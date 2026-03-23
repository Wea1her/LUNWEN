"""一键实验编排: 多种子并行训练 → 评估 → 聚合 → LaTeX 表格"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import time

import numpy as np
import torch

import config as C
from device_utils import resolve_device, seed_everything
from env import TxOrderingEnv
from networks import ActorCritic
from train import train as train_model
from evaluate import (build_shared_pools, save_shared_pools, load_shared_pools,
                      evaluate_rl, evaluate_baseline, aggregate,
                      plot_training_curve)
from baselines import run_baseline
from metrics import compute_all_metrics
from latex_tables import (generate_main_table, generate_robustness_table,
                          generate_ablation_table, ABLATION_ORDER,
                          ABLATION_LABELS, STRUCT_ABLATION_ORDER,
                          STRUCT_ABLATION_LABELS)
from method_registry import MAIN_METHOD_ORDER, get_baseline_method_ids, normalize_method_id
from stat_tests import (SCIPY_AVAILABLE, run_paired_significance_tests,
                        format_paired_significance_table,
                        generate_paired_significance_latex)

STAGE_ORDER = ["main", "robustness", "ablation"]


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: str, payload: dict | list):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _episode_rows(raw: dict[str, list[dict]], setting: str, setting_value, seed: int) -> list[dict]:
    rows = []
    for method, metrics_seq in raw.items():
        for episode_id, metrics in enumerate(metrics_seq):
            rows.append({
                "seed": seed,
                "setting": setting,
                "setting_value": setting_value,
                "method": method,
                "episode_id": episode_id,
                "shared_pool_id": episode_id,
                "metrics": metrics,
            })
    return rows


def _select_case_episode_ids(raw: dict[str, list[dict]], baseline_methods: list[str], max_cases: int = 3) -> list[int]:
    ours_metrics = raw.get("ours", [])
    if not ours_metrics:
        return []
    candidates = []
    for episode_id in range(len(ours_metrics)):
        ours = ours_metrics[episode_id]
        baseline_metrics = [raw[m][episode_id] for m in baseline_methods if m in raw]
        if not baseline_metrics:
            continue
        mean_risk = float(np.mean([b["risk_exposure"] for b in baseline_metrics]))
        mean_fee = float(np.mean([b["block_fee"] for b in baseline_metrics]))
        score = abs(ours["risk_exposure"] - mean_risk) + 0.001 * abs(ours["block_fee"] - mean_fee)
        candidates.append((score, episode_id))
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = [eid for _, eid in candidates[:max_cases]]
    if not chosen:
        chosen = list(range(min(max_cases, len(ours_metrics))))
    return chosen


def _build_compact_order(selected: list, head: int = 12, tail: int = 12) -> dict:
    seq = [int(tx.tid) for tx in selected]
    head_items = [{
        "tid": int(tx.tid),
        "fee": float(tx.fee),
        "risk_score": float(tx.risk_score),
        "arrival_time": float(tx.arrival_time),
    } for tx in selected[:head]]
    tail_items = [{
        "tid": int(tx.tid),
        "fee": float(tx.fee),
        "risk_score": float(tx.risk_score),
        "arrival_time": float(tx.arrival_time),
    } for tx in selected[-tail:]] if selected else []
    return {
        "selected_count": len(selected),
        "selected_tid_sequence": seq,
        "head": head_items,
        "tail": tail_items,
    }


def _evaluate_method_on_pool(pool: list, method_id: str, model, device, env: TxOrderingEnv) -> dict:
    if method_id == "ours":
        obs, _ = env.reset_with_pool(pool)
        while True:
            action, _, _ = model.act(obs, device, greedy=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        selected = env.get_selected_transactions()
    else:
        selected = run_baseline(deepcopy(pool), method_id)
    metrics = compute_all_metrics(selected, pool)
    return {
        "metrics": metrics,
        "order": _build_compact_order(selected),
    }


def _build_case_study_for_seed(
    model,
    device,
    args,
    seed: int,
    shared_pools: list[list],
    raw: dict[str, list[dict]],
    result_dir: str,
) -> str | None:
    case_episode_ids = _select_case_episode_ids(raw, args.baseline_methods, max_cases=3)
    if not case_episode_ids:
        return None

    env = TxOrderingEnv(pool_size=args.pool_size, risk_ratio=C.RISK_RATIO, seed=seed)
    episodes_payload = []
    for episode_id in case_episode_ids:
        pool = deepcopy(shared_pools[episode_id])
        risk_tx_count = sum(1 for tx in pool if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD)
        method_details = {}
        for method_id in ["ours", *args.baseline_methods]:
            method_details[method_id] = _evaluate_method_on_pool(pool, method_id, model, device, env)
        episodes_payload.append({
            "episode_id": episode_id,
            "shared_pool_id": episode_id,
            "pool_summary": {
                "pool_size": len(pool),
                "risk_ratio_observed": (risk_tx_count / len(pool)) if pool else 0.0,
                "mean_fee": float(np.mean([tx.fee for tx in pool])) if pool else 0.0,
                "mean_arrival": float(np.mean([tx.arrival_time for tx in pool])) if pool else 0.0,
            },
            "methods": method_details,
        })

    payload = {
        "seed": seed,
        "setting": "main",
        "selection_rule": "top_gap_episodes_by_risk_exposure_and_fee",
        "baseline_methods": args.baseline_methods,
        "episodes": episodes_payload,
    }
    case_path = os.path.join(result_dir, "case_study_seed.json")
    _write_json(case_path, payload)
    return case_path


def _run_main_evaluation(model, device, args, seed, result_dir):
    env = TxOrderingEnv(pool_size=args.pool_size, risk_ratio=C.RISK_RATIO, seed=seed)
    shared_pools = build_shared_pools(args.eval_episodes, args.pool_size, C.RISK_RATIO, seed)
    shared_pool_path = os.path.join(result_dir, f"shared_pools_main_seed{seed}.json")
    save_shared_pools(shared_pool_path, shared_pools, metadata={
        "seed": seed,
        "eval_episodes": args.eval_episodes,
        "pool_size": args.pool_size,
        "risk_ratio": C.RISK_RATIO,
        "setting": "main",
        "baseline_methods": args.baseline_methods,
    })

    raw = {"ours": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
    for bl in args.baseline_methods:
        raw[bl] = evaluate_baseline(bl, env, args.eval_episodes, shared_pools)
    agg = {method: aggregate(metrics) for method, metrics in raw.items()}

    _write_json(os.path.join(result_dir, "main_results.json"), agg)
    _write_json(os.path.join(result_dir, "main_aggregated_metrics.json"), agg)
    _write_json(os.path.join(result_dir, "main_episode_metrics.json"), {
        "seed": seed,
        "setting": "main",
        "records": _episode_rows(raw, "main", {"pool_size": args.pool_size, "risk_ratio": C.RISK_RATIO}, seed),
    })
    case_study_seed_path = _build_case_study_for_seed(
        model=model,
        device=device,
        args=args,
        seed=seed,
        shared_pools=shared_pools,
        raw=raw,
        result_dir=result_dir,
    )
    return agg, case_study_seed_path


def _run_risk_robustness(model, device, args, seed, result_dir):
    all_data = {}
    all_rows = []
    pool_index = {
        "seed": seed,
        "dimension": "risk_ratio",
        "files": {},
    }

    for rr in C.ROBUSTNESS_RISK_RATIOS:
        env = TxOrderingEnv(pool_size=args.pool_size, risk_ratio=rr, seed=seed)
        shared_pools = build_shared_pools(args.eval_episodes, args.pool_size, rr, seed)
        rr_tag = str(rr).replace(".", "p")
        rr_pool_file = os.path.join(result_dir, f"shared_pools_robust_risk_seed{seed}_rr_{rr_tag}.json")
        save_shared_pools(rr_pool_file, shared_pools, metadata={
            "seed": seed,
            "eval_episodes": args.eval_episodes,
            "pool_size": args.pool_size,
            "risk_ratio": rr,
            "setting": "robustness_risk",
            "baseline_methods": args.baseline_methods,
        })
        pool_index["files"][str(rr)] = os.path.basename(rr_pool_file)

        raw = {"ours": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
        for bl in args.baseline_methods:
            raw[bl] = evaluate_baseline(bl, env, args.eval_episodes, shared_pools)
        all_data[str(rr)] = {method: aggregate(metrics) for method, metrics in raw.items()}
        all_rows.extend(_episode_rows(raw, "robustness_risk", {"risk_ratio": rr}, seed))

    _write_json(os.path.join(result_dir, "shared_pools_robust_risk_seed{seed}.json".format(seed=seed)), pool_index)
    _write_json(os.path.join(result_dir, "robustness_results.json"), all_data)
    _write_json(os.path.join(result_dir, "robustness_risk_episode_metrics.json"), {
        "seed": seed,
        "dimension": "risk_ratio",
        "records": all_rows,
    })
    return all_data


def _run_pool_robustness(model, device, args, seed, result_dir):
    all_data = {}
    all_rows = []
    pool_index = {
        "seed": seed,
        "dimension": "pool_size",
        "files": {},
    }

    for ps in C.ROBUSTNESS_POOL_SIZES:
        env = TxOrderingEnv(pool_size=ps, risk_ratio=C.RISK_RATIO, seed=seed)
        shared_pools = build_shared_pools(args.eval_episodes, ps, C.RISK_RATIO, seed)
        pool_file = os.path.join(result_dir, f"shared_pools_robust_pool_seed{seed}_N{ps}.json")
        save_shared_pools(pool_file, shared_pools, metadata={
            "seed": seed,
            "eval_episodes": args.eval_episodes,
            "pool_size": ps,
            "risk_ratio": C.RISK_RATIO,
            "setting": "robustness_pool_size",
            "baseline_methods": args.baseline_methods,
        })
        pool_index["files"][str(ps)] = os.path.basename(pool_file)

        raw = {"ours": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
        for bl in args.baseline_methods:
            raw[bl] = evaluate_baseline(bl, env, args.eval_episodes, shared_pools)
        all_data[str(ps)] = {method: aggregate(metrics) for method, metrics in raw.items()}
        all_rows.extend(_episode_rows(raw, "robustness_pool_size", {"pool_size": ps}, seed))

    _write_json(os.path.join(result_dir, "shared_pools_robust_pool_seed{seed}.json".format(seed=seed)), pool_index)
    _write_json(os.path.join(result_dir, "robustness_pool_size.json"), all_data)
    _write_json(os.path.join(result_dir, "robustness_pool_episode_metrics.json"), {
        "seed": seed,
        "dimension": "pool_size",
        "records": all_rows,
    })
    return all_data


def _run_fee_robustness(model, device, args, seed, result_dir):
    all_data = {}
    all_rows = []
    base_shared_pools = build_shared_pools(args.eval_episodes, args.pool_size, C.RISK_RATIO, seed)
    base_pool_file = os.path.join(result_dir, f"shared_pools_robust_fee_seed{seed}_base.json")
    save_shared_pools(base_pool_file, base_shared_pools, metadata={
        "seed": seed,
        "eval_episodes": args.eval_episodes,
        "pool_size": args.pool_size,
        "risk_ratio": C.RISK_RATIO,
        "setting": "robustness_fee_multiplier_base",
        "baseline_methods": args.baseline_methods,
    })
    _write_json(os.path.join(result_dir, f"shared_pools_robust_fee_seed{seed}.json"), {
        "seed": seed,
        "dimension": "fee_multiplier",
        "base_shared_pool_file": os.path.basename(base_pool_file),
        "multipliers": C.ROBUSTNESS_FEE_MULTIPLIERS,
        "transform_rule": (
            "if tx.risk_score >= HEURISTIC_RISK_THRESHOLD: "
            "tx.fee = tx.fee / RISK_FEE_MULTIPLIER * multiplier"
        ),
    })

    for mult in C.ROBUSTNESS_FEE_MULTIPLIERS:
        env = TxOrderingEnv(pool_size=args.pool_size, risk_ratio=C.RISK_RATIO, seed=seed)
        raw_rl = []
        raw_bl = {bl: [] for bl in args.baseline_methods}

        for base_pool in base_shared_pools:
            pool = deepcopy(base_pool)
            for tx in pool:
                if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD:
                    tx.fee = tx.fee / C.RISK_FEE_MULTIPLIER * mult

            obs, _ = env.reset_with_pool(pool)
            while True:
                action, _, _ = model.act(obs, device, greedy=True)
                obs, _, done, _, _ = env.step(action)
                if done:
                    break
            selected_rl = env.get_selected_transactions()
            raw_rl.append(compute_all_metrics(selected_rl, pool))

            for bl in args.baseline_methods:
                selected_bl = run_baseline(deepcopy(pool), bl)
                raw_bl[bl].append(compute_all_metrics(selected_bl, pool))

        raw = {"ours": raw_rl}
        raw.update(raw_bl)
        all_data[str(mult)] = {method: aggregate(metrics) for method, metrics in raw.items()}
        all_rows.extend(_episode_rows(raw, "robustness_fee_multiplier", {"fee_multiplier": mult}, seed))

    _write_json(os.path.join(result_dir, "robustness_fee_mult.json"), all_data)
    _write_json(os.path.join(result_dir, "robustness_fee_episode_metrics.json"), {
        "seed": seed,
        "dimension": "fee_multiplier",
        "records": all_rows,
    })
    return all_data


def run_single_seed(seed, args_dict):
    """单个种子: 训练 + 分阶段评估 (独立进程)"""
    args = argparse.Namespace(**args_dict)
    if isinstance(getattr(args, "device_map", None), dict) and seed in args.device_map:
        args.device = args.device_map[seed]
    stages = set(args.stages)
    device = resolve_device(args.device)
    total_start = time.perf_counter()

    seed_dir = os.path.join(args.output, f"seed_{seed}")
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    result_dir = os.path.join(seed_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    model_path = os.path.join(ckpt_dir, C.FORMAL_EVAL_CHECKPOINT_NAME)
    log_path = os.path.join(ckpt_dir, "train_log.json")
    seed_summary = {
        "seed": seed,
        "policy_source": "trained",
        "stages": list(args.stages),
        "result_dir": result_dir,
        "status": "success",
        "timing_seconds": {},
        "requested_device": args.device,
        "resolved_device": str(device),
    }
    timing = {
        "train": 0.0,
        "eval": 0.0,
        "robustness": 0.0,
        "robustness_risk": 0.0,
        "robustness_pool": 0.0,
        "robustness_fee": 0.0,
    }

    # 训练
    if not args.skip_train:
        print(f"[seed={seed}] 开始训练", flush=True)
        seed_everything(seed)
        train_args = argparse.Namespace(
            episodes=args.episodes,
            pool_size=args.pool_size,
            risk_ratio=C.RISK_RATIO,
            seed=seed,
            log_interval=C.LOG_INTERVAL,
            output=ckpt_dir,
            device=args.device,
            val_episodes=args.val_episodes,
            val_interval=args.val_interval,
            val_metric=args.val_metric,
        )
        t0 = time.perf_counter()
        train_model(train_args)
        timing["train"] = time.perf_counter() - t0
        print(f"[seed={seed}] 训练完成", flush=True)

    main_results = None
    rob_risk = None
    rob_pool = None
    rob_fee = None
    main_episode_metrics_path = None
    case_study_seed_path = None
    requires_policy = bool(stages & {"main", "robustness"})

    if requires_policy:
        model = ActorCritic().to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        elif args.allow_random_fallback:
            seed_summary["policy_source"] = "random_fallback"
            print(f"[seed={seed}] Warning: {model_path} not found, using random policy", flush=True)
        else:
            raise FileNotFoundError(
                f"[seed={seed}] required checkpoint not found: {model_path}. "
                "Use --allow-random-fallback to force random-policy evaluation."
            )
        model.eval()

        if "main" in stages:
            print(f"[seed={seed}] 主实验评估", flush=True)
            t0 = time.perf_counter()
            main_results, case_study_seed_path = _run_main_evaluation(model, device, args, seed, result_dir)
            timing["eval"] = time.perf_counter() - t0
            main_episode_metrics_path = os.path.join(result_dir, "main_episode_metrics.json")

        if "robustness" in stages:
            print(f"[seed={seed}] 鲁棒性实验 (风险比例)", flush=True)
            t0 = time.perf_counter()
            rob_risk = _run_risk_robustness(model, device, args, seed, result_dir)
            timing["robustness_risk"] = time.perf_counter() - t0
            print(f"[seed={seed}] 鲁棒性实验 (池大小)", flush=True)
            t1 = time.perf_counter()
            rob_pool = _run_pool_robustness(model, device, args, seed, result_dir)
            timing["robustness_pool"] = time.perf_counter() - t1
            print(f"[seed={seed}] 鲁棒性实验 (费率倍率)", flush=True)
            t2 = time.perf_counter()
            rob_fee = _run_fee_robustness(model, device, args, seed, result_dir)
            timing["robustness_fee"] = time.perf_counter() - t2
            timing["robustness"] = (
                timing["robustness_risk"] + timing["robustness_pool"] + timing["robustness_fee"]
            )

    if os.path.exists(log_path):
        plot_training_curve(log_path, os.path.join(result_dir, "training_curve.png"))

    timing["total"] = time.perf_counter() - total_start
    seed_summary["timing_seconds"] = {k: float(v) for k, v in timing.items()}
    _write_json(os.path.join(result_dir, "seed_summary.json"), seed_summary)
    print(f"[seed={seed}] 完成", flush=True)
    return {
        "seed": seed,
        "main": main_results,
        "rob_risk": rob_risk,
        "rob_pool": rob_pool,
        "rob_fee": rob_fee,
        "main_episode_metrics_path": main_episode_metrics_path,
        "case_study_seed_path": case_study_seed_path,
        "timing": timing,
        "seed_summary": seed_summary,
    }


def aggregate_metric_grid(all_results):
    """聚合形如 {setting: {method: metric_stats}} 的结果网格。"""
    all_settings = set()
    for result in all_results:
        all_settings.update(result.keys())
    settings = sorted(all_settings, key=str)
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util",
               "risky_rank", "packing_ratio", "top10_risk", "late_promo"]

    agg = {}
    for setting in settings:
        methods = [m for m in MAIN_METHOD_ORDER if any(m in r.get(setting, {}) for r in all_results)]
        if not methods:
            method_set = set()
            for result in all_results:
                method_set.update(result.get(setting, {}).keys())
            methods = sorted(method_set, key=str)
        agg[setting] = {}
        for method in methods:
            agg[setting][method] = {}
            for metric in metrics:
                vals = []
                for result in all_results:
                    if method not in result.get(setting, {}):
                        continue
                    vals.append(result[setting][method][f"{metric}_mean"])
                if vals:
                    agg[setting][method][f"{metric}_mean"] = float(np.mean(vals))
                    agg[setting][method][f"{metric}_std"] = float(np.std(vals))
                else:
                    agg[setting][method][f"{metric}_mean"] = float("nan")
                    agg[setting][method][f"{metric}_std"] = float("nan")
    return agg


def aggregate_across_seeds(all_main=None, all_rob_risk=None, all_rob_pool=None, all_rob_fee=None):
    """跨种子聚合: 对每个方法的 mean 取均值和标准差"""
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util",
               "risky_rank", "packing_ratio", "top10_risk", "late_promo"]

    agg_main = {}
    if all_main:
        methods = [m for m in MAIN_METHOD_ORDER if any(m in r for r in all_main)]
        if not methods:
            method_set = set()
            for result in all_main:
                method_set.update(result.keys())
            methods = sorted(method_set, key=str)
        for method in methods:
            agg_main[method] = {}
            for metric in metrics:
                vals = []
                for result in all_main:
                    if method not in result:
                        continue
                    vals.append(result[method][f"{metric}_mean"])
                if vals:
                    agg_main[method][f"{metric}_mean"] = float(np.mean(vals))
                    agg_main[method][f"{metric}_std"] = float(np.std(vals))
                else:
                    agg_main[method][f"{metric}_mean"] = float("nan")
                    agg_main[method][f"{metric}_std"] = float("nan")

    agg_rob_risk = aggregate_metric_grid(all_rob_risk) if all_rob_risk else {}
    agg_rob_pool = aggregate_metric_grid(all_rob_pool) if all_rob_pool else {}
    agg_rob_fee = aggregate_metric_grid(all_rob_fee) if all_rob_fee else {}
    return agg_main, agg_rob_risk, agg_rob_pool, agg_rob_fee


# ================================================================
# 消融实验
# ================================================================

REWARD_ABLATION_CONFIGS = {
    "Ours-FeeOnly":   {"alpha": 1.0, "beta": 0.0,         "gamma_r": 0.0},
    "Ours-Fee+Fair":  {"alpha": C.ALPHA, "beta": C.BETA,   "gamma_r": 0.0},
    "Ours-Fee+Risk":  {"alpha": C.ALPHA, "beta": 0.0,      "gamma_r": C.GAMMA_R},
    "Ours-Full":      {"alpha": C.ALPHA, "beta": C.BETA,   "gamma_r": C.GAMMA_R},
}

STRUCT_ABLATION_CONFIGS = {
    "No-SeqSummary":  {"no_seq_summary": True,  "no_stop": False, "no_action_mask": False},
    "No-ActionMask":  {"no_seq_summary": False, "no_stop": False, "no_action_mask": True},
    "No-STOP":        {"no_seq_summary": False, "no_stop": True,  "no_action_mask": False},
    "Ours-Full":      {"no_seq_summary": False, "no_stop": False, "no_action_mask": False},
}


def _train_ablation_single(name, seed, args_dict, env_kwargs, shared_pool_path=None):
    """单个消融变体+种子的训练与评估 (独立进程)"""
    args = argparse.Namespace(**args_dict)
    if isinstance(getattr(args, "device_map", None), dict) and seed in args.device_map:
        args.device = args.device_map[seed]
    device = resolve_device(args.device)

    abl_dir = os.path.join(args.output, "ablation", name, f"seed_{seed}")
    ckpt_dir = os.path.join(abl_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, C.FORMAL_EVAL_CHECKPOINT_NAME)

    if not args.skip_train:
        print(f"  [Ablation] {name} seed={seed} 开始训练", flush=True)
        seed_everything(seed)
        train_args = argparse.Namespace(
            episodes=args.episodes,
            pool_size=args.pool_size,
            risk_ratio=C.RISK_RATIO,
            seed=seed,
            log_interval=C.LOG_INTERVAL,
            output=ckpt_dir,
            device=args.device,
            val_episodes=args.val_episodes,
            val_interval=args.val_interval,
            val_metric=args.val_metric,
        )
        train_model(train_args, env_kwargs=env_kwargs)
        print(f"  [Ablation] {name} seed={seed} 训练完成", flush=True)

    model = ActorCritic().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    elif not args.allow_random_fallback:
        raise FileNotFoundError(
            f"[Ablation] required checkpoint not found: {model_path}. "
            "Use --allow-random-fallback to force random-policy evaluation."
        )
    model.eval()

    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=C.RISK_RATIO, seed=seed,
                        **env_kwargs)
    if shared_pool_path and os.path.exists(shared_pool_path):
        shared_pools = load_shared_pools(shared_pool_path)
    else:
        shared_pools = build_shared_pools(args.eval_episodes, args.pool_size, C.RISK_RATIO, seed)
    result = aggregate(evaluate_rl(model, env, args.eval_episodes, device, shared_pools))
    result_path = os.path.join(abl_dir, "eval_result.json")
    _write_json(result_path, {
        "variant": name,
        "seed": seed,
        "shared_pool_path": shared_pool_path,
        "metrics": result,
    })
    return name, seed, result


def run_ablation_parallel(configs, args, label_prefix):
    """并行运行消融实验"""
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util"]
    args_dict = vars(args)
    shared_pool_dir = os.path.join(args.output, "ablation", "shared_pools")
    os.makedirs(shared_pool_dir, exist_ok=True)
    shared_pool_paths = {}
    for seed in args.seeds:
        shared_pool_path = os.path.join(shared_pool_dir, f"seed_{seed}.json")
        shared_pools = build_shared_pools(args.eval_episodes, args.pool_size, C.RISK_RATIO, seed)
        save_shared_pools(shared_pool_path, shared_pools, metadata={
            "seed": seed,
            "eval_episodes": args.eval_episodes,
            "pool_size": args.pool_size,
            "risk_ratio": C.RISK_RATIO,
            "source": "ablation_evaluation_shared_pools",
        })
        shared_pool_paths[seed] = shared_pool_path

    max_workers = max(1, min(args.workers, len(configs) * len(args.seeds)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, env_kwargs in configs.items():
            for seed in args.seeds:
                fut = executor.submit(
                    _train_ablation_single,
                    f"{label_prefix}_{name}",
                    seed,
                    args_dict,
                    env_kwargs,
                    shared_pool_paths[seed],
                )
                futures[fut] = (name, seed)

        results = {name: [] for name in configs}
        failures = []
        for fut in as_completed(futures):
            orig_name, seed = futures[fut]
            try:
                _name_full, _seed, res = fut.result()
                results[orig_name].append(res)
            except Exception as exc:
                failures.append({
                    "variant": orig_name,
                    "seed": seed,
                    "error": repr(exc),
                })
                print(f"  [Ablation] {orig_name} seed={seed} 失败: {exc}", flush=True)

    agg = {}
    for name in configs:
        agg[name] = {}
        for metric in metrics:
            vals = [r[f"{metric}_mean"] for r in results[name]]
            if vals:
                agg[name][f"{metric}_mean"] = float(np.mean(vals))
                agg[name][f"{metric}_std"] = float(np.std(vals))
            else:
                agg[name][f"{metric}_mean"] = float("nan")
                agg[name][f"{metric}_std"] = float("nan")
    if failures:
        print(f"  [Ablation] {len(failures)} 个任务失败，已跳过失败项继续聚合。", flush=True)
    return agg


def _resolve_stages(args):
    stages = set(args.stages or [])
    if args.ablation:
        stages.add("ablation")
    if not stages:
        stages.add("main")
    args.stages = [stage for stage in STAGE_ORDER if stage in stages]


def _collect_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            text=True,
        ).strip()
    except Exception:
        return None


def _collect_config_snapshot(args):
    constants = {}
    for name in dir(C):
        if name.isupper():
            constants[name] = _to_jsonable(getattr(C, name))
    return {
        "timestamp": datetime.now().isoformat(),
        "git_commit": _collect_git_commit(),
        "args": _to_jsonable(vars(args)),
        "config_constants": constants,
    }


STAGE_REQUIRED_FILES = {
    "main": [
        "main_results.json",
        "main_episode_metrics.json",
        "case_study_seed.json",
    ],
    "robustness": [
        "robustness_results.json",
        "robustness_pool_size.json",
        "robustness_fee_mult.json",
        "robustness_risk_episode_metrics.json",
        "robustness_pool_episode_metrics.json",
        "robustness_fee_episode_metrics.json",
    ],
}


def _seed_stage_outputs_exist(output_dir: str, seed: int, stages: list[str]) -> bool:
    result_dir = os.path.join(output_dir, f"seed_{seed}", "results")
    if not os.path.isdir(result_dir):
        return False
    for stage in stages:
        for filename in STAGE_REQUIRED_FILES.get(stage, []):
            if not os.path.exists(os.path.join(result_dir, filename)):
                return False
    return True


def _parse_device_map(raw: str | None) -> dict[int, str]:
    if not raw:
        return {}
    candidate = raw.strip()
    if not candidate:
        return {}

    if os.path.isfile(candidate):
        with open(candidate) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("device map JSON must be an object: {\"42\": \"cuda:0\", ...}")
        return {int(k): str(v) for k, v in payload.items()}

    result = {}
    for item in candidate.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "device-map format must be `seed:device` comma-separated, "
                "e.g. 42:cuda:0,123:cuda:1"
            )
        seed_str, device = item.split(":", 1)
        result[int(seed_str.strip())] = device.strip()
    return result


def _load_main_episode_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            payload = json.load(f)
        payload_seed = payload.get("seed")
        for row in payload.get("records", []):
            method_key = row.get("method")
            if method_key is not None:
                try:
                    row["method"] = normalize_method_id(method_key)
                except KeyError:
                    pass
            if "seed" not in row:
                row["seed"] = payload_seed
            rows.append(row)
    return rows


def _read_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _method_metric_summary(main_episode_rows: list[dict], method_id: str, metric: str) -> dict:
    vals = []
    for row in main_episode_rows:
        if row.get("setting", "main") != "main":
            continue
        if row.get("method") != method_id:
            continue
        value = row.get("metrics", {}).get(metric)
        if value is not None:
            vals.append(float(value))
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "count": len(vals)}


def _build_behavior_probe(main_episode_rows: list[dict], baseline_methods: list[str]) -> dict:
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util", "risky_rank", "top10_risk", "late_promo"]
    summary_by_method = {}
    for method_id in ["ours", *baseline_methods]:
        summary_by_method[method_id] = {m: _method_metric_summary(main_episode_rows, method_id, m) for m in metrics}

    row_map: dict[tuple, dict[str, dict]] = {}
    for row in main_episode_rows:
        if row.get("setting", "main") != "main":
            continue
        key = (row.get("seed"), row.get("shared_pool_id", row.get("episode_id")))
        method = row.get("method")
        row_map.setdefault(key, {})[method] = row.get("metrics", {})

    paired_deltas = {}
    for baseline in baseline_methods:
        paired_deltas[baseline] = {}
        for metric in metrics:
            diffs = []
            better_flags = []
            for methods in row_map.values():
                if "ours" not in methods or baseline not in methods:
                    continue
                ours = methods["ours"].get(metric)
                base = methods[baseline].get(metric)
                if ours is None or base is None:
                    continue
                ours = float(ours)
                base = float(base)
                diff = ours - base
                diffs.append(diff)
                if metric in {"risk_exposure", "top10_risk"}:
                    better_flags.append(1.0 if ours < base else 0.0)
                elif metric == "risky_rank":
                    better_flags.append(1.0 if abs(ours - 0.5) < abs(base - 0.5) else 0.0)
                else:
                    better_flags.append(1.0 if ours > base else 0.0)
            if diffs:
                paired_deltas[baseline][metric] = {
                    "mean_delta": float(np.mean(diffs)),
                    "std_delta": float(np.std(diffs)),
                    "median_delta": float(np.median(diffs)),
                    "ours_better_rate": float(np.mean(better_flags)) if better_flags else float("nan"),
                    "n_pairs": len(diffs),
                }
            else:
                paired_deltas[baseline][metric] = {
                    "mean_delta": float("nan"),
                    "std_delta": float("nan"),
                    "median_delta": float("nan"),
                    "ours_better_rate": float("nan"),
                    "n_pairs": 0,
                }

    return {
        "metrics": metrics,
        "summary_by_method": summary_by_method,
        "paired_deltas_ours_vs_baselines": paired_deltas,
        "note": "paired_deltas are computed on shared_pool_id aligned episodes",
    }


def _merge_case_studies(case_study_seed_paths: list[str]) -> dict:
    seeds_payload = []
    for path in case_study_seed_paths:
        payload = _read_json_if_exists(path)
        if payload is not None:
            seeds_payload.append(payload)
    return {
        "generated_at": datetime.now().isoformat(),
        "seed_case_studies": seeds_payload,
    }


def main():
    parser = argparse.ArgumentParser(description="一键实验编排 (并行版)")
    parser.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    parser.add_argument("--episodes", type=int, default=C.TOTAL_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=C.EVAL_EPISODES)
    parser.add_argument("--pool-size", type=C.validate_pool_size, default=C.POOL_SIZE_DEFAULT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results_final")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练, 仅评估已有模型")
    parser.add_argument("--ablation", action="store_true", help="兼容参数: 在 stages 中追加 ablation")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=["main"],
        help="运行阶段。默认仅 main，可选: main robustness ablation",
    )
    parser.add_argument("--workers", type=int, default=None, help="并行进程数（默认按设备自动设置）")
    parser.add_argument("--max-gpu-workers", type=int, default=5,
                        help="CUDA 场景最大并行进程数（默认 5）")
    parser.add_argument("--device-map", type=str, default="",
                        help="可选 seed->device 映射，如 42:cuda:0,123:cuda:1 或 JSON 文件路径")
    parser.add_argument("--resume", action="store_true",
                        help="断点恢复：已完成 seed 自动跳过，其余 seed 继续运行")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有完整输出的 seed")
    parser.add_argument("--allow-random-fallback", action="store_true",
                        help="当 checkpoint 缺失时允许随机策略回退")
    parser.add_argument("--enable-center-aware-baseline", action="store_true",
                        help="启用更强启发式基线 Center-Aware Greedy")
    parser.add_argument("--val-episodes", type=int, default=C.VALIDATION_EPISODES)
    parser.add_argument("--val-interval", type=int, default=C.VALIDATION_INTERVAL)
    parser.add_argument("--val-metric", type=str, default=C.VALIDATION_METRIC)
    args = parser.parse_args()
    _resolve_stages(args)
    args.device_map = _parse_device_map(args.device_map)
    args.baseline_methods = get_baseline_method_ids(args.enable_center_aware_baseline)

    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)
    if args.workers is None:
        if device.type == "cuda":
            mapped_gpu_devices = {d for d in args.device_map.values() if d.startswith("cuda")}
            gpu_slots = len(mapped_gpu_devices) if mapped_gpu_devices else args.max_gpu_workers
            args.workers = min(len(args.seeds), max(1, gpu_slots))
        else:
            args.workers = min(len(args.seeds), max(1, os.cpu_count() or 1))
    else:
        args.workers = max(1, args.workers)

    if device.type == "cuda":
        allowed = max(1, args.max_gpu_workers)
        if args.workers > allowed:
            print(f"Warning: CUDA 场景 workers={args.workers} 超过 max-gpu-workers={allowed}，已自动收敛。")
            args.workers = allowed

    run_start = time.perf_counter()
    print(f"Device: {device}")
    print(f"Stages: {args.stages}")
    print(f"Seeds: {args.seeds}")
    print(f"Workers: {args.workers}")
    print(f"Resume: {args.resume}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Episodes: {args.episodes}")
    print(f"Pool size: {args.pool_size}")
    print(f"Baselines: {args.baseline_methods}")
    if args.device_map:
        print(f"Device map: {args.device_map}")

    config_snapshot_path = os.path.join(args.output, "config_snapshot.json")
    _write_json(config_snapshot_path, _collect_config_snapshot(args))

    status_counts = {
        "success": 0,
        "failed": 0,
        "skipped_existing": 0,
        "resumed": 0,
    }
    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "stages": args.stages,
        "allow_random_fallback": args.allow_random_fallback,
        "resume": args.resume,
        "skip_existing": args.skip_existing,
        "workers": args.workers,
        "device": str(device),
        "device_map": {str(k): v for k, v in args.device_map.items()},
        "baseline_methods": args.baseline_methods,
        "config_snapshot": os.path.basename(config_snapshot_path),
        "seeds": args.seeds,
        "seed_runs": [],
        "status_counts": status_counts,
        "outputs": {},
    }

    all_main = []
    all_rob_risk = []
    all_rob_pool = []
    all_rob_fee = []
    main_episode_metric_paths = []
    case_study_seed_paths = []
    seed_timings = {}
    ablation_reward_seconds = 0.0
    ablation_struct_seconds = 0.0

    if set(args.stages) & {"main", "robustness"}:
        seeds_to_run = []
        resume_candidates = set()

        for seed in args.seeds:
            has_existing = _seed_stage_outputs_exist(args.output, seed, args.stages)
            seed_dir = os.path.join(args.output, f"seed_{seed}")
            result_dir = os.path.join(seed_dir, "results")

            if has_existing and (args.skip_existing or args.resume):
                summary = {
                    "seed": seed,
                    "status": "skipped_existing",
                    "policy_source": "existing",
                    "stages": list(args.stages),
                    "result_dir": result_dir,
                }
                run_summary["seed_runs"].append(summary)
                status_counts["skipped_existing"] += 1

                main_payload = _read_json_if_exists(os.path.join(result_dir, "main_results.json"))
                if main_payload is not None:
                    all_main.append(main_payload)
                    main_episode_metric_paths.append(os.path.join(result_dir, "main_episode_metrics.json"))
                    case_study_path = os.path.join(result_dir, "case_study_seed.json")
                    if os.path.exists(case_study_path):
                        case_study_seed_paths.append(case_study_path)

                rob_risk_payload = _read_json_if_exists(os.path.join(result_dir, "robustness_results.json"))
                rob_pool_payload = _read_json_if_exists(os.path.join(result_dir, "robustness_pool_size.json"))
                rob_fee_payload = _read_json_if_exists(os.path.join(result_dir, "robustness_fee_mult.json"))
                if rob_risk_payload is not None:
                    all_rob_risk.append(rob_risk_payload)
                if rob_pool_payload is not None:
                    all_rob_pool.append(rob_pool_payload)
                if rob_fee_payload is not None:
                    all_rob_fee.append(rob_fee_payload)

                seed_summary_path = os.path.join(result_dir, "seed_summary.json")
                existing_summary = _read_json_if_exists(seed_summary_path)
                if isinstance(existing_summary, dict) and "timing_seconds" in existing_summary:
                    seed_timings[str(seed)] = existing_summary["timing_seconds"]
                continue

            if args.resume and os.path.isdir(seed_dir):
                resume_candidates.add(seed)
            seeds_to_run.append(seed)

        if seeds_to_run:
            max_workers = max(1, min(args.workers, len(seeds_to_run)))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for seed in seeds_to_run:
                    seed_args_dict = dict(vars(args))
                    if seed in args.device_map:
                        seed_args_dict["device"] = args.device_map[seed]
                    futures[executor.submit(run_single_seed, seed, seed_args_dict)] = (
                        seed,
                        seed_args_dict["device"],
                    )

                for fut in as_completed(futures):
                    seed, seed_device = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        status_counts["failed"] += 1
                        run_summary["seed_runs"].append({
                            "seed": seed,
                            "status": "failed",
                            "device": seed_device,
                            "error": repr(exc),
                        })
                        print(f"[seed={seed}] 失败: {exc}", flush=True)
                        continue

                    seed_status = "resumed" if args.resume and seed in resume_candidates else "success"
                    status_counts[seed_status] += 1
                    seed_summary = result["seed_summary"]
                    seed_summary["status"] = seed_status
                    seed_summary["requested_device"] = seed_device
                    seed_summary.setdefault("resolved_device", seed_device)
                    run_summary["seed_runs"].append(seed_summary)
                    _write_json(os.path.join(seed_summary["result_dir"], "seed_summary.json"), seed_summary)

                    if result["main"] is not None:
                        all_main.append(result["main"])
                    if result["rob_risk"] is not None:
                        all_rob_risk.append(result["rob_risk"])
                    if result["rob_pool"] is not None:
                        all_rob_pool.append(result["rob_pool"])
                    if result["rob_fee"] is not None:
                        all_rob_fee.append(result["rob_fee"])
                    if result["main_episode_metrics_path"] is not None:
                        main_episode_metric_paths.append(result["main_episode_metrics_path"])
                    if result.get("case_study_seed_path") is not None:
                        case_study_seed_paths.append(result["case_study_seed_path"])
                    if result.get("timing") is not None:
                        seed_timings[str(seed)] = result["timing"]

                    print(
                        f"[seed={seed}] 结果已收集 "
                        f"({status_counts['success'] + status_counts['resumed'] + status_counts['failed']}/"
                        f"{len(seeds_to_run)})",
                        flush=True,
                    )

    agg_main, agg_rob_risk, agg_rob_pool, agg_rob_fee = aggregate_across_seeds(
        all_main=all_main if "main" in args.stages else None,
        all_rob_risk=all_rob_risk if "robustness" in args.stages else None,
        all_rob_pool=all_rob_pool if "robustness" in args.stages else None,
        all_rob_fee=all_rob_fee if "robustness" in args.stages else None,
    )

    if "main" in args.stages and agg_main:
        agg_main_path = os.path.join(args.output, "aggregated_main.json")
        _write_json(agg_main_path, agg_main)
        run_summary["outputs"]["aggregated_main"] = os.path.basename(agg_main_path)

        t2 = generate_main_table(agg_main)
        t2_path = os.path.join(args.output, "table2_content.tex")
        with open(t2_path, "w") as f:
            f.write(t2)
        run_summary["outputs"]["table2"] = os.path.basename(t2_path)

    if "robustness" in args.stages and agg_rob_risk and agg_rob_pool and agg_rob_fee:
        agg_rob_risk_path = os.path.join(args.output, "aggregated_robustness_risk.json")
        agg_rob_pool_path = os.path.join(args.output, "aggregated_robustness_pool.json")
        agg_rob_fee_path = os.path.join(args.output, "aggregated_robustness_fee.json")
        _write_json(agg_rob_risk_path, agg_rob_risk)
        _write_json(agg_rob_pool_path, agg_rob_pool)
        _write_json(agg_rob_fee_path, agg_rob_fee)
        run_summary["outputs"]["aggregated_robustness_risk"] = os.path.basename(agg_rob_risk_path)
        run_summary["outputs"]["aggregated_robustness_pool"] = os.path.basename(agg_rob_pool_path)
        run_summary["outputs"]["aggregated_robustness_fee"] = os.path.basename(agg_rob_fee_path)

        t3_risk = generate_robustness_table(agg_rob_risk)
        t3_pool = generate_robustness_table(agg_rob_pool)
        t3_fee = generate_robustness_table(agg_rob_fee)
        t3_risk_path = os.path.join(args.output, "table3_risk_content.tex")
        t3_pool_path = os.path.join(args.output, "table3_pool_content.tex")
        t3_fee_path = os.path.join(args.output, "table3_fee_content.tex")
        with open(t3_risk_path, "w") as f:
            f.write(t3_risk)
        with open(t3_pool_path, "w") as f:
            f.write(t3_pool)
        with open(t3_fee_path, "w") as f:
            f.write(t3_fee)
        run_summary["outputs"]["table3_risk"] = os.path.basename(t3_risk_path)
        run_summary["outputs"]["table3_pool"] = os.path.basename(t3_pool_path)
        run_summary["outputs"]["table3_fee"] = os.path.basename(t3_fee_path)

    main_episode_rows = _load_main_episode_rows(main_episode_metric_paths) if "main" in args.stages else []
    if "main" in args.stages:
        if main_episode_rows:
            behavior_probe = _build_behavior_probe(main_episode_rows, args.baseline_methods)
            behavior_probe_path = os.path.join(args.output, "behavior_probe.json")
            _write_json(behavior_probe_path, behavior_probe)
            run_summary["outputs"]["behavior_probe"] = os.path.basename(behavior_probe_path)

        merged_case_study = _merge_case_studies(case_study_seed_paths)
        if merged_case_study["seed_case_studies"]:
            case_study_path = os.path.join(args.output, "case_study.json")
            _write_json(case_study_path, merged_case_study)
            run_summary["outputs"]["case_study"] = os.path.basename(case_study_path)

        if SCIPY_AVAILABLE:
            if main_episode_rows:
                print("\n===== Episode 级配对显著性检验 =====")
                sig = run_paired_significance_tests(main_episode_rows)
                print(format_paired_significance_table(sig))
                sig_path = os.path.join(args.output, "paired_significance_tests.json")
                _write_json(sig_path, sig)
                sig_tex = os.path.join(args.output, "table_significance_paired.tex")
                generate_paired_significance_latex(sig, sig_tex)
                run_summary["outputs"]["paired_significance_json"] = os.path.basename(sig_path)
                run_summary["outputs"]["paired_significance_table"] = os.path.basename(sig_tex)
            else:
                print("\n===== Episode 级配对显著性检验 =====")
                print("Warning: 未找到 main_episode_metrics.json，已跳过。")
        else:
            print("\n===== Episode 级配对显著性检验 =====")
            print("Warning: scipy 未安装, 跳过显著性检验与对应 LaTeX 表格生成。")

    if "ablation" in args.stages:
        print("\n===== 奖励消融实验 =====", flush=True)
        t_abl_reward = time.perf_counter()
        agg_reward_abl = run_ablation_parallel(REWARD_ABLATION_CONFIGS, args, "reward")
        ablation_reward_seconds = time.perf_counter() - t_abl_reward
        t_rw = generate_ablation_table(agg_reward_abl, ABLATION_ORDER, ABLATION_LABELS)
        rw_path = os.path.join(args.output, "table_ablation_reward.tex")
        with open(rw_path, "w") as f:
            f.write(t_rw)
        abl_rw_json = os.path.join(args.output, "ablation_reward.json")
        _write_json(abl_rw_json, agg_reward_abl)

        print("\n===== 结构消融实验 =====", flush=True)
        t_abl_struct = time.perf_counter()
        agg_struct_abl = run_ablation_parallel(STRUCT_ABLATION_CONFIGS, args, "struct")
        ablation_struct_seconds = time.perf_counter() - t_abl_struct
        t_st = generate_ablation_table(agg_struct_abl, STRUCT_ABLATION_ORDER, STRUCT_ABLATION_LABELS)
        st_path = os.path.join(args.output, "table_ablation_struct.tex")
        with open(st_path, "w") as f:
            f.write(t_st)
        abl_st_json = os.path.join(args.output, "ablation_struct.json")
        _write_json(abl_st_json, agg_struct_abl)

        run_summary["outputs"]["ablation_reward_json"] = os.path.basename(abl_rw_json)
        run_summary["outputs"]["ablation_struct_json"] = os.path.basename(abl_st_json)
        run_summary["outputs"]["ablation_reward_table"] = os.path.basename(rw_path)
        run_summary["outputs"]["ablation_struct_table"] = os.path.basename(st_path)
        print(f"消融表格已保存: {rw_path}, {st_path}")

    timing_payload = {
        "timestamp": datetime.now().isoformat(),
        "seed_timings": seed_timings,
        "stage_totals_seconds": {
            "train": float(sum(t.get("train", 0.0) for t in seed_timings.values())),
            "eval": float(sum(t.get("eval", 0.0) for t in seed_timings.values())),
            "robustness": float(sum(t.get("robustness", 0.0) for t in seed_timings.values())),
            "ablation_reward": float(ablation_reward_seconds),
            "ablation_struct": float(ablation_struct_seconds),
        },
        "total_runtime_seconds": float(time.perf_counter() - run_start),
    }
    timing_path = os.path.join(args.output, "timing.json")
    _write_json(timing_path, timing_payload)
    run_summary["outputs"]["timing"] = os.path.basename(timing_path)

    run_summary_path = os.path.join(args.output, "run_summary.json")
    _write_json(run_summary_path, run_summary)
    print(f"运行摘要已保存: {run_summary_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
