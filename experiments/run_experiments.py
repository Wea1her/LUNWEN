"""一键实验编排: 多种子训练 → 评估 → 聚合 → LaTeX 表格"""

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
                      load_shared_pool_metadata,
                      evaluate_rl, evaluate_baseline, aggregate,
                      build_fairness_decomposition, build_constrained_eval_summary,
                      build_operating_points_summary, build_constraint_bottleneck_report,
                      plot_training_curve)
from baselines import run_baseline, grid_search_baseline_params
from metrics import (compute_all_metrics, constrained_ranking, pareto_dominates,
                     summary_metric_bundle)
from latex_tables import (generate_main_table, generate_robustness_table,
                          generate_ablation_table, generate_composite_table,
                          generate_tradeoff_table,
                          generate_fairness_decomp_table, generate_constrained_main_table,
                          generate_main_core_table, generate_main_fullmetrics_table,
                          generate_pareto_main_table, generate_protocol_ablation_table,
                          generate_operating_points_table, generate_constraint_bottleneck_table,
                          generate_baseline_params_table,
                          append_exploratory_note, append_primary_decision_rule_note,
                          ABLATION_ORDER, PROTOCOL_ABLATION_ORDER,
                          ABLATION_LABELS, STRUCT_ABLATION_ORDER,
                          STRUCT_ABLATION_LABELS, PROTOCOL_ABLATION_LABELS)
from method_registry import (MAIN_METHOD_ORDER, get_baseline_method_ids, normalize_method_id,
                             method_registry_payload)
from stat_tests import (SCIPY_AVAILABLE, run_paired_significance_tests,
                        format_paired_significance_table,
                        generate_paired_significance_latex,
                        run_seed_level_paired_tests,
                        format_seed_level_significance_table,
                        generate_seed_level_significance_latex)

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


def _shared_pool_path(output_dir: str, role: str, filename: str) -> str:
    path = os.path.join(output_dir, "pools", role, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


RISK_TUNING_FIELDS = {
    "gamma_r": "GAMMA_R",
    "terminal_risk_exposure_weight": "TERMINAL_RISK_EXPOSURE_WEIGHT",
    "terminal_top10_risk_weight": "TERMINAL_TOP10_RISK_WEIGHT",
    "risk_adjusted_fee_lambda": "RISK_ADJUSTED_FEE_LAMBDA",
}


def _risk_tuning_payload(args: argparse.Namespace) -> dict:
    payload = {}
    label = getattr(args, "risk_tune_label", None)
    if label:
        payload["label"] = label
    for attr, const_name in RISK_TUNING_FIELDS.items():
        value = getattr(args, attr, None)
        if value is not None:
            payload[attr] = float(value)
            payload[f"baseline_{attr}"] = float(getattr(C, const_name))
    return payload


def _apply_risk_tuning_overrides(args: argparse.Namespace) -> dict:
    payload = _risk_tuning_payload(args)
    for attr, const_name in RISK_TUNING_FIELDS.items():
        value = getattr(args, attr, None)
        if value is not None:
            setattr(C, const_name, float(value))
    return payload


def _risk_env_kwargs_from_args(args: argparse.Namespace) -> dict:
    env_kwargs = {}
    for attr in ("gamma_r", "terminal_risk_exposure_weight", "terminal_top10_risk_weight"):
        value = getattr(args, attr, None)
        if value is not None:
            env_kwargs[attr] = float(value)
    return env_kwargs


def _get_or_create_shared_pools(
    path: str,
    n_episodes: int,
    pool_size: int,
    risk_ratio: float,
    generation_seed: int,
    metadata: dict,
) -> tuple[list[list], dict]:
    """加载已冻结共享池，或首次生成并落盘。"""
    if os.path.exists(path):
        pools = load_shared_pools(path)
        meta = load_shared_pool_metadata(path)
        if len(pools) != n_episodes:
            raise ValueError(
                f"frozen pool episode mismatch for {path}: "
                f"expected {n_episodes}, got {len(pools)}"
            )
        return pools, meta
    pools = build_shared_pools(n_episodes, pool_size, risk_ratio, generation_seed)
    meta = dict(metadata)
    meta["generation_seed"] = generation_seed
    pool_hash = save_shared_pools(path, pools, metadata=meta)
    meta["pool_hash"] = pool_hash
    meta["n_episodes"] = n_episodes
    return pools, meta


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


def _evaluate_method_on_pool(
    pool: list,
    method_id: str,
    model,
    device,
    env: TxOrderingEnv,
    baseline_params: dict | None = None,
) -> dict:
    if method_id == "ours":
        obs, _ = env.reset_with_pool(pool)
        while True:
            action, _, _ = model.act(obs, device, greedy=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        selected = env.get_selected_transactions()
    else:
        params = (baseline_params or {}).get(method_id, {})
        selected = run_baseline(deepcopy(pool), method_id, params=params)
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
    baseline_params: dict | None = None,
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
            method_details[method_id] = _evaluate_method_on_pool(
                pool, method_id, model, device, env, baseline_params=baseline_params
            )
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


def _run_main_evaluation(model, device, args, seed, result_dir, baseline_params=None):
    env = TxOrderingEnv(pool_size=args.pool_size, risk_ratio=C.RISK_RATIO, seed=seed)
    test_seed = seed + getattr(C, "TEST_POOL_SEED_OFFSET", 20000)
    shared_pool_path = _shared_pool_path(
        args.output,
        "test",
        f"test_pool_seed_{seed}_main.json",
    )
    shared_pools, test_pool_meta = _get_or_create_shared_pools(
        shared_pool_path,
        args.eval_episodes,
        args.pool_size,
        C.RISK_RATIO,
        test_seed,
        metadata={
            "pool_role": "test",
            "seed": seed,
            "eval_episodes": args.eval_episodes,
            "pool_size": args.pool_size,
            "risk_ratio": C.RISK_RATIO,
            "setting": "main",
            "baseline_methods": args.baseline_methods,
            "frozen": True,
        },
    )

    raw = {"ours": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
    for bl in args.baseline_methods:
        raw[bl] = evaluate_baseline(
            bl, env, args.eval_episodes, shared_pools,
            baseline_params=(baseline_params or {}).get(bl, {}),
        )
    agg = {method: aggregate(metrics) for method, metrics in raw.items()}

    _write_json(os.path.join(result_dir, "main_results.json"), agg)
    _write_json(os.path.join(result_dir, "main_aggregated_metrics.json"), {
        "protocol_version": getattr(C, "EXPERIMENT_PROTOCOL_VERSION", "unknown"),
        "test_pool": {
            "path": shared_pool_path,
            "metadata": test_pool_meta,
        },
        "metrics": agg,
    })
    _write_json(
        os.path.join(result_dir, "fairness_decomposition.json"),
        build_fairness_decomposition(raw),
    )
    constrained_seed = build_constrained_eval_summary(raw)
    _write_json(os.path.join(result_dir, "constrained_eval_summary.json"), constrained_seed)
    _write_json(
        os.path.join(result_dir, "operating_points_summary.json"),
        build_operating_points_summary(constrained_seed),
    )
    _write_json(
        os.path.join(result_dir, "constraint_bottleneck_report.json"),
        build_constraint_bottleneck_report(constrained_seed),
    )
    _write_json(os.path.join(result_dir, "main_episode_metrics.json"), {
        "protocol_version": getattr(C, "EXPERIMENT_PROTOCOL_VERSION", "unknown"),
        "seed": seed,
        "setting": "main",
        "test_pool": {
            "path": shared_pool_path,
            "metadata": test_pool_meta,
        },
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
        baseline_params=baseline_params,
    )
    return agg, case_study_seed_path


def _run_risk_robustness(model, device, args, seed, result_dir, baseline_params=None):
    all_data = {}
    all_rows = []
    pool_index = {
        "seed": seed,
        "dimension": "risk_ratio",
        "files": {},
    }

    for rr in C.ROBUSTNESS_RISK_RATIOS:
        env = TxOrderingEnv(pool_size=args.pool_size, risk_ratio=rr, seed=seed)
        rr_tag = str(rr).replace(".", "p")
        rr_pool_file = _shared_pool_path(args.output, "test", f"test_pool_seed_{seed}_robust_risk_{rr_tag}.json")
        shared_pools, rr_pool_meta = _get_or_create_shared_pools(
            rr_pool_file,
            args.eval_episodes,
            args.pool_size,
            rr,
            seed + getattr(C, "TEST_POOL_SEED_OFFSET", 20000) + 1000 + int(round(rr * 1000)),
            metadata={
                "pool_role": "test",
                "seed": seed,
                "eval_episodes": args.eval_episodes,
                "pool_size": args.pool_size,
                "risk_ratio": rr,
                "setting": "robustness_risk",
                "frozen_model_from_default_scene": True,
                "baseline_methods": args.baseline_methods,
            },
        )
        pool_index["files"][str(rr)] = rr_pool_file
        pool_index.setdefault("metadata", {})[str(rr)] = rr_pool_meta

        raw = {"ours": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
        for bl in args.baseline_methods:
            raw[bl] = evaluate_baseline(
                bl, env, args.eval_episodes, shared_pools,
                baseline_params=(baseline_params or {}).get(bl, {}),
            )
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


def _run_pool_robustness(model, device, args, seed, result_dir, baseline_params=None):
    all_data = {}
    all_rows = []
    pool_index = {
        "seed": seed,
        "dimension": "pool_size",
        "files": {},
    }

    for ps in C.ROBUSTNESS_POOL_SIZES:
        env = TxOrderingEnv(pool_size=ps, risk_ratio=C.RISK_RATIO, seed=seed)
        pool_file = _shared_pool_path(args.output, "test", f"test_pool_seed_{seed}_robust_pool_N{ps}.json")
        shared_pools, pool_meta = _get_or_create_shared_pools(
            pool_file,
            args.eval_episodes,
            ps,
            C.RISK_RATIO,
            seed + getattr(C, "TEST_POOL_SEED_OFFSET", 20000) + 2000 + int(ps),
            metadata={
                "pool_role": "test",
                "seed": seed,
                "eval_episodes": args.eval_episodes,
                "pool_size": ps,
                "risk_ratio": C.RISK_RATIO,
                "setting": "robustness_pool_size",
                "frozen_model_from_default_scene": True,
                "baseline_methods": args.baseline_methods,
            },
        )
        pool_index["files"][str(ps)] = pool_file
        pool_index.setdefault("metadata", {})[str(ps)] = pool_meta

        raw = {"ours": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
        for bl in args.baseline_methods:
            raw[bl] = evaluate_baseline(
                bl, env, args.eval_episodes, shared_pools,
                baseline_params=(baseline_params or {}).get(bl, {}),
            )
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


def _run_fee_robustness(model, device, args, seed, result_dir, baseline_params=None):
    all_data = {}
    all_rows = []
    base_pool_file = _shared_pool_path(args.output, "test", f"test_pool_seed_{seed}_robust_fee_base.json")
    base_shared_pools, base_pool_meta = _get_or_create_shared_pools(
        base_pool_file,
        args.eval_episodes,
        args.pool_size,
        C.RISK_RATIO,
        seed + getattr(C, "TEST_POOL_SEED_OFFSET", 20000) + 3000,
        metadata={
            "pool_role": "test",
            "seed": seed,
            "eval_episodes": args.eval_episodes,
            "pool_size": args.pool_size,
            "risk_ratio": C.RISK_RATIO,
            "setting": "robustness_fee_multiplier_base",
            "frozen_model_from_default_scene": True,
            "baseline_methods": args.baseline_methods,
        },
    )
    _write_json(os.path.join(result_dir, f"shared_pools_robust_fee_seed{seed}.json"), {
        "seed": seed,
        "dimension": "fee_multiplier",
        "base_shared_pool_file": base_pool_file,
        "base_shared_pool_metadata": base_pool_meta,
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
                selected_bl = run_baseline(
                    deepcopy(pool), bl, params=(baseline_params or {}).get(bl, {})
                )
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


def _prepare_baseline_params_for_seed(seed: int, args: argparse.Namespace, result_dir: str) -> tuple[dict, dict]:
    """在固定验证池上为含参数基线选参，并落盘。"""
    default_params = {method: {} for method in args.baseline_methods}
    paths = {}
    if getattr(args, "skip_baseline_tuning", False) or not getattr(C, "BASELINE_TUNING_ENABLED", True):
        best_path = os.path.join(result_dir, "baseline_best_params.json")
        payload = {
            "enabled": False,
            "reason": "disabled_by_cli_or_config",
            "best_params": default_params,
        }
        _write_json(best_path, payload)
        table_path = os.path.join(result_dir, "table_baseline_params.tex")
        with open(table_path, "w") as f:
            f.write(generate_baseline_params_table(default_params))
        return default_params, {"baseline_best_params": best_path, "table_baseline_params": table_path}

    validation_seed = seed + getattr(C, "VALIDATION_SEED_OFFSET", 10000) + 7000
    val_pool_path = _shared_pool_path(
        args.output,
        "validation",
        f"validation_pool_seed_{seed}_baseline_tuning.json",
    )
    val_pools, val_meta = _get_or_create_shared_pools(
        val_pool_path,
        args.val_episodes,
        args.pool_size,
        C.RISK_RATIO,
        validation_seed,
        metadata={
            "pool_role": "validation",
            "seed": seed,
            "val_episodes": args.val_episodes,
            "pool_size": args.pool_size,
            "risk_ratio": C.RISK_RATIO,
            "setting": "baseline_tuning",
            "test_pool_not_used": True,
            "baseline_methods": args.baseline_methods,
        },
    )
    constraints = {
        "fairness_floor": args.val_fairness_floor,
        "oldest_coverage_floor": args.val_oldest_coverage_floor,
        "risk_ceil": args.val_risk_ceil,
        "edge10_risk_ceil": args.val_edge10_risk_ceil,
        "top10_risk_ceil": args.val_top10_risk_ceil,
    }
    tuning, best_params = grid_search_baseline_params(
        args.baseline_methods,
        val_pools,
        constraints=constraints,
        operating_mode=getattr(args, "operating_mode", getattr(C, "OPERATING_MODE", "balanced")),
    )
    tuning_path = os.path.join(result_dir, "baseline_tuning.json")
    best_path = os.path.join(result_dir, "baseline_best_params.json")
    table_path = os.path.join(result_dir, "table_baseline_params.tex")
    _write_json(tuning_path, {
        "enabled": True,
        "selection_pool": {"path": val_pool_path, "metadata": val_meta},
        "selection_metric": "two_stage_selection_score",
        "constraints": constraints,
        "operating_mode": getattr(args, "operating_mode", getattr(C, "OPERATING_MODE", "balanced")),
        "tuning": tuning,
    })
    _write_json(best_path, {
        "enabled": True,
        "selection_pool": {"path": val_pool_path, "metadata": val_meta},
        "best_params": best_params,
    })
    with open(table_path, "w") as f:
        f.write(generate_baseline_params_table(best_params))
    paths.update({
        "baseline_tuning": tuning_path,
        "baseline_best_params": best_path,
        "table_baseline_params": table_path,
        "baseline_validation_pool": val_pool_path,
    })
    return best_params, paths


def run_single_seed(seed, args_dict):
    """单个种子: 训练 + 分阶段评估 (独立进程)"""
    args = argparse.Namespace(**args_dict)
    risk_tuning = _apply_risk_tuning_overrides(args)
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
    if risk_tuning:
        seed_summary["risk_tuning"] = risk_tuning
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
        train_env_kwargs = _risk_env_kwargs_from_args(args)
        if hasattr(C, "resolve_constraints"):
            train_cons = C.resolve_constraints(profile=args.training_constraint_profile, for_training=True)
            train_env_kwargs["fairness_gate_threshold"] = float(
                train_cons.get("fairness_floor", C.FAIRNESS_GATE_THRESHOLD)
            )
            train_env_kwargs["fairness_oldest_coverage_floor"] = float(
                train_cons.get("oldest_coverage_floor", C.FAIRNESS_OLDEST_COVERAGE_FLOOR)
            )
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
            val_fairness_floor=args.val_fairness_floor,
            val_oldest_coverage_floor=args.val_oldest_coverage_floor,
            val_risk_ceil=args.val_risk_ceil,
            val_edge10_risk_ceil=args.val_edge10_risk_ceil,
            val_top10_risk_ceil=args.val_top10_risk_ceil,
            pretrain_policy=args.pretrain_policy,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_episodes_per_epoch=args.pretrain_episodes_per_epoch,
            curriculum=args.curriculum,
            curriculum_stage_episodes=args.curriculum_stage_episodes,
            fairness_first=args.fairness_first,
            operating_mode=args.operating_mode,
        )
        t0 = time.perf_counter()
        train_model(train_args, env_kwargs=train_env_kwargs)
        timing["train"] = time.perf_counter() - t0
        print(f"[seed={seed}] 训练完成", flush=True)

    main_results = None
    rob_risk = None
    rob_pool = None
    rob_fee = None
    main_episode_metrics_path = None
    case_study_seed_path = None
    baseline_best_params = {method: {} for method in getattr(args, "baseline_methods", [])}
    baseline_tuning_paths = {}
    requires_policy = bool(stages & {"main", "robustness"})

    if requires_policy:
        baseline_best_params, baseline_tuning_paths = _prepare_baseline_params_for_seed(seed, args, result_dir)
        seed_summary["baseline_best_params"] = baseline_best_params
        seed_summary["baseline_tuning_outputs"] = {
            key: os.path.relpath(path, args.output) for key, path in baseline_tuning_paths.items()
        }

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
            main_results, case_study_seed_path = _run_main_evaluation(
                model, device, args, seed, result_dir, baseline_params=baseline_best_params
            )
            timing["eval"] = time.perf_counter() - t0
            main_episode_metrics_path = os.path.join(result_dir, "main_episode_metrics.json")

        if "robustness" in stages:
            print(f"[seed={seed}] 鲁棒性实验 (风险比例)", flush=True)
            t0 = time.perf_counter()
            rob_risk = _run_risk_robustness(
                model, device, args, seed, result_dir, baseline_params=baseline_best_params
            )
            timing["robustness_risk"] = time.perf_counter() - t0
            print(f"[seed={seed}] 鲁棒性实验 (池大小)", flush=True)
            t1 = time.perf_counter()
            rob_pool = _run_pool_robustness(
                model, device, args, seed, result_dir, baseline_params=baseline_best_params
            )
            timing["robustness_pool"] = time.perf_counter() - t1
            print(f"[seed={seed}] 鲁棒性实验 (费率倍率)", flush=True)
            t2 = time.perf_counter()
            rob_fee = _run_fee_robustness(
                model, device, args, seed, result_dir, baseline_params=baseline_best_params
            )
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
        "baseline_best_params": baseline_best_params,
        "baseline_tuning_paths": baseline_tuning_paths,
    }


def _metric_stat_value(stats: dict, metric: str):
    """兼容普通 episode 指标和已聚合的特殊指标。"""
    mean_key = f"{metric}_mean"
    if mean_key in stats:
        return stats[mean_key]
    if metric in stats:
        return stats[metric]
    return None


def _v5_metric_list() -> list[str]:
    return [
        "block_fee",
        "fairness",
        "risk_exposure",
        "edge_risk_ratio",
        "edge10_risk",
        "risky_inclusion_rate",
        "gas_util",
        "risky_rank",
        "packing_ratio",
        "top10_risk",
        "late_promo",
        "oldest_coverage",
        "old_tx_pack_rate",
        "starvation_gap",
        "starvation_ratio",
        "tail_wait_reduction",
        "selected_wait_std",
        "wait_p95",
        "wait_p99",
        "wait_gini",
        "composite_score",
        "constrained_fee_score",
        "risk_adjusted_fee_score",
        "risk_safety_score",
        "edge_risk_safety_score",
        "trade_score",
        "risk_aware_trade_score",
        "constrained_trade_score",
        "invalid_action_count",
        "invalid_action_rate",
        "invalid_truncation_count",
        "invalid_truncation_rate",
        "max_invalid_streak",
        "mean_consecutive_invalid_actions",
        "mean_inference_time",
        "p95_inference_time",
        "max_inference_time",
    ]


def aggregate_metric_grid(all_results):
    """聚合形如 {setting: {method: metric_stats}} 的结果网格。"""
    all_settings = set()
    for result in all_results:
        all_settings.update(result.keys())
    settings = sorted(all_settings, key=str)
    metrics = _v5_metric_list()

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
                    value = _metric_stat_value(result[setting][method], metric)
                    if value is not None:
                        vals.append(value)
                if vals:
                    agg[setting][method][f"{metric}_mean"] = float(np.mean(vals))
                    agg[setting][method][f"{metric}_std"] = float(np.std(vals))
                else:
                    agg[setting][method][f"{metric}_mean"] = float("nan")
                    agg[setting][method][f"{metric}_std"] = float("nan")
    return agg


def aggregate_across_seeds(all_main=None, all_rob_risk=None, all_rob_pool=None, all_rob_fee=None):
    """跨种子聚合: 对每个方法的 mean 取均值和标准差"""
    metrics = _v5_metric_list()

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
                    value = _metric_stat_value(result[method], metric)
                    if value is not None:
                        vals.append(value)
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
    "Ours-AgeOnly": {
        "alpha": C.ALPHA,
        "beta_age": C.BETA_AGE,
        "beta_oldest_cover": 0.0,
        "beta_terminal_fair": 0.0,
        "gamma_r": 0.0,
        "gamma_starvation": 0.0,
    },
    "Ours-Age+Risk": {
        "alpha": C.ALPHA,
        "beta_age": C.BETA_AGE,
        "beta_oldest_cover": 0.0,
        "beta_terminal_fair": 0.0,
        "gamma_r": C.GAMMA_R,
        "gamma_starvation": 0.0,
    },
    "Ours-Age+TerminalFair": {
        "alpha": C.ALPHA,
        "beta_age": C.BETA_AGE,
        "beta_oldest_cover": 0.0,
        "beta_terminal_fair": C.BETA_TERMINAL_FAIR,
        "gamma_r": 0.0,
        "gamma_starvation": 0.0,
    },
    "Ours-FullBalanced": {
        "alpha": C.ALPHA,
        "beta_age": C.BETA_AGE,
        "beta_oldest_cover": C.BETA_OLDEST_COVER,
        "beta_terminal_fair": C.BETA_TERMINAL_FAIR,
        "gamma_r": C.GAMMA_R,
        "gamma_starvation": C.GAMMA_STARVATION,
    },
}

STRUCT_ABLATION_CONFIGS = {
    "No-SeqSummary":  {"no_seq_summary": True,  "no_stop": False, "no_action_mask": False},
    "No-ActionMask":  {"no_seq_summary": False, "no_stop": False, "no_action_mask": True},
    "No-STOP":        {"no_seq_summary": False, "no_stop": True,  "no_action_mask": False},
    "Ours-Full":      {"no_seq_summary": False, "no_stop": False, "no_action_mask": False},
}

PROTOCOL_ABLATION_CONFIGS = {
    "Proto-Composite-NoWarm-NoCurr": {
        "env_kwargs": {},
        "train_overrides": {
            "val_metric": "composite_score",
            "pretrain_policy": "none",
            "pretrain_epochs": 0,
            "curriculum": False,
        },
    },
    "Proto-Constrained-NoWarm-NoCurr": {
        "env_kwargs": {},
        "train_overrides": {
            "val_metric": "constrained_fee",
            "pretrain_policy": "none",
            "pretrain_epochs": 0,
            "curriculum": False,
        },
    },
    "Proto-Constrained-Warm-Curr": {
        "env_kwargs": {},
        "train_overrides": {
            "val_metric": "constrained_fee",
            "pretrain_policy": "mixed",
            "pretrain_epochs": 1,
            "curriculum": True,
        },
    },
    "Proto-Constrained-NoFairGate": {
        "env_kwargs": {
            "fairness_gate_type": "hard",
            "fairness_gate_min": 1.0,
        },
        "train_overrides": {
            "val_metric": "constrained_fee",
            "pretrain_policy": "mixed",
            "pretrain_epochs": 1,
            "curriculum": True,
        },
    },
    "Proto-Constrained-NoTerminalRisk": {
        "env_kwargs": {
            "terminal_risk_exposure_weight": 0.0,
            "terminal_top10_risk_weight": 0.0,
            "terminal_risky_rank_dev_weight": 0.0,
        },
        "train_overrides": {
            "val_metric": "constrained_fee",
            "pretrain_policy": "mixed",
            "pretrain_epochs": 1,
            "curriculum": True,
        },
    },
    "Proto-Hypervolume-Warm-Curr": {
        "env_kwargs": {},
        "train_overrides": {
            "val_metric": "hypervolume",
            "pretrain_policy": "mixed",
            "pretrain_epochs": 1,
            "curriculum": True,
        },
    },
}


def _train_ablation_single(
    name,
    seed,
    args_dict,
    env_kwargs=None,
    train_overrides=None,
    shared_pool_path=None,
):
    """单个消融变体+种子的训练与评估 (独立进程)"""
    args = argparse.Namespace(**args_dict)
    env_kwargs = dict(env_kwargs or {})
    train_overrides = dict(train_overrides or {})
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
        train_args_dict = {
            "episodes": args.episodes,
            "pool_size": args.pool_size,
            "risk_ratio": C.RISK_RATIO,
            "seed": seed,
            "log_interval": C.LOG_INTERVAL,
            "output": ckpt_dir,
            "device": args.device,
            "val_episodes": args.val_episodes,
            "val_interval": args.val_interval,
            "val_metric": args.val_metric,
            "val_fairness_floor": args.val_fairness_floor,
            "val_oldest_coverage_floor": args.val_oldest_coverage_floor,
            "val_risk_ceil": args.val_risk_ceil,
            "val_top10_risk_ceil": args.val_top10_risk_ceil,
            "pretrain_policy": args.pretrain_policy,
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_episodes_per_epoch": args.pretrain_episodes_per_epoch,
            "curriculum": args.curriculum,
            "curriculum_stage_episodes": args.curriculum_stage_episodes,
        }
        train_args_dict.update(train_overrides)
        train_args = argparse.Namespace(**train_args_dict)
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


def run_ablation_parallel(configs, args, label_prefix, metrics=None):
    """并行运行消融实验"""
    if metrics is None:
        metrics = ["block_fee", "fairness", "risk_exposure", "gas_util"]
    args_dict = vars(args)
    parsed_configs = {}
    for name, config in configs.items():
        if isinstance(config, dict) and ("env_kwargs" in config or "train_overrides" in config):
            parsed_configs[name] = {
                "env_kwargs": dict(config.get("env_kwargs", {})),
                "train_overrides": dict(config.get("train_overrides", {})),
            }
        else:
            parsed_configs[name] = {
                "env_kwargs": dict(config or {}),
                "train_overrides": {},
            }

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

    max_workers = max(1, min(args.workers, len(parsed_configs) * len(args.seeds)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, config_payload in parsed_configs.items():
            for seed in args.seeds:
                fut = executor.submit(
                    _train_ablation_single,
                    f"{label_prefix}_{name}",
                    seed,
                    args_dict,
                    config_payload["env_kwargs"],
                    config_payload["train_overrides"],
                    shared_pool_paths[seed],
                )
                futures[fut] = (name, seed)

        results = {name: [] for name in parsed_configs}
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
    for name in parsed_configs:
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


def _infer_pareto_tag(delta_vs_full: dict) -> str:
    fee_delta = float(delta_vs_full.get("block_fee", {}).get("mean_delta", 0.0))
    fairness_delta = float(delta_vs_full.get("fairness", {}).get("mean_delta", 0.0))
    risk_delta = float(delta_vs_full.get("risk_exposure", {}).get("mean_delta", 0.0))
    top10_delta = float(delta_vs_full.get("top10_risk", {}).get("mean_delta", 0.0))

    risk_improved = (risk_delta < 0.0) or (top10_delta < 0.0)
    if fee_delta > 0.0 and fairness_delta <= 0.0 and not risk_improved:
        return "Fee-lean"
    if fairness_delta > 0.0 and fee_delta <= 0.0 and risk_improved:
        return "Fair-lean"
    if risk_improved and fee_delta <= 0.0:
        return "Risk-lean"
    return "Balanced-shift"


def _annotate_ablation_tradeoff(agg: dict, full_key: str, core_metrics: list[str]) -> tuple[dict, bool]:
    if full_key not in agg:
        return agg, False
    full_payload = agg[full_key]
    tradeoff_found = False
    lower_is_better = {"risk_exposure", "top10_risk", "starvation_gap"}
    for variant, payload in agg.items():
        delta = {}
        n_better = 0
        for metric in core_metrics:
            metric_key = f"{metric}_mean"
            if metric_key not in payload or metric_key not in full_payload:
                continue
            mean_delta = float(payload[metric_key]) - float(full_payload[metric_key])
            higher_better = metric not in lower_is_better
            is_better = mean_delta > 0.0 if higher_better else mean_delta < 0.0
            if variant != full_key and is_better:
                n_better += 1
            delta[metric] = {
                "mean_delta": mean_delta,
                "is_better_than_full": bool(is_better),
                "higher_is_better": bool(higher_better),
            }
        payload["delta_vs_full"] = delta
        if variant == full_key:
            payload["pareto_tag"] = "Full-Reference"
            payload["n_better_core_metrics"] = 0
            continue
        payload["pareto_tag"] = _infer_pareto_tag(delta)
        payload["n_better_core_metrics"] = n_better
        if n_better >= 1:
            tradeoff_found = True
    return agg, tradeoff_found


def _resolve_stages(args):
    stages = set(args.stages or [])
    if args.ablation:
        stages.add("ablation")
    if not stages:
        stages.add("main")
    args.stages = [stage for stage in STAGE_ORDER if stage in stages]


def _apply_track_defaults(args):
    if args.track == "fairness_recovery_track":
        if args.val_metric == C.VALIDATION_METRIC:
            args.val_metric = "constrained_fee"
        args.val_fairness_floor = max(args.val_fairness_floor, 0.92)
        args.val_oldest_coverage_floor = max(args.val_oldest_coverage_floor, 0.95)
        args.val_risk_ceil = min(args.val_risk_ceil, 0.35)
        args.val_top10_risk_ceil = min(args.val_top10_risk_ceil, 0.35)
        return

    if args.track == "composite_optimal_track":
        if args.val_metric == C.VALIDATION_METRIC:
            args.val_metric = "two_stage"
        args.val_fairness_floor = max(args.val_fairness_floor, 0.90)
        args.val_oldest_coverage_floor = max(args.val_oldest_coverage_floor, 0.90)
        args.val_risk_ceil = min(args.val_risk_ceil, 0.30)
        args.val_top10_risk_ceil = min(args.val_top10_risk_ceil, 0.30)


def _infer_evidence_level(seed_runs: list[dict]) -> str:
    seed_set = {
        int(item.get("seed"))
        for item in seed_runs
        if item.get("seed") is not None and item.get("status") in {"success", "resumed", "skipped_existing"}
    }
    n = len(seed_set)
    if n <= 1:
        return "dryrun_single_seed"
    if n < int(getattr(C, "EVIDENCE_FORMAL_MIN_SEEDS", 3)):
        return "multi_seed_exploratory"
    return "formal_multi_seed"


def _metric_winner_by_dimension(agg_main: dict) -> dict:
    specs = [
        ("block_fee_mean", True),
        ("fairness_mean", True),
        ("risk_exposure_mean", False),
        ("edge10_risk_mean", False),  # main risk metric
        ("packing_ratio_mean", True),
        ("top10_risk_mean", False),  # diagnostic only
        ("composite_score_mean", True),
        ("constrained_fee_score_mean", True),
        ("trade_score_mean", True),
        ("risk_aware_trade_score_mean", True),
        ("constrained_trade_score_mean", True),
    ]
    winner_map = {}
    for metric, higher in specs:
        cand = [(m, d.get(metric)) for m, d in agg_main.items() if metric in d]
        if not cand:
            continue
        ordered = sorted(cand, key=lambda kv: kv[1], reverse=higher)
        winner_map[metric] = ordered[0][0]
    return winner_map


def _build_narrative_guard(agg_main: dict | None) -> dict:
    if not agg_main:
        return {
            "guard_version": "v20260324",
            "can_claim_all_dimensions_best": False,
            "blocking_reasons": ["missing_aggregated_main"],
            "metric_winner_by_dimension": {},
        }
    winners = _metric_winner_by_dimension(agg_main)
    non_ours = [m for m, winner in winners.items() if winner != "ours"]
    return {
        "guard_version": "v20260324",
        "can_claim_all_dimensions_best": len(non_ours) == 0,
        "blocking_reasons": [f"ours_not_best_on_{m}" for m in non_ours],
        "metric_winner_by_dimension": winners,
    }


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
        "fairness_decomposition.json",
        "constrained_eval_summary.json",
        "operating_points_summary.json",
        "constraint_bottleneck_report.json",
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


def _build_seed_level_statistics(main_episode_rows: list[dict], baseline_methods: list[str], evidence_level: str) -> dict:
    """V5 统计摘要：先按独立训练 seed 聚合，再比较方法差值。"""
    metrics = _v5_metric_list()
    methods = ["ours", *baseline_methods]
    by_seed: dict[int, dict[str, dict[str, float]]] = {}
    for row in main_episode_rows:
        if row.get("setting", "main") != "main":
            continue
        seed = row.get("seed")
        method = row.get("method")
        if seed is None or method not in methods:
            continue
        by_seed.setdefault(int(seed), {}).setdefault(method, {m: [] for m in metrics})
        row_metrics = row.get("metrics", {})
        for metric in metrics:
            value = row_metrics.get(metric)
            if value is not None:
                by_seed[int(seed)][method][metric].append(float(value))

    seed_means: dict[str, dict[str, dict[str, float]]] = {}
    for seed, method_payload in sorted(by_seed.items()):
        seed_means[str(seed)] = {}
        for method, metric_payload in method_payload.items():
            seed_means[str(seed)][method] = {
                metric: float(np.mean(values)) if values else float("nan")
                for metric, values in metric_payload.items()
            }

    lower_is_better = {
        "risk_exposure", "edge10_risk", "top10_risk", "starvation_ratio", "starvation_gap",
        "selected_wait_std", "wait_p95", "wait_p99", "wait_gini",
        "invalid_action_count", "invalid_action_rate", "invalid_truncation_count",
        "invalid_truncation_rate", "max_invalid_streak", "mean_consecutive_invalid_actions",
        "mean_inference_time", "p95_inference_time", "max_inference_time",
    }
    paired = {}
    for baseline in baseline_methods:
        paired[baseline] = {}
        for metric in metrics:
            diffs = []
            better_flags = []
            for seed_payload in seed_means.values():
                if "ours" not in seed_payload or baseline not in seed_payload:
                    continue
                ours = seed_payload["ours"].get(metric)
                base = seed_payload[baseline].get(metric)
                if ours is None or base is None or np.isnan(ours) or np.isnan(base):
                    continue
                diff = float(ours) - float(base)
                diffs.append(diff)
                if metric in lower_is_better:
                    better_flags.append(1.0 if diff < 0.0 else 0.0)
                else:
                    better_flags.append(1.0 if diff > 0.0 else 0.0)
            paired[baseline][metric] = {
                "n_seed_pairs": len(diffs),
                "mean_delta": float(np.mean(diffs)) if diffs else float("nan"),
                "std_delta": float(np.std(diffs)) if diffs else float("nan"),
                "ours_better_seed_rate": float(np.mean(better_flags)) if better_flags else float("nan"),
                "higher_is_better": metric not in lower_is_better,
            }

    n_seeds = len(seed_means)
    formal_min = int(getattr(C, "EVIDENCE_FORMAL_MIN_SEEDS", 5))
    return {
        "protocol_version": getattr(C, "EXPERIMENT_PROTOCOL_VERSION", "unknown"),
        "statistical_unit": "independent_training_seed",
        "episode_role": "within-seed evaluation samples, not independent training repeats",
        "evidence_level": evidence_level,
        "formal_statistics_ready": evidence_level == "formal_multi_seed" and n_seeds >= formal_min,
        "formal_min_seeds": formal_min,
        "n_completed_seeds": n_seeds,
        "metrics": metrics,
        "seed_means": seed_means,
        "paired_deltas_ours_vs_baselines_by_seed": paired,
    }


def _build_behavior_probe(main_episode_rows: list[dict], baseline_methods: list[str]) -> dict:
    metrics = _v5_metric_list()
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
                if metric in {
                    "risk_exposure", "edge10_risk", "top10_risk", "starvation_gap", "starvation_ratio",
                    "selected_wait_std", "wait_p95", "wait_p99", "wait_gini",
                    "invalid_action_count", "invalid_action_rate", "invalid_truncation_count",
                    "invalid_truncation_rate", "max_invalid_streak", "mean_consecutive_invalid_actions",
                    "mean_inference_time", "p95_inference_time", "max_inference_time",
                }:
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


def _build_fairness_decomposition_from_rows(main_episode_rows: list[dict], baseline_methods: list[str]) -> dict:
    metrics = [
        "fairness",
        "oldest_coverage",
        "starvation_gap",
        "starvation_ratio",
        "tail_wait_reduction",
        "selected_wait_std",
        "wait_p95",
        "wait_p99",
        "wait_gini",
        "composite_score",
    ]
    payload = {
        "settings": {
            "oldest_ratio": C.FAIR_OLDEST_RATIO,
            "tail_quantile": C.FAIR_TAIL_QUANTILE,
        },
        "metrics": metrics,
        "methods": {},
    }

    for method_id in ["ours", *baseline_methods]:
        rows = [
            row for row in main_episode_rows
            if row.get("setting", "main") == "main" and row.get("method") == method_id
        ]
        payload["methods"][method_id] = {}
        for metric in metrics:
            vals = [float(row.get("metrics", {}).get(metric, 0.0)) for row in rows]
            payload["methods"][method_id][f"{metric}_mean"] = float(np.mean(vals)) if vals else 0.0
            payload["methods"][method_id][f"{metric}_std"] = float(np.std(vals)) if vals else 0.0
    return payload


def _build_constrained_eval_summary_from_rows(
    main_episode_rows: list[dict],
    baseline_methods: list[str],
    args: argparse.Namespace,
) -> dict:
    constraints = {
        "fairness_floor": args.val_fairness_floor,
        "oldest_coverage_floor": args.val_oldest_coverage_floor,
        "risk_ceil": args.val_risk_ceil,
        "top10_risk_ceil": args.val_top10_risk_ceil,
    }
    payload = {
        "score_policy_version": C.SCORE_POLICY_VERSION,
        "ranking_policy_version": C.RANKING_POLICY_VERSION,
        "selection_policy_version": getattr(C, "SELECTION_POLICY_VERSION", C.RANKING_POLICY_VERSION),
        "operating_mode": getattr(args, "operating_mode", getattr(C, "OPERATING_MODE", "balanced")),
        "constraints": constraints,
        "methods": {},
    }
    for method_id in ["ours", *baseline_methods]:
        metrics_seq = []
        for row in main_episode_rows:
            if row.get("setting", "main") != "main" or row.get("method") != method_id:
                continue
            metrics_seq.append(row.get("metrics", {}))
        bundle = summary_metric_bundle(
            metrics_seq,
            constraints=constraints,
            target_metric="block_fee_norm",
            mode=getattr(args, "operating_mode", getattr(C, "OPERATING_MODE", "balanced")),
        )
        payload["methods"][method_id] = {
            "feasible_rate": bundle["feasible_rate"],
            "feasible_rate_tier": bundle["feasible_rate_tier"],
            "feasible_fee_mean": bundle["feasible_set_fee_mean"],
            "feasible_fee_std": bundle["feasible_set_fee_std"],
            # backward compatibility
            "constrained_fee_mean": bundle["feasible_set_fee_mean"],
            "constrained_fee_std": bundle["feasible_set_fee_std"],
            "all_episode_fee_mean": bundle["all_episode_fee_mean"],
            "all_episode_fee_std": bundle["all_episode_fee_std"],
            "risk_adjusted_fee_mean": bundle["risk_adjusted_fee_mean"],
            "risk_adjusted_fee_std": bundle["risk_adjusted_fee_std"],
            "infeasible_count": bundle["infeasible_count"],
            "violation_count": bundle["violation_count"],
            "n_episodes": bundle["n_episodes"],
            "two_stage_selection_score": bundle["two_stage_selection_score"],
            "selection_policy_version": bundle["selection_policy_version"],
            "operating_mode": bundle["operating_mode"],
            "violation_breakdown": bundle["violation_breakdown"],
            "constraint_violation_top1": bundle["constraint_violation_top1"],
            "effective_variance_checks": bundle["effective_variance_checks"],
            "low_variance_flags": bundle["low_variance_flags"],
        }
    payload["ranking"] = constrained_ranking(
        payload["methods"],
        min_feasible_rate=C.CONSTRAINED_RANK_MIN_FEASIBLE_RATE,
        mode=getattr(args, "operating_mode", getattr(C, "OPERATING_MODE", "balanced")),
    )
    return payload


def _build_pareto_outputs_from_rows(main_episode_rows: list[dict], baseline_methods: list[str]) -> tuple[dict, dict]:
    objectives = ["block_fee", "fairness", "risk_exposure", "oldest_coverage"]
    lower_is_better = {"risk_exposure"}
    methods = ["ours", *baseline_methods]
    pair_map: dict[tuple, dict[str, dict]] = {}
    for row in main_episode_rows:
        if row.get("setting", "main") != "main":
            continue
        method = row.get("method")
        if method not in methods:
            continue
        key = (row.get("seed"), row.get("shared_pool_id", row.get("episode_id")))
        pair_map.setdefault(key, {})[method] = row.get("metrics", {})

    matrix = {m: {} for m in methods}
    for a in methods:
        for b in methods:
            if a == b:
                matrix[a][b] = {
                    "ours_dominates_rate": 0.0,
                    "baseline_dominates_rate": 0.0,
                    "non_dominated_rate": 1.0,
                    "n_pairs": len(pair_map),
                }
                continue
            ours_dom = 0
            base_dom = 0
            n = 0
            for items in pair_map.values():
                if a not in items or b not in items:
                    continue
                n += 1
                if pareto_dominates(items[a], items[b], objectives, lower_is_better):
                    ours_dom += 1
                elif pareto_dominates(items[b], items[a], objectives, lower_is_better):
                    base_dom += 1
            non_dom = n - ours_dom - base_dom
            matrix[a][b] = {
                "ours_dominates_rate": (ours_dom / n) if n else 0.0,
                "baseline_dominates_rate": (base_dom / n) if n else 0.0,
                "non_dominated_rate": (non_dom / n) if n else 0.0,
                "n_pairs": n,
            }

    pareto_episode = {
        "objectives": objectives,
        "lower_is_better": list(lower_is_better),
        "episodes": [],
    }
    for (seed, pool_id), items in sorted(pair_map.items(), key=lambda kv: kv[0]):
        dom = {}
        for a in methods:
            if a not in items:
                continue
            dom[a] = {}
            for b in methods:
                if b not in items or a == b:
                    dom[a][b] = False
                    continue
                dom[a][b] = pareto_dominates(items[a], items[b], objectives, lower_is_better)
        pareto_episode["episodes"].append({
            "seed": seed,
            "shared_pool_id": pool_id,
            "methods": items,
            "dominance": dom,
        })
    return pareto_episode, matrix


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
    parser = argparse.ArgumentParser(description="一键实验编排")
    parser.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    parser.add_argument("--episodes", type=int, default=C.TOTAL_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=C.EVAL_EPISODES)
    parser.add_argument("--pool-size", type=C.validate_pool_size, default=C.POOL_SIZE_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="results_final")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练, 仅评估已有模型")
    parser.add_argument("--ablation", action="store_true", help="兼容参数: 在 stages 中追加 ablation")
    parser.add_argument("--protocol-ablation", action="store_true",
                        help="在 ablation 阶段额外运行协议消融并输出 table_protocol_ablation.tex")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=["main"],
        help="运行阶段。默认仅 main，可选: main robustness ablation",
    )
    parser.add_argument("--workers", type=int, default=None, help="并行进程数（默认 1，即串行）")
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
    parser.add_argument("--enable-strong-baseline", dest="enable_strong_baseline",
                        action="store_true", default=getattr(C, "ENABLE_STRONG_BASELINE_DEFAULT", True),
                        help="启用 V5 强启发式基线 Center-Insertion 和 Dynamic Tri-Objective Greedy（默认启用）")
    parser.add_argument("--disable-strong-baseline", dest="enable_strong_baseline",
                        action="store_false",
                        help="关闭 V5 强启发式基线，仅保留基础规则基线")
    parser.add_argument("--skip-baseline-tuning", action="store_true",
                        help="跳过含参数基线的验证池网格调参，使用 config.py 默认参数")
    parser.add_argument("--track", type=str, default="composite_optimal_track",
                        choices=["fairness_recovery_track", "composite_optimal_track"],
                        help="正式协议轨道：fairness_recovery_track 或 composite_optimal_track")
    parser.add_argument("--operating-mode", type=str,
                        choices=list(getattr(C, "OPERATING_MODES", ("aggressive", "balanced", "conservative"))),
                        default=getattr(C, "OPERATING_MODE", "balanced"),
                        help="运行策略档位：aggressive / balanced / conservative / risk_aware")
    parser.add_argument("--constraint-profile", type=str,
                        choices=["strict", "relaxed_for_training"],
                        default=getattr(C, "CONSTRAINT_PROFILE", "strict"),
                        help="评估约束 profile，默认 strict")
    parser.add_argument("--training-constraint-profile", type=str,
                        choices=["strict", "relaxed_for_training"],
                        default=getattr(C, "TRAINING_CONSTRAINT_PROFILE", "relaxed_for_training"),
                        help="训练阶段约束 profile，默认 relaxed_for_training")
    parser.add_argument("--val-episodes", type=int, default=C.VALIDATION_EPISODES)
    parser.add_argument("--val-interval", type=int, default=C.VALIDATION_INTERVAL)
    parser.add_argument("--val-metric", type=str, default=C.VALIDATION_METRIC)
    parser.add_argument("--val-fairness-floor", type=float, default=C.VALIDATION_FAIRNESS_FLOOR)
    parser.add_argument("--val-oldest-coverage-floor", type=float, default=C.VALIDATION_OLDEST_COVERAGE_FLOOR)
    parser.add_argument("--val-risk-ceil", type=float, default=C.VALIDATION_RISK_CEIL)
    parser.add_argument("--val-edge10-risk-ceil", type=float, default=C.VALIDATION_EDGE10_RISK_CEIL)
    parser.add_argument("--val-top10-risk-ceil", type=float, default=C.VALIDATION_TOP10_RISK_CEIL)
    parser.add_argument("--risk-tune-label", type=str, default="",
                        help="风险增强预验标签，如 A/B/C")
    parser.add_argument("--gamma-r", type=float, default=None,
                        help="覆盖训练环境的风险惩罚权重 GAMMA_R")
    parser.add_argument("--terminal-risk-exposure-weight", type=float, default=None,
                        help="覆盖终止风险暴露惩罚权重")
    parser.add_argument("--terminal-top10-risk-weight", type=float, default=None,
                        help="覆盖终止 top/edge 风险惩罚权重")
    parser.add_argument("--risk-adjusted-fee-lambda", type=float, default=None,
                        help="覆盖风险调整收益分中的风险惩罚系数")
    parser.add_argument("--pretrain-policy", type=str, default=C.PRETRAIN_POLICY,
                        choices=["none", "fifo", "fair_fee", "mixed"])
    parser.add_argument("--pretrain-epochs", type=int, default=C.PRETRAIN_EPOCHS)
    parser.add_argument("--pretrain-episodes-per-epoch", type=int, default=C.PRETRAIN_EPISODES_PER_EPOCH)
    parser.add_argument("--curriculum", action="store_true", default=C.CURRICULUM_ENABLED)
    parser.add_argument("--curriculum-stage-episodes", type=float, nargs=3, default=C.CURRICULUM_STAGE_EPISODES)
    parser.add_argument("--fairness-first", dest="fairness_first", action="store_true",
                        default=getattr(C, "FAIRNESS_FIRST_CURRICULUM_ENABLED", True))
    parser.add_argument("--no-fairness-first", dest="fairness_first", action="store_false")
    args = parser.parse_args()
    if hasattr(C, "resolve_constraints"):
        eval_cons = C.resolve_constraints(profile=args.constraint_profile, for_training=False)
        if args.val_fairness_floor == C.VALIDATION_FAIRNESS_FLOOR:
            args.val_fairness_floor = float(eval_cons.get("fairness_floor", args.val_fairness_floor))
        if args.val_oldest_coverage_floor == C.VALIDATION_OLDEST_COVERAGE_FLOOR:
            args.val_oldest_coverage_floor = float(
                eval_cons.get("oldest_coverage_floor", args.val_oldest_coverage_floor)
            )
        if args.val_risk_ceil == C.VALIDATION_RISK_CEIL:
            args.val_risk_ceil = float(eval_cons.get("risk_ceil", args.val_risk_ceil))
        if args.val_edge10_risk_ceil == C.VALIDATION_EDGE10_RISK_CEIL:
            args.val_edge10_risk_ceil = float(eval_cons.get("edge10_risk_ceil", args.val_edge10_risk_ceil))
        if args.val_top10_risk_ceil == C.VALIDATION_TOP10_RISK_CEIL:
            args.val_top10_risk_ceil = float(eval_cons.get("top10_risk_ceil", args.val_top10_risk_ceil))
    risk_tuning = _apply_risk_tuning_overrides(args)
    _resolve_stages(args)
    _apply_track_defaults(args)
    args.device_map = _parse_device_map(args.device_map)
    args.baseline_methods = get_baseline_method_ids(args.enable_strong_baseline)

    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)
    if args.workers is None:
        args.workers = 1
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
    print(f"Protocol ablation: {args.protocol_ablation}")
    print(f"Episodes: {args.episodes}")
    print(f"Pool size: {args.pool_size}")
    print(f"Baselines: {args.baseline_methods}")
    print(f"Baseline tuning: {not args.skip_baseline_tuning and getattr(C, 'BASELINE_TUNING_ENABLED', True)}")
    print(f"Track: {args.track}")
    print(f"Operating mode: {args.operating_mode}")
    print(f"Constraint profile (eval/train): {args.constraint_profile}/{args.training_constraint_profile}")
    print(
        "Validation protocol: "
        f"metric={args.val_metric}, fairness>={args.val_fairness_floor}, "
        f"oldest_cov>={args.val_oldest_coverage_floor}, risk<={args.val_risk_ceil}, "
        f"edge10_risk<={args.val_edge10_risk_ceil}, "
        f"top10_risk<={args.val_top10_risk_ceil}"
    )
    if risk_tuning:
        print(f"Risk tuning: {risk_tuning}")
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
        "protocol_version": getattr(C, "EXPERIMENT_PROTOCOL_VERSION", "unknown"),
        "protocol_name": getattr(C, "EXPERIMENT_PROTOCOL_NAME", "unknown"),
        "stages": args.stages,
        "track": args.track,
        "allow_random_fallback": args.allow_random_fallback,
        "resume": args.resume,
        "skip_existing": args.skip_existing,
        "protocol_ablation": args.protocol_ablation,
        "workers": args.workers,
        "device": str(device),
        "device_map": {str(k): v for k, v in args.device_map.items()},
        "baseline_methods": args.baseline_methods,
        "operating_mode": args.operating_mode,
        "selection_policy_version": getattr(C, "SELECTION_POLICY_VERSION", C.RANKING_POLICY_VERSION),
        "constraint_profile": args.constraint_profile,
        "training_constraint_profile": args.training_constraint_profile,
        "validation_protocol": {
            "metric": args.val_metric,
            "fairness_floor": args.val_fairness_floor,
            "oldest_coverage_floor": args.val_oldest_coverage_floor,
            "risk_ceil": args.val_risk_ceil,
            "edge10_risk_ceil": args.val_edge10_risk_ceil,
            "top10_risk_ceil": args.val_top10_risk_ceil,
        },
        "risk_tuning": risk_tuning,
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
    baseline_best_params_by_seed = {}
    seed_timings = {}
    ablation_reward_seconds = 0.0
    ablation_struct_seconds = 0.0
    ablation_protocol_seconds = 0.0

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
                best_params_payload = _read_json_if_exists(os.path.join(result_dir, "baseline_best_params.json"))
                if isinstance(best_params_payload, dict):
                    baseline_best_params_by_seed[str(seed)] = best_params_payload.get("best_params", best_params_payload)
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
                    if result.get("baseline_best_params") is not None:
                        baseline_best_params_by_seed[str(seed)] = result["baseline_best_params"]

                    print(
                        f"[seed={seed}] 结果已收集 "
                        f"({status_counts['success'] + status_counts['resumed'] + status_counts['failed']}/"
                        f"{len(seeds_to_run)})",
                        flush=True,
                    )

    if baseline_best_params_by_seed:
        baseline_params_payload = {
            "selection_unit": "per_training_seed_fixed_validation_pool",
            "selection_metric": "two_stage_selection_score",
            "test_pool_not_used_for_tuning": True,
            "by_seed": baseline_best_params_by_seed,
        }
        baseline_best_path = os.path.join(args.output, "baseline_best_params.json")
        baseline_table_path = os.path.join(args.output, "table_baseline_params.tex")
        _write_json(baseline_best_path, baseline_params_payload)
        with open(baseline_table_path, "w") as f:
            f.write(generate_baseline_params_table(baseline_params_payload))
        run_summary["outputs"]["baseline_best_params"] = os.path.basename(baseline_best_path)
        run_summary["outputs"]["table_baseline_params"] = os.path.basename(baseline_table_path)

    method_registry_path = os.path.join(args.output, "method_registry.json")
    _write_json(method_registry_path, method_registry_payload())
    run_summary["outputs"]["method_registry"] = os.path.basename(method_registry_path)

    run_summary["evidence_level"] = _infer_evidence_level(run_summary["seed_runs"])

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

        t2 = append_exploratory_note(generate_main_table(agg_main), run_summary["evidence_level"])
        t2_path = os.path.join(args.output, "table2_content.tex")
        with open(t2_path, "w") as f:
            f.write(t2)
        run_summary["outputs"]["table2"] = os.path.basename(t2_path)

        core_table = append_primary_decision_rule_note(generate_main_core_table(agg_main))
        core_table = append_exploratory_note(core_table, run_summary["evidence_level"])
        core_table_path = os.path.join(args.output, "table_main_core.tex")
        with open(core_table_path, "w") as f:
            f.write(core_table)
        run_summary["outputs"]["table_main_core"] = os.path.basename(core_table_path)

        full_table = append_exploratory_note(
            generate_main_fullmetrics_table(agg_main),
            run_summary["evidence_level"],
        )
        full_table_path = os.path.join(args.output, "table_main_fullmetrics.tex")
        with open(full_table_path, "w") as f:
            f.write(full_table)
        run_summary["outputs"]["table_main_fullmetrics"] = os.path.basename(full_table_path)

        table_composite = generate_composite_table(agg_main)
        if table_composite:
            table_composite_path = os.path.join(args.output, "table_composite_main.tex")
            with open(table_composite_path, "w") as f:
                f.write(append_exploratory_note(table_composite, run_summary["evidence_level"]))
            run_summary["outputs"]["table_composite_main"] = os.path.basename(table_composite_path)

        table_tradeoff = generate_tradeoff_table(agg_main)
        if table_tradeoff:
            table_tradeoff_path = os.path.join(args.output, "table_tradeoff_main.tex")
            with open(table_tradeoff_path, "w") as f:
                f.write(append_exploratory_note(table_tradeoff, run_summary["evidence_level"]))
            run_summary["outputs"]["table_tradeoff_main"] = os.path.basename(table_tradeoff_path)

        winner_payload = _metric_winner_by_dimension(agg_main)
        winner_path = os.path.join(args.output, "metric_winner_by_dimension.json")
        _write_json(winner_path, winner_payload)
        run_summary["outputs"]["metric_winner_by_dimension"] = os.path.basename(winner_path)

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
            f.write(append_exploratory_note(t3_risk, run_summary["evidence_level"]))
        with open(t3_pool_path, "w") as f:
            f.write(append_exploratory_note(t3_pool, run_summary["evidence_level"]))
        with open(t3_fee_path, "w") as f:
            f.write(append_exploratory_note(t3_fee, run_summary["evidence_level"]))
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

            seed_level_stats = _build_seed_level_statistics(
                main_episode_rows,
                args.baseline_methods,
                run_summary["evidence_level"],
            )
            seed_level_stats_path = os.path.join(args.output, "seed_level_statistics.json")
            _write_json(seed_level_stats_path, seed_level_stats)
            run_summary["outputs"]["seed_level_statistics"] = os.path.basename(seed_level_stats_path)

            if SCIPY_AVAILABLE:
                print("\n===== Seed 级正式配对显著性检验 =====")
                seed_sig = run_seed_level_paired_tests(seed_level_stats)
                print(format_seed_level_significance_table(seed_sig))
                seed_sig_path = os.path.join(args.output, "seed_level_paired_tests.json")
                seed_sig_tex = os.path.join(args.output, "table_seed_level_significance.tex")
                _write_json(seed_sig_path, seed_sig)
                generate_seed_level_significance_latex(seed_sig, seed_sig_tex)
                if seed_sig.get("exploratory_only"):
                    with open(seed_sig_tex, "a") as f:
                        f.write(
                            f"% NOTE: Exploratory evidence ({run_summary['evidence_level']}); "
                            "formal claims require enough independent training seeds.\n"
                        )
                run_summary["outputs"]["seed_level_paired_tests"] = os.path.basename(seed_sig_path)
                run_summary["outputs"]["table_seed_level_significance"] = os.path.basename(seed_sig_tex)
            else:
                print("\n===== Seed 级正式配对显著性检验 =====")
                print("Warning: scipy 未安装, 跳过 seed-level paired tests。")

            fairness_decomp = _build_fairness_decomposition_from_rows(
                main_episode_rows,
                args.baseline_methods,
            )
            fairness_decomp_path = os.path.join(args.output, "fairness_decomposition.json")
            _write_json(fairness_decomp_path, fairness_decomp)
            run_summary["outputs"]["fairness_decomposition"] = os.path.basename(fairness_decomp_path)
            fairness_tex = generate_fairness_decomp_table(fairness_decomp)
            fairness_tex_path = os.path.join(args.output, "table_fairness_decomp.tex")
            with open(fairness_tex_path, "w") as f:
                f.write(append_exploratory_note(fairness_tex, run_summary["evidence_level"]))
            run_summary["outputs"]["table_fairness_decomp"] = os.path.basename(fairness_tex_path)

            constrained_summary = _build_constrained_eval_summary_from_rows(
                main_episode_rows,
                args.baseline_methods,
                args,
            )
            constrained_path = os.path.join(args.output, "constrained_eval_summary.json")
            _write_json(constrained_path, constrained_summary)
            run_summary["outputs"]["constrained_eval_summary"] = os.path.basename(constrained_path)
            constrained_tex = append_primary_decision_rule_note(generate_constrained_main_table(constrained_summary))
            constrained_tex_path = os.path.join(args.output, "table_constrained_main.tex")
            with open(constrained_tex_path, "w") as f:
                f.write(append_exploratory_note(constrained_tex, run_summary["evidence_level"]))
            run_summary["outputs"]["table_constrained_main"] = os.path.basename(constrained_tex_path)
            table_constraints_path = os.path.join(args.output, "table_main_constraints.tex")
            with open(table_constraints_path, "w") as f:
                f.write(append_exploratory_note(constrained_tex, run_summary["evidence_level"]))
            run_summary["outputs"]["table_main_constraints"] = os.path.basename(table_constraints_path)

            operating_points = build_operating_points_summary(constrained_summary)
            operating_points_path = os.path.join(args.output, "operating_points_summary.json")
            _write_json(operating_points_path, operating_points)
            run_summary["outputs"]["operating_points_summary"] = os.path.basename(operating_points_path)
            operating_points_tex = generate_operating_points_table(operating_points)
            operating_points_tex_path = os.path.join(args.output, "table_operating_points.tex")
            with open(operating_points_tex_path, "w") as f:
                f.write(append_exploratory_note(operating_points_tex, run_summary["evidence_level"]))
            run_summary["outputs"]["table_operating_points"] = os.path.basename(operating_points_tex_path)

            constraint_bottleneck = build_constraint_bottleneck_report(constrained_summary)
            constraint_bottleneck_path = os.path.join(args.output, "constraint_bottleneck_report.json")
            _write_json(constraint_bottleneck_path, constraint_bottleneck)
            run_summary["outputs"]["constraint_bottleneck_report"] = os.path.basename(constraint_bottleneck_path)
            bottleneck_tex = generate_constraint_bottleneck_table(constraint_bottleneck)
            bottleneck_tex_path = os.path.join(args.output, "table_constraint_bottleneck.tex")
            with open(bottleneck_tex_path, "w") as f:
                f.write(append_exploratory_note(bottleneck_tex, run_summary["evidence_level"]))
            run_summary["outputs"]["table_constraint_bottleneck"] = os.path.basename(bottleneck_tex_path)

            pareto_episode, dominance_matrix = _build_pareto_outputs_from_rows(
                main_episode_rows,
                args.baseline_methods,
            )
            pareto_path = os.path.join(args.output, "pareto_episode_analysis.json")
            matrix_path = os.path.join(args.output, "dominance_matrix.json")
            _write_json(pareto_path, pareto_episode)
            _write_json(matrix_path, dominance_matrix)
            run_summary["outputs"]["pareto_episode_analysis"] = os.path.basename(pareto_path)
            run_summary["outputs"]["dominance_matrix"] = os.path.basename(matrix_path)
            pareto_tex = generate_pareto_main_table(dominance_matrix, anchor_method="ours")
            pareto_tex_path = os.path.join(args.output, "table_pareto_main.tex")
            with open(pareto_tex_path, "w") as f:
                f.write(append_exploratory_note(pareto_tex, run_summary["evidence_level"]))
            run_summary["outputs"]["table_pareto_main"] = os.path.basename(pareto_tex_path)

        merged_case_study = _merge_case_studies(case_study_seed_paths)
        if merged_case_study["seed_case_studies"]:
            case_study_path = os.path.join(args.output, "case_study.json")
            _write_json(case_study_path, merged_case_study)
            run_summary["outputs"]["case_study"] = os.path.basename(case_study_path)

        if SCIPY_AVAILABLE:
            if main_episode_rows:
                print("\n===== Episode 级配对显著性检验（诊断/附录） =====")
                sig = run_paired_significance_tests(main_episode_rows)
                print(format_paired_significance_table(sig))
                sig_path = os.path.join(args.output, "paired_significance_tests.json")
                _write_json(sig_path, sig)
                sig_tex = os.path.join(args.output, "table_significance_paired.tex")
                generate_paired_significance_latex(sig, sig_tex)
                if run_summary["evidence_level"] != "formal_multi_seed":
                    with open(sig_tex, "a") as f:
                        f.write(
                            f"% NOTE: Exploratory evidence ({run_summary['evidence_level']}); "
                            "do not claim formal superiority.\n"
                        )
                run_summary["outputs"]["paired_significance_json"] = os.path.basename(sig_path)
                run_summary["outputs"]["paired_significance_table"] = os.path.basename(sig_tex)
            else:
                print("\n===== Episode 级配对显著性检验（诊断/附录） =====")
                print("Warning: 未找到 main_episode_metrics.json，已跳过。")
        else:
            print("\n===== Episode 级配对显著性检验（诊断/附录） =====")
            print("Warning: scipy 未安装, 跳过显著性检验与对应 LaTeX 表格生成。")

    if "ablation" in args.stages:
        ablation_core_metrics = [
            "block_fee",
            "fairness",
            "risk_exposure",
            "top10_risk",
            "composite_score",
            "constrained_fee_score",
            "risk_adjusted_fee_score",
            "gas_util",
        ]
        tradeoff_note = "Ablation indicates trade-off shift, not monotonic degradation."

        print("\n===== 奖励消融实验 =====", flush=True)
        t_abl_reward = time.perf_counter()
        agg_reward_abl = run_ablation_parallel(
            REWARD_ABLATION_CONFIGS,
            args,
            "reward",
            metrics=ablation_core_metrics,
        )
        agg_reward_abl, reward_tradeoff = _annotate_ablation_tradeoff(
            agg_reward_abl,
            full_key="Ours-FullBalanced",
            core_metrics=ablation_core_metrics,
        )
        ablation_reward_seconds = time.perf_counter() - t_abl_reward
        t_rw = generate_ablation_table(agg_reward_abl, ABLATION_ORDER, ABLATION_LABELS)
        rw_path = os.path.join(args.output, "table_ablation_reward.tex")
        with open(rw_path, "w") as f:
            f.write(append_exploratory_note(t_rw, run_summary["evidence_level"]))
        if reward_tradeoff:
            with open(rw_path, "a") as f:
                f.write(f"% NOTE: {tradeoff_note}\n")
        abl_rw_json = os.path.join(args.output, "ablation_reward.json")
        _write_json(abl_rw_json, agg_reward_abl)

        print("\n===== 结构消融实验 =====", flush=True)
        t_abl_struct = time.perf_counter()
        agg_struct_abl = run_ablation_parallel(
            STRUCT_ABLATION_CONFIGS,
            args,
            "struct",
            metrics=ablation_core_metrics,
        )
        agg_struct_abl, struct_tradeoff = _annotate_ablation_tradeoff(
            agg_struct_abl,
            full_key="Ours-Full",
            core_metrics=ablation_core_metrics,
        )
        ablation_struct_seconds = time.perf_counter() - t_abl_struct
        t_st = generate_ablation_table(agg_struct_abl, STRUCT_ABLATION_ORDER, STRUCT_ABLATION_LABELS)
        st_path = os.path.join(args.output, "table_ablation_struct.tex")
        with open(st_path, "w") as f:
            f.write(append_exploratory_note(t_st, run_summary["evidence_level"]))
        if struct_tradeoff:
            with open(st_path, "a") as f:
                f.write(f"% NOTE: {tradeoff_note}\n")
        abl_st_json = os.path.join(args.output, "ablation_struct.json")
        _write_json(abl_st_json, agg_struct_abl)

        if reward_tradeoff or struct_tradeoff:
            run_summary.setdefault("ablation_notes", []).append(tradeoff_note)

        run_summary["outputs"]["ablation_reward_json"] = os.path.basename(abl_rw_json)
        run_summary["outputs"]["ablation_struct_json"] = os.path.basename(abl_st_json)
        run_summary["outputs"]["ablation_reward_table"] = os.path.basename(rw_path)
        run_summary["outputs"]["ablation_struct_table"] = os.path.basename(st_path)
        print(f"消融表格已保存: {rw_path}, {st_path}")

        if args.protocol_ablation:
            print("\n===== 协议消融实验 =====", flush=True)
            t_abl_protocol = time.perf_counter()
            agg_protocol_abl = run_ablation_parallel(
                PROTOCOL_ABLATION_CONFIGS,
                args,
                "protocol",
                metrics=[
                    "block_fee",
                    "fairness",
                    "risk_exposure",
                    "top10_risk",
                    "composite_score",
                    "constrained_fee_score",
                    "risk_adjusted_fee_score",
                ],
            )
            ablation_protocol_seconds = time.perf_counter() - t_abl_protocol
            protocol_tex = generate_protocol_ablation_table(
                agg_protocol_abl,
                PROTOCOL_ABLATION_ORDER,
                PROTOCOL_ABLATION_LABELS,
            )
            protocol_tex_path = os.path.join(args.output, "table_protocol_ablation.tex")
            with open(protocol_tex_path, "w") as f:
                f.write(append_exploratory_note(protocol_tex, run_summary["evidence_level"]))
            protocol_json_path = os.path.join(args.output, "ablation_protocol.json")
            _write_json(protocol_json_path, agg_protocol_abl)
            run_summary["outputs"]["ablation_protocol_json"] = os.path.basename(protocol_json_path)
            run_summary["outputs"]["ablation_protocol_table"] = os.path.basename(protocol_tex_path)
            print(f"协议消融表格已保存: {protocol_tex_path}")

    narrative_guard = _build_narrative_guard(agg_main if "main" in args.stages else None)
    narrative_guard_path = os.path.join(args.output, "narrative_guard.json")
    _write_json(narrative_guard_path, narrative_guard)
    run_summary["outputs"]["narrative_guard"] = os.path.basename(narrative_guard_path)

    v5_protocol_manifest = {
        "protocol_version": getattr(C, "EXPERIMENT_PROTOCOL_VERSION", "unknown"),
        "protocol_name": getattr(C, "EXPERIMENT_PROTOCOL_NAME", "unknown"),
        "evidence_level": run_summary.get("evidence_level", "unknown"),
        "formal_min_seeds": int(getattr(C, "EVIDENCE_FORMAL_MIN_SEEDS", 5)),
        "formal_statistics_ready": run_summary.get("evidence_level") == "formal_multi_seed",
        "data_separation": {
            "train_seed_offset": getattr(C, "TRAIN_POOL_SEED_OFFSET", 0),
            "validation_seed_offset": getattr(C, "VALIDATION_SEED_OFFSET", 10000),
            "test_seed_offset": getattr(C, "TEST_POOL_SEED_OFFSET", 20000),
            "test_pools_frozen_on_disk": os.path.isdir(os.path.join(args.output, "pools", "test")),
        },
        "baseline_protocol": {
            "strong_baseline_enabled": bool(args.enable_strong_baseline),
            "strong_baseline_names": ["Center-Insertion Heuristic", "Dynamic Tri-Objective Greedy"],
            "dynamic_tri_objective_is_stepwise": True,
            "baseline_tuning": "fixed_validation_pool_grid_search",
            "baseline_methods": args.baseline_methods,
        },
        "statistics_protocol": {
            "primary_unit": "independent_training_seed",
            "seed_level_paired_tests": "paired_t_with_cohens_dz_bootstrap_ci_holm",
            "episode_level_tests_are_diagnostic_only": True,
            "multiple_comparison_correction": "Holm-Bonferroni",
        },
        "tradeoff_score_protocol": {
            "policy_version": getattr(C, "TRADE_SCORE_POLICY_VERSION", "unknown"),
            "trade_score_weights": getattr(C, "TRADE_SCORE_WEIGHTS", {}),
            "risk_aware_trade_score_weights": getattr(C, "RISK_AWARE_TRADE_SCORE_WEIGHTS", {}),
            "risk_ref": getattr(C, "TRADE_SCORE_RISK_REF", None),
            "risk_tuning": risk_tuning,
            "edge_ref": getattr(C, "TRADE_SCORE_EDGE_REF", None),
            "constrained_thresholds": {
                "fairness_min": getattr(C, "CONSTRAINED_TRADE_FAIRNESS_MIN", None),
                "risk_max": getattr(C, "CONSTRAINED_TRADE_RISK_MAX", None),
                "edge_max": getattr(C, "CONSTRAINED_TRADE_EDGE_MAX", None),
                "gas_min": getattr(C, "CONSTRAINED_TRADE_GAS_MIN", None),
            },
        },
        "scope_boundary": (
            "risk_score is a proxy for position-sensitive ordering risk; "
            "results do not claim real on-chain MEV-defense effectiveness"
        ),
    }
    v5_protocol_manifest_path = os.path.join(args.output, "v5_protocol_manifest.json")
    _write_json(v5_protocol_manifest_path, v5_protocol_manifest)
    run_summary["outputs"]["v5_protocol_manifest"] = os.path.basename(v5_protocol_manifest_path)

    required_for_main_story = [
        "table_main_core",
        "table_main_fullmetrics",
        "table_main_constraints",
        "table_tradeoff_main",
        "table_operating_points",
        "table_constraint_bottleneck",
        "constrained_eval_summary",
        "operating_points_summary",
        "constraint_bottleneck_report",
        "seed_level_statistics",
        "seed_level_paired_tests",
        "table_seed_level_significance",
        "baseline_best_params",
        "table_baseline_params",
        "method_registry",
        "v5_protocol_manifest",
        "narrative_guard",
    ]
    missing = [k for k in required_for_main_story if k not in run_summary["outputs"]]
    outputs_manifest = {
        "version": "v5_experiment_package",
        "protocol_version": getattr(C, "EXPERIMENT_PROTOCOL_VERSION", "unknown"),
        "required_keys": required_for_main_story,
        "missing_keys": missing,
        "report_incomplete": len(missing) > 0,
        "evidence_level": run_summary.get("evidence_level", "unknown"),
        "score_policy_version": C.SCORE_POLICY_VERSION,
        "ranking_policy_version": C.RANKING_POLICY_VERSION,
        "selection_policy_version": getattr(C, "SELECTION_POLICY_VERSION", C.RANKING_POLICY_VERSION),
        "trade_score_policy_version": getattr(C, "TRADE_SCORE_POLICY_VERSION", "unknown"),
        "operating_mode": args.operating_mode,
    }
    outputs_manifest_path = os.path.join(args.output, "outputs_manifest.json")
    _write_json(outputs_manifest_path, outputs_manifest)
    run_summary["outputs"]["outputs_manifest"] = os.path.basename(outputs_manifest_path)

    timing_payload = {
        "timestamp": datetime.now().isoformat(),
        "seed_timings": seed_timings,
        "stage_totals_seconds": {
            "train": float(sum(t.get("train", 0.0) for t in seed_timings.values())),
            "eval": float(sum(t.get("eval", 0.0) for t in seed_timings.values())),
            "robustness": float(sum(t.get("robustness", 0.0) for t in seed_timings.values())),
            "ablation_reward": float(ablation_reward_seconds),
            "ablation_struct": float(ablation_struct_seconds),
            "ablation_protocol": float(ablation_protocol_seconds),
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
