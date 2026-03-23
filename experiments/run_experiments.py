"""一键实验编排: 多种子并行训练 → 评估 → 聚合 → LaTeX 表格"""

from __future__ import annotations

import argparse
from copy import deepcopy
import dataclasses
from datetime import datetime
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

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
from stat_tests import (SCIPY_AVAILABLE, run_significance_tests,
                        format_significance_table, generate_significance_latex)

BASELINE_METHODS = ["fifo", "gas", "heuristic", "fee_risk_linear", "fair_fee"]
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


def _baseline_label(method: str) -> str:
    return method.upper().replace("FEE_RISK_LINEAR", "FeeRiskLinear").replace(
        "FAIR_FEE", "FairFee"
    )


def _serialize_shared_pools(shared_pools: list[list]) -> list[list[dict]]:
    return [
        [dataclasses.asdict(tx) for tx in pool]
        for pool in shared_pools
    ]


def _episode_rows(raw: dict[str, list[dict]], setting: str, setting_value) -> list[dict]:
    rows = []
    for method, metrics_seq in raw.items():
        for episode_id, metrics in enumerate(metrics_seq):
            rows.append({
                "setting": setting,
                "setting_value": setting_value,
                "method": method,
                "episode_id": episode_id,
                "shared_pool_id": episode_id,
                "metrics": metrics,
            })
    return rows


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
    })

    raw = {"RL (Ours)": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
    for bl in BASELINE_METHODS:
        raw[_baseline_label(bl)] = evaluate_baseline(bl, env, args.eval_episodes, shared_pools)
    agg = {method: aggregate(metrics) for method, metrics in raw.items()}

    _write_json(os.path.join(result_dir, "main_results.json"), agg)
    _write_json(os.path.join(result_dir, "main_aggregated_metrics.json"), agg)
    _write_json(os.path.join(result_dir, "main_episode_metrics.json"), {
        "seed": seed,
        "setting": "main",
        "records": _episode_rows(raw, "main", {"pool_size": args.pool_size, "risk_ratio": C.RISK_RATIO}),
    })
    return agg


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
        })
        pool_index["files"][str(rr)] = os.path.basename(rr_pool_file)

        raw = {"RL": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
        for bl in BASELINE_METHODS:
            raw[bl] = evaluate_baseline(bl, env, args.eval_episodes, shared_pools)
        all_data[str(rr)] = {method: aggregate(metrics) for method, metrics in raw.items()}
        all_rows.extend(_episode_rows(raw, "robustness_risk", {"risk_ratio": rr}))

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
        })
        pool_index["files"][str(ps)] = os.path.basename(pool_file)

        raw = {"RL": evaluate_rl(model, env, args.eval_episodes, device, shared_pools)}
        for bl in BASELINE_METHODS:
            raw[bl] = evaluate_baseline(bl, env, args.eval_episodes, shared_pools)
        all_data[str(ps)] = {method: aggregate(metrics) for method, metrics in raw.items()}
        all_rows.extend(_episode_rows(raw, "robustness_pool_size", {"pool_size": ps}))

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
        raw_bl = {bl: [] for bl in BASELINE_METHODS}

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

            for bl in BASELINE_METHODS:
                selected_bl = run_baseline(deepcopy(pool), bl)
                raw_bl[bl].append(compute_all_metrics(selected_bl, pool))

        raw = {"RL": raw_rl}
        raw.update(raw_bl)
        all_data[str(mult)] = {method: aggregate(metrics) for method, metrics in raw.items()}
        all_rows.extend(_episode_rows(raw, "robustness_fee_multiplier", {"fee_multiplier": mult}))

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
    stages = set(args.stages)
    device = resolve_device(args.device)

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
        "status": "ok",
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
        )
        train_model(train_args)
        print(f"[seed={seed}] 训练完成", flush=True)

    main_results = None
    rob_risk = None
    rob_pool = None
    rob_fee = None
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
            main_results = _run_main_evaluation(model, device, args, seed, result_dir)

        if "robustness" in stages:
            print(f"[seed={seed}] 鲁棒性实验 (风险比例)", flush=True)
            rob_risk = _run_risk_robustness(model, device, args, seed, result_dir)
            print(f"[seed={seed}] 鲁棒性实验 (池大小)", flush=True)
            rob_pool = _run_pool_robustness(model, device, args, seed, result_dir)
            print(f"[seed={seed}] 鲁棒性实验 (费率倍率)", flush=True)
            rob_fee = _run_fee_robustness(model, device, args, seed, result_dir)

    if os.path.exists(log_path):
        plot_training_curve(log_path, os.path.join(result_dir, "training_curve.png"))

    _write_json(os.path.join(result_dir, "seed_summary.json"), seed_summary)
    print(f"[seed={seed}] 完成", flush=True)
    return {
        "seed": seed,
        "main": main_results,
        "rob_risk": rob_risk,
        "rob_pool": rob_pool,
        "rob_fee": rob_fee,
        "seed_summary": seed_summary,
    }


def aggregate_metric_grid(all_results):
    """聚合形如 {setting: {method: metric_stats}} 的结果网格。"""
    settings = list(all_results[0].keys())
    methods = list(all_results[0][settings[0]].keys())
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util",
               "risky_rank", "packing_ratio", "top10_risk", "late_promo"]

    agg = {}
    for setting in settings:
        agg[setting] = {}
        for method in methods:
            agg[setting][method] = {}
            for metric in metrics:
                vals = [r[setting][method][f"{metric}_mean"] for r in all_results]
                agg[setting][method][f"{metric}_mean"] = float(np.mean(vals))
                agg[setting][method][f"{metric}_std"] = float(np.std(vals))
    return agg


def aggregate_across_seeds(all_main=None, all_rob_risk=None, all_rob_pool=None, all_rob_fee=None):
    """跨种子聚合: 对每个方法的 mean 取均值和标准差"""
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util",
               "risky_rank", "packing_ratio", "top10_risk", "late_promo"]

    agg_main = {}
    if all_main:
        methods = list(all_main[0].keys())
        for method in methods:
            agg_main[method] = {}
            for metric in metrics:
                vals = [r[method][f"{metric}_mean"] for r in all_main]
                agg_main[method][f"{metric}_mean"] = float(np.mean(vals))
                agg_main[method][f"{metric}_std"] = float(np.std(vals))

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
    from ppo import PPOTrainer, RolloutBuffer
    args = argparse.Namespace(**args_dict)
    device = resolve_device(args.device)

    abl_dir = os.path.join(args.output, "ablation", name, f"seed_{seed}")
    ckpt_dir = os.path.join(abl_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, C.BEST_CHECKPOINT_NAME)
    final_model_path = os.path.join(ckpt_dir, C.FINAL_CHECKPOINT_NAME)
    model_path = os.path.join(ckpt_dir, C.FORMAL_EVAL_CHECKPOINT_NAME)

    if not args.skip_train:
        print(f"  [Ablation] {name} seed={seed} 开始训练", flush=True)
        seed_everything(seed)
        env = TxOrderingEnv(pool_size=args.pool_size,
                            risk_ratio=C.RISK_RATIO,
                            seed=seed, **env_kwargs)
        model = ActorCritic().to(device)
        trainer = PPOTrainer(model, device=device)
        buffer = RolloutBuffer()
        best_reward = -float("inf")

        for _ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            ep_reward = 0.0
            while True:
                action, log_prob, value = model.act(obs, device)
                next_obs, reward, done, _, _info = env.step(action)
                buffer.store(obs, action, log_prob, reward, value, done)
                ep_reward += reward
                obs = next_obs
                if done:
                    break
            trainer.update(buffer)
            buffer.clear()
            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), final_model_path)
        _write_json(os.path.join(ckpt_dir, "checkpoint_meta.json"), {
            "formal_eval_checkpoint": C.FORMAL_EVAL_CHECKPOINT_NAME,
            "formal_eval_rule": C.FORMAL_EVAL_CHECKPOINT_RULE,
            "best_checkpoint": C.BEST_CHECKPOINT_NAME,
            "final_checkpoint": C.FINAL_CHECKPOINT_NAME,
        })
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

    max_workers = min(args.workers, len(configs) * len(args.seeds))
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
        for fut in as_completed(futures):
            _name_full, _seed, res = fut.result()
            orig_name = futures[fut][0]
            results[orig_name].append(res)

    agg = {}
    for name in configs:
        agg[name] = {}
        for metric in metrics:
            vals = [r[f"{metric}_mean"] for r in results[name]]
            agg[name][f"{metric}_mean"] = float(np.mean(vals))
            agg[name][f"{metric}_std"] = float(np.std(vals))
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
    parser.add_argument("--workers", type=int, default=5, help="并行进程数 (默认 5)")
    parser.add_argument("--allow-random-fallback", action="store_true",
                        help="当 checkpoint 缺失时允许随机策略回退")
    args = parser.parse_args()
    _resolve_stages(args)

    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Stages: {args.stages}")
    print(f"Seeds: {args.seeds}")
    print(f"Workers: {args.workers}")
    print(f"Episodes: {args.episodes}")
    print(f"Pool size: {args.pool_size}")

    config_snapshot_path = os.path.join(args.output, "config_snapshot.json")
    _write_json(config_snapshot_path, _collect_config_snapshot(args))

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "stages": args.stages,
        "allow_random_fallback": args.allow_random_fallback,
        "config_snapshot": os.path.basename(config_snapshot_path),
        "seeds": args.seeds,
        "seed_runs": [],
        "outputs": {},
    }

    all_main = []
    all_rob_risk = []
    all_rob_pool = []
    all_rob_fee = []

    if set(args.stages) & {"main", "robustness"}:
        args_dict = vars(args)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for seed in args.seeds:
                fut = executor.submit(run_single_seed, seed, args_dict)
                futures[fut] = seed

            for fut in as_completed(futures):
                seed = futures[fut]
                result = fut.result()
                run_summary["seed_runs"].append(result["seed_summary"])
                if result["main"] is not None:
                    all_main.append(result["main"])
                if result["rob_risk"] is not None:
                    all_rob_risk.append(result["rob_risk"])
                if result["rob_pool"] is not None:
                    all_rob_pool.append(result["rob_pool"])
                if result["rob_fee"] is not None:
                    all_rob_fee.append(result["rob_fee"])
                print(f"[seed={seed}] 结果已收集 ({len(run_summary['seed_runs'])}/{len(args.seeds)})",
                      flush=True)

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

    if "main" in args.stages:
        if SCIPY_AVAILABLE and len(all_main) >= 2:
            print("\n===== 统计显著性检验 =====")
            sig = run_significance_tests(all_main)
            print(format_significance_table(sig))
            sig_path = os.path.join(args.output, "significance_tests.json")
            _write_json(sig_path, sig)
            sig_tex = os.path.join(args.output, "table_significance.tex")
            generate_significance_latex(sig, sig_tex)
            run_summary["outputs"]["significance_json"] = os.path.basename(sig_path)
            run_summary["outputs"]["significance_table"] = os.path.basename(sig_tex)
        else:
            print("\n===== 统计显著性检验 =====")
            if not SCIPY_AVAILABLE:
                print("Warning: scipy 未安装, 跳过显著性检验与对应 LaTeX 表格生成。")
            else:
                print("Warning: 显著性检验至少需要 2 个 seed, 当前已跳过。")

    if "ablation" in args.stages:
        print("\n===== 奖励消融实验 =====", flush=True)
        agg_reward_abl = run_ablation_parallel(REWARD_ABLATION_CONFIGS, args, "reward")
        t_rw = generate_ablation_table(agg_reward_abl, ABLATION_ORDER, ABLATION_LABELS)
        rw_path = os.path.join(args.output, "table_ablation_reward.tex")
        with open(rw_path, "w") as f:
            f.write(t_rw)
        abl_rw_json = os.path.join(args.output, "ablation_reward.json")
        _write_json(abl_rw_json, agg_reward_abl)

        print("\n===== 结构消融实验 =====", flush=True)
        agg_struct_abl = run_ablation_parallel(STRUCT_ABLATION_CONFIGS, args, "struct")
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

    run_summary_path = os.path.join(args.output, "run_summary.json")
    _write_json(run_summary_path, run_summary)
    print(f"运行摘要已保存: {run_summary_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
