"""训练入口"""

from __future__ import annotations

import argparse
import os
import json
import random
import time
import torch
import numpy as np

import config as C
from env import TxOrderingEnv
from networks import ActorCritic
from ppo import PPOTrainer, RolloutBuffer
from device_utils import resolve_device, seed_everything
from evaluate import aggregate, build_shared_pools, evaluate_rl, save_shared_pools
from metrics import compute_all_metrics, two_stage_selection_score


PROGRESS_FILENAME = "training_progress.json"


def _write_training_progress(output_dir: str, payload: dict) -> None:
    """原子写入训练进度，便于外部实时查询当前 episode。"""
    os.makedirs(output_dir, exist_ok=True)
    progress = dict(payload)
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    path = os.path.join(output_dir, PROGRESS_FILENAME)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _progress_payload(
    args: argparse.Namespace,
    status: str,
    current_episode: int,
    started_at: float,
    current_stage: int | None = None,
    last_reward: float | None = None,
    last_fee: float | None = None,
    last_steps: int | None = None,
    best_episode: int | None = None,
    best_score: float | None = None,
    best_frozen: bool | None = None,
    last_validation_episode: int | None = None,
    note: str | None = None,
) -> dict:
    total = int(args.episodes)
    current = int(current_episode)
    payload = {
        "status": status,
        "seed": int(args.seed),
        "current_episode": current,
        "total_episodes": total,
        "progress_pct": round((current / total) * 100.0, 4) if total else 0.0,
        "remaining_episodes": max(total - current, 0),
        "current_stage": current_stage,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "device": str(args.device),
        "pool_size": int(args.pool_size),
        "risk_ratio": float(args.risk_ratio),
        "val_interval": int(args.val_interval),
        "val_episodes": int(args.val_episodes),
        "val_metric": str(args.val_metric),
        "best_episode": best_episode,
        "best_score": best_score,
        "last_validation_episode": last_validation_episode,
        "last_reward": last_reward,
        "last_fee": last_fee,
        "last_steps": last_steps,
    }
    if best_frozen is not None:
        payload["best_frozen"] = bool(best_frozen)
    if note:
        payload["note"] = note
    return payload


def _higher_is_better(metric: str) -> bool:
    return metric not in C.LOWER_IS_BETTER_METRICS


def _run_validation(model: ActorCritic,
                    device: torch.device,
                    args: argparse.Namespace,
                    validation_pools: list[list],
                    env_kwargs: dict) -> dict:
    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=args.risk_ratio,
                        seed=args.seed,
                        **env_kwargs)
    metrics = evaluate_rl(model, env, len(validation_pools), device, validation_pools)
    agg = aggregate(metrics)
    strict_constraints = C.resolve_constraints(profile="strict", for_training=False) if hasattr(C, "resolve_constraints") else {}
    strict_constraints.update({
        "fairness_floor": getattr(args, "val_fairness_floor", C.VALIDATION_FAIRNESS_FLOOR),
        "oldest_coverage_floor": getattr(args, "val_oldest_coverage_floor", C.VALIDATION_OLDEST_COVERAGE_FLOOR),
        "risk_ceil": getattr(args, "val_risk_ceil", C.VALIDATION_RISK_CEIL),
        "edge10_risk_ceil": getattr(args, "val_edge10_risk_ceil", C.VALIDATION_EDGE10_RISK_CEIL),
    })
    if args.val_metric == "constrained_fee":
        fee = float(agg.get("block_fee_mean", 0.0))
        fairness = float(agg.get("fairness_mean", 0.0))
        oldest_cov = float(agg.get("oldest_coverage_mean", 0.0))
        risk = float(agg.get("risk_exposure_mean", 1.0))
        edge10_risk = float(agg.get("edge10_risk_mean", 1.0))
        feasible = (
            fairness >= getattr(args, "val_fairness_floor", C.VALIDATION_FAIRNESS_FLOOR)
            and oldest_cov >= getattr(args, "val_oldest_coverage_floor", C.VALIDATION_OLDEST_COVERAGE_FLOOR)
            and risk <= getattr(args, "val_risk_ceil", C.VALIDATION_RISK_CEIL)
            and edge10_risk <= getattr(args, "val_edge10_risk_ceil", C.VALIDATION_EDGE10_RISK_CEIL)
        )
        metric_key = "block_fee_mean"
        metric_value = fee
        score = fee if feasible else -float("inf")
    elif args.val_metric == "hypervolume":
        fee_norm = float(agg.get("block_fee_norm_mean", 0.0))
        fairness = float(np.clip(agg.get("fairness_mean", 0.0), 0.0, 1.0))
        risk = float(np.clip(agg.get("risk_exposure_mean", 1.0), 0.0, 1.0))
        edge10_risk = float(np.clip(agg.get("edge10_risk_mean", 1.0), 0.0, 1.0))
        oldest_cov = float(np.clip(agg.get("oldest_coverage_mean", 0.0), 0.0, 1.0))
        metric_key = "hypervolume_proxy"
        metric_value = fee_norm * fairness * (1.0 - risk) * (1.0 - edge10_risk) * oldest_cov
        score = metric_value
    elif args.val_metric == "pareto_score":
        # 轻量 Pareto proxy：鼓励收益/公平/oldest 覆盖，不允许高风险与高edge10_risk。
        fee_norm = float(np.clip(agg.get("block_fee_norm_mean", 0.0), 0.0, 1.0))
        fairness = float(np.clip(agg.get("fairness_mean", 0.0), 0.0, 1.0))
        oldest = float(np.clip(agg.get("oldest_coverage_mean", 0.0), 0.0, 1.0))
        risk = float(np.clip(agg.get("risk_exposure_mean", 1.0), 0.0, 1.0))
        edge10 = float(np.clip(agg.get("edge10_risk_mean", 1.0), 0.0, 1.0))
        metric_key = "pareto_proxy"
        metric_value = 0.3 * fee_norm + 0.25 * fairness + 0.25 * oldest + 0.1 * (1.0 - risk) + 0.1 * (1.0 - edge10)
        score = metric_value
    elif args.val_metric == "two_stage":
        two_stage = two_stage_selection_score(
            metrics,
            constraints=strict_constraints,
            mode=getattr(args, "operating_mode", getattr(C, "OPERATING_MODE", "balanced")),
            target_metric="block_fee_norm",
            min_feasible_rate=C.CONSTRAINED_RANK_MIN_FEASIBLE_RATE,
        )
        metric_key = "two_stage_selection_score"
        metric_value = float(two_stage["two_stage_selection_score"])
        score = metric_value
    else:
        metric_key = f"{args.val_metric}_mean"
        if metric_key not in agg:
            raise KeyError(f"validation metric '{args.val_metric}' not found in evaluation metrics.")
        metric_value = float(agg[metric_key])
        score = metric_value if _higher_is_better(args.val_metric) else -metric_value
    return {
        "metric_key": metric_key,
        "metric_value": metric_value,
        "score": float(score),
        "aggregate": agg,
    }


def _teacher_action_from_env(env: TxOrderingEnv, policy: str) -> int:
    valid = sorted(env._valid_indices())  # noqa: SLF001
    if not valid:
        return len(env._candidates)  # noqa: SLF001

    candidates = env._candidates  # noqa: SLF001
    if policy == "fifo":
        return min(valid, key=lambda i: candidates[i].arrival_time)
    if policy == "fair_fee":
        t_now = env._t_now  # noqa: SLF001
        max_fee = max(env._max_fee, 1e-8)  # noqa: SLF001
        t_max = max(env._t_max, 1e-8)  # noqa: SLF001
        return max(
            valid,
            key=lambda i: (candidates[i].fee / max_fee) + ((t_now - candidates[i].arrival_time) / t_max),
        )
    if policy == "gas":
        return max(valid, key=lambda i: candidates[i].fee)

    # mixed teacher
    sampled = random.choice(["fifo", "fair_fee", "gas"])
    return _teacher_action_from_env(env, sampled)


def _behavior_clone_warmstart(
    model: ActorCritic,
    device: torch.device,
    env: TxOrderingEnv,
    policy: str,
    epochs: int,
    episodes_per_epoch: int,
) -> list[dict]:
    if policy == "none" or epochs <= 0:
        return []

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": C.LR_ACTOR},
            {"params": model.actor_fc.parameters(), "lr": C.LR_ACTOR},
            {"params": model.actor_score.parameters(), "lr": C.LR_ACTOR},
            {"params": [model.stop_embed], "lr": C.LR_ACTOR},
        ]
    )
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        n_steps = 0
        for _ in range(max(episodes_per_epoch, 1)):
            obs, _ = env.reset()
            while True:
                action = _teacher_action_from_env(env, policy)
                tx_f = torch.as_tensor(obs["tx_features"], dtype=torch.float32, device=device)
                bs = torch.as_tensor(obs["block_state"], dtype=torch.float32, device=device)
                mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
                n = int(obs["num_candidates"])
                log_probs, _ = model.forward(tx_f, bs, mask, n)
                losses.append(-log_probs[action])
                n_steps += 1
                obs, _, done, _, _ = env.step(action)
                if done:
                    break

        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            history.append({
                "epoch": epoch,
                "policy": policy,
                "bc_loss": float(loss.item()),
                "steps": n_steps,
            })
    return history


def _resolve_curriculum_stage(ep: int, total_episodes: int, cuts: tuple[float, float, float]) -> int:
    c1 = max(int(total_episodes * cuts[0]), 1)
    c2 = max(int(total_episodes * cuts[1]), c1 + 1)
    if ep <= c1:
        return 1
    if ep <= c2:
        return 2
    return 3


def _curriculum_env_kwargs(base_env_kwargs: dict, args: argparse.Namespace, stage: int) -> dict:
    if not args.curriculum:
        return dict(base_env_kwargs)
    if not getattr(args, "fairness_first", getattr(C, "FAIRNESS_FIRST_CURRICULUM_ENABLED", True)):
        return dict(base_env_kwargs)

    kwargs = dict(base_env_kwargs)
    if stage == 1:
        kwargs["beta_age"] = kwargs.get("beta_age", C.BETA_AGE) * 1.35
        kwargs["beta_oldest_cover"] = kwargs.get("beta_oldest_cover", C.BETA_OLDEST_COVER) * 1.35
        kwargs["gamma_r"] = kwargs.get("gamma_r", C.GAMMA_R) * 0.7
        kwargs["terminal_risk_exposure_weight"] = kwargs.get(
            "terminal_risk_exposure_weight",
            C.TERMINAL_RISK_EXPOSURE_WEIGHT,
        ) * 0.6
        kwargs["terminal_top10_risk_weight"] = kwargs.get(
            "terminal_top10_risk_weight",
            C.TERMINAL_TOP10_RISK_WEIGHT,
        ) * 0.6
        kwargs["fairness_gate_min"] = max(kwargs.get("fairness_gate_min", C.FAIRNESS_GATE_MIN), 0.55)
    elif stage == 2:
        kwargs["gamma_r"] = kwargs.get("gamma_r", C.GAMMA_R) * 1.15
        kwargs["terminal_risk_exposure_weight"] = kwargs.get(
            "terminal_risk_exposure_weight",
            C.TERMINAL_RISK_EXPOSURE_WEIGHT,
        ) * 1.2
        kwargs["terminal_top10_risk_weight"] = kwargs.get(
            "terminal_top10_risk_weight",
            C.TERMINAL_TOP10_RISK_WEIGHT,
        ) * 1.2
        kwargs["fairness_gate_min"] = max(kwargs.get("fairness_gate_min", C.FAIRNESS_GATE_MIN), 0.45)
    elif stage == 3:
        kwargs["gamma_r"] = kwargs.get("gamma_r", C.GAMMA_R) * getattr(
            C, "CURRICULUM_STAGE3_GAMMA_R_MULT", 1.0
        )
        kwargs["terminal_risk_exposure_weight"] = kwargs.get(
            "terminal_risk_exposure_weight",
            C.TERMINAL_RISK_EXPOSURE_WEIGHT,
        ) * getattr(C, "CURRICULUM_STAGE3_TERMINAL_RISK_MULT", 1.0)
        kwargs["terminal_top10_risk_weight"] = kwargs.get(
            "terminal_top10_risk_weight",
            C.TERMINAL_TOP10_RISK_WEIGHT,
        ) * getattr(C, "CURRICULUM_STAGE3_TERMINAL_TOP10_MULT", 1.0)
        kwargs["fairness_gate_min"] = max(
            kwargs.get("fairness_gate_min", C.FAIRNESS_GATE_MIN),
            getattr(C, "CURRICULUM_STAGE3_FAIRNESS_GATE_MIN", C.FAIRNESS_GATE_MIN),
        )
    return kwargs


def _lr_scale_for_stage(args: argparse.Namespace, stage: int) -> float:
    if not getattr(args, "lr_schedule", getattr(C, "LR_SCHEDULE_ENABLED", False)):
        return 1.0
    scales = tuple(getattr(args, "lr_stage_scales", getattr(C, "LR_STAGE_SCALES", (1.0, 0.5, 0.2))))
    idx = max(min(int(stage), len(scales)), 1) - 1
    return float(scales[idx])


def _stability_adjusted_score(raw_scores: list[float], window_size: int, variance_penalty: float) -> dict:
    window_size = max(int(window_size), 1)
    window = [float(score) for score in raw_scores[-window_size:]]
    if not window:
        return {
            "score": -float("inf"),
            "mean": -float("inf"),
            "std": float("inf"),
            "count": 0,
            "window_size": window_size,
        }
    if not all(np.isfinite(score) for score in window):
        return {
            "score": -float("inf"),
            "mean": -float("inf"),
            "std": float("inf"),
            "count": len(window),
            "window_size": window_size,
        }
    mean_score = float(np.mean(window))
    std_score = float(np.std(window)) if len(window) > 1 else 0.0
    return {
        "score": mean_score - float(variance_penalty) * std_score,
        "mean": mean_score,
        "std": std_score,
        "count": len(window),
        "window_size": window_size,
    }


def _multi_checkpoint_scores(agg: dict, args: argparse.Namespace) -> dict:
    fairness = float(agg.get("fairness_mean", 0.0))
    oldest = float(agg.get("oldest_coverage_mean", 0.0))
    starvation = float(agg.get("starvation_gap_mean", 1.0))
    risk = float(agg.get("risk_exposure_mean", 1.0))
    edge10 = float(agg.get("edge10_risk_mean", 1.0))
    fee_norm = float(agg.get("block_fee_norm_mean", 0.0))
    constrained_fee = (
        fee_norm
        if (
            fairness >= getattr(args, "val_fairness_floor", C.VALIDATION_FAIRNESS_FLOOR)
            and oldest >= getattr(args, "val_oldest_coverage_floor", C.VALIDATION_OLDEST_COVERAGE_FLOOR)
            and risk <= getattr(args, "val_risk_ceil", C.VALIDATION_RISK_CEIL)
            and edge10 <= getattr(args, "val_edge10_risk_ceil", C.VALIDATION_EDGE10_RISK_CEIL)
        )
        else -float("inf")
    )
    return {
        "fairness_recovery": fairness + 0.4 * oldest - 0.2 * starvation,
        # 显式纳入 edge10_risk，避免仅靠 risk_exposure 选模。
        "risk_aligned": fairness - 0.8 * risk - 0.5 * edge10,
        "constrained_fee": constrained_fee,
        "hypervolume": (
            fee_norm
            * max(fairness, 0.0)
            * max(1.0 - risk, 0.0)
            * max(1.0 - edge10, 0.0)
            * max(oldest, 1e-6)
        ),
        "trade_score": float(agg.get("trade_score_mean", 0.0)),
        "risk_aware_trade": float(agg.get("risk_aware_trade_score_mean", 0.0)),
        "constrained_trade": float(agg.get("constrained_trade_score_mean", 0.0)),
    }


def train(args, env_kwargs: dict | None = None):
    base_env_kwargs = dict(env_kwargs or {})
    device = resolve_device(args.device)
    progress_started_at = time.time()
    _write_training_progress(
        args.output,
        _progress_payload(args, "initializing", 0, progress_started_at, note="building model and data pools"),
    )
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print(f"Device: {device}", flush=True)

    curriculum_enabled = bool(getattr(args, "curriculum", C.CURRICULUM_ENABLED))
    curriculum_cuts = tuple(getattr(args, "curriculum_stage_episodes", C.CURRICULUM_STAGE_EPISODES))
    fairness_first = bool(getattr(args, "fairness_first", getattr(C, "FAIRNESS_FIRST_CURRICULUM_ENABLED", True)))
    pretrain_policy = getattr(args, "pretrain_policy", C.PRETRAIN_POLICY)
    pretrain_epochs = int(getattr(args, "pretrain_epochs", C.PRETRAIN_EPOCHS))
    pretrain_episodes_per_epoch = int(
        getattr(args, "pretrain_episodes_per_epoch", C.PRETRAIN_EPISODES_PER_EPOCH)
    )

    if curriculum_enabled:
        current_stage = _resolve_curriculum_stage(
            ep=1,
            total_episodes=args.episodes,
            cuts=curriculum_cuts,
        )
    else:
        current_stage = 3
    args.curriculum = curriculum_enabled
    args.curriculum_stage_episodes = curriculum_cuts
    args.fairness_first = fairness_first
    args.lr_schedule = bool(getattr(args, "lr_schedule", getattr(C, "LR_SCHEDULE_ENABLED", False)))
    args.lr_stage_scales = tuple(getattr(args, "lr_stage_scales", getattr(C, "LR_STAGE_SCALES", (1.0, 0.5, 0.2))))
    args.val_smoothing_window = max(1, int(getattr(args, "val_smoothing_window", getattr(C, "VALIDATION_SMOOTHING_WINDOW", 1))))
    args.val_variance_penalty = max(0.0, float(getattr(args, "val_variance_penalty", getattr(C, "VALIDATION_VARIANCE_PENALTY", 0.0))))
    args.best_freeze = bool(getattr(args, "best_freeze", getattr(C, "BEST_FREEZE_ENABLED", False)))
    args.best_freeze_patience = max(1, int(getattr(args, "best_freeze_patience", getattr(C, "BEST_FREEZE_PATIENCE", 8))))
    args.best_freeze_min_episodes = max(0, int(getattr(args, "best_freeze_min_episodes", getattr(C, "BEST_FREEZE_MIN_EPISODES", 300))))
    active_env_kwargs = _curriculum_env_kwargs(base_env_kwargs, args, current_stage)
    train_pool_seed = args.seed + getattr(C, "TRAIN_POOL_SEED_OFFSET", 0)
    env = TxOrderingEnv(
        pool_size=args.pool_size,
        risk_ratio=args.risk_ratio,
        seed=train_pool_seed,
        **active_env_kwargs,
    )

    model = ActorCritic().to(device)
    trainer = PPOTrainer(model, device=device)
    current_lr_scale = _lr_scale_for_stage(args, current_stage)
    trainer.set_lr_scale(current_lr_scale)
    buffer = RolloutBuffer()

    warmstart_history = _behavior_clone_warmstart(
        model=model,
        device=device,
        env=env,
        policy=pretrain_policy,
        epochs=pretrain_epochs,
        episodes_per_epoch=pretrain_episodes_per_epoch,
    )

    log = {
        "episode": [],
        "stage": [],
        "reward": [],
        "fee": [],
        "steps": [],
        "actor_loss": [],
        "critic_loss": [],
        "proxy_block_fee": [],
        "proxy_fairness": [],
        "proxy_risk_exposure": [],
        "proxy_gas_util": [],
        "proxy_age_reward": [],
        "proxy_oldest_cover": [],
        "proxy_terminal_fair_reward": [],
        "proxy_starvation_penalty": [],
        "proxy_terminal_risk_penalty": [],
        "proxy_packing_reward": [],
        "proxy_unused_gas_penalty": [],
        "proxy_fairness_gate": [],
        "lr_scale": [],
        "val_episode": [],
        "val_metric": [],
        "val_score": [],
        "val_selection_score": [],
        "val_score_mean": [],
        "val_score_std": [],
        "val_selection_window_count": [],
        "val_selection_ready": [],
        "val_best_frozen": [],
        "val_no_improvement_count": [],
        "warmstart": warmstart_history,
        "stage_switches": [],
    }

    save_train_pools_limit = int(getattr(C, "V5_SAVE_FULL_TRAIN_POOLS_MAX_EPISODES", 0))
    materialize_train_pools = args.episodes <= save_train_pools_limit
    training_pools = None
    train_pool_path = os.path.join(args.output, "train_pools.json")
    train_manifest_path = os.path.join(args.output, "train_pool_manifest.json")
    if materialize_train_pools:
        training_pools = build_shared_pools(args.episodes, args.pool_size, args.risk_ratio, train_pool_seed)
        train_pool_hash = save_shared_pools(train_pool_path, training_pools, metadata={
            "pool_role": "train",
            "seed": args.seed,
            "generation_seed": train_pool_seed,
            "episodes": args.episodes,
            "pool_size": args.pool_size,
            "risk_ratio": args.risk_ratio,
            "materialized": True,
        })
        train_pool_record = {
            "pool_role": "train",
            "materialized": True,
            "file": os.path.basename(train_pool_path),
            "pool_hash": train_pool_hash,
            "generation_seed": train_pool_seed,
            "episodes": args.episodes,
        }
    else:
        train_pool_record = {
            "pool_role": "train",
            "materialized": False,
            "file": None,
            "pool_hash": None,
            "generation_seed": train_pool_seed,
            "episodes": args.episodes,
            "reconstruction": "TxOrderingEnv seeded with generation_seed; pools generated deterministically per reset",
        }
        with open(train_manifest_path, "w") as f:
            json.dump(train_pool_record, f, indent=2, ensure_ascii=False)

    validation_seed = args.seed + C.VALIDATION_SEED_OFFSET
    validation_pools = build_shared_pools(args.val_episodes, args.pool_size, args.risk_ratio, validation_seed)
    validation_pool_path = os.path.join(args.output, "validation_pools.json")
    validation_pool_hash = save_shared_pools(validation_pool_path, validation_pools, metadata={
        "pool_role": "validation",
        "seed": args.seed,
        "validation_seed": validation_seed,
        "validation_episodes": args.val_episodes,
        "validation_interval": args.val_interval,
        "pool_size": args.pool_size,
        "risk_ratio": args.risk_ratio,
        "validation_metric": args.val_metric,
        "higher_is_better": _higher_is_better(args.val_metric),
        "env_kwargs": base_env_kwargs,
        "curriculum": curriculum_enabled,
        "validation_fairness_floor": args.val_fairness_floor,
        "validation_oldest_coverage_floor": args.val_oldest_coverage_floor,
        "validation_risk_ceil": args.val_risk_ceil,
        "validation_edge10_risk_ceil": args.val_edge10_risk_ceil,
        "validation_top10_risk_ceil": args.val_top10_risk_ceil,  # diagnostic
        "selection_smoothing_window": args.val_smoothing_window,
        "selection_variance_penalty": args.val_variance_penalty,
        "best_freeze_enabled": args.best_freeze,
        "best_freeze_patience": args.best_freeze_patience,
        "best_freeze_min_episodes": args.best_freeze_min_episodes,
    })
    _write_training_progress(
        args.output,
        _progress_payload(args, "training", 0, progress_started_at, current_stage=current_stage),
    )

    best_score = -float("inf")
    best_episode = None
    best_metric_value = None
    best_metric_key = None
    best_raw_score = None
    best_score_mean = None
    best_score_std = None
    best_selection_window_count = 0
    best_frozen = False
    no_improvement_validations = 0
    multi_best = {
        "fairness_recovery": {"score": -float("inf"), "episode": None, "file": "best_fairness_recovery.pt"},
        "risk_aligned": {"score": -float("inf"), "episode": None, "file": "best_risk_aligned.pt"},
        "constrained_fee": {"score": -float("inf"), "episode": None, "file": "best_constrained_fee.pt"},
        "hypervolume": {"score": -float("inf"), "episode": None, "file": "best_hypervolume.pt"},
        "trade_score": {"score": -float("inf"), "episode": None, "file": "best_trade_score.pt"},
        "risk_aware_trade": {"score": -float("inf"), "episode": None, "file": "best_risk_aware_trade.pt"},
        "constrained_trade": {"score": -float("inf"), "episode": None, "file": "best_constrained_trade.pt"},
    }

    for ep in range(1, args.episodes + 1):
        if curriculum_enabled:
            stage = _resolve_curriculum_stage(ep, args.episodes, curriculum_cuts)
        else:
            stage = 3
        if stage != current_stage:
            current_stage = stage
            active_env_kwargs = _curriculum_env_kwargs(base_env_kwargs, args, current_stage)
            env = TxOrderingEnv(
                pool_size=args.pool_size,
                risk_ratio=args.risk_ratio,
                seed=train_pool_seed,
                **active_env_kwargs,
            )
            current_lr_scale = _lr_scale_for_stage(args, current_stage)
            trainer.set_lr_scale(current_lr_scale)
            log["stage_switches"].append({
                "episode": ep,
                "stage": current_stage,
                "lr_scale": current_lr_scale,
                "env_kwargs": active_env_kwargs,
            })

        if training_pools is not None:
            obs, _ = env.reset_with_pool(training_pools[ep - 1])
        else:
            obs, _ = env.reset()
        ep_reward = 0.0

        while True:
            action, log_prob, value = model.act(obs, device)
            next_obs, reward, done, _, info = env.step(action)

            buffer.store(obs, action, log_prob, reward, value, done)
            ep_reward += reward
            obs = next_obs

            if done:
                break

        # PPO 更新
        a_loss, c_loss, entropy = trainer.update(buffer)
        buffer.clear()

        # 训练期代理指标（与论文评估口径一致）
        selected = env.get_selected_transactions()
        pool = env.get_pool()
        proxy = compute_all_metrics(selected, pool)

        # 记录
        log["episode"].append(ep)
        log["stage"].append(current_stage)
        log["reward"].append(ep_reward)
        log["fee"].append(info.get("total_fee", 0))
        log["steps"].append(info.get("num_selected", 0))
        log["actor_loss"].append(a_loss)
        log["critic_loss"].append(c_loss)
        log["proxy_block_fee"].append(proxy["block_fee"])
        log["proxy_fairness"].append(proxy["fairness"])
        log["proxy_risk_exposure"].append(proxy["risk_exposure"])
        log["proxy_gas_util"].append(proxy["gas_util"])
        log["proxy_age_reward"].append(info.get("proxy_age_reward", 0.0))
        log["proxy_oldest_cover"].append(info.get("proxy_oldest_cover", 0.0))
        reward_decomp = info.get("reward_decomposition", {})
        log["proxy_terminal_fair_reward"].append(
            reward_decomp.get("terminal_fair_reward", 0.0)
        )
        log["proxy_starvation_penalty"].append(info.get("proxy_starvation_penalty", 0.0))
        log["proxy_terminal_risk_penalty"].append(info.get("proxy_terminal_risk_penalty", 0.0))
        log["proxy_packing_reward"].append(info.get("proxy_packing_reward", 0.0))
        log["proxy_unused_gas_penalty"].append(info.get("proxy_unused_gas_penalty", 0.0))
        log["proxy_fairness_gate"].append(info.get("proxy_fairness_gate", 1.0))
        log["lr_scale"].append(current_lr_scale)
        _write_training_progress(
            args.output,
            _progress_payload(
                args,
                "training",
                ep,
                progress_started_at,
                current_stage=current_stage,
                last_reward=float(ep_reward),
                last_fee=float(info.get("total_fee", 0)),
                last_steps=int(info.get("num_selected", 0)),
                best_episode=best_episode,
                best_score=None if best_score == -float("inf") else float(best_score),
                best_frozen=best_frozen,
                last_validation_episode=log["val_episode"][-1] if log["val_episode"] else None,
            ),
        )

        if ep % args.log_interval == 0:
            recent = log["reward"][-args.log_interval:]
            avg_r = np.mean(recent)
            avg_fee = np.mean(log["fee"][-args.log_interval:])
            print(f"Episode {ep:>6d} | "
                  f"AvgReward {avg_r:>8.2f} | "
                  f"AvgFee {avg_fee:>10.1f} | "
                  f"ActorLoss {a_loss:.4f} | "
                  f"CriticLoss {c_loss:.4f} | "
                  f"Entropy {entropy:.4f}", flush=True)

        if ep % args.val_interval == 0 or ep == args.episodes:
            _write_training_progress(
                args.output,
                _progress_payload(
                    args,
                    "validating",
                    ep,
                    progress_started_at,
                    current_stage=current_stage,
                    last_reward=float(ep_reward),
                    last_fee=float(info.get("total_fee", 0)),
                    last_steps=int(info.get("num_selected", 0)),
                    best_episode=best_episode,
                    best_score=None if best_score == -float("inf") else float(best_score),
                    best_frozen=best_frozen,
                    last_validation_episode=log["val_episode"][-1] if log["val_episode"] else None,
                ),
            )
            model.eval()
            val = _run_validation(model, device, args, validation_pools, base_env_kwargs)
            model.train()
            raw_val_score = float(val["score"])
            log["val_episode"].append(ep)
            log["val_metric"].append(val["metric_value"])
            log["val_score"].append(raw_val_score)
            selection = _stability_adjusted_score(
                log["val_score"],
                args.val_smoothing_window,
                args.val_variance_penalty,
            )
            selection_score = float(selection["score"])
            selection_ready = selection["count"] >= args.val_smoothing_window or ep == args.episodes
            log["val_selection_score"].append(selection_score)
            log["val_score_mean"].append(selection["mean"])
            log["val_score_std"].append(selection["std"])
            log["val_selection_window_count"].append(selection["count"])
            log["val_selection_ready"].append(bool(selection_ready))

            if selection_ready and not best_frozen:
                if best_episode is None or selection_score > best_score:
                    best_score = selection_score
                    best_episode = ep
                    best_metric_value = val["metric_value"]
                    best_metric_key = val["metric_key"]
                    best_raw_score = raw_val_score
                    best_score_mean = selection["mean"]
                    best_score_std = selection["std"]
                    best_selection_window_count = selection["count"]
                    no_improvement_validations = 0
                    torch.save(model.state_dict(),
                               os.path.join(args.output, C.BEST_CHECKPOINT_NAME))
                elif best_episode is not None:
                    no_improvement_validations += 1

                if (
                    args.best_freeze
                    and best_episode is not None
                    and ep >= args.best_freeze_min_episodes
                    and no_improvement_validations >= args.best_freeze_patience
                ):
                    best_frozen = True

            log["val_best_frozen"].append(bool(best_frozen))
            log["val_no_improvement_count"].append(int(no_improvement_validations))

            extra_scores = _multi_checkpoint_scores(val["aggregate"], args)
            for ckpt_name, payload in multi_best.items():
                candidate = float(extra_scores.get(ckpt_name, -float("inf")))
                if candidate > payload["score"]:
                    payload["score"] = candidate
                    payload["episode"] = ep
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output, payload["file"]),
                    )
            _write_training_progress(
                args.output,
                _progress_payload(
                    args,
                    "training" if ep < args.episodes else "finalizing",
                    ep,
                    progress_started_at,
                    current_stage=current_stage,
                    last_reward=float(ep_reward),
                    last_fee=float(info.get("total_fee", 0)),
                    last_steps=int(info.get("num_selected", 0)),
                    best_episode=best_episode,
                    best_score=None if best_score == -float("inf") else float(best_score),
                    best_frozen=best_frozen,
                    last_validation_episode=ep,
                ),
            )

    # 保存最终模型和训练日志
    torch.save(model.state_dict(),
               os.path.join(args.output, C.FINAL_CHECKPOINT_NAME))
    with open(os.path.join(args.output, "train_log.json"), "w") as f:
        json.dump(log, f)
    with open(os.path.join(args.output, "checkpoint_meta.json"), "w") as f:
        json.dump({
            "formal_eval_checkpoint": C.FORMAL_EVAL_CHECKPOINT_NAME,
            "formal_eval_rule": C.FORMAL_EVAL_CHECKPOINT_RULE,
            "best_checkpoint": C.BEST_CHECKPOINT_NAME,
            "final_checkpoint": C.FINAL_CHECKPOINT_NAME,
            "data_pools": {
                "train": train_pool_record,
                "validation": {
                    "pool_role": "validation",
                    "file": os.path.basename(validation_pool_path),
                    "pool_hash": validation_pool_hash,
                    "generation_seed": validation_seed,
                    "episodes": args.val_episodes,
                },
            },
            "reward_config": {
                "env_kwargs": base_env_kwargs,
                "risk_adjusted_fee_lambda": C.RISK_ADJUSTED_FEE_LAMBDA,
            },
            "selection_rule": {
                "type": "fixed_validation_pool_smoothed_stability_score",
                "validation_metric": args.val_metric,
                "higher_is_better": _higher_is_better(args.val_metric),
                "validation_episodes": args.val_episodes,
                "validation_interval": args.val_interval,
                "selection_score_log_key": "val_selection_score",
                "raw_score_log_key": "val_score",
                "smoothing_window": args.val_smoothing_window,
                "variance_penalty": args.val_variance_penalty,
                "best_freeze_enabled": args.best_freeze,
                "best_freeze_patience": args.best_freeze_patience,
                "best_freeze_min_episodes": args.best_freeze_min_episodes,
                "best_frozen": best_frozen,
                "no_improvement_validations": no_improvement_validations,
                "validation_seed": validation_seed,
                "validation_pool_file": os.path.basename(validation_pool_path),
                "validation_pool_hash": validation_pool_hash,
                "validation_fairness_floor": args.val_fairness_floor,
                "validation_oldest_coverage_floor": args.val_oldest_coverage_floor,
                "validation_risk_ceil": args.val_risk_ceil,
                "validation_edge10_risk_ceil": args.val_edge10_risk_ceil,
                "validation_top10_risk_ceil": args.val_top10_risk_ceil,  # diagnostic
                "best_episode": best_episode,
                "best_metric_key": best_metric_key,
                "best_metric_value": best_metric_value,
                "best_raw_score": best_raw_score,
                "best_score_mean": best_score_mean,
                "best_score_std": best_score_std,
                "best_selection_window_count": best_selection_window_count,
                "best_selection_score": best_score,
            },
            "warmstart": {
                "policy": pretrain_policy,
                "epochs": pretrain_epochs,
                "episodes_per_epoch": pretrain_episodes_per_epoch,
                "history": warmstart_history,
            },
            "curriculum": {
                "enabled": curriculum_enabled,
                "fairness_first": fairness_first,
                "stage_episode_cuts": list(curriculum_cuts),
                "stage_switches": log["stage_switches"],
            },
            "lr_schedule": {
                "enabled": args.lr_schedule,
                "stage_scales": list(args.lr_stage_scales),
            },
            "multi_checkpoints": {
                name: {
                    "file": payload["file"],
                    "best_score": payload["score"],
                    "best_episode": payload["episode"],
                }
                for name, payload in multi_best.items()
            },
            "training_proxy_metrics": [
                "proxy_block_fee",
                "proxy_fairness",
                "proxy_risk_exposure",
                "proxy_gas_util",
                "proxy_age_reward",
                "proxy_oldest_cover",
                "proxy_terminal_fair_reward",
                "proxy_starvation_penalty",
                "proxy_terminal_risk_penalty",
                "proxy_packing_reward",
                "proxy_unused_gas_penalty",
                "proxy_fairness_gate",
            ],
            "reward_eval_note": (
                "Training reward is a proxy objective; formal reporting metrics are "
                "computed by the shared metric pipeline in evaluation."
            ),
        }, f, indent=2, ensure_ascii=False)
    _write_training_progress(
        args.output,
        _progress_payload(
            args,
            "completed",
            args.episodes,
            progress_started_at,
            current_stage=current_stage,
            best_episode=best_episode,
            best_score=None if best_score == -float("inf") else float(best_score),
            best_frozen=best_frozen,
            last_validation_episode=log["val_episode"][-1] if log["val_episode"] else None,
        ),
    )
    print(f"Training complete. Models saved to {args.output}/", flush=True)
    return {
        "best_episode": best_episode,
        "best_metric_key": best_metric_key,
        "best_metric_value": best_metric_value,
        "best_score": best_score,
    }


def main():
    parser = argparse.ArgumentParser(description="PPO 交易排序训练")
    parser.add_argument("--episodes", type=int, default=C.TOTAL_EPISODES)
    parser.add_argument("--pool-size", type=C.validate_pool_size,
                        default=C.POOL_SIZE_DEFAULT)
    parser.add_argument("--risk-ratio", type=float, default=C.RISK_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=C.LOG_INTERVAL)
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练设备: 默认 cuda:0，可手动指定 cpu/cuda/cuda:1 ...")
    parser.add_argument("--val-episodes", type=int, default=C.VALIDATION_EPISODES)
    parser.add_argument("--val-interval", type=int, default=C.VALIDATION_INTERVAL)
    parser.add_argument("--val-metric", type=str, default=C.VALIDATION_METRIC)
    parser.add_argument("--val-smoothing-window", type=int, default=getattr(C, "VALIDATION_SMOOTHING_WINDOW", 1))
    parser.add_argument("--val-variance-penalty", type=float, default=getattr(C, "VALIDATION_VARIANCE_PENALTY", 0.0))
    parser.add_argument("--best-freeze", dest="best_freeze", action="store_true", default=getattr(C, "BEST_FREEZE_ENABLED", False))
    parser.add_argument("--no-best-freeze", dest="best_freeze", action="store_false")
    parser.add_argument("--best-freeze-patience", type=int, default=getattr(C, "BEST_FREEZE_PATIENCE", 8))
    parser.add_argument("--best-freeze-min-episodes", type=int, default=getattr(C, "BEST_FREEZE_MIN_EPISODES", 300))
    parser.add_argument("--val-fairness-floor", type=float, default=C.VALIDATION_FAIRNESS_FLOOR)
    parser.add_argument("--val-oldest-coverage-floor", type=float, default=C.VALIDATION_OLDEST_COVERAGE_FLOOR)
    parser.add_argument("--val-risk-ceil", type=float, default=C.VALIDATION_RISK_CEIL)
    parser.add_argument("--val-edge10-risk-ceil", type=float, default=C.VALIDATION_EDGE10_RISK_CEIL)
    parser.add_argument("--val-top10-risk-ceil", type=float, default=C.VALIDATION_TOP10_RISK_CEIL)  # diagnostic
    parser.add_argument("--pretrain-policy", type=str, default=C.PRETRAIN_POLICY,
                        choices=["none", "fifo", "fair_fee", "mixed"])
    parser.add_argument("--pretrain-epochs", type=int, default=C.PRETRAIN_EPOCHS)
    parser.add_argument("--pretrain-episodes-per-epoch", type=int, default=C.PRETRAIN_EPISODES_PER_EPOCH)
    parser.add_argument("--curriculum", action="store_true", default=C.CURRICULUM_ENABLED)
    parser.add_argument("--curriculum-stage-episodes", type=float, nargs=3, default=C.CURRICULUM_STAGE_EPISODES)
    parser.add_argument("--fairness-first", dest="fairness_first", action="store_true",
                        default=getattr(C, "FAIRNESS_FIRST_CURRICULUM_ENABLED", True))
    parser.add_argument("--no-fairness-first", dest="fairness_first", action="store_false")
    parser.add_argument("--lr-schedule", action="store_true", default=getattr(C, "LR_SCHEDULE_ENABLED", False))
    parser.add_argument("--lr-stage-scales", type=float, nargs=3, default=getattr(C, "LR_STAGE_SCALES", (1.0, 0.5, 0.2)))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main_seed(args)


def main_seed(args):
    seed_everything(args.seed)
    train(args)


if __name__ == "__main__":
    main()
