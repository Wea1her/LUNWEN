"""训练入口"""

from __future__ import annotations

import argparse
import os
import json
import torch
import numpy as np

import config as C
from env import TxOrderingEnv
from networks import ActorCritic
from ppo import PPOTrainer, RolloutBuffer
from device_utils import resolve_device, seed_everything
from evaluate import aggregate, build_shared_pools, evaluate_rl, save_shared_pools
from metrics import compute_all_metrics


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


def train(args, env_kwargs: dict | None = None):
    env_kwargs = env_kwargs or {}
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=args.risk_ratio,
                        seed=args.seed,
                        **env_kwargs)

    model = ActorCritic().to(device)
    trainer = PPOTrainer(model, device=device)
    buffer = RolloutBuffer()

    log = {
        "episode": [],
        "reward": [],
        "fee": [],
        "steps": [],
        "actor_loss": [],
        "critic_loss": [],
        "proxy_block_fee": [],
        "proxy_fairness": [],
        "proxy_risk_exposure": [],
        "proxy_gas_util": [],
        "val_episode": [],
        "val_metric": [],
        "val_score": [],
    }

    validation_seed = args.seed + C.VALIDATION_SEED_OFFSET
    validation_pools = build_shared_pools(args.val_episodes, args.pool_size, args.risk_ratio, validation_seed)
    validation_pool_path = os.path.join(args.output, "validation_pools.json")
    save_shared_pools(validation_pool_path, validation_pools, metadata={
        "seed": args.seed,
        "validation_seed": validation_seed,
        "validation_episodes": args.val_episodes,
        "validation_interval": args.val_interval,
        "pool_size": args.pool_size,
        "risk_ratio": args.risk_ratio,
        "validation_metric": args.val_metric,
        "higher_is_better": _higher_is_better(args.val_metric),
        "env_kwargs": env_kwargs,
    })

    best_score = -float("inf")
    best_episode = None
    best_metric_value = None
    best_metric_key = None

    for ep in range(1, args.episodes + 1):
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
        log["reward"].append(ep_reward)
        log["fee"].append(info.get("total_fee", 0))
        log["steps"].append(info.get("num_selected", 0))
        log["actor_loss"].append(a_loss)
        log["critic_loss"].append(c_loss)
        log["proxy_block_fee"].append(proxy["block_fee"])
        log["proxy_fairness"].append(proxy["fairness"])
        log["proxy_risk_exposure"].append(proxy["risk_exposure"])
        log["proxy_gas_util"].append(proxy["gas_util"])

        if ep % args.log_interval == 0:
            recent = log["reward"][-args.log_interval:]
            avg_r = np.mean(recent)
            avg_fee = np.mean(log["fee"][-args.log_interval:])
            print(f"Episode {ep:>6d} | "
                  f"AvgReward {avg_r:>8.2f} | "
                  f"AvgFee {avg_fee:>10.1f} | "
                  f"ActorLoss {a_loss:.4f} | "
                  f"CriticLoss {c_loss:.4f} | "
                  f"Entropy {entropy:.4f}")

        if ep % args.val_interval == 0 or ep == args.episodes:
            model.eval()
            val = _run_validation(model, device, args, validation_pools, env_kwargs)
            model.train()
            log["val_episode"].append(ep)
            log["val_metric"].append(val["metric_value"])
            log["val_score"].append(val["score"])
            if val["score"] > best_score:
                best_score = val["score"]
                best_episode = ep
                best_metric_value = val["metric_value"]
                best_metric_key = val["metric_key"]
                torch.save(model.state_dict(),
                           os.path.join(args.output, C.BEST_CHECKPOINT_NAME))

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
            "selection_rule": {
                "type": "fixed_validation_pool",
                "validation_metric": args.val_metric,
                "higher_is_better": _higher_is_better(args.val_metric),
                "validation_episodes": args.val_episodes,
                "validation_interval": args.val_interval,
                "validation_seed": validation_seed,
                "validation_pool_file": os.path.basename(validation_pool_path),
                "best_episode": best_episode,
                "best_metric_key": best_metric_key,
                "best_metric_value": best_metric_value,
                "best_selection_score": best_score,
            },
            "training_proxy_metrics": [
                "proxy_block_fee",
                "proxy_fairness",
                "proxy_risk_exposure",
                "proxy_gas_util",
            ],
            "reward_eval_note": (
                "Training reward is a proxy objective; formal reporting metrics are "
                "computed by the shared metric pipeline in evaluation."
            ),
        }, f, indent=2, ensure_ascii=False)
    print(f"Training complete. Models saved to {args.output}/")
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
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备: auto/cpu/cuda/cuda:0/cuda:1...")
    parser.add_argument("--val-episodes", type=int, default=C.VALIDATION_EPISODES)
    parser.add_argument("--val-interval", type=int, default=C.VALIDATION_INTERVAL)
    parser.add_argument("--val-metric", type=str, default=C.VALIDATION_METRIC)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main_seed(args)


def main_seed(args):
    seed_everything(args.seed)
    train(args)


if __name__ == "__main__":
    main()
