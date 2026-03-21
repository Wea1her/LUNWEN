"""训练入口"""

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


def train(args):
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=args.risk_ratio,
                        seed=args.seed)

    model = ActorCritic().to(device)
    trainer = PPOTrainer(model, device=device)
    buffer = RolloutBuffer()

    log = {"episode": [], "reward": [], "fee": [],
           "steps": [], "actor_loss": [], "critic_loss": []}

    best_reward = -float("inf")

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

        # 记录
        log["episode"].append(ep)
        log["reward"].append(ep_reward)
        log["fee"].append(info.get("total_fee", 0))
        log["steps"].append(info.get("num_selected", 0))
        log["actor_loss"].append(a_loss)
        log["critic_loss"].append(c_loss)

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

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(model.state_dict(),
                       os.path.join(args.output, "best_model.pt"))

    # 保存最终模型和训练日志
    torch.save(model.state_dict(),
               os.path.join(args.output, "final_model.pt"))
    with open(os.path.join(args.output, "train_log.json"), "w") as f:
        json.dump(log, f)
    print(f"Training complete. Models saved to {args.output}/")


def main():
    parser = argparse.ArgumentParser(description="PPO 交易排序训练")
    parser.add_argument("--episodes", type=int, default=C.TOTAL_EPISODES)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--risk-ratio", type=float, default=C.RISK_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=C.LOG_INTERVAL)
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "cuda:0"],
                        help="训练设备: auto/cpu/cuda/cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main_seed(args)


def main_seed(args):
    seed_everything(args.seed)
    train(args)


if __name__ == "__main__":
    main()
