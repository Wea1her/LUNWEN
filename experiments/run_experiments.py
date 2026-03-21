"""一键实验编排: 多种子训练 → 评估 → 聚合 → LaTeX 表格"""

import argparse
import os
import json
import numpy as np

import config as C
from device_utils import resolve_device, seed_everything
from env import TxOrderingEnv
from networks import ActorCritic
from train import train as train_model
from evaluate import (evaluate_rl, evaluate_baseline, aggregate,
                      run_robustness, plot_training_curve)
from latex_tables import generate_main_table, generate_robustness_table


def run_single_seed(seed, args, device):
    """单个种子: 训练 + 评估 + 鲁棒性"""
    seed_dir = os.path.join(args.output, f"seed_{seed}")
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    result_dir = os.path.join(seed_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    model_path = os.path.join(ckpt_dir, "best_model.pt")
    log_path = os.path.join(ckpt_dir, "train_log.json")

    # 训练
    if not args.skip_train:
        print(f"\n{'='*50}")
        print(f"训练 seed={seed}")
        print(f"{'='*50}")
        seed_everything(seed)
        train_args = argparse.Namespace(
            episodes=args.episodes, pool_size=args.pool_size,
            risk_ratio=C.RISK_RATIO, seed=seed,
            log_interval=C.LOG_INTERVAL, output=ckpt_dir,
            device=args.device,
        )
        train_model(train_args)

    # 加载模型
    import torch
    model = ActorCritic().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
    else:
        print(f"Warning: {model_path} not found, using random policy")
    model.eval()

    # 主实验评估
    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=C.RISK_RATIO, seed=seed)
    main_results = {}
    main_results["RL (Ours)"] = aggregate(
        evaluate_rl(model, env, args.eval_episodes, device))
    for bl in ["fifo", "gas", "heuristic"]:
        main_results[bl.upper()] = aggregate(
            evaluate_baseline(bl, env, args.eval_episodes))
    with open(os.path.join(result_dir, "main_results.json"), "w") as f:
        json.dump(main_results, f, indent=2, ensure_ascii=False)

    # 鲁棒性实验
    rob_data = run_robustness(model, device, args.eval_episodes,
                              args.pool_size, seed, result_dir)

    # 训练曲线
    if os.path.exists(log_path):
        plot_training_curve(log_path,
                            os.path.join(result_dir, "training_curve.png"))

    return main_results, rob_data


def aggregate_across_seeds(all_main, all_rob):
    """跨种子聚合: 对每个方法的 mean 取均值和标准差"""
    methods = list(all_main[0].keys())
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util"]

    agg_main = {}
    for m in methods:
        agg_main[m] = {}
        for met in metrics:
            vals = [r[m][f"{met}_mean"] for r in all_main]
            agg_main[m][f"{met}_mean"] = float(np.mean(vals))
            agg_main[m][f"{met}_std"] = float(np.std(vals))

    risk_ratios = list(all_rob[0].keys())
    rob_methods = list(all_rob[0][risk_ratios[0]].keys())
    agg_rob = {}
    for rr in risk_ratios:
        agg_rob[rr] = {}
        for m in rob_methods:
            agg_rob[rr][m] = {}
            for met in metrics:
                vals = [r[rr][m][f"{met}_mean"] for r in all_rob]
                agg_rob[rr][m][f"{met}_mean"] = float(np.mean(vals))
                agg_rob[rr][m][f"{met}_std"] = float(np.std(vals))

    return agg_main, agg_rob


def main():
    parser = argparse.ArgumentParser(description="一键实验编排")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--episodes", type=int, default=C.TOTAL_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=C.EVAL_EPISODES)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results_final")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练, 仅评估已有模型")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")

    all_main = []
    all_rob = []
    for seed in args.seeds:
        main_res, rob_res = run_single_seed(seed, args, device)
        all_main.append(main_res)
        all_rob.append(rob_res)

    # 跨种子聚合
    agg_main, agg_rob = aggregate_across_seeds(all_main, all_rob)

    agg_main_path = os.path.join(args.output, "aggregated_main.json")
    agg_rob_path = os.path.join(args.output, "aggregated_robustness.json")
    with open(agg_main_path, "w") as f:
        json.dump(agg_main, f, indent=2, ensure_ascii=False)
    with open(agg_rob_path, "w") as f:
        json.dump(agg_rob, f, indent=2, ensure_ascii=False)
    print(f"\n聚合结果已保存: {agg_main_path}, {agg_rob_path}")

    # 生成 LaTeX 表格
    t2 = generate_main_table(agg_main)
    t3 = generate_robustness_table(agg_rob)
    t2_path = os.path.join(args.output, "table2_content.tex")
    t3_path = os.path.join(args.output, "table3_content.tex")
    with open(t2_path, "w") as f:
        f.write(t2)
    with open(t3_path, "w") as f:
        f.write(t3)
    print(f"LaTeX 表格已保存: {t2_path}, {t3_path}")


if __name__ == "__main__":
    main()
