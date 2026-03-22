"""一键实验编排: 多种子并行训练 → 评估 → 聚合 → LaTeX 表格"""

import argparse
import os
import json
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import config as C
from device_utils import resolve_device, seed_everything
from env import TxOrderingEnv
from networks import ActorCritic
from train import train as train_model
from evaluate import (build_shared_pools, evaluate_rl, evaluate_baseline, aggregate,
                      run_robustness, run_pool_size_robustness,
                      run_fee_multiplier_robustness,
                      plot_training_curve)
from latex_tables import (generate_main_table, generate_robustness_table,
                          generate_ablation_table, ABLATION_ORDER,
                          ABLATION_LABELS, STRUCT_ABLATION_ORDER,
                          STRUCT_ABLATION_LABELS)
from stat_tests import (SCIPY_AVAILABLE, run_significance_tests,
                        format_significance_table, generate_significance_latex)


def run_single_seed(seed, args_dict):
    """单个种子: 训练 + 评估 + 鲁棒性 (独立进程)"""
    args = argparse.Namespace(**args_dict)
    device = resolve_device(args.device)

    seed_dir = os.path.join(args.output, f"seed_{seed}")
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    result_dir = os.path.join(seed_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    model_path = os.path.join(ckpt_dir, "best_model.pt")
    log_path = os.path.join(ckpt_dir, "train_log.json")

    # 训练
    if not args.skip_train:
        print(f"[seed={seed}] 开始训练", flush=True)
        seed_everything(seed)
        train_args = argparse.Namespace(
            episodes=args.episodes, pool_size=args.pool_size,
            risk_ratio=C.RISK_RATIO, seed=seed,
            log_interval=C.LOG_INTERVAL, output=ckpt_dir,
            device=args.device,
        )
        train_model(train_args)
        print(f"[seed={seed}] 训练完成", flush=True)

    # 加载模型
    model = ActorCritic().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
    else:
        print(f"[seed={seed}] Warning: {model_path} not found, using random policy")
    model.eval()

    # 主实验评估
    print(f"[seed={seed}] 主实验评估", flush=True)
    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=C.RISK_RATIO, seed=seed)
    shared_pools = build_shared_pools(args.eval_episodes, args.pool_size,
                                      C.RISK_RATIO, seed)
    main_results = {}
    main_results["RL (Ours)"] = aggregate(
        evaluate_rl(model, env, args.eval_episodes, device, shared_pools))
    for bl in ["fifo", "gas", "heuristic", "fee_risk_linear", "fair_fee"]:
        label = bl.upper().replace("FEE_RISK_LINEAR", "FeeRiskLinear").replace("FAIR_FEE", "FairFee")
        main_results[label] = aggregate(
            evaluate_baseline(bl, env, args.eval_episodes, shared_pools))
    with open(os.path.join(result_dir, "main_results.json"), "w") as f:
        json.dump(main_results, f, indent=2, ensure_ascii=False)

    # 鲁棒性实验: 风险比例
    print(f"[seed={seed}] 鲁棒性实验 (风险比例)", flush=True)
    rob_data = run_robustness(model, device, args.eval_episodes,
                              args.pool_size, seed, result_dir)

    # 鲁棒性实验: 候选池规模
    print(f"[seed={seed}] 鲁棒性实验 (池大小)", flush=True)
    rob_pool = run_pool_size_robustness(model, device, args.eval_episodes,
                                        C.RISK_RATIO, seed, result_dir)

    # 鲁棒性实验: 风险费率倍率
    print(f"[seed={seed}] 鲁棒性实验 (费率倍率)", flush=True)
    rob_fee_mult = run_fee_multiplier_robustness(
        model, device, args.eval_episodes,
        args.pool_size, seed, result_dir)

    # 训练曲线
    if os.path.exists(log_path):
        plot_training_curve(log_path,
                            os.path.join(result_dir, "training_curve.png"))

    print(f"[seed={seed}] 全部完成", flush=True)
    return seed, main_results, rob_data, rob_pool, rob_fee_mult


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


def aggregate_across_seeds(all_main, all_rob_risk, all_rob_pool, all_rob_fee):
    """跨种子聚合: 对每个方法的 mean 取均值和标准差"""
    methods = list(all_main[0].keys())
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util",
               "risky_rank", "packing_ratio", "top10_risk", "late_promo"]

    agg_main = {}
    for m in methods:
        agg_main[m] = {}
        for met in metrics:
            vals = [r[m][f"{met}_mean"] for r in all_main]
            agg_main[m][f"{met}_mean"] = float(np.mean(vals))
            agg_main[m][f"{met}_std"] = float(np.std(vals))

    agg_rob_risk = aggregate_metric_grid(all_rob_risk)
    agg_rob_pool = aggregate_metric_grid(all_rob_pool)
    agg_rob_fee = aggregate_metric_grid(all_rob_fee)

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


def _train_ablation_single(name, seed, args_dict, env_kwargs):
    """单个消融变体+种子的训练与评估 (独立进程)"""
    from ppo import PPOTrainer, RolloutBuffer
    args = argparse.Namespace(**args_dict)
    device = resolve_device(args.device)

    abl_dir = os.path.join(args.output, "ablation", name, f"seed_{seed}")
    ckpt_dir = os.path.join(abl_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, "best_model.pt")

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
            trainer.update(buffer)
            buffer.clear()
            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(model.state_dict(), model_path)
        print(f"  [Ablation] {name} seed={seed} 训练完成", flush=True)

    model = ActorCritic().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
    model.eval()

    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=C.RISK_RATIO, seed=seed,
                        **env_kwargs)
    result = aggregate(evaluate_rl(model, env, args.eval_episodes, device))
    return name, seed, result


def run_ablation_parallel(configs, args, label_prefix):
    """并行运行消融实验"""
    metrics = ["block_fee", "fairness", "risk_exposure", "gas_util"]
    args_dict = vars(args)
    tasks = []

    max_workers = min(args.workers, len(configs) * len(args.seeds))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, env_kwargs in configs.items():
            for seed in args.seeds:
                fut = executor.submit(_train_ablation_single,
                                      f"{label_prefix}_{name}",
                                      seed, args_dict, env_kwargs)
                futures[fut] = (name, seed)

        results = {name: [] for name in configs}
        for fut in as_completed(futures):
            name_full, seed, res = fut.result()
            orig_name = futures[fut][0]
            results[orig_name].append(res)

    agg = {}
    for name in configs:
        agg[name] = {}
        for met in metrics:
            vals = [r[f"{met}_mean"] for r in results[name]]
            agg[name][f"{met}_mean"] = float(np.mean(vals))
            agg[name][f"{met}_std"] = float(np.std(vals))
    return agg


def main():
    parser = argparse.ArgumentParser(description="一键实验编排 (并行版)")
    parser.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    parser.add_argument("--episodes", type=int, default=C.TOTAL_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=C.EVAL_EPISODES)
    parser.add_argument("--pool-size", type=int, default=C.POOL_SIZE_DEFAULT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results_final")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练, 仅评估已有模型")
    parser.add_argument("--ablation", action="store_true",
                        help="运行消融实验 (奖励 + 结构)")
    parser.add_argument("--workers", type=int, default=3,
                        help="并行进程数 (默认 3)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Workers: {args.workers}")
    print(f"Episodes: {args.episodes}")
    print(f"Pool size: {args.pool_size}")

    # 并行训练+评估所有 seed
    args_dict = vars(args)
    all_main = [None] * len(args.seeds)
    all_rob_risk = [None] * len(args.seeds)
    all_rob_pool = [None] * len(args.seeds)
    all_rob_fee = [None] * len(args.seeds)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, seed in enumerate(args.seeds):
            fut = executor.submit(run_single_seed, seed, args_dict)
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            seed, main_res, rob_risk_res, rob_pool_res, rob_fee_res = fut.result()
            all_main[idx] = main_res
            all_rob_risk[idx] = rob_risk_res
            all_rob_pool[idx] = rob_pool_res
            all_rob_fee[idx] = rob_fee_res
            print(f"[seed={seed}] 结果已收集 ({idx+1}/{len(args.seeds)})",
                  flush=True)

    # 跨种子聚合
    agg_main, agg_rob_risk, agg_rob_pool, agg_rob_fee = aggregate_across_seeds(
        all_main, all_rob_risk, all_rob_pool, all_rob_fee)

    agg_main_path = os.path.join(args.output, "aggregated_main.json")
    agg_rob_risk_path = os.path.join(args.output, "aggregated_robustness_risk.json")
    agg_rob_pool_path = os.path.join(args.output, "aggregated_robustness_pool.json")
    agg_rob_fee_path = os.path.join(args.output, "aggregated_robustness_fee.json")
    with open(agg_main_path, "w") as f:
        json.dump(agg_main, f, indent=2, ensure_ascii=False)
    with open(agg_rob_risk_path, "w") as f:
        json.dump(agg_rob_risk, f, indent=2, ensure_ascii=False)
    with open(agg_rob_pool_path, "w") as f:
        json.dump(agg_rob_pool, f, indent=2, ensure_ascii=False)
    with open(agg_rob_fee_path, "w") as f:
        json.dump(agg_rob_fee, f, indent=2, ensure_ascii=False)
    print(
        "\n聚合结果已保存: "
        f"{agg_main_path}, {agg_rob_risk_path}, {agg_rob_pool_path}, {agg_rob_fee_path}"
    )

    # 生成 LaTeX 表格
    t2 = generate_main_table(agg_main)
    t3_risk = generate_robustness_table(agg_rob_risk)
    t3_pool = generate_robustness_table(agg_rob_pool)
    t3_fee = generate_robustness_table(agg_rob_fee)
    t2_path = os.path.join(args.output, "table2_content.tex")
    t3_risk_path = os.path.join(args.output, "table3_risk_content.tex")
    t3_pool_path = os.path.join(args.output, "table3_pool_content.tex")
    t3_fee_path = os.path.join(args.output, "table3_fee_content.tex")
    with open(t2_path, "w") as f:
        f.write(t2)
    with open(t3_risk_path, "w") as f:
        f.write(t3_risk)
    with open(t3_pool_path, "w") as f:
        f.write(t3_pool)
    with open(t3_fee_path, "w") as f:
        f.write(t3_fee)
    print(
        "LaTeX 表格已保存: "
        f"{t2_path}, {t3_risk_path}, {t3_pool_path}, {t3_fee_path}"
    )

    # 统计显著性检验
    if SCIPY_AVAILABLE and len(all_main) >= 2:
        print("\n===== 统计显著性检验 =====")
        sig = run_significance_tests(all_main)
        print(format_significance_table(sig))
        sig_path = os.path.join(args.output, "significance_tests.json")
        with open(sig_path, "w") as f:
            json.dump(sig, f, indent=2, ensure_ascii=False)
        sig_tex = os.path.join(args.output, "table_significance.tex")
        generate_significance_latex(sig, sig_tex)
        print(f"显著性检验已保存: {sig_path}, {sig_tex}")
    else:
        print("\n===== 统计显著性检验 =====")
        if not SCIPY_AVAILABLE:
            print("Warning: scipy 未安装, 跳过显著性检验与对应 LaTeX 表格生成。")
        else:
            print("Warning: 显著性检验至少需要 2 个 seed, 当前已跳过。")

    # 消融实验 (并行)
    if args.ablation:
        print("\n===== 奖励消融实验 =====", flush=True)
        agg_reward_abl = run_ablation_parallel(
            REWARD_ABLATION_CONFIGS, args, "reward")
        t_rw = generate_ablation_table(agg_reward_abl, ABLATION_ORDER,
                                       ABLATION_LABELS)
        rw_path = os.path.join(args.output, "table_ablation_reward.tex")
        with open(rw_path, "w") as f:
            f.write(t_rw)
        abl_rw_json = os.path.join(args.output, "ablation_reward.json")
        with open(abl_rw_json, "w") as f:
            json.dump(agg_reward_abl, f, indent=2, ensure_ascii=False)

        print("\n===== 结构消融实验 =====", flush=True)
        agg_struct_abl = run_ablation_parallel(
            STRUCT_ABLATION_CONFIGS, args, "struct")
        t_st = generate_ablation_table(agg_struct_abl, STRUCT_ABLATION_ORDER,
                                       STRUCT_ABLATION_LABELS)
        st_path = os.path.join(args.output, "table_ablation_struct.tex")
        with open(st_path, "w") as f:
            f.write(t_st)
        abl_st_json = os.path.join(args.output, "ablation_struct.json")
        with open(abl_st_json, "w") as f:
            json.dump(agg_struct_abl, f, indent=2, ensure_ascii=False)
        print(f"消融表格已保存: {rw_path}, {st_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
