"""评估脚本: 对比 RL 方法与基线, 生成指标表格与图表"""

import argparse
from copy import deepcopy
import dataclasses
import os
import json
import numpy as np
import torch

import config as C
from env import TxOrderingEnv
from networks import ActorCritic
from baselines import run_baseline
from metrics import compute_all_metrics
from device_utils import resolve_device
from transaction import Transaction, generate_pool


def build_shared_pools(n_episodes: int, pool_size: int,
                       risk_ratio: float, seed: int) -> list[list]:
    """为不同方法预生成完全一致的评估交易池。"""
    rng = np.random.default_rng(seed)
    return [generate_pool(rng, pool_size, risk_ratio) for _ in range(n_episodes)]


def save_shared_pools(path: str, shared_pools: list[list[Transaction]],
                      metadata: dict | None = None):
    """将共享评估池落盘, 便于不同实验变体复用和复核。"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "pools": [
            [dataclasses.asdict(tx) for tx in pool]
            for pool in shared_pools
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_shared_pools(path: str) -> list[list[Transaction]]:
    """从磁盘恢复共享评估池。"""
    with open(path) as f:
        payload = json.load(f)
    raw_pools = payload["pools"] if isinstance(payload, dict) else payload
    return [
        [Transaction(**tx_data) for tx_data in pool]
        for pool in raw_pools
    ]


def evaluate_rl(model: ActorCritic, env: TxOrderingEnv,
                n_episodes: int, device: torch.device,
                shared_pools: list[list] | None = None) -> list[dict]:
    """运行训练好的 RL 模型, 收集指标"""
    results = []
    pools = shared_pools if shared_pools is not None else [None] * n_episodes
    for pool in pools:
        if pool is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset_with_pool(pool)
        while True:
            action, _, _ = model.act(obs, device, greedy=True)
            obs, _, done, _, info = env.step(action)
            if done:
                break
        selected = env.get_selected_transactions()
        pool = env.get_pool()
        m = compute_all_metrics(selected, pool)
        results.append(m)
    return results


def evaluate_baseline(method: str, env: TxOrderingEnv,
                      n_episodes: int,
                      shared_pools: list[list] | None = None) -> list[dict]:
    """运行基线方法, 收集指标"""
    results = []
    pools = shared_pools if shared_pools is not None else [None] * n_episodes
    for pool in pools:
        if pool is None:
            env.reset()
        else:
            env.reset_with_pool(pool)
        current_pool = env.get_pool()
        selected = run_baseline(deepcopy(current_pool), method)
        m = compute_all_metrics(selected, current_pool)
        results.append(m)
    return results


def aggregate(results: list[dict]) -> dict:
    """聚合指标: 计算均值和标准差"""
    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


def print_table(all_results: dict[str, dict]):
    """打印结果表"""
    print(f"{'方法':<20s} {'区块收益':>10s} {'公平性':>10s} "
          f"{'风险暴露':>10s} {'Gas利用':>10s} "
          f"{'风险排名':>10s} {'打包比':>10s}")
    print("-" * 90)
    for name, agg in all_results.items():
        print(f"{name:<20s} "
              f"{agg['block_fee_mean']:>8.1f}±{agg['block_fee_std']:<4.1f} "
              f"{agg['fairness_mean']:>8.4f} "
              f"{agg['risk_exposure_mean']:>8.4f} "
              f"{agg['gas_util_mean']:>8.4f} "
              f"{agg.get('risky_rank_mean', 0.5):>8.4f} "
              f"{agg.get('packing_ratio_mean', 0.0):>8.4f}")


def run_robustness(model, device, n_episodes, pool_size, seed,
                   output_dir="results"):
    """不同风险比例下的鲁棒性实验, 返回并保存 JSON"""
    risk_ratios = C.ROBUSTNESS_RISK_RATIOS
    all_data = {}
    print("\n===== 鲁棒性分析 =====")
    for rr in risk_ratios:
        env = TxOrderingEnv(pool_size=pool_size, risk_ratio=rr, seed=seed)
        shared_pools = build_shared_pools(n_episodes, pool_size, rr, seed)
        results = {}
        results["RL"] = aggregate(
            evaluate_rl(model, env, n_episodes, device, shared_pools))
        for bl in ["fifo", "gas", "heuristic", "fee_risk_linear", "fair_fee"]:
            results[bl] = aggregate(
                evaluate_baseline(bl, env, n_episodes, shared_pools))
        print(f"\n--- risk_ratio = {rr:.0%} ---")
        print_table(results)
        all_data[str(rr)] = results

    path = os.path.join(output_dir, "robustness_results.json")
    with open(path, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"鲁棒性结果已保存: {path}")
    return all_data


def run_pool_size_robustness(model, device, n_episodes, risk_ratio, seed,
                             output_dir="results"):
    """不同候选池规模下的鲁棒性实验"""
    pool_sizes = C.ROBUSTNESS_POOL_SIZES
    all_data = {}
    print("\n===== 候选池规模鲁棒性 =====")
    for ps in pool_sizes:
        env = TxOrderingEnv(pool_size=ps, risk_ratio=risk_ratio, seed=seed)
        shared_pools = build_shared_pools(n_episodes, ps, risk_ratio, seed)
        results = {}
        results["RL"] = aggregate(
            evaluate_rl(model, env, n_episodes, device, shared_pools))
        for bl in ["fifo", "gas", "heuristic", "fee_risk_linear", "fair_fee"]:
            results[bl] = aggregate(
                evaluate_baseline(bl, env, n_episodes, shared_pools))
        print(f"\n--- pool_size = {ps} ---")
        print_table(results)
        all_data[str(ps)] = results

    path = os.path.join(output_dir, "robustness_pool_size.json")
    with open(path, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"池大小鲁棒性结果已保存: {path}")
    return all_data


def run_fee_multiplier_robustness(model, device, n_episodes, pool_size, seed,
                                  output_dir="results"):
    """不同风险费率倍率下的鲁棒性实验"""
    multipliers = C.ROBUSTNESS_FEE_MULTIPLIERS
    all_data = {}
    print("\n===== 风险费率倍率鲁棒性 =====")
    for mult in multipliers:
        # 临时修改倍率, 通过自定义交易池实现
        env = TxOrderingEnv(pool_size=pool_size, risk_ratio=C.RISK_RATIO,
                            seed=seed)
        shared_pools = build_shared_pools(n_episodes, pool_size, C.RISK_RATIO,
                                          seed)
        results = {}
        eval_results_rl = []
        eval_results_bl = {bl: [] for bl in ["fifo", "gas", "heuristic",
                                              "fee_risk_linear", "fair_fee"]}
        for base_pool in shared_pools:
            pool = deepcopy(base_pool)
            # 修改风险交易费率倍率
            for tx in pool:
                if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD:
                    tx.fee = tx.fee / C.RISK_FEE_MULTIPLIER * mult
            obs, _ = env.reset_with_pool(pool)

            # RL 评估
            while True:
                action, _, _ = model.act(obs, device, greedy=True)
                obs, _, done, _, info = env.step(action)
                if done:
                    break
            sel_rl = env.get_selected_transactions()
            eval_results_rl.append(compute_all_metrics(sel_rl, pool))

            # 基线评估
            for bl in eval_results_bl:
                sel_bl = run_baseline(deepcopy(pool), bl)
                eval_results_bl[bl].append(compute_all_metrics(sel_bl, pool))

        results["RL"] = aggregate(eval_results_rl)
        for bl in eval_results_bl:
            results[bl] = aggregate(eval_results_bl[bl])

        print(f"\n--- fee_multiplier = {mult} ---")
        print_table(results)
        all_data[str(mult)] = results

    path = os.path.join(output_dir, "robustness_fee_mult.json")
    with open(path, "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"费率倍率鲁棒性结果已保存: {path}")
    return all_data


def plot_training_curve(log_path: str, save_path: str):
    """绘制训练收敛曲线 (3 子图: 奖励、损失、区块收益)"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装, 跳过绘图")
        return

    with open(log_path) as f:
        log = json.load(f)

    episodes = log["episode"]
    rewards = log["reward"]

    window = min(100, len(rewards) // 5) if len(rewards) > 10 else 1
    smooth = lambda v: np.convolve(v, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 子图1: 奖励曲线 (原始 + 平滑)
    axes[0].plot(episodes, rewards, alpha=0.2, linewidth=0.5, color="C0",
                 label="Raw")
    if window > 1:
        sm = smooth(rewards)
        axes[0].plot(episodes[:len(sm)], sm, linewidth=1.2, color="C0",
                     label=f"Smoothed (w={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Reward")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 子图2: 损失曲线
    if "actor_loss" in log:
        axes[1].plot(episodes, log["actor_loss"], label="Actor",
                     linewidth=0.6, alpha=0.7)
        axes[1].plot(episodes, log["critic_loss"], label="Critic",
                     linewidth=0.6, alpha=0.7)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    # 子图3: 区块收益曲线
    if "fee" in log:
        fees = log["fee"]
        axes[2].plot(episodes, fees, alpha=0.2, linewidth=0.5, color="C2")
        if window > 1:
            sm_fee = smooth(fees)
            axes[2].plot(episodes[:len(sm_fee)], sm_fee, linewidth=1.2,
                         color="C2")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Block Fee (Gwei)")
        axes[2].set_title("Block Fee Revenue")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="评估与对比")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("checkpoints", C.FORMAL_EVAL_CHECKPOINT_NAME),
        help=(
            "待评估模型路径。默认使用正式实验规则指定的 checkpoint "
            f"({C.FORMAL_EVAL_CHECKPOINT_NAME}; {C.FORMAL_EVAL_CHECKPOINT_RULE})."
        ),
    )
    parser.add_argument("--episodes", type=int, default=C.EVAL_EPISODES)
    parser.add_argument("--pool-size", type=C.validate_pool_size,
                        default=C.POOL_SIZE_DEFAULT)
    parser.add_argument("--risk-ratio", type=float, default=C.RISK_RATIO)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "cuda:0"],
                        help="评估设备: auto/cpu/cuda/cuda:0")
    parser.add_argument("--robustness", action="store_true",
                        help="运行鲁棒性实验")
    parser.add_argument("--plot", action="store_true",
                        help="绘制训练曲线")
    parser.add_argument("--train-log", type=str,
                        default="checkpoints/train_log.json")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    # 加载模型
    model = ActorCritic().to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device,
                                         weights_only=True))
        print(f"Loaded model: {args.model}")
    else:
        print(f"Warning: {args.model} not found, using random policy")
    model.eval()

    # 主对比实验
    env = TxOrderingEnv(pool_size=args.pool_size,
                        risk_ratio=args.risk_ratio, seed=args.seed)
    print("===== 主实验结果 =====")
    all_results = {}
    shared_pools = build_shared_pools(args.episodes, args.pool_size,
                                      args.risk_ratio, args.seed)
    all_results["RL (Ours)"] = aggregate(
        evaluate_rl(model, env, args.episodes, device, shared_pools))
    for bl in ["fifo", "gas", "heuristic", "fee_risk_linear", "fair_fee"]:
        label = bl.upper().replace("FEE_RISK_LINEAR", "FeeRiskLinear").replace("FAIR_FEE", "FairFee")
        all_results[label] = aggregate(
            evaluate_baseline(bl, env, args.episodes, shared_pools))
    print_table(all_results)

    # 保存结果
    with open(os.path.join(args.output, "main_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 鲁棒性实验
    if args.robustness:
        run_robustness(model, device, args.episodes,
                       args.pool_size, args.seed, args.output)

    # 绘制训练曲线
    if args.plot and os.path.exists(args.train_log):
        plot_training_curve(args.train_log,
                            os.path.join(args.output, "training_curve.png"))


if __name__ == "__main__":
    main()
