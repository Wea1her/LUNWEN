"""统计检验脚手架: Welch t-test, 效应量 Cohen's d, 置信区间"""

import json
import numpy as np
from scipy import stats
from typing import Optional


def welch_ttest(vals_a: list[float], vals_b: list[float]) -> dict:
    """双样本 Welch t 检验 (不假设等方差)"""
    a, b = np.array(vals_a), np.array(vals_b)
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": 0.0, "p_value": 1.0}
    if a.std() < 1e-12 and b.std() < 1e-12:
        # 两组均为常数: 若均值相同则不显著, 否则极显著
        if abs(a.mean() - b.mean()) < 1e-12:
            return {"t_stat": 0.0, "p_value": 1.0}
        return {"t_stat": float("inf"), "p_value": 0.0}
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def cohens_d(vals_a: list[float], vals_b: list[float]) -> float:
    """Cohen's d 效应量 (pooled std)"""
    a, b = np.array(vals_a), np.array(vals_b)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 +
                          (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def confidence_interval(vals: list[float], confidence: float = 0.95) -> tuple:
    """均值的 t 分布置信区间"""
    a = np.array(vals)
    n = len(a)
    mean = a.mean()
    se = a.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return (float(mean - t_crit * se), float(mean + t_crit * se))


def mann_whitney_u(vals_a: list[float], vals_b: list[float]) -> dict:
    """Mann-Whitney U 检验 (非参数)"""
    u_stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {"u_stat": float(u_stat), "p_value": float(p_value)}


def run_significance_tests(all_main: list[dict],
                           ours_key: str = "RL (Ours)",
                           metrics: Optional[list[str]] = None) -> dict:
    """对主实验结果跑统计检验: 本文方法 vs 每个基线

    Args:
        all_main: 每个 seed 的 main_results dict 列表
        ours_key: 本文方法在结果中的 key
        metrics: 要检验的指标列表

    Returns:
        {baseline: {metric: {t_stat, p_value, cohens_d, ci_ours, ci_baseline, significant}}}
    """
    if metrics is None:
        metrics = ["block_fee", "fairness", "risk_exposure", "gas_util",
                   "risky_rank", "packing_ratio", "top10_risk", "late_promo"]

    baselines = [k for k in all_main[0].keys() if k != ours_key]
    results = {}

    for bl in baselines:
        results[bl] = {}
        for met in metrics:
            ours_vals = [r[ours_key][f"{met}_mean"] for r in all_main]
            bl_vals = [r[bl][f"{met}_mean"] for r in all_main]

            tt = welch_ttest(ours_vals, bl_vals)
            d = cohens_d(ours_vals, bl_vals)
            ci_ours = confidence_interval(ours_vals)
            ci_bl = confidence_interval(bl_vals)

            results[bl][met] = {
                "t_stat": tt["t_stat"],
                "p_value": tt["p_value"],
                "cohens_d": d,
                "ci_ours_95": ci_ours,
                "ci_baseline_95": ci_bl,
                "ours_mean": float(np.mean(ours_vals)),
                "baseline_mean": float(np.mean(bl_vals)),
                "significant": tt["p_value"] < 0.05,
            }

    return results


def format_significance_table(sig_results: dict) -> str:
    """格式化为可读的显著性报告"""
    lines = []
    lines.append(f"{'Baseline':<20s} {'Metric':<16s} {'Ours':>8s} {'Base':>8s} "
                 f"{'d':>6s} {'p':>8s} {'Sig':>4s}")
    lines.append("-" * 80)
    for bl, metrics in sig_results.items():
        for met, r in metrics.items():
            sig_mark = "***" if r["p_value"] < 0.001 else \
                       "**" if r["p_value"] < 0.01 else \
                       "*" if r["p_value"] < 0.05 else "ns"
            lines.append(
                f"{bl:<20s} {met:<16s} {r['ours_mean']:>8.4f} "
                f"{r['baseline_mean']:>8.4f} {r['cohens_d']:>6.2f} "
                f"{r['p_value']:>8.4f} {sig_mark:>4s}")
    return "\n".join(lines)


def generate_significance_latex(sig_results: dict, output_path: str):
    """生成显著性检验的 LaTeX 表格"""
    lines = []
    for bl, metrics in sig_results.items():
        for met, r in metrics.items():
            sig_mark = "^{***}" if r["p_value"] < 0.001 else \
                       "^{**}" if r["p_value"] < 0.01 else \
                       "^{*}" if r["p_value"] < 0.05 else ""
            lines.append(
                f"{bl} & {met} & "
                f"${r['ours_mean']:.4f}$ & ${r['baseline_mean']:.4f}$ & "
                f"${r['cohens_d']:.2f}$ & "
                f"${r['p_value']:.4f}{sig_mark}$ \\\\")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stat_tests.py <results_dir>")
        print("  reads seed_*/results/main_results.json, runs significance tests")
        sys.exit(1)

    results_dir = sys.argv[1]
    import os
    import glob

    # 加载各 seed 的主实验结果
    seed_dirs = sorted(glob.glob(os.path.join(results_dir, "seed_*")))
    all_main = []
    for sd in seed_dirs:
        path = os.path.join(sd, "results", "main_results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_main.append(json.load(f))

    if len(all_main) < 2:
        print(f"需要至少 2 个 seed 的结果, 当前只有 {len(all_main)} 个")
        sys.exit(1)

    print(f"加载了 {len(all_main)} 个 seed 的结果\n")

    sig = run_significance_tests(all_main)
    print(format_significance_table(sig))

    # 保存 JSON
    sig_path = os.path.join(results_dir, "significance_tests.json")
    with open(sig_path, "w") as f:
        json.dump(sig, f, indent=2, ensure_ascii=False)
    print(f"\n显著性检验结果已保存: {sig_path}")

    # 保存 LaTeX
    tex_path = os.path.join(results_dir, "table_significance.tex")
    generate_significance_latex(sig, tex_path)
    print(f"LaTeX 表格已保存: {tex_path}")
