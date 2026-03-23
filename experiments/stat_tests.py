"""统计检验: seed 级 Welch t-test + episode 级配对检验"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional

from method_registry import MAIN_METHOD_ORDER, display_name, normalize_method_id

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False


def ensure_scipy():
    """确保统计检验依赖可用。"""
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for significance testing. "
            "Install it with `pip install -r experiments/requirements.txt`."
        )


def latex_escape(text: str) -> str:
    """最小化转义 LaTeX 表格中的普通文本。"""
    return text.replace("_", r"\_")


def _normalize_method_results(result: dict) -> dict:
    normalized = {}
    for method_key, metrics in result.items():
        try:
            method_id = normalize_method_id(method_key)
        except KeyError:
            method_id = method_key
        normalized[method_id] = metrics
    return normalized


def welch_ttest(vals_a: list[float], vals_b: list[float]) -> dict:
    """双样本 Welch t 检验 (不假设等方差)"""
    ensure_scipy()
    a, b = np.array(vals_a), np.array(vals_b)
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": 0.0, "p_value": 1.0}
    if a.std() < 1e-12 and b.std() < 1e-12:
        if abs(a.mean() - b.mean()) < 1e-12:
            return {"t_stat": 0.0, "p_value": 1.0}
        return {"t_stat": float("inf"), "p_value": 0.0}
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def paired_ttest(vals_a: list[float], vals_b: list[float]) -> dict:
    """配对 t 检验。"""
    ensure_scipy()
    a, b = np.array(vals_a), np.array(vals_b)
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": 0.0, "p_value": 1.0}
    diff = a - b
    if np.all(np.abs(diff) < 1e-12):
        return {"t_stat": 0.0, "p_value": 1.0}
    t_stat, p_value = stats.ttest_rel(a, b)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def wilcoxon_signed_rank(vals_a: list[float], vals_b: list[float]) -> dict:
    """Wilcoxon signed-rank 检验。"""
    ensure_scipy()
    a, b = np.array(vals_a), np.array(vals_b)
    if len(a) == 0 or len(b) == 0:
        return {"w_stat": 0.0, "p_value": 1.0}
    diff = a - b
    if np.all(np.abs(diff) < 1e-12):
        return {"w_stat": 0.0, "p_value": 1.0}
    w_stat, p_value = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"w_stat": float(w_stat), "p_value": float(p_value)}


def cohens_d(vals_a: list[float], vals_b: list[float]) -> float:
    """Cohen's d 效应量 (pooled std)"""
    a, b = np.array(vals_a), np.array(vals_b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def paired_cohens_d(vals_a: list[float], vals_b: list[float]) -> float:
    """配对 Cohen's d (dz)。"""
    a, b = np.array(vals_a), np.array(vals_b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    diff = a - b
    std = diff.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(diff.mean() / std)


def confidence_interval(vals: list[float], confidence: float = 0.95) -> tuple:
    """均值的 t 分布置信区间"""
    ensure_scipy()
    a = np.array(vals)
    n = len(a)
    mean = a.mean() if n else 0.0
    if n < 2:
        return (float(mean), float(mean))
    se = a.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return (float(mean - t_crit * se), float(mean + t_crit * se))


def mann_whitney_u(vals_a: list[float], vals_b: list[float]) -> dict:
    """Mann-Whitney U 检验 (非参数)"""
    ensure_scipy()
    u_stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {"u_stat": float(u_stat), "p_value": float(p_value)}


def run_significance_tests(
    all_main: list[dict], ours_key: str = "ours", metrics: Optional[list[str]] = None
) -> dict:
    """seed 级统计检验: 本文方法 vs 每个基线 (Welch t-test)。"""
    if metrics is None:
        metrics = [
            "block_fee",
            "fairness",
            "risk_exposure",
            "gas_util",
            "risky_rank",
            "packing_ratio",
            "top10_risk",
            "late_promo",
            "oldest_coverage",
            "starvation_gap",
            "tail_wait_reduction",
            "composite_score",
        ]

    normalized_main = [_normalize_method_results(m) for m in all_main]
    baselines = [k for k in normalized_main[0].keys() if k != ours_key]
    results = {}

    for bl in baselines:
        results[bl] = {}
        for met in metrics:
            ours_vals = [r[ours_key][f"{met}_mean"] for r in normalized_main]
            bl_vals = [r[bl][f"{met}_mean"] for r in normalized_main]

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


def _episode_pair_key(record: dict) -> tuple:
    setting_value = record.get("setting_value")
    if isinstance(setting_value, dict):
        setting_value = json.dumps(setting_value, ensure_ascii=False, sort_keys=True)
    return (
        record.get("seed"),
        record.get("setting", "main"),
        setting_value,
        record.get("shared_pool_id", record.get("episode_id")),
    )


def run_paired_significance_tests(
    episode_records: list[dict], ours_key: str = "ours", metrics: Optional[list[str]] = None
) -> dict:
    """episode 级配对检验: paired t-test + Wilcoxon signed-rank。"""
    ensure_scipy()
    if metrics is None:
        metrics = [
            "block_fee",
            "fairness",
            "risk_exposure",
            "gas_util",
            "risky_rank",
            "packing_ratio",
            "top10_risk",
            "late_promo",
            "oldest_coverage",
            "starvation_gap",
            "tail_wait_reduction",
            "composite_score",
        ]

    paired_rows: dict[tuple, dict] = {}
    for record in episode_records:
        if record.get("setting", "main") != "main":
            continue
        method_key = record.get("method")
        if method_key is None:
            continue
        try:
            method_id = normalize_method_id(method_key)
        except KeyError:
            continue
        row_key = _episode_pair_key(record)
        if row_key not in paired_rows:
            paired_rows[row_key] = {}
        paired_rows[row_key][method_id] = record.get("metrics", {})

    methods_present = set()
    for row in paired_rows.values():
        methods_present.update(row.keys())
    baselines = [m for m in MAIN_METHOD_ORDER if m != ours_key and m in methods_present]

    results: dict[str, dict] = {}
    for bl in baselines:
        results[bl] = {}
        for metric in metrics:
            ours_vals = []
            bl_vals = []
            for row in paired_rows.values():
                if ours_key not in row or bl not in row:
                    continue
                ours_metric = row[ours_key].get(metric)
                bl_metric = row[bl].get(metric)
                if ours_metric is None or bl_metric is None:
                    continue
                ours_vals.append(float(ours_metric))
                bl_vals.append(float(bl_metric))

            n_pairs = len(ours_vals)
            if n_pairs == 0:
                results[bl][metric] = {
                    "n_pairs": 0,
                    "paired_t": {"t_stat": 0.0, "p_value": 1.0},
                    "wilcoxon": {"w_stat": 0.0, "p_value": 1.0},
                    "cohens_dz": 0.0,
                    "ci_diff_95": [0.0, 0.0],
                    "ours_mean": 0.0,
                    "baseline_mean": 0.0,
                    "mean_diff": 0.0,
                    "significant_t": False,
                    "significant_wilcoxon": False,
                }
                continue

            tt = paired_ttest(ours_vals, bl_vals)
            wx = wilcoxon_signed_rank(ours_vals, bl_vals)
            diff = (np.array(ours_vals) - np.array(bl_vals)).tolist()
            ci_diff = confidence_interval(diff)
            dz = paired_cohens_d(ours_vals, bl_vals)

            results[bl][metric] = {
                "n_pairs": n_pairs,
                "paired_t": tt,
                "wilcoxon": wx,
                "cohens_dz": dz,
                "ci_diff_95": ci_diff,
                "ours_mean": float(np.mean(ours_vals)),
                "baseline_mean": float(np.mean(bl_vals)),
                "mean_diff": float(np.mean(diff)),
                "significant_t": tt["p_value"] < 0.05,
                "significant_wilcoxon": wx["p_value"] < 0.05,
            }

    return results


def _significance_mark(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_significance_table(sig_results: dict) -> str:
    """格式化 seed 级显著性报告。"""
    lines = []
    lines.append(f"{'Baseline':<22s} {'Metric':<16s} {'Ours':>8s} {'Base':>8s} {'d':>6s} {'p':>8s} {'Sig':>4s}")
    lines.append("-" * 88)
    for bl, metrics in sig_results.items():
        for met, r in metrics.items():
            sig_mark = _significance_mark(r["p_value"])
            lines.append(
                f"{display_name(bl):<22s} {met:<16s} {r['ours_mean']:>8.4f} "
                f"{r['baseline_mean']:>8.4f} {r['cohens_d']:>6.2f} {r['p_value']:>8.4f} {sig_mark:>4s}"
            )
    return "\n".join(lines)


def format_paired_significance_table(sig_results: dict) -> str:
    """格式化 episode 级配对显著性报告。"""
    lines = []
    lines.append(
        f"{'Baseline':<22s} {'Metric':<16s} {'N':>6s} {'Ours':>8s} {'Base':>8s} "
        f"{'Diff':>8s} {'p_t':>8s} {'p_w':>8s}"
    )
    lines.append("-" * 98)
    for bl, metrics in sig_results.items():
        for met, r in metrics.items():
            lines.append(
                f"{display_name(bl):<22s} {met:<16s} {r['n_pairs']:>6d} {r['ours_mean']:>8.4f} "
                f"{r['baseline_mean']:>8.4f} {r['mean_diff']:>8.4f} "
                f"{r['paired_t']['p_value']:>8.4f} {r['wilcoxon']['p_value']:>8.4f}"
            )
    return "\n".join(lines)


def generate_significance_latex(sig_results: dict, output_path: str):
    """生成 seed 级显著性检验 LaTeX 表格。"""
    lines = []
    for bl, metrics in sig_results.items():
        for met, r in metrics.items():
            sig_mark = _significance_mark(r["p_value"])
            sig_mark = "" if sig_mark == "ns" else f"^{{{sig_mark}}}"
            lines.append(
                f"{latex_escape(display_name(bl))} & {latex_escape(met)} & "
                f"${r['ours_mean']:.4f}$ & ${r['baseline_mean']:.4f}$ & "
                f"${r['cohens_d']:.2f}$ & "
                f"${r['p_value']:.4f}{sig_mark}$ \\\\"
            )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_paired_significance_latex(sig_results: dict, output_path: str):
    """生成 episode 级配对显著性 LaTeX 表格。"""
    lines = []
    for bl, metrics in sig_results.items():
        for met, r in metrics.items():
            sig_t = _significance_mark(r["paired_t"]["p_value"])
            sig_w = _significance_mark(r["wilcoxon"]["p_value"])
            sig_t = "" if sig_t == "ns" else f"^{{{sig_t}}}"
            sig_w = "" if sig_w == "ns" else f"^{{{sig_w}}}"
            lines.append(
                f"{latex_escape(display_name(bl))} & {latex_escape(met)} & "
                f"${r['n_pairs']}$ & "
                f"${r['ours_mean']:.4f}$ & ${r['baseline_mean']:.4f}$ & "
                f"${r['mean_diff']:.4f}$ & "
                f"${r['paired_t']['p_value']:.4f}{sig_t}$ & "
                f"${r['wilcoxon']['p_value']:.4f}{sig_w}$ \\\\"
            )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys
    import os
    import glob

    if len(sys.argv) < 2:
        print("Usage: python stat_tests.py <results_dir>")
        print("  reads seed_*/results/main_episode_metrics.json and runs paired significance tests")
        sys.exit(1)

    results_dir = sys.argv[1]
    seed_dirs = sorted(glob.glob(os.path.join(results_dir, "seed_*")))
    all_main = []
    all_episode_rows = []

    for sd in seed_dirs:
        main_path = os.path.join(sd, "results", "main_results.json")
        if os.path.exists(main_path):
            with open(main_path) as f:
                all_main.append(json.load(f))

        episode_path = os.path.join(sd, "results", "main_episode_metrics.json")
        if os.path.exists(episode_path):
            with open(episode_path) as f:
                payload = json.load(f)
            seed = payload.get("seed")
            for row in payload.get("records", []):
                if "seed" not in row:
                    row["seed"] = seed
                all_episode_rows.append(row)

    if all_episode_rows:
        sig = run_paired_significance_tests(all_episode_rows)
        print(format_paired_significance_table(sig))
        sig_path = os.path.join(results_dir, "paired_significance_tests.json")
        with open(sig_path, "w") as f:
            json.dump(sig, f, indent=2, ensure_ascii=False)
        print(f"\n配对显著性检验结果已保存: {sig_path}")

        tex_path = os.path.join(results_dir, "table_significance_paired.tex")
        generate_paired_significance_latex(sig, tex_path)
        print(f"LaTeX 表格已保存: {tex_path}")
    else:
        if len(all_main) < 2:
            print(f"需要至少 2 个 seed 的结果, 当前只有 {len(all_main)} 个")
            sys.exit(1)
        sig = run_significance_tests(all_main)
        print(format_significance_table(sig))
        sig_path = os.path.join(results_dir, "significance_tests.json")
        with open(sig_path, "w") as f:
            json.dump(sig, f, indent=2, ensure_ascii=False)
        print(f"\n显著性检验结果已保存: {sig_path}")
        tex_path = os.path.join(results_dir, "table_significance.tex")
        generate_significance_latex(sig, tex_path)
        print(f"LaTeX 表格已保存: {tex_path}")
