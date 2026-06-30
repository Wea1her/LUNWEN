"""统计检验: seed 级 Welch t-test + episode 级配对检验"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional

import config as C
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
            "composite_score",
            "constrained_fee_score",
            "risk_adjusted_fee_score",
        ]

    normalized_main = [_normalize_method_results(m) for m in all_main]
    baselines = [k for k in normalized_main[0].keys() if k != ours_key]
    exploratory_only = len(normalized_main) < int(getattr(C, "EVIDENCE_FORMAL_MIN_SEEDS", 3))
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
                "significant": (tt["p_value"] < 0.05) and (not exploratory_only),
                "exploratory_only": exploratory_only,
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
            "composite_score",
            "constrained_fee_score",
            "risk_adjusted_fee_score",
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
    seed_set = {k[0] for k in paired_rows.keys() if k[0] is not None}
    exploratory_only = len(seed_set) < int(getattr(C, "EVIDENCE_FORMAL_MIN_SEEDS", 3))

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
                if metric == "constrained_fee_score":
                    # 仅在可行子集上比较，避免不可行占位值污染统计。
                    if float(ours_metric) < 0.0 or float(bl_metric) < 0.0:
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
                    "exploratory_only": exploratory_only,
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
                "significant_t": (tt["p_value"] < 0.05) and (not exploratory_only),
                "significant_wilcoxon": (wx["p_value"] < 0.05) and (not exploratory_only),
                "exploratory_only": exploratory_only,
            }

    return results


def bootstrap_mean_ci(vals: list[float], confidence: float = 0.95, n_boot: int = 10000, seed: int = 20260326) -> tuple:
    """均值差的 bootstrap 百分位置信区间。"""
    arr = np.array(vals, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = min((m - rank) * float(p_values[idx]), 1.0)
        running_max = max(running_max, raw)
        adjusted[idx] = running_max
    return adjusted


def run_seed_level_paired_tests(seed_level_statistics: dict, ours_key: str = "ours", metrics: Optional[list[str]] = None) -> dict:
    """以独立训练 seed 为单位做正式配对统计。"""
    ensure_scipy()
    seed_means = seed_level_statistics.get("seed_means", {})
    if metrics is None:
        metrics = list(seed_level_statistics.get("metrics") or [
            "block_fee",
            "fairness",
            "risk_exposure",
            "edge10_risk",
            "risky_inclusion_rate",
            "gas_util",
            "packing_ratio",
            "old_tx_pack_rate",
            "starvation_ratio",
            "composite_score",
            "risk_adjusted_fee_score",
        ])
    lower_is_better = {
        "risk_exposure",
        "edge10_risk",
        "top10_risk",
        "starvation_ratio",
        "starvation_gap",
        "selected_wait_std",
        "wait_p95",
        "wait_p99",
        "wait_gini",
        "invalid_action_count",
        "invalid_action_rate",
        "invalid_truncation_count",
        "invalid_truncation_rate",
        "max_invalid_streak",
        "mean_consecutive_invalid_actions",
        "mean_inference_time",
        "p95_inference_time",
        "max_inference_time",
    }

    methods_present = set()
    for payload in seed_means.values():
        methods_present.update(payload.keys())
    baselines = [m for m in MAIN_METHOD_ORDER if m != ours_key and m in methods_present]
    exploratory_only = not bool(seed_level_statistics.get("formal_statistics_ready", False))

    flat_keys = []
    flat_p = []
    results: dict[str, dict] = {}
    for bl in baselines:
        results[bl] = {}
        for metric in metrics:
            ours_vals = []
            bl_vals = []
            seeds_used = []
            for seed, payload in sorted(seed_means.items(), key=lambda kv: int(kv[0])):
                if ours_key not in payload or bl not in payload:
                    continue
                ours = payload[ours_key].get(metric)
                base = payload[bl].get(metric)
                if ours is None or base is None or np.isnan(float(ours)) or np.isnan(float(base)):
                    continue
                ours_vals.append(float(ours))
                bl_vals.append(float(base))
                seeds_used.append(int(seed))

            diff = (np.array(ours_vals) - np.array(bl_vals)).tolist()
            tt = paired_ttest(ours_vals, bl_vals) if len(diff) >= 2 else {"t_stat": 0.0, "p_value": 1.0}
            dz = paired_cohens_d(ours_vals, bl_vals)
            ci = bootstrap_mean_ci(diff, seed=20260326 + len(flat_keys)) if diff else (0.0, 0.0)
            higher_is_better = metric not in lower_is_better
            if diff:
                better = [(d > 0.0) if higher_is_better else (d < 0.0) for d in diff]
            else:
                better = []
            item = {
                "statistical_unit": "independent_training_seed",
                "seeds": seeds_used,
                "n_seed_pairs": len(diff),
                "ours_mean": float(np.mean(ours_vals)) if ours_vals else 0.0,
                "baseline_mean": float(np.mean(bl_vals)) if bl_vals else 0.0,
                "mean_diff": float(np.mean(diff)) if diff else 0.0,
                "paired_t": tt,
                "cohens_dz": dz,
                "bootstrap_ci_diff_95": ci,
                "ours_better_seed_rate": float(np.mean(better)) if better else 0.0,
                "higher_is_better": higher_is_better,
                "exploratory_only": exploratory_only,
            }
            results[bl][metric] = item
            flat_keys.append((bl, metric))
            flat_p.append(float(tt["p_value"]))

    adjusted = holm_adjust(flat_p)
    for (bl, metric), p_holm in zip(flat_keys, adjusted):
        item = results[bl][metric]
        item["p_value_holm"] = float(p_holm)
        item["significant_holm"] = (p_holm < 0.05) and (not item.get("exploratory_only", False))

    return {
        "protocol": "seed_level_paired_tests_with_bootstrap_ci_and_holm",
        "statistical_unit": "independent_training_seed",
        "episode_level_role": "diagnostic_only",
        "multiple_comparison_correction": "Holm-Bonferroni across all baseline-metric tests",
        "formal_statistics_ready": bool(seed_level_statistics.get("formal_statistics_ready", False)),
        "exploratory_only": exploratory_only,
        "metrics": metrics,
        "results": results,
    }


def format_seed_level_significance_table(sig_payload: dict) -> str:
    """格式化 seed 级正式统计报告。"""
    results = sig_payload.get("results", {})
    lines = []
    lines.append(
        f"{'Baseline':<28s} {'Metric':<28s} {'N':>3s} {'Diff':>10s} {'CI95':>23s} {'dz':>7s} {'pH':>8s}"
    )
    lines.append("-" * 112)
    for bl, metrics in results.items():
        for met, r in metrics.items():
            lo, hi = r.get("bootstrap_ci_diff_95", [0.0, 0.0])
            lines.append(
                f"{display_name(bl):<28s} {met:<28s} {r['n_seed_pairs']:>3d} "
                f"{r['mean_diff']:>10.4f} [{lo:>7.4f},{hi:>7.4f}] "
                f"{r['cohens_dz']:>7.2f} {r.get('p_value_holm', 1.0):>8.4f}"
            )
    return "\n".join(lines)


def generate_seed_level_significance_latex(sig_payload: dict, output_path: str):
    """生成 seed 级正式显著性检验 LaTeX 表格。"""
    lines = []
    for bl, metrics in sig_payload.get("results", {}).items():
        for met, r in metrics.items():
            sig_mark = _significance_mark(r.get("p_value_holm", 1.0), r.get("exploratory_only", False))
            sig_mark = "" if sig_mark == "ns" else f"^{{{sig_mark}}}"
            lo, hi = r.get("bootstrap_ci_diff_95", [0.0, 0.0])
            lines.append(
                f"{latex_escape(display_name(bl))} & {latex_escape(met)} & "
                f"${r['n_seed_pairs']}$ & ${r['mean_diff']:.4f}$ & "
                f"$[{lo:.4f}, {hi:.4f}]$ & ${r['cohens_dz']:.2f}$ & "
                f"${r.get('p_value_holm', 1.0):.4f}{sig_mark}$ \\\\"
            )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _significance_mark(p_value: float, exploratory_only: bool = False) -> str:
    if exploratory_only:
        return "ns"
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
            sig_mark = _significance_mark(r["p_value"], r.get("exploratory_only", False))
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
            sig_mark = _significance_mark(r["p_value"], r.get("exploratory_only", False))
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
            sig_t = _significance_mark(r["paired_t"]["p_value"], r.get("exploratory_only", False))
            sig_w = _significance_mark(r["wilcoxon"]["p_value"], r.get("exploratory_only", False))
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
        print("\nNote: episode-level paired tests are diagnostic; formal claims should use seed-level paired tests.")
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
