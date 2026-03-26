"""将聚合 JSON 结果转为论文 LaTeX 表格行内容"""

import json

from method_registry import MAIN_METHOD_ORDER, latex_name, normalize_method_id

ABLATION_ORDER = ["Ours-AgeOnly", "Ours-Age+Risk", "Ours-Age+TerminalFair", "Ours-FullBalanced"]
ABLATION_LABELS = {
    "Ours-AgeOnly": "$r=\\alpha r_{\\text{fee}}+\\beta_{age}r_{age}$",
    "Ours-Age+Risk": "$r=\\alpha r_{\\text{fee}}+\\beta_{age}r_{age}-\\gamma r_{risk}$",
    "Ours-Age+TerminalFair": "$r=\\alpha r_{\\text{fee}}+\\beta_{age}r_{age}+\\beta_T r_{fair}^{terminal}$",
    "Ours-FullBalanced": "完整奖励（Age+Oldest+Risk+Terminal）",
}

STRUCT_ABLATION_ORDER = ["No-SeqSummary", "No-ActionMask", "No-STOP", "Ours-Full"]
STRUCT_ABLATION_LABELS = {
    "No-SeqSummary": "去除序列摘要",
    "No-ActionMask": "去除动作掩码",
    "No-STOP": "去除 STOP 动作",
    "Ours-Full": "完整模型（本文）",
}

PROTOCOL_ABLATION_ORDER = [
    "Proto-Composite-NoWarm-NoCurr",
    "Proto-Constrained-NoWarm-NoCurr",
    "Proto-Constrained-Warm-Curr",
    "Proto-Constrained-NoFairGate",
    "Proto-Constrained-NoTerminalRisk",
    "Proto-Hypervolume-Warm-Curr",
]
PROTOCOL_ABLATION_LABELS = {
    "Proto-Composite-NoWarm-NoCurr": "Composite选模 + 无Warm/Curr",
    "Proto-Constrained-NoWarm-NoCurr": "Constrained选模 + 无Warm/Curr",
    "Proto-Constrained-Warm-Curr": "Constrained选模 + Warm/Curr",
    "Proto-Constrained-NoFairGate": "Constrained选模 + 去FairnessGate",
    "Proto-Constrained-NoTerminalRisk": "Constrained选模 + 去TerminalRisk",
    "Proto-Hypervolume-Warm-Curr": "Hypervolume选模 + Warm/Curr",
}


def _fmt(mean, std, precision=2):
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def _normalize_method_keyed_dict(data: dict) -> dict:
    normalized = {}
    for key, value in data.items():
        try:
            method_id = normalize_method_id(key)
        except KeyError:
            method_id = key
        normalized[method_id] = value
    return normalized


def _highlight_ranked(raw_values, rendered_values, higher_better=True):
    """按数值排序后，对最优值加粗、次优值下划线。"""
    indexed = [(v, i) for i, v in enumerate(raw_values)]
    indexed.sort(key=lambda x: x[0], reverse=higher_better)
    result = list(rendered_values)
    for rank, (_, i) in enumerate(indexed):
        if rank == 0:
            result[i] = f"\\textbf{{{rendered_values[i]}}}"
        elif rank == 1:
            result[i] = f"\\underline{{{rendered_values[i]}}}"
    return result


def generate_main_table(agg: dict) -> str:
    """生成主实验表: 6 方法 × 6 指标, 最优加粗, 次优下划线"""
    agg = _normalize_method_keyed_dict(agg)
    metrics = [
        ("block_fee", 1, True),
        ("fairness", 4, True),
        ("risk_exposure", 4, False),
        ("gas_util", 4, True),
        ("risky_rank", 4, None),  # 接近 0.5 最好, 特殊处理
        ("packing_ratio", 4, True),
    ]
    return _generate_metric_table(agg, metrics)


def _generate_metric_table(agg: dict, metrics: list[tuple[str, int, bool | None]]) -> str:
    """通用指标表生成器。"""
    agg = _normalize_method_keyed_dict(agg)

    present = [m for m in MAIN_METHOD_ORDER if m in agg]

    # 收集每个指标的格式化值
    col_data = []
    for met, prec, higher in metrics:
        raw_strs = []
        raw_means = []
        for m in present:
            d = agg[m]
            mean_val = d.get(f"{met}_mean", 0.0)
            std_val = d.get(f"{met}_std", 0.0)
            raw_strs.append(_fmt(mean_val, std_val, prec))
            raw_means.append(mean_val)
        if higher is not None:
            bolded = _highlight_ranked(raw_means, raw_strs, higher)
        else:
            # risky_rank: 接近 0.5 最好
            dists = [abs(v - 0.5) for v in raw_means]
            idx_sorted = sorted(range(len(dists)), key=lambda i: dists[i])
            bolded = list(raw_strs)
            bolded[idx_sorted[0]] = f"\\textbf{{{raw_strs[idx_sorted[0]]}}}"
            if len(idx_sorted) > 1:
                bolded[idx_sorted[1]] = f"\\underline{{{raw_strs[idx_sorted[1]]}}}"
        col_data.append(bolded)

    lines = []
    for row_idx, m in enumerate(present):
        cells = [latex_name(m)]
        for col in col_data:
            cells.append(col[row_idx])
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def generate_main_core_table(agg: dict) -> str:
    """主叙事核心表（收益-公平-风险-约束）。"""
    metrics = [
        ("block_fee", 1, True),
        ("fairness", 4, True),
        ("risk_exposure", 4, False),
        ("top10_risk", 4, False),
        ("gas_util", 4, True),
        ("packing_ratio", 4, True),
        ("composite_score", 4, True),
        ("constrained_fee_score", 4, True),
    ]
    return _generate_metric_table(agg, metrics)


def generate_main_fullmetrics_table(agg: dict) -> str:
    """全指标主表（防止选择性呈现）。"""
    metrics = [
        ("block_fee", 1, True),
        ("fairness", 4, True),
        ("risk_exposure", 4, False),
        ("edge_risk_ratio", 4, False),
        ("top10_risk", 4, False),
        ("risky_rank", 4, None),
        ("gas_util", 4, True),
        ("packing_ratio", 4, True),
        ("late_promo", 4, True),
        ("oldest_coverage", 4, True),
        ("starvation_gap", 4, False),
        ("tail_wait_reduction", 4, True),
        ("selected_wait_std", 4, False),
        ("wait_p95", 4, False),
        ("wait_p99", 4, False),
        ("wait_gini", 4, False),
        ("composite_score", 4, True),
        ("constrained_fee_score", 4, True),
        ("risk_adjusted_fee_score", 4, True),
    ]
    return _generate_metric_table(agg, metrics)


def generate_robustness_table(agg_rob: dict) -> str:
    """生成鲁棒性表"""
    ratios = sorted(agg_rob.keys(), key=float)
    normalized = {ratio: _normalize_method_keyed_dict(agg_rob[ratio]) for ratio in ratios}
    present = [m for m in MAIN_METHOD_ORDER if m in normalized[ratios[0]]]
    lines = []
    for m in present:
        cells = [latex_name(m)]
        for rr in ratios:
            d = normalized[rr][m]
            cells.append(_fmt(d["block_fee_mean"], d["block_fee_std"], 1))
        for rr in ratios:
            d = normalized[rr][m]
            cells.append(_fmt(d["risk_exposure_mean"], d["risk_exposure_std"], 4))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def generate_ablation_table(agg: dict, order=None, labels=None) -> str:
    """生成消融实验表"""
    if order is None:
        order = ABLATION_ORDER
    if labels is None:
        labels = ABLATION_LABELS
    metrics = [
        ("block_fee", 1),
        ("fairness", 4),
        ("risk_exposure", 4),
        ("gas_util", 4),
    ]
    present = [m for m in order if m in agg]
    lines = []
    for m in present:
        d = agg[m]
        cells = [labels[m]]
        for met, prec in metrics:
            cells.append(_fmt(d[f"{met}_mean"], d[f"{met}_std"], prec))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def generate_composite_table(agg: dict) -> str:
    """生成综合目标主表: method × composite_score。"""
    agg = _normalize_method_keyed_dict(agg)
    present = [m for m in MAIN_METHOD_ORDER if m in agg and "composite_score_mean" in agg[m]]
    if not present:
        return ""
    raw = [agg[m]["composite_score_mean"] for m in present]
    rendered = [_fmt(agg[m]["composite_score_mean"], agg[m]["composite_score_std"], 4) for m in present]
    highlighted = _highlight_ranked(raw, rendered, higher_better=True)
    lines = []
    for idx, method_id in enumerate(present):
        lines.append(f"{latex_name(method_id)} & {highlighted[idx]} \\\\")
    return "\n".join(lines) + "\n"


def generate_fairness_decomp_table(payload: dict) -> str:
    """生成 fairness 分解表。"""
    methods = payload.get("methods", {})
    metrics = ["fairness", "oldest_coverage", "starvation_gap", "tail_wait_reduction", "selected_wait_std"]
    present = [m for m in MAIN_METHOD_ORDER if m in methods]
    lines = []
    for method_id in present:
        values = methods[method_id]
        cells = [latex_name(method_id)]
        for metric in metrics:
            cells.append(_fmt(values.get(f"{metric}_mean", 0.0), values.get(f"{metric}_std", 0.0), 4))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def generate_constrained_main_table(payload: dict) -> str:
    """生成约束式主表。"""
    methods = payload.get("methods", {})
    present = [m for m in MAIN_METHOD_ORDER if m in methods]
    lines = []
    for method_id in present:
        v = methods[method_id]
        feas = f"${v.get('feasible_rate', 0.0):.4f}$"
        cmean_raw = v.get("feasible_fee_mean", v.get("constrained_fee_mean"))
        cstd_raw = v.get("feasible_fee_std", v.get("constrained_fee_std"))
        cmean = "--" if cmean_raw is None else f"${float(cmean_raw):.4f}$"
        cstd = "--" if cstd_raw is None else f"${float(cstd_raw):.4f}$"
        infeasible = int(v.get("infeasible_count", 0))
        violation_count = int(v.get("violation_count", infeasible))
        top1 = str(v.get("constraint_violation_top1", "none")).replace("_", r"\_")
        lines.append(
            f"{latex_name(method_id)} & {feas} & {cmean} & {cstd} & ${infeasible}$ & ${violation_count}$ & {top1} \\\\"
        )
    return "\n".join(lines) + "\n"


def generate_operating_points_table(payload: dict) -> str:
    """生成三档 operating points 对比表。"""
    modes = payload.get("modes", {})
    mode_order = ["aggressive", "balanced", "conservative"]
    lines = []
    for mode in mode_order:
        row = modes.get(mode, {})
        top_method = row.get("top_method")
        metric = row.get("top_method_metrics", {})
        if top_method is None:
            lines.append(f"{mode} & -- & -- & -- & -- \\\\")
            continue
        top_method_latex = latex_name(top_method) if top_method in MAIN_METHOD_ORDER else top_method
        feas = metric.get("feasible_rate", 0.0)
        fee = metric.get("feasible_set_fee_mean")
        risk_adj = metric.get("risk_adjusted_fee_mean")
        fee_str = "--" if fee is None else f"${float(fee):.4f}$"
        risk_str = "--" if risk_adj is None else f"${float(risk_adj):.4f}$"
        lines.append(
            f"{mode} & {top_method_latex} & ${float(feas):.4f}$ & {fee_str} & {risk_str} \\\\"
        )
    return "\n".join(lines) + "\n"


def generate_constraint_bottleneck_table(payload: dict) -> str:
    """生成各方法约束瓶颈表。"""
    methods = payload.get("methods", {})
    present = [m for m in MAIN_METHOD_ORDER if m in methods]
    lines = []
    for method_id in present:
        item = methods[method_id]
        feas = float(item.get("feasible_rate", 0.0))
        violation_count = int(item.get("violation_count", 0))
        infeasible_count = int(item.get("infeasible_count", 0))
        top1 = str(item.get("constraint_violation_top1", "none")).replace("_", r"\_")
        lines.append(
            f"{latex_name(method_id)} & ${feas:.4f}$ & ${violation_count}$ & ${infeasible_count}$ & {top1} \\\\"
        )
    return "\n".join(lines) + "\n"


def append_primary_decision_rule_note(table_rows: str) -> str:
    """在表格末尾追加主决策规则说明。"""
    return table_rows + "% NOTE: Primary decision rule: feasible-rate first, feasible-set fee second.\n"


def append_exploratory_note(table_rows: str, evidence_level: str) -> str:
    """在 dryrun/exploratory 情况下追加表注。"""
    if evidence_level == "formal_multi_seed":
        return table_rows
    return table_rows + f"% NOTE: This table is exploratory ({evidence_level}).\n"


def generate_pareto_main_table(dominance_matrix: dict, anchor_method: str = "ours") -> str:
    """生成 anchor 方法（默认 Ours）相对各基线的 Pareto 表。"""
    if anchor_method not in dominance_matrix:
        return ""
    lines = []
    for method_id in MAIN_METHOD_ORDER:
        if method_id == anchor_method:
            continue
        rel = dominance_matrix.get(anchor_method, {}).get(method_id)
        if not rel:
            continue
        dom = f"${rel.get('ours_dominates_rate', 0.0):.4f}$"
        back = f"${rel.get('baseline_dominates_rate', 0.0):.4f}$"
        non = f"${rel.get('non_dominated_rate', 0.0):.4f}$"
        n_pairs = rel.get("n_pairs", 0)
        lines.append(f"{latex_name(anchor_method)} vs {latex_name(method_id)} & {dom} & {back} & {non} & ${n_pairs}$ \\\\")
    return "\n".join(lines) + "\n"


def generate_protocol_ablation_table(agg: dict, order=None, labels=None) -> str:
    """生成协议消融表。"""
    if order is None:
        order = PROTOCOL_ABLATION_ORDER
    if labels is None:
        labels = PROTOCOL_ABLATION_LABELS
    metrics = [
        ("block_fee", 1),
        ("fairness", 4),
        ("risk_exposure", 4),
        ("top10_risk", 4),
        ("constrained_fee_score", 4),
        ("risk_adjusted_fee_score", 4),
        ("composite_score", 4),
    ]
    present = [m for m in order if m in agg]
    lines = []
    for m in present:
        d = agg[m]
        cells = [labels.get(m, m)]
        for met, prec in metrics:
            cells.append(_fmt(d.get(f"{met}_mean", 0.0), d.get(f"{met}_std", 0.0), prec))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 2:
        print("Usage: python latex_tables.py <results_dir>")
        sys.exit(1)
    d = sys.argv[1]
    with open(f"{d}/aggregated_main.json") as f:
        print("=== Main Table ===")
        print(generate_main_table(json.load(f)))
    robustness_files = [
        ("Risk Robustness Table", "aggregated_robustness_risk.json"),
        ("Pool Robustness Table", "aggregated_robustness_pool.json"),
        ("Fee Robustness Table", "aggregated_robustness_fee.json"),
        ("Robustness Table", "aggregated_robustness.json"),
    ]
    for title, filename in robustness_files:
        path = f"{d}/{filename}"
        if os.path.exists(path):
            with open(path) as f:
                print(f"=== {title} ===")
                print(generate_robustness_table(json.load(f)))
