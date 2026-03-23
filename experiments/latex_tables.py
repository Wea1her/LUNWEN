"""将聚合 JSON 结果转为论文 LaTeX 表格行内容"""

import json

from method_registry import MAIN_METHOD_ORDER, latex_name, normalize_method_id

ABLATION_ORDER = ["Ours-FeeOnly", "Ours-Fee+Fair", "Ours-Fee+Risk", "Ours-Full"]
ABLATION_LABELS = {
    "Ours-FeeOnly": "$r=\\alpha r_{\\text{fee}}$",
    "Ours-Fee+Fair": "$r=\\alpha r_{\\text{fee}}+\\beta r_{\\text{fair}}$",
    "Ours-Fee+Risk": "$r=\\alpha r_{\\text{fee}}-\\gamma r_{\\text{risk}}$",
    "Ours-Full": "完整奖励（本文）",
}

STRUCT_ABLATION_ORDER = ["No-SeqSummary", "No-ActionMask", "No-STOP", "Ours-Full"]
STRUCT_ABLATION_LABELS = {
    "No-SeqSummary": "去除序列摘要",
    "No-ActionMask": "去除动作掩码",
    "No-STOP": "去除 STOP 动作",
    "Ours-Full": "完整模型（本文）",
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
