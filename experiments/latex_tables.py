"""将聚合 JSON 结果转为论文 Table 2 / Table 3 的 LaTeX 行内容"""

import json

METHOD_ORDER = ["FIFO", "GAS", "HEURISTIC", "RL (Ours)"]
METHOD_LABELS = {
    "FIFO": "FIFO",
    "GAS": "Gas 优先",
    "HEURISTIC": "Heuristic",
    "RL (Ours)": "本文方法",
}

ROB_METHOD_ORDER = ["fifo", "gas", "heuristic", "RL"]
ROB_METHOD_LABELS = {
    "fifo": "FIFO",
    "gas": "Gas 优先",
    "heuristic": "Heuristic",
    "RL": "本文方法",
}


def _fmt(mean, std, precision=2):
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def generate_main_table(agg: dict) -> str:
    """生成 Table 2 数据行 (4 方法 × 4 指标)"""
    metrics = [
        ("block_fee", 1),
        ("fairness", 4),
        ("risk_exposure", 4),
        ("gas_util", 4),
    ]
    lines = []
    for m in METHOD_ORDER:
        d = agg[m]
        cells = [METHOD_LABELS[m]]
        for met, prec in metrics:
            cells.append(_fmt(d[f"{met}_mean"], d[f"{met}_std"], prec))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def generate_robustness_table(agg_rob: dict) -> str:
    """生成 Table 3 数据行 (4 方法 × 3 风险比例 × 2 指标)"""
    ratios = ["0.05", "0.15", "0.3"]
    lines = []
    for m in ROB_METHOD_ORDER:
        cells = [ROB_METHOD_LABELS[m]]
        for rr in ratios:
            d = agg_rob[rr][m]
            cells.append(_fmt(d["risk_exposure_mean"], d["risk_exposure_std"], 4))
        for rr in ratios:
            d = agg_rob[rr][m]
            cells.append(_fmt(d["block_fee_mean"], d["block_fee_std"], 1))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python latex_tables.py <results_dir>")
        sys.exit(1)
    d = sys.argv[1]
    with open(f"{d}/aggregated_main.json") as f:
        print("=== Table 2 ===")
        print(generate_main_table(json.load(f)))
    with open(f"{d}/aggregated_robustness.json") as f:
        print("=== Table 3 ===")
        print(generate_robustness_table(json.load(f)))
