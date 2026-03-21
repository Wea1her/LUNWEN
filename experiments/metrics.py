"""评价指标: 区块收益、公平性指数、风险暴露度、Gas 利用率"""

from __future__ import annotations
import numpy as np
from typing import List

import config as C
from transaction import Transaction


def block_fee_revenue(selected: List[Transaction]) -> float:
    """区块手续费总收益 (Gwei)"""
    return sum(tx.fee for tx in selected)


def jain_fairness_index(selected: List[Transaction], t_now: float) -> float:
    """
    Jain 公平性指数: J = (Σw_i)² / (K · Σw_i²)
    w_i = 交易等待时间
    """
    if not selected:
        return 1.0
    waits = np.array([t_now - tx.arrival_time for tx in selected])
    waits = np.maximum(waits, 0)
    K = len(waits)
    sum_w = waits.sum()
    sum_w2 = (waits ** 2).sum()
    if sum_w2 < 1e-12:
        return 1.0
    return float((sum_w ** 2) / (K * sum_w2))


def risk_exposure(selected: List[Transaction],
                  ratio: float = C.RISK_POSITION_RATIO,
                  threshold: float = C.HEURISTIC_RISK_THRESHOLD) -> float:
    """高风险交易出现在头尾敏感位置的比例"""
    K = len(selected)
    if K == 0:
        return 0.0
    n_edge = max(int(K * ratio), 1)
    head = selected[:n_edge]
    tail = selected[-n_edge:]
    sensitive = head + tail

    risky_in_sensitive = sum(1 for tx in sensitive
                            if tx.risk_score >= threshold)
    total_risky = sum(1 for tx in selected if tx.risk_score >= threshold)
    if total_risky == 0:
        return 0.0
    return risky_in_sensitive / total_risky


def gas_utilization(selected: List[Transaction]) -> float:
    """Gas 利用率"""
    used = sum(tx.gas for tx in selected)
    return used / C.MAX_BLOCK_GAS


def compute_all_metrics(selected: List[Transaction],
                        pool: List[Transaction]) -> dict:
    """计算全部四项指标"""
    t_now = max(tx.arrival_time for tx in pool) + 1.0 if pool else 1.0
    return {
        "block_fee": block_fee_revenue(selected),
        "fairness": jain_fairness_index(selected, t_now),
        "risk_exposure": risk_exposure(selected),
        "gas_util": gas_utilization(selected),
    }
