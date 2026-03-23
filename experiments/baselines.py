"""基线排序方法: FIFO, Gas 优先, 启发式风险感知与中心感知贪心"""

from __future__ import annotations
from typing import List

import config as C
from transaction import Transaction


def _apply_nonce_order(txs: List[Transaction]) -> List[Transaction]:
    """保证同地址交易按 nonce 顺序, 在原有排序基础上做拓扑修正"""
    selected: List[Transaction] = []
    remaining = list(txs)
    selected_nonces: dict[int, int] = {}  # sender -> max selected nonce
    gas_left = C.MAX_BLOCK_GAS

    while remaining:
        progress = False
        next_remaining = []
        for tx in remaining:
            # 容量检查
            if tx.gas > gas_left:
                next_remaining.append(tx)
                continue
            # Nonce 依赖检查
            if tx.nonce > 0:
                prev = selected_nonces.get(tx.sender, -1)
                if prev < tx.nonce - 1:
                    next_remaining.append(tx)
                    continue
            # 可选
            selected.append(tx)
            gas_left -= tx.gas
            selected_nonces[tx.sender] = max(
                selected_nonces.get(tx.sender, -1), tx.nonce)
            progress = True
        remaining = next_remaining
        if not progress:
            break
    return selected


def fifo_sort(pool: List[Transaction]) -> List[Transaction]:
    """按到达时间排序"""
    ordered = sorted(pool, key=lambda tx: tx.arrival_time)
    return _apply_nonce_order(ordered)


def gas_priority_sort(pool: List[Transaction]) -> List[Transaction]:
    """按手续费从高到低排序"""
    ordered = sorted(pool, key=lambda tx: -tx.fee)
    return _apply_nonce_order(ordered)


def heuristic_sort(pool: List[Transaction],
                   risk_threshold: float = C.HEURISTIC_RISK_THRESHOLD
                   ) -> List[Transaction]:
    """Gas 优先 + 高风险交易降级到区块中间位置"""
    normal = [tx for tx in pool if tx.risk_score < risk_threshold]
    risky = [tx for tx in pool if tx.risk_score >= risk_threshold]

    normal_sorted = sorted(normal, key=lambda tx: -tx.fee)
    risky_sorted = sorted(risky, key=lambda tx: -tx.fee)

    # 将风险交易插入到中间位置
    result = list(normal_sorted)
    mid = len(result) // 2
    for tx in risky_sorted:
        result.insert(mid, tx)
        mid += 1

    return _apply_nonce_order(result)


def fee_risk_linear_sort(pool: List[Transaction],
                         lambda_f: float = C.LINEAR_LAMBDA_F,
                         lambda_r: float = C.LINEAR_LAMBDA_R
                         ) -> List[Transaction]:
    """score(tx) = λ_f · fee_norm − λ_r · risk_score, 按分数降序"""
    if not pool:
        return []
    max_fee = max(tx.fee for tx in pool)
    ordered = sorted(pool,
                     key=lambda tx: -(lambda_f * (tx.fee / max_fee)
                                      - lambda_r * tx.risk_score))
    return _apply_nonce_order(ordered)


def fair_fee_sort(pool: List[Transaction],
                  lambda_f: float = C.FAIR_LAMBDA_F,
                  lambda_w: float = C.FAIR_LAMBDA_W
                  ) -> List[Transaction]:
    """score(tx) = λ_f · fee_norm + λ_w · wait_norm, 按分数降序"""
    if not pool:
        return []
    max_fee = max(tx.fee for tx in pool)
    t_now = max(tx.arrival_time for tx in pool) + 1.0
    t_max = max(tx.arrival_time for tx in pool)
    if t_max < 1e-8:
        t_max = 1.0
    ordered = sorted(pool,
                     key=lambda tx: -(lambda_f * (tx.fee / max_fee)
                                      + lambda_w * ((t_now - tx.arrival_time) / t_max)))
    return _apply_nonce_order(ordered)


def center_aware_greedy_sort(
    pool: List[Transaction],
    lambda_f: float = C.CENTER_AWARE_LAMBDA_F,
    lambda_w: float = C.CENTER_AWARE_LAMBDA_W,
    late_promo: float = C.CENTER_AWARE_LATE_PROMO,
    risk_threshold: float = C.HEURISTIC_RISK_THRESHOLD,
) -> List[Transaction]:
    """收益驱动 + 迟到高费提升 + 风险交易居中放置。"""
    if not pool:
        return []
    max_fee = max(tx.fee for tx in pool)
    t_now = max(tx.arrival_time for tx in pool) + 1.0
    t_max = max(tx.arrival_time for tx in pool)
    if t_max < 1e-8:
        t_max = 1.0
    median_arrival = float(sorted(tx.arrival_time for tx in pool)[len(pool) // 2])
    median_fee = float(sorted(tx.fee for tx in pool)[len(pool) // 2])

    scored = []
    for tx in pool:
        fee_norm = tx.fee / max_fee
        wait_norm = (t_now - tx.arrival_time) / t_max
        late_high = 1.0 if (tx.arrival_time > median_arrival and tx.fee > median_fee) else 0.0
        score = lambda_f * fee_norm + lambda_w * wait_norm + late_promo * late_high
        scored.append((score, tx.fee, tx))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered = [tx for _, _, tx in scored]

    normal = [tx for tx in ordered if tx.risk_score < risk_threshold]
    risky = [tx for tx in ordered if tx.risk_score >= risk_threshold]
    risky.sort(key=lambda tx: (-tx.fee, -tx.risk_score))

    # 将高风险交易插入中部，降低头尾暴露
    result = list(normal)
    mid = len(result) // 2
    for tx in risky:
        result.insert(mid, tx)
        mid += 1
    return _apply_nonce_order(result)


def run_baseline(pool: List[Transaction],
                 method: str) -> List[Transaction]:
    """统一接口"""
    if method == "fifo":
        return fifo_sort(pool)
    elif method == "gas":
        return gas_priority_sort(pool)
    elif method == "heuristic":
        return heuristic_sort(pool)
    elif method == "fee_risk_linear":
        return fee_risk_linear_sort(pool)
    elif method == "fair_fee":
        return fair_fee_sort(pool)
    elif method == "center_aware":
        return center_aware_greedy_sort(pool)
    else:
        raise ValueError(f"Unknown baseline: {method}")
