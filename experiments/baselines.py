"""基线排序方法: FIFO, Gas 优先, 线性/公平启发式、中心插入启发式与动态三目标贪心。"""

from __future__ import annotations

from copy import deepcopy
import itertools
import math
from typing import Any, List

import numpy as np

import config as C
from transaction import Transaction


def _effective_gas_limit(txs: List[Transaction]) -> int:
    return C.effective_block_gas_limit(len(txs))


def _valid_candidate_indices(
    remaining: List[Transaction],
    selected: List[Transaction],
    gas_left: int,
) -> list[int]:
    """返回当前 remaining 中满足 Gas 与 Nonce 约束的索引。"""
    selected_nonces: dict[int, int] = {}
    for tx in selected:
        selected_nonces[tx.sender] = max(selected_nonces.get(tx.sender, -1), tx.nonce)

    valid = []
    for idx, tx in enumerate(remaining):
        if tx.gas > gas_left:
            continue
        if tx.nonce > 0 and selected_nonces.get(tx.sender, -1) < tx.nonce - 1:
            continue
        valid.append(idx)
    return valid


def _apply_nonce_order(txs: List[Transaction], block_gas_limit: int | None = None) -> List[Transaction]:
    """保证同地址交易按 nonce 顺序, 在原有排序基础上做拓扑修正。"""
    selected: List[Transaction] = []
    remaining = list(txs)
    gas_left = int(block_gas_limit) if block_gas_limit is not None else _effective_gas_limit(txs)

    while remaining:
        progress = False
        next_remaining = []
        valid_indices = set(_valid_candidate_indices(remaining, selected, gas_left))
        for idx, tx in enumerate(remaining):
            if idx not in valid_indices:
                next_remaining.append(tx)
                continue
            selected.append(tx)
            gas_left -= tx.gas
            progress = True
        remaining = next_remaining
        if not progress:
            break
    return selected


def fifo_sort(pool: List[Transaction]) -> List[Transaction]:
    """按到达时间排序。"""
    ordered = sorted(pool, key=lambda tx: tx.arrival_time)
    return _apply_nonce_order(ordered, _effective_gas_limit(pool))


def gas_priority_sort(pool: List[Transaction]) -> List[Transaction]:
    """按手续费从高到低排序。"""
    ordered = sorted(pool, key=lambda tx: -tx.fee)
    return _apply_nonce_order(ordered, _effective_gas_limit(pool))


def heuristic_sort(
    pool: List[Transaction],
    risk_threshold: float = C.HEURISTIC_RISK_THRESHOLD,
) -> List[Transaction]:
    """Gas 优先 + 高风险交易降级到区块中间位置。"""
    normal = [tx for tx in pool if tx.risk_score < risk_threshold]
    risky = [tx for tx in pool if tx.risk_score >= risk_threshold]

    normal_sorted = sorted(normal, key=lambda tx: -tx.fee)
    risky_sorted = sorted(risky, key=lambda tx: -tx.fee)

    result = list(normal_sorted)
    mid = len(result) // 2
    for tx in risky_sorted:
        result.insert(mid, tx)
        mid += 1

    return _apply_nonce_order(result, _effective_gas_limit(pool))


def fee_risk_linear_sort(
    pool: List[Transaction],
    lambda_f: float = C.LINEAR_LAMBDA_F,
    lambda_r: float = C.LINEAR_LAMBDA_R,
) -> List[Transaction]:
    """score(tx) = lambda_f * fee_norm - lambda_r * risk_score。"""
    if not pool:
        return []
    max_fee = max(tx.fee for tx in pool)
    ordered = sorted(
        pool,
        key=lambda tx: -(lambda_f * (tx.fee / max(max_fee, 1e-8)) - lambda_r * tx.risk_score),
    )
    return _apply_nonce_order(ordered, _effective_gas_limit(pool))


def fair_fee_sort(
    pool: List[Transaction],
    lambda_f: float = C.FAIR_LAMBDA_F,
    lambda_w: float = C.FAIR_LAMBDA_W,
) -> List[Transaction]:
    """score(tx) = lambda_f * fee_norm + lambda_w * wait_norm。"""
    if not pool:
        return []
    max_fee = max(tx.fee for tx in pool)
    t_now = max(tx.arrival_time for tx in pool) + 1.0
    min_arrival = min(tx.arrival_time for tx in pool)
    wait_denom = max(t_now - min_arrival, 1e-8)
    ordered = sorted(
        pool,
        key=lambda tx: -(
            lambda_f * (tx.fee / max(max_fee, 1e-8))
            + lambda_w * ((t_now - tx.arrival_time) / wait_denom)
        ),
    )
    return _apply_nonce_order(ordered, _effective_gas_limit(pool))


def center_insertion_heuristic_sort(
    pool: List[Transaction],
    lambda_f: float = C.CENTER_AWARE_LAMBDA_F,
    lambda_w: float = C.CENTER_AWARE_LAMBDA_W,
    late_promo: float = C.CENTER_AWARE_LATE_PROMO,
    risk_threshold: float = C.HEURISTIC_RISK_THRESHOLD,
) -> List[Transaction]:
    """收益/等待静态打分后，将高风险交易插入中部的启发式。"""
    if not pool:
        return []
    max_fee = max(tx.fee for tx in pool)
    t_now = max(tx.arrival_time for tx in pool) + 1.0
    min_arrival = min(tx.arrival_time for tx in pool)
    wait_denom = max(t_now - min_arrival, 1e-8)
    median_arrival = float(np.median([tx.arrival_time for tx in pool]))
    median_fee = float(np.median([tx.fee for tx in pool]))

    scored = []
    for tx in pool:
        fee_norm = tx.fee / max(max_fee, 1e-8)
        wait_norm = (t_now - tx.arrival_time) / wait_denom
        late_high = 1.0 if (tx.arrival_time > median_arrival and tx.fee > median_fee) else 0.0
        score = lambda_f * fee_norm + lambda_w * wait_norm + late_promo * late_high
        scored.append((score, tx.fee, tx))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered = [tx for _, _, tx in scored]

    normal = [tx for tx in ordered if tx.risk_score < risk_threshold]
    risky = [tx for tx in ordered if tx.risk_score >= risk_threshold]
    risky.sort(key=lambda tx: (-tx.fee, -tx.risk_score))

    result = list(normal)
    mid = len(result) // 2
    for tx in risky:
        result.insert(mid, tx)
        mid += 1
    return _apply_nonce_order(result, _effective_gas_limit(pool))


def center_aware_greedy_sort(*args, **kwargs) -> List[Transaction]:
    """旧入口兼容: 实际语义为 Center-Insertion Heuristic。"""
    return center_insertion_heuristic_sort(*args, **kwargs)


def _risk_position_penalty(pos_ratio: float) -> float:
    dist = abs(pos_ratio - C.RISK_CENTER)
    sigma = max(C.RISK_POSITION_SIGMA, 1e-6)
    return float(1.0 - math.exp(-0.5 * (dist / sigma) ** 2))


def _estimated_block_capacity(pool: List[Transaction], block_gas_limit: int) -> int:
    if not pool:
        return 1
    avg_gas = sum(tx.gas for tx in pool) / max(len(pool), 1)
    return max(int(block_gas_limit / avg_gas), 1) if avg_gas > 0 else max(len(pool), 1)


def dynamic_tri_objective_greedy_sort(
    pool: List[Transaction],
    lambda_f: float = C.DYNAMIC_TRI_LAMBDA_F,
    lambda_w: float = C.DYNAMIC_TRI_LAMBDA_W,
    lambda_r: float = C.DYNAMIC_TRI_LAMBDA_R,
) -> List[Transaction]:
    """逐步动态三目标贪心。

    每一步只在当前合法集合内选择，并根据当前位置重新计算风险位置权重。
    目标项与环境保持同一组归一化口径：fee_norm、wait_norm 和平滑位置风险。
    """
    if not pool:
        return []

    remaining = list(pool)
    selected: List[Transaction] = []
    block_gas_limit = _effective_gas_limit(pool)
    gas_left = block_gas_limit
    max_fee = max(tx.fee for tx in pool)
    t_now = max(tx.arrival_time for tx in pool) + 1.0
    min_arrival = min(tx.arrival_time for tx in pool)
    wait_denom = max(t_now - min_arrival, 1e-8)
    est_capacity = _estimated_block_capacity(pool, block_gas_limit)

    while remaining:
        valid_indices = _valid_candidate_indices(remaining, selected, gas_left)
        if not valid_indices:
            break

        pos_ratio = len(selected) / max(est_capacity - 1, 1)
        pos_weight = _risk_position_penalty(pos_ratio)
        scored = []
        for idx in valid_indices:
            tx = remaining[idx]
            fee_norm = tx.fee / max(max_fee, 1e-8)
            wait_norm = (t_now - tx.arrival_time) / wait_denom
            risk_term = tx.risk_score * pos_weight
            score = lambda_f * fee_norm + lambda_w * wait_norm - lambda_r * risk_term
            scored.append((score, tx.fee, wait_norm, -tx.gas, idx))

        _score, _fee, _wait, _neg_gas, chosen_idx = max(scored, key=lambda item: item[:4])
        chosen = remaining.pop(chosen_idx)
        selected.append(chosen)
        gas_left -= chosen.gas

    return selected


def _normalize_baseline_method(method: str) -> str:
    if method == "center_aware":
        return "center_insertion"
    return method


def run_baseline(
    pool: List[Transaction],
    method: str,
    params: dict[str, Any] | None = None,
) -> List[Transaction]:
    """统一接口。params 仅作用于含参数基线。"""
    params = dict(params or {})
    method_norm = _normalize_baseline_method(method)
    if method_norm == "fifo":
        return fifo_sort(pool)
    if method_norm == "gas":
        return gas_priority_sort(pool)
    if method_norm == "heuristic":
        return heuristic_sort(pool, **{k: v for k, v in params.items() if k == "risk_threshold"})
    if method_norm == "fee_risk_linear":
        return fee_risk_linear_sort(pool, **params)
    if method_norm == "fair_fee":
        return fair_fee_sort(pool, **params)
    if method_norm == "center_insertion":
        return center_insertion_heuristic_sort(pool, **params)
    if method_norm == "dynamic_tri_objective":
        return dynamic_tri_objective_greedy_sort(pool, **params)
    raise ValueError(f"Unknown baseline: {method}")


def baseline_param_grid(method: str) -> list[dict[str, float]]:
    """返回验证池调参网格；无参数基线返回默认空参数。"""
    method_norm = _normalize_baseline_method(method)
    lambda_f_grid = list(getattr(C, "BASELINE_TUNING_LAMBDA_F_GRID", [1.0]))
    lambda_w_grid = list(getattr(C, "BASELINE_TUNING_LAMBDA_W_GRID", [0.5]))
    lambda_r_grid = list(getattr(C, "BASELINE_TUNING_LAMBDA_R_GRID", [0.5]))
    late_grid = list(getattr(C, "BASELINE_TUNING_LATE_PROMO_GRID", [C.CENTER_AWARE_LATE_PROMO]))

    if method_norm == "fee_risk_linear":
        return [
            {"lambda_f": float(lf), "lambda_r": float(lr)}
            for lf, lr in itertools.product(lambda_f_grid, lambda_r_grid)
        ]
    if method_norm == "fair_fee":
        return [
            {"lambda_f": float(lf), "lambda_w": float(lw)}
            for lf, lw in itertools.product(lambda_f_grid, lambda_w_grid)
        ]
    if method_norm == "center_insertion":
        return [
            {"lambda_f": float(lf), "lambda_w": float(lw), "late_promo": float(lp)}
            for lf, lw, lp in itertools.product(lambda_f_grid, lambda_w_grid, late_grid)
        ]
    if method_norm == "dynamic_tri_objective":
        return [
            {"lambda_f": float(lf), "lambda_w": float(lw), "lambda_r": float(lr)}
            for lf, lw, lr in itertools.product(lambda_f_grid, lambda_w_grid, lambda_r_grid)
        ]
    return [{}]


def grid_search_baseline_params(
    methods: list[str],
    val_pools: list[list[Transaction]],
    constraints: dict | None = None,
    operating_mode: str | None = None,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, Any]]]:
    """在固定验证池上为含参数基线做网格搜索。"""
    from metrics import compute_all_metrics, summary_metric_bundle

    tuning: dict[str, list[dict]] = {}
    best_params: dict[str, dict[str, Any]] = {}
    mode = operating_mode or getattr(C, "OPERATING_MODE", "balanced")

    for method in methods:
        candidates = baseline_param_grid(method)
        rows = []
        for params in candidates:
            metrics_seq = []
            for pool in val_pools:
                selected = run_baseline(deepcopy(pool), method, params=params)
                metrics_seq.append(compute_all_metrics(selected, pool))
            bundle = summary_metric_bundle(
                metrics_seq,
                constraints=constraints,
                target_metric="block_fee_norm",
                mode=mode,
            )
            row = {
                "params": params,
                "validation_score": float(bundle["two_stage_selection_score"]),
                "feasible_rate": float(bundle["feasible_rate"]),
                "feasible_fee_mean": bundle["feasible_set_fee_mean"],
                "risk_adjusted_fee_mean": float(bundle["risk_adjusted_fee_mean"]),
                "composite_score_mean": float(np.mean([m.get("composite_score", 0.0) for m in metrics_seq])) if metrics_seq else 0.0,
                "n_val_episodes": len(metrics_seq),
            }
            rows.append(row)
        rows.sort(
            key=lambda item: (
                item["validation_score"],
                item["feasible_rate"],
                -1.0 if item["feasible_fee_mean"] is None else float(item["feasible_fee_mean"]),
                item["risk_adjusted_fee_mean"],
            ),
            reverse=True,
        )
        tuning[method] = rows
        best_params[method] = rows[0]["params"] if rows else {}
    return tuning, best_params
