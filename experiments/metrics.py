"""评价指标: 区块收益、公平性、风险暴露与综合目标"""

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


def _oldest_pool_ids(pool: List[Transaction], q: float = C.FAIR_OLDEST_RATIO) -> set[int]:
    if not pool:
        return set()
    k = max(int(len(pool) * q), 1)
    oldest = sorted(pool, key=lambda tx: tx.arrival_time)[:k]
    return {tx.tid for tx in oldest}


def oldest_coverage_ratio(selected: List[Transaction],
                          pool: List[Transaction],
                          q: float = C.FAIR_OLDEST_RATIO) -> float:
    """候选池 oldest-q 交易被服务比例。"""
    oldest_ids = _oldest_pool_ids(pool, q=q)
    if not oldest_ids:
        return 0.0
    selected_ids = {tx.tid for tx in selected}
    served = len(oldest_ids & selected_ids)
    return served / len(oldest_ids)


def starvation_gap(selected: List[Transaction],
                   pool: List[Transaction],
                   q: float = C.FAIR_OLDEST_RATIO) -> float:
    """候选池 oldest-q 中未被服务比例。"""
    return 1.0 - oldest_coverage_ratio(selected, pool, q=q)


def tail_wait_reduction(selected: List[Transaction],
                        pool: List[Transaction],
                        q: float = C.FAIR_TAIL_QUANTILE) -> float:
    """高分位等待时间“被服务覆盖率”近似量化。"""
    if not pool:
        return 0.0
    t_now = max(tx.arrival_time for tx in pool) + 1.0
    pool_waits = np.array([max(t_now - tx.arrival_time, 0.0) for tx in pool], dtype=np.float64)
    threshold = float(np.quantile(pool_waits, q))
    tail_ids = {tx.tid for tx in pool if (t_now - tx.arrival_time) >= threshold}
    if not tail_ids:
        return 0.0
    selected_ids = {tx.tid for tx in selected}
    served_tail = len(tail_ids & selected_ids)
    return served_tail / len(tail_ids)


def risk_exposure(selected: List[Transaction],
                  ratio: float = C.RISK_POSITION_RATIO,
                  threshold: float = C.HEURISTIC_RISK_THRESHOLD) -> float:
    """高风险交易出现在头尾敏感位置的比例"""
    K = len(selected)
    if K == 0:
        return 0.0
    n_edge = max(int(K * ratio), 1)
    head_idx = set(range(n_edge))
    tail_idx = set(range(max(K - n_edge, 0), K))
    sensitive_idx = head_idx | tail_idx  # 去重
    sensitive = [selected[i] for i in sensitive_idx]

    risky_in_sensitive = sum(1 for tx in sensitive
                            if tx.risk_score >= threshold)
    total_risky = sum(1 for tx in selected if tx.risk_score >= threshold)
    if total_risky == 0:
        return 0.0
    return risky_in_sensitive / total_risky


def edge_risk_ratio(selected: List[Transaction],
                    ratio: float = C.RISK_POSITION_RATIO,
                    threshold: float = C.HEURISTIC_RISK_THRESHOLD) -> float:
    """头尾敏感区内高风险交易占比。"""
    K = len(selected)
    if K == 0:
        return 0.0
    n_edge = max(int(K * ratio), 1)
    sensitive_idx = set(range(n_edge)) | set(range(max(K - n_edge, 0), K))
    sensitive = [selected[i] for i in sensitive_idx]
    if not sensitive:
        return 0.0
    risky_in_sensitive = sum(1 for tx in sensitive if tx.risk_score >= threshold)
    return risky_in_sensitive / len(sensitive)


def gas_utilization(selected: List[Transaction],
                    pool_size: int | None = None) -> float:
    """Gas 利用率"""
    used = sum(tx.gas for tx in selected)
    gas_limit = C.effective_block_gas_limit(pool_size)
    return used / max(gas_limit, 1)


def avg_risky_rank(selected: List[Transaction],
                   threshold: float = C.HEURISTIC_RISK_THRESHOLD) -> float:
    """高风险交易在区块中的平均相对位置 (0=头, 0.5=中, 1=尾)"""
    K = len(selected)
    if K <= 1:
        return 0.5
    positions = [i / (K - 1) for i, tx in enumerate(selected)
                 if tx.risk_score >= threshold]
    if not positions:
        return 0.5
    return float(np.mean(positions))


def packing_ratio(selected: List[Transaction],
                  pool: List[Transaction]) -> float:
    """打包比例: 实际选入数 / 候选池大小"""
    if not pool:
        return 0.0
    return len(selected) / len(pool)


def top10_risk_ratio(selected: List[Transaction],
                     threshold: float = C.HEURISTIC_RISK_THRESHOLD) -> float:
    """区块前 10% 位置中高风险交易占比"""
    K = len(selected)
    if K == 0:
        return 0.0
    n_top = max(int(K * 0.1), 1)
    top_txs = selected[:n_top]
    risky_count = sum(1 for tx in top_txs if tx.risk_score >= threshold)
    return risky_count / n_top


def risky_inclusion_rate(selected: List[Transaction],
                         pool: List[Transaction],
                         threshold: float = C.HEURISTIC_RISK_THRESHOLD) -> float:
    """候选池中高风险交易被打包的比例，用于排除选择性不打包造成的伪风险下降。"""
    risky_pool = [tx for tx in pool if tx.risk_score >= threshold]
    if not risky_pool:
        return 0.0
    selected_ids = {tx.tid for tx in selected}
    included = sum(1 for tx in risky_pool if tx.tid in selected_ids)
    return included / len(risky_pool)


def late_arrival_promotion_rate(selected: List[Transaction],
                                pool: List[Transaction]) -> float:
    """后到达高手续费交易被提前打包的比例"""
    if not pool or not selected:
        return 0.0
    median_arrival = float(np.median([tx.arrival_time for tx in pool]))
    median_fee = float(np.median([tx.fee for tx in pool]))
    # 后到达且高手续费的交易
    late_high = [tx for tx in pool
                 if tx.arrival_time > median_arrival and tx.fee > median_fee]
    if not late_high:
        return 0.0
    K = len(selected)
    top_half_tids = {selected[i].tid for i in range(K // 2)}
    promoted = sum(1 for tx in late_high if tx.tid in top_half_tids)
    return promoted / len(late_high)


def composite_score(metrics: dict,
                    pool: List[Transaction] | None = None) -> float:
    """综合目标分数: fee + fairness - risk + oldest_coverage。"""
    fee = float(metrics.get("block_fee", 0.0))
    if pool:
        pool_fee = sum(tx.fee for tx in pool)
        fee_norm = fee / max(pool_fee, 1e-8)
    else:
        fee_norm = float(metrics.get("block_fee_norm", 0.0))

    fairness = float(np.clip(metrics.get("fairness", 0.0), 0.0, 1.0))
    risk = float(np.clip(metrics.get("risk_exposure", 0.0), 0.0, 1.0))
    oldest_cov = float(np.clip(metrics.get("oldest_coverage", 0.0), 0.0, 1.0))

    return (
        C.COMPOSITE_W_FEE * fee_norm
        + C.COMPOSITE_W_FAIRNESS * fairness
        - C.COMPOSITE_W_RISK * risk
        + C.COMPOSITE_W_OLDEST_COVERAGE * oldest_cov
    )


def _fee_norm(metrics: dict, pool: List[Transaction] | None = None) -> float:
    if pool:
        pool_fee = sum(tx.fee for tx in pool)
        return float(metrics.get("block_fee", 0.0)) / max(pool_fee, 1e-8)
    return float(metrics.get("block_fee_norm", 0.0))


def risk_safety_score(metrics: dict, risk_ref: float | None = None) -> float:
    """将越低越好的 risk_exposure 转换为 [0,1] 风险安全得分。"""
    ref = float(risk_ref if risk_ref is not None else getattr(C, "TRADE_SCORE_RISK_REF", C.VALIDATION_RISK_CEIL))
    risk = float(np.clip(metrics.get("risk_exposure", 0.0), 0.0, 1.0))
    return float(1.0 - min(risk / max(ref, 1e-8), 1.0))


def edge_risk_safety_score(metrics: dict, edge_ref: float | None = None) -> float:
    """将越低越好的 edge10_risk 转换为 [0,1] 头尾风险安全得分。"""
    ref = float(edge_ref if edge_ref is not None else getattr(C, "TRADE_SCORE_EDGE_REF", C.VALIDATION_TOP10_RISK_CEIL))
    edge = float(np.clip(metrics.get("edge10_risk", 0.0), 0.0, 1.0))
    return float(1.0 - min(edge / max(ref, 1e-8), 1.0))


def multi_objective_trade_score(
    metrics: dict,
    pool: List[Transaction] | None = None,
    weights: dict | None = None,
    risk_ref: float | None = None,
    edge_ref: float | None = None,
) -> float:
    """计划文档中的 S_trade: 收益、等待公平、效率和风险安全的加权折中分。"""
    w = dict(weights or getattr(C, "TRADE_SCORE_WEIGHTS", {}))
    fee = float(np.clip(_fee_norm(metrics, pool), 0.0, 1.0))
    fairness = float(np.clip(metrics.get("fairness", 0.0), 0.0, 1.0))
    old_tx = float(np.clip(metrics.get("old_tx_pack_rate", metrics.get("oldest_coverage", 0.0)), 0.0, 1.0))
    gas = float(np.clip(metrics.get("gas_util", 0.0), 0.0, 1.0))
    packing = float(np.clip(metrics.get("packing_ratio", 0.0), 0.0, 1.0))
    risk_safe = risk_safety_score(metrics, risk_ref=risk_ref)
    edge_safe = edge_risk_safety_score(metrics, edge_ref=edge_ref)
    return float(
        float(w.get("fee", 0.0)) * fee
        + float(w.get("fairness", 0.0)) * fairness
        + float(w.get("old_tx", 0.0)) * old_tx
        + float(w.get("gas", 0.0)) * gas
        + float(w.get("packing", 0.0)) * packing
        + float(w.get("risk_safety", 0.0)) * risk_safe
        + float(w.get("edge_safety", 0.0)) * edge_safe
    )


def risk_aware_trade_score(metrics: dict, pool: List[Transaction] | None = None) -> float:
    """计划文档中的 S_risk-aware: 强化风险安全权重的折中分。"""
    return multi_objective_trade_score(
        metrics,
        pool=pool,
        weights=getattr(C, "RISK_AWARE_TRADE_SCORE_WEIGHTS", C.TRADE_SCORE_WEIGHTS),
    )


def constrained_trade_score(metrics: dict, pool: List[Transaction] | None = None) -> float:
    """计划文档中的 S_constrained: 以收益/效率为基础，对约束违反施加 hinge penalty。"""
    fee = float(np.clip(_fee_norm(metrics, pool), 0.0, 1.0))
    gas = float(np.clip(metrics.get("gas_util", 0.0), 0.0, 1.0))
    packing = float(np.clip(metrics.get("packing_ratio", 0.0), 0.0, 1.0))
    fairness = float(metrics.get("fairness", 0.0))
    risk = float(metrics.get("risk_exposure", 1.0))
    edge = float(metrics.get("edge10_risk", 1.0))

    fair_gap = max(float(getattr(C, "CONSTRAINED_TRADE_FAIRNESS_MIN", 0.90)) - fairness, 0.0)
    risk_gap = max(risk - float(getattr(C, "CONSTRAINED_TRADE_RISK_MAX", 0.15)), 0.0)
    edge_gap = max(edge - float(getattr(C, "CONSTRAINED_TRADE_EDGE_MAX", 0.08)), 0.0)
    gas_gap = max(float(getattr(C, "CONSTRAINED_TRADE_GAS_MIN", 0.95)) - gas, 0.0)

    return float(
        fee
        + float(getattr(C, "CONSTRAINED_TRADE_MU_GAS", 0.10)) * gas
        + float(getattr(C, "CONSTRAINED_TRADE_MU_PACKING", 0.05)) * packing
        - float(getattr(C, "CONSTRAINED_TRADE_LAMBDA_FAIRNESS", 1.0)) * fair_gap
        - float(getattr(C, "CONSTRAINED_TRADE_LAMBDA_RISK", 1.0)) * risk_gap
        - float(getattr(C, "CONSTRAINED_TRADE_LAMBDA_EDGE", 1.0)) * edge_gap
        - float(getattr(C, "CONSTRAINED_TRADE_LAMBDA_GAS", 1.0)) * gas_gap
    )


def constrained_success_score(metrics: dict,
                              constraints: dict | None = None,
                              target_metric: str = "block_fee_norm",
                              infeasible_score: float | None = -1.0) -> float | None:
    """约束式得分: 满足约束时返回目标指标，否则给不可行分。"""
    cons = _resolve_constraints(constraints)
    feasible = _is_feasible(metrics, cons)
    if not feasible:
        if infeasible_score is None:
            return None
        return float(infeasible_score)
    return float(metrics.get(target_metric, 0.0))


def _resolve_constraints(constraints: dict | None = None) -> dict:
    """统一解析约束配置，默认使用 strict 评估口径。"""
    if constraints is not None:
        return constraints
    if hasattr(C, "resolve_constraints"):
        return C.resolve_constraints(profile="strict", for_training=False)
    return {
        "fairness_floor": C.VALIDATION_FAIRNESS_FLOOR,
        "oldest_coverage_floor": C.VALIDATION_OLDEST_COVERAGE_FLOOR,
        "risk_ceil": C.VALIDATION_RISK_CEIL,
        "edge10_risk_ceil": C.VALIDATION_EDGE10_RISK_CEIL,
        "top10_risk_ceil": C.VALIDATION_TOP10_RISK_CEIL,  # diagnostic
    }


def _is_feasible(metrics: dict, constraints: dict) -> bool:
    fairness_floor = float(constraints.get("fairness_floor", C.VALIDATION_FAIRNESS_FLOOR))
    oldest_floor = float(constraints.get("oldest_coverage_floor", C.VALIDATION_OLDEST_COVERAGE_FLOOR))
    risk_ceil = float(constraints.get("risk_ceil", C.VALIDATION_RISK_CEIL))
    edge10_risk_ceil = float(constraints.get("edge10_risk_ceil", C.VALIDATION_EDGE10_RISK_CEIL))
    fairness = float(metrics.get("fairness", 0.0))
    oldest_cov = float(metrics.get("oldest_coverage", 0.0))
    risk = float(metrics.get("risk_exposure", 1.0))
    edge10_risk = float(metrics.get("edge10_risk", 1.0))
    return (
        fairness >= fairness_floor
        and oldest_cov >= oldest_floor
        and risk <= risk_ceil
        and edge10_risk <= edge10_risk_ceil
    )


def violation_breakdown(metrics_seq: list[dict],
                        constraints: dict | None = None) -> dict:
    """统计每类约束违约次数。"""
    cons = _resolve_constraints(constraints)
    fairness_floor = float(cons.get("fairness_floor", C.VALIDATION_FAIRNESS_FLOOR))
    oldest_floor = float(cons.get("oldest_coverage_floor", C.VALIDATION_OLDEST_COVERAGE_FLOOR))
    risk_ceil = float(cons.get("risk_ceil", C.VALIDATION_RISK_CEIL))
    edge10_risk_ceil = float(cons.get("edge10_risk_ceil", C.VALIDATION_EDGE10_RISK_CEIL))

    breakdown = {
        "fairness_floor": 0,
        "oldest_coverage_floor": 0,
        "risk_ceil": 0,
        "edge10_risk_ceil": 0,
        "any_violation": 0,
    }
    for metrics in metrics_seq:
        violated = False
        if float(metrics.get("fairness", 0.0)) < fairness_floor:
            breakdown["fairness_floor"] += 1
            violated = True
        if float(metrics.get("oldest_coverage", 0.0)) < oldest_floor:
            breakdown["oldest_coverage_floor"] += 1
            violated = True
        if float(metrics.get("risk_exposure", 1.0)) > risk_ceil:
            breakdown["risk_ceil"] += 1
            violated = True
        if float(metrics.get("edge10_risk", 1.0)) > edge10_risk_ceil:
            breakdown["edge10_risk_ceil"] += 1
            violated = True
        if violated:
            breakdown["any_violation"] += 1
    return breakdown


def feasible_rate(metrics_seq: list[dict], constraints: dict | None = None) -> float:
    """可行样本占比。"""
    if not metrics_seq:
        return 0.0
    cons = _resolve_constraints(constraints)
    feasible = [1 for m in metrics_seq if _is_feasible(m, cons)]
    return len(feasible) / len(metrics_seq)


def feasible_set_fee_score(metrics_seq: list[dict],
                           constraints: dict | None = None,
                           target_metric: str = "block_fee_norm") -> dict:
    """仅在可行子集上计算收益统计；不可行样本不参与均值。"""
    cons = _resolve_constraints(constraints)
    scores = [
        constrained_success_score(m, constraints=cons, target_metric=target_metric, infeasible_score=None)
        for m in metrics_seq
    ]
    feasible = [float(v) for v in scores if v is not None]
    return {
        "feasible_fee_mean": float(np.mean(feasible)) if feasible else None,
        "feasible_fee_std": float(np.std(feasible)) if feasible else None,
        "feasible_count": len(feasible),
        "infeasible_count": len(scores) - len(feasible),
        "n_episodes": len(scores),
    }


def effective_variance_check(values: list[float], epsilon: float = 1e-10) -> dict:
    """低区分度检测：方差极小时标记指标饱和。"""
    if not values:
        return {"effective_variance": 0.0, "is_low_variance": True}
    var = float(np.var(values))
    return {
        "effective_variance": var,
        "is_low_variance": bool(var < epsilon),
    }


def two_stage_selection_score(
    metrics_seq: list[dict],
    constraints: dict | None = None,
    mode: str | None = None,
    target_metric: str = "block_fee_norm",
    min_feasible_rate: float = C.CONSTRAINED_RANK_MIN_FEASIBLE_RATE,
) -> dict:
    """两阶段选优分：可行率优先，其次可行域收益，再看风险调整收益。"""
    cons = _resolve_constraints(constraints)
    fr = feasible_rate(metrics_seq, constraints=cons)
    feasible_stats = feasible_set_fee_score(metrics_seq, constraints=cons, target_metric=target_metric)
    feasible_risk_adj = [
        risk_adjusted_fee_score(m)
        for m in metrics_seq
        if _is_feasible(m, cons)
    ]
    risk_adj_mean = float(np.mean(feasible_risk_adj)) if feasible_risk_adj else None

    if hasattr(C, "normalize_operating_mode"):
        mode_norm = C.normalize_operating_mode(mode)
    else:
        mode_norm = str(mode or "balanced").lower()
    if hasattr(C, "operating_mode_weights"):
        weights = C.operating_mode_weights(mode_norm)
    else:
        weights = {"tier": 7.0, "feasible_rate": 5.0, "feasible_fee": 6.0, "risk_adjusted_fee": 3.0}

    tier = 1 if fr >= min_feasible_rate else 0
    fee_for_score = float(feasible_stats["feasible_fee_mean"]) if feasible_stats["feasible_fee_mean"] is not None else -1.0
    risk_for_score = float(risk_adj_mean) if risk_adj_mean is not None else -1.0
    selection_score = (
        float(weights.get("tier", 0.0)) * tier
        + float(weights.get("feasible_rate", 0.0)) * fr
        + float(weights.get("feasible_fee", 0.0)) * fee_for_score
        + float(weights.get("risk_adjusted_fee", 0.0)) * risk_for_score
    )

    return {
        "selection_policy_version": getattr(C, "SELECTION_POLICY_VERSION", C.RANKING_POLICY_VERSION),
        "operating_mode": mode_norm,
        "feasible_rate_tier": tier,
        "feasible_rate": fr,
        "feasible_set_fee_mean": feasible_stats["feasible_fee_mean"],
        "risk_adjusted_fee": risk_adj_mean,
        "two_stage_selection_score": float(selection_score),
    }


def operating_point_rank(
    method_payload: dict[str, dict],
    mode: str | None = None,
    min_feasible_rate: float = C.CONSTRAINED_RANK_MIN_FEASIBLE_RATE,
) -> list[dict]:
    """输出三档 operating mode 下的方法排序与关键分解字段。"""
    if hasattr(C, "normalize_operating_mode"):
        mode_norm = C.normalize_operating_mode(mode)
    else:
        mode_norm = str(mode or "balanced").lower()
    if hasattr(C, "operating_mode_weights"):
        weights = C.operating_mode_weights(mode_norm)
    else:
        weights = {"tier": 7.0, "feasible_rate": 5.0, "feasible_fee": 6.0, "risk_adjusted_fee": 3.0}

    rows = []
    for method, item in method_payload.items():
        fr = float(item.get("feasible_rate", 0.0))
        tier = int(item.get("feasible_rate_tier", 1 if fr >= min_feasible_rate else 0))
        fee_raw = item.get("feasible_fee_mean", item.get("feasible_set_fee_mean", item.get("constrained_fee_mean")))
        risk_adj_raw = item.get("risk_adjusted_fee_mean", item.get("risk_adjusted_fee"))
        fee = float(fee_raw) if fee_raw is not None else -1.0
        risk_adj = float(risk_adj_raw) if risk_adj_raw is not None else -1.0
        score = (
            float(weights.get("tier", 0.0)) * tier
            + float(weights.get("feasible_rate", 0.0)) * fr
            + float(weights.get("feasible_fee", 0.0)) * fee
            + float(weights.get("risk_adjusted_fee", 0.0)) * risk_adj
        )
        rows.append({
            "method": method,
            "operating_mode": mode_norm,
            "selection_policy_version": getattr(C, "SELECTION_POLICY_VERSION", C.RANKING_POLICY_VERSION),
            "feasible_rate_tier": tier,
            "feasible_rate": fr,
            "feasible_set_fee_mean": None if fee_raw is None else float(fee_raw),
            "risk_adjusted_fee_mean": None if risk_adj_raw is None else float(risk_adj_raw),
            "two_stage_selection_score": float(score),
        })
    rows.sort(
        key=lambda x: (
            x["two_stage_selection_score"],
            x["feasible_rate_tier"],
            x["feasible_rate"],
            -1.0 if x["feasible_set_fee_mean"] is None else x["feasible_set_fee_mean"],
            -1.0 if x["risk_adjusted_fee_mean"] is None else x["risk_adjusted_fee_mean"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def constrained_ranking(method_payload: dict[str, dict],
                        min_feasible_rate: float = C.CONSTRAINED_RANK_MIN_FEASIBLE_RATE,
                        mode: str = "balanced") -> list[str]:
    """两段式 constrained 排序：先可行率档位，再可行域收益。"""
    rows = operating_point_rank(
        method_payload,
        mode=mode,
        min_feasible_rate=min_feasible_rate,
    )
    return [row["method"] for row in rows]


def summary_metric_bundle(metrics_seq: list[dict],
                          constraints: dict | None = None,
                          target_metric: str = "block_fee_norm",
                          mode: str | None = None) -> dict:
    """统一输出 constrained 与解释性统计字段。"""
    cons = _resolve_constraints(constraints)
    feasible_stats = feasible_set_fee_score(metrics_seq, constraints=cons, target_metric=target_metric)
    two_stage = two_stage_selection_score(
        metrics_seq,
        constraints=cons,
        mode=mode,
        target_metric=target_metric,
        min_feasible_rate=C.CONSTRAINED_RANK_MIN_FEASIBLE_RATE,
    )
    vb = violation_breakdown(metrics_seq, constraints=cons)
    all_fee_vals = [float(m.get(target_metric, 0.0)) for m in metrics_seq]
    all_risk_adjusted = [risk_adjusted_fee_score(m) for m in metrics_seq]
    variance_metrics = ("block_fee_norm", "oldest_coverage", "tail_wait_reduction", "wait_p95", "wait_p99", "wait_gini")
    effective_variance_checks = {
        metric: effective_variance_check([float(m.get(metric, 0.0)) for m in metrics_seq])
        for metric in variance_metrics
    }
    low_variance_flags = [m for m, c in effective_variance_checks.items() if c.get("is_low_variance")]
    top_violation = max(
        ("fairness_floor", "oldest_coverage_floor", "risk_ceil", "edge10_risk_ceil"),
        key=lambda k: vb[k],
    ) if metrics_seq else "none"
    return {
        "score_policy_version": C.SCORE_POLICY_VERSION,
        "ranking_policy_version": C.RANKING_POLICY_VERSION,
        "feasible_rate": feasible_rate(metrics_seq, constraints=cons),
        "feasible_rate_tier": two_stage["feasible_rate_tier"],
        "feasible_set_fee_mean": feasible_stats["feasible_fee_mean"],
        "feasible_set_fee_std": feasible_stats["feasible_fee_std"],
        "feasible_count": feasible_stats["feasible_count"],
        "infeasible_count": feasible_stats["infeasible_count"],
        "violation_count": vb["any_violation"],
        "n_episodes": feasible_stats["n_episodes"],
        "all_episode_fee_mean": float(np.mean(all_fee_vals)) if all_fee_vals else 0.0,
        "all_episode_fee_std": float(np.std(all_fee_vals)) if all_fee_vals else 0.0,
        "risk_adjusted_fee_mean": float(np.mean(all_risk_adjusted)) if all_risk_adjusted else 0.0,
        "risk_adjusted_fee_std": float(np.std(all_risk_adjusted)) if all_risk_adjusted else 0.0,
        "two_stage_selection_score": two_stage["two_stage_selection_score"],
        "selection_policy_version": two_stage["selection_policy_version"],
        "operating_mode": two_stage["operating_mode"],
        "violation_breakdown": vb,
        "constraint_violation_top1": top_violation,
        "effective_variance_checks": effective_variance_checks,
        "low_variance_flags": low_variance_flags,
    }


def risk_adjusted_fee_score(metrics: dict,
                            lambda_risk: float = C.RISK_ADJUSTED_FEE_LAMBDA) -> float:
    """风险调整收益分：fee_norm - λ·risk_exposure。"""
    fee_norm = float(metrics.get("block_fee_norm", 0.0))
    risk = float(np.clip(metrics.get("risk_exposure", 0.0), 0.0, 1.0))
    return fee_norm - lambda_risk * risk


def pareto_dominates(a: dict,
                     b: dict,
                     objectives: list[str],
                     lower_is_better: set[str] | None = None,
                     eps: float = 1e-12) -> bool:
    """判断 a 是否 Pareto 支配 b。"""
    lower = lower_is_better or set()
    no_worse_all = True
    strictly_better = False
    for m in objectives:
        av = float(a.get(m, 0.0))
        bv = float(b.get(m, 0.0))
        if m in lower:
            if av > bv + eps:
                no_worse_all = False
                break
            if av < bv - eps:
                strictly_better = True
        else:
            if av < bv - eps:
                no_worse_all = False
                break
            if av > bv + eps:
                strictly_better = True
    return no_worse_all and strictly_better


def pareto_dominance_rate(
    ours_metrics_seq: list[dict],
    baseline_metrics_seq: list[dict],
    objectives: list[str] | None = None,
    lower_is_better: set[str] | None = None,
) -> dict:
    """逐 episode 计算 Pareto dominance 比率。"""
    if objectives is None:
        objectives = ["block_fee", "fairness", "risk_exposure", "oldest_coverage"]
    if lower_is_better is None:
        lower_is_better = {"risk_exposure"}
    n = min(len(ours_metrics_seq), len(baseline_metrics_seq))
    if n == 0:
        return {
            "n_pairs": 0,
            "ours_dominates_rate": 0.0,
            "baseline_dominates_rate": 0.0,
            "non_dominated_rate": 0.0,
        }
    ours_dom = 0
    base_dom = 0
    for i in range(n):
        ours = ours_metrics_seq[i]
        base = baseline_metrics_seq[i]
        if pareto_dominates(ours, base, objectives, lower_is_better):
            ours_dom += 1
        elif pareto_dominates(base, ours, objectives, lower_is_better):
            base_dom += 1
    non_dom = n - ours_dom - base_dom
    return {
        "n_pairs": n,
        "ours_dominates_rate": ours_dom / n,
        "baseline_dominates_rate": base_dom / n,
        "non_dominated_rate": non_dom / n,
    }


def compute_all_metrics(selected: List[Transaction],
                        pool: List[Transaction]) -> dict:
    """计算主指标 + 公平分解指标 + 综合分数。"""
    t_now = max(tx.arrival_time for tx in pool) + 1.0 if pool else 1.0
    oldest_cov = oldest_coverage_ratio(selected, pool, q=C.FAIR_OLDEST_RATIO)
    metrics = {
        "block_fee": block_fee_revenue(selected),
        "fairness": jain_fairness_index(selected, t_now),
        "risk_exposure": risk_exposure(selected),
        "edge_risk_ratio": edge_risk_ratio(selected),
        "edge10_risk": edge_risk_ratio(selected),
        "gas_util": gas_utilization(selected, pool_size=len(pool)),
        "risky_rank": avg_risky_rank(selected),
        "packing_ratio": packing_ratio(selected, pool),
        "top10_risk": top10_risk_ratio(selected),
        "late_promo": late_arrival_promotion_rate(selected, pool),
        "oldest_coverage": oldest_cov,
        "old_tx_pack_rate": oldest_cov,
        "starvation_gap": 1.0 - oldest_cov,
        "starvation_ratio": 1.0 - oldest_cov,
        "tail_wait_reduction": tail_wait_reduction(selected, pool, q=C.FAIR_TAIL_QUANTILE),
        "risky_inclusion_rate": risky_inclusion_rate(selected, pool),
    }
    if selected:
        waits = np.array([max(t_now - tx.arrival_time, 0.0) for tx in selected], dtype=np.float64)
        metrics["selected_wait_std"] = float(np.std(waits))
        metrics["wait_p95"] = float(np.quantile(waits, 0.95))
        metrics["wait_p99"] = float(np.quantile(waits, 0.99))
        mean_wait = float(np.mean(waits))
        if mean_wait < 1e-12:
            metrics["wait_gini"] = 0.0
        else:
            diff = np.abs(waits[:, None] - waits[None, :])
            metrics["wait_gini"] = float(diff.mean() / (2.0 * mean_wait))
    else:
        metrics["selected_wait_std"] = 0.0
        metrics["wait_p95"] = 0.0
        metrics["wait_p99"] = 0.0
        metrics["wait_gini"] = 0.0
    pool_fee = sum(tx.fee for tx in pool) if pool else 0.0
    metrics["block_fee_norm"] = metrics["block_fee"] / max(pool_fee, 1e-8)
    metrics["composite_score"] = composite_score(metrics, pool)
    metrics["constrained_fee_score"] = constrained_success_score(
        metrics,
        target_metric="block_fee_norm",
    )
    metrics["risk_adjusted_fee_score"] = risk_adjusted_fee_score(metrics)
    metrics["risk_safety_score"] = risk_safety_score(metrics)
    metrics["edge_risk_safety_score"] = edge_risk_safety_score(metrics)
    metrics["trade_score"] = multi_objective_trade_score(metrics, pool)
    metrics["risk_aware_trade_score"] = risk_aware_trade_score(metrics, pool)
    metrics["constrained_trade_score"] = constrained_trade_score(metrics, pool)
    return metrics
