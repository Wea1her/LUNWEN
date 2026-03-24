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


def constrained_success_score(metrics: dict,
                              constraints: dict | None = None,
                              target_metric: str = "block_fee_norm",
                              infeasible_score: float | None = -1.0) -> float | None:
    """约束式得分: 满足约束时返回目标指标，否则给不可行分。"""
    cons = constraints or {}
    fairness_floor = float(cons.get("fairness_floor", C.VALIDATION_FAIRNESS_FLOOR))
    oldest_floor = float(cons.get("oldest_coverage_floor", C.VALIDATION_OLDEST_COVERAGE_FLOOR))
    risk_ceil = float(cons.get("risk_ceil", C.VALIDATION_RISK_CEIL))
    top10_risk_ceil = float(cons.get("top10_risk_ceil", C.VALIDATION_TOP10_RISK_CEIL))

    fairness = float(metrics.get("fairness", 0.0))
    oldest_cov = float(metrics.get("oldest_coverage", 0.0))
    risk = float(metrics.get("risk_exposure", 1.0))
    top10_risk = float(metrics.get("top10_risk", 1.0))
    feasible = (
        fairness >= fairness_floor
        and oldest_cov >= oldest_floor
        and risk <= risk_ceil
        and top10_risk <= top10_risk_ceil
    )
    if not feasible:
        if infeasible_score is None:
            return None
        return float(infeasible_score)
    return float(metrics.get(target_metric, 0.0))


def violation_breakdown(metrics_seq: list[dict],
                        constraints: dict | None = None) -> dict:
    """统计每类约束违约次数。"""
    cons = constraints or {}
    fairness_floor = float(cons.get("fairness_floor", C.VALIDATION_FAIRNESS_FLOOR))
    oldest_floor = float(cons.get("oldest_coverage_floor", C.VALIDATION_OLDEST_COVERAGE_FLOOR))
    risk_ceil = float(cons.get("risk_ceil", C.VALIDATION_RISK_CEIL))
    top10_risk_ceil = float(cons.get("top10_risk_ceil", C.VALIDATION_TOP10_RISK_CEIL))

    breakdown = {
        "fairness_floor": 0,
        "oldest_coverage_floor": 0,
        "risk_ceil": 0,
        "top10_risk_ceil": 0,
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
        if float(metrics.get("top10_risk", 1.0)) > top10_risk_ceil:
            breakdown["top10_risk_ceil"] += 1
            violated = True
        if violated:
            breakdown["any_violation"] += 1
    return breakdown


def feasible_rate(metrics_seq: list[dict], constraints: dict | None = None) -> float:
    """可行样本占比。"""
    if not metrics_seq:
        return 0.0
    cons = constraints or {}
    scores = [
        constrained_success_score(m, constraints=cons, target_metric="block_fee_norm", infeasible_score=None)
        for m in metrics_seq
    ]
    feasible = [v for v in scores if v is not None]
    return len(feasible) / len(scores)


def feasible_set_fee_score(metrics_seq: list[dict],
                           constraints: dict | None = None,
                           target_metric: str = "block_fee_norm") -> dict:
    """仅在可行子集上计算收益统计；不可行样本不参与均值。"""
    cons = constraints or {}
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


def constrained_ranking(method_payload: dict[str, dict],
                        min_feasible_rate: float = C.CONSTRAINED_RANK_MIN_FEASIBLE_RATE) -> list[str]:
    """两段式 constrained 排序：先可行率档位，再可行域收益。"""
    scored = []
    for method, item in method_payload.items():
        fr = float(item.get("feasible_rate", 0.0))
        tier = 1 if fr >= min_feasible_rate else 0
        fee = item.get("feasible_fee_mean")
        fee_key = float("-inf") if fee is None else float(fee)
        risk_adj = float(item.get("risk_adjusted_fee_mean", float("-inf")))
        scored.append((method, tier, fr, fee_key, risk_adj))
    scored.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    return [m for m, *_ in scored]


def summary_metric_bundle(metrics_seq: list[dict],
                          constraints: dict | None = None,
                          target_metric: str = "block_fee_norm") -> dict:
    """统一输出 constrained 与解释性统计字段。"""
    cons = constraints or {}
    feasible_stats = feasible_set_fee_score(metrics_seq, constraints=cons, target_metric=target_metric)
    vb = violation_breakdown(metrics_seq, constraints=cons)
    all_fee_vals = [float(m.get(target_metric, 0.0)) for m in metrics_seq]
    all_risk_adjusted = [risk_adjusted_fee_score(m) for m in metrics_seq]
    top_violation = max(
        ("fairness_floor", "oldest_coverage_floor", "risk_ceil", "top10_risk_ceil"),
        key=lambda k: vb[k],
    ) if metrics_seq else "none"
    return {
        "score_policy_version": C.SCORE_POLICY_VERSION,
        "ranking_policy_version": C.RANKING_POLICY_VERSION,
        "feasible_rate": feasible_rate(metrics_seq, constraints=cons),
        "feasible_set_fee_mean": feasible_stats["feasible_fee_mean"],
        "feasible_set_fee_std": feasible_stats["feasible_fee_std"],
        "feasible_count": feasible_stats["feasible_count"],
        "infeasible_count": feasible_stats["infeasible_count"],
        "n_episodes": feasible_stats["n_episodes"],
        "all_episode_fee_mean": float(np.mean(all_fee_vals)) if all_fee_vals else 0.0,
        "all_episode_fee_std": float(np.std(all_fee_vals)) if all_fee_vals else 0.0,
        "risk_adjusted_fee_mean": float(np.mean(all_risk_adjusted)) if all_risk_adjusted else 0.0,
        "risk_adjusted_fee_std": float(np.std(all_risk_adjusted)) if all_risk_adjusted else 0.0,
        "violation_breakdown": vb,
        "constraint_violation_top1": top_violation,
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
        "gas_util": gas_utilization(selected, pool_size=len(pool)),
        "risky_rank": avg_risky_rank(selected),
        "packing_ratio": packing_ratio(selected, pool),
        "top10_risk": top10_risk_ratio(selected),
        "late_promo": late_arrival_promotion_rate(selected, pool),
        "oldest_coverage": oldest_cov,
        "starvation_gap": 1.0 - oldest_cov,
        "tail_wait_reduction": tail_wait_reduction(selected, pool, q=C.FAIR_TAIL_QUANTILE),
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
    return metrics
