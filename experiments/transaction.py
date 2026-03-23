"""交易数据结构与交易池生成"""

from __future__ import annotations
import dataclasses
import numpy as np
from typing import List

import config as C


@dataclasses.dataclass
class Transaction:
    tid: int
    fee: float            # Gwei
    gas: int
    arrival_time: float   # 秒
    tx_type: int           # 0=转账, 1=合约调用
    risk_score: float      # [0, 1]
    sender: int            # 发送地址编码
    nonce: int             # 同地址 nonce
    success_rate: float    # 历史成功率
    hist_delay: float      # 历史确认延迟 (归一化)

    def feature_vector(self, max_fee: float, max_gas: int, t_max: float) -> np.ndarray:
        """返回归一化后的 d=8 维特征向量"""
        return np.array([
            self.fee / max(max_fee, 1e-8),
            self.gas / max(max_gas, 1),
            self.arrival_time / max(t_max, 1e-8),
            float(self.tx_type == 0),   # is_transfer
            float(self.tx_type == 1),   # is_contract
            self.risk_score,
            self.success_rate,
            self.hist_delay,
        ], dtype=np.float32)


def generate_pool(rng: np.random.Generator,
                  pool_size: int | None = None,
                  risk_ratio: float = C.RISK_RATIO) -> List[Transaction]:
    """根据配置采样生成候选交易池。"""
    if C.POOL_REALISM_MODE == "iid_baseline":
        return _generate_pool_iid(rng, pool_size, risk_ratio)
    return _generate_pool_correlated_v1(rng, pool_size, risk_ratio)


def _generate_pool_iid(rng: np.random.Generator,
                       pool_size: int | None = None,
                       risk_ratio: float = C.RISK_RATIO) -> List[Transaction]:
    """旧版独立采样交易池（保留作为对照）。"""
    if pool_size is None:
        pool_size = rng.integers(C.POOL_SIZE_MIN, C.POOL_SIZE_MAX + 1)

    # 决定每笔交易是否为风险交易
    is_risk = rng.random(pool_size) < risk_ratio

    # 发送地址: 生成若干不同地址, 部分地址有多笔交易 (产生 nonce 依赖)
    n_senders = max(pool_size // 3, 1)
    senders = rng.integers(0, n_senders, size=pool_size)

    # 按 sender 分组分配 nonce
    sender_nonce_counter: dict[int, int] = {}
    nonces = np.zeros(pool_size, dtype=int)
    for i in range(pool_size):
        s = int(senders[i])
        nonces[i] = sender_nonce_counter.get(s, 0)
        sender_nonce_counter[s] = nonces[i] + 1

    # 到达时间: 泊松过程累积
    intervals = rng.exponential(1.0 / C.ARRIVAL_RATE, size=pool_size)
    arrivals = np.cumsum(intervals)

    txs: List[Transaction] = []
    for i in range(pool_size):
        # 交易类型
        tx_type = 1 if rng.random() < 0.6 else 0  # 60% 合约调用
        gas = C.GAS_TRANSFER if tx_type == 0 else int(
            rng.integers(C.GAS_CONTRACT_MIN, C.GAS_CONTRACT_MAX + 1))

        # 手续费
        fee = float(rng.lognormal(C.FEE_LOG_MEAN, C.FEE_LOG_STD))
        if is_risk[i]:
            fee *= C.RISK_FEE_MULTIPLIER

        # 风险评分
        if is_risk[i]:
            risk = float(rng.uniform(*C.RISK_SCORE_HIGH))
        else:
            risk = float(rng.uniform(*C.RISK_SCORE_LOW))

        txs.append(Transaction(
            tid=i,
            fee=fee,
            gas=gas,
            arrival_time=float(arrivals[i]),
            tx_type=tx_type,
            risk_score=risk,
            sender=int(senders[i]),
            nonce=int(nonces[i]),
            success_rate=float(rng.uniform(0.7, 1.0)),
            hist_delay=float(rng.uniform(0.0, 1.0)),
        ))
    return txs


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _generate_pool_correlated_v1(rng: np.random.Generator,
                                 pool_size: int | None = None,
                                 risk_ratio: float = C.RISK_RATIO) -> List[Transaction]:
    """相关性增强版交易池:

    - 到达时间受市场拥堵状态影响（高拥堵 -> 到达更密集）；
    - 风险概率与拥堵状态相关；
    - 手续费与风险、延迟、拥堵强度存在正相关；
    - success_rate 与风险/延迟负相关。
    """
    if pool_size is None:
        pool_size = int(rng.integers(C.POOL_SIZE_MIN, C.POOL_SIZE_MAX + 1))

    # 市场拥堵强度（按片段波动），用于耦合 arrival / risk / fee
    segment_len = max(pool_size // 10, 8)
    levels = np.array([0.75, 1.0, 1.35], dtype=np.float64)
    segment_count = int(np.ceil(pool_size / segment_len))
    segment_ids = np.repeat(np.arange(segment_count), segment_len)[:pool_size]
    market_levels = rng.choice(levels, size=segment_count, p=[0.25, 0.5, 0.25])
    market_state = market_levels[segment_ids]

    # 风险标签概率与拥堵状态相关
    risk_prob = np.clip(risk_ratio * (0.85 + 0.45 * (market_state - 1.0)), 0.01, 0.95)
    is_risk = rng.random(pool_size) < risk_prob

    # 发送地址与 nonce
    n_senders = max(pool_size // 3, 1)
    senders = rng.integers(0, n_senders, size=pool_size)
    sender_nonce_counter: dict[int, int] = {}
    nonces = np.zeros(pool_size, dtype=int)
    for i in range(pool_size):
        s = int(senders[i])
        nonces[i] = sender_nonce_counter.get(s, 0)
        sender_nonce_counter[s] = nonces[i] + 1

    # 到达过程：拥堵越高，间隔越短
    intervals = np.zeros(pool_size, dtype=np.float64)
    for i in range(pool_size):
        lam = max(C.ARRIVAL_RATE * float(market_state[i]), 1e-3)
        intervals[i] = rng.exponential(1.0 / lam)
    arrivals = np.cumsum(intervals)

    txs: List[Transaction] = []
    for i in range(pool_size):
        congest = float(market_state[i] - 1.0)  # [-0.25, 0.35] 附近
        tx_type_contract_prob = _clip01(0.58 + 0.08 * congest + (0.06 if is_risk[i] else -0.03))
        tx_type = 1 if rng.random() < tx_type_contract_prob else 0
        gas = C.GAS_TRANSFER if tx_type == 0 else int(
            rng.integers(C.GAS_CONTRACT_MIN, C.GAS_CONTRACT_MAX + 1)
        )

        if is_risk[i]:
            risk = float(rng.uniform(*C.RISK_SCORE_HIGH))
        else:
            # 普通交易在高拥堵下也可能略抬升风险分
            base = rng.uniform(*C.RISK_SCORE_LOW)
            risk = float(min(base + max(congest, 0.0) * 0.05, C.RISK_SCORE_HIGH[0] - 1e-3))

        # 历史延迟与拥堵、风险正相关
        hist_delay = _clip01(
            rng.normal(
                loc=0.3 + 0.35 * max(congest, 0.0) + 0.35 * risk,
                scale=0.15,
            )
        )

        # 手续费受拥堵、风险、历史延迟共同影响
        fee = float(
            rng.lognormal(
                mean=C.FEE_LOG_MEAN + 0.22 * congest + 0.25 * risk + 0.20 * hist_delay,
                sigma=max(C.FEE_LOG_STD * (0.90 + 0.20 * hist_delay), 1e-3),
            )
        )
        if is_risk[i]:
            # 风险交易不再使用单一固定倍率，倍率随风险与延迟浮动
            dynamic_mult = 1.0 + (C.RISK_FEE_MULTIPLIER - 1.0) * (0.45 + 0.35 * risk + 0.20 * hist_delay)
            fee *= dynamic_mult

        success_rate = _clip01(
            rng.normal(
                loc=0.95 - 0.25 * risk - 0.20 * hist_delay + 0.05 * max(-congest, 0.0),
                scale=0.06,
            )
        )

        txs.append(Transaction(
            tid=i,
            fee=fee,
            gas=gas,
            arrival_time=float(arrivals[i]),
            tx_type=tx_type,
            risk_score=risk,
            sender=int(senders[i]),
            nonce=int(nonces[i]),
            success_rate=success_rate,
            hist_delay=float(hist_delay),
        ))
    return txs
