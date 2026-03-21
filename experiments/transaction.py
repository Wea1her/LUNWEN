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
    """根据论文参数分布采样生成候选交易池"""
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
