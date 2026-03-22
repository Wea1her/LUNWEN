"""交易排序仿真环境 (Gymnasium 兼容)"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any

import config as C
from transaction import Transaction, generate_pool


class TxOrderingEnv(gym.Env):
    """
    将一次区块构建建模为 episode:
      state  = (候选交易特征矩阵, 区块状态向量)
      action = 从合法候选集中选一笔交易的索引, 或 STOP
      reward = α·r_fee + β·r_fair − γ·r_risk
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 pool_size: int | None = None,
                 risk_ratio: float = C.RISK_RATIO,
                 max_pool: int = C.POOL_SIZE_MAX,
                 seed: int | None = None,
                 alpha: float = C.ALPHA,
                 beta: float = C.BETA,
                 gamma_r: float = C.GAMMA_R,
                 eta: float = C.ETA,
                 no_seq_summary: bool = False,
                 no_stop: bool = False,
                 no_action_mask: bool = False):
        super().__init__()
        self.pool_size = pool_size
        self.risk_ratio = risk_ratio
        self.max_pool = max_pool
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.beta = beta
        self.gamma_r = gamma_r
        self.eta = eta
        self.no_seq_summary = no_seq_summary
        self.no_stop = no_stop
        self.no_action_mask = no_action_mask

        # 观测 / 动作空间 (用 dict 传递变长数据)
        self.observation_space = spaces.Dict({
            "tx_features": spaces.Box(-np.inf, np.inf,
                                      shape=(max_pool, C.TX_FEATURE_DIM), dtype=np.float32),
            "block_state": spaces.Box(-np.inf, np.inf,
                                      shape=(3 + C.HIDDEN_DIM,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(max_pool + 1),
            "num_candidates": spaces.Discrete(max_pool + 1),
        })
        # 动作: 0 ~ max_pool-1 选交易, max_pool = STOP
        self.action_space = spaces.Discrete(max_pool + 1)

        # episode 状态
        self._pool: list[Transaction] = []
        self._candidates: list[Transaction] = []
        self._selected: list[Transaction] = []
        self._remaining_gas = C.MAX_BLOCK_GAS
        self._acc_fee = 0.0
        self._step_count = 0
        self._t_now = 0.0
        self._max_fee = 1.0
        self._max_gas = 1
        self._t_max = 1.0
        self._done = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._pool = generate_pool(self.rng, self.pool_size, self.risk_ratio)
        self._candidates = list(self._pool)
        self._selected = []
        self._remaining_gas = C.MAX_BLOCK_GAS
        self._acc_fee = 0.0
        self._step_count = 0
        self._done = False

        self._max_fee = max(tx.fee for tx in self._pool) if self._pool else 1.0
        self._max_gas = max(tx.gas for tx in self._pool) if self._pool else 1
        self._t_max = max(tx.arrival_time for tx in self._pool) if self._pool else 1.0
        self._t_now = self._t_max + 1.0

        return self._obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._done:
            return self._obs(), 0.0, True, False, {}

        valid = self._valid_indices()
        stop_idx = len(self._candidates)

        # STOP 动作
        if action == stop_idx:
            stop_penalty = 0.0
            if len(valid) > 0:
                remaining_fees = [self._candidates[i].fee for i in valid]
                avg_remaining = sum(remaining_fees) / (len(remaining_fees) * max(self._max_fee, 1e-8))
                import math
                stop_penalty = -self.eta * (1.0 + avg_remaining * math.log1p(len(valid)))
            self._done = True
            return self._obs(), stop_penalty, True, False, self._info()

        # 无效动作: 不终止, 施加小惩罚, 状态不变
        if action not in valid:
            return self._obs(), -self.eta, False, False, self._info()

        tx = self._candidates[action]

        # 计算即时奖励
        reward = self._compute_reward(tx)

        # 更新状态
        self._selected.append(tx)
        self._remaining_gas -= tx.gas
        self._acc_fee += tx.fee
        self._step_count += 1
        self._candidates.pop(action)

        # 检查终止
        if not self._candidates or not self._valid_indices():
            self._done = True

        return self._obs(), reward, self._done, False, self._info()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _valid_indices(self) -> set[int]:
        """返回当前候选列表中满足合法性约束的交易索引集合"""
        # 已选交易的 (sender, max_nonce) 记录
        selected_nonces: dict[int, int] = {}
        for tx in self._selected:
            prev = selected_nonces.get(tx.sender, -1)
            selected_nonces[tx.sender] = max(prev, tx.nonce)

        valid = set()
        for idx, tx in enumerate(self._candidates):
            # 容量约束
            if tx.gas > self._remaining_gas:
                continue
            # Nonce 依赖约束: 同地址前序 nonce 必须已被选入
            if tx.nonce > 0:
                prev_max = selected_nonces.get(tx.sender, -1)
                if prev_max < tx.nonce - 1:
                    continue
            valid.add(idx)
        return valid

    def _jain_index(self, waits: np.ndarray) -> float:
        """计算等待时间序列的 Jain 公平性指数"""
        if len(waits) == 0:
            return 1.0
        s = waits.sum()
        s2 = (waits ** 2).sum()
        if s2 < 1e-12:
            return 1.0
        return float(s ** 2 / (len(waits) * s2))

    def _compute_reward(self, tx: Transaction) -> float:
        """r = α·r_fee + β·Δr_fair − γ·r_risk"""
        # 手续费收益
        r_fee = tx.fee / self._max_fee

        # 增量 Jain 公平性: J(S_t ∪ {tx}) - J(S_t)
        old_waits = np.array([self._t_now - s.arrival_time
                              for s in self._selected], dtype=np.float64)
        new_wait = self._t_now - tx.arrival_time
        new_waits = np.append(old_waits, new_wait)
        delta_fair = self._jain_index(new_waits) - self._jain_index(old_waits)

        # 分段式风险惩罚 (基于实际打包进度, 非池大小)
        avg_gas = sum(tx.gas for tx in self._pool) / max(len(self._pool), 1)
        estimated_block_size = max(int(C.MAX_BLOCK_GAS / avg_gas), 1) if avg_gas > 0 else len(self._pool)
        if estimated_block_size <= 1:
            pos_ratio = 0.5
        else:
            pos_ratio = self._step_count / estimated_block_size
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            phi = 1.0      # 头尾 10%: 强惩罚
        elif 0.4 <= pos_ratio <= 0.6:
            phi = 0.0      # 中间 40%-60%: 无惩罚
        else:
            phi = 0.5      # 其余位置: 中等惩罚
        r_risk = tx.risk_score * phi

        return self.alpha * r_fee + self.beta * delta_fair - self.gamma_r * r_risk

    def _obs(self) -> dict:
        """构造 padded 观测"""
        n = len(self._candidates)
        features = np.zeros((self.max_pool, C.TX_FEATURE_DIM), dtype=np.float32)
        for i, tx in enumerate(self._candidates):
            features[i] = tx.feature_vector(self._max_fee, self._max_gas, self._t_max)

        # 区块状态
        rem_gas_norm = self._remaining_gas / C.MAX_BLOCK_GAS
        n_sel_norm = len(self._selected) / max(len(self._pool), 1)
        acc_fee_norm = self._acc_fee / (self._max_fee * max(len(self._pool), 1))

        # 已选交易序列摘要 (均值池化)
        if self._selected and not self.no_seq_summary:
            sel_feats = np.stack([
                tx.feature_vector(self._max_fee, self._max_gas, self._t_max)
                for tx in self._selected
            ])
            seq_summary = np.zeros(C.HIDDEN_DIM, dtype=np.float32)
            mean_feat = sel_feats.mean(axis=0)
            seq_summary[:len(mean_feat)] = mean_feat
        else:
            seq_summary = np.zeros(C.HIDDEN_DIM, dtype=np.float32)

        block_state = np.concatenate([
            np.array([rem_gas_norm, n_sel_norm, acc_fee_norm], dtype=np.float32),
            seq_summary,
        ])

        # 动作掩码
        mask = np.zeros(self.max_pool + 1, dtype=np.int8)
        if self.no_action_mask:
            # 消融: 所有候选均标记为合法, 但 env 仍通过 _valid_indices 强制约束
            mask[:n] = 1
        else:
            valid = self._valid_indices()
            for idx in valid:
                mask[idx] = 1
        # STOP 掩码: 早期强制禁用, 要求至少打包一定比例后才允许 STOP
        min_steps_before_stop = max(int(len(self._pool) * 0.3), 10)
        if self.no_stop:
            mask[n] = 0
        elif self._step_count < min_steps_before_stop:
            mask[n] = 0  # 打包不足 30% 前禁止 STOP
        else:
            mask[n] = 1

        return {
            "tx_features": features,
            "block_state": block_state,
            "action_mask": mask,
            "num_candidates": n,
        }

    def _info(self) -> dict:
        return {
            "selected": [tx.tid for tx in self._selected],
            "total_fee": self._acc_fee,
            "gas_used": C.MAX_BLOCK_GAS - self._remaining_gas,
            "num_selected": len(self._selected),
            "pool_size": len(self._pool),
        }

    # ------------------------------------------------------------------
    # 公开辅助
    # ------------------------------------------------------------------

    def get_selected_transactions(self) -> list[Transaction]:
        return list(self._selected)

    def get_pool(self) -> list[Transaction]:
        return list(self._pool)
