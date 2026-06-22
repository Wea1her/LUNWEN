"""交易排序仿真环境 (Gymnasium 兼容)"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C
from transaction import Transaction, generate_pool


class TxOrderingEnv(gym.Env):
    """
    将一次区块构建建模为 episode:
      state  = (候选交易特征矩阵, 区块状态向量)
      action = 从合法候选集中选一笔交易的索引, 或 STOP
      reward = fee + age-aware + oldest-cover-aware - risk
               + terminal_fairness - terminal_starvation
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pool_size: int | None = None,
        risk_ratio: float = C.RISK_RATIO,
        max_pool: int = C.POOL_SIZE_MAX,
        seed: int | None = None,
        alpha: float = C.ALPHA,
        beta: float = C.BETA,  # 兼容旧参数，映射到 beta_terminal_fair
        gamma_r: float = C.GAMMA_R,
        eta: float = C.ETA,
        beta_age: float = C.BETA_AGE,
        beta_oldest_cover: float = C.BETA_OLDEST_COVER,
        beta_terminal_fair: float | None = None,
        gamma_starvation: float = C.GAMMA_STARVATION,
        fairness_gate_type: str = C.FAIRNESS_GATE_TYPE,
        fairness_gate_threshold: float = C.FAIRNESS_GATE_THRESHOLD,
        fairness_gate_sharpness: float = C.FAIRNESS_GATE_SHARPNESS,
        fairness_gate_min: float = C.FAIRNESS_GATE_MIN,
        fairness_oldest_coverage_floor: float = C.FAIRNESS_OLDEST_COVERAGE_FLOOR,
        fairness_starvation_ceil: float = C.FAIRNESS_STARVATION_CEIL,
        packing_reward_weight: float = C.PACKING_REWARD_WEIGHT,
        late_fee_recovery_weight: float = C.LATE_FEE_RECOVERY_WEIGHT,
        unused_gas_penalty_weight: float = C.UNUSED_GAS_PENALTY_WEIGHT,
        terminal_risk_exposure_weight: float = C.TERMINAL_RISK_EXPOSURE_WEIGHT,
        terminal_top10_risk_weight: float = C.TERMINAL_TOP10_RISK_WEIGHT,
        terminal_risky_rank_dev_weight: float = C.TERMINAL_RISKY_RANK_DEV_WEIGHT,
        no_seq_summary: bool = False,
        no_stop: bool = False,
        no_action_mask: bool = False,
    ):
        super().__init__()
        if pool_size is not None:
            C.validate_pool_size(pool_size)
        self.pool_size = pool_size
        self.risk_ratio = risk_ratio
        self.max_pool = max_pool
        self.rng = np.random.default_rng(seed)

        self.alpha = alpha
        self.beta = beta
        self.gamma_r = gamma_r
        self.eta = eta
        self.beta_age = beta_age
        self.beta_oldest_cover = beta_oldest_cover
        self.beta_terminal_fair = beta if beta_terminal_fair is None else beta_terminal_fair
        self.gamma_starvation = gamma_starvation
        self.fairness_gate_type = fairness_gate_type
        self.fairness_gate_threshold = fairness_gate_threshold
        self.fairness_gate_sharpness = fairness_gate_sharpness
        self.fairness_gate_min = fairness_gate_min
        self.fairness_oldest_coverage_floor = fairness_oldest_coverage_floor
        self.fairness_starvation_ceil = fairness_starvation_ceil
        self.packing_reward_weight = packing_reward_weight
        self.late_fee_recovery_weight = late_fee_recovery_weight
        self.unused_gas_penalty_weight = unused_gas_penalty_weight
        self.terminal_risk_exposure_weight = terminal_risk_exposure_weight
        self.terminal_top10_risk_weight = terminal_top10_risk_weight
        self.terminal_risky_rank_dev_weight = terminal_risky_rank_dev_weight

        self.no_seq_summary = no_seq_summary
        self.no_stop = no_stop
        self.no_action_mask = no_action_mask

        # 观测 / 动作空间 (用 dict 传递变长数据)
        self.observation_space = spaces.Dict({
            "tx_features": spaces.Box(
                -np.inf,
                np.inf,
                shape=(max_pool, C.TX_FEATURE_DIM),
                dtype=np.float32,
            ),
            "block_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(C.BLOCK_STATE_DIM,),
                dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary(max_pool + 1),
            "num_candidates": spaces.Discrete(max_pool + 1),
        })
        # 动作: 0 ~ max_pool-1 选交易, max_pool = STOP
        self.action_space = spaces.Discrete(max_pool + 1)

        # episode 状态
        self._pool: list[Transaction] = []
        self._candidates: list[Transaction] = []
        self._selected: list[Transaction] = []
        self._block_gas_limit = C.MAX_BLOCK_GAS
        self._remaining_gas = self._block_gas_limit
        self._acc_fee = 0.0
        self._step_count = 0
        self._t_now = 0.0
        self._t_max = 1.0
        self._min_arrival = 0.0
        self._median_arrival = 0.0
        self._median_fee = 0.0
        self._wait_norm_denom = 1.0
        self._max_fee = 1.0
        self._max_gas = 1
        self._done = False
        self._invalid_action_streak = 0
        self._terminal_reward_applied = False
        self._pool_oldest_ids: set[int] = set()
        self._pool_late_high_ids: set[int] = set()

        # reward 分解跟踪（按加权贡献累计）
        self._proxy_fee_reward = 0.0
        self._proxy_age_reward = 0.0
        self._proxy_oldest_cover_reward = 0.0
        self._proxy_risk_penalty = 0.0
        self._proxy_fairness_gate = 1.0
        self._proxy_packing_reward = 0.0
        self._proxy_late_fee_recovery = 0.0
        self._proxy_terminal_fair = 0.0
        self._proxy_starvation_penalty = 0.0
        self._proxy_terminal_risk_penalty = 0.0
        self._proxy_unused_gas_penalty = 0.0
        self._proxy_stop_penalty = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._pool = generate_pool(self.rng, self.pool_size, self.risk_ratio)
        return self._reset_from_current_pool(), {}

    def reset_with_pool(self, pool: list[Transaction]) -> tuple[dict, dict]:
        """使用给定交易池重置环境，便于方法间公平对比。"""
        self._pool = deepcopy(pool)
        return self._reset_from_current_pool(), {}

    def _reset_from_current_pool(self) -> dict:
        self._candidates = list(self._pool)
        self._selected = []
        self._block_gas_limit = C.effective_block_gas_limit(len(self._pool))
        self._remaining_gas = self._block_gas_limit
        self._acc_fee = 0.0
        self._step_count = 0
        self._done = False
        self._invalid_action_streak = 0
        self._terminal_reward_applied = False

        self._max_fee = max(tx.fee for tx in self._pool) if self._pool else 1.0
        self._max_gas = max(tx.gas for tx in self._pool) if self._pool else 1
        self._t_max = max(tx.arrival_time for tx in self._pool) if self._pool else 1.0
        self._min_arrival = min(tx.arrival_time for tx in self._pool) if self._pool else 0.0
        self._median_arrival = float(np.median([tx.arrival_time for tx in self._pool])) if self._pool else 0.0
        self._median_fee = float(np.median([tx.fee for tx in self._pool])) if self._pool else 0.0
        self._t_now = self._t_max + 1.0
        self._wait_norm_denom = max(self._t_now - self._min_arrival, 1e-8)
        self._pool_oldest_ids = self._oldest_ids(self._pool, C.FAIR_OLDEST_RATIO)
        self._pool_late_high_ids = {
            tx.tid for tx in self._pool
            if tx.arrival_time > self._median_arrival and tx.fee > self._median_fee
        }

        self._proxy_fee_reward = 0.0
        self._proxy_age_reward = 0.0
        self._proxy_oldest_cover_reward = 0.0
        self._proxy_risk_penalty = 0.0
        self._proxy_fairness_gate = 1.0
        self._proxy_packing_reward = 0.0
        self._proxy_late_fee_recovery = 0.0
        self._proxy_terminal_fair = 0.0
        self._proxy_starvation_penalty = 0.0
        self._proxy_terminal_risk_penalty = 0.0
        self._proxy_unused_gas_penalty = 0.0
        self._proxy_stop_penalty = 0.0

        return self._obs()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self._done:
            return self._obs(), 0.0, True, False, {}

        valid = self._valid_indices()
        stop_idx = len(self._candidates)

        # STOP 动作
        if action == stop_idx:
            stop_penalty = self._compute_stop_penalty(valid)
            self._proxy_stop_penalty += stop_penalty
            terminal_reward = self._apply_terminal_reward()
            self._done = True
            reward = stop_penalty + terminal_reward
            return self._obs(), reward, True, False, self._info()

        # 无效动作: 不终止, 施加小惩罚, 状态不变
        if action not in valid:
            self._invalid_action_streak += 1
            penalty = -self.eta * (1.0 + 0.1 * self._invalid_action_streak)
            if self.no_action_mask or self._invalid_action_streak >= max(len(self._candidates), 10):
                terminal_reward = self._apply_terminal_reward()
                self._done = True
                return self._obs(), penalty + terminal_reward, True, False, self._info()
            return self._obs(), penalty, False, False, self._info()

        tx = self._candidates[action]
        self._invalid_action_streak = 0

        # 计算即时奖励
        reward = self._compute_step_reward(tx)

        # 更新状态
        self._selected.append(tx)
        self._remaining_gas -= tx.gas
        self._acc_fee += tx.fee
        self._step_count += 1
        self._candidates.pop(action)

        # 检查终止
        if not self._candidates or not self._valid_indices():
            self._done = True
            reward += self._apply_terminal_reward()

        return self._obs(), reward, self._done, False, self._info()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _valid_indices(self) -> set[int]:
        """返回当前候选列表中满足合法性约束的交易索引集合。"""
        selected_nonces: dict[int, int] = {}
        for tx in self._selected:
            prev = selected_nonces.get(tx.sender, -1)
            selected_nonces[tx.sender] = max(prev, tx.nonce)

        valid = set()
        for idx, tx in enumerate(self._candidates):
            if tx.gas > self._remaining_gas:
                continue
            if tx.nonce > 0:
                prev_max = selected_nonces.get(tx.sender, -1)
                if prev_max < tx.nonce - 1:
                    continue
            valid.add(idx)
        return valid

    def _jain_index(self, waits: np.ndarray) -> float:
        """计算等待时间序列的 Jain 公平性指数。"""
        if len(waits) == 0:
            return 1.0
        s = waits.sum()
        s2 = (waits ** 2).sum()
        if s2 < 1e-12:
            return 1.0
        return float((s ** 2) / (len(waits) * s2))

    def _wait(self, tx: Transaction) -> float:
        return max(self._t_now - tx.arrival_time, 0.0)

    def _oldest_ids(self, txs: list[Transaction], q: float) -> set[int]:
        if not txs:
            return set()
        k = max(int(len(txs) * q), 1)
        oldest = sorted(txs, key=lambda tx: tx.arrival_time)[:k]
        return {tx.tid for tx in oldest}

    def _oldest_unserved_ratio(self) -> float:
        if not self._pool_oldest_ids:
            return 0.0
        selected_ids = {tx.tid for tx in self._selected}
        unserved = len([tid for tid in self._pool_oldest_ids if tid not in selected_ids])
        return unserved / len(self._pool_oldest_ids)

    def _oldest_coverage_ratio(self) -> float:
        if not self._pool_oldest_ids:
            return 0.0
        selected_ids = {tx.tid for tx in self._selected}
        served = len([tid for tid in self._pool_oldest_ids if tid in selected_ids])
        return served / len(self._pool_oldest_ids)

    def _oldest_wait_mass_norm(self) -> float:
        if not self._pool_oldest_ids:
            return 0.0
        selected_ids = {tx.tid for tx in self._selected}
        unserved_wait = 0.0
        for tx in self._pool:
            if tx.tid in self._pool_oldest_ids and tx.tid not in selected_ids:
                unserved_wait += self._wait(tx)
        denom = max(len(self._pool_oldest_ids) * self._wait_norm_denom, 1e-8)
        return unserved_wait / denom

    def _sender_starvation_stats(self) -> tuple[float, float]:
        if not self._pool:
            return 0.0, 0.0
        selected_ids = {tx.tid for tx in self._selected}
        sender_total: dict[int, int] = {}
        sender_unserved: dict[int, int] = {}
        for tx in self._pool:
            sender_total[tx.sender] = sender_total.get(tx.sender, 0) + 1
            if tx.tid not in selected_ids:
                sender_unserved[tx.sender] = sender_unserved.get(tx.sender, 0) + 1
        ratios = []
        for sender, total in sender_total.items():
            unserved = sender_unserved.get(sender, 0)
            ratios.append(unserved / max(total, 1))
        if not ratios:
            return 0.0, 0.0
        return float(max(ratios)), float(np.mean(ratios))

    def _fairness_proxy(self) -> float:
        if not self._selected:
            return 0.0
        waits = np.array([self._wait(tx) for tx in self._selected], dtype=np.float64)
        return self._jain_index(waits)

    def _fairness_gate(self) -> float:
        fair = self._fairness_proxy()
        oldest_cov = self._oldest_coverage_ratio()
        starvation = self._oldest_unserved_ratio()
        feasible = (
            fair >= self.fairness_gate_threshold
            and oldest_cov >= self.fairness_oldest_coverage_floor
            and starvation <= self.fairness_starvation_ceil
        )
        if self.fairness_gate_type == "hard":
            return self.fairness_gate_min if feasible else 1.0

        # sigmoid: 达标后逐步收缩公平项权重；未达标时保持接近 1
        x = self.fairness_gate_sharpness * (fair - self.fairness_gate_threshold)
        gate = self.fairness_gate_min + (1.0 - self.fairness_gate_min) * (1.0 / (1.0 + math.exp(x)))
        if not feasible:
            gate = max(gate, 0.75)
        return float(min(max(gate, self.fairness_gate_min), 1.0))

    def _risk_exposure_selected(self) -> float:
        if not self._selected:
            return 0.0
        k = len(self._selected)
        n_edge = max(int(k * C.RISK_POSITION_RATIO), 1)
        idx = set(range(n_edge)) | set(range(max(k - n_edge, 0), k))
        sensitive = [self._selected[i] for i in idx]
        risky_sensitive = sum(1 for tx in sensitive if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD)
        total_risky = sum(1 for tx in self._selected if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD)
        if total_risky == 0:
            return 0.0
        return risky_sensitive / total_risky

    def _top10_risk_ratio_selected(self) -> float:
        if not self._selected:
            return 0.0
        k = len(self._selected)
        n_top = max(int(k * 0.1), 1)
        top = self._selected[:n_top]
        risky = sum(1 for tx in top if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD)
        return risky / n_top

    def _edge_risk_ratio_selected(self) -> float:
        if not self._selected:
            return 0.0
        k = len(self._selected)
        n_edge = max(int(k * C.RISK_POSITION_RATIO), 1)
        idx = set(range(n_edge)) | set(range(max(k - n_edge, 0), k))
        sensitive = [self._selected[i] for i in idx]
        if not sensitive:
            return 0.0
        risky_sensitive = sum(1 for tx in sensitive if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD)
        return risky_sensitive / len(sensitive)

    def _avg_risky_rank_selected(self) -> float:
        if len(self._selected) <= 1:
            return 0.5
        pos = [
            i / (len(self._selected) - 1)
            for i, tx in enumerate(self._selected)
            if tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD
        ]
        if not pos:
            return 0.5
        return float(np.mean(pos))

    def _remaining_fee_mass_norm(self) -> float:
        selected_ids = {tx.tid for tx in self._selected}
        rem_fee = sum(tx.fee for tx in self._pool if tx.tid not in selected_ids)
        total_fee = sum(tx.fee for tx in self._pool)
        return rem_fee / max(total_fee, 1e-8)

    def _late_high_fee_unserved_ratio(self) -> float:
        if not self._pool_late_high_ids:
            return 0.0
        selected_ids = {tx.tid for tx in self._selected}
        unserved = len([tid for tid in self._pool_late_high_ids if tid not in selected_ids])
        return unserved / len(self._pool_late_high_ids)

    def _risk_position_penalty(self, pos_ratio: float) -> float:
        """越接近两端惩罚越大，中心位置惩罚越小。"""
        dist = abs(pos_ratio - C.RISK_CENTER)
        sigma = max(C.RISK_POSITION_SIGMA, 1e-6)
        return float(1.0 - math.exp(-0.5 * (dist / sigma) ** 2))

    def _compute_step_reward(self, tx: Transaction) -> float:
        # fee 收益
        r_fee = tx.fee / max(self._max_fee, 1e-8)

        # age-aware 奖励
        r_age = self._wait(tx) / self._wait_norm_denom

        # oldest-q 覆盖奖励
        oldest_candidate_ids = self._oldest_ids(self._candidates, C.FAIR_OLDEST_RATIO)
        r_oldest_cover = 1.0 if tx.tid in oldest_candidate_ids else 0.0

        # 平滑风险位置惩罚
        avg_gas = sum(item.gas for item in self._pool) / max(len(self._pool), 1)
        est_block_size = max(int(self._block_gas_limit / avg_gas), 1) if avg_gas > 0 else len(self._pool)
        pos_ratio = self._step_count / max(est_block_size - 1, 1)
        r_risk = tx.risk_score * self._risk_position_penalty(pos_ratio)
        fairness_gate = self._fairness_gate()

        # packing 激励（弱项，避免主导训练）
        r_packing = tx.gas / max(self._block_gas_limit, 1)

        # late-high-fee recovery（仅作弱激励，并按风险折减）
        is_late_high = (tx.arrival_time > self._median_arrival) and (tx.fee > self._median_fee)
        r_late_fee = (1.0 - tx.risk_score) if is_late_high else 0.0

        fee_contrib = self.alpha * r_fee
        age_contrib = fairness_gate * self.beta_age * r_age
        oldest_contrib = fairness_gate * self.beta_oldest_cover * r_oldest_cover
        risk_contrib = -self.gamma_r * r_risk
        packing_contrib = self.packing_reward_weight * r_packing
        late_fee_contrib = self.late_fee_recovery_weight * r_late_fee

        self._proxy_fee_reward += fee_contrib
        self._proxy_age_reward += age_contrib
        self._proxy_oldest_cover_reward += oldest_contrib
        self._proxy_risk_penalty += risk_contrib
        self._proxy_fairness_gate = fairness_gate
        self._proxy_packing_reward += packing_contrib
        self._proxy_late_fee_recovery += late_fee_contrib
        return (
            fee_contrib
            + age_contrib
            + oldest_contrib
            + risk_contrib
            + packing_contrib
            + late_fee_contrib
        )

    def _compute_stop_penalty(self, valid_indices: set[int]) -> float:
        if not self._pool:
            return 0.0
        if valid_indices:
            remaining_fees = [self._candidates[i].fee for i in valid_indices]
            avg_remaining_fee_norm = float(np.mean(remaining_fees)) / max(self._max_fee, 1e-8)
        else:
            avg_remaining_fee_norm = 0.0

        oldest_unserved = self._oldest_unserved_ratio()
        oldest_wait_mass = self._oldest_wait_mass_norm()
        packing_ratio = len(self._selected) / max(len(self._pool), 1)
        packing_gap = 1.0 - packing_ratio
        unused_gas_ratio = self._remaining_gas / max(self._block_gas_limit, 1)
        late_high_unserved = self._late_high_fee_unserved_ratio()
        remaining_fee_mass_norm = self._remaining_fee_mass_norm()

        penalty_strength = (
            C.STOP_FEE_WEIGHT * (0.5 * avg_remaining_fee_norm + 0.5 * remaining_fee_mass_norm)
            + C.STOP_OLDEST_UNSERVED_WEIGHT * oldest_unserved
            + C.STOP_OLDEST_WAIT_WEIGHT * oldest_wait_mass
            + C.STOP_PACKING_WEIGHT * packing_gap
            + C.STOP_UNUSED_GAS_WEIGHT * unused_gas_ratio
            + C.STOP_LATE_HIGH_FEE_UNSERVED_WEIGHT * late_high_unserved
        )
        if self._fairness_gate() <= (self.fairness_gate_min + 0.05):
            # fairness 已达标后，进一步抑制“早停躺平”
            penalty_strength *= 1.15
        return -self.eta * penalty_strength

    def _apply_terminal_reward(self) -> float:
        if self._terminal_reward_applied:
            return 0.0
        self._terminal_reward_applied = True

        if self._selected:
            waits = np.array([self._wait(tx) for tx in self._selected], dtype=np.float64)
            terminal_fair = self._jain_index(waits)
        else:
            terminal_fair = 0.0
        starvation = self._oldest_unserved_ratio()
        fairness_gate = self._fairness_gate()

        terminal_fair_contrib = fairness_gate * self.beta_terminal_fair * terminal_fair
        starvation_penalty = -fairness_gate * self.gamma_starvation * starvation

        terminal_risk_exposure = self._risk_exposure_selected()
        terminal_top10_risk = self._top10_risk_ratio_selected()
        terminal_risky_rank_dev = abs(self._avg_risky_rank_selected() - 0.5)
        terminal_risk_penalty = -(
            self.terminal_risk_exposure_weight * terminal_risk_exposure
            + self.terminal_top10_risk_weight * terminal_top10_risk
            + self.terminal_risky_rank_dev_weight * terminal_risky_rank_dev
        )
        unused_gas_penalty = -self.unused_gas_penalty_weight * (
            self._remaining_gas / max(self._block_gas_limit, 1)
        )

        self._proxy_terminal_fair += terminal_fair_contrib
        self._proxy_starvation_penalty += starvation_penalty
        self._proxy_terminal_risk_penalty += terminal_risk_penalty
        self._proxy_unused_gas_penalty += unused_gas_penalty
        self._proxy_fairness_gate = fairness_gate
        return (
            terminal_fair_contrib
            + starvation_penalty
            + terminal_risk_penalty
            + unused_gas_penalty
        )

    def _fairness_block_summary(self) -> np.ndarray:
        # backlog wait 特征
        if self._candidates:
            candidate_waits = np.array(
                [self._wait(tx) / self._wait_norm_denom for tx in self._candidates],
                dtype=np.float32,
            )
            oldest_wait_norm = float(np.max(candidate_waits))
            p90_wait_norm = float(np.quantile(candidate_waits, 0.9))
        else:
            oldest_wait_norm = 0.0
            p90_wait_norm = 0.0

        # selected wait 特征
        if self._selected:
            selected_waits = np.array(
                [self._wait(tx) / self._wait_norm_denom for tx in self._selected],
                dtype=np.float32,
            )
            mean_wait_selected = float(selected_waits.mean())
            std_wait_selected = float(selected_waits.std())
        else:
            mean_wait_selected = 0.0
            std_wait_selected = 0.0

        oldest_unserved_ratio = self._oldest_unserved_ratio()
        if self._selected:
            selected_ids = {tx.tid for tx in self._selected}
            selected_oldest20_ratio = (
                len(selected_ids & self._pool_oldest_ids) / len(self._selected)
            ) if self._pool_oldest_ids else 0.0
            late_selected_ratio = (
                sum(1 for tx in self._selected if tx.arrival_time > self._median_arrival)
                / len(self._selected)
            )
        else:
            selected_oldest20_ratio = 0.0
            late_selected_ratio = 0.0

        sender_starvation_max, sender_starvation_mean = self._sender_starvation_stats()

        return np.array([
            oldest_wait_norm,
            p90_wait_norm,
            mean_wait_selected,
            std_wait_selected,
            oldest_unserved_ratio,
            selected_oldest20_ratio,
            late_selected_ratio,
            sender_starvation_max,
            sender_starvation_mean,
        ], dtype=np.float32)

    def _risk_block_summary(self) -> np.ndarray:
        selected_top10_risk_ratio = self._top10_risk_ratio_selected()
        risk_exposure_selected = self._risk_exposure_selected()
        selected_edge_risk_ratio = self._edge_risk_ratio_selected()
        selected_risky_rank_mean = self._avg_risky_rank_selected()

        if self._candidates:
            oldest_candidates = self._oldest_ids(self._candidates, C.FAIR_OLDEST_RATIO)
            overlap = [
                tx for tx in self._candidates
                if tx.tid in oldest_candidates and tx.risk_score >= C.HEURISTIC_RISK_THRESHOLD
            ]
            remaining_risky_oldest_overlap_ratio = len(overlap) / max(len(oldest_candidates), 1)
        else:
            remaining_risky_oldest_overlap_ratio = 0.0

        late_high_fee_unserved_ratio = self._late_high_fee_unserved_ratio()

        return np.array([
            selected_top10_risk_ratio,
            risk_exposure_selected,
            selected_edge_risk_ratio,
            selected_risky_rank_mean,
            remaining_risky_oldest_overlap_ratio,
            late_high_fee_unserved_ratio,
        ], dtype=np.float32)

    def _obs(self) -> dict:
        """构造 padded 观测。"""
        n = len(self._candidates)
        features = np.zeros((self.max_pool, C.TX_FEATURE_DIM), dtype=np.float32)
        for i, tx in enumerate(self._candidates):
            features[i] = tx.feature_vector(self._max_fee, self._max_gas, self._t_max)

        # 区块基础状态
        rem_gas_norm = self._remaining_gas / max(self._block_gas_limit, 1)
        n_sel_norm = len(self._selected) / max(len(self._pool), 1)
        acc_fee_norm = self._acc_fee / (self._max_fee * max(len(self._pool), 1))

        fairness_summary = self._fairness_block_summary()
        risk_summary = self._risk_block_summary()

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
            fairness_summary,
            risk_summary,
            seq_summary,
        ])

        # 动作掩码
        mask = np.zeros(self.max_pool + 1, dtype=np.int8)
        if self.no_action_mask:
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
            mask[n] = 0
        else:
            mask[n] = 1

        return {
            "tx_features": features,
            "block_state": block_state,
            "action_mask": mask,
            "num_candidates": n,
        }

    def _info(self) -> dict[str, Any]:
        if self._selected:
            waits = np.array([self._wait(tx) for tx in self._selected], dtype=np.float64)
            wait_p95 = float(np.quantile(waits, 0.95))
            wait_p99 = float(np.quantile(waits, 0.99))
            mean_wait = float(np.mean(waits))
            if mean_wait < 1e-12:
                wait_gini = 0.0
            else:
                diff = np.abs(waits[:, None] - waits[None, :])
                wait_gini = float(diff.mean() / (2.0 * mean_wait))
        else:
            wait_p95 = 0.0
            wait_p99 = 0.0
            wait_gini = 0.0

        return {
            "selected": [tx.tid for tx in self._selected],
            "total_fee": self._acc_fee,
            "gas_used": self._block_gas_limit - self._remaining_gas,
            "block_gas_limit": self._block_gas_limit,
            "num_selected": len(self._selected),
            "pool_size": len(self._pool),
            "wait_p95": wait_p95,
            "wait_p99": wait_p99,
            "wait_gini": wait_gini,
            "reward_decomposition": {
                "fee_reward": self._proxy_fee_reward,
                "age_reward": self._proxy_age_reward,
                "oldest_cover_reward": self._proxy_oldest_cover_reward,
                "risk_penalty": self._proxy_risk_penalty,
                "packing_reward": self._proxy_packing_reward,
                "late_fee_recovery_reward": self._proxy_late_fee_recovery,
                "terminal_fair_reward": self._proxy_terminal_fair,
                "starvation_penalty": self._proxy_starvation_penalty,
                "terminal_risk_penalty": self._proxy_terminal_risk_penalty,
                "unused_gas_penalty": self._proxy_unused_gas_penalty,
                "fairness_gate": self._proxy_fairness_gate,
                "stop_penalty": self._proxy_stop_penalty,
            },
            "proxy_age_reward": self._proxy_age_reward,
            "proxy_oldest_cover": self._proxy_oldest_cover_reward,
            "proxy_starvation_penalty": self._proxy_starvation_penalty,
            "proxy_terminal_risk_penalty": self._proxy_terminal_risk_penalty,
            "proxy_packing_reward": self._proxy_packing_reward,
            "proxy_unused_gas_penalty": self._proxy_unused_gas_penalty,
            "proxy_fairness_gate": self._proxy_fairness_gate,
        }

    # ------------------------------------------------------------------
    # 公开辅助
    # ------------------------------------------------------------------

    def get_selected_transactions(self) -> list[Transaction]:
        return list(self._selected)

    def get_pool(self) -> list[Transaction]:
        return list(self._pool)
