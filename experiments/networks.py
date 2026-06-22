"""Actor-Critic 网络与特征编码器"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config as C


class FeatureEncoder(nn.Module):
    """两层 MLP 将 d 维原始交易特征编码为 d_h 维稠密表示"""

    def __init__(self, input_dim: int = C.TX_FEATURE_DIM,
                 hidden_dim: int = C.HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, N, d) or (N, d)
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class ActorCritic(nn.Module):
    """
    Actor: 对每笔候选交易计算选择得分 -> softmax (含 STOP)
    Critic: 均值池化候选交易 + 区块状态 -> 标量价值
    """

    def __init__(self, tx_dim: int = C.TX_FEATURE_DIM,
                 block_dim: int = C.BLOCK_STATE_DIM,
                 hidden_dim: int = C.HIDDEN_DIM):
        super().__init__()
        self.encoder = FeatureEncoder(tx_dim, hidden_dim)

        # Actor head
        self.actor_fc = nn.Linear(hidden_dim + block_dim, hidden_dim)
        self.actor_score = nn.Linear(hidden_dim, 1)

        # STOP token 可学习嵌入
        self.stop_embed = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.stop_embed, std=0.01)

        # Critic head
        self.critic_fc = nn.Linear(hidden_dim + block_dim, hidden_dim)
        self.critic_val = nn.Linear(hidden_dim, 1)

    def forward(self, tx_features: torch.Tensor,
                block_state: torch.Tensor,
                action_mask: torch.Tensor,
                num_candidates: int):
        """
        Args:
            tx_features: (max_pool, d)  padded
            block_state: (block_dim,)
            action_mask: (max_pool+1,) int
            num_candidates: 实际候选数
        Returns:
            log_probs: (num_candidates+1,)  含 STOP
            value: scalar
        """
        n = num_candidates

        # 编码候选交易
        if n > 0:
            h = self.encoder(tx_features[:n])  # (n, hidden)
        else:
            h = torch.zeros(0, self.encoder.fc2.out_features,
                            device=tx_features.device)

        bs = block_state.unsqueeze(0).expand(max(n, 1), -1)  # (n, block_dim)

        # ----- Actor -----
        # 候选交易得分
        if n > 0:
            actor_in = torch.cat([h, bs[:n]], dim=-1)         # (n, hidden+block)
            scores_tx = self.actor_score(F.relu(self.actor_fc(actor_in))).squeeze(-1)  # (n,)
        else:
            scores_tx = torch.tensor([], device=tx_features.device)

        # STOP 得分
        stop_in = torch.cat([self.stop_embed, block_state])    # (hidden+block,)
        score_stop = self.actor_score(F.relu(self.actor_fc(stop_in.unsqueeze(0)))).squeeze()

        # 拼接 + 掩码
        all_scores = torch.cat([scores_tx, score_stop.unsqueeze(0)])  # (n+1,)
        mask = action_mask[:n + 1].float()

        # 将无效动作设为 -inf
        all_scores = all_scores + (1.0 - mask) * (-1e9)
        log_probs = F.log_softmax(all_scores, dim=0)

        # ----- Critic -----
        if n > 0:
            h_mean = h.mean(dim=0)  # (hidden,)
        else:
            h_mean = torch.zeros(self.encoder.fc2.out_features,
                                 device=tx_features.device)
        critic_in = torch.cat([h_mean, block_state])  # (hidden+block,)
        value = self.critic_val(F.relu(self.critic_fc(critic_in.unsqueeze(0)))).squeeze()

        return log_probs, value

    def act(self, obs: dict, device: torch.device = torch.device("cpu"),
            greedy: bool = False):
        """采样或贪心选择一个动作, 返回 (action, log_prob, value)"""
        tx_f = torch.as_tensor(obs["tx_features"], dtype=torch.float32, device=device)
        bs = torch.as_tensor(obs["block_state"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
        n = int(obs["num_candidates"])

        with torch.no_grad():
            log_probs, value = self.forward(tx_f, bs, mask, n)

        if greedy:
            action = log_probs[:n + 1].argmax()
        else:
            probs = log_probs[:n + 1].exp()
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), log_probs[action].item(), value.item()

    def evaluate(self, obs_batch: list[dict],
                 actions: torch.Tensor,
                 device: torch.device = torch.device("cpu")):
        """
        批量评估: 返回 (log_probs, values, entropy) 用于 PPO 更新
        由于各样本的候选数不同, 逐条处理
        """
        log_probs_list = []
        values_list = []
        entropy_list = []

        for i, obs in enumerate(obs_batch):
            tx_f = torch.as_tensor(obs["tx_features"], dtype=torch.float32, device=device)
            bs = torch.as_tensor(obs["block_state"], dtype=torch.float32, device=device)
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
            n = int(obs["num_candidates"])

            lp, v = self.forward(tx_f, bs, mask, n)
            a = actions[i].long()

            log_probs_list.append(lp[a])
            values_list.append(v)

            probs = lp[:n + 1].exp()
            entropy_list.append(-(probs * lp[:n + 1]).sum())

        return (torch.stack(log_probs_list),
                torch.stack(values_list),
                torch.stack(entropy_list))
