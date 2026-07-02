"""PPO 训练算法"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

import config as C
from networks import ActorCritic


class RolloutBuffer:
    """存储一个 episode 的轨迹数据"""

    def __init__(self):
        self.obs: List[dict] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


def compute_gae(rewards, values, dones,
                gamma=C.DISCOUNT, lam=C.GAE_LAMBDA):
    """广义优势估计 (GAE)"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]

    return advantages, returns


class PPOTrainer:
    """PPO 截断目标训练器"""

    def __init__(self, model: ActorCritic,
                 lr_actor=C.LR_ACTOR,
                 lr_critic=C.LR_CRITIC,
                 clip_eps=C.PPO_CLIP,
                 epochs=C.PPO_EPOCHS,
                 entropy_coef=C.ENTROPY_COEF,
                 device=torch.device("cpu")):
        self.model = model
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.device = device

        param_groups = [
            {"params": model.encoder.parameters(), "lr": lr_actor},
            {"params": model.actor_fc.parameters(), "lr": lr_actor},
            {"params": model.actor_score.parameters(), "lr": lr_actor},
            {"params": [model.stop_embed], "lr": lr_actor},
            {"params": model.critic_fc.parameters(), "lr": lr_critic},
            {"params": model.critic_val.parameters(), "lr": lr_critic},
        ]
        self._base_lrs = [group["lr"] for group in param_groups]
        self.optimizer = torch.optim.Adam(param_groups)

    def set_lr_scale(self, scale: float) -> None:
        """Scale all optimizer group learning rates from their initial values."""
        scale = float(scale)
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * scale

    def update(self, buffer: RolloutBuffer):
        """执行 PPO 更新, 返回 (actor_loss, critic_loss, entropy)"""
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones)

        # 归一化优势
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        if adv_t.std() > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(
            buffer.log_probs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            buffer.actions, dtype=torch.long, device=self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs):
            log_probs, values, entropy = self.model.evaluate(
                buffer.obs, actions_t, self.device)

            # PPO-Clip 目标
            ratio = (log_probs - old_log_probs_t).exp()
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                1.0 + self.clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic 损失
            critic_loss = nn.functional.mse_loss(values, ret_t)

            # 总损失
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

        n = self.epochs
        return total_actor_loss / n, total_critic_loss / n, total_entropy / n
