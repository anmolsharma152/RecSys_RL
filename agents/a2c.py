from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import A2CConfig
from models.actor_critic import ActorCritic


class A2C:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[A2CConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or A2CConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> tuple:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, value = self.model(state_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def update(self, states, actions, rewards, dones, log_probs, values, next_value: float) -> dict:
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device)

        returns = torch.zeros_like(rewards_t)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards_t[t] + self.config.gamma * next_value * (1 - dones_t[t]) - values[t]
            gae = delta + self.config.gamma * 0.95 * (1 - dones_t[t]) * gae
            returns[t] = gae + values[t]

        log_probs_new, entropy, pred_values = self.model.evaluate(states_t, actions_t)
        advantage = returns - pred_values.detach()

        policy_loss = -(log_probs_new * advantage).mean()
        value_loss = nn.MSELoss()(pred_values, returns)
        loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
