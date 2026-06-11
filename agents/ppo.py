from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import PPOConfig
from models.actor_critic import ActorCritic


class RolloutBuffer:
    def __init__(self):
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.dones: list = []
        self.log_probs: list = []
        self.values: list = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def to_tensor(self, device: torch.device) -> dict:
        return {
            "states": torch.as_tensor(np.array(self.states), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.array(self.actions), dtype=torch.long, device=device),
            "rewards": torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.array(self.dones), dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
            "values": torch.as_tensor(np.array(self.values), dtype=torch.float32, device=device),
        }


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or PPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> tuple:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, value = self.model.act(state_t)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

    def compute_gae(self, data: dict, last_value: float) -> tuple:
        rewards = data["rewards"]
        dones = data["dones"]
        values = data["values"]
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, last_value: float) -> dict:
        data = self.buffer.to_tensor(self.device)
        advantages, returns = self.compute_gae(data, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = data["log_probs"]
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(len(data["states"]))
        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.batch_size):
                batch = indices[start : start + self.config.batch_size]
                batch_states = data["states"][batch]
                batch_actions = data["actions"][batch]
                batch_adv = advantages[batch]
                batch_ret = returns[batch]
                batch_old_log_probs = old_log_probs[batch]

                log_probs, entropy, values = self.model.evaluate(batch_states, batch_actions)
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, batch_ret)
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
