from typing import Optional

import numpy as np

from env.user_model import UserModel


class RecEnv:
    def __init__(
        self,
        num_items: int = 1000,
        num_users: int = 100,
        state_dim: int = 64,
        max_episode_steps: int = 20,
        reward_decay: float = 0.99,
        user_feature_dim: int = 32,
        item_feature_dim: int = 32,
    ):
        self.num_items = num_items
        self.num_users = num_users
        self.state_dim = state_dim
        self.max_episode_steps = max_episode_steps
        self.reward_decay = reward_decay
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim

        self._rng = np.random.default_rng()
        self.item_embeddings: Optional[np.ndarray] = None
        self.user_model: Optional[UserModel] = None
        self.current_step = 0
        self._init_items()

    def _init_items(self):
        self.item_embeddings = self._rng.standard_normal(
            (self.num_items, self.item_feature_dim)
        ).astype(np.float32)
        self.item_embeddings /= (
            np.linalg.norm(self.item_embeddings, axis=1, keepdims=True) + 1e-8
        )

    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        if user_id is None:
            user_id = int(self._rng.integers(0, self.num_users))
        self.user_model = UserModel(user_id, self.user_feature_dim)
        self.current_step = 0
        return self._build_observation()

    def _build_observation(self) -> np.ndarray:
        user_vec = self.user_model.get_state()
        obs = np.concatenate([user_vec, np.zeros(self.item_feature_dim, dtype=np.float32)])
        assert obs.shape[0] == self.state_dim, f"{obs.shape=} != {self.state_dim=}"
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.current_step += 1
        item_emb = self.item_embeddings[action]
        user_vec = self.user_model.get_state()
        raw_reward = float(np.dot(user_vec, item_emb))
        reward = raw_reward * (self.reward_decay ** (self.current_step - 1))
        next_user_state = self._simulate_user_response(user_vec, item_emb)
        self.user_model.update_state(action, reward, next_user_state)
        next_obs = self._build_observation_with_item(item_emb)
        terminated = self.current_step >= self.max_episode_steps
        return next_obs, reward, terminated, {"user_id": self.user_model.user_id}

    def _simulate_user_response(self, user_vec: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
        return user_vec + 0.05 * item_emb

    def _build_observation_with_item(self, item_emb: np.ndarray) -> np.ndarray:
        user_vec = self.user_model.get_state()
        obs = np.concatenate([user_vec, item_emb])
        assert obs.shape[0] == self.state_dim, f"{obs.shape=} != {self.state_dim=}"
        return obs

    def render(self):
        pass
