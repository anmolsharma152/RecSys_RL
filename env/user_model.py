from typing import Optional

import numpy as np


class UserModel:
    def __init__(self, user_id: int, feature_dim: int = 32):
        self.user_id = user_id
        self.feature_dim = feature_dim
        self.features: Optional[np.ndarray] = None
        self.state: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.features = np.random.randn(self.feature_dim).astype(np.float32)
        self.features /= np.linalg.norm(self.features) + 1e-8
        self.state = self.features.copy()
        return self.state

    def update_state(self, action: int, reward: float, next_state: np.ndarray):
        self.state = next_state

    def get_state(self) -> np.ndarray:
        return self.state
