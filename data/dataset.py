from typing import Optional
from pathlib import Path

import numpy as np


class RecSysDataset:
    def __init__(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        interactions: np.ndarray,
    ):
        self.user_features = user_features
        self.item_features = item_features
        self.interactions = interactions

    @property
    def num_users(self) -> int:
        return self.user_features.shape[0]

    @property
    def num_items(self) -> int:
        return self.item_features.shape[0]

    def get_user_features(self, user_id: int) -> np.ndarray:
        return self.user_features[user_id]

    def get_item_features(self, item_id: int) -> np.ndarray:
        return self.item_features[item_id]


def load_movielens(
    path: str,
    user_feature_dim: int = 32,
    item_feature_dim: int = 32,
) -> Optional[RecSysDataset]:
    data_path = Path(path)
    if not data_path.exists():
        return None
    raise NotImplementedError("MovieLens loader not yet implemented")
