from typing import Sequence

import numpy as np


def compute_metrics(
    rewards: Sequence[float],
    top_k: int = 10,
) -> dict:
    rewards_arr = np.array(rewards)
    return {
        "total_reward": float(rewards_arr.sum()),
        "mean_reward": float(rewards_arr.mean()),
        "max_reward": float(rewards_arr.max()),
        "min_reward": float(rewards_arr.min()),
        "std_reward": float(rewards_arr.std()),
    }
