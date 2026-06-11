import argparse
from pathlib import Path

import torch
import numpy as np

from config import Config
from env.rec_env import RecEnv
from agents.ppo import PPO
from agents.a2c import A2C
from utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "a2c"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = Config()
    cfg.algorithm = args.algo
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    env = RecEnv(
        num_items=cfg.env.num_items,
        num_users=cfg.env.num_users,
        state_dim=cfg.env.state_dim,
        max_episode_steps=cfg.env.max_episode_steps,
        reward_decay=cfg.env.reward_decay,
        user_feature_dim=cfg.env.user_feature_dim,
        item_feature_dim=cfg.env.item_feature_dim,
    )

    if cfg.algorithm == "ppo":
        agent = PPO(env.state_dim, env.num_items, cfg.ppo, device)
    else:
        agent = A2C(env.state_dim, env.num_items, cfg.a2c, device)

    agent.load(args.checkpoint)
    agent.model.eval()

    all_metrics = []
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        rewards = []
        while not done:
            with torch.no_grad():
                action, *_ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
        metrics = compute_metrics(rewards)
        all_metrics.append(metrics)
        print(f"Episode {ep + 1}: total_reward={metrics['total_reward']:.4f}")

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print("\n=== Average over {} episodes ===".format(args.episodes))
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
