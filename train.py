import argparse
from pathlib import Path

import torch
import numpy as np

from config import Config
from env.rec_env import RecEnv
from agents.ppo import PPO
from agents.a2c import A2C
from utils.logger import Logger
from utils.metrics import compute_metrics


def make_env(cfg: Config) -> RecEnv:
    return RecEnv(
        num_items=cfg.env.num_items,
        num_users=cfg.env.num_users,
        state_dim=cfg.env.state_dim,
        max_episode_steps=cfg.env.max_episode_steps,
        reward_decay=cfg.env.reward_decay,
        user_feature_dim=cfg.env.user_feature_dim,
        item_feature_dim=cfg.env.item_feature_dim,
    )


def make_agent(cfg: Config, env: RecEnv):
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    if cfg.algorithm == "ppo":
        return PPO(env.state_dim, env.num_items, cfg.ppo, device)
    elif cfg.algorithm == "a2c":
        return A2C(env.state_dim, env.num_items, cfg.a2c, device)
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")


def train_ppo(cfg: Config, agent: PPO, env: RecEnv, logger: Logger):
    num_envs = cfg.ppo.num_envs if hasattr(cfg.ppo, 'num_envs') else 1
    envs = [make_env(cfg) for _ in range(num_envs)]
    states = [e.reset() for e in envs]
    episode_rewards = [[] for _ in range(num_envs)]
    episode_count = 0
    step = 0

    while step < cfg.train.total_timesteps:
        for env_idx in range(num_envs):
            action, log_prob, value = agent.select_action(states[env_idx])
            next_state, reward, done, info = envs[env_idx].step(action)
            agent.store_transition(states[env_idx], action, reward, done, log_prob, value)
            episode_rewards[env_idx].append(reward)
            states[env_idx] = next_state

            if done:
                metrics = compute_metrics(episode_rewards[env_idx])
                logger.log(step, {f"env_{env_idx}/{k}": v for k, v in metrics.items()})
                episode_rewards[env_idx].clear()
                states[env_idx] = envs[env_idx].reset()
                episode_count += 1

            step += 1

        if step % cfg.ppo.n_steps == 0:
            last_values = [agent.model.get_value(
                torch.as_tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
            ).item() for s in states]
            avg_last_value = float(np.mean(last_values))
            update_info = agent.update(avg_last_value)
            if step % cfg.train.log_interval < num_envs:
                logger.log(step, update_info)

    return agent


def train_a2c(cfg: Config, agent: A2C, env: RecEnv, logger: Logger):
    state = env.reset()
    episode_rewards = []
    step = 0
    episode = 0

    while step < cfg.train.total_timesteps:
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        for _ in range(cfg.a2c.n_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            episode_rewards.append(reward)
            state = next_state
            step += 1

            if done:
                metrics = compute_metrics(episode_rewards)
                logger.log(step, metrics)
                episode_rewards.clear()
                state = env.reset()
                episode += 1

        with torch.no_grad():
            next_value = agent.model.get_value(
                torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            ).item()
        update_info = agent.update(states, actions, rewards, dones, log_probs, values, next_value)
        if step % cfg.train.log_interval < cfg.a2c.n_steps:
            logger.log(step, update_info)

    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"])
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.algorithm = args.algo
    cfg.train.total_timesteps = args.total_timesteps
    cfg.train.log_dir = args.log_dir
    cfg.train.seed = args.seed

    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    env = make_env(cfg)
    agent = make_agent(cfg, env)
    logger = Logger(cfg.train.log_dir, tag=cfg.algorithm)

    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if cfg.algorithm == "ppo":
        agent = train_ppo(cfg, agent, env, logger)
    else:
        agent = train_a2c(cfg, agent, env, logger)

    save_path = Path(cfg.train.checkpoint_dir) / f"{cfg.algorithm}.pt"
    agent.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
