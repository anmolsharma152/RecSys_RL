from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    num_items: int = 1000
    num_users: int = 100
    state_dim: int = 64
    max_episode_steps: int = 20
    reward_decay: float = 0.99
    user_feature_dim: int = 32
    item_feature_dim: int = 32


@dataclass
class ModelConfig:
    actor_hidden_dims: list = field(default_factory=lambda: [128, 64])
    critic_hidden_dims: list = field(default_factory=lambda: [128, 64])
    activation: str = "tanh"


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    batch_size: int = 64
    n_steps: int = 2048
    num_envs: int = 8


@dataclass
class A2CConfig:
    lr: float = 7e-4
    gamma: float = 0.99
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 5


@dataclass
class TrainConfig:
    total_timesteps: int = 100_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    algorithm: str = "ppo"
    device: str = "auto"
