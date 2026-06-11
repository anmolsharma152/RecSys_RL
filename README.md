# RecSys_RL

Reinforcement Learning for Recommender Systems — PPO & A2C from scratch.

```mermaid
flowchart TB
    subgraph Environment["Recommendation Environment"]
        U[("User Model<br/>(embedding + history)")]
        I[("Item Pool<br/>(learned embeddings)")]
        S[("State<br/>sₜ = user_emb ⊕ last_item_emb")]
    end

    subgraph Agent["RL Agent (Actor-Critic)"]
        A[("Actor π(a|s)<br/>→ policy over items")]
        C[("Critic V(s)<br/>→ expected return")]
    end

    subgraph Training["PPO / A2C Update"]
        B[("Rollout Buffer<br/>(s, a, r, logπ, V)")]
        GAE[("GAE: advantage<br/>Â = δ + γλ·Âₜ₊₁")]
        CLIP[("Clipped surrogate<br/>Lᴾᴾᴼ = min(rÂ, clip(r,1±ε)Â)")]
    end

    User -->|"reset / update"| S
    I -->|"item_emb[action]"| S
    S -->|"sₜ"| A
    S -->|"sₜ"| C
    A -->|"aₜ ~ π(·|sₜ)"| EnvStep["Environment<br/>step(aₜ)"]
    EnvStep -->|"rₜ = sim(user_emb, item_emb)"| R[("Reward rₜ")]
    EnvStep -->|"sₜ₊₁"| S
    R --> B
    EnvStep --> B
    B --> GAE
    GAE --> CLIP
    CLIP -->|"gradient update"| A
    CLIP -->|"gradient update"| C
```

## Overview

A recommender system cast as a Markov Decision Process:

| Component     | This project                                              |
| ------------- | --------------------------------------------------------- |
| **State**     | User embedding ⊕ last recommended item embedding (64-dim) |
| **Action**    | Select one item from the candidate pool (discrete)        |
| **Reward**    | Dot-product similarity between user and item embeddings   |
| **Agent**     | Actor-critic neural network                               |
| **Algorithm** | PPO (clipped surrogate + GAE) or A2C (N-step)             |

## Quickstart

```bash
pip install numpy torch
python -m train --algo ppo --total-timesteps 100000
python -m evaluate --algo ppo --checkpoint checkpoints/ppo.pt
```

## Project Layout

```
├── config.py           # Hyperparameter dataclasses (Env / PPO / A2C / Train)
├── env/                # RecEnv (gym-like) + UserModel
├── agents/             # PPO + A2C algorithm implementations
├── models/             # Actor & Critic neural networks
├── data/               # Dataset loader stubs
├── utils/              # Logger (stdout + CSV) + metrics
├── train.py            # Training entrypoint
├── evaluate.py         # Evaluation entrypoint
└── docs/               # Theory, math, and deep-dive explanations
```

## Docs

| Document                                                   | What it covers                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [docs/01_intro_recsys.md](docs/01_intro_recsys.md)         | How TikTok, YouTube, & social media RecSys actually work     |
| [docs/02_rl_formulation.md](docs/02_rl_formulation.md)     | MDP formulation: state, action, reward design                |
| [docs/03_theory_ppo.md](docs/03_theory_ppo.md)             | PPO theory and full math derivation                          |
| [docs/04_theory_a2c.md](docs/04_theory_a2c.md)             | A2C theory and math                                          |
| [docs/05_code_walkthrough.md](docs/05_code_walkthrough.md) | Line-by-line explanation of this codebase                    |
| [docs/06_imitation_agents.md](docs/06_imitation_agents.md) | Imitation agents inspired by TikTok, YouTube, & social feeds |
| [docs/07_extensions.md](docs/07_extensions.md)             | Scaling ideas: Transformers, multi-stage, exploration        |

## Config

| Config class  | Key fields                                            |
| ------------- | ----------------------------------------------------- |
| `EnvConfig`   | num_items, state_dim, max_episode_steps, reward_decay |
| `PPOConfig`   | lr, clip_eps, gae_lambda, update_epochs, batch_size   |
| `A2CConfig`   | lr, n_steps, ent_coef                                 |
| `TrainConfig` | total_timesteps, log_interval, seed                   |
