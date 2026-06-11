# Code Walkthrough

A line-by-line explanation of how this project's files fit together.

---

## 1. Entry Points

### `train.py`

Parses CLI args, builds `Config`, instantiates the environment and agent, then runs the training loop:

```
Argparse → Config → make_env() → make_agent() → train_ppo() / train_a2c() → save checkpoint
```

### `evaluate.py`

Loads a saved checkpoint, runs `N` episodes with `agent.model.eval()`, prints aggregate metrics.

---

## 2. Configuration (`config.py`)

Uses `dataclasses` for clean, type-annotated config:

```python
@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    algorithm: str = "ppo"       # switch between algorithms
    device: str = "auto"         # auto → cuda if available
```

Each sub-config only contains fields relevant to that component — no monolithic argument dicts.

---

## 3. Environment (`env/`)

### `env/user_model.py`

```python
class UserModel:
    def __init__(self, user_id, feature_dim=32):
        self.features = random_unit_vector(feature_dim)
        self.state = self.features.copy()
```

Stores the user's current interest vector. `get_state()` returns it; `update_state()` shifts it toward the item just consumed.

### `env/rec_env.py`

A gym-like environment:

```python
class RecEnv:
    def reset(self, user_id=None) -> np.ndarray:
        # pick a random user → build s₀ = [user_emb | zeros]

    def step(self, action) -> (obs, reward, done, info):
        # item_emb = item_embeddings[action]
        # reward = dot(user_emb, item_emb) * γ^(step-1)
        # user_emb += 0.05 * item_emb  (simulate interest shift)
        # sₜ₊₁ = [updated_user_emb | item_emb]
```

The observation concatenates user and item embeddings (64-dim total). The action space is `num_items` discrete choices.

---

## 4. Models (`models/actor_critic.py`)

### Shared backbone

```python
def build_mlp(in_dim, hidden_dims, activation):
    # Linear → Act → Linear → Act → ... → hidden_dims[-1]
```

### Actor heads

```python
class DiscreteActor:     # categorical over items
class GaussianActor:     # continuous actions (unused here)
```

### Critic

```python
class Critic:
    # same backbone → single scalar V(s)
```

### Combined

```python
class ActorCritic(nn.Module):
    def act(self, s):      # sample action → (a, logπ, V)
    def evaluate(self, s, a):  # (logπ(a|s), entropy, V(s))
```

---

## 5. Agents

### `agents/ppo.py`

**RolloutBuffer**: stores `(s, a, r, done, logπ, V)` for a full batch of environment steps.

**PPO.update()**:
1. Compute GAE advantages from stored values and rewards
2. Normalise advantages (zero-mean, unit-variance)
3. For `update_epochs` epochs, shuffle and split into mini-batches
4. For each mini-batch:
   - Compute probability ratio `r = exp(logπ_new - logπ_old)`
   - Clipped surrogate: `min(r·Â, clip(r, 1±ε)·Â)`
   - MSE for value function
   - Entropy bonus
   - Gradient step with `max_grad_norm` clipping

### `agents/a2c.py`

Simpler: collects `n_steps` transitions, computes returns with GAE, takes a single gradient step.

---

## 6. Training Loops

### PPO training (`train.py:train_ppo`)

```
for each timestep:
    for each parallel env:
        a, logπ, V = agent.select_action(s)
        s', r, done = env.step(a)
        buffer.store(s, a, r, done, logπ, V)
        if buffer is full (n_steps * num_envs):
            agent.update(last_values)
```

Parallel environments (`num_envs=8`) collect diverse trajectories simultaneously.

### A2C training (`train.py:train_a2c`)

```
for each timestep:
    collect n_steps of (s, a, r) in lists
    agent.update(states, actions, rewards, dones, log_probs, values, next_value)
```

Single environment, synchronous collection.

---

## 7. Utilities

- **`utils/logger.py`**: writes to stdout + CSV file for later analysis.
- **`utils/metrics.py`**: computes episode-level statistics (total, mean, std reward).

---

## 8. Data Flow Summary

```
┌──────────┐   sₜ    ┌──────────┐   aₜ    ┌──────────┐   rₜ    ┌──────────┐
│ RecEnv   │ ──────> │   PPO    │ ──────> │ RecEnv   │ ──────> │  Buffer  │
│ (state)  │         │ (policy) │         │ (reward) │         │          │
└──────────┘         └──────────┘         └──────────┘         └────┬─────┘
                                                                    │
                                                                    ▼
                                                              ┌──────────┐
                                                              │  Update  │
                                                              │ (GAE +   │
                                                              │  PPO)    │
                                                              └──────────┘
```

---

**Next**: [06_imitation_agents.md](06_imitation_agents.md) — building agents inspired by real platforms.
