# RL Formulation of a Recommender System

This document explains how a recommendation problem is cast as a Markov Decision Process (MDP), and how it maps to the code in this project.

---

## 1. The MDP Quintuple

An MDP is defined by `(S, A, P, R, γ)`:

| Symbol | Meaning | In our code |
|---|---|---|
| `S` | State space | `RecEnv.state_dim = 64` — user embedding ⊕ item embedding |
| `A` | Action space | `RecEnv.num_items = 1000` — choose one item from the pool |
| `P(s' | s, a)` | Transition dynamics | `UserModel.update_state()` + environment |
| `R(s, a)` | Reward function | `dot(user_emb, item_emb)` with decay |
| `γ` | Discount factor | `PPOConfig.gamma = 0.99` |

---

## 2. State Design

The state at timestep `t` is:

```
sₜ = [user_features | item_embedding_last_shown]
```

In `RecEnv._build_observation_with_item`:

```python
obs = np.concatenate([user_vec, item_emb])  # shape (64,)
```

- **user_vec** (32-dim): the user's current interest representation. In the simulator, this starts as a random unit-norm vector.
- **item_emb** (32-dim): the embedding of the item that was just recommended (zero-padded at the start of an episode).

The concatenation gives the agent both *who the user is* and *what they last saw*, enabling it to model:
- User preferences (via user vector)
- Sequential dynamics (via what was just served)
- Contextual similarity (dot-product similarity)

---

## 3. Action Space

Each action is the index of an item in the pool:

```python
action = 422  # select item #422
env.step(action)
```

This is a **discrete action space** with `num_items` choices. In production, the action space is much larger (millions), so techniques like **candidate generation** (retrieval) or **hierarchical RL** are used.

Our code uses `Categorical` distribution over logits produced by `DiscreteActor`:

```python
class DiscreteActor(nn.Module):
    def forward(self, x):
        x = self.backbone(x)
        return Categorical(logits=self.logits(x))
```

---

## 4. Reward Design

The reward in this project is a **structured proxy**:

```python
raw_reward = dot(user_emb, item_emb)
reward = raw_reward * γ^{t-1}
```

- The dot product measures alignment between user interest and item content.
- The discount `γ^{t-1}` (applied per step, not per episode) models **diminishing returns**: later recommendations are weighted less, encouraging the agent to get high-reward items early.

In production, reward is a **weighted composite**:

| Platform | Reward signals |
|---|---|
| TikTok | `α·watch_time + β·like + γ·share + δ·comment - η·skip_penalty` |
| YouTube | `watch_time / video_length * completion_indicator` |
| Instagram | `α·save + β·share + γ·profile_visit` |
| Social feed | `α·click + β·dwell_time + γ·ad_revenue` |

---

## 5. Episode Structure

An episode in `RecEnv`:

1. **Reset**: sample a random user, initialise their embedding, zero-pad the last-item slot → `s₀`.

2. **Loop** (max `max_episode_steps=20`):
   - Agent observes `sₜ` → selects action `aₜ` (which item to show).
   - Environment computes reward `rₜ` and updates user state (simulated response).
   - New observation `sₜ₊₁` = [updated_user_vec | item_emb of aₜ].

3. **Termination**: after `max_episode_steps` steps (simulating a finite browsing session).

---

## 6. Transition Dynamics (User Model)

In production, the user's state changes in complex ways (habituation, satisfaction, boredom). Our simple simulator approximates this as:

```python
def _simulate_user_response(self, user_vec, item_emb):
    return user_vec + 0.05 * item_emb
```

The user's embedding drifts slightly toward the items they've seen, simulating **interest adaptation**. In a real system this is learned from data (e.g., a GRU over the user's interaction history).

---

## 7. The Policy Optimization Objective

Given the MDP, we want to find a policy `π(a|s)` that maximises expected discounted return:

```
J(π) = E[ Σ γᵗ · rₜ ]
```

This is where PPO and A2C come in — see:
- [03_theory_ppo.md](03_theory_ppo.md)
- [04_theory_a2c.md](04_theory_a2c.md)

For the code implementation, see [05_code_walkthrough.md](05_code_walkthrough.md).
