# A2C — Advantage Actor-Critic

A2C is the synchronous, on-policy variant of the A3C (Asynchronous Advantage Actor-Critic) algorithm. It is simpler and lighter than PPO.

---

## 1. The Actor-Critic Architecture

A2C maintains two components:

```
                        ┌──────────────┐
                        │    State s   │
                        └──────┬───────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
              ┌─────┴──────┐      ┌──────┴─────┐
              │    Actor   │      │   Critic   │
              │  π(a | s)  │      │   V(s)     │
              └─────┬──────┘      └──────┬─────┘
                    │                     │
              ┌─────┴──────┐      ┌──────┴─────┐
              │ action a   │      │   value    │
              └────────────┘      └────────────┘
```

- **Actor** (policy): decides which item to recommend.
- **Critic** (value function): estimates how good the current state is.

Both share the same backbone but have separate output heads:

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.actor = DiscreteActor(state_dim, action_dim, ...)
        self.critic = Critic(state_dim, ...)
```

---

## 2. The A2C Gradient

A2C updates the actor using the **advantage**:

```
∇J(θ) = E[ ∇log π(aₜ|sₜ) · (Rₜ - V(sₜ)) ]
```

Where:

| Symbol | Meaning |
|---|---|
| `Rₜ` | The N-step return |
| `V(sₜ)` | Critic's estimate of the state value |
| `Âₜ = Rₜ - V(sₜ)` | The **advantage** |

The critic is trained to minimise `MSE(Rₜ, V(sₜ))`.

---

## 3. N-Step Returns

A2C doesn't wait for the episode to finish. It collects `n_steps` transitions, then computes truncated returns:

```
Rₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ... + γⁿ⁻¹·rₜ₊ₙ₋₁ + γⁿ·V(sₜ₊ₙ)
```

In `agents/a2c.py`:

```python
returns = torch.zeros_like(rewards_t)
gae = 0.0
for t in reversed(range(len(rewards))):
    delta = rewards_t[t] + gamma * next_value * (1 - dones_t[t]) - values[t]
    gae = delta + gamma * 0.95 * (1 - dones_t[t]) * gae
    returns[t] = gae + values[t]
```

The code above actually uses a GAE-like return (same formula as PPO's GAE, with λ=0.95 hardcoded), but A2C can also work with simple N-step bootstrapping.

---

## 4. Full A2C Loss

```
L(θ) = -E[ logπ(aₜ|sₜ) · Âₜ ] + c₁ · MSE(V(sₜ), Rₜ) - c₂ · H(π(·|sₜ))
```

In code:

```python
policy_loss = -(log_probs_new * advantage).mean()
value_loss = nn.MSELoss()(pred_values, returns)
loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
```

---

## 5. A2C vs PPO

| Aspect | A2C | PPO |
|---|---|---|
| Update frequency | Once per `n_steps` | After full rollout (batch) |
| Stability | Lower (can be noisy) | Higher (clipped surrogate) |
| Sample reuse | One pass | Multi-epoch + mini-batches |
| Complexity | Minimal | Moderate |
| When to use | Quick prototyping, simple envs | Complex envs requiring stability |
| Entropy bonus | Same | Same |

For RecSys, PPO is generally preferred because:
- The recommendation space is large and noisy — clipping helps.
- User sessions provide limited data — multi-epoch reuse is valuable.
- Stability matters — you don't want a user's feed to abruptly flip.

---

## 6. A2C Pseudocode

```
Initialize policy π and value function V
for iteration = 1, 2, ... do
    s = env.reset()
    for step = 1, ..., n_steps do
        aₜ = π(sₜ)              # sample action
        sₜ₊₁, rₜ, done = env.step(aₜ)
        store (sₜ, aₜ, rₜ, done, logπ(aₜ|sₜ), V(sₜ))
    end for
    R = V(sₙ) if not done else 0
    for t = n-1, ..., 0 do
        R = rₜ + γ · R          # compute returns backwards
        Âₜ = R - V(sₜ)          # advantage
    end for
    L = -logπ(aₜ|sₜ)·Âₜ + c₁·MSE(V(sₜ), R) - c₂·H(π)
    θ ← θ + α · ∇L
end for
```

---

**Next**: [05_code_walkthrough.md](05_code_walkthrough.md) — full walkthrough of this project's code.
