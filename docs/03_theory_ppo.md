# PPO — Proximal Policy Optimization

PPO (Schulman et al., 2017) is a policy-gradient method that improves training stability by constraining how much the policy can change in a single update.

---

## 1. Policy Gradient

The gradient of the expected return `J(π)` with respect to policy parameters θ is:

```
∇J(θ) = Eₜ[ ∇log π(aₜ|sₜ) · Âₜ ]
```

where `Âₜ` is the **advantage** — how much better `aₜ` was than average in state `sₜ`.

---

## 2. The Trust-Region Problem

Vanilla policy gradients are brittle. A bad update can collapse the policy. TRPO (Trust Region Policy Optimization) addressed this by enforcing a KL-divergence constraint between old and new policy:

```
max Eₜ[ π(a|s) / π_old(a|s) · Âₜ ]
s.t. KL(π_old || π) ≤ δ
```

TRPO works but is complex (conjugate gradients, line searches). PPO simplifies this with a **clipped surrogate objective**.

---

## 3. The PPO Clipped Objective

PPO replaces the constraint with a **clipped probability ratio**:

```
Lᶜᴸᴵᴾ(θ) = Eₜ[ min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ) ]
```

Where:

| Symbol | Meaning |
|---|---|
| `rₜ(θ)` | Probability ratio: `π(aₜ | sₜ) / π_old(aₜ | sₜ)` |
| `Âₜ` | Advantage estimate at timestep `t` |
| `ε` | Clip threshold (hyperparameter, default 0.2) |

**Intuition**:

- If `Âₜ > 0` (action was good): the ratio `rₜ` is clipped at `1+ε`. The policy can increase the probability of this action, but only up to a factor of `1+ε`. Beyond that, the gradient is zeroed out — preventing over-optimisation.
- If `Âₜ < 0` (action was bad): the ratio is clipped at `1-ε`. The policy can decrease the probability, but no further than the lower bound.

This keeps the new policy within a "trust region" of the old policy while being much simpler than TRPO.

---

## 4. Full PPO Loss

PPO uses a **combined** loss:

```
L(θ) = Lᶜᴸᴵᴾ(θ) - c₁ · Lⱽ(θ) + c₂ · H(π(·|s))
```

| Term | Purpose |
|---|---|
| `Lᶜᴸᴵᴾ` | Clipped policy gradient — maximises advantage |
| `Lⱽ` | Value function loss — MSE between predicted and actual returns |
| `H` | Entropy bonus — encourages exploration |

In our code (`agents/ppo.py:update`):

```python
ratio = (log_probs - batch_old_log_probs).exp()
surr1 = ratio * batch_adv
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
policy_loss = -torch.min(surr1, surr2).mean()

value_loss = nn.MSELoss()(values, batch_ret)

loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
```

---

## 5. Generalized Advantage Estimation (GAE)

GAE (Schulman et al., 2016) provides a low-variance, bias-controlled advantage estimate:

```
δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)         # TD residual
Âₜᴳᴬᴱ = Σ (γλ)ᵏ · δₜ₊ₖ               # Exponentially weighted sum
```

| Parameter | Effect |
|---|---|
| `λ = 0` | High bias, low variance (TD(0)) |
| `λ = 1` | Low bias, high variance (Monte Carlo) |
| `λ = 0.95` | Sweet spot (default in our config) |

In `agents/ppo.py:compute_gae`:

```python
for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
    gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
    advantages[t] = gae
    next_value = values[t]
returns = advantages + values
```

---

## 6. Multi-Epoch Mini-Batch Updates

Unlike A2C (which updates once per rollout), PPO reuses each rollout multiple times:

```python
for _ in range(update_epochs):        # 10 epochs
    np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):  # mini-batches
        # compute loss and update
```

This is **data-efficient**: each collected trajectory is reused for several gradient steps, but clipping prevents the policy from walking too far.

---

## 7. PPO in the RecSys Context

| Property | Benefit |
|---|---|
| Stable updates | The clip prevents the recommendation policy from suddenly changing — users don't see a jarring content shift |
| Multi-epoch reuse | Each user session (rollout) informs multiple gradient steps — sample-efficient |
| Entropy bonus | The agent maintains some randomness in recommendations, enabling exploration of new content |
| GAE | Smooths the noisy reward signal (users are inconsistent) |

---

## 8. Algorithm Pseudocode

```
for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        Run policy π_old for T timesteps
        Collect {sₜ, aₜ, rₜ, doneₜ, logπ_old(aₜ|sₜ), V(sₜ)}
    end for
    Compute advantages Âₜ via GAE (using V(sₜ) and rₜ)
    for epoch = 1, 2, ..., K do
        for minibatch in shuffled data do
            rₜ(θ) = exp(logπ(aₜ|sₜ) - logπ_old(aₜ|sₜ))
            L = -E[min(rₜÂₜ, clip(rₜ, 1±ε)Âₜ)] + c₁·MSE(V, R) - c₂·H(π)
            θ ← θ + α · ∇L
        end for
    end for
end for
```

---

**Next**: [04_theory_a2c.md](04_theory_a2c.md) — the simpler alternative.
