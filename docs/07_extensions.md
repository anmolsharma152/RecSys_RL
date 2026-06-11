# Extensions & Scaling Ideas

This project is designed to be a minimal, readable foundation. Here are concrete ways to grow it.

---

## 1. Better State Representations

### Current
`state = [user_embedding | item_embedding]` — a 64-dim concatenation.

### Extension: Transformer State Encoder

Replace the concatenation with a causal transformer over the user's interaction history:

```
sₜ = Transformer([item_emb₀, item_emb₁, ..., item_embₜ₋₁, user_emb])
```

This lets the agent attend to *which specific past items* drive the next recommendation, rather than compressing everything into one vector.

### Extension: Graph-Based State

For social platforms, include the user's social graph:

```
sₜ = [user_emb | item_emb | GNN(user_node, neighbour_items)]
```

This captures viral / social signals.

---

## 2. Hierarchical / Multi-Stage Policies

### Current
Single `ActorCritic` that outputs one item from the full pool.

### Extension: Retrieval + Ranking (Two-Tower)

```
π_retrieval(s) → 100 candidate items (fast, cheap embedding lookup)
π_ranker(s, candidate_set) → 1 item (slower, deeper network)
```

This mirrors production pipelines and makes training tractable with millions of items.

### Extension: Slate Recommendation

Recommend a *set* of K items at once (e.g., a grid of 4 videos). Use:

- **Slate-Q learning**: treat the slate as a joint action.
- **Sequence-wise policy gradient**: optimise the whole slate, not individual items.

---

## 3. Offline RL & Real Data

### Current
Synthetic environment with random embeddings.

### Extension: Offline Dataset from Logs

Collect logs from a deployed RecSys: `(s, a, r, s', done, π_behavior)`. Train with **offline RL** algorithms:

| Algorithm | Use case |
|---|---|
| **CQL** (Conservative Q-Learning) | Prevents over-estimation of unseen actions |
| **IQL** (Implicit Q-Learning) | Simple, stable, no explicit policy constraint |
| **BCQ** (Batch-Constrained Q-Learning) | Stays close to behavioural policy |

This lets you train on real user data without needing a simulator.

### Extension: World Model (Model-Based RL)

Train a **transition model** `P(s' | s, a)` from logged data, then train the policy in the learned simulator. This is what TikTok and ByteDance research teams are known to use.

---

## 4. Exploration Strategies

### Current
Entropy bonus in PPO (`ent_coef = 0.01`) — uniform exploration.

### Extension: Uncertainty-Based Exploration

- **Bootstrapped DQN**: multiple Q-heads, pick actions with high variance.
- **Intrinsic motivation**: add `r_intrinsic = novelty(s, a)` to reward.

### Extension: Contextual Bandits for Cold-Start

Use a separate bandit policy for new users / items, then hand off to the RL policy once sufficient data is collected.

---

## 5. Multi-Objective & Constrained Optimisation

### Current
Single scalar reward.

### Extension: Multi-Objective PPO

Optimise multiple reward streams simultaneously:

```
L = w₁·L_engagement + w₂·L_diversity + w₃·L_freshness - λ·L_ad_frequency
```

Use **Lagrangian methods** to enforce hard constraints (e.g., "at most 1 ad per 5 items").

### Extension: Fairness Constraints

Add demographic parity constraints to the policy gradient:

```
L_fairness = |E[r | group A] - E[r | group B]|
```

---

## 6. Infrastructure Scaling

| Direction | How |
|---|---|
| **Multi-GPU** | `torch.nn.DistributedDataParallel` + `num_envs × num_gpus` |
| **Vectorised envs** | `torch.multiprocessing` for parallel environment stepping |
| **Replay buffer** | Replace `RolloutBuffer` with a circular buffer for off-policy (SAC, TD3) |
| **Scalable logger** | Swap `Logger` for TensorBoard / Weights & Biases / MLflow |

---

## 7. Alternative Algorithms to Implement

| Algorithm | Why |
|---|---|
| **SAC** (Soft Actor-Critic) | Off-policy, maximum entropy — sample efficient for expensive simulators |
| **Rainbow DQN** | If action space is small enough, discrete Q-learning with distributional RL |
| **Decision Transformer** | Cast RL as sequence modelling — recommend items auto-regressively |
| **GRPO** (Group Relative PPO) | DeepSeek's variant — uses group baselines instead of a critic |

---

## 8. Evaluation Beyond Simulations

### Offline Metrics

| Metric | What it measures |
|---|---|
| **NDCG@K** | Ranking quality |
| **Recall@K** | Coverage of relevant items |
| **Hit rate** | Did the user engage? |
| **Policy cost** | Cumulative regret vs. optimal |
| **Diversity** | Intra-list similarity (ILS) |

### Online Metrics (A/B Test)

| Metric | What it measures |
|---|---|
| **User retention D7 / D30** | Long-term value |
| **Session length** | Engagement depth |
| **Churn rate** | User fatigue / dissatisfaction |
| **Ad revenue** | Business outcome |

---

## 9. The Ultimate Goal

A deployable RL RecSys stack:

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
│  Logs   │ → │ Offline  │ → │   RL    │ → │  Online  │
│ (HDFS)  │   │ Dataset  │   │  Agent  │   │ A/B Test │
└─────────┘   └──────────┘   └────┬────┘   └──────────┘
                                  │ learns from logs
                                  │ improved by online feedback
                                  ▼
                          ┌────────────────┐
                          │ Data Flywheel  │
                          └────────────────┘
```

Every component here (`agent`, `env`, `models`) is designed to be swapped out independently as you scale.
