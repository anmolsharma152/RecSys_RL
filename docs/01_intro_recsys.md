# How Real-World Recommender Systems Work

Recommender systems are the engine behind every major content platform. Understanding the production systems gives context for why RL is a natural fit.

---

## 1. The Two-Stage Pipeline

Almost every large-scale RecSys (TikTok, YouTube, Netflix, Instagram) uses a **cascade**:

```
       ┌──────────┐     ┌──────────┐     ┌──────────┐
       │ Retrieval │ ──> │ Ranking  │ ──> │  Re-rank │
       │ (hundreds │     │ (tens)   │     │  (final) │
       └──────────┘     └──────────┘     └──────────┘
```

1. **Retrieval (candidate generation)** — fast, cheap model (e.g., cosine similarity, two-tower, matrix factorisation) that scores millions of items down to hundreds.
2. **Ranking** — a deeper model (often a DNN with cross features) that scores the hundreds.
3. **Re-ranking** — apply business rules: diversity, freshness, fairness, hard filters, position bias.

The RL agent in this project replaces the **ranking + re-ranking** stages: given a user state, it selects the next item to show.

---

## 2. Platform-Specific Designs

### TikTok / Instagram Reels / YouTube Shorts

These are **short-form video** platforms. Key characteristics:

- **Session-based**: users swipe up/down, each swipe is an implicit feedback signal (watch time, like, share, skip).
- **The "For You" feed**: a single vertical scroll; the RL agent decides *which video to serve next* based on the user's real-time reaction to the previous video.
- **Reward signal**: weighted combination of `watch_time`, `likes`, `shares`, `comments`, `completion_rate`. The weights are tuned to maximise long-term user retention.
- **State**: user embedding (from past interactions) + video embedding + context (time of day, device, network speed).

**Imitation agent** (see [06_imitation_agents.md](06_imitation_agents.md)): continuous scrolling, watch-time reward, item embeddings from content features.

### YouTube (long-form)

Long-form (watch a 10-minute video) has different dynamics:

- **Watch time** is the primary optimisation target (Google's stated metric).
- **Suggestions sidebar**: the agent recommends the *next* video after the current one ends, or a related video alongside.
- **State**: user history (watched channels, search queries) + the *current video* context.
- **Reward**: `watch_time / video_length` (completion ratio) + downstream metrics (subscriptions).

### Social Media Feeds (Twitter/X, Facebook, LinkedIn)

These are **heterogeneous feeds**: posts, ads, suggested follows, sponsored content.

- **Action space is mixed**: which post to show, whether to insert an ad, whether to insert a "suggested follow" card.
- **Reward**: engagement (clicks, retweets, replies) weighted by advertiser bids for ads.
- **State**: user graph (who they follow, what their network engages with) + real-time trends.

---

## 3. Why RL?

Traditional RecSys uses **supervised learning**: predict the rating / click / watch-time for each item independently, then rank by predicted score. This is a **greedy, myopic** approach.

RL reasons about the **long-term** consequence of each recommendation:

| Aspect | Supervised | RL |
|---|---|---|
| Objective | Maximise *immediate* engagement | Maximise *cumulative* engagement |
| Exploration | None (exploitation only) | Explicit exploration via policy entropy |
| User state | Ignored or handcrafted | Learned and updated after each action |
| Dopamine loops | Short bursts | Models user saturation / fatigue |

This is especially important for short-form video: showing one more cat video might get a watch now, but it causes user fatigue and churn later. RL can learn to **diversify** and **pace** content to maximise *lifetime value*.

---

## 4. State of the Art in Industry

| Platform | Known technique | Reference |
|---|---|---|
| YouTube | Deep Q-Network for slate recommendation | "Deep Reinforcement Learning for Page-wise Recommendations" (2019) |
| Facebook | Contextual bandits for feed ranking | "A Contextual Bandit Approach to Recommended Content" |
| Alibaba | Multi-agent RL for e-commerce | "Multi-Agent Reinforcement Learning for Recommender Systems" |
| Google Play | PPO for app discovery | Internal papers |
| ByteDance (TikTok) | Offline RL + Simulator | Known from research blog posts |

---

## 5. Open Challenges

- **Offline evaluation**: you cannot A/B test every policy iteration. Off-policy evaluation (importance sampling, doubly robust) is critical.
- **Simulation fidelity**: training RL in a simulator and deploying to real users suffers from sim-to-real gap.
- **Cold-start**: new users / items have no history — RL must combine with bandit exploration.
- **Multi-stakeholder**: satisfying users, creators, and advertisers simultaneously.

---

**Next**: [02_rl_formulation.md](02_rl_formulation.md) — how we model recommendation as an MDP.
