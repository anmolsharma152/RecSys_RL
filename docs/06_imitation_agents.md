# Imitation Agents for Real-World Platforms

This document sketches how you would extend this project's codebase to approximate recommendation policies from TikTok, YouTube, and social media feeds. These are **imitation agents** — not exact replicas (those are proprietary), but faithful approximations of the *behavioural logic* each platform uses.

Each agent reuses the core `RecEnv`, `PPO`, and `ActorCritic` infrastructure, only changing the reward function, state representation, and action semantics.

---

## 1. TikTok / Short-Form Video Agent

### Behavioural Signature

TikTok's "For You" feed:
- Vertical swipe: one video at a time, full screen.
- User signals: **watch time** (continuous), **like**, **share**, **follow creator**, **skip** (< 1s = negative).
- The agent learns which video to show next based on real-time reaction to the current video.

### State

```
sₜ = [user_embedding | video_embedding | engagement_context]
```

Where `engagement_context` encodes:
- Watch time ratio of the last video (0.0–1.0)
- Whether the user liked / shared / followed
- Time of day, day of week
- How many videos in a row the user has watched from the same creator (fatigue)

### Reward

```python
def reward(watch_ratio, liked, shared, followed, skipped):
    if skipped:
        return -0.5
    r = 0.4 * watch_ratio + 0.3 * liked + 0.2 * shared + 0.1 * followed
    return r
```

The weights encode the platform's true objective: **long-term user retention**, approximated by engagement depth.

### Imitation Implementation

The key differences from the base agent:

| Aspect | Base (this project) | TikTok imitation |
|---|---|---|
| Item embeddings | Random | Pre-trained CLIP / video features |
| User initialisation | Random | From past interaction history |
| Reward | Dot product | Weighted engagement composite |
| Episode length | 20 | 50–200 (scrolling session) |
| User state update | Linear drift | GRU over interaction history |
| Action constraint | All items | Candidate set from retrieval stage |

### To build it

1. Replace `RecEnv.item_embeddings` with real video features (e.g., from a pre-trained vision model).
2. Change `RecEnv.step` reward to the composite formula above.
3. Replace `UserModel` with a small GRU that processes the history of watched video embeddings.
4. Increase `max_episode_steps` to 100–200 to simulate a real scrolling session.

---

## 2. YouTube / Long-Form Video Agent

### Behavioural Signature

YouTube's recommendation sidebar and homepage:
- Suggested videos alongside the current video or on the homepage.
- Primary reward: **watch time** (Google's stated metric).
- Secondary rewards: clicks, subscriptions, ad clicks.
- State includes the *current video context* + long-term user history.

### State

```
sₜ = [user_embedding | current_video_embedding | watch_progress | history_summary]
```

Where `history_summary` might be a learned aggregation of the last 50 watched videos (e.g., a mean pooling or attention-weighted sum).

### Reward

```python
def reward(watch_seconds, video_length, clicked, subscribed):
    completion = watch_seconds / max(video_length, 1)
    return 0.6 * completion + 0.2 * clicked + 0.2 * subscribed
```

### Imitation Implementation

| Aspect | Base (this project) | YouTube imitation |
|---|---|---|
| Episode structure | Single session | Multi-video sequence with context |
| Action semantics | Show one item | Recommend from sidebar + homepage |
| User model | Simple drift | Long-term + short-term embeddings |
| Reward | Immediate engagement | Completion-weighted watch time |
| Diversity | None | Explicit diversity bonus |

### To build it

1. Add a `current_video` field to the state that persists across several steps (the user watches one video and receives recommendations).
2. Make `done` fire only when the user "leaves" the session (e.g., after 5 recommended videos with no click).
3. Add a diversity bonus: `diversity_penalty = λ * cosine_sim(prev_videos, candidate).mean()` subtracted from reward.

---

## 3. Social Media Feed Agent (Twitter / Instagram / Facebook)

### Behavioural Signature

Heterogeneous feed with multiple content types:
- Organic posts from followed accounts
- Sponsored posts / ads
- Suggested follows
- Stories / ephemeral content

The agent decides **what type of content to insert next**, not just which item.

### State

```
sₜ = [user_embedding | last_content_type | time_since_last_ad | feed_diversity_vector]
```

### Action Space

```
Actions = [
    SELECT_POST(post_id),    # show an organic post
    SHOW_AD(ad_id),          # show a sponsored post
    SUGGEST_FOLLOW(user_id), # show 'suggested for you'
]
```

This is a **multi-discrete** action space: first pick the *type*, then pick the *content* for that type. Our `DiscreteActor` would need to be extended to a hierarchical or multi-head policy.

### Reward

```python
def reward(engagement_type, dwell_time, ad_revenue):
    r = 0.3 * engagement_type + 0.3 * dwell_time_normalized + 0.4 * ad_revenue
    return r
```

Where `engagement_type = {click: 0.5, like: 0.3, share: 1.0, reply: 0.8, none: 0.0}`.

### Imitation Implementation

| Aspect | Base (this project) | Social feed imitation |
|---|---|---|
| Action space | Single discrete | Hierarchical (type → content) |
| Item pool | Homogeneous | Heterogeneous (posts, ads, follows) |
| Business rules | None | Ad frequency capping, diversity |
| Reward | User engagement | User engagement + advertiser revenue |
| State features | User + item | User + item type + ad history |

### To build it

1. Refactor `RecEnv` to return a *set* of candidate items per type (organic pool, ad pool, suggest pool).
2. Extend `ActorCritic` with a hierarchical policy: first a `GatingNetwork` picks the type, then a per-type item selector.
3. Add budget constraints: e.g., at most 1 ad every 5 items.
4. Add `ad_revenue` to reward.

---

## 4. Hybrid: TikTok + E-Commerce Agent

Combine short-form video with shoppable content (TikTok Shop, Instagram Shopping):

### Reward

```python
def reward(watch_ratio, liked, purchased, product_saved):
    return 0.4 * watch_ratio + 0.2 * liked + 0.3 * purchased + 0.1 * product_saved
```

### Action Space

```
Actions = [SHOW_VIDEO(v), SHOW_PRODUCT(p), SHOW_SHOPPABLE_VIDEO(v, p)]
```

The agent learns when to push entertainment vs. commerce — a delicate balance that RL handles naturally.

---

## 5. Learning Guidance Agent (Duolingo / Khan Academy)

### Behavioural Signature

Educational platforms recommend *what to learn next* — this maps almost perfectly to RL:

- **State**: `[knowledge_state | last_exercise_type | performance_history | time_spent]`
- **Action**: `[RECOMMEND_LESSON(l), RECOMMEND_REVIEW(r), RECOMMEND_PRACTICE(p)]`
- **Reward**: `completion + correctness - frustration_penalty`

### Key Difference

The reward function models **learning outcomes** (knowledge retention, engagement, not quitting), which naturally have long-term dependencies — exactly the setting where RL outperforms supervised heuristics.

---

## Summary Table

| Platform | State | Action | Reward | Challenge |
|---|---|---|---|---|
| TikTok | user + video + engagement context | next video | watch time + engagement | fatigue modelling |
| YouTube | user + video + watch progress | sidebar suggestions | completion-weighted watch time | long sessions |
| Social feed | user + feed context + ad history | content type + item | engagement + ad revenue | multi-stakeholder |
| Educational | knowledge state + activity history | lesson type | completion + correctness | knowledge modelling |

---

**Next**: [07_extensions.md](07_extensions.md) — scaling ideas and advanced techniques.
