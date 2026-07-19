# RecSys RL (PPO) — status handoff

| Field | Value |
|-------|--------|
| **As of** | 2026-07-19 |
| **Branch** | `main` |
| **Product** | Deep RL for recommendations — PPO/A2C-style agents over a recsys env |

Theory docs: `docs/01_*.md` … `07_*.md`.

---

## What exists

| Area | Path |
|------|------|
| Train / eval | `train.py`, `evaluate.py`, `main.py` |
| Agents / env / models | `agents/`, `env/`, `models/` |
| Config | `config.py` |

---

## Resume

```bash
cd ~/Projects/RecSys_RL
# uv / pip install -e . or requirements per README
python train.py
python evaluate.py
```

See [setup.md](./setup.md).
