# Results targets (design goals only)

> These are **simulated targets for prototype scope** — not measurements from a
> production-scale deployment. This repository does **not** claim or ship
> accuracy numbers.

| Metric | Target | Why it matters |
|--------|--------|----------------|
| Vision classification accuracy | ≥ 95% | Correctly names PET bottle, can, rubbish, food, or tampered item |
| False-positive rate (fraud paid) | < 2% | Limits how often an ineligible item still collects the 10¢ |
| False-negative rate (valid item held) | < 5% | Keeps genuine recyclers from being wrongly turned away |
| End-to-end decision latency | < 2 s | Keeps the queue moving at busy Return Right bays |

Constants live in `smartsort/config.py` (`TARGET_*`). Score your own labelled
trials with `smartsort/metrics.py` / `scripts/evaluate.py` — never invent results.
