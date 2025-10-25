# Homework 2 — Multi-Armed Bandits (A/B Testing)

This project implements two algorithms to explore and compare A/B testing strategies using the **Multi-Armed Bandit** approach:
- **Epsilon-Greedy** (with decaying exploration εₜ = ε₀ / t)
- **Thompson Sampling** (Normal rewards with known precision)

---

## 🧠 Experiment Setup
- True arm means: `[1, 2, 3, 4]`
- Number of trials: `20,000`
- Rewards: Normal(μ_arm, σ=1)
- Random seed: 42 (for reproducibility)

---

## ⚙️ Requirements
Install dependencies once before running:

```bash
pip install -r requirements.txt
