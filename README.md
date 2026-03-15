# Soft Actuator PINN — Zig-Zag Pneumatic Bending Actuator

Physics-Informed Neural Network (PINN) for inverse parameter identification of a
pneumatic zig-zag soft bending actuator.  
**This is the original v3 codebase** (width=64, depth=4, n_iters=4000) with three additions:
1. Extended `evaluate.py` — 8-pressure centerline comparison + complete summary figure
2. New `visualize_zigzag_geometry.py` — 4 publication-quality 3-D geometry figures
3. New `experiments.py` — two ablation experiments (see below)

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate synthetic dataset (original settings)
python src/data_generation.py

# 2. Train PINN  (original hyperparameters, unchanged)
python src/train_pinn.py

# 3. Train baseline MLP
python src/train_mlp.py

# 4. Evaluate — generates all result figures including:
#    - shape_examples.png        (8 pressures: 10,20,30,40,55,70,85,100 kPa)
#    - complete_summary.png      (3×4 comprehensive panel)
#    - training_validation_histories.png
#    - identified_parameters.png
#    - tip_response_curve.png
#    - blocked_force_curve.png
python src/evaluate.py

# 5. Generate 3-D geometry visualisation (no training required)
python src/visualize_zigzag_geometry.py

# 6. Run ablation experiments
python src/experiments.py                  # both experiments
python src/experiments.py --exp 1          # Exp 1 only
python src/experiments.py --exp 2          # Exp 2 only
python src/experiments.py --exp 2 --n_seeds 5   # more seeds for Exp 2
```

---

## Experiment 1 — Blocked-force loss ablation

**Goal:** Does the blocked-force observation actually help parameter identification?

| Version | w_block | Expected result |
|---------|---------|-----------------|
| A | 0 (removed) | k_p drifts; EI harder to pin |
| B | 10 (original) | k_p converges faster; EI more accurate |

Output: `results/exp1/exp1_blocked_force_ablation.png`  
Shows: EI error, k_p error, tip RMSE, blocked-force RMSE, and convergence curves for both versions.

---

## Experiment 2 — PINN vs MLP under different data regimes

**Goal:** Is PINN more data-efficient than MLP?

| Regime | Training pressures | Description |
|--------|--------------------|-------------|
| Low | 3 levels | Very sparse |
| Medium | 6 levels | Original split |
| High | 8 levels | Denser sampling |

Averaged over `--n_seeds` random seeds (default 3).  
Output: `results/exp2/exp2_data_regime_comparison.png`  
Shows: tip RMSE, shape RMSE, EI error as grouped bar charts with error bars.

---

## File structure

```
soft_actuator_pinn_starter/
├── data/synthetic/         ← generated CSVs
├── results/
│   ├── figures/            ← geometry visualisation figures
│   ├── exp1/               ← Exp 1 outputs
│   └── exp2/               ← Exp 2 outputs
└── src/
    ├── actuator_config.py          ← geometry + geo_* keys for 3-D viz
    ├── reduced_model.py            ← BVP solver (unchanged)
    ├── pinn_model.py               ← PINN + MLP (unchanged)
    ├── data_generation.py          ← dataset generation (unchanged)
    ├── train_pinn.py               ← PINN training (unchanged)
    ├── train_mlp.py                ← MLP training (unchanged)
    ├── evaluate.py                 ← ★ extended: 8-pressure shapes + summary
    ├── visualize_actuator_case.py  ← quick centerline preview (unchanged)
    ├── visualize_zigzag_geometry.py ← ★ NEW: 4 geometry figures
    └── experiments.py             ← ★ NEW: Exp 1 + Exp 2
```
