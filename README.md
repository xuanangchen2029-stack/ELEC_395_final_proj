# Soft Actuator PINN — Zig-Zag Pneumatic Bending Actuator

Physics-Informed Neural Network (PINN) for **inverse parameter identification** of a
pneumatic zig-zag soft bending actuator. The PINN simultaneously learns the
deformed centerline field and identifies the effective mechanical parameters
`EI_eff` (bending stiffness) and `k_p` (pressure-to-actuation coefficient).

---

## Actuator geometry (default)

| Parameter | Value |
|-----------|-------|
| Total length L | **200 mm** |
| Cross-section (w × h) | 22 × 22 mm |
| Wall thickness t | 2 mm |
| Inlet plain region l₁ | 20 mm |
| Free-tip plain region l₂ | 20 mm |
| Active zig-zag region | 160 mm |
| Number of teeth N | 10 |
| Tooth pitch (auto) | 16 mm |
| Tooth depth (auto) | 9 mm |

All parameters are in `src/actuator_config.py` and can be changed freely.

---

## Reduced-order model

```
x'(s) = cos θ(s)          — inextensible kinematics
y'(s) = sin θ(s)
EI_eff · θ''(s) + k_p · p = 0    — moment balance
θ(0) = x(0) = y(0) = 0           — clamped base
θ'(L) = 0                         — free tip
```

The blocked force is the reaction force that keeps `y(L) = 0` under pressure.

---

## Quick start

```bash
pip install -r requirements.txt

# 1 — Regenerate the synthetic dataset
python src/data_generation.py

# 2 — Visualise the reduced-order model predictions (no training required)
python src/visualize_actuator_case.py

# 3 — NEW: Full 3-D zig-zag geometry visualisation
python src/visualize_zigzag_geometry.py

# 4 — Train the PINN
python src/train_pinn.py

# 5 — Train the baseline MLP
python src/train_mlp.py

# 6 — Evaluate and generate result figures
python src/evaluate.py
```

All scripts are callable from the **project root** (the folder containing `src/`).

---

## File structure

```
soft_actuator_pinn_starter/
├── data/
│   ├── synthetic/          ← auto-generated CSVs
│   └── fem/                ← placeholder for FEM data
├── results/
│   ├── figures/            ← all output plots
│   └── tables/
├── src/
│   ├── actuator_config.py          ← geometry + material config (edit here)
│   ├── reduced_model.py            ← BVP solver + blocked-force solver
│   ├── data_generation.py          ← synthetic dataset generation
│   ├── pinn_model.py               ← SoftActuatorPINN + MLP definitions
│   ├── train_pinn.py               ← PINN training loop
│   ├── train_mlp.py                ← MLP training loop
│   ├── evaluate.py                 ← metrics + result figures
│   ├── visualize_actuator_case.py  ← quick model inspection plots
│   └── visualize_zigzag_geometry.py  ← *** 3-D zig-zag geometry viewer ***
├── report/
├── slides/
├── README.md
└── requirements.txt
```

---

## Key improvements over v3 starter

| Area | Change |
|------|--------|
| **Geometry** | `L_mm` corrected to **200 mm** (was 150 mm); full zig-zag 3-D geometry parameters added to config |
| **Visualisation** | New `visualize_zigzag_geometry.py` with 4 figures: 2-D side view, 3-D undeformed, 3-D deformed family, geometry summary table |
| **Training** | Cosine-annealing LR schedule; gradient clipping; separate LR group for `log_EI` / `log_kp`; normalised physics residual |
| **Evaluate** | Relative L2 error; per-pressure tip error plot; parameter identification bar chart; nonlinear blocked-force comparison |
| **Code quality** | `sys.path` handled in every script; docstrings; consistent style |

---

## Customising the geometry

Open `src/actuator_config.py` and edit the `geo_*` keys:

```python
"geo_L_mm":          200.0,   # total length [mm]
"geo_N_teeth":        10,     # number of zig-zag teeth
"geo_tooth_pitch_mm": None,   # None = auto-computed
```

Then run `visualize_zigzag_geometry.py` to see the updated shape immediately,
and `data_generation.py` to regenerate the training data.

---

## Limitations and next steps

- The model is a **reduced-order surrogate** — not a 3-D FEM solution.
- Next step: export FEM centerline data and replace the synthetic dataset.
- Potential extensions: add geometry-dependent compliance factor (variable `g(s; μ)`),
  multi-chamber actuators, or neural operator (DeepONet) for parametric sweep.
