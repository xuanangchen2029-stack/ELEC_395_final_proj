from __future__ import annotations
"""
evaluate.py  ── extended (v4)
==============================
Changes vs original:
  1. plot_shape_examples: 8 pressures [10,20,30,40,55,70,85,100] in 2×4 layout
  2. New plot_complete_summary: single comprehensive figure for paper
  3. Geometry visualisation: calls visualize_zigzag_geometry.run_all() if available
  4. All original functions preserved unchanged.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pinn_model  import SoftActuatorPINN, MLP, blocked_force_hat_linear
from reduced_model import solve_shape, blocked_force_constrained

plt.style.use("ggplot")

# ── constants ────────────────────────────────────────────────────────────────
# All 8 pressures used for detailed centerline comparison.
# Low/mid/high regions make it easy to see where PINN works well.
SHAPE_PRESSURES = [10.0, 20.0, 30.0, 40.0, 55.0, 70.0, 85.0, 100.0]


def to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device).view(-1, 1)


def unpack_outputs(model, s, p):
    out = model(s, p)
    if isinstance(out, tuple):
        return out[0], out[1], out[2]
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def load_models(out_dir, meta, device):
    L_mm      = float(meta.loc[0, "L_mm"])
    p_max_kpa = float(meta.loc[0, "p_max_kpa"])
    EI_true   = float(meta.loc[0, "EI_eff_true_Nmm2"])
    kp_true   = float(meta.loc[0, "k_p_true_N_per_kPa"])

    pinn = SoftActuatorPINN(width=64, depth=4,
                            length_scale_mm=L_mm, pressure_scale_kpa=p_max_kpa,
                            EI_scale_Nmm2=EI_true, kp_scale_N_per_kPa=kp_true).to(device)
    pinn.load_state_dict(torch.load(os.path.join(out_dir, "pinn_model_best.pt"),
                                    map_location=device, weights_only=True))
    pinn.eval()

    mlp = MLP(in_dim=2, out_dim=3, width=64, depth=4,
              length_scale_mm=L_mm, pressure_scale_kpa=p_max_kpa).to(device)
    mlp.load_state_dict(torch.load(os.path.join(out_dir, "mlp_model_best.pt"),
                                   map_location=device, weights_only=True))
    mlp.eval()
    return pinn, mlp


def metrics(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    rmse   = float(np.sqrt(np.mean((a - b) ** 2)))
    mae    = float(np.mean(np.abs(a - b)))
    rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    return {"rmse": rmse, "mae": mae, "rel_l2": rel_l2}


def eval_on_split(shape_df, tip_df, block_df, split, pinn, mlp, device, meta):
    sdf = shape_df[shape_df["split"] == split].reset_index(drop=True)
    tdf = tip_df[tip_df["split"]   == split].reset_index(drop=True)
    bdf = block_df[block_df["split"]== split].reset_index(drop=True)
    L_mm = float(meta.loc[0, "L_mm"])

    with torch.no_grad():
        s = to_tensor(sdf["s_mm"].values, device)
        p = to_tensor(sdf["p_kpa"].values, device)
        px, py, _ = unpack_outputs(pinn, s, p)
        mx, my, _ = unpack_outputs(mlp,  s, p)

        st = L_mm * torch.ones((len(tdf), 1), dtype=torch.float32, device=device)
        pt = to_tensor(tdf["p_kpa"].values, device)
        _, py_tip, pth_tip = unpack_outputs(pinn, st, pt)
        _, my_tip, mth_tip = unpack_outputs(mlp,  st, pt)
        F_b_hat = blocked_force_hat_linear(pinn, to_tensor(bdf["p_kpa"].values, device))

    out = {
        "split": split,
        "pinn_shape_x_rmse_mm":   metrics(px.cpu().numpy(), sdf["x_mm"].values)["rmse"],
        "pinn_shape_y_rmse_mm":   metrics(py.cpu().numpy(), sdf["y_mm"].values)["rmse"],
        "mlp_shape_x_rmse_mm":    metrics(mx.cpu().numpy(), sdf["x_mm"].values)["rmse"],
        "mlp_shape_y_rmse_mm":    metrics(my.cpu().numpy(), sdf["y_mm"].values)["rmse"],
        "pinn_tip_y_rmse_mm":     metrics(py_tip.cpu().numpy(), tdf["tip_y_mm"].values)["rmse"],
        "mlp_tip_y_rmse_mm":      metrics(my_tip.cpu().numpy(), tdf["tip_y_mm"].values)["rmse"],
        "pinn_tip_angle_rmse_rad":metrics(pth_tip.cpu().numpy(), tdf["tip_angle_rad"].values)["rmse"],
        "mlp_tip_angle_rmse_rad": metrics(mth_tip.cpu().numpy(), tdf["tip_angle_rad"].values)["rmse"],
        "pinn_block_rmse_N":      metrics(F_b_hat.cpu().numpy(), bdf["F_b_N"].values)["rmse"],
    }
    if split == "test":
        out.update({
            "EI_identified_Nmm2":     float(pinn.EI_eff.item()),
            "k_p_identified_N_per_kPa": float(pinn.k_p.item()),
            "EI_true_Nmm2":           float(meta.loc[0, "EI_eff_true_Nmm2"]),
            "k_p_true_N_per_kPa":     float(meta.loc[0, "k_p_true_N_per_kPa"]),
            "EI_pct_err": abs(float(pinn.EI_eff.item()) - float(meta.loc[0,"EI_eff_true_Nmm2"]))
                          / float(meta.loc[0,"EI_eff_true_Nmm2"]) * 100,
            "kp_pct_err": abs(float(pinn.k_p.item()) - float(meta.loc[0,"k_p_true_N_per_kPa"]))
                          / float(meta.loc[0,"k_p_true_N_per_kPa"]) * 100,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Original plots (preserved exactly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_histories(out_dir, meta):
    pinn_hist = pd.read_csv(os.path.join(out_dir, "pinn_history.csv"))
    mlp_hist  = pd.read_csv(os.path.join(out_dir, "mlp_history.csv"))
    EI_true   = float(meta.loc[0, "EI_eff_true_Nmm2"])
    kp_true   = float(meta.loc[0, "k_p_true_N_per_kPa"])

    plt.figure(figsize=(7, 4.2))
    plt.semilogy(pinn_hist["iter"], pinn_hist["loss"],      label="PINN train")
    plt.semilogy(pinn_hist["iter"], pinn_hist["val_total"], label="PINN val")
    plt.semilogy(mlp_hist["iter"],  mlp_hist["loss"],       label="MLP train")
    plt.semilogy(mlp_hist["iter"],  mlp_hist["val_total"],  label="MLP val")
    plt.xlabel("Training iteration"); plt.ylabel("Total loss (log scale)")
    plt.title("Training and validation histories"); plt.grid(True, alpha=0.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_validation_histories.png"), dpi=220)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(pinn_hist["iter"], pinn_hist["EI_eff_Nmm2"],   label=r"Identified $EI_{eff}$", lw=2)
    ax1.axhline(EI_true, color="C0", ls="--", alpha=0.6,   label=r"True $EI_{eff}$")
    ax1.set_xlabel("Training iteration"); ax1.set_ylabel(r"$EI_{eff}$ (N$\cdot$mm$^2$)")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(pinn_hist["iter"], pinn_hist["k_p_N_per_kPa"], color="C1", label=r"Identified $k_p$", lw=2)
    ax2.axhline(kp_true, color="C1", ls="--", alpha=0.6,   label=r"True $k_p$")
    ax2.set_ylabel(r"$k_p$ (N/kPa)")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, loc="best")
    plt.title("PINN parameter trajectories"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "identified_parameters.png"), dpi=220)
    plt.close()


def plot_tip_curves(out_dir, pinn, mlp, meta, device):
    L_mm      = float(meta.loc[0, "L_mm"])
    EI_true   = float(meta.loc[0, "EI_eff_true_Nmm2"])
    k_p_true  = float(meta.loc[0, "k_p_true_N_per_kPa"])
    p_min_kpa = float(meta.loc[0, "p_min_kpa"])
    p_max_kpa = float(meta.loc[0, "p_max_kpa"])
    p_grid    = np.linspace(p_min_kpa, p_max_kpa, 120)

    with torch.no_grad():
        s_tip   = L_mm * torch.ones((len(p_grid), 1), dtype=torch.float32, device=device)
        p_t     = to_tensor(p_grid, device)
        _, y_pinn, _ = unpack_outputs(pinn, s_tip, p_t)
        _, y_mlp, _  = unpack_outputs(mlp,  s_tip, p_t)
        tip_pinn     = y_pinn.cpu().numpy().ravel()
        tip_mlp      = y_mlp.cpu().numpy().ravel()
        block_pinn   = blocked_force_hat_linear(pinn, p_t).cpu().numpy().ravel()

    tip_true  = [solve_shape(L_mm=L_mm, p_input_kpa=p, EI_eff_Nmm2=EI_true,
                             k_p_N_per_kPa=k_p_true)["tip_y_mm"] for p in p_grid]
    block_true= [blocked_force_constrained(L_mm=L_mm, p_input_kpa=p, EI_eff_Nmm2=EI_true,
                                           k_p_N_per_kPa=k_p_true) for p in p_grid]

    plt.figure(figsize=(7, 4.2))
    plt.plot(p_grid, tip_true,  lw=2, label="Reference")
    plt.plot(p_grid, tip_pinn, "--", lw=2, label="PINN")
    plt.plot(p_grid, tip_mlp,  ":",  lw=2, label="MLP")
    plt.xlabel("Pressure (kPa)"); plt.ylabel("Tip displacement (mm)")
    plt.title("Pressure-to-tip response"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tip_response_curve.png"), dpi=220); plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.plot(p_grid, block_true,  lw=2, label="Reference blocked-force")
    plt.plot(p_grid, block_pinn, "--", lw=2, label="PINN linear blocked-force")
    plt.xlabel("Pressure (kPa)"); plt.ylabel("Tip-blocked force (N)")
    plt.title("Pressure-to-tip-blocked-force response"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "blocked_force_curve.png"), dpi=220); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Extended shape examples: 8 pressures in 2×4 grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_shape_examples(out_dir, pinn, mlp, meta, device,
                        pressures=None):
    """
    Plot deformed centerline (Reference / PINN / MLP) at multiple pressures.
    Default: [10, 20, 30, 40, 55, 70, 85, 100] kPa  →  2 rows × 4 cols.
    """
    if pressures is None:
        pressures = SHAPE_PRESSURES

    L_mm     = float(meta.loc[0, "L_mm"])
    EI_true  = float(meta.loc[0, "EI_eff_true_Nmm2"])
    k_p_true = float(meta.loc[0, "k_p_true_N_per_kPa"])
    s_grid   = np.linspace(0.0, L_mm, 200)

    n     = len(pressures)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.6))
    axes = np.array(axes).flatten()

    for idx, p in enumerate(pressures):
        ax = axes[idx]
        ref = solve_shape(L_mm=L_mm, p_input_kpa=p,
                          EI_eff_Nmm2=EI_true, k_p_N_per_kPa=k_p_true,
                          n_points=len(s_grid))
        s_t = to_tensor(s_grid, device)
        p_t = to_tensor(np.full_like(s_grid, p), device)
        with torch.no_grad():
            px, py, _ = unpack_outputs(pinn, s_t, p_t)
            mx, my, _ = unpack_outputs(mlp,  s_t, p_t)

        px = px.cpu().numpy().ravel(); py = py.cpu().numpy().ravel()
        mx = mx.cpu().numpy().ravel(); my = my.cpu().numpy().ravel()

        ax.plot(ref["x_mm"], ref["y_mm"], lw=2,    color="C0", label="Reference")
        ax.plot(px,          py,          lw=1.8, color="C1", ls="--", label="PINN")
        ax.plot(mx,          my,          lw=1.8, color="C2", ls=":",  label="MLP")

        # Tip-y error
        pinn_err = abs(py[-1] - ref["tip_y_mm"])
        mlp_err  = abs(my[-1] - ref["tip_y_mm"])
        ax.set_title(f"p = {p:.0f} kPa\nPINN err={pinn_err:.1f} mm  MLP err={mlp_err:.1f} mm",
                     fontsize=8.5)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.5)

    # Hide empty subplots
    for idx in range(len(pressures), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Deformed Centerline Comparison: Reference vs PINN vs MLP\n"
        "Tip error labels show where the PINN fits well (small error) "
        "vs where the reduced-order model breaks down (large error)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shape_examples.png"), dpi=220, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# New: complete summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_complete_summary(out_dir, pinn, mlp, meta, device):
    """
    Single comprehensive 3×4 panel figure.
    Row 0: training summary (loss, PINN breakdown, param convergence, param bar)
    Row 1: tip response + absolute error + blocked force
    Row 2: three centreline panels at low/mid/high pressure
    """
    L_mm      = float(meta.loc[0, "L_mm"])
    EI_true   = float(meta.loc[0, "EI_eff_true_Nmm2"])
    k_p_true  = float(meta.loc[0, "k_p_true_N_per_kPa"])
    p_min_kpa = float(meta.loc[0, "p_min_kpa"])
    p_max_kpa = float(meta.loc[0, "p_max_kpa"])
    p_grid    = np.linspace(p_min_kpa, p_max_kpa, 150)

    pinn_hist = pd.read_csv(os.path.join(out_dir, "pinn_history.csv"))
    mlp_hist  = pd.read_csv(os.path.join(out_dir, "mlp_history.csv"))

    # Reference curves
    ref_ty = np.array([solve_shape(L_mm=L_mm, p_input_kpa=p,
                                    EI_eff_Nmm2=EI_true, k_p_N_per_kPa=k_p_true)["tip_y_mm"]
                        for p in p_grid])
    ref_bl = np.array([blocked_force_constrained(L_mm=L_mm, p_input_kpa=p,
                        EI_eff_Nmm2=EI_true, k_p_N_per_kPa=k_p_true) for p in p_grid])

    with torch.no_grad():
        s_tip = L_mm * torch.ones(len(p_grid), 1, dtype=torch.float32, device=device)
        p_t   = to_tensor(p_grid, device)
        _, py_, _ = unpack_outputs(pinn, s_tip, p_t)
        _, my_, _ = unpack_outputs(mlp,  s_tip, p_t)
        Fb_hat    = blocked_force_hat_linear(pinn, p_t).cpu().numpy().ravel()
    py_ = py_.cpu().numpy().ravel(); my_ = my_.cpu().numpy().ravel()

    EI_id   = float(pinn.EI_eff.item()); kp_id = float(pinn.k_p.item())
    EI_err  = abs(EI_id - EI_true) / EI_true * 100
    kp_err  = abs(kp_id - k_p_true) / k_p_true * 100
    pinn_rmse = np.sqrt(np.mean((py_ - ref_ty)**2))
    mlp_rmse  = np.sqrt(np.mean((my_ - ref_ty)**2))

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.46, wspace=0.34)

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(pinn_hist["iter"], pinn_hist["loss"],      lw=1.5, label="PINN train")
    ax.semilogy(pinn_hist["iter"], pinn_hist["val_total"], lw=1.5, ls="--", label="PINN val")
    ax.semilogy(mlp_hist["iter"],  mlp_hist["loss"],       lw=1.5, label="MLP train")
    ax.semilogy(mlp_hist["iter"],  mlp_hist["val_total"],  lw=1.5, ls="--", label="MLP val")
    ax.set_title("(a) Loss history"); ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3); ax.set_xlabel("Iteration")

    ax = fig.add_subplot(gs[0, 1])
    for col, lbl, ls_ in [("phys","Physics","-"),("bc","BC","-"),
                            ("shape","Shape","-"),("tip","Tip","--"),("block","Block F",":")]:
        if col in pinn_hist.columns:
            ax.semilogy(pinn_hist["iter"], pinn_hist[col], lw=1.2, label=lbl, ls=ls_)
    ax.set_title("(b) PINN loss breakdown"); ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3); ax.set_xlabel("Iteration")

    ax = fig.add_subplot(gs[0, 2]); ax2_ = ax.twinx()
    ax.plot(pinn_hist["iter"], pinn_hist["EI_eff_Nmm2"],   color="C0", lw=2, label="EI_eff")
    ax.axhline(EI_true, color="C0", ls="--", alpha=0.7,   label=f"True {EI_true:.0f}")
    ax2_.plot(pinn_hist["iter"], pinn_hist["k_p_N_per_kPa"], color="C1", lw=2, label="k_p")
    ax2_.axhline(k_p_true, color="C1", ls="--", alpha=0.7)
    ax.set_ylabel("EI_eff [N·mm²]", color="C0", fontsize=8)
    ax2_.set_ylabel("k_p [N/kPa]", color="C1", fontsize=8)
    ax.tick_params(axis="y", labelcolor="C0"); ax2_.tick_params(axis="y", labelcolor="C1")
    lines = ax.get_lines() + ax2_.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=6.5, loc="lower right")
    ax.set_title("(c) Parameter convergence"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    clr_ei = "#70AD47" if EI_err < 5 else ("#ED7D31" if EI_err < 15 else "#FF4444")
    clr_kp = "#70AD47" if kp_err < 5 else ("#ED7D31" if kp_err < 15 else "#FF4444")
    ax.bar([0, 1], [EI_true, EI_id], color=["#5B9BD5", clr_ei],
           width=0.5, edgecolor="k", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["True", "Identified"], fontsize=8)
    ax.set_ylabel("EI_eff [N·mm²]")
    for xi, v in enumerate([EI_true, EI_id]):
        ax.text(xi, v * 1.01, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title(f"(d) EI err={EI_err:.1f}%  kp err={kp_err:.1f}%")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(p_grid, ref_ty, lw=2.5, label="Reference", color="C0")
    ax.plot(p_grid, py_,    lw=2,   label=f"PINN  RMSE={pinn_rmse:.1f} mm", ls="--", color="C1")
    ax.plot(p_grid, my_,    lw=2,   label=f"MLP   RMSE={mlp_rmse:.1f} mm",  ls=":",  color="C2")
    ax.set_xlabel("Pressure [kPa]"); ax.set_ylabel("Tip y-displacement [mm]")
    ax.set_title("(e) Tip displacement"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(p_grid, np.abs(py_ - ref_ty), lw=2, color="C1",
            label=f"PINN max={np.max(np.abs(py_-ref_ty)):.1f}mm")
    ax.plot(p_grid, np.abs(my_ - ref_ty), lw=2, color="C2", ls="--",
            label=f"MLP  max={np.max(np.abs(my_-ref_ty)):.1f}mm")
    ax.set_xlabel("Pressure [kPa]"); ax.set_ylabel("|Tip error| [mm]")
    ax.set_title("(f) Tip error"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(p_grid, ref_bl, lw=2.5, label="Ref (nonlinear)", color="C0")
    ax.plot(p_grid, Fb_hat, lw=2,   label="PINN (linear)",   ls="--", color="C1")
    Fb_rmse = np.sqrt(np.mean((Fb_hat - ref_bl)**2))
    ax.set_xlabel("Pressure [kPa]"); ax.set_ylabel("Blocked force [N]")
    ax.set_title(f"(g) Blocked force  RMSE={Fb_rmse*1e3:.2f} mN")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Row 2: three centrelines ──────────────────────────────────────────────
    s_g   = np.linspace(0, L_mm, 200)
    ex_ps = [p_min_kpa, (p_min_kpa + p_max_kpa) / 2, p_max_kpa]
    for i, p in enumerate(ex_ps):
        ax = fig.add_subplot(gs[2, i])
        ref = solve_shape(L_mm=L_mm, p_input_kpa=p, EI_eff_Nmm2=EI_true,
                          k_p_N_per_kPa=k_p_true, n_points=200)
        s_t = to_tensor(s_g, device); p_t = to_tensor(np.full_like(s_g, p), device)
        with torch.no_grad():
            ppx, ppy, _ = unpack_outputs(pinn, s_t, p_t)
            mpx, mpy, _ = unpack_outputs(mlp,  s_t, p_t)
        ppx=ppx.cpu().numpy().ravel(); ppy=ppy.cpu().numpy().ravel()
        mpx=mpx.cpu().numpy().ravel(); mpy=mpy.cpu().numpy().ravel()
        ax.plot(ref["x_mm"], ref["y_mm"], lw=2.5, label="Reference", color="C0")
        ax.plot(ppx, ppy, lw=2, ls="--", label="PINN", color="C1")
        ax.plot(mpx, mpy, lw=2, ls=":",  label="MLP",  color="C2")
        te = abs(ppy[-1] - ref["tip_y_mm"])
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"(h{i+1}) p={p:.0f} kPa  PINN tip err={te:.1f} mm")
        if i == 0: ax.legend(fontsize=8)

    # Metrics table (last panel)
    ax = fig.add_subplot(gs[2, 3]); ax.axis("off")
    t_rows = [
        ["Metric", "PINN", "MLP"],
        ["Tip RMSE (mm)", f"{pinn_rmse:.2f}", f"{mlp_rmse:.2f}"],
        ["Tip max err (mm)", f"{np.max(np.abs(py_-ref_ty)):.1f}", f"{np.max(np.abs(my_-ref_ty)):.1f}"],
        ["Blocked F RMSE (mN)", f"{Fb_rmse*1e3:.2f}", "N/A"],
        ["EI_eff error (%)", f"{EI_err:.2f}", "N/A"],
        ["k_p error (%)", f"{kp_err:.2f}", "N/A"],
    ]
    tbl = ax.table(cellText=[r for r in t_rows[1:]], colLabels=t_rows[0],
                   cellLoc="center", loc="center", bbox=[0, 0.2, 1, 0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#2F5496"); cell.set_text_props(color="white", fontweight="bold")
        elif ri % 2 == 0:
            cell.set_facecolor("#D9E1F2")
        cell.set_edgecolor("#BFBFBF")
    ax.set_title("(i) Metrics summary", pad=10)

    fig.suptitle(
        f"Complete Results Summary — Original PINN (width=64, depth=4, n_iters=4000)\n"
        f"EI err={EI_err:.2f}%  |  kp err={kp_err:.2f}%  |  "
        f"PINN tip RMSE={pinn_rmse:.1f} mm  |  MLP tip RMSE={mlp_rmse:.1f} mm",
        fontsize=11, fontweight="bold", y=1.01,
    )
    out = os.path.join(out_dir, "complete_summary.png")
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(data_dir="data/synthetic", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape_df = pd.read_csv(os.path.join(data_dir, "shape_data.csv"))
    tip_df   = pd.read_csv(os.path.join(data_dir, "tip_data.csv"))
    block_df = pd.read_csv(os.path.join(data_dir, "blocked_force_data.csv"))
    meta     = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    pinn, mlp = load_models(out_dir, meta, device)

    rows = [eval_on_split(shape_df, tip_df, block_df, sp,
                          pinn, mlp, device, meta)
            for sp in ["train", "val", "test"]]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("Generating figures ...")
    plot_histories(out_dir, meta)
    plot_tip_curves(out_dir, pinn, mlp, meta, device)
    plot_shape_examples(out_dir, pinn, mlp, meta, device)      # 8-pressure 2×4 grid
    plot_complete_summary(out_dir, pinn, mlp, meta, device)    # comprehensive panel

    # Generate 3-D geometry figures
    try:
        from visualize_zigzag_geometry import run_all
        fig_dir = os.path.join(out_dir, "..", "results", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        run_all(save_dir=os.path.join(out_dir, "figures"))
    except Exception as e:
        print(f"  [Geometry viz skipped: {e}]")

    print(f"\nSaved figures and metrics to {out_dir}")
    print(metrics_df[["split","pinn_shape_y_rmse_mm","mlp_shape_y_rmse_mm",
                       "pinn_tip_y_rmse_mm","mlp_tip_y_rmse_mm"]].to_string(index=False))


if __name__ == "__main__":
    main()
