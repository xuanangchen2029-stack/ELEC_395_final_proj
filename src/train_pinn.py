from __future__ import annotations

import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pinn_model import SoftActuatorPINN, physics_residuals, gradients, blocked_force_hat_linear



def to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device).view(-1, 1)



def split_df(df, split):
    return df[df["split"] == split].reset_index(drop=True)



def sample_collocation(nc, device, L_mm, p_min_kpa, p_max_kpa):
    s = L_mm * torch.rand(nc, 1, device=device, requires_grad=True)
    p = p_min_kpa + (p_max_kpa - p_min_kpa) * torch.rand(nc, 1, device=device, requires_grad=True)
    return s, p



def _make_bundle(shape_df, tip_df, block_df, split, device):
    shape_df = split_df(shape_df, split)
    tip_df = split_df(tip_df, split)
    block_df = split_df(block_df, split)
    return {
        "shape_df": shape_df,
        "tip_df": tip_df,
        "block_df": block_df,
        "s_d": to_tensor(shape_df["s_mm"].values, device),
        "p_d": to_tensor(shape_df["p_kpa"].values, device),
        "x_d": to_tensor(shape_df["x_mm"].values, device),
        "y_d": to_tensor(shape_df["y_mm"].values, device),
        "p_t": to_tensor(tip_df["p_kpa"].values, device),
        "tip_y": to_tensor(tip_df["tip_y_mm"].values, device),
        "tip_angle": to_tensor(tip_df["tip_angle_rad"].values, device),
        "p_bf": to_tensor(block_df["p_kpa"].values, device),
        "F_b": to_tensor(block_df["F_b_N"].values, device),
    }



def supervised_losses(model, bundle, mse, L_mm, F_scale_N):
    x_hat, y_hat, _ = model(bundle["s_d"], bundle["p_d"])
    loss_shape = mse(x_hat / L_mm, bundle["x_d"] / L_mm) + mse(y_hat / L_mm, bundle["y_d"] / L_mm)

    s_tip = L_mm * torch.ones((len(bundle["tip_df"]), 1), dtype=torch.float32, device=bundle["p_t"].device)
    _, y_tip_hat, th_tip_hat = model(s_tip, bundle["p_t"])
    loss_tip = mse(y_tip_hat / L_mm, bundle["tip_y"] / L_mm) + mse(th_tip_hat, bundle["tip_angle"])

    F_b_hat = blocked_force_hat_linear(model, bundle["p_bf"])
    loss_block = mse(F_b_hat / F_scale_N, bundle["F_b"] / F_scale_N)
    return loss_shape, loss_tip, loss_block



def evaluate_val(model, bundle, mse, L_mm, F_scale_N):
    model.eval()
    with torch.no_grad():
        loss_shape, loss_tip, loss_block = supervised_losses(model, bundle, mse, L_mm, F_scale_N)
        val_total = 20.0 * loss_shape + 20.0 * loss_tip + 10.0 * loss_block
    model.train()
    return {
        "val_total": float(val_total.item()),
        "val_shape": float(loss_shape.item()),
        "val_tip": float(loss_tip.item()),
        "val_block": float(loss_block.item()),
    }



def train(data_dir="data/synthetic", out_dir="results", n_iters=4000, lr=1e-3, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shape_df = pd.read_csv(os.path.join(data_dir, "shape_data.csv"))
    tip_df = pd.read_csv(os.path.join(data_dir, "tip_data.csv"))
    block_df = pd.read_csv(os.path.join(data_dir, "blocked_force_data.csv"))
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    L_mm = float(meta.loc[0, "L_mm"])
    p_min_kpa = float(meta.loc[0, "p_min_kpa"])
    p_max_kpa = float(meta.loc[0, "p_max_kpa"])
    EI_true = float(meta.loc[0, "EI_eff_true_Nmm2"])
    kp_true = float(meta.loc[0, "k_p_true_N_per_kPa"])
    F_scale_N = max((8.0 / 3.0) * kp_true * p_max_kpa, 1e-6)

    train_bundle = _make_bundle(shape_df, tip_df, block_df, "train", device)
    val_bundle = _make_bundle(shape_df, tip_df, block_df, "val", device)

    model = SoftActuatorPINN(
        width=64,
        depth=4,
        length_scale_mm=L_mm,
        pressure_scale_kpa=p_max_kpa,
        EI_scale_Nmm2=EI_true,
        kp_scale_N_per_kPa=kp_true,
    ).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best = {"val_total": np.inf, "iter": -1, "state_dict": None}

    for it in range(n_iters):
        opt.zero_grad()

        s_f, p_f = sample_collocation(512, device, L_mm, p_min_kpa, p_max_kpa)
        r_x, r_y, r_theta = physics_residuals(model, s_f, p_f)
        loss_phys = r_x.pow(2).mean() + r_y.pow(2).mean() + (r_theta / (kp_true * p_max_kpa)).pow(2).mean()

        p_bc = p_min_kpa + (p_max_kpa - p_min_kpa) * torch.rand(128, 1, device=device)
        s0 = torch.zeros_like(p_bc, requires_grad=True)
        sL = L_mm * torch.ones_like(p_bc, requires_grad=True)
        x0, y0, th0 = model(s0, p_bc)
        _, _, thL = model(sL, p_bc)
        thL_s = gradients(thL, sL)
        loss_bc = (x0 / L_mm).pow(2).mean() + (y0 / L_mm).pow(2).mean() + th0.pow(2).mean() + (thL_s * L_mm).pow(2).mean()

        loss_shape, loss_tip, loss_block = supervised_losses(model, train_bundle, mse, L_mm, F_scale_N)
        loss = 1.0 * loss_phys + 10.0 * loss_bc + 20.0 * loss_shape + 20.0 * loss_tip + 10.0 * loss_block
        loss.backward()
        opt.step()

        val_stats = evaluate_val(model, val_bundle, mse, L_mm, F_scale_N)
        row = {
            "iter": it,
            "loss": float(loss.item()),
            "phys": float(loss_phys.item()),
            "bc": float(loss_bc.item()),
            "shape": float(loss_shape.item()),
            "tip": float(loss_tip.item()),
            "block": float(loss_block.item()),
            "EI_eff_Nmm2": float(model.EI_eff.item()),
            "k_p_N_per_kPa": float(model.k_p.item()),
            **val_stats,
        }
        history.append(row)

        if val_stats["val_total"] < best["val_total"]:
            best = {"val_total": val_stats["val_total"], "iter": it, "state_dict": copy.deepcopy(model.state_dict())}

        if it % 200 == 0:
            print(
                f"it={it:5d} | train={loss.item():.3e} | val={val_stats['val_total']:.3e} "
                f"| EI={model.EI_eff.item():.2f} Nmm^2 | k_p={model.k_p.item():.5f} N/kPa"
            )

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "pinn_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "pinn_model_last.pt"))
    if best["state_dict"] is not None:
        torch.save(best["state_dict"], os.path.join(out_dir, "pinn_model_best.pt"))
    pd.DataFrame([{"best_iter": best["iter"], "best_val_total": best["val_total"]}]).to_csv(
        os.path.join(out_dir, "pinn_best_summary.csv"), index=False
    )
    print("Saved PINN results.")


if __name__ == "__main__":
    train()
