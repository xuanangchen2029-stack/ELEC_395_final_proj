import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pinn_model import MLP


def to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device).view(-1, 1)


def split_df(df, split):
    return df[df["split"] == split].reset_index(drop=True)


def _make_bundle(shape_df, tip_df, split, device):
    shape_df = split_df(shape_df, split)
    tip_df = split_df(tip_df, split)
    return {
        "shape_df": shape_df,
        "tip_df": tip_df,
        "s_d": to_tensor(shape_df["s_mm"].values, device),
        "p_d": to_tensor(shape_df["p_kpa"].values, device),
        "x_d": to_tensor(shape_df["x_mm"].values, device),
        "y_d": to_tensor(shape_df["y_mm"].values, device),
        "p_t": to_tensor(tip_df["p_kpa"].values, device),
        "tip_y": to_tensor(tip_df["tip_y_mm"].values, device),
        "tip_angle": to_tensor(tip_df["tip_angle_rad"].values, device),
    }


def compute_losses(model, bundle, mse, L_mm):
    out = model(bundle["s_d"], bundle["p_d"])
    x_hat = out[:, 0:1]
    y_hat = out[:, 1:2]
    loss_shape = mse(x_hat / L_mm, bundle["x_d"] / L_mm) + mse(y_hat / L_mm, bundle["y_d"] / L_mm)

    s_tip = L_mm * torch.ones((len(bundle["tip_df"]), 1), dtype=torch.float32, device=bundle["p_t"].device)
    out_tip = model(s_tip, bundle["p_t"])
    y_tip_hat = out_tip[:, 1:2]
    th_tip_hat = out_tip[:, 2:3]
    loss_tip = mse(y_tip_hat / L_mm, bundle["tip_y"] / L_mm) + mse(th_tip_hat, bundle["tip_angle"])
    return loss_shape, loss_tip


def train(data_dir="data/synthetic", out_dir="results", n_iters=3000, lr=1e-3, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shape_df = pd.read_csv(os.path.join(data_dir, "shape_data.csv"))
    tip_df = pd.read_csv(os.path.join(data_dir, "tip_data.csv"))
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    L_mm = float(meta.loc[0, "L_mm"])
    p_max_kpa = float(meta.loc[0, "p_max_kpa"])

    train_bundle = _make_bundle(shape_df, tip_df, "train", device)
    val_bundle = _make_bundle(shape_df, tip_df, "val", device)

    model = MLP(in_dim=2, out_dim=3, width=64, depth=4, length_scale_mm=L_mm, pressure_scale_kpa=p_max_kpa).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best = {"val_total": np.inf, "iter": -1, "state_dict": None}
    for it in range(n_iters):
        opt.zero_grad()
        loss_shape, loss_tip = compute_losses(model, train_bundle, mse, L_mm)
        loss = 20.0 * loss_shape + 20.0 * loss_tip
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_shape, val_tip = compute_losses(model, val_bundle, mse, L_mm)
            val_total = 20.0 * val_shape + 20.0 * val_tip
        model.train()

        history.append({
            "iter": it,
            "loss": float(loss.item()),
            "shape": float(loss_shape.item()),
            "tip": float(loss_tip.item()),
            "val_total": float(val_total.item()),
            "val_shape": float(val_shape.item()),
            "val_tip": float(val_tip.item()),
        })

        if val_total.item() < best["val_total"]:
            best = {
                "val_total": float(val_total.item()),
                "iter": it,
                "state_dict": copy.deepcopy(model.state_dict()),
            }

        if it % 200 == 0:
            print(f"it={it:5d} | train={loss.item():.3e} | val={val_total.item():.3e}")

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "mlp_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "mlp_model_last.pt"))
    if best["state_dict"] is not None:
        torch.save(best["state_dict"], os.path.join(out_dir, "mlp_model_best.pt"))
    pd.DataFrame([{"best_iter": best["iter"], "best_val_total": best["val_total"]}]).to_csv(
        os.path.join(out_dir, "mlp_best_summary.csv"), index=False
    )
    print("Saved baseline MLP results.")


if __name__ == "__main__":
    train()
