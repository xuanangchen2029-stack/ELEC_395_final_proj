from __future__ import annotations

import os
import numpy as np
import pandas as pd

from reduced_model import get_default_physical_config, solve_shape, blocked_force_constrained


def _assign_split(pressures, train_frac=0.6, val_frac=0.2):
    pressures = np.array(sorted(np.unique(np.asarray(pressures, dtype=float))))
    n = len(pressures)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    split_map = {}
    for i, p in enumerate(pressures):
        if i < n_train:
            split_map[p] = "train"
        elif i < n_train + n_val:
            split_map[p] = "val"
        else:
            split_map[p] = "test"
    return split_map



def generate_synthetic_dataset(
    save_dir="data/synthetic",
    config=None,
    pressures_kpa=None,
    n_shape_points=None,
    noise_std_mm=0.0,
    noise_std_angle_rad=0.0,
    noise_std_force_N=0.0,
    seed=0,
    train_frac=0.6,
    val_frac=0.2,
):
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    cfg = get_default_physical_config()
    if config is not None:
        cfg.update(config)
        cfg["EI_eff_Nmm2"] = cfg.get("EI_eff_Nmm2", get_default_physical_config()["EI_eff_Nmm2"])

    if n_shape_points is None:
        n_shape_points = int(cfg.get("default_n_shape_points", 25))

    if pressures_kpa is None:
        pressures_kpa = np.arange(cfg["p_min_kpa"], cfg["p_max_kpa"] + 1e-9, cfg.get("pressure_step_kpa", 10.0))

    split_map = _assign_split(pressures_kpa, train_frac=train_frac, val_frac=val_frac)

    rows_shape, rows_tip, rows_block = [], [], []

    for p_input_kpa in pressures_kpa:
        split = split_map[float(p_input_kpa)]
        sol = solve_shape(
            L_mm=cfg["L_mm"],
            p_input_kpa=p_input_kpa,
            EI_eff_Nmm2=cfg["EI_eff_Nmm2"],
            k_p_N_per_kPa=cfg["k_p_N_per_kPa"],
            n_points=240,
        )
        idx = np.linspace(0, len(sol["s_mm"]) - 1, n_shape_points, dtype=int)

        for j in idx:
            rows_shape.append({
                "split": split,
                "s_mm": sol["s_mm"][j],
                "s_over_L": sol["s_mm"][j] / cfg["L_mm"],
                "p_kpa": p_input_kpa,
                "x_mm": sol["x_mm"][j] + noise_std_mm * rng.normal(),
                "y_mm": sol["y_mm"][j] + noise_std_mm * rng.normal(),
            })

        rows_tip.append({
            "split": split,
            "p_kpa": p_input_kpa,
            "tip_y_mm": sol["tip_y_mm"] + noise_std_mm * rng.normal(),
            "tip_angle_rad": sol["tip_angle_rad"] + noise_std_angle_rad * rng.normal(),
        })

        rows_block.append({
            "split": split,
            "p_kpa": p_input_kpa,
            "F_b_N": blocked_force_constrained(
                L_mm=cfg["L_mm"],
                p_input_kpa=p_input_kpa,
                EI_eff_Nmm2=cfg["EI_eff_Nmm2"],
                k_p_N_per_kPa=cfg["k_p_N_per_kPa"],
                n_points=240,
            ) + noise_std_force_N * rng.normal(),
        })

    df_shape = pd.DataFrame(rows_shape)
    df_tip = pd.DataFrame(rows_tip)
    df_block = pd.DataFrame(rows_block)

    df_shape.to_csv(os.path.join(save_dir, "shape_data.csv"), index=False)
    df_tip.to_csv(os.path.join(save_dir, "tip_data.csv"), index=False)
    df_block.to_csv(os.path.join(save_dir, "blocked_force_data.csv"), index=False)

    meta = pd.DataFrame([{
        "actuator_name": cfg["name"],
        "L_mm": cfg["L_mm"],
        "outer_width_mm": cfg["outer_width_mm"],
        "outer_height_mm": cfg["outer_height_mm"],
        "wall_thickness_mm": cfg["wall_thickness_mm"],
        "zigzag_compliance_factor": cfg["zigzag_compliance_factor"],
        "effective_modulus_kpa": cfg["effective_modulus_kpa"],
        "EI_eff_true_Nmm2": cfg["EI_eff_Nmm2"],
        "k_p_true_N_per_kPa": cfg["k_p_N_per_kPa"],
        "p_min_kpa": cfg["p_min_kpa"],
        "p_max_kpa": cfg["p_max_kpa"],
        "pressure_step_kpa": cfg["pressure_step_kpa"],
        "noise_std_mm": noise_std_mm,
        "noise_std_angle_rad": noise_std_angle_rad,
        "noise_std_force_N": noise_std_force_N,
        "x_unit": cfg["length_unit"],
        "y_unit": cfg["length_unit"],
        "s_unit": cfg["length_unit"],
        "pressure_unit": cfg["pressure_unit"],
        "force_unit": cfg["force_unit"],
        "angle_unit": cfg["angle_unit"],
        "blocked_force_definition": "tip reaction force required to enforce y_tip = 0 under pressure loading",
        "pressures_train_kpa": ",".join(f"{p:.3f}" for p, s in split_map.items() if s == "train"),
        "pressures_val_kpa": ",".join(f"{p:.3f}" for p, s in split_map.items() if s == "val"),
        "pressures_test_kpa": ",".join(f"{p:.3f}" for p, s in split_map.items() if s == "test"),
    }])
    meta.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

    print(f"Saved synthetic dataset to {save_dir}")
    print(meta.T)
    print(df_tip.groupby("split").size())


if __name__ == "__main__":
    generate_synthetic_dataset()
