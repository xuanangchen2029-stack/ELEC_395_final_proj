from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from actuator_config import default_actuator_config
from reduced_model import get_default_physical_config, solve_shape, blocked_force_constrained


plt.style.use("ggplot")


def main(save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    cfg = get_default_physical_config()
    L_mm = cfg["L_mm"]
    EI_eff = cfg["EI_eff_Nmm2"]
    kp = cfg["k_p_N_per_kPa"]
    p_grid = np.arange(cfg["p_min_kpa"], cfg["p_max_kpa"] + 1e-9, cfg["pressure_step_kpa"])

    # Shape examples
    s_grid_n = 220
    plt.figure(figsize=(8, 5))
    for p in [p_grid[0], p_grid[len(p_grid)//2], p_grid[-1]]:
        sol = solve_shape(L_mm=L_mm, p_input_kpa=p, EI_eff_Nmm2=EI_eff, k_p_N_per_kPa=kp, n_points=s_grid_n)
        plt.plot(sol["x_mm"], sol["y_mm"], label=f"{p:.0f} kPa")
    plt.xlabel("Axial position x (mm)")
    plt.ylabel("Transverse displacement y (mm)")
    plt.title("Simplified zig-zag actuator centerline predictions")
    plt.axis("equal")
    plt.legend(title="Pressure")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualized_shape_family.png"), dpi=220)
    plt.close()

    # Curves
    tip = []
    block = []
    for p in p_grid:
        sol = solve_shape(L_mm=L_mm, p_input_kpa=p, EI_eff_Nmm2=EI_eff, k_p_N_per_kPa=kp)
        tip.append(sol["tip_y_mm"])
        block.append(blocked_force_constrained(L_mm=L_mm, p_input_kpa=p, EI_eff_Nmm2=EI_eff, k_p_N_per_kPa=kp))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(p_grid, tip, marker="o")
    ax1.set_xlabel("Pressure (kPa)")
    ax1.set_ylabel("Tip displacement (mm)")
    ax1.set_title("Pressure response of the simplified zig-zag actuator")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualized_tip_response.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(p_grid, block, marker="o")
    plt.xlabel("Pressure (kPa)")
    plt.ylabel("Tip-blocked force (N)")
    plt.title("Constrained tip-blocked force in the simplified model")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualized_blocked_force.png"), dpi=220)
    plt.close()

    print("Saved visualization figures to", save_dir)
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
