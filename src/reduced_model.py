from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from actuator_config import default_actuator_config


DEFAULT_CONFIG = default_actuator_config()


def hollow_box_second_moment_mm4(outer_width_mm, outer_height_mm, wall_thickness_mm):
    bi = max(outer_width_mm - 2.0 * wall_thickness_mm, 1e-6)
    hi = max(outer_height_mm - 2.0 * wall_thickness_mm, 1e-6)
    Io = outer_width_mm * outer_height_mm**3 / 12.0
    Ii = bi * hi**3 / 12.0
    return Io - Ii


def effective_bending_stiffness_from_config(config=None):
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    E_eff_N_per_mm2 = cfg["effective_modulus_kpa"] * 1.0e-3
    I_hollow_mm4 = hollow_box_second_moment_mm4(
        cfg["outer_width_mm"], cfg["outer_height_mm"], cfg["wall_thickness_mm"]
    )
    return E_eff_N_per_mm2 * I_hollow_mm4 * cfg["zigzag_compliance_factor"]



def get_default_physical_config():
    cfg = DEFAULT_CONFIG.copy()
    cfg["EI_eff_Nmm2"] = effective_bending_stiffness_from_config(cfg)
    cfg["tip_force_scale_N"] = constrained_tip_blocked_force_linear(
        p_input_kpa=cfg["p_max_kpa"],
        k_p_N_per_kPa=cfg["k_p_N_per_kPa"],
    )
    return cfg



def ode_system(s_mm, z, L_mm, p_input_kpa, EI_eff_Nmm2, k_p_N_per_kPa, tip_constraint_force_N=0.0):
    """
    z = [theta, theta_s, x, y]

    Reduced-order dimensional model:
        EI_eff * theta_ss + k_p * p - F_tip * (1 - s/L) = 0
        x_s = cos(theta)
        y_s = sin(theta)

    Interpretation:
    - k_p * p is an equivalent distributed actuation term that bends the actuator.
    - F_tip * (1 - s/L) is an effective reaction-force term opposing the pressure-induced bend.
      It is introduced so that the blocked-force case can be posed as a tip-constrained problem.

    This remains a reduced-order surrogate and is not a full 3D chamber-resolved model.
    """
    theta = z[0]
    theta_s = z[1]
    dtheta = theta_s
    force_shape = 1.0 - s_mm / max(L_mm, 1e-9)
    dtheta_s = -(k_p_N_per_kPa * p_input_kpa - tip_constraint_force_N * force_shape) / EI_eff_Nmm2
    dx = np.cos(theta)
    dy = np.sin(theta)
    return np.vstack([dtheta, dtheta_s, dx, dy])



def bc(za, zb):
    return np.array([
        za[0],  # theta(0)=0
        za[2],  # x(0)=0
        za[3],  # y(0)=0
        zb[1],  # theta_s(L)=0
    ])



def solve_shape(
    L_mm=150.0,
    p_input_kpa=50.0,
    EI_eff_Nmm2=2000.0,
    k_p_N_per_kPa=0.0012,
    tip_constraint_force_N=0.0,
    n_points=200,
):
    s = np.linspace(0.0, L_mm, n_points)

    theta_guess = 1e-3 * (s / max(L_mm, 1e-12)) * (L_mm - 0.5 * s)
    theta_s_guess = 1e-3 * (1.0 - s / max(L_mm, 1e-12))
    x_guess = s
    y_guess = np.zeros_like(s)
    z_guess = np.vstack([theta_guess, theta_s_guess, x_guess, y_guess])

    sol = solve_bvp(
        lambda ss, zz: ode_system(ss, zz, L_mm, p_input_kpa, EI_eff_Nmm2, k_p_N_per_kPa, tip_constraint_force_N),
        bc,
        s,
        z_guess,
        max_nodes=10000,
    )
    if not sol.success:
        raise RuntimeError(f"BVP solver failed: {sol.message}")

    z = sol.sol(s)
    return {
        "s_mm": s,
        "theta_rad": z[0],
        "theta_s_per_mm": z[1],
        "x_mm": z[2],
        "y_mm": z[3],
        "tip_angle_rad": z[0, -1],
        "tip_y_mm": z[3, -1],
        "tip_x_mm": z[2, -1],
        "tip_constraint_force_N": tip_constraint_force_N,
    }



def constrained_tip_blocked_force_linear(p_input_kpa, k_p_N_per_kPa):
    """
    Linearized closed-form estimate for the reaction force needed to enforce y(L)=0
    in the present reduced-order model.
    """
    return (8.0 / 3.0) * k_p_N_per_kPa * p_input_kpa



def blocked_force_constrained(
    L_mm=150.0,
    p_input_kpa=50.0,
    EI_eff_Nmm2=2000.0,
    k_p_N_per_kPa=0.0012,
    n_points=200,
):
    """
    Compute the tip-blocked force as the reaction force required to enforce
    y_tip = 0 under pressure loading.

    This is closer to the user's intended experimental definition than the old proxy.
    """

    def tip_y_given_force(F_tip_N):
        sol = solve_shape(
            L_mm=L_mm,
            p_input_kpa=p_input_kpa,
            EI_eff_Nmm2=EI_eff_Nmm2,
            k_p_N_per_kPa=k_p_N_per_kPa,
            tip_constraint_force_N=F_tip_N,
            n_points=n_points,
        )
        return sol["tip_y_mm"]

    y_free = tip_y_given_force(0.0)
    if abs(y_free) < 1e-9:
        return 0.0

    guess = constrained_tip_blocked_force_linear(p_input_kpa, k_p_N_per_kPa)
    upper = max(guess, 1e-6)
    y_upper = tip_y_given_force(upper)
    expand_count = 0
    while y_free * y_upper > 0 and expand_count < 20:
        upper *= 2.0
        y_upper = tip_y_given_force(upper)
        expand_count += 1

    if y_free * y_upper > 0:
        raise RuntimeError("Could not bracket constrained blocked-force root.")

    root = brentq(lambda F: tip_y_given_force(F), 0.0, upper, xtol=1e-8, rtol=1e-8, maxiter=200)
    return float(root)
