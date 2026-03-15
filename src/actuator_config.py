from __future__ import annotations


def default_actuator_config():
    """
    Simplified geometry for a zig-zag / bellows-type pneumatic soft bending actuator.

    The model is intentionally reduced-order. It does NOT resolve the full 3D chambered
    geometry. Instead, it uses a hollow rectangular section plus a compliance reduction
    factor to represent the extra flexibility induced by the zig-zag top wall.

    geo_* keys are used only by visualize_zigzag_geometry.py for 3-D rendering.
    They do NOT affect training or the mechanical model.
    """
    return {
        # ── Reduced-order mechanical model (training / evaluation) ────────────
        "name": "zigzag_soft_bending_actuator",
        "L_mm": 200.0,
        "outer_width_mm": 22.0,
        "outer_height_mm": 22.0,
        "wall_thickness_mm": 2.0,
        "zigzag_compliance_factor": 0.85,
        "effective_modulus_kpa": 150.0,
        "k_p_N_per_kPa": 0.0012,
        "p_min_kpa": 10.0,
        "p_max_kpa": 100.0,
        "pressure_step_kpa": 10.0,
        "default_n_shape_points": 25,
        "length_unit": "mm",
        "pressure_unit": "kPa",
        "force_unit": "N",
        "angle_unit": "rad",

        # ── Explicit 3-D zig-zag geometry (visualize_zigzag_geometry.py only) ─
        # Spec: L=20cm, w=22mm, h=22mm, t=2mm, l1=2cm, l2=2cm, N=10 teeth
        "geo_L_mm":          200.0,   # mirrors L_mm above
        "geo_w_mm":           22.0,
        "geo_h_mm":           22.0,
        "geo_t_mm":            2.0,
        "geo_l1_mm":          20.0,   # inlet plain region  [mm]
        "geo_l2_mm":          20.0,   # free-tip plain region [mm]
        "geo_N_teeth":        10,
        "geo_tooth_pitch_mm": None,   # None → auto = active_len / N_teeth
    }


def compute_zigzag_geometry(config=None):
    """
    Derive all dependent zig-zag geometry quantities from the config.
    Returns the config dict enriched with:
        active_length_mm, tooth_pitch_mm, tooth_depth_mm, inner_h_mm, inner_w_mm
    """
    cfg = default_actuator_config()
    if config is not None:
        cfg.update(config)

    L  = cfg["geo_L_mm"]
    l1 = cfg["geo_l1_mm"]
    l2 = cfg["geo_l2_mm"]
    N  = cfg["geo_N_teeth"]
    t  = cfg["geo_t_mm"]
    h  = cfg["geo_h_mm"]
    w  = cfg["geo_w_mm"]

    active = L - l1 - l2
    if active <= 0:
        raise ValueError(f"l1+l2 ({l1+l2}) must be < L ({L}).")
    if N < 1:
        raise ValueError("geo_N_teeth must be >= 1.")

    pitch = (active / N) if cfg["geo_tooth_pitch_mm"] is None \
            else float(cfg["geo_tooth_pitch_mm"])

    inner_h = h - 2.0 * t
    inner_w = w - 2.0 * t
    if inner_h <= 0 or inner_w <= 0:
        raise ValueError("Wall thickness too large.")

    return {
        **cfg,
        "active_length_mm": active,
        "tooth_pitch_mm":   pitch,
        "tooth_depth_mm":   inner_h / 2.0,
        "inner_h_mm":       inner_h,
        "inner_w_mm":       inner_w,
    }
