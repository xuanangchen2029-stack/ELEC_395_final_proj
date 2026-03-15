from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, width=64, depth=4, length_scale_mm=150.0, pressure_scale_kpa=100.0):
        super().__init__()
        self.length_scale_mm = float(length_scale_mm)
        self.pressure_scale_kpa = float(pressure_scale_kpa)

        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s_mm, p_kpa):
        s_scaled = 2.0 * (s_mm / self.length_scale_mm) - 1.0
        p_scaled = 2.0 * (p_kpa / self.pressure_scale_kpa) - 1.0
        return self.net(torch.cat([s_scaled, p_scaled], dim=1))


class SoftActuatorPINN(nn.Module):
    def __init__(
        self,
        width=64,
        depth=4,
        length_scale_mm=150.0,
        pressure_scale_kpa=100.0,
        EI_scale_Nmm2=1000.0,
        kp_scale_N_per_kPa=1e-3,
    ):
        super().__init__()
        self.length_scale_mm = float(length_scale_mm)
        self.pressure_scale_kpa = float(pressure_scale_kpa)
        self.EI_scale_Nmm2 = float(EI_scale_Nmm2)
        self.kp_scale_N_per_kPa = float(kp_scale_N_per_kPa)

        self.field_net = MLP(
            in_dim=2,
            out_dim=3,
            width=width,
            depth=depth,
            length_scale_mm=length_scale_mm,
            pressure_scale_kpa=pressure_scale_kpa,
        )
        self.log_EI = nn.Parameter(torch.tensor(0.0))
        self.log_kp = nn.Parameter(torch.tensor(0.0))

    @property
    def EI_eff(self):
        return self.EI_scale_Nmm2 * torch.exp(self.log_EI)

    @property
    def k_p(self):
        return self.kp_scale_N_per_kPa * torch.exp(self.log_kp)

    def forward(self, s_mm, p_kpa):
        out = self.field_net(s_mm, p_kpa)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]



def gradients(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]



def physics_residuals(model, s_mm, p_kpa):
    s_mm.requires_grad_(True)
    p_kpa.requires_grad_(True)
    x_hat_mm, y_hat_mm, theta_hat_rad = model(s_mm, p_kpa)

    x_s = gradients(x_hat_mm, s_mm)
    y_s = gradients(y_hat_mm, s_mm)
    theta_s = gradients(theta_hat_rad, s_mm)
    theta_ss = gradients(theta_s, s_mm)

    r_x = x_s - torch.cos(theta_hat_rad)
    r_y = y_s - torch.sin(theta_hat_rad)
    r_theta = model.EI_eff * theta_ss + model.k_p * p_kpa
    return r_x, r_y, r_theta



def blocked_force_hat_linear(model, p_kpa):
    return (8.0 / 3.0) * model.k_p * p_kpa
