"""Physics-Informed Wildfire Spread Loss Engine (PINN).

Supervises neural network predictions using fundamental physical wildfire propagation principles:
1. Rothermel Surface Fire Spread Equation:
   Rate of spread (ROS) R = R_0 * (1 + phi_w + phi_s)
   where phi_w is the wind coefficient and phi_s is the slope acceleration factor.
2. 2D Spatiotemporal Advection-Diffusion Partial Differential Equation (PDE):
   dp/dt + v_wind . grad(p) - D * laplacian(p) - S(fuel, dryness) = 0
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


class PhysicsInformedSpreadLoss(nn.Module):
    """Computes physical consistency loss penalizing unphysical flame propagation.
    
    Penalizes:
    - Downwind advection violations (fire moving opposite to hurricane-force winds).
    - Topographical slope violations (fire failing to accelerate up steep inclines).
    - Fuel depletion conservation (fire persisting in zero-fuel unburnable zones).
    """

    def __init__(
        self,
        diffusion_coef: float = 0.05,
        wind_weight: float = 0.4,
        slope_weight: float = 0.35,
        fuel_weight: float = 0.25
    ):
        super().__init__()
        self.diffusion_coef = diffusion_coef
        self.wind_weight = wind_weight
        self.slope_weight = slope_weight
        self.fuel_weight = fuel_weight

        # Spatial Sobel filters for spatial gradients dx and dy
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        laplacian = torch.tensor([[0.0,  1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0,  1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    def compute_spatial_derivatives(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes dp/dx, dp/dy, and laplacian(p) using 2D convolutions."""
        b, t, c, h, w = tensor.shape
        # Flatten B*T to 2D batch
        flat = tensor.view(b * t * c, 1, h, w)

        grad_x = F.conv2d(flat, self.sobel_x, padding=1)
        grad_y = F.conv2d(flat, self.sobel_y, padding=1)
        lap = F.conv2d(flat, self.laplacian, padding=1)

        return (
            grad_x.view(b, t, c, h, w),
            grad_y.view(b, t, c, h, w),
            lap.view(b, t, c, h, w)
        )

    def forward(
        self,
        pred_presence: torch.Tensor,     # (B, T_out=3, 1, H, W)
        input_6ch: torch.Tensor          # (B, T_in=5, 6, H, W)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculates physical PDE residual loss across predicted frames.
        
        Channels of input_6ch:
            C1: fire, C2: ndvi, C3: terrain, C4: wind_speed, C5: wind_dir, C6: dryness
        """
        device = pred_presence.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.laplacian = self.laplacian.to(device)

        # Extract latest environmental conditions from t0
        t0_env = input_6ch[:, -1] # (B, 6, H, W)
        ndvi = t0_env[:, 1:2].unsqueeze(1)      # (B, 1, 1, H, W)
        terrain = t0_env[:, 2:3].unsqueeze(1)   # (B, 1, 1, H, W)
        wind_spd = t0_env[:, 3:4].unsqueeze(1)  # (B, 1, 1, H, W)
        wind_dir = t0_env[:, 4:5].unsqueeze(1)  # (B, 1, 1, H, W)
        dryness = t0_env[:, 5:6].unsqueeze(1)   # (B, 1, 1, H, W)

        # Convert circular wind direction to vector field (u, v)
        wind_rad = wind_dir * (2.0 * 3.1415926535)
        u_wind = wind_spd * torch.cos(wind_rad)
        v_wind = wind_spd * torch.sin(wind_rad)

        # 1. Temporal derivative dp/dt across future horizons
        # dp/dt ~ pred[:, 1:] - pred[:, :-1]
        dp_dt = pred_presence[:, 1:] - pred_presence[:, :-1] # (B, 2, 1, H, W)
        p_mid = (pred_presence[:, 1:] + pred_presence[:, :-1]) * 0.5

        # 2. Spatial derivatives of flame presence
        grad_x, grad_y, lap = self.compute_spatial_derivatives(p_mid)

        # 3. Advection term: v . grad(p) = u * dp/dx + v * dp/dy
        advection = u_wind * grad_x + v_wind * grad_y

        # 4. Diffusion term: D * laplacian(p)
        diffusion = self.diffusion_coef * lap

        # 5. Rothermel Topographical Slope Acceleration:
        # Fire accelerates up slope: grad(terrain)
        t_grad_x, t_grad_y, _ = self.compute_spatial_derivatives(terrain.repeat(1, 2, 1, 1, 1))
        slope_alignment = -(t_grad_x * grad_x + t_grad_y * grad_y) # Positive when fire spreads uphill

        # 6. Physical PDE Residual:
        # dp/dt should align with: advection (wind push) + diffusion + slope boost + fuel availability
        expected_spread = advection + diffusion + self.slope_weight * slope_alignment
        
        # Fuel constraint: Fire cannot spread where NDVI/fuel is zero
        fuel_limiter = torch.clamp(ndvi, 0.05, 1.0)
        pde_residual = (dp_dt - expected_spread * fuel_limiter) ** 2

        # Physics loss is the mean squared PDE residual
        loss_pinn = pde_residual.mean()

        return loss_pinn, {
            "pinn_loss": round(float(loss_pinn.item()), 5),
            "mean_advection": round(float(advection.abs().mean().item()), 5),
            "mean_diffusion": round(float(diffusion.abs().mean().item()), 5)
        }

