"""Differentiable Poisson surface reconstruction.

This is a compact PyTorch adaptation of the MIT-licensed Shape As Points DPSR
implementation: https://github.com/autonomousvision/shape_as_points
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def point_rasterize(points: torch.Tensor, values: torch.Tensor, resolution: int) -> torch.Tensor:
    """Trilinearly splat point values onto a periodic cubic grid."""
    if points.shape != values.shape:
        raise ValueError(f"points and values must match, got {points.shape} and {values.shape}")
    batch, count, channels = values.shape
    scaled = points.clamp(0.0, 1.0 - 1e-6) * resolution
    base = torch.floor(scaled).long()
    fraction = scaled - base.to(scaled.dtype)
    raster = values.new_zeros(batch, channels, resolution**3)

    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = torch.tensor([dx, dy, dz], device=points.device)
                indices = (base + corner) % resolution
                weights_xyz = torch.where(
                    corner.view(1, 1, 3).bool(),
                    fraction,
                    1.0 - fraction,
                )
                weights = weights_xyz.prod(dim=-1)
                linear = (
                    indices[..., 0] * resolution * resolution
                    + indices[..., 1] * resolution
                    + indices[..., 2]
                )
                raster.scatter_add_(
                    2,
                    linear.unsqueeze(1).expand(batch, channels, count),
                    (values * weights.unsqueeze(-1)).transpose(1, 2),
                )
    return raster.view(batch, channels, resolution, resolution, resolution)


def sample_grid(grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample a scalar grid at points in [0, 1]."""
    query = points[..., [2, 1, 0]] * 2.0 - 1.0
    query = query.view(points.shape[0], points.shape[1], 1, 1, 3)
    sampled = F.grid_sample(
        grid.unsqueeze(1),
        query,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0, :, 0, 0]


class DPSR(nn.Module):
    """Map oriented points in [0, 1]^3 to a differentiable indicator grid."""

    def __init__(self, resolution: int = 64, sigma: float = 2.0) -> None:
        super().__init__()
        self.resolution = int(resolution)
        freq = torch.fft.fftfreq(self.resolution, d=1.0 / self.resolution)
        freq_last = torch.fft.rfftfreq(self.resolution, d=1.0 / self.resolution)
        omega = torch.stack(
            torch.meshgrid(freq, freq, freq_last, indexing="ij"),
            dim=-1,
        )
        omega = omega * (2.0 * math.pi)
        distance = torch.linalg.norm(omega, dim=-1)
        gaussian = torch.exp(
            -0.5 * ((sigma * 2.0 * distance / self.resolution) ** 2)
        )
        self.register_buffer("omega", omega)
        self.register_buffer("gaussian", gaussian)

    def forward(self, points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        normals = F.normalize(normals, dim=-1, eps=1e-6)
        raster = point_rasterize(points, normals, self.resolution)
        spectrum = torch.fft.rfftn(raster, dim=(2, 3, 4))
        spectrum = spectrum.permute(0, 2, 3, 4, 1)
        filtered = spectrum * self.gaussian[None, ..., None]

        divergence = (-1j * filtered * self.omega[None]).sum(dim=-1)
        laplacian = -(self.omega.square().sum(dim=-1))
        phi_spectrum = divergence / (laplacian[None] + 1e-6)
        dc_mask = torch.ones_like(laplacian)
        dc_mask[0, 0, 0] = 0.0
        phi_spectrum = phi_spectrum * dc_mask[None]
        phi = torch.fft.irfftn(
            phi_spectrum,
            s=(self.resolution, self.resolution, self.resolution),
            dim=(1, 2, 3),
        )

        surface_values = sample_grid(phi, points)
        phi = phi - surface_values.mean(dim=1).view(-1, 1, 1, 1)
        corner = phi[:, 0, 0, 0].abs().clamp_min(1e-6)
        return -0.5 * phi / corner.view(-1, 1, 1, 1)
