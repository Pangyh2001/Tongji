from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.net(points)
        return features.max(dim=1).values


class M0CrownNet(nn.Module):
    """M0 baseline: prep ROI + antagonist ROI + tooth/arch embedding -> crown points.

    This intentionally does not consume margin_points. M1 can reuse the same
    interface and add a margin encoder.
    """

    def __init__(
        self,
        *,
        output_points: int = 16384,
        output_channels: int = 6,
        feature_dim: int = 512,
        latent_dim: int = 1024,
        tooth_count: int = 16,
        arch_count: int = 3,
    ) -> None:
        super().__init__()
        self.output_points = output_points
        self.output_channels = output_channels
        self.prep_encoder = PointNetEncoder(6, feature_dim)
        self.antagonist_encoder = PointNetEncoder(6, feature_dim)
        self.tooth_embedding = nn.Embedding(tooth_count, 64)
        self.arch_embedding = nn.Embedding(arch_count, 16)

        fused_dim = feature_dim * 2 + 64 + 16
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, output_points * output_channels),
        )

    def forward(
        self,
        prep: torch.Tensor,
        antagonist: torch.Tensor,
        tooth_index: torch.Tensor,
        prep_arch_index: torch.Tensor,
    ) -> torch.Tensor:
        prep_feat = self.prep_encoder(prep)
        ant_feat = self.antagonist_encoder(antagonist)
        tooth_feat = self.tooth_embedding(tooth_index)
        arch_feat = self.arch_embedding(prep_arch_index)
        fused = torch.cat([prep_feat, ant_feat, tooth_feat, arch_feat], dim=1)
        out = self.decoder(fused)
        return out.view(prep.shape[0], self.output_points, self.output_channels)

