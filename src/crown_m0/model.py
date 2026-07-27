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


class M0TemplateDeformNet(nn.Module):
    """M0 template-deformation baseline.

    The network predicts a displacement vector for each vertex in a fixed crown
    template. The final STL reuses the template faces, so the output has stable
    topology instead of reconstructing triangles from an unordered point cloud.
    """

    def __init__(
        self,
        *,
        template_vertices: int,
        feature_dim: int = 512,
        latent_dim: int = 1024,
        tooth_count: int = 16,
        arch_count: int = 3,
        max_displacement_mm: float = 4.0,
    ) -> None:
        super().__init__()
        self.template_vertices = template_vertices
        self.max_displacement_mm = max_displacement_mm
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
            nn.Linear(latent_dim, template_vertices * 3),
        )

    def forward(
        self,
        prep: torch.Tensor,
        antagonist: torch.Tensor,
        tooth_index: torch.Tensor,
        prep_arch_index: torch.Tensor,
        template_vertices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prep_feat = self.prep_encoder(prep)
        ant_feat = self.antagonist_encoder(antagonist)
        tooth_feat = self.tooth_embedding(tooth_index)
        arch_feat = self.arch_embedding(prep_arch_index)
        fused = torch.cat([prep_feat, ant_feat, tooth_feat, arch_feat], dim=1)
        delta = self.decoder(fused).view(prep.shape[0], self.template_vertices, 3)
        delta = torch.tanh(delta) * self.max_displacement_mm
        if template_vertices.dim() == 2:
            base_vertices = template_vertices.unsqueeze(0).expand_as(delta)
        elif template_vertices.dim() == 3:
            base_vertices = template_vertices
        else:
            raise ValueError(f"template_vertices must be (N, 3) or (B, N, 3), got {tuple(template_vertices.shape)}")
        vertices = base_vertices + delta
        return vertices, delta
