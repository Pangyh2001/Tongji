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


class PointUpsampleBlock(nn.Module):
    """Learn child points as bounded local offsets from each parent point."""

    def __init__(
        self,
        *,
        factor: int,
        context_dim: int,
        hidden_dim: int = 192,
        max_offset_mm: float,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.max_offset_mm = max_offset_mm
        self.point_projection = nn.Linear(6, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.child_codes = nn.Parameter(torch.randn(factor, hidden_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Linear(hidden_dim, 3)
        self.normal_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        parents: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = (
            self.point_projection(parents).unsqueeze(2)
            + self.context_projection(context)[:, None, None, :]
            + self.child_codes[None, None, :, :]
        )
        features = self.refine(features)
        offsets = torch.tanh(self.offset_head(features)) * self.max_offset_mm

        parent_xyz = parents[..., :3].unsqueeze(2)
        parent_normals = parents[..., 3:6].unsqueeze(2)
        xyz = parent_xyz + offsets
        normal_delta = 0.25 * torch.tanh(self.normal_head(features))
        normals = torch.nn.functional.normalize(parent_normals + normal_delta, dim=-1, eps=1e-6)
        children = torch.cat([xyz, normals], dim=-1)
        return children.flatten(1, 2), offsets


class M0CoarseToFineNet(nn.Module):
    """Generate a structured crown point cloud in 8k -> 32k -> 64k stages."""

    def __init__(
        self,
        *,
        coarse_points: int = 8192,
        first_factor: int = 4,
        second_factor: int = 2,
        feature_dim: int = 512,
        latent_dim: int = 768,
        seed_dim: int = 64,
        tooth_count: int = 16,
        arch_count: int = 3,
        first_max_offset_mm: float = 0.35,
        second_max_offset_mm: float = 0.16,
    ) -> None:
        super().__init__()
        self.coarse_points = coarse_points
        self.first_factor = first_factor
        self.second_factor = second_factor
        self.output_points = coarse_points * first_factor * second_factor
        self.output_channels = 6

        self.prep_encoder = PointNetEncoder(6, feature_dim)
        self.antagonist_encoder = PointNetEncoder(6, feature_dim)
        self.tooth_embedding = nn.Embedding(tooth_count, 64)
        self.arch_embedding = nn.Embedding(arch_count, 16)

        fused_dim = feature_dim * 2 + 64 + 16
        self.context_decoder = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.coarse_seeds = nn.Embedding(coarse_points, seed_dim)
        self.coarse_seed_projection = nn.Linear(seed_dim, 256)
        self.coarse_context_projection = nn.Linear(latent_dim, 256)
        self.coarse_decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )
        self.upsample_32k = PointUpsampleBlock(
            factor=first_factor,
            context_dim=latent_dim,
            max_offset_mm=first_max_offset_mm,
        )
        self.upsample_64k = PointUpsampleBlock(
            factor=second_factor,
            context_dim=latent_dim,
            max_offset_mm=second_max_offset_mm,
        )

    def forward(
        self,
        prep: torch.Tensor,
        antagonist: torch.Tensor,
        tooth_index: torch.Tensor,
        prep_arch_index: torch.Tensor,
        *,
        return_stages: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        prep_feat = self.prep_encoder(prep)
        ant_feat = self.antagonist_encoder(antagonist)
        tooth_feat = self.tooth_embedding(tooth_index)
        arch_feat = self.arch_embedding(prep_arch_index)
        context = self.context_decoder(torch.cat([prep_feat, ant_feat, tooth_feat, arch_feat], dim=1))

        seed_features = self.coarse_seed_projection(self.coarse_seeds.weight)[None, :, :]
        coarse_features = seed_features + self.coarse_context_projection(context)[:, None, :]
        coarse_raw = self.coarse_decoder(coarse_features)
        coarse = torch.cat(
            [
                coarse_raw[..., :3],
                torch.nn.functional.normalize(coarse_raw[..., 3:6], dim=-1, eps=1e-6),
            ],
            dim=-1,
        )
        middle, offsets_32k = self.upsample_32k(coarse, context)
        fine, offsets_64k = self.upsample_64k(middle, context)
        if return_stages:
            return {
                "stages": [coarse, middle, fine],
                "offsets": [offsets_32k, offsets_64k],
            }
        return fine


class TangentPointUpsampleBlock(nn.Module):
    """Expand children in the local tangent plane with tightly bounded normal drift."""

    def __init__(
        self,
        *,
        factor: int,
        context_dim: int,
        hidden_dim: int = 192,
        max_tangent_offset_mm: float,
        max_normal_offset_mm: float,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.max_tangent_offset_mm = max_tangent_offset_mm
        self.max_normal_offset_mm = max_normal_offset_mm
        self.point_projection = nn.Linear(6, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.child_codes = nn.Parameter(torch.randn(factor, hidden_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.tangent_head = nn.Linear(hidden_dim, 2)
        self.normal_offset_head = nn.Linear(hidden_dim, 1)
        self.normal_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        parents: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = (
            self.point_projection(parents).unsqueeze(2)
            + self.context_projection(context)[:, None, None, :]
            + self.child_codes[None, None, :, :]
        )
        features = self.refine(features)

        parent_normals = torch.nn.functional.normalize(parents[..., 3:6], dim=-1, eps=1e-6)
        reference_z = torch.zeros_like(parent_normals)
        reference_z[..., 2] = 1.0
        reference_x = torch.zeros_like(parent_normals)
        reference_x[..., 0] = 1.0
        use_x = parent_normals[..., 2].abs() > 0.9
        reference = torch.where(use_x.unsqueeze(-1), reference_x, reference_z)
        tangent_1 = torch.nn.functional.normalize(
            torch.cross(parent_normals, reference, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        tangent_2 = torch.cross(parent_normals, tangent_1, dim=-1)

        tangent_uv = torch.tanh(self.tangent_head(features)) * self.max_tangent_offset_mm
        normal_offset = torch.tanh(self.normal_offset_head(features)) * self.max_normal_offset_mm
        offsets = (
            tangent_uv[..., :1] * tangent_1.unsqueeze(2)
            + tangent_uv[..., 1:] * tangent_2.unsqueeze(2)
            + normal_offset * parent_normals.unsqueeze(2)
        )

        xyz = parents[..., :3].unsqueeze(2) + offsets
        normal_delta = 0.15 * torch.tanh(self.normal_head(features))
        normals = torch.nn.functional.normalize(
            parent_normals.unsqueeze(2) + normal_delta,
            dim=-1,
            eps=1e-6,
        )
        children = torch.cat([xyz, normals], dim=-1)
        return children.flatten(1, 2), offsets


class M0TangentCoarseToFineNet(M0CoarseToFineNet):
    """Coarse-to-fine M0 with tangent-plane constrained local upsampling."""

    def __init__(
        self,
        *,
        coarse_points: int = 8192,
        first_factor: int = 4,
        second_factor: int = 2,
        feature_dim: int = 512,
        latent_dim: int = 768,
        seed_dim: int = 64,
        tooth_count: int = 16,
        arch_count: int = 3,
        first_max_tangent_offset_mm: float = 0.12,
        first_max_normal_offset_mm: float = 0.02,
        second_max_tangent_offset_mm: float = 0.06,
        second_max_normal_offset_mm: float = 0.01,
    ) -> None:
        super().__init__(
            coarse_points=coarse_points,
            first_factor=first_factor,
            second_factor=second_factor,
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            seed_dim=seed_dim,
            tooth_count=tooth_count,
            arch_count=arch_count,
        )
        self.upsample_32k = TangentPointUpsampleBlock(
            factor=first_factor,
            context_dim=latent_dim,
            max_tangent_offset_mm=first_max_tangent_offset_mm,
            max_normal_offset_mm=first_max_normal_offset_mm,
        )
        self.upsample_64k = TangentPointUpsampleBlock(
            factor=second_factor,
            context_dim=latent_dim,
            max_tangent_offset_mm=second_max_tangent_offset_mm,
            max_normal_offset_mm=second_max_normal_offset_mm,
        )


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
