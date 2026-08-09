from __future__ import annotations

import torch
from torch import nn

from .dpsr import DPSR


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


class ProgressivePointSplitBlock(nn.Module):
    """Split each parent into learned children with optional local input skip attention."""

    def __init__(
        self,
        *,
        model_dim: int,
        factor: int = 4,
        max_offset_mm: float = 0.25,
        use_local_skip: bool = False,
        skip_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.factor = int(factor)
        self.max_offset_mm = float(max_offset_mm)
        self.use_local_skip = bool(use_local_skip)
        self.skip_neighbors = int(skip_neighbors)
        self.parent_position = nn.Linear(3, model_dim)
        self.parent_normal = nn.Linear(3, model_dim)
        self.child_codes = nn.Parameter(torch.randn(self.factor, model_dim) * 0.02)
        if self.use_local_skip:
            self.skip_norm = nn.LayerNorm(model_dim)
            self.skip_attention = nn.MultiheadAttention(
                model_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.skip_gate = nn.Parameter(torch.tensor(-1.0))
        self.refine = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.offset_head = nn.Linear(model_dim, 3)
        self.normal_head = nn.Linear(model_dim, 3)

    def _add_local_skip(
        self,
        parent_features: torch.Tensor,
        parent_xyz: torch.Tensor,
        memory_xyz: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_count = min(self.skip_neighbors, memory.shape[1])
        with torch.no_grad():
            neighbor_index = torch.cdist(parent_xyz, memory_xyz).topk(
                neighbor_count,
                dim=-1,
                largest=False,
            ).indices
        batch, parents, _ = parent_xyz.shape
        memory_expanded = memory[:, None].expand(-1, parents, -1, -1)
        neighbors = torch.gather(
            memory_expanded,
            2,
            neighbor_index[..., None].expand(-1, -1, -1, memory.shape[-1]),
        )
        query = self.skip_norm(parent_features).reshape(batch * parents, 1, -1)
        key_value = neighbors.reshape(batch * parents, neighbor_count, -1)
        skip, _ = self.skip_attention(query, key_value, key_value, need_weights=False)
        skip = skip.reshape(batch, parents, -1)
        return parent_features + torch.sigmoid(self.skip_gate) * skip

    def forward(
        self,
        parents: torch.Tensor,
        parent_features: torch.Tensor,
        *,
        memory_xyz: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        parent_xyz = parents[..., :3]
        parent_normals = torch.nn.functional.normalize(
            parents[..., 3:6], dim=-1, eps=1e-6
        )
        features = (
            parent_features
            + self.parent_position(parent_xyz)
            + self.parent_normal(parent_normals)
        )
        if self.use_local_skip:
            features = self._add_local_skip(features, parent_xyz, memory_xyz, memory)
        children = features.unsqueeze(2) + self.child_codes[None, None]
        children = children + self.refine(children)
        offsets = torch.tanh(self.offset_head(children)) * self.max_offset_mm
        xyz = parent_xyz.unsqueeze(2) + offsets
        normal_delta = 0.2 * torch.tanh(self.normal_head(children))
        normals = torch.nn.functional.normalize(
            parent_normals.unsqueeze(2) + normal_delta,
            dim=-1,
            eps=1e-6,
        )
        return torch.cat([xyz, normals], dim=-1).flatten(1, 2)


class M0DMCDPSRNet(nn.Module):
    """DMC-style Transformer + Folding decoder with differentiable PSR."""

    def __init__(
        self,
        *,
        model_dim: int = 256,
        context_tokens_per_input: int = 256,
        num_queries: int = 256,
        fold_step: int = 8,
        transformer_layers: int = 3,
        tooth_count: int = 16,
        arch_count: int = 3,
        dpsr_resolution: int = 128,
        dpsr_sigma: float = 2.0,
        roi_half_extent_mm: float = 12.0,
        use_margin: bool = False,
        margin_anchor_queries: int = 0,
        ring_groups: int = 0,
        margin_anchor_max_offset_mm: float = 1.5,
        detail_decoder: str = "folding",
        spd_parent_step: int = 4,
        spd_factor: int = 4,
        spd_max_offset_mm: float = 0.25,
        spd_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.context_tokens_per_input = context_tokens_per_input
        self.num_queries = num_queries
        self.fold_step = fold_step
        self.patch_points = fold_step * fold_step
        self.output_points = num_queries * self.patch_points
        self.output_channels = 6
        self.roi_half_extent_mm = roi_half_extent_mm
        self.use_margin = use_margin
        self.margin_anchor_queries = int(margin_anchor_queries)
        self.ring_groups = int(ring_groups)
        self.margin_anchor_max_offset_mm = float(margin_anchor_max_offset_mm)
        self.detail_decoder = detail_decoder
        if self.margin_anchor_queries > self.num_queries:
            raise ValueError("margin_anchor_queries cannot exceed num_queries")
        if self.margin_anchor_queries and not self.use_margin:
            raise ValueError("margin-anchored queries require use_margin=True")
        if detail_decoder not in {"folding", "spd", "spd_skip"}:
            raise ValueError(f"unsupported detail decoder: {detail_decoder}")
        if detail_decoder != "folding" and (
            spd_parent_step * spd_parent_step * spd_factor != self.patch_points
        ):
            raise ValueError(
                "spd_parent_step^2 * spd_factor must equal fold_step^2"
            )

        self.point_projection = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, model_dim),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, model_dim),
        )
        self.input_type_embedding = nn.Embedding(3 if use_margin else 2, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim),
        )

        self.query_embedding = nn.Embedding(num_queries, model_dim)
        self.ring_embedding = nn.Embedding(ring_groups, model_dim) if ring_groups > 0 else None
        self.tooth_embedding = nn.Embedding(tooth_count, 64)
        self.arch_embedding = nn.Embedding(arch_count, 16)
        self.condition_projection = nn.Linear(80, model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.crown_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.coarse_head = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

        grid_axis = torch.linspace(-1.0, 1.0, steps=fold_step)
        grid_u, grid_v = torch.meshgrid(grid_axis, grid_axis, indexing="ij")
        self.register_buffer("folding_grid", torch.stack([grid_u.flatten(), grid_v.flatten()], dim=0))
        self.folding_1 = nn.Sequential(
            nn.Conv1d(model_dim + 2, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1),
        )
        self.folding_2 = nn.Sequential(
            nn.Conv1d(model_dim + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1),
        )
        self.normal_head = nn.Sequential(
            nn.Conv1d(model_dim + 3, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1),
        )
        if detail_decoder != "folding":
            parent_axis = torch.linspace(-1.0, 1.0, steps=spd_parent_step)
            parent_u, parent_v = torch.meshgrid(parent_axis, parent_axis, indexing="ij")
            self.register_buffer(
                "spd_parent_grid",
                torch.stack([parent_u.flatten(), parent_v.flatten()], dim=0),
            )
            self.detail_upsampler = ProgressivePointSplitBlock(
                model_dim=model_dim,
                factor=spd_factor,
                max_offset_mm=spd_max_offset_mm,
                use_local_skip=detail_decoder == "spd_skip",
                skip_neighbors=spd_neighbors,
            )
        self.dpsr = DPSR(resolution=dpsr_resolution, sigma=dpsr_sigma)

    def _sample_context(self, points: torch.Tensor) -> torch.Tensor:
        count = min(self.context_tokens_per_input, points.shape[1])
        indices = torch.linspace(
            0,
            points.shape[1] - 1,
            steps=count,
            device=points.device,
        ).long()
        return points[:, indices]

    def points_to_grid(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[..., :3]
        normals = points[..., 3:6]
        normalized_xyz = (
            xyz + self.roi_half_extent_mm
        ) / (2.0 * self.roi_half_extent_mm)
        normalized_xyz = normalized_xyz.clamp(1e-4, 1.0 - 1e-4)
        return self.dpsr(normalized_xyz, normals)

    def forward(
        self,
        prep: torch.Tensor,
        antagonist: torch.Tensor,
        tooth_index: torch.Tensor,
        prep_arch_index: torch.Tensor,
        *,
        margin: torch.Tensor | None = None,
        return_grid: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        prep_tokens = self._sample_context(prep)
        antagonist_tokens = self._sample_context(antagonist)
        context_parts = [prep_tokens, antagonist_tokens]
        type_parts = [
            torch.zeros(prep_tokens.shape[1], dtype=torch.long, device=prep.device),
            torch.ones(antagonist_tokens.shape[1], dtype=torch.long, device=prep.device),
        ]
        if self.use_margin:
            if margin is None:
                raise ValueError("margin input is required for this DMC-DPSR model")
            margin_tokens = self._sample_context(margin)
            margin_features = torch.nn.functional.pad(margin_tokens, (0, 3))
            context_parts.append(margin_features)
            type_parts.append(
                torch.full(
                    (margin_tokens.shape[1],),
                    2,
                    dtype=torch.long,
                    device=prep.device,
                )
            )
        context_points = torch.cat(context_parts, dim=1)
        token_types = torch.cat(type_parts)
        context = (
            self.point_projection(context_points)
            + self.position_projection(context_points[..., :3])
            + self.input_type_embedding(token_types)[None]
        )
        memory = self.context_encoder(context)

        condition = self.condition_projection(
            torch.cat(
                [
                    self.tooth_embedding(tooth_index),
                    self.arch_embedding(prep_arch_index),
                ],
                dim=-1,
            )
        )
        queries = self.query_embedding.weight[None].expand(prep.shape[0], -1, -1)
        if self.ring_embedding is not None:
            ring_index = (
                torch.arange(self.num_queries, device=prep.device) * self.ring_groups
            ) // self.num_queries
            queries = queries + self.ring_embedding(ring_index)[None]
        query_features = self.crown_decoder(queries + condition[:, None, :], memory)
        coarse = self.coarse_head(query_features)
        if self.margin_anchor_queries > 0:
            anchor_index = torch.linspace(
                0,
                margin.shape[1] - 1,
                steps=self.margin_anchor_queries,
                device=margin.device,
            ).long()
            anchors = margin[:, anchor_index, :3]
            anchor_offsets = (
                torch.tanh(coarse[:, : self.margin_anchor_queries])
                * self.margin_anchor_max_offset_mm
            )
            coarse = torch.cat(
                [
                    anchors + anchor_offsets,
                    coarse[:, self.margin_anchor_queries :],
                ],
                dim=1,
            )

        batch, queries_count, channels = query_features.shape
        patch_points = (
            self.patch_points
            if self.detail_decoder == "folding"
            else self.spd_parent_grid.shape[1]
        )
        features = query_features.reshape(batch * queries_count, channels, 1).expand(
            -1, -1, patch_points
        )
        detail_grid = (
            self.folding_grid
            if self.detail_decoder == "folding"
            else self.spd_parent_grid
        )
        grid = detail_grid[None].expand(batch * queries_count, -1, -1)
        folded_1 = self.folding_1(torch.cat([features, grid], dim=1))
        folded_2 = self.folding_2(torch.cat([features, folded_1], dim=1))
        relative = folded_2.transpose(1, 2).reshape(
            batch, queries_count, patch_points, 3
        )
        xyz_raw = coarse.unsqueeze(2) + relative
        xyz = torch.tanh(xyz_raw / self.roi_half_extent_mm) * self.roi_half_extent_mm

        normal_input = torch.cat([features, folded_2], dim=1)
        normals = self.normal_head(normal_input).transpose(1, 2)
        normals = normals.reshape(batch, queries_count, patch_points, 3)
        normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
        parents = torch.cat([xyz, normals], dim=-1).flatten(1, 2)
        if self.detail_decoder == "folding":
            points = parents
        else:
            parent_features = query_features.unsqueeze(2).expand(
                -1, -1, patch_points, -1
            ).flatten(1, 2)
            points = self.detail_upsampler(
                parents,
                parent_features,
                memory_xyz=context_points[..., :3],
                memory=memory,
            )
        if return_grid:
            return {"points": points, "psr_grid": self.points_to_grid(points)}
        return points


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
