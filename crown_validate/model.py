from __future__ import annotations

import torch
from torch import nn


def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_support, channels = values.shape
    _, num_queries, num_neighbors = indices.shape
    batch_offsets = torch.arange(batch_size, device=values.device).view(batch_size, 1, 1) * num_support
    flat_indices = (indices + batch_offsets).reshape(-1)
    flat_values = values.reshape(batch_size * num_support, channels)
    gathered = flat_values[flat_indices]
    return gathered.reshape(batch_size, num_queries, num_neighbors, channels)


def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    num_points = points.shape[1]
    if num_points <= 1:
        return torch.zeros((points.shape[0], num_points, 1), dtype=torch.long, device=points.device)
    actual_k = min(k + 1, num_points)
    distances = torch.cdist(points, points)
    indices = torch.topk(distances, k=actual_k, largest=False).indices
    if actual_k > 1:
        return indices[:, :, 1:]
    return indices


class SharedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class EdgeConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = SharedMLP(input_dim * 2, output_dim, output_dim, dropout=dropout)

    def forward(self, features: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        neighbors = _gather_neighbors(features, neighbor_indices)
        center = features.unsqueeze(2).expand_as(neighbors)
        edge_features = torch.cat([center, neighbors - center], dim=-1)
        return self.edge_mlp(edge_features).max(dim=2).values


class LocalGeometryEncoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, dropout: float, encoder_knn: int) -> None:
        super().__init__()
        mid_dim = max(hidden_dim // 2, 32)
        self.encoder_knn = encoder_knn
        self.stem = SharedMLP(3, mid_dim, mid_dim, dropout=dropout)
        self.edge_block_1 = EdgeConvBlock(mid_dim, hidden_dim, dropout=dropout)
        self.edge_block_2 = EdgeConvBlock(hidden_dim, hidden_dim, dropout=dropout)
        self.point_projection = nn.Sequential(
            nn.Linear(mid_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(mid_dim + hidden_dim * 2, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        knn = _knn_indices(points, k=self.encoder_knn)
        stem = self.stem(points)
        edge_1 = self.edge_block_1(stem, knn)
        edge_2 = self.edge_block_2(edge_1, knn)
        multi_scale = torch.cat([stem, edge_1, edge_2], dim=-1)
        point_features = self.point_projection(multi_scale)
        global_feature = self.global_projection(multi_scale.max(dim=1).values)
        return point_features, global_feature


class TemplateQueryDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
        decoder_knn: int,
        max_offset: float,
    ) -> None:
        super().__init__()
        self.decoder_knn = decoder_knn
        self.max_offset = max_offset
        fusion_input_dim = 3 + (feature_dim * 2) + 6 + (latent_dim * 4)
        self.decoder = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def _query_support(
        self,
        query_vertices: torch.Tensor,
        support_points: torch.Tensor,
        support_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_support = support_points.shape[1]
        actual_k = min(self.decoder_knn, num_support)
        distances = torch.cdist(query_vertices, support_points)
        knn_distances, knn_indices = torch.topk(distances, k=actual_k, largest=False)
        neighbor_features = _gather_neighbors(support_features, knn_indices)
        neighbor_points = _gather_neighbors(support_points, knn_indices)
        query_expanded = query_vertices.unsqueeze(2).expand_as(neighbor_points)
        relative_offsets = neighbor_points - query_expanded
        weights = torch.softmax(-knn_distances * 10.0, dim=-1).unsqueeze(-1)
        local_features = torch.sum(weights * neighbor_features, dim=2)
        local_offsets = torch.sum(weights * relative_offsets, dim=2)
        return local_features, local_offsets

    def forward(
        self,
        template_vertices: torch.Tensor,
        prep_points: torch.Tensor,
        prep_features: torch.Tensor,
        prep_global: torch.Tensor,
        opposing_points: torch.Tensor,
        opposing_features: torch.Tensor,
        opposing_global: torch.Tensor,
    ) -> torch.Tensor:
        prep_local_features, prep_local_offsets = self._query_support(template_vertices, prep_points, prep_features)
        opposing_local_features, opposing_local_offsets = self._query_support(template_vertices, opposing_points, opposing_features)

        prep_global_expanded = prep_global.unsqueeze(1).expand(-1, template_vertices.shape[1], -1)
        opposing_global_expanded = opposing_global.unsqueeze(1).expand(-1, template_vertices.shape[1], -1)
        decoder_input = torch.cat(
            [
                template_vertices,
                prep_local_features,
                opposing_local_features,
                prep_local_offsets,
                opposing_local_offsets,
                prep_global_expanded,
                opposing_global_expanded,
                torch.abs(prep_global_expanded - opposing_global_expanded),
                prep_global_expanded * opposing_global_expanded,
            ],
            dim=-1,
        )
        offsets = torch.tanh(self.decoder(decoder_input)) * self.max_offset
        return template_vertices + offsets


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    batch_size, num_vertices, _ = vertices.shape
    device = vertices.device
    vertex_normals = torch.zeros((batch_size, num_vertices, 3), device=device, dtype=vertices.dtype)

    a = vertices[:, faces[:, 0], :]
    b = vertices[:, faces[:, 1], :]
    c = vertices[:, faces[:, 2], :]
    face_normals = torch.cross(b - a, c - a, dim=-1)

    for vertex_slot in range(3):
        indices = faces[:, vertex_slot]
        expanded = indices.view(1, -1, 1).expand(batch_size, -1, 3)
        vertex_normals.scatter_add_(1, expanded, face_normals)

    return torch.nn.functional.normalize(vertex_normals, dim=-1, eps=1e-6)


class CrownDeformationNet(nn.Module):
    def __init__(
        self,
        template_vertices: torch.Tensor,
        template_faces: torch.Tensor,
        boundary_indices: torch.Tensor,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        max_offset: float,
        encoder_knn: int = 16,
        decoder_knn: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = LocalGeometryEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            encoder_knn=encoder_knn,
        )
        self.decoder = TemplateQueryDecoder(
            feature_dim=hidden_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            decoder_knn=decoder_knn,
            max_offset=max_offset,
        )
        self.register_buffer("template_vertices", template_vertices.float())
        self.register_buffer("template_faces", template_faces.long())
        self.register_buffer("boundary_indices", boundary_indices.long())

    def forward(self, prep_points: torch.Tensor, opposing_points: torch.Tensor) -> dict[str, torch.Tensor]:
        prep_features, prep_global = self.encoder(prep_points)
        opposing_features, opposing_global = self.encoder(opposing_points)
        template = self.template_vertices.unsqueeze(0).expand(prep_points.shape[0], -1, -1)
        pred_vertices = self.decoder(
            template_vertices=template,
            prep_points=prep_points,
            prep_features=prep_features,
            prep_global=prep_global,
            opposing_points=opposing_points,
            opposing_features=opposing_features,
            opposing_global=opposing_global,
        )
        pred_normals = _vertex_normals(pred_vertices, self.template_faces)
        pred_margin = pred_vertices[:, self.boundary_indices, :]
        return {
            "pred_points": pred_vertices,
            "pred_normals": pred_normals,
            "pred_margin": pred_margin,
        }
