from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_points(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[1] <= count:
        return points
    idx = torch.randperm(points.shape[1], device=points.device)[:count]
    return points[:, idx]


def chamfer_distance(pred_xyz: torch.Tensor, target_xyz: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred_xyz, target_xyz, p=2)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def normal_cosine_loss(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    pred_normals: torch.Tensor,
    target_normals: torch.Tensor,
) -> torch.Tensor:
    # Match normals using nearest geometry points, not nearest normal vectors.
    pred_n = F.normalize(pred_normals, dim=-1)
    target_n = F.normalize(target_normals, dim=-1)
    dist = torch.cdist(pred_xyz, target_xyz, p=2)
    idx = dist.argmin(dim=2)
    gathered = torch.gather(target_n, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    return (1.0 - (pred_n * gathered).sum(dim=-1)).mean()


def m0_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    chamfer_points: int = 2048,
    normal_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_sample = sample_points(pred, chamfer_points)
    target_sample = sample_points(target, chamfer_points)
    cd = chamfer_distance(pred_sample[..., :3], target_sample[..., :3])
    loss = cd
    normal_loss = pred.new_tensor(0.0)
    if normal_weight > 0 and pred.shape[-1] >= 6 and target.shape[-1] >= 6:
        normal_loss = normal_cosine_loss(
            pred_sample[..., :3],
            target_sample[..., :3],
            pred_sample[..., 3:6],
            target_sample[..., 3:6],
        )
        loss = loss + normal_weight * normal_loss
    return loss, {"loss": float(loss.detach().cpu()), "chamfer": float(cd.detach().cpu()), "normal": float(normal_loss.detach().cpu())}


def dmc_dpsr_loss(
    pred_points: torch.Tensor,
    pred_grid: torch.Tensor,
    target_points: torch.Tensor,
    target_grid: torch.Tensor,
    *,
    chamfer_points: int = 4096,
    normal_weight: float = 0.05,
    grid_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Jointly supervise DMC points and the reconstructed Poisson indicator."""
    point_loss, point_metrics = m0_loss(
        pred_points,
        target_points,
        chamfer_points=chamfer_points,
        normal_weight=normal_weight,
    )
    pred_indicator = torch.tanh(pred_grid)
    target_indicator = torch.tanh(target_grid)
    grid_mse = F.mse_loss(pred_indicator, target_indicator)
    grid_l1 = F.l1_loss(pred_indicator, target_indicator)
    loss = point_loss + grid_weight * grid_mse
    return loss, {
        "loss": float(loss.detach().cpu()),
        "chamfer": point_metrics["chamfer"],
        "normal": point_metrics["normal"],
        "grid_mse": float(grid_mse.detach().cpu()),
        "grid_l1": float(grid_l1.detach().cpu()),
    }


def sibling_repulsion_loss(offsets: torch.Tensor, min_distance: float) -> torch.Tensor:
    """Keep children generated from one parent from collapsing together."""
    factor = offsets.shape[2]
    if factor < 2:
        return offsets.new_tensor(0.0)
    distances = torch.cdist(offsets, offsets)
    eye = torch.eye(factor, device=offsets.device, dtype=torch.bool)
    distances = distances.masked_fill(eye[None, None, :, :], float("inf"))
    nearest = distances.min(dim=-1).values
    return torch.relu(min_distance - nearest).square().mean()


def point_uniformity_loss(points: torch.Tensor, count: int = 1024, neighbors: int = 4) -> torch.Tensor:
    """Penalize large variation in local nearest-neighbor spacing."""
    xyz = sample_points(points, count)[..., :3]
    distances = torch.cdist(xyz, xyz)
    eye = torch.eye(xyz.shape[1], device=xyz.device, dtype=torch.bool)
    distances = distances.masked_fill(eye[None, :, :], float("inf"))
    local_spacing = distances.topk(k=min(neighbors, xyz.shape[1] - 1), largest=False).values.mean(dim=-1)
    mean = local_spacing.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return ((local_spacing - mean) / mean).square().mean()


def coarse_to_fine_loss(
    stages: list[torch.Tensor],
    offsets: list[torch.Tensor],
    target: torch.Tensor,
    *,
    chamfer_points: int = 2048,
    normal_weight: float = 0.05,
    stage_weights: tuple[float, float, float] = (0.2, 0.3, 0.5),
    repulsion_weight: float = 0.10,
    uniformity_weight: float = 0.01,
    offset_weight: float = 0.001,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(stages) != 3 or len(offsets) != 2:
        raise ValueError("coarse-to-fine loss expects three stages and two offset tensors")

    total = target.new_tensor(0.0)
    stage_metrics = []
    for stage, weight in zip(stages, stage_weights):
        stage_loss, metrics = m0_loss(
            stage,
            target,
            chamfer_points=chamfer_points,
            normal_weight=normal_weight,
        )
        total = total + weight * stage_loss
        stage_metrics.append(metrics)

    repulsion_32k = sibling_repulsion_loss(offsets[0], min_distance=0.08)
    repulsion_64k = sibling_repulsion_loss(offsets[1], min_distance=0.04)
    repulsion = repulsion_32k + repulsion_64k
    uniformity = point_uniformity_loss(stages[-1])
    offset_regularization = sum(item.square().mean() for item in offsets)
    total = (
        total
        + repulsion_weight * repulsion
        + uniformity_weight * uniformity
        + offset_weight * offset_regularization
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "coarse_chamfer": stage_metrics[0]["chamfer"],
        "middle_chamfer": stage_metrics[1]["chamfer"],
        "fine_chamfer": stage_metrics[2]["chamfer"],
        "fine_normal": stage_metrics[2]["normal"],
        "repulsion": float(repulsion.detach().cpu()),
        "uniformity": float(uniformity.detach().cpu()),
        "offset": float(offset_regularization.detach().cpu()),
    }


def tangent_sibling_repulsion_loss(
    offsets: torch.Tensor,
    parent_normals: torch.Tensor,
    min_distance: float,
) -> torch.Tensor:
    factor = offsets.shape[2]
    if factor < 2:
        return offsets.new_tensor(0.0)
    normals = F.normalize(parent_normals, dim=-1).unsqueeze(2)
    tangent_offsets = offsets - (offsets * normals).sum(dim=-1, keepdim=True) * normals
    distances = torch.cdist(tangent_offsets, tangent_offsets)
    eye = torch.eye(factor, device=offsets.device, dtype=torch.bool)
    distances = distances.masked_fill(eye[None, None, :, :], float("inf"))
    nearest = distances.min(dim=-1).values
    return torch.relu(min_distance - nearest).square().mean()


def local_plane_consistency_loss(points: torch.Tensor, count: int = 1024, neighbors: int = 8) -> torch.Tensor:
    sampled = sample_points(points, count)
    xyz = sampled[..., :3]
    normals = F.normalize(sampled[..., 3:6], dim=-1)
    distances = torch.cdist(xyz, xyz)
    eye = torch.eye(xyz.shape[1], device=xyz.device, dtype=torch.bool)
    distances = distances.masked_fill(eye[None, :, :], float("inf"))
    neighbor_idx = distances.topk(k=min(neighbors, xyz.shape[1] - 1), largest=False).indices
    batch_idx = torch.arange(xyz.shape[0], device=xyz.device)[:, None, None]
    neighbor_xyz = xyz[batch_idx, neighbor_idx]
    signed_height = ((neighbor_xyz - xyz.unsqueeze(2)) * normals.unsqueeze(2)).sum(dim=-1)
    return signed_height.abs().mean()


def stage_surface_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_sample = sample_points(pred, count)
    target_sample = sample_points(target, count)
    pred_xyz = pred_sample[..., :3]
    target_xyz = target_sample[..., :3]
    distances = torch.cdist(pred_xyz, target_xyz)
    pred_min, pred_to_target = distances.min(dim=2)
    target_min = distances.min(dim=1).values
    chamfer = pred_min.mean() + target_min.mean()

    pred_normals = F.normalize(pred_sample[..., 3:6], dim=-1)
    target_normals = F.normalize(target_sample[..., 3:6], dim=-1)
    gathered_target_xyz = torch.gather(
        target_xyz,
        1,
        pred_to_target.unsqueeze(-1).expand(-1, -1, 3),
    )
    gathered_target_normals = torch.gather(
        target_normals,
        1,
        pred_to_target.unsqueeze(-1).expand(-1, -1, 3),
    )
    normal = (1.0 - (pred_normals * gathered_target_normals).sum(dim=-1)).mean()
    point_to_plane = (
        (pred_xyz - gathered_target_xyz) * gathered_target_normals
    ).sum(dim=-1).abs().mean()
    return chamfer, normal, point_to_plane


def tangent_coarse_to_fine_loss(
    stages: list[torch.Tensor],
    offsets: list[torch.Tensor],
    target: torch.Tensor,
    *,
    chamfer_points: int = 8192,
    normal_weight: float = 0.20,
    point_to_plane_weight: float = 0.50,
    local_plane_weight: float = 0.20,
    repulsion_weight: float = 0.10,
    uniformity_weight: float = 0.01,
    normal_drift_weight: float = 0.50,
    stage_weights: tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(stages) != 3 or len(offsets) != 2:
        raise ValueError("tangent coarse-to-fine loss expects three stages and two offset tensors")

    total = target.new_tensor(0.0)
    stage_values = []
    for stage, weight in zip(stages, stage_weights):
        chamfer, normal, point_to_plane = stage_surface_losses(stage, target, count=chamfer_points)
        stage_loss = chamfer + normal_weight * normal + point_to_plane_weight * point_to_plane
        total = total + weight * stage_loss
        stage_values.append((chamfer, normal, point_to_plane))

    repulsion = tangent_sibling_repulsion_loss(
        offsets[0],
        stages[0][..., 3:6],
        min_distance=0.05,
    ) + tangent_sibling_repulsion_loss(
        offsets[1],
        stages[1][..., 3:6],
        min_distance=0.025,
    )
    uniformity = point_uniformity_loss(stages[-1])
    local_plane = local_plane_consistency_loss(stages[-1])
    normal_drift = sum(
        ((item * F.normalize(parent[..., 3:6], dim=-1).unsqueeze(2)).sum(dim=-1)).abs().mean()
        for item, parent in zip(offsets, stages[:-1])
    )
    total = (
        total
        + repulsion_weight * repulsion
        + uniformity_weight * uniformity
        + local_plane_weight * local_plane
        + normal_drift_weight * normal_drift
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "coarse_chamfer": float(stage_values[0][0].detach().cpu()),
        "middle_chamfer": float(stage_values[1][0].detach().cpu()),
        "fine_chamfer": float(stage_values[2][0].detach().cpu()),
        "fine_normal": float(stage_values[2][1].detach().cpu()),
        "fine_point_to_plane": float(stage_values[2][2].detach().cpu()),
        "repulsion": float(repulsion.detach().cpu()),
        "uniformity": float(uniformity.detach().cpu()),
        "local_plane": float(local_plane.detach().cpu()),
        "normal_drift": float(normal_drift.detach().cpu()),
    }


def edge_length_loss(vertices: torch.Tensor, template_vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return vertices.new_tensor(0.0)
    pred_len = torch.linalg.norm(vertices[:, edges[:, 0]] - vertices[:, edges[:, 1]], dim=-1)
    template_len = torch.linalg.norm(template_vertices[edges[:, 0]] - template_vertices[edges[:, 1]], dim=-1)
    return F.l1_loss(pred_len, template_len.unsqueeze(0).expand_as(pred_len))


def laplacian_delta_loss(delta: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return delta.new_tensor(0.0)
    return torch.mean(torch.linalg.norm(delta[:, edges[:, 0]] - delta[:, edges[:, 1]], dim=-1))


def template_deform_loss(
    pred_vertices: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    template_vertices: torch.Tensor,
    edges: torch.Tensor,
    *,
    chamfer_points: int = 4096,
    edge_weight: float = 0.05,
    laplacian_weight: float = 0.10,
    displacement_weight: float = 0.001,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sample = sample_points(target, chamfer_points)
    pred_sample = sample_points(pred_vertices, chamfer_points)
    cd = chamfer_distance(pred_sample[..., :3], target_sample[..., :3])
    edge = edge_length_loss(pred_vertices, template_vertices, edges)
    lap = laplacian_delta_loss(delta, edges)
    disp = torch.mean(delta.square())
    loss = cd + edge_weight * edge + laplacian_weight * lap + displacement_weight * disp
    return loss, {
        "loss": float(loss.detach().cpu()),
        "chamfer": float(cd.detach().cpu()),
        "edge": float(edge.detach().cpu()),
        "laplacian": float(lap.detach().cpu()),
        "displacement": float(disp.detach().cpu()),
    }
