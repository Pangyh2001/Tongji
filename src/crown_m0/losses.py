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
