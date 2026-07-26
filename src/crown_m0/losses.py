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
