from __future__ import annotations

import torch


def _gather_last_dim(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))


def _local_roughness(points: torch.Tensor, knn: int) -> torch.Tensor:
    distances = torch.cdist(points, points)
    neighbor_distances = torch.topk(distances, k=knn + 1, largest=False).values[:, :, 1:]
    return neighbor_distances.mean(dim=-1)


def _margin_weights(points: torch.Tensor, margin_points: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
    if margin_points.shape[1] == 0:
        return torch.ones(points.shape[:2], device=points.device, dtype=points.dtype)
    distances = torch.cdist(points, margin_points).min(dim=-1).values
    return 1.0 + alpha * torch.exp(-(distances**2) / (2.0 * sigma * sigma + 1e-12))


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
    loss_cfg: dict[str, float | bool | int],
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = outputs["pred_points"]
    pred_normals = outputs["pred_normals"]
    gt = batch["crown_points"]
    gt_normals = batch["crown_normals"]
    margin_points = batch["margin_points"]

    pairwise = torch.cdist(pred, gt)
    pred_min, pred_to_gt_idx = pairwise.min(dim=-1)
    gt_min, _ = pairwise.min(dim=1)

    if bool(loss_cfg["use_margin_weighting"]) and margin_points.shape[1] > 0:
        pred_weights = _margin_weights(pred, margin_points, float(loss_cfg["margin_alpha"]), float(loss_cfg["margin_sigma"]))
        gt_weights = _margin_weights(gt, margin_points, float(loss_cfg["margin_alpha"]), float(loss_cfg["margin_sigma"]))
    else:
        pred_weights = torch.ones_like(pred_min)
        gt_weights = torch.ones_like(gt_min)

    chamfer = ((pred_weights * pred_min.pow(2)).mean(dim=1) + (gt_weights * gt_min.pow(2)).mean(dim=1)).mean()
    total = chamfer

    roughness_weight = float(loss_cfg["curvature_weight"])
    curvature_loss = torch.tensor(0.0, device=pred.device)
    if roughness_weight > 0.0:
        knn = int(loss_cfg["roughness_knn"])
        pred_rough = _local_roughness(pred, knn=knn)
        gt_rough = _local_roughness(gt, knn=knn)
        matched_gt_rough = torch.gather(gt_rough, 1, pred_to_gt_idx)
        curvature_loss = torch.abs(pred_rough - matched_gt_rough).mean()
        total = total + roughness_weight * curvature_loss

    normal_weight = float(loss_cfg["normal_weight"])
    normal_loss = torch.tensor(0.0, device=pred.device)
    if normal_weight > 0.0:
        matched_gt_normals = _gather_last_dim(gt_normals, pred_to_gt_idx)
        normal_loss = torch.mean((pred_normals - matched_gt_normals) ** 2)
        total = total + normal_weight * normal_loss

    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_chamfer": float(chamfer.detach().cpu()),
        "loss_curvature": float(curvature_loss.detach().cpu()),
        "loss_normal": float(normal_loss.detach().cpu()),
    }
