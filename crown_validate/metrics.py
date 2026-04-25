from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = cKDTree(target)
    distances, _ = tree.query(source, k=1)
    return distances.astype(np.float64)


def compute_case_metrics(
    pred_points_mm: np.ndarray,
    gt_points_mm: np.ndarray,
    pred_margin_mm: np.ndarray | None,
    gt_margin_mm: np.ndarray | None,
    fscore_threshold_mm: float,
) -> dict[str, float]:
    pred_to_gt = _nearest_distances(pred_points_mm, gt_points_mm)
    gt_to_pred = _nearest_distances(gt_points_mm, pred_points_mm)

    precision = float(np.mean(pred_to_gt <= fscore_threshold_mm))
    recall = float(np.mean(gt_to_pred <= fscore_threshold_mm))
    fscore = 0.0
    if precision + recall > 0.0:
        fscore = 2.0 * precision * recall / (precision + recall)

    result = {
        "mean_pred_to_gt_mm": float(np.mean(pred_to_gt)),
        "mean_gt_to_pred_mm": float(np.mean(gt_to_pred)),
        "chamfer_l2_mm2": float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)),
        "hd95_mm": float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95))),
        "fscore": float(fscore),
    }

    if pred_margin_mm is not None and gt_margin_mm is not None and pred_margin_mm.size and gt_margin_mm.size:
        pred_margin_to_gt = _nearest_distances(pred_margin_mm, gt_margin_mm)
        gt_margin_to_pred = _nearest_distances(gt_margin_mm, pred_margin_mm)
        result["margin_chamfer_l2_mm2"] = float(np.mean(pred_margin_to_gt**2) + np.mean(gt_margin_to_pred**2))
        result["margin_hd95_mm"] = float(max(np.percentile(pred_margin_to_gt, 95), np.percentile(gt_margin_to_pred, 95)))
        result["margin_mean_mm"] = float((np.mean(pred_margin_to_gt) + np.mean(gt_margin_to_pred)) * 0.5)
    else:
        result["margin_chamfer_l2_mm2"] = np.nan
        result["margin_hd95_mm"] = np.nan
        result["margin_mean_mm"] = np.nan

    return result
