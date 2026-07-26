from __future__ import annotations

import numpy as np


def ensure_2d_float32(array: np.ndarray, columns: int, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def resample_rows(
    points: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    replace_if_needed: bool = True,
) -> np.ndarray:
    """Randomly resample rows to a fixed count.

    This is used at training time as a guardrail. The preprocessing script should
    create fixed-density FPS samples; this function prevents one malformed case
    from breaking a training run.
    """
    if points.shape[0] == count:
        return points
    replace = points.shape[0] < count
    if replace and not replace_if_needed:
        raise ValueError(f"Cannot sample {count} rows from {points.shape[0]} without replacement")
    idx = rng.choice(points.shape[0], size=count, replace=replace)
    return points[idx]


def farthest_point_sample(points_xyz: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices for farthest point sampling on xyz coordinates."""
    xyz = ensure_2d_float32(points_xyz, 3, "points_xyz")
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("Cannot farthest-point sample an empty point set")
    if n <= count:
        extra = rng.choice(n, size=count - n, replace=True) if n < count else np.array([], dtype=np.int64)
        return np.concatenate([np.arange(n, dtype=np.int64), extra])

    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(0, n))
    min_dist2 = np.full(n, np.inf, dtype=np.float32)

    for i in range(1, count):
        last = xyz[selected[i - 1]]
        dist2 = np.sum((xyz - last) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected[i] = int(np.argmax(min_dist2))

    return selected


def resample_closed_polyline(points_xyz: np.ndarray, count: int) -> np.ndarray:
    """Arc-length resample an ordered closed polyline."""
    pts = ensure_2d_float32(points_xyz, 3, "margin_points")
    if pts.shape[0] < 3:
        raise ValueError("A closed margin line needs at least 3 points")

    closed = np.concatenate([pts, pts[:1]], axis=0)
    seg = closed[1:] - closed[:-1]
    length = np.linalg.norm(seg, axis=1)
    total = float(length.sum())
    if total <= 0:
        raise ValueError("Margin line has zero length")

    cumulative = np.concatenate([[0.0], np.cumsum(length)])
    targets = np.linspace(0.0, total, count, endpoint=False)
    out = np.empty((count, 3), dtype=np.float32)

    seg_idx = np.searchsorted(cumulative, targets, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(length) - 1)
    denom = np.maximum(length[seg_idx], 1e-8)
    alpha = ((targets - cumulative[seg_idx]) / denom).astype(np.float32)
    out = closed[seg_idx] * (1.0 - alpha[:, None]) + closed[seg_idx + 1] * alpha[:, None]
    return out.astype(np.float32)

