from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

from .sampling import farthest_point_sample, resample_closed_polyline


@dataclass(frozen=True)
class PreprocessConfig:
    roi_side_mm: float = 20.0
    prep_points: int = 8192
    antagonist_points: int = 8192
    crown_points: int = 16384
    margin_points: int = 1024
    candidate_factor: float = 4.0
    seed: int = 20260621


def iter_processed_cases(data_dir: Path) -> Iterable[Path]:
    for processed in sorted(data_dir.rglob("processed")):
        if (processed / "upper_local.stl").exists() and (processed / "lower_local.stl").exists():
            yield processed.parent


def _read_margin(path: Path) -> np.ndarray:
    margin = np.loadtxt(path, dtype=np.float32)
    if margin.ndim == 1:
        margin = margin.reshape(1, -1)
    if margin.shape[1] > 3:
        margin = margin[:, :3]
    if margin.shape[1] != 3:
        raise ValueError(f"Expected margin xyz with 3 columns: {path}")
    return margin.astype(np.float32)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry in scene: {path}")
        mesh = trimesh.util.concatenate(geoms)
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Empty mesh: {path}")
    mesh.remove_unreferenced_vertices()
    return mesh


def _mean_nearest_distance(a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(b_xyz).query(a_xyz, k=1)
        return float(np.mean(dist))
    except Exception:
        # Fallback avoids making scipy mandatory for simple inspections.
        chunks = []
        for start in range(0, a_xyz.shape[0], 256):
            q = a_xyz[start : start + 256]
            dist2 = np.sum((q[:, None, :] - b_xyz[None, :, :]) ** 2, axis=2)
            chunks.append(np.sqrt(np.min(dist2, axis=1)))
        return float(np.mean(np.concatenate(chunks)))


def _sample_mesh_points(
    mesh: trimesh.Trimesh,
    count: int,
    center: np.ndarray,
    rng: np.random.Generator,
    *,
    roi_half_side: float | None,
    candidate_factor: float,
) -> np.ndarray:
    target_candidates = max(count, int(count * candidate_factor))
    points_accum: list[np.ndarray] = []
    normals_accum: list[np.ndarray] = []
    attempts = 0

    while sum(p.shape[0] for p in points_accum) < count and attempts < 12:
        attempts += 1
        sample_count = target_candidates * attempts
        pts, face_idx = trimesh.sample.sample_surface(mesh, sample_count)
        normals = mesh.face_normals[face_idx]
        pts = pts.astype(np.float32) - center.astype(np.float32)
        normals = normals.astype(np.float32)

        if roi_half_side is not None:
            keep = np.all((pts >= -roi_half_side) & (pts <= roi_half_side), axis=1)
            pts = pts[keep]
            normals = normals[keep]

        if pts.size:
            points_accum.append(pts)
            normals_accum.append(normals)

    if not points_accum:
        raise ValueError("No sampled mesh points inside ROI")

    points = np.concatenate(points_accum, axis=0)
    normals = np.concatenate(normals_accum, axis=0)
    idx = farthest_point_sample(points, count, rng)
    sampled = np.concatenate([points[idx], normals[idx]], axis=1)
    return sampled.astype(np.float32)


def _find_crown_mesh(processed: Path, tooth_id: str | None) -> Path:
    if tooth_id:
        candidate = processed / f"crown_{tooth_id}.stl"
        if candidate.exists():
            return candidate
    crowns = sorted(processed.glob("crown_*.stl"))
    if not crowns:
        raise FileNotFoundError(f"No crown_*.stl found in {processed}")
    return crowns[0]


def _case_name_and_tooth(case_dir: Path) -> tuple[str, str | None]:
    if case_dir.name.isdigit():
        return case_dir.parent.name, case_dir.name
    if "_" in case_dir.name:
        prefix, suffix = case_dir.name.rsplit("_", 1)
        return prefix, suffix if suffix.isdigit() else None
    return case_dir.name, None


def preprocess_case(case_dir: Path, config: PreprocessConfig) -> dict:
    stable_case_hash = int(hashlib.sha1(str(case_dir).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(config.seed + stable_case_hash % 1_000_000)
    processed = case_dir / "processed"
    train = case_dir / "train"
    train.mkdir(parents=True, exist_ok=True)

    case_name, tooth_id = _case_name_and_tooth(case_dir)
    margin_raw = _read_margin(processed / "margin_local.xyz")
    center = margin_raw.mean(axis=0).astype(np.float32)
    margin = resample_closed_polyline(margin_raw - center, config.margin_points)

    upper_mesh = _load_mesh(processed / "upper_local.stl")
    lower_mesh = _load_mesh(processed / "lower_local.stl")
    crown_mesh = _load_mesh(_find_crown_mesh(processed, tooth_id))

    upper_dist = _mean_nearest_distance(margin_raw, np.asarray(upper_mesh.vertices, dtype=np.float32))
    lower_dist = _mean_nearest_distance(margin_raw, np.asarray(lower_mesh.vertices, dtype=np.float32))
    prep_arch = "upper" if upper_dist <= lower_dist else "lower"
    antagonist_arch = "lower" if prep_arch == "upper" else "upper"
    prep_mesh = upper_mesh if prep_arch == "upper" else lower_mesh
    antagonist_mesh = lower_mesh if prep_arch == "upper" else upper_mesh

    roi_half = config.roi_side_mm / 2.0
    prep_points = _sample_mesh_points(
        prep_mesh, config.prep_points, center, rng, roi_half_side=roi_half, candidate_factor=config.candidate_factor
    )
    antagonist_points = _sample_mesh_points(
        antagonist_mesh,
        config.antagonist_points,
        center,
        rng,
        roi_half_side=roi_half,
        candidate_factor=config.candidate_factor,
    )
    crown_points = _sample_mesh_points(
        crown_mesh,
        config.crown_points,
        center,
        rng,
        roi_half_side=None,
        candidate_factor=max(1.5, config.candidate_factor / 2.0),
    )

    np.save(train / "prep_points.npy", prep_points)
    np.save(train / "antagonist_points.npy", antagonist_points)
    np.save(train / "crown_points.npy", crown_points)
    np.save(train / "margin_points.npy", margin)

    metadata = {
        "case": case_dir.name,
        "patient_id": case_name,
        "tooth_id": tooth_id,
        "method": {
            "name": "margin_centered_fixed_cube_roi_fps",
            "roi_side_mm": config.roi_side_mm,
            "candidate_factor": config.candidate_factor,
            "scale_applied": False,
            "units": "millimeters",
        },
        "role_assignment": {
            "prep_arch": prep_arch,
            "antagonist_arch": antagonist_arch,
            "margin_to_upper_mean": upper_dist,
            "margin_to_lower_mean": lower_dist,
        },
        "coordinate_processing": {
            "center_xyz_mm": center.tolist(),
            "all_points_subtract_center_xyz_mm": True,
            "scale_applied": False,
        },
        "outputs": {
            "prep_points.npy": list(prep_points.shape),
            "antagonist_points.npy": list(antagonist_points.shape),
            "crown_points.npy": list(crown_points.shape),
            "margin_points.npy": list(margin.shape),
        },
    }
    (train / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (train / "bbox_scale.json").write_text(
        json.dumps(
            {
                "case": case_dir.name,
                "tooth_id": tooth_id,
                "units": "millimeters",
                "scale_applied": False,
                "roi_center_xyz_mm": center.tolist(),
                "roi_side_mm": config.roi_side_mm,
                "roi_half_side_mm": roi_half,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata
