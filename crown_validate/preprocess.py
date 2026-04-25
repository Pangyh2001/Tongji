from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .manifest import load_manifest
from .stl import extract_boundary_vertices, load_mesh, resample_points, sample_surface
from .utils import ensure_dir, json_dump


def _normalize(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return ((points - center[None, :]) / scale).astype(np.float32)


def _fallback_margin_vertices(crown_vertices: np.ndarray, prep_vertices: np.ndarray) -> np.ndarray:
    prep_tree = cKDTree(prep_vertices)
    nearest_distances, _ = prep_tree.query(crown_vertices, k=1)
    percentile_threshold = float(np.percentile(nearest_distances, 12.0))
    margin_mask = nearest_distances <= percentile_threshold
    fallback = crown_vertices[margin_mask]
    if fallback.shape[0] < 32:
        closest_indices = np.argsort(nearest_distances)[: max(32, crown_vertices.shape[0] // 20)]
        fallback = crown_vertices[closest_indices]
    return fallback.astype(np.float32)


def preprocess_case(
    row: dict[str, str],
    output_dir: str | Path,
    prep_points: int,
    opposing_points: int,
    crown_points: int,
    margin_points: int,
    seed: int,
) -> Path:
    case_dir = ensure_dir(Path(output_dir) / row["case_id"])
    prep_mesh = load_mesh(row["prep_path"])
    opposing_mesh = load_mesh(row["opposing_path"])
    crown_mesh = load_mesh(row["crown_path"])

    center = prep_mesh.vertices.mean(axis=0).astype(np.float32)
    support_vertices = np.concatenate(
        [prep_mesh.vertices, opposing_mesh.vertices, crown_mesh.vertices],
        axis=0,
    )
    scale = float(np.linalg.norm(support_vertices - center[None, :], axis=1).max())
    if scale <= 1e-8:
        scale = 1.0

    prep_sampled, prep_normals = sample_surface(prep_mesh, prep_points, seed + 11)
    opposing_sampled, opposing_normals = sample_surface(opposing_mesh, opposing_points, seed + 23)
    crown_sampled, crown_normals = sample_surface(crown_mesh, crown_points, seed + 37)

    boundary_vertices = extract_boundary_vertices(crown_mesh)
    margin_source = "mesh_boundary"
    if boundary_vertices.size == 0:
        boundary_vertices = _fallback_margin_vertices(crown_mesh.vertices, prep_mesh.vertices)
        margin_source = "prep_distance_fallback"
    boundary_sampled = resample_points(boundary_vertices, margin_points, seed + 51) if boundary_vertices.size else np.empty((0, 3), dtype=np.float32)

    np.savez_compressed(
        case_dir / "processed_case.npz",
        prep_points=_normalize(prep_sampled, center, scale),
        prep_normals=prep_normals.astype(np.float32),
        opposing_points=_normalize(opposing_sampled, center, scale),
        opposing_normals=opposing_normals.astype(np.float32),
        crown_points=_normalize(crown_sampled, center, scale),
        crown_normals=crown_normals.astype(np.float32),
        margin_points=_normalize(boundary_sampled, center, scale) if boundary_sampled.size else boundary_sampled,
        center=center.astype(np.float32),
        scale=np.asarray([scale], dtype=np.float32),
    )

    json_dump(
        {
            "case_id": row["case_id"],
            "patient_name": row["patient_name"],
            "tooth_position": row.get("tooth_position", ""),
            "prep_path": row["prep_path"],
            "opposing_path": row["opposing_path"],
            "crown_path": row["crown_path"],
            "needs_manual_confirmation": row.get("needs_manual_confirmation", "false"),
            "notes": row.get("notes", ""),
            "margin_source": margin_source,
        },
        case_dir / "metadata.json",
    )
    return case_dir / "processed_case.npz"


def preprocess_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    prep_points: int,
    opposing_points: int,
    crown_points: int,
    margin_points: int,
    seed: int,
    include_unconfirmed: bool = True,
) -> list[Path]:
    rows = load_manifest(manifest_path, include_unconfirmed=include_unconfirmed)
    outputs: list[Path] = []
    for index, row in enumerate(rows):
        outputs.append(
            preprocess_case(
                row=row,
                output_dir=output_dir,
                prep_points=prep_points,
                opposing_points=opposing_points,
                crown_points=crown_points,
                margin_points=margin_points,
                seed=seed + index * 101,
            )
        )
    return outputs
