from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_m0_official_experiment import (
    margin_distance_metrics,
    mesh_stats,
    read_mesh,
    reconstruct_dpsr_grid,
    regional_surface_metrics,
    restore_original_coordinates,
    sample_mesh_points,
    surface_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a topology-safe DPSR iso-level from saved prediction grids."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", default="-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08")
    parser.add_argument("--roi-half-extent-mm", type=float, default=12.0)
    parser.add_argument("--smooth-iterations", type=int, default=5)
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_score(
    mesh: o3d.geometry.TriangleMesh,
    pred_xyz: np.ndarray,
    margin_xyz: np.ndarray,
    level: float,
) -> tuple:
    stats = mesh_stats(mesh)
    vertices = np.asarray(mesh.vertices)
    tree = cKDTree(vertices)
    pred_dist, _ = tree.query(pred_xyz[:: max(1, len(pred_xyz) // 4096)], k=1)
    margin_dist, _ = tree.query(margin_xyz, k=1)
    genus = float(stats["genus"])
    topology_rank = 0 if stats["topology_ok"] else (1 if stats["watertight"] else 2)
    genus_rank = abs(genus) if math.isfinite(genus) else 999.0
    return (
        topology_rank,
        genus_rank,
        float(np.sqrt(np.mean(margin_dist**2))),
        float(np.sqrt(np.mean(pred_dist**2))),
        abs(level),
    )


def main() -> None:
    args = parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    levels = [float(value) for value in args.levels.split(",")]
    source_rows = read_rows(args.source / "test" / "metrics_by_case.csv")
    if args.max_cases > 0:
        source_rows = source_rows[: args.max_cases]
    output_test = args.output / "test"
    output_test.mkdir(parents=True, exist_ok=True)
    rows = []
    level_rows = []

    for source_row in source_rows:
        case_path = Path(source_row["case"])
        stem = source_row["stem"]
        source_case = args.source / "test" / "cases" / stem
        output_case = output_test / "cases" / stem
        output_case.mkdir(parents=True, exist_ok=True)
        grid = np.load(source_case / f"{stem}_pred_psr_grid.npy")
        pred_local = np.load(source_case / f"{stem}_pred.npy")
        pred_original = restore_original_coordinates(pred_local, case_path)[:, :3]
        margin_local = np.load(case_path / "train" / "margin_points.npy")
        margin_original = restore_original_coordinates(
            np.pad(margin_local[:, :3], ((0, 0), (0, 3))), case_path
        )[:, :3]

        candidates = []
        for level in levels:
            try:
                mesh = reconstruct_dpsr_grid(
                    grid,
                    case_path,
                    roi_half_extent_mm=args.roi_half_extent_mm,
                    smooth_iterations=args.smooth_iterations,
                    level=level,
                )
                score = candidate_score(mesh, pred_original, margin_original, level)
                stats = mesh_stats(mesh)
                candidates.append((score, level, mesh, stats))
                level_rows.append(
                    {
                        "case": str(case_path),
                        "stem": stem,
                        "level": level,
                        "score": repr(score),
                        **stats,
                    }
                )
            except Exception as exc:
                level_rows.append(
                    {
                        "case": str(case_path),
                        "stem": stem,
                        "level": level,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not candidates:
            rows.append({"case": str(case_path), "stem": stem, "ok": False})
            continue

        score, level, pred_mesh, stats = min(candidates, key=lambda item: item[0])
        pred_stl = output_case / f"{stem}_pred_dpsr_iso_topology_sweep.stl"
        o3d.io.write_triangle_mesh(str(pred_stl), pred_mesh, write_ascii=False)
        gt_source = source_case / f"{stem}_GT_technician.stl"
        gt_copy = output_case / gt_source.name
        shutil.copy2(gt_source, gt_copy)
        gt_mesh = read_mesh(gt_source)
        pred_surface = sample_mesh_points(pred_mesh, args.sample_points)
        gt_surface = sample_mesh_points(gt_mesh, args.sample_points)
        row = {
            "case": str(case_path),
            "stem": stem,
            "ok": True,
            "selected_level": level,
            "selection_score": repr(score),
            **{f"stl_{k}": v for k, v in surface_metrics(pred_surface, gt_surface).items()},
            **{
                f"margin_stl_{k}": v
                for k, v in margin_distance_metrics(margin_original, pred_surface).items()
            },
            **{
                f"r1_stl_{k}": v
                for k, v in regional_surface_metrics(
                    pred_surface, gt_surface, margin_original, radius_mm=1.0
                ).items()
            },
            **stats,
            "pred_stl": str(pred_stl),
            "gt_stl": str(gt_copy),
        }
        rows.append(row)

    write_rows(output_test / "metrics_by_case.csv", rows)
    write_rows(args.output / "level_candidates.csv", level_rows)
    ok = [row for row in rows if row.get("ok")]
    summary = {
        "n": len(ok),
        "topology_ok": sum(bool(row["topology_ok"]) for row in ok),
        "topology_failures": sum(not bool(row["topology_ok"]) for row in ok),
    }
    for key in (
        "stl_symmetric_rms",
        "stl_fscore_0p3",
        "margin_stl_rms",
        "r1_stl_symmetric_rms",
    ):
        values = [float(row[key]) for row in ok if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
