from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


METHOD = "m2_template_deform_m0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deform retrieved template meshes using M0 predictions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--m1-dir", type=Path, default=Path("visualizations/m1_retrieval_representative10"))
    parser.add_argument("--m1-method", default="m1a_rigid_pca")
    parser.add_argument("--prediction-dir", type=Path, default=Path("predictions/m0_new_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/m2_template_deform_m0_representative10"))
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument("--alpha", type=float, default=0.35, help="Blend weight toward nearest M0 predicted points.")
    parser.add_argument("--max-displacement", type=float, default=1.2)
    parser.add_argument("--smooth-iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    rows_in = read_m1_rows(args.m1_dir / "metrics_by_case.csv", args.m1_method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gt_stl").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "template_stl").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index, row_in in enumerate(rows_in, start=1):
        stem = row_in["stem"]
        case = row_in["case"]
        print(f"[{index}/{len(rows_in)}] {stem}", flush=True)
        row = {
            "case": case,
            "stem": stem,
            "method": METHOD,
            "ok": False,
            "template_case": row_in.get("template_case", ""),
            "template_stl": "",
            "gt_stl": "",
            "error": "",
        }
        try:
            init_mesh = read_mesh(Path(row_in["stl"]))
            gt_stl = Path(row_in["gt_stl"])
            template_stl = Path(row_in["template_stl"])
            pred_path = args.prediction_dir / f"{stem}.npy"
            if not pred_path.exists():
                raise FileNotFoundError(f"missing M0 prediction: {pred_path}")

            gt_copy = args.output_dir / "gt_stl" / gt_stl.name
            template_copy = args.output_dir / "template_stl" / template_stl.name
            if not gt_copy.exists():
                shutil.copy2(gt_stl, gt_copy)
            if not template_copy.exists():
                shutil.copy2(template_stl, template_copy)

            target_center = load_center(Path(case))
            m0_points = np.load(pred_path).astype(np.float64)[:, :3] + target_center[None, :]
            mesh = deform_mesh_to_points(init_mesh, m0_points, args.alpha, args.max_displacement)
            if args.smooth_iterations > 0:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iterations)
            mesh = cleanup_mesh(mesh)

            out_case = args.output_dir / METHOD / stem
            out_case.mkdir(parents=True, exist_ok=True)
            out_stl = out_case / f"{stem}_{METHOD}.stl"
            ok = o3d.io.write_triangle_mesh(str(out_stl), mesh, write_ascii=False)
            if not ok:
                raise RuntimeError(f"failed to write {out_stl}")

            gt_mesh = read_mesh(gt_stl)
            metrics = surface_metrics(sample_mesh_points(mesh, args.sample_points), sample_mesh_points(gt_mesh, args.sample_points))
            row.update(metrics)
            row.update(mesh_stats(mesh))
            row.update({"ok": True, "stl": str(out_stl), "gt_stl": str(gt_copy), "template_stl": str(template_copy)})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write_outputs(args.output_dir, rows)

    write_outputs(args.output_dir, rows)
    print(f"wrote {args.output_dir}", flush=True)


def read_m1_rows(path: Path, method: str) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("method") == method and row.get("ok") == "True"]
    if not rows:
        raise SystemExit(f"No successful rows for {method} in {path}")
    return rows


def load_center(case_path: Path) -> np.ndarray:
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    return np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float64)


def deform_mesh_to_points(mesh: o3d.geometry.TriangleMesh, target_xyz: np.ndarray, alpha: float, max_displacement: float) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(out.vertices, dtype=np.float64)
    distances, indices = cKDTree(target_xyz).query(vertices, k=1)
    nearest = target_xyz[indices]
    displacement = nearest - vertices
    norm = np.linalg.norm(displacement, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_displacement / np.maximum(norm, 1e-8))
    new_vertices = vertices + alpha * displacement * scale
    out.vertices = o3d.utility.Vector3dVector(new_vertices)
    return out


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"empty mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def sample_mesh_points(mesh: o3d.geometry.TriangleMesh, count: int) -> np.ndarray:
    pcd = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(pcd.points)


def surface_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    pred_to_gt, _ = cKDTree(gt_xyz).query(pred_xyz, k=1)
    gt_to_pred, _ = cKDTree(pred_xyz).query(gt_xyz, k=1)
    both = np.concatenate([pred_to_gt, gt_to_pred])
    return {
        "pred_to_gt_mean": mean(pred_to_gt),
        "pred_to_gt_rms": rms(pred_to_gt),
        "pred_to_gt_hd95": percentile(pred_to_gt, 95),
        "gt_to_pred_mean": mean(gt_to_pred),
        "gt_to_pred_rms": rms(gt_to_pred),
        "gt_to_pred_hd95": percentile(gt_to_pred, 95),
        "symmetric_mean": mean(both),
        "symmetric_rms": rms(both),
    }


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts_np = np.asarray(counts)
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "components": int(len(counts_np)),
        "largest_component_triangles": int(counts_np.max()) if counts_np.size else 0,
        "surface_area": float(mesh.get_surface_area()),
        "edge_manifold": bool(mesh.is_edge_manifold()),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
    }


def write_outputs(output_dir: Path, rows: list[dict]) -> None:
    fieldnames = [
        "case", "stem", "method", "ok", "template_case",
        "pred_to_gt_mean", "pred_to_gt_rms", "pred_to_gt_hd95",
        "gt_to_pred_mean", "gt_to_pred_rms", "gt_to_pred_hd95",
        "symmetric_mean", "symmetric_rms",
        "vertices", "triangles", "components", "largest_component_triangles", "surface_area",
        "edge_manifold", "vertex_manifold", "stl", "gt_stl", "template_stl", "error",
    ]
    with (output_dir / "metrics_by_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = summarize([row for row in rows if row.get("ok")])
    (output_dir / "summary_by_method.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary_by_method.csv", summary)


def summarize(rows: list[dict]) -> list[dict]:
    metric_keys = [
        "pred_to_gt_rms", "gt_to_pred_rms", "symmetric_rms",
        "pred_to_gt_hd95", "gt_to_pred_hd95", "triangles", "surface_area",
    ]
    if not rows:
        return []
    item = {"method": METHOD, "n": len(rows)}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows if row.get(key) not in ("", None)])
        if values.size:
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_median"] = float(np.median(values))
            item[f"{key}_max"] = float(np.max(values))
    return [item]


def write_summary_csv(path: Path, summary: list[dict]) -> None:
    keys = sorted({key for row in summary for key in row})
    fieldnames = ["method", "n"] + [key for key in keys if key not in ("method", "n")]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def rms(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values**2))))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


if __name__ == "__main__":
    main()
