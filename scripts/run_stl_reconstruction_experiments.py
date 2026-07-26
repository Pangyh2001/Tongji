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


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

METHODS = ("alpha_raw", "alpha_clean", "poisson_raw", "poisson_clean_taubin", "bpa_clean")
OFFICIAL_STL_METHOD = "alpha_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct crown point clouds as STL meshes. The official experiment "
            "default is alpha_clean; pass --methods explicitly only for STL-method ablations."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("predictions/m0_new_all"))
    parser.add_argument(
        "--source",
        choices=["prediction", "gt-crown-points"],
        default="prediction",
        help="Reconstruct M0 predictions or GT crown_points.npy for reconstruction-upper-bound tests.",
    )
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--case-list",
        type=Path,
        default=None,
        help="Optional newline-delimited case list. Overrides --split-file/--split when provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/m0_stl_reconstruction_experiments"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=[OFFICIAL_STL_METHOD],
        help="STL reconstruction methods. Default is the official unified pipeline: alpha_clean.",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--poisson-depth", type=int, default=6)
    parser.add_argument("--density-quantile", type=float, default=0.03)
    parser.add_argument("--normal-radius", type=float, default=1.0)
    parser.add_argument("--normal-max-nn", type=int, default=40)
    parser.add_argument(
        "--consistent-normals",
        action="store_true",
        help="Use Open3D's slower tangent-plane normal orientation before Poisson/BPA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.case_list:
        cases = [line.strip() for line in args.case_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(args.split_file.read_text(encoding="utf-8"))
        cases = payload[args.split]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = args.output_dir / "gt_stl"
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for case_index, case in enumerate(cases, start=1):
        case_path = Path(case)
        stem = safe_case_name(case_path, args.data_dir)
        print(f"[{case_index}/{len(cases)}] {stem}", flush=True)

        pred_npy = args.prediction_dir / f"{stem}.npy"
        gt_points_npy = case_path / "train" / "crown_points.npy"
        gt_stl = find_gt_stl(case_path)
        if args.source == "prediction" and not pred_npy.exists():
            rows.append({"case": case, "method": "", "ok": False, "error": "missing prediction"})
            continue
        if args.source == "gt-crown-points" and not gt_points_npy.exists():
            rows.append({"case": case, "method": "", "ok": False, "error": "missing GT crown_points.npy"})
            continue
        if gt_stl is None or not gt_stl.exists():
            rows.append({"case": case, "method": "", "ok": False, "error": "missing GT STL"})
            continue

        gt_copy = gt_dir / f"{stem}_GT_technician.stl"
        if not gt_copy.exists():
            shutil.copy2(gt_stl, gt_copy)

        source_npy = pred_npy if args.source == "prediction" else gt_points_npy
        pred_arr = np.load(source_npy).astype(np.float64)
        pred_arr = restore_original_coordinates(pred_arr, case_path)
        gt_mesh = read_mesh(gt_stl)
        gt_sample = sample_mesh_points(gt_mesh, args.sample_points)

        for method in args.methods:
            print(f"  - {method}", flush=True)
            row = {"case": case, "stem": stem, "method": method, "ok": False, "error": ""}
            try:
                mesh = reconstruct(pred_arr, method, args)
                mesh = cleanup_mesh(mesh, keep_largest=True)
                out_case = args.output_dir / method / stem
                out_case.mkdir(parents=True, exist_ok=True)
                out_stl = out_case / f"{stem}_{args.source}_{method}.stl"
                ok = o3d.io.write_triangle_mesh(str(out_stl), mesh, write_ascii=False)
                if not ok:
                    raise RuntimeError(f"failed to write {out_stl}")

                pred_sample = sample_mesh_points(mesh, args.sample_points)
                metrics = surface_metrics(pred_sample, gt_sample)
                row.update(metrics)
                row.update(mesh_stats(mesh))
                row.update({"ok": True, "source": args.source, "stl": str(out_stl), "gt_stl": str(gt_copy)})
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_outputs(args.output_dir, rows)

    write_outputs(args.output_dir, rows)
    print(f"wrote {args.output_dir}", flush=True)


def reconstruct(pred_arr: np.ndarray, method: str, args: argparse.Namespace) -> o3d.geometry.TriangleMesh:
    xyz = pred_arr[:, :3]
    pcd = make_point_cloud(xyz)

    if method == "alpha_raw":
        return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, args.alpha)

    if method == "alpha_clean":
        clean = clean_point_cloud(pcd)
        return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(clean, args.alpha)

    if method == "poisson_raw":
        p = prepare_normals(pcd, args)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p, depth=args.poisson_depth)
        return postprocess_poisson(mesh, densities, xyz, args, smooth_iterations=0)

    if method == "poisson_clean_taubin":
        clean = clean_point_cloud(pcd)
        p = prepare_normals(clean, args)
        clean_xyz = np.asarray(p.points)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p, depth=args.poisson_depth)
        return postprocess_poisson(mesh, densities, clean_xyz, args, smooth_iterations=15)

    if method == "bpa_clean":
        clean = clean_point_cloud(pcd)
        p = prepare_normals(clean, args)
        distances = np.asarray(p.compute_nearest_neighbor_distance())
        median_spacing = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 0.25
        radii = o3d.utility.DoubleVector([median_spacing * x for x in (1.5, 2.5, 4.0)])
        return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(p, radii)

    raise ValueError(f"Unsupported method: {method}")


def make_point_cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError(f"invalid xyz array: {xyz.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def clean_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    clean, _ = clean.remove_radius_outlier(nb_points=8, radius=0.55)
    if len(clean.points) < 1000:
        clean = pcd
    return clean.voxel_down_sample(voxel_size=0.06)


def prepare_normals(pcd: o3d.geometry.PointCloud, args: argparse.Namespace) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(pcd)
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=args.normal_max_nn)
    )
    if args.consistent_normals:
        try:
            p.orient_normals_consistent_tangent_plane(30)
            return p
        except RuntimeError:
            pass
    try:
        p.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 100.0]))
    except RuntimeError:
        pass
    return p


def postprocess_poisson(
    mesh: o3d.geometry.TriangleMesh,
    densities: o3d.utility.DoubleVector,
    source_xyz: np.ndarray,
    args: argparse.Namespace,
    *,
    smooth_iterations: int,
) -> o3d.geometry.TriangleMesh:
    densities_np = np.asarray(densities)
    if 0.0 < args.density_quantile < 0.5 and densities_np.size:
        threshold = np.quantile(densities_np, args.density_quantile)
        mesh.remove_vertices_by_mask(densities_np < threshold)

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=source_xyz.min(axis=0) - 0.8,
        max_bound=source_xyz.max(axis=0) + 0.8,
    )
    mesh = mesh.crop(bbox)
    if smooth_iterations > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    return mesh


def cleanup_mesh(mesh: o3d.geometry.TriangleMesh, *, keep_largest: bool) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if keep_largest and len(mesh.triangles) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_np = np.asarray(labels)
        counts_np = np.asarray(counts)
        if counts_np.size > 1:
            keep_label = int(np.argmax(counts_np))
            mesh.remove_triangles_by_mask(labels_np != keep_label)
            mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    if len(mesh.triangles) == 0:
        raise ValueError("empty reconstructed mesh")
    return mesh


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"empty mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def sample_mesh_points(mesh: o3d.geometry.TriangleMesh, count: int) -> np.ndarray:
    if len(mesh.triangles) == 0:
        raise ValueError("cannot sample empty mesh")
    pcd = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(pcd.points)


def surface_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    pred_to_gt, _ = cKDTree(gt_xyz).query(pred_xyz, k=1)
    gt_to_pred, _ = cKDTree(pred_xyz).query(gt_xyz, k=1)
    return {
        "pred_to_gt_mean": mean(pred_to_gt),
        "pred_to_gt_rms": rms(pred_to_gt),
        "pred_to_gt_hd95": percentile(pred_to_gt, 95),
        "gt_to_pred_mean": mean(gt_to_pred),
        "gt_to_pred_rms": rms(gt_to_pred),
        "gt_to_pred_hd95": percentile(gt_to_pred, 95),
        "symmetric_mean": mean(np.concatenate([pred_to_gt, gt_to_pred])),
        "symmetric_rms": rms(np.concatenate([pred_to_gt, gt_to_pred])),
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
    metrics_path = output_dir / "metrics_by_case.csv"
    fieldnames = [
        "case",
        "stem",
        "method",
        "source",
        "ok",
        "pred_to_gt_mean",
        "pred_to_gt_rms",
        "pred_to_gt_hd95",
        "gt_to_pred_mean",
        "gt_to_pred_rms",
        "gt_to_pred_hd95",
        "symmetric_mean",
        "symmetric_rms",
        "vertices",
        "triangles",
        "components",
        "largest_component_triangles",
        "surface_area",
        "edge_manifold",
        "vertex_manifold",
        "stl",
        "gt_stl",
        "error",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ok_rows = [r for r in rows if r.get("ok")]
    summary = summarize(ok_rows)
    (output_dir / "summary_by_method.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(output_dir / "summary_by_method.csv", summary)


def summarize(rows: list[dict]) -> list[dict]:
    metric_keys = [
        "pred_to_gt_mean",
        "pred_to_gt_rms",
        "pred_to_gt_hd95",
        "gt_to_pred_mean",
        "gt_to_pred_rms",
        "gt_to_pred_hd95",
        "symmetric_mean",
        "symmetric_rms",
        "triangles",
        "surface_area",
    ]
    out = []
    for method in METHODS:
        method_rows = [r for r in rows if r.get("method") == method]
        if not method_rows:
            continue
        item = {"method": method, "n": len(method_rows)}
        for key in metric_keys:
            values = np.asarray([float(r[key]) for r in method_rows if r.get(key) not in ("", None)], dtype=float)
            if values.size:
                item[f"{key}_mean"] = float(np.mean(values))
                item[f"{key}_median"] = float(np.median(values))
                item[f"{key}_max"] = float(np.max(values))
        out.append(item)
    return out


def write_summary_csv(path: Path, summary: list[dict]) -> None:
    keys = sorted({key for row in summary for key in row.keys()})
    preferred = ["method", "n"]
    fieldnames = preferred + [key for key in keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def find_gt_stl(case_path: Path) -> Path | None:
    crowns = sorted((case_path / "processed").glob("crown_*.stl"))
    return crowns[0] if crowns else None


def restore_original_coordinates(pred_arr: np.ndarray, case_path: Path) -> np.ndarray:
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float64)
    restored = pred_arr.copy()
    restored[:, :3] += center[None, :]
    return restored


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        rel = case_path
    return "__".join(rel.parts)


def mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def rms(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values**2))))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


if __name__ == "__main__":
    main()
