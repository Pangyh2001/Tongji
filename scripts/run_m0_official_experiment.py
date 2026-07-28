from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from skimage import measure
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crown_m0.dataset import CrownDataset, discover_cases
from src.crown_m0.io import write_ply, write_xyz
from src.crown_m0.model import (
    M0CoarseToFineNet,
    M0CrownNet,
    M0DMCDPSRNet,
    M0TangentCoarseToFineNet,
)


DEFAULT_STL_METHOD = "tangent_mls_poisson"


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Run the official M0 baseline evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/m0_new/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument(
        "--representative-list",
        type=Path,
        default=None,
        help="Optional extra case list for visual STL comparison. Official runs only write test by default.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("result"))
    parser.add_argument("--date", default=today, help="Result date folder, formatted as YYYYMMDD.")
    parser.add_argument("--experiment-name", default="m0_baseline")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument(
        "--stl-method",
        choices=["alpha_clean_taubin", "tangent_mls_poisson", "dmc_dpsr_marching_cubes"],
        default=DEFAULT_STL_METHOD,
    )
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--smooth-iterations", type=int, default=25)
    parser.add_argument("--subdivide-iterations", type=int, default=1)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-threads", type=int, default=8)
    parser.add_argument("--mls-iterations", type=int, default=2)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    output_dir = args.output_root / args.date / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_cases(args.data_dir)
    by_case = {str(record.train_dir.parent): record for record in records}
    sample_sets = load_sample_sets(args)

    model = load_model(args)
    if isinstance(model, M0DMCDPSRNet):
        args.roi_half_extent_mm = model.roi_half_extent_mm
    write_config(output_dir / "config.json", args)
    all_summary_rows = []
    for sample_set, cases in sample_sets.items():
        selected_records = [by_case[case] for case in cases if case in by_case]
        missing = [case for case in cases if case not in by_case]
        if missing:
            print(f"[warn] {sample_set}: missing {len(missing)} cases", flush=True)
        sample_dir = output_dir / sample_set
        rows = run_sample_set(model, selected_records, sample_dir, args)
        write_sample_outputs(sample_dir, rows)
        summary = summarize_rows(rows)
        for key, value in summary.items():
            all_summary_rows.append({"sample_set": sample_set, **value})

    write_csv(output_dir / "summary_by_sample_set.csv", all_summary_rows)
    print(f"wrote {output_dir}", flush=True)


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("decoder") == "dmc_dpsr":
        model = M0DMCDPSRNet(
            model_dim=int(checkpoint_args.get("dmc_model_dim", 256)),
            context_tokens_per_input=int(checkpoint_args.get("dmc_context_tokens", 256)),
            num_queries=int(checkpoint_args.get("dmc_queries", 256)),
            fold_step=int(checkpoint_args.get("dmc_fold_step", 8)),
            transformer_layers=int(checkpoint_args.get("dmc_transformer_layers", 3)),
            dpsr_resolution=int(checkpoint_args.get("dpsr_resolution", 128)),
            dpsr_sigma=float(checkpoint_args.get("dpsr_sigma", 2.0)),
            roi_half_extent_mm=float(checkpoint_args.get("roi_half_extent_mm", 12.0)),
        ).to(args.device)
    elif checkpoint_args.get("decoder") == "coarse_to_fine_tangent":
        model = M0TangentCoarseToFineNet(
            coarse_points=int(checkpoint_args.get("coarse_points", 8192)),
            first_factor=int(checkpoint_args.get("first_factor", 4)),
            second_factor=int(checkpoint_args.get("second_factor", 2)),
        ).to(args.device)
    elif checkpoint_args.get("decoder") == "coarse_to_fine":
        model = M0CoarseToFineNet(
            coarse_points=int(checkpoint_args.get("coarse_points", 8192)),
            first_factor=int(checkpoint_args.get("first_factor", 4)),
            second_factor=int(checkpoint_args.get("second_factor", 2)),
        ).to(args.device)
    else:
        model = M0CrownNet(output_points=int(checkpoint_args.get("output_points", 16384))).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_sample_sets(args: argparse.Namespace) -> dict[str, list[str]]:
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    sample_sets = {"test": split["test"]}
    if args.representative_list and args.representative_list.exists():
        sample_sets["representative10"] = [
            line.strip()
            for line in args.representative_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sample_sets


def run_sample_set(
    model: M0CrownNet,
    records,
    sample_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    loader = DataLoader(CrownDataset(records), batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=sample_dir.name):
            model_args = (
                batch["prep"].to(args.device),
                batch["antagonist"].to(args.device),
                batch["tooth_index"].to(args.device),
                batch["prep_arch_index"].to(args.device),
            )
            if isinstance(model, M0DMCDPSRNet):
                outputs = model(*model_args, return_grid=True)
                pred = outputs["points"].cpu().numpy()
                pred_grids = outputs["psr_grid"].cpu().numpy()
            else:
                pred = model(*model_args).cpu().numpy()
                pred_grids = [None] * len(pred)
            crown = batch["crown"].cpu().numpy()
            for i, case in enumerate(batch["case_id"]):
                case_path = Path(case)
                row = evaluate_case(
                    case_path,
                    pred[i].astype(np.float32),
                    crown[i].astype(np.float32),
                    sample_dir,
                    args,
                    pred_grid=None if pred_grids[i] is None else pred_grids[i].astype(np.float32),
                )
                rows.append(row)
    return rows


def evaluate_case(
    case_path: Path,
    pred_local: np.ndarray,
    gt_local: np.ndarray,
    sample_dir: Path,
    args: argparse.Namespace,
    *,
    pred_grid: np.ndarray | None = None,
) -> dict:
    stem = safe_case_name(case_path, args.data_dir)
    case_dir = sample_dir / "cases" / stem
    case_dir.mkdir(parents=True, exist_ok=True)

    row = {"case": str(case_path), "stem": stem, "ok": False, "stl_method": args.stl_method, "error": ""}
    pred_npy = case_dir / f"{stem}_pred.npy"
    pred_xyz = case_dir / f"{stem}_pred.xyz"
    pred_ply = case_dir / f"{stem}_pred.ply"
    pred_grid_npy = case_dir / f"{stem}_pred_psr_grid.npy"
    pred_stl = case_dir / f"{stem}_pred_{args.stl_method}.stl"
    gt_stl_copy = case_dir / f"{stem}_GT_technician.stl"

    try:
        np.save(pred_npy, pred_local)
        write_xyz(pred_xyz, pred_local)
        write_ply(pred_ply, pred_local)

        row.update(prefix_metrics("point", point_metrics(pred_local[:, :3], gt_local[:, :3])))

        pred_original = restore_original_coordinates(pred_local, case_path)
        if args.stl_method == "dmc_dpsr_marching_cubes":
            if pred_grid is None:
                raise ValueError("DMC DPSR STL export requires a predicted PSR grid")
            np.save(pred_grid_npy, pred_grid)
            pred_mesh = reconstruct_dpsr_grid(
                pred_grid,
                case_path,
                roi_half_extent_mm=float(args.roi_half_extent_mm),
                smooth_iterations=min(args.smooth_iterations, 5),
            )
        elif args.stl_method == "tangent_mls_poisson":
            pred_mesh = reconstruct_tangent_mls_poisson(
                pred_original,
                depth=args.poisson_depth,
                threads=args.poisson_threads,
                mls_iterations=args.mls_iterations,
                density_quantile=args.poisson_density_quantile,
                smooth_iterations=min(args.smooth_iterations, 10),
            )
        else:
            pred_mesh = reconstruct_alpha_clean(
                pred_original[:, :3],
                args.alpha,
                smooth_iterations=args.smooth_iterations,
                subdivide_iterations=args.subdivide_iterations,
            )
        ok = o3d.io.write_triangle_mesh(str(pred_stl), pred_mesh, write_ascii=False)
        if not ok:
            raise RuntimeError(f"failed to write {pred_stl}")

        gt_stl = find_gt_stl(case_path)
        if gt_stl is None:
            raise FileNotFoundError(f"missing GT STL for {case_path}")
        shutil.copy2(gt_stl, gt_stl_copy)
        gt_mesh = read_mesh(gt_stl)
        stl_metrics = surface_metrics(
            sample_mesh_points(pred_mesh, args.sample_points),
            sample_mesh_points(gt_mesh, args.sample_points),
        )
        row.update(prefix_metrics("stl", stl_metrics))
        row.update(mesh_stats(pred_mesh))
        row.update(
            {
                "ok": True,
                "pred_npy": str(pred_npy),
                "pred_xyz": str(pred_xyz),
                "pred_ply": str(pred_ply),
                "pred_psr_grid": str(pred_grid_npy) if pred_grid is not None else "",
                "pred_stl": str(pred_stl),
                "gt_stl": str(gt_stl_copy),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def reconstruct_dpsr_grid(
    grid: np.ndarray,
    case_path: Path,
    *,
    roi_half_extent_mm: float,
    smooth_iterations: int,
) -> o3d.geometry.TriangleMesh:
    """Extract the model's learned zero level set without point-cloud remeshing."""
    if not (float(grid.min()) <= 0.0 <= float(grid.max())):
        raise ValueError(
            f"PSR grid has no zero crossing: min={float(grid.min()):.6f}, "
            f"max={float(grid.max()):.6f}"
        )
    vertices, faces, _, _ = measure.marching_cubes(grid, level=0.0)
    resolution = np.asarray(grid.shape, dtype=np.float64)
    local_vertices = (
        vertices.astype(np.float64) / resolution[None, :]
    ) * (2.0 * roi_half_extent_mm) - roi_half_extent_mm
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float64)
    vertices_original = local_vertices + center[None, :]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_original),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh = keep_largest_mesh_component(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = keep_largest_mesh_component(mesh)
    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=smooth_iterations,
            lambda_filter=0.5,
            mu=-0.53,
        )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    if len(mesh.triangles) == 0:
        raise ValueError("empty DPSR marching-cubes mesh")
    return mesh


def reconstruct_alpha_clean(
    xyz: np.ndarray,
    alpha: float,
    *,
    smooth_iterations: int,
    subdivide_iterations: int,
) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    clean, _ = clean.remove_radius_outlier(nb_points=8, radius=0.55)
    if len(clean.points) < 1000:
        clean = pcd
    clean = clean.voxel_down_sample(voxel_size=0.06)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(clean, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_np = np.asarray(labels)
        counts_np = np.asarray(counts)
        if counts_np.size > 1:
            keep_label = int(np.argmax(counts_np))
            mesh.remove_triangles_by_mask(labels_np != keep_label)
            mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise ValueError("empty alpha_clean mesh")
    if subdivide_iterations > 0:
        mesh = mesh.subdivide_midpoint(number_of_iterations=subdivide_iterations)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=smooth_iterations,
            lambda_filter=0.5,
            mu=-0.53,
        )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    if len(mesh.triangles) == 0:
        raise ValueError("empty alpha_clean mesh")
    return mesh


def project_points_mls(
    xyz: np.ndarray,
    reference_normals: np.ndarray,
    *,
    iterations: int,
    neighbors: int = 24,
    strength: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    projected = np.asarray(xyz, dtype=np.float64).copy()
    reference = np.asarray(reference_normals, dtype=np.float64).copy()
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-8)
    for _ in range(max(iterations, 0)):
        tree = cKDTree(projected)
        _, indices = tree.query(projected, k=min(neighbors, len(projected)))
        neighborhoods = projected[indices]
        centroids = neighborhoods.mean(axis=1)
        normals = reference[indices].mean(axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        flip = np.sum(normals * reference, axis=1) < 0
        normals[flip] *= -1.0
        signed = np.sum((centroids - projected) * normals, axis=1)
        signed = np.clip(signed, -0.08, 0.08)
        projected += strength * signed[:, None] * normals
        reference = normals
    return projected, reference


def keep_largest_mesh_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts_np = np.asarray(counts)
    if counts_np.size > 1:
        labels_np = np.asarray(labels)
        mesh.remove_triangles_by_mask(labels_np != int(np.argmax(counts_np)))
        mesh.remove_unreferenced_vertices()
    return mesh


def reconstruct_tangent_mls_poisson(
    points: np.ndarray,
    *,
    depth: int,
    threads: int,
    mls_iterations: int,
    density_quantile: float,
    smooth_iterations: int,
) -> o3d.geometry.TriangleMesh:
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    normals = np.asarray(points[:, 3:6], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    clean, indices = pcd.remove_statistical_outlier(nb_neighbors=32, std_ratio=1.75)
    if len(clean.points) >= 1000:
        clean.normals = o3d.utility.Vector3dVector(normals[np.asarray(indices)])
        pcd = clean
    pcd = pcd.voxel_down_sample(voxel_size=0.08)
    pcd.normalize_normals()
    xyz = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    xyz, normals = project_points_mls(
        xyz,
        normals,
        iterations=mls_iterations,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.05,
        linear_fit=False,
        n_threads=threads,
    )
    densities_np = np.asarray(densities)
    if densities_np.size:
        threshold = float(np.quantile(densities_np, density_quantile))
        mesh.remove_vertices_by_mask(densities_np < threshold)

    xyz_clean = np.asarray(pcd.points)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        xyz_clean.min(axis=0) - 0.20,
        xyz_clean.max(axis=0) + 0.20,
    )
    mesh = mesh.crop(bbox)
    mesh = keep_largest_mesh_component(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = keep_largest_mesh_component(mesh)
    if len(mesh.triangles) > 50000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        mesh = keep_largest_mesh_component(mesh)
    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=smooth_iterations,
            lambda_filter=0.5,
            mu=-0.53,
        )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    if len(mesh.triangles) == 0:
        raise ValueError("empty tangent MLS Poisson mesh")
    return mesh


def restore_original_coordinates(arr: np.ndarray, case_path: Path) -> np.ndarray:
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float64)
    restored = arr.astype(np.float64).copy()
    restored[:, :3] += center[None, :]
    return restored


def find_gt_stl(case_path: Path) -> Path | None:
    crowns = sorted((case_path / "processed").glob("crown_*.stl"))
    return crowns[0] if crowns else None


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"empty mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def sample_mesh_points(mesh: o3d.geometry.TriangleMesh, count: int) -> np.ndarray:
    pcd = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(pcd.points)


def point_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    return surface_metrics(pred_xyz, gt_xyz)


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


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_sample_outputs(sample_dir: Path, rows: list[dict]) -> None:
    write_csv(sample_dir / "metrics_by_case.csv", rows)
    summary = summarize_rows(rows)
    (sample_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(sample_dir / "summary_metrics.csv", list(summary.values()))


def summarize_rows(rows: list[dict]) -> dict[str, dict]:
    ok_rows = [row for row in rows if row.get("ok")]
    metric_keys = [
        "point_pred_to_gt_rms",
        "point_gt_to_pred_rms",
        "point_symmetric_rms",
        "point_pred_to_gt_hd95",
        "point_gt_to_pred_hd95",
        "stl_pred_to_gt_rms",
        "stl_gt_to_pred_rms",
        "stl_symmetric_rms",
        "stl_pred_to_gt_hd95",
        "stl_gt_to_pred_hd95",
        "triangles",
        "surface_area",
    ]
    source_rows = ok_rows or rows
    method = str(source_rows[0].get("stl_method", DEFAULT_STL_METHOD)) if source_rows else DEFAULT_STL_METHOD
    summary = {method: {"method": method, "n": len(ok_rows)}}
    item = summary[method]
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in ok_rows if row.get(key) not in ("", None)], dtype=float)
        if values.size:
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_median"] = float(np.median(values))
            item[f"{key}_max"] = float(np.max(values))
    return summary


def write_config(path: Path, args: argparse.Namespace) -> None:
    payload = vars(args).copy()
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["official_stl_method"] = args.stl_method
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "sample_set",
        "method",
        "n",
        "case",
        "stem",
        "ok",
        "stl_method",
        "point_pred_to_gt_rms",
        "point_gt_to_pred_rms",
        "point_symmetric_rms",
        "stl_pred_to_gt_rms",
        "stl_gt_to_pred_rms",
        "stl_symmetric_rms",
        "triangles",
        "surface_area",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


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
