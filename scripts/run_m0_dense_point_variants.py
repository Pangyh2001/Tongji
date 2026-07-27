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
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crown_m0.dataset import CrownDataset, discover_cases
from src.crown_m0.io import write_ply, write_xyz
from src.crown_m0.model import M0CrownNet


VARIANTS = (
    "raw16k_alpha",
    "dense32k_pca_alpha",
    "dense64k_pca_alpha",
)


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Evaluate denser point-cloud STL reconstruction variants for M0.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/m0_new/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--output-root", type=Path, default=Path("result"))
    parser.add_argument("--date", default=today)
    parser.add_argument("--experiment-name", default="m0_dense_point_variants")
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    rng = np.random.default_rng(args.seed)

    output_dir = args.output_root / args.date / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    records = discover_cases(args.data_dir)
    by_case = {str(record.train_dir.parent): record for record in records}
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    cases = split["test"][: args.max_cases if args.max_cases > 0 else None]
    selected_records = [by_case[case] for case in cases if case in by_case]

    rows = run_sample_set(model, selected_records, output_dir / "test", args, rng)
    write_csv(output_dir / "test" / "metrics_by_case.csv", rows)
    summaries = summarize_rows(rows)
    write_csv(output_dir / "test" / "summary_metrics.csv", list(summaries.values()))
    (output_dir / "test" / "summary_metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "variants": list(VARIANTS),
                "max_cases": args.max_cases,
                "sample_points": args.sample_points,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}", flush=True)


def load_model(args: argparse.Namespace) -> M0CrownNet:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = M0CrownNet().to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def run_sample_set(model: M0CrownNet, records, sample_dir: Path, args: argparse.Namespace, rng) -> list[dict]:
    loader = DataLoader(CrownDataset(records), batch_size=1, shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            pred = model(
                batch["prep"].to(args.device),
                batch["antagonist"].to(args.device),
                batch["tooth_index"].to(args.device),
                batch["prep_arch_index"].to(args.device),
            ).cpu().numpy()[0].astype(np.float32)
            gt = batch["crown"].cpu().numpy()[0].astype(np.float32)
            rows.extend(evaluate_case(Path(batch["case_id"][0]), pred, gt, sample_dir, args, rng))
    return rows


def evaluate_case(case_path: Path, pred_local: np.ndarray, gt_local: np.ndarray, sample_dir: Path, args, rng) -> list[dict]:
    stem = safe_case_name(case_path, args.data_dir)
    case_dir = sample_dir / "cases" / stem
    case_dir.mkdir(parents=True, exist_ok=True)
    gt_stl = find_gt_stl(case_path)
    if gt_stl is None:
        raise FileNotFoundError(f"missing GT STL for {case_path}")
    gt_stl_copy = case_dir / f"{stem}_GT_technician.stl"
    if not gt_stl_copy.exists():
        shutil.copy2(gt_stl, gt_stl_copy)
    gt_mesh = read_mesh(gt_stl)
    gt_mesh_points = sample_mesh_points(gt_mesh, args.sample_points)
    gt_xyz = gt_local[:, :3]

    raw_xyz = pred_local[:, :3].astype(np.float64)
    variants = {
        "raw16k_alpha": raw_xyz,
        "dense32k_pca_alpha": upsample_pca(raw_xyz, target_points=32768, rng=rng),
        "dense64k_pca_alpha": upsample_pca(raw_xyz, target_points=65536, rng=rng),
    }
    rows = []
    for method, xyz_local in variants.items():
        pred_npy = case_dir / f"{stem}_{method}.npy"
        pred_xyz = case_dir / f"{stem}_{method}.xyz"
        pred_ply = case_dir / f"{stem}_{method}.ply"
        pred_stl = case_dir / f"{stem}_pred_{method}.stl"
        np.save(pred_npy, xyz_local.astype(np.float32))
        write_xyz(pred_xyz, xyz_local)
        write_ply(pred_ply, np.pad(xyz_local, ((0, 0), (0, 3)), constant_values=0.0))

        xyz_original = restore_original_coordinates(xyz_local, case_path)
        pred_mesh = reconstruct_alpha_clean(xyz_original, alpha=1.0 if "dense" in method else 1.2)
        ok = o3d.io.write_triangle_mesh(str(pred_stl), pred_mesh, write_ascii=False)
        if not ok:
            raise RuntimeError(f"failed to write {pred_stl}")

        point_metric = point_metrics(xyz_local, gt_xyz)
        stl_metric = point_metrics(sample_mesh_points(pred_mesh, args.sample_points), gt_mesh_points)
        row = {
            "case": str(case_path),
            "stem": stem,
            "method": method,
            "ok": True,
            "points": int(len(xyz_local)),
            "pred_npy": str(pred_npy),
            "pred_ply": str(pred_ply),
            "pred_stl": str(pred_stl),
            "gt_stl": str(gt_stl_copy),
        }
        row.update(prefix_metrics("point", point_metric))
        row.update(prefix_metrics("stl", stl_metric))
        row.update(mesh_stats(pred_mesh))
        rows.append(row)
    return rows


def upsample_pca(xyz: np.ndarray, target_points: int, rng) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) >= target_points:
        return xyz[:target_points].copy()
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz, k=12)
    local_scale = np.clip(np.median(distances[:, 1:6], axis=1), 0.03, 0.18)
    normals = estimate_normals_from_neighbors(xyz, indices[:, 1:])
    n_extra = target_points - len(xyz)
    source = rng.integers(0, len(xyz), size=n_extra)
    offsets = np.zeros((n_extra, 3), dtype=np.float64)
    for out_idx, point_idx in enumerate(source):
        normal = normals[point_idx]
        basis_u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(basis_u) < 1e-6:
            basis_u = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        basis_u /= np.linalg.norm(basis_u) + 1e-8
        basis_v = np.cross(normal, basis_u)
        radius = local_scale[point_idx] * 0.45
        a, b = rng.normal(size=2)
        offsets[out_idx] = radius * (a * basis_u + b * basis_v)
    dense = np.concatenate([xyz, xyz[source] + offsets], axis=0)
    return voxel_unique(dense, target_points=target_points, voxel=0.015, rng=rng)


def estimate_normals_from_neighbors(xyz: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(xyz)
    center = xyz.mean(axis=0)
    for idx, neighbors in enumerate(neighbor_indices):
        pts = xyz[neighbors]
        cov = np.cov((pts - pts.mean(axis=0)).T)
        _, vecs = np.linalg.eigh(cov)
        normal = vecs[:, 0]
        if np.dot(normal, xyz[idx] - center) < 0:
            normal = -normal
        normals[idx] = normal / (np.linalg.norm(normal) + 1e-8)
    return normals


def voxel_unique(xyz: np.ndarray, target_points: int, voxel: float, rng) -> np.ndarray:
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    out = xyz[np.sort(keep)]
    if len(out) >= target_points:
        choice = rng.choice(len(out), size=target_points, replace=False)
        return out[np.sort(choice)]
    if len(out) < target_points:
        extra = xyz[rng.choice(len(xyz), size=target_points - len(out), replace=True)]
        out = np.concatenate([out, extra], axis=0)
    return out


def reconstruct_alpha_clean(xyz: np.ndarray, alpha: float) -> o3d.geometry.TriangleMesh:
    pcd = make_pcd(xyz)
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=32, std_ratio=2.0)
    clean, _ = clean.remove_radius_outlier(nb_points=10, radius=0.45)
    if len(clean.points) < 1000:
        clean = pcd
    clean = clean.voxel_down_sample(voxel_size=0.04)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(clean, alpha)
    mesh = clean_mesh(mesh)
    mesh = mesh.subdivide_midpoint(number_of_iterations=1)
    mesh = clean_mesh(mesh)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=20, lambda_filter=0.5, mu=-0.53)
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise ValueError("empty alpha mesh")
    return mesh


def reconstruct_poisson(xyz: np.ndarray) -> o3d.geometry.TriangleMesh:
    pcd = make_pcd(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=0.035)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.45, max_nn=40))
    try:
        pcd.orient_normals_consistent_tangent_plane(30)
    except RuntimeError:
        pass
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=1.05)
    densities_np = np.asarray(densities)
    if densities_np.size:
        cutoff = np.percentile(densities_np, 5)
        mesh.remove_vertices_by_mask(densities_np < cutoff)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.08, bbox.get_center())
    mesh = mesh.crop(bbox)
    mesh = clean_mesh(mesh)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=15, lambda_filter=0.5, mu=-0.53)
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise ValueError("empty poisson mesh")
    return mesh


def make_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return pcd


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_np = np.asarray(labels)
        counts_np = np.asarray(counts)
        if counts_np.size > 1:
            mesh.remove_triangles_by_mask(labels_np != int(np.argmax(counts_np)))
            mesh.remove_unreferenced_vertices()
    return mesh


def restore_original_coordinates(xyz: np.ndarray, case_path: Path) -> np.ndarray:
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float64)
    return xyz.astype(np.float64) + center[None, :]


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
    return np.asarray(pcd.points, dtype=np.float64)


def point_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    pred_to_gt = cKDTree(gt_xyz).query(pred_xyz, k=1)[0]
    gt_to_pred = cKDTree(pred_xyz).query(gt_xyz, k=1)[0]
    both = np.concatenate([pred_to_gt, gt_to_pred])
    return {
        "pred_to_gt_rms": rms(pred_to_gt),
        "gt_to_pred_rms": rms(gt_to_pred),
        "symmetric_rms": rms(both),
        "pred_to_gt_hd95": percentile(pred_to_gt, 95),
        "gt_to_pred_hd95": percentile(gt_to_pred, 95),
    }


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts_np = np.asarray(counts)
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "components": int(len(counts_np)),
        "surface_area": float(mesh.get_surface_area()),
        "edge_manifold": bool(mesh.is_edge_manifold()),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
    }


def summarize_rows(rows: list[dict]) -> dict[str, dict]:
    metric_keys = [
        "points",
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
        "vertices",
        "triangles",
        "components",
        "surface_area",
    ]
    summaries = {}
    for method in VARIANTS:
        method_rows = [row for row in rows if row.get("method") == method and row.get("ok")]
        item = {"method": method, "n": len(method_rows)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in method_rows if row.get(key) not in ("", None)], dtype=float)
            if values.size:
                item[f"{key}_mean"] = float(np.mean(values))
                item[f"{key}_median"] = float(np.median(values))
                item[f"{key}_max"] = float(np.max(values))
        summaries[method] = item
    return summaries


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    priority = ["case", "stem", "method", "ok", "points", "point_symmetric_rms", "stl_symmetric_rms"]
    ordered = [key for key in priority if key in fieldnames] + [key for key in fieldnames if key not in priority]
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


def rms(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(values**2))))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


if __name__ == "__main__":
    main()
