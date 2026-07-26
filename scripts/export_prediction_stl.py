from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert predicted crown point clouds to STL meshes.")
    parser.add_argument("--input-dir", type=Path, default=Path("predictions/m0_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/m0_stl"))
    parser.add_argument(
        "--method",
        choices=["alpha_clean", "alpha_raw", "poisson"],
        default="alpha_clean",
        help="Official experiment default is alpha_clean.",
    )
    parser.add_argument("--alpha", type=float, default=1.2, help="Alpha radius for alpha-shape reconstruction.")
    parser.add_argument("--depth", type=int, default=7, help="Poisson reconstruction depth.")
    parser.add_argument("--density-quantile", type=float, default=0.02, help="Remove lowest-density mesh vertices.")
    parser.add_argument("--single", type=Path, default=None, help="Optional single .npy prediction to convert.")
    return parser.parse_args()


def prediction_to_stl(
    npy_path: Path,
    output_path: Path,
    *,
    method: str,
    alpha: float,
    depth: int,
    density_quantile: float,
) -> None:
    arr = np.load(npy_path).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected (N, >=3) prediction array, got {arr.shape}: {npy_path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3])

    if method == "alpha_clean":
        clean, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
        clean, _ = clean.remove_radius_outlier(nb_points=8, radius=0.55)
        if len(clean.points) < 1000:
            clean = pcd
        clean = clean.voxel_down_sample(voxel_size=0.06)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(clean, alpha)
    elif method == "alpha_raw":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        if arr.shape[1] >= 6 and np.isfinite(arr[:, 3:6]).all():
            normals = arr[:, 3:6]
            norm = np.linalg.norm(normals, axis=1, keepdims=True)
            valid = norm[:, 0] > 1e-8
            normals = np.where(valid[:, None], normals / np.maximum(norm, 1e-8), 0.0)
            pcd.normals = o3d.utility.Vector3dVector(normals)
        else:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        densities_np = np.asarray(densities)
        if 0.0 < density_quantile < 0.5 and densities_np.size:
            threshold = np.quantile(densities_np, density_quantile)
            mesh.remove_vertices_by_mask(densities_np < threshold)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write STL: {output_path}")


def main() -> None:
    args = parse_args()
    files = [args.single] if args.single else sorted(args.input_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy predictions found in {args.input_dir}")

    for npy_path in tqdm(files, desc="stl"):
        output_path = args.output_dir / f"{npy_path.stem}.stl"
        prediction_to_stl(
            npy_path,
            output_path,
            method=args.method,
            alpha=args.alpha,
            depth=args.depth,
            density_quantile=args.density_quantile,
        )


if __name__ == "__main__":
    main()
