from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crown_m0.dataset import discover_cases
from src.crown_m0.template_mesh import write_template_mesh_stl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed-topology crown template mesh.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--output", type=Path, default=Path("templates/m0_global_template_4096.npz"))
    parser.add_argument("--preview-stl", type=Path, default=Path("templates/m0_global_template_4096.stl"))
    parser.add_argument("--source-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--target-triangles", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_cases(args.data_dir)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    split_cases = set(split[args.source_split])
    candidates = [record for record in records if str(record.train_dir.parent) in split_cases]
    if not candidates:
        raise SystemExit(f"No {args.source_split} cases found in {args.split_file}")

    record = choose_median_vertex_case(candidates)
    mesh_path = find_gt_stl(record.train_dir.parent)
    if mesh_path is None:
        raise SystemExit(f"Missing GT crown STL for template case {record.train_dir.parent}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise SystemExit(f"Cannot read template mesh: {mesh_path}")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    center = load_case_center(record.train_dir.parent)
    vertices = np.asarray(mesh.vertices, dtype=np.float32) - center[None, :]
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vertices=vertices,
        faces=faces,
        source_case=str(record.train_dir.parent),
        source_stl=str(mesh_path),
        center_xyz_mm=center,
        coordinate_frame="margin-centered local millimeters",
    )
    write_template_mesh_stl(args.preview_stl, vertices, faces)
    print(json.dumps({
        "template": str(args.output),
        "preview_stl": str(args.preview_stl),
        "source_case": str(record.train_dir.parent),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
    }, ensure_ascii=False, indent=2))


def choose_median_vertex_case(records):
    counts = []
    for record in records:
        mesh_path = find_gt_stl(record.train_dir.parent)
        if mesh_path is None:
            continue
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.is_empty():
            counts.append((len(mesh.vertices), record))
    if not counts:
        raise SystemExit("No readable GT STL found for template construction")
    counts.sort(key=lambda item: item[0])
    return counts[len(counts) // 2][1]


def find_gt_stl(case_path: Path) -> Path | None:
    crowns = sorted((case_path / "processed").glob("crown_*.stl"))
    return crowns[0] if crowns else None


def load_case_center(case_path: Path) -> np.ndarray:
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    return np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float32)


if __name__ == "__main__":
    main()
