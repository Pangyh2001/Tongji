from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crown_m0.dataset import discover_cases
from src.crown_m0.template_mesh import write_template_mesh_stl
from src.crown_m0.template_selection import case_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dynamic crown template library from train GT STL files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("templates/m0_dynamic_library_4096"))
    parser.add_argument("--source-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--target-triangles", type=int, default=8192)
    parser.add_argument(
        "--expected-vertices",
        type=int,
        default=4098,
        help="Keep only simplified templates with this vertex count; use 0 to accept the first mesh count.",
    )
    parser.add_argument("--max-templates-per-group", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_cases(args.data_dir)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    split_cases = set(split[args.source_split])
    candidates = [record for record in records if str(record.train_dir.parent) in split_cases]
    if not candidates:
        raise SystemExit(f"No {args.source_split} cases found in {args.split_file}")

    grouped = defaultdict(list)
    for record in candidates:
        grouped[(record.tooth_id, record.prep_arch)].append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = args.output_dir / "cases"
    templates_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    expected_vertices = args.expected_vertices if args.expected_vertices > 0 else None

    for key, records_in_group in sorted(grouped.items()):
        selected = select_representative_records(records_in_group, args.max_templates_per_group)
        for rank, record in enumerate(selected):
            mesh_path = find_gt_stl(record.train_dir.parent)
            if mesh_path is None:
                continue
            mesh = read_and_simplify(mesh_path, args.target_triangles)
            center = load_case_center(record.train_dir.parent)
            vertices = np.asarray(mesh.vertices, dtype=np.float32) - center[None, :]
            faces = np.asarray(mesh.triangles, dtype=np.int64)
            if expected_vertices is None:
                expected_vertices = vertices.shape[0]
            if vertices.shape[0] != expected_vertices:
                print(
                    f"[skip] {record.train_dir.parent}: vertex count {vertices.shape[0]} != {expected_vertices}",
                    flush=True,
                )
                continue

            stem = safe_case_name(record.train_dir.parent, args.data_dir)
            npz_rel = Path("cases") / f"{stem}.npz"
            stl_rel = Path("cases") / f"{stem}.stl"
            np.savez_compressed(
                args.output_dir / npz_rel,
                vertices=vertices,
                faces=faces,
                source_case=str(record.train_dir.parent),
                source_stl=str(mesh_path),
                center_xyz_mm=center,
                coordinate_frame="margin-centered local millimeters",
            )
            write_template_mesh_stl(args.output_dir / stl_rel, vertices, faces)
            index_rows.append(
                {
                    "template": str(npz_rel),
                    "preview_stl": str(stl_rel),
                    "source_case": str(record.train_dir.parent),
                    "source_stl": str(mesh_path),
                    "tooth_id": record.tooth_id,
                    "prep_arch": record.prep_arch,
                    "rank_in_group": rank,
                    "vertices": int(vertices.shape[0]),
                    "faces": int(faces.shape[0]),
                    "feature": case_feature(record.train_dir.parent).astype(float).tolist(),
                }
            )

    payload = {
        "method": "dynamic_template_library",
        "source_split": args.source_split,
        "target_triangles": args.target_triangles,
        "max_templates_per_group": args.max_templates_per_group,
        "templates": index_rows,
    }
    (args.output_dir / "template_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "templates"} | {"n": len(index_rows)}, ensure_ascii=False, indent=2))


def select_representative_records(records, max_count: int):
    if len(records) <= max_count:
        return sorted(records, key=lambda record: str(record.train_dir.parent))
    scored = [(case_feature(record.train_dir.parent), record) for record in records]
    features = np.stack([item[0] for item in scored], axis=0)
    center = np.median(features, axis=0)
    dist = np.linalg.norm(features - center[None, :], axis=1)
    order = np.argsort(dist)
    picked = np.linspace(0, len(order) - 1, max_count, dtype=int)
    return [scored[int(order[i])][1] for i in picked]


def read_and_simplify(path: Path, target_triangles: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"empty mesh: {path}")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def find_gt_stl(case_path: Path) -> Path | None:
    crowns = sorted((case_path / "processed").glob("crown_*.stl"))
    return crowns[0] if crowns else None


def load_case_center(case_path: Path) -> np.ndarray:
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    return np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float32)


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.relative_to(data_dir)
        return "__".join(rel.parts)
    except ValueError:
        return case_path.name


if __name__ == "__main__":
    main()
