from __future__ import annotations

import argparse
import csv
import json
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
from src.crown_m0.model import M0TemplateDeformNet
from src.crown_m0.template_mesh import load_template_npz, mesh_edges_from_faces, write_template_mesh_stl
from src.crown_m0.template_selection import case_feature, load_template_index, select_template


VARIANTS = (
    "raw",
    "raw_taubin30",
    "delta_clip_2mm_taubin30",
    "delta_clip_2mm_smooth",
    "neighbor_despike_smooth",
    "neighbor_despike_taubin30",
    "template_blend_075_smooth",
)


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Evaluate M0 dynamic-template STL post-processing variants.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/m0_template/best.pt"))
    parser.add_argument("--template-index", type=Path, default=Path("templates/m0_dynamic_library_4096/template_index.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--output-root", type=Path, default=Path("result"))
    parser.add_argument("--date", default=today)
    parser.add_argument("--experiment-name", default="m0_template_stl_variants")
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    output_dir = args.output_root / args.date / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    template_entries = load_template_index(args.template_index)
    first_vertices, _ = load_template_npz(template_entries[0].template)
    model = load_model(args, first_vertices.shape[0])

    records = discover_cases(args.data_dir)
    by_case = {str(record.train_dir.parent): record for record in records}
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    selected_records = [by_case[case] for case in split["test"] if case in by_case]

    rows = run_sample_set(model, selected_records, template_entries, output_dir / "test", args)
    write_csv(output_dir / "test" / "metrics_by_case.csv", rows)
    summaries = summarize_rows(rows)
    write_csv(output_dir / "test" / "summary_metrics.csv", list(summaries.values()))
    (output_dir / "test" / "summary_metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "summary_by_method.csv", list(summaries.values()))
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "template_index": str(args.template_index),
                "variants": list(VARIANTS),
                "sample_points": args.sample_points,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}", flush=True)


def load_model(args: argparse.Namespace, template_vertices: int) -> M0TemplateDeformNet:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = M0TemplateDeformNet(
        template_vertices=template_vertices,
        max_displacement_mm=float(checkpoint_args.get("max_displacement_mm", 4.0)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def run_sample_set(model, records, template_entries, sample_dir: Path, args: argparse.Namespace) -> list[dict]:
    loader = DataLoader(CrownDataset(records), batch_size=1, shuffle=False, num_workers=0)
    rows = []
    template_cache = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            case_path = Path(batch["case_id"][0])
            record = next(record for record in records if str(record.train_dir.parent) == str(case_path))
            entry = select_template(
                template_entries,
                tooth_id=record.tooth_id,
                prep_arch=record.prep_arch,
                feature=case_feature(case_path),
            )
            template_vertices, faces = load_eval_template(entry, template_cache, args.device)
            pred_vertices, _ = model(
                batch["prep"].to(args.device),
                batch["antagonist"].to(args.device),
                batch["tooth_index"].to(args.device),
                batch["prep_arch_index"].to(args.device),
                template_vertices,
            )
            pred_local = pred_vertices[0].detach().cpu().numpy()
            template_local = template_vertices.detach().cpu().numpy()
            rows.extend(evaluate_variants(case_path, pred_local, template_local, faces, entry, sample_dir, args))
    return rows


def load_eval_template(entry, cache, device):
    cached = cache.get(entry.template)
    if cached is None:
        vertices_np, faces_np = load_template_npz(entry.template)
        cached = (torch.from_numpy(vertices_np).to(device), faces_np)
        cache[entry.template] = cached
    return cached


def evaluate_variants(case_path: Path, pred_local: np.ndarray, template_local: np.ndarray, faces: np.ndarray, entry, sample_dir: Path, args) -> list[dict]:
    stem = safe_case_name(case_path, args.data_dir)
    case_dir = sample_dir / "cases" / stem
    case_dir.mkdir(parents=True, exist_ok=True)

    gt_stl = find_gt_stl(case_path)
    if gt_stl is None:
        raise FileNotFoundError(f"missing GT crown STL in {case_path / 'processed'}")
    gt_stl_copy = case_dir / f"{stem}_GT_technician.stl"
    template_stl_copy = case_dir / f"{stem}_selected_template.stl"
    if not gt_stl_copy.exists():
        shutil.copy2(gt_stl, gt_stl_copy)
    if not template_stl_copy.exists():
        shutil.copy2(entry.preview_stl, template_stl_copy)

    gt_mesh = read_mesh(gt_stl)
    gt_points = sample_mesh_points(gt_mesh, args.sample_points)
    gt_local = np.load(case_path / "train" / "crown_points.npy").astype(np.float64)[:, :3]
    adjacency = build_adjacency(len(template_local), faces)

    rows = []
    for method, vertices_local in make_variants(pred_local, template_local, faces, adjacency).items():
        pred_stl = case_dir / f"{stem}_pred_{method}.stl"
        pred_original = restore_original_coordinates(vertices_local, case_path)
        write_template_mesh_stl(pred_stl, pred_original, faces)
        pred_mesh = read_mesh(pred_stl)
        stl_metrics = point_metrics(sample_mesh_points(pred_mesh, args.sample_points), gt_points)
        vertex_metrics = point_metrics(vertices_local, gt_local)
        row = {
            "case": str(case_path),
            "stem": stem,
            "method": method,
            "ok": True,
            "pred_stl": str(pred_stl),
            "gt_stl": str(gt_stl_copy),
            "selected_template_stl": str(template_stl_copy),
            "selected_template_case": entry.source_case,
            "selected_template_tooth_id": entry.tooth_id,
            "selected_template_prep_arch": entry.prep_arch,
        }
        row.update(prefix_metrics("stl", stl_metrics))
        row.update(prefix_metrics("vertex", vertex_metrics))
        row.update(mesh_stats(pred_mesh))
        row.update(spike_stats(vertices_local, template_local, faces))
        rows.append(row)
    return rows


def make_variants(pred: np.ndarray, template: np.ndarray, faces: np.ndarray, adjacency: list[np.ndarray]) -> dict[str, np.ndarray]:
    delta = pred - template
    clipped_only = template + clip_displacement(delta, max_norm=2.0)
    clipped = template + clip_displacement(delta, max_norm=2.0)
    clipped = smooth_vertices(clipped, adjacency, iterations=8, lam=0.35)

    despiked_only = despike_vertices(pred, adjacency, iterations=5)
    despiked = smooth_vertices(despiked_only, adjacency, iterations=8, lam=0.30)

    blended = template + 0.75 * delta
    blended = despike_vertices(blended, adjacency, iterations=4)
    blended = smooth_vertices(blended, adjacency, iterations=10, lam=0.35)

    return {
        "raw": pred.astype(np.float32),
        "raw_taubin30": taubin_vertices(pred, faces, iterations=30).astype(np.float32),
        "delta_clip_2mm_taubin30": taubin_vertices(clipped_only, faces, iterations=30).astype(np.float32),
        "delta_clip_2mm_smooth": clipped.astype(np.float32),
        "neighbor_despike_smooth": despiked.astype(np.float32),
        "neighbor_despike_taubin30": taubin_vertices(despiked_only, faces, iterations=30).astype(np.float32),
        "template_blend_075_smooth": blended.astype(np.float32),
    }


def clip_displacement(delta: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1e-8))
    return delta * scale


def despike_vertices(vertices: np.ndarray, adjacency: list[np.ndarray], iterations: int) -> np.ndarray:
    out = vertices.astype(np.float64).copy()
    for _ in range(iterations):
        neighbor_mean = neighbor_average(out, adjacency)
        deviation = np.linalg.norm(out - neighbor_mean, axis=1)
        median = float(np.median(deviation))
        mad = float(np.median(np.abs(deviation - median))) + 1e-6
        threshold = max(0.30, median + 4.0 * mad)
        mask = deviation > threshold
        out[mask] = 0.25 * out[mask] + 0.75 * neighbor_mean[mask]
    return out


def smooth_vertices(vertices: np.ndarray, adjacency: list[np.ndarray], iterations: int, lam: float) -> np.ndarray:
    out = vertices.astype(np.float64).copy()
    for _ in range(iterations):
        out = (1.0 - lam) * out + lam * neighbor_average(out, adjacency)
    return out


def taubin_vertices(vertices: np.ndarray, faces: np.ndarray, iterations: int) -> np.ndarray:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    smoothed = mesh.filter_smooth_taubin(
        number_of_iterations=iterations,
        lambda_filter=0.5,
        mu=-0.53,
    )
    return np.asarray(smoothed.vertices, dtype=np.float64)


def neighbor_average(vertices: np.ndarray, adjacency: list[np.ndarray]) -> np.ndarray:
    avg = vertices.copy()
    for idx, neighbors in enumerate(adjacency):
        if neighbors.size:
            avg[idx] = vertices[neighbors].mean(axis=0)
    return avg


def build_adjacency(n_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    neighbors = [set() for _ in range(n_vertices)]
    for a, b in mesh_edges_from_faces(faces):
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))
    return [np.asarray(sorted(item), dtype=np.int64) for item in neighbors]


def spike_stats(vertices: np.ndarray, template: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    edges = mesh_edges_from_faces(faces)
    if edges.size == 0:
        return {}
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    template_edge_lengths = np.linalg.norm(template[edges[:, 0]] - template[edges[:, 1]], axis=1)
    displacement = np.linalg.norm(vertices - template, axis=1)
    return {
        "edge_length_p95": float(np.percentile(edge_lengths, 95)),
        "edge_length_max": float(np.max(edge_lengths)),
        "edge_stretch_p95": float(np.percentile(edge_lengths / np.maximum(template_edge_lengths, 1e-6), 95)),
        "edge_stretch_max": float(np.max(edge_lengths / np.maximum(template_edge_lengths, 1e-6))),
        "displacement_p95": float(np.percentile(displacement, 95)),
        "displacement_max": float(np.max(displacement)),
    }


def restore_original_coordinates(xyz: np.ndarray, case_path: Path) -> np.ndarray:
    meta = json.loads((case_path / "train" / "metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float64)
    return xyz.astype(np.float64) + center[None, :]


def find_gt_stl(case_path: Path) -> Path | None:
    crowns = sorted((case_path / "processed").glob("crown_*.stl"))
    return crowns[0] if crowns else None


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"empty mesh: {path}")
    return mesh


def sample_mesh_points(mesh: o3d.geometry.TriangleMesh, count: int) -> np.ndarray:
    pcd = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(pcd.points, dtype=np.float64)


def point_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    pred_to_gt = cKDTree(pred_xyz).query(gt_xyz, k=1)[0]
    gt_to_pred = cKDTree(gt_xyz).query(pred_xyz, k=1)[0]
    pred_rms = float(np.sqrt(np.mean(pred_to_gt**2)))
    gt_rms = float(np.sqrt(np.mean(gt_to_pred**2)))
    return {
        "pred_to_gt_rms": pred_rms,
        "gt_to_pred_rms": gt_rms,
        "symmetric_rms": float((pred_rms + gt_rms) / 2.0),
        "pred_to_gt_hd95": float(np.percentile(pred_to_gt, 95)),
        "gt_to_pred_hd95": float(np.percentile(gt_to_pred, 95)),
    }


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict[str, float]:
    return {
        "vertices": float(len(mesh.vertices)),
        "triangles": float(len(mesh.triangles)),
        "surface_area": float(mesh.get_surface_area()),
    }


def summarize_rows(rows: list[dict]) -> dict[str, dict]:
    metric_keys = [
        "vertex_pred_to_gt_rms",
        "vertex_gt_to_pred_rms",
        "vertex_symmetric_rms",
        "vertex_pred_to_gt_hd95",
        "vertex_gt_to_pred_hd95",
        "stl_pred_to_gt_rms",
        "stl_gt_to_pred_rms",
        "stl_symmetric_rms",
        "stl_pred_to_gt_hd95",
        "stl_gt_to_pred_hd95",
        "vertices",
        "triangles",
        "surface_area",
        "edge_length_p95",
        "edge_length_max",
        "edge_stretch_p95",
        "edge_stretch_max",
        "displacement_p95",
        "displacement_max",
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    priority = ["case", "stem", "method", "ok", "error"]
    ordered = [key for key in priority if key in fieldnames] + [key for key in fieldnames if key not in priority]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.relative_to(data_dir)
        return "__".join(rel.parts)
    except ValueError:
        return case_path.name


if __name__ == "__main__":
    main()
