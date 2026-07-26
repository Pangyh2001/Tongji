from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


METHODS = ("m1a_rigid_pca", "m1b_estimated_bbox_scale", "m1c_oracle_bbox_scale")


@dataclass(frozen=True)
class CaseInfo:
    case: str
    case_path: Path
    stem: str
    tooth_id: str
    prep_arch: str
    center: np.ndarray
    prep_xyz: np.ndarray
    antagonist_xyz: np.ndarray
    crown_xyz: np.ndarray
    margin_xyz: np.ndarray
    crown_stl: Path
    prep_span: np.ndarray
    antagonist_span: np.ndarray
    crown_span: np.ndarray
    margin_span: np.ndarray
    prep_frame: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval/template STL baselines for crown generation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--case-list", type=Path, default=Path("splits/m0_stl_representative_10.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/m1_retrieval_representative10"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train_cases = split["train"]
    eval_cases = [line.strip() for line in args.case_list.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_infos = [load_case(Path(case), args.data_dir) for case in train_cases]
    eval_infos = [load_case(Path(case), args.data_dir) for case in eval_cases]
    tooth_ratios = build_tooth_ratios(train_infos, args.eps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gt_stl").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "template_stl").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index, target in enumerate(eval_infos, start=1):
        print(f"[{index}/{len(eval_infos)}] {target.stem}", flush=True)
        candidates = [
            info
            for info in train_infos
            if info.tooth_id == target.tooth_id and info.prep_arch == target.prep_arch
        ]
        if not candidates:
            candidates = [info for info in train_infos if info.tooth_id == target.tooth_id]
        if not candidates:
            rows.append({"case": target.case, "stem": target.stem, "ok": False, "error": "no same-tooth candidates"})
            continue
        template = min(candidates, key=lambda info: retrieval_score(target, info, tooth_ratios, args.eps))

        gt_copy = args.output_dir / "gt_stl" / f"{target.stem}_GT_technician.stl"
        if not gt_copy.exists():
            shutil.copy2(target.crown_stl, gt_copy)
        template_copy = args.output_dir / "template_stl" / f"{target.stem}_TEMPLATE_from_{template.stem}.stl"
        if not template_copy.exists():
            shutil.copy2(template.crown_stl, template_copy)

        gt_mesh = read_mesh(target.crown_stl)
        gt_sample = sample_mesh_points(gt_mesh, args.sample_points)

        for method in args.methods:
            row = {
                "case": target.case,
                "stem": target.stem,
                "method": method,
                "ok": False,
                "template_case": template.case,
                "template_stl": str(template_copy),
                "gt_stl": str(gt_copy),
                "error": "",
            }
            try:
                pred_mesh = transform_template_mesh(template, target, method, tooth_ratios, args.eps)
                pred_mesh = cleanup_mesh(pred_mesh)
                out_case = args.output_dir / method / target.stem
                out_case.mkdir(parents=True, exist_ok=True)
                out_stl = out_case / f"{target.stem}_{method}.stl"
                ok = o3d.io.write_triangle_mesh(str(out_stl), pred_mesh, write_ascii=False)
                if not ok:
                    raise RuntimeError(f"failed to write {out_stl}")

                pred_sample = sample_mesh_points(pred_mesh, args.sample_points)
                row.update(surface_metrics(pred_sample, gt_sample))
                row.update(mesh_stats(pred_mesh))
                row.update({"ok": True, "stl": str(out_stl)})
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_outputs(args.output_dir, rows)

    write_outputs(args.output_dir, rows)
    print(f"wrote {args.output_dir}", flush=True)


def load_case(case_path: Path, data_dir: Path) -> CaseInfo:
    meta_path = case_path / "train" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tooth_id = str(meta.get("tooth_id") or case_path.name.rsplit("_", 1)[-1])
    prep_arch = str(meta.get("role_assignment", {}).get("prep_arch", "unknown")).lower()
    center = np.asarray(meta["coordinate_processing"]["center_xyz_mm"], dtype=np.float64)

    prep = np.load(case_path / "train" / "prep_points.npy").astype(np.float64)[:, :3]
    antagonist = np.load(case_path / "train" / "antagonist_points.npy").astype(np.float64)[:, :3]
    crown = np.load(case_path / "train" / "crown_points.npy").astype(np.float64)[:, :3]
    margin_path = case_path / "train" / "margin_points.npy"
    margin = np.load(margin_path).astype(np.float64)[:, :3] if margin_path.exists() else np.zeros((1, 3))
    crown_stl = find_gt_stl(case_path)
    if crown_stl is None:
        raise FileNotFoundError(f"missing crown STL for {case_path}")

    return CaseInfo(
        case=str(case_path),
        case_path=case_path,
        stem=safe_case_name(case_path, data_dir),
        tooth_id=tooth_id,
        prep_arch=prep_arch,
        center=center,
        prep_xyz=prep,
        antagonist_xyz=antagonist,
        crown_xyz=crown,
        margin_xyz=margin,
        crown_stl=crown_stl,
        prep_span=span(prep),
        antagonist_span=span(antagonist),
        crown_span=span(crown),
        margin_span=span(margin),
        prep_frame=pca_frame(prep),
    )


def build_tooth_ratios(infos: list[CaseInfo], eps: float) -> dict[tuple[str, str], np.ndarray]:
    by_key: dict[tuple[str, str], list[np.ndarray]] = {}
    for info in infos:
        ratio = info.crown_span / np.maximum(info.prep_span, eps)
        by_key.setdefault((info.tooth_id, info.prep_arch), []).append(ratio)
    return {key: np.median(np.vstack(values), axis=0) for key, values in by_key.items()}


def retrieval_score(target: CaseInfo, candidate: CaseInfo, ratios: dict[tuple[str, str], np.ndarray], eps: float) -> float:
    key = (target.tooth_id, target.prep_arch)
    target_est_crown_span = target.prep_span * ratios.get(key, np.ones(3))
    parts = [
        normalized_l2(target.prep_span, candidate.prep_span, eps),
        0.5 * normalized_l2(target.antagonist_span, candidate.antagonist_span, eps),
        0.5 * normalized_l2(target.margin_span, candidate.margin_span, eps),
        normalized_l2(target_est_crown_span, candidate.crown_span, eps),
    ]
    return float(sum(parts))


def transform_template_mesh(
    template: CaseInfo,
    target: CaseInfo,
    method: str,
    ratios: dict[tuple[str, str], np.ndarray],
    eps: float,
) -> o3d.geometry.TriangleMesh:
    mesh = read_mesh(template.crown_stl)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) - template.center[None, :]
    vertices = local_frame_transform(vertices, template.prep_frame, target.prep_frame)

    if method == "m1a_rigid_pca":
        scaled = vertices
    elif method == "m1b_estimated_bbox_scale":
        key = (target.tooth_id, target.prep_arch)
        target_span = target.prep_span * ratios.get(key, np.ones(3))
        scaled = scale_to_span(vertices, target_span, eps)
    elif method == "m1c_oracle_bbox_scale":
        scaled = scale_to_span(vertices, target.crown_span, eps)
    else:
        raise ValueError(method)

    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(scaled + target.center[None, :])
    return out


def local_frame_transform(points: np.ndarray, source_frame: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
    source_local = points @ source_frame
    return source_local @ target_frame.T


def scale_to_span(points: np.ndarray, target_span: np.ndarray, eps: float) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    current_span = span(points)
    scale = target_span / np.maximum(current_span, eps)
    scale = np.clip(scale, 0.65, 1.45)
    return (points - center) * scale[None, :] + center


def pca_frame(xyz: np.ndarray) -> np.ndarray:
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    frame = vecs[:, ::-1]
    for axis in range(3):
        if frame[axis, axis] < 0:
            frame[:, axis] *= -1
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return frame


def find_gt_stl(case_path: Path) -> Path | None:
    crowns = sorted((case_path / "processed").glob("crown_*.stl"))
    return crowns[0] if crowns else None


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        rel = case_path
    return "__".join(rel.parts)


def span(xyz: np.ndarray) -> np.ndarray:
    return np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1e-6)


def normalized_l2(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    return float(np.linalg.norm((a - b) / np.maximum((a + b) * 0.5, eps)))


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
        "case",
        "stem",
        "method",
        "ok",
        "template_case",
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
        "template_stl",
        "error",
    ]
    with (output_dir / "metrics_by_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = summarize([row for row in rows if row.get("ok")])
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
        method_rows = [row for row in rows if row.get("method") == method]
        if not method_rows:
            continue
        item = {"method": method, "n": len(method_rows)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in method_rows if row.get(key) not in ("", None)])
            if values.size:
                item[f"{key}_mean"] = float(np.mean(values))
                item[f"{key}_median"] = float(np.median(values))
                item[f"{key}_max"] = float(np.max(values))
        out.append(item)
    return out


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
