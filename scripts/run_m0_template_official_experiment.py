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
from src.crown_m0.io import write_ply, write_xyz
from src.crown_m0.model import M0TemplateDeformNet
from src.crown_m0.template_mesh import load_template_npz, write_template_mesh_stl
from src.crown_m0.template_selection import case_feature, load_template_index, select_template


OFFICIAL_STL_METHOD = "template_displacement"


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Run the official M0 template-deformation evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/m0_template/best.pt"))
    parser.add_argument("--template-index", type=Path, default=Path("templates/m0_dynamic_library_4096/template_index.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--output-root", type=Path, default=Path("result"))
    parser.add_argument("--date", default=today, help="Result date folder, formatted as YYYYMMDD.")
    parser.add_argument("--experiment-name", default="m0_template")
    parser.add_argument("--batch-size", type=int, default=1)
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
    write_config(output_dir / "config.json", args, first_vertices, template_entries)

    records = discover_cases(args.data_dir)
    by_case = {str(record.train_dir.parent): record for record in records}
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    cases = split["test"]
    selected_records = [by_case[case] for case in cases if case in by_case]
    missing = [case for case in cases if case not in by_case]
    if missing:
        print(f"[warn] test: missing {len(missing)} cases", flush=True)

    sample_dir = output_dir / "test"
    rows = run_sample_set(model, selected_records, template_entries, sample_dir, args)
    write_sample_outputs(sample_dir, rows)
    summary = summarize_rows(rows)
    write_csv(output_dir / "summary_by_sample_set.csv", [{"sample_set": "test", **summary[OFFICIAL_STL_METHOD]}])
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


def run_sample_set(model, records, template_entries, sample_dir: Path, args: argparse.Namespace):
    loader = DataLoader(CrownDataset(records), batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    template_cache = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=sample_dir.name):
            if len(batch["case_id"]) != 1:
                raise ValueError("Dynamic template evaluation currently requires --batch-size 1")
            case_path = Path(batch["case_id"][0])
            record = next(record for record in records if str(record.train_dir.parent) == str(case_path))
            entry = select_template(
                template_entries,
                tooth_id=record.tooth_id,
                prep_arch=record.prep_arch,
                feature=case_feature(case_path),
            )
            template_vertices, faces_np = load_eval_template(entry, template_cache, args.device)
            pred_vertices, _ = model(
                batch["prep"].to(args.device),
                batch["antagonist"].to(args.device),
                batch["tooth_index"].to(args.device),
                batch["prep_arch_index"].to(args.device),
                template_vertices,
            )
            for i, case in enumerate(batch["case_id"]):
                row = evaluate_case(
                    Path(case),
                    pred_vertices[i].detach().cpu().numpy(),
                    faces_np,
                    entry,
                    sample_dir,
                    args,
                )
                rows.append(row)
    return rows


def load_eval_template(entry, cache, device):
    cached = cache.get(entry.template)
    if cached is None:
        vertices_np, faces_np = load_template_npz(entry.template)
        cached = (torch.from_numpy(vertices_np).to(device), faces_np)
        cache[entry.template] = cached
    return cached


def evaluate_case(case_path: Path, pred_local_xyz: np.ndarray, faces_np: np.ndarray, entry, sample_dir: Path, args) -> dict:
    stem = safe_case_name(case_path, args.data_dir)
    case_dir = sample_dir / "cases" / stem
    case_dir.mkdir(parents=True, exist_ok=True)

    row = {"case": str(case_path), "stem": stem, "ok": False, "stl_method": OFFICIAL_STL_METHOD, "error": ""}
    pred_npy = case_dir / f"{stem}_pred_vertices.npy"
    pred_xyz = case_dir / f"{stem}_pred_vertices.xyz"
    pred_ply = case_dir / f"{stem}_pred_vertices.ply"
    pred_stl = case_dir / f"{stem}_pred_{OFFICIAL_STL_METHOD}.stl"
    template_stl_copy = case_dir / f"{stem}_selected_template.stl"
    gt_stl_copy = case_dir / f"{stem}_GT_technician.stl"

    try:
        np.save(pred_npy, pred_local_xyz.astype(np.float32))
        write_xyz(pred_xyz, pred_local_xyz)
        write_ply(pred_ply, np.pad(pred_local_xyz, ((0, 0), (0, 3)), constant_values=0.0))

        pred_original = restore_original_coordinates(pred_local_xyz, case_path)
        write_template_mesh_stl(pred_stl, pred_original, faces_np)
        shutil.copy2(entry.preview_stl, template_stl_copy)

        gt_stl = find_gt_stl(case_path)
        if gt_stl is None:
            raise FileNotFoundError(f"missing GT crown STL in {case_path / 'processed'}")
        shutil.copy2(gt_stl, gt_stl_copy)
        gt_mesh = read_mesh(gt_stl)
        pred_mesh = read_mesh(pred_stl)
        stl_metrics = surface_metrics(
            sample_mesh_points(pred_mesh, args.sample_points),
            sample_mesh_points(gt_mesh, args.sample_points),
        )
        row.update(prefix_metrics("stl", stl_metrics))
        row.update(mesh_stats(pred_mesh))

        gt_local = np.load(case_path / "train" / "crown_points.npy").astype(np.float64)[:, :3]
        point_metrics_row = point_metrics(pred_local_xyz, gt_local)
        row.update(prefix_metrics("vertex", point_metrics_row))
        row.update(
            {
                "ok": True,
                "pred_stl": str(pred_stl),
                "gt_stl": str(gt_stl_copy),
                "selected_template_stl": str(template_stl_copy),
                "selected_template_case": entry.source_case,
                "selected_template_tooth_id": entry.tooth_id,
                "selected_template_prep_arch": entry.prep_arch,
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


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


def surface_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    return point_metrics(pred_xyz, gt_xyz)


def point_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    pred_to_gt = cKDTree(gt_xyz).query(pred_xyz, k=1)[0]
    gt_to_pred = cKDTree(pred_xyz).query(gt_xyz, k=1)[0]
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
    ok_rows = [row for row in rows if row.get("ok")]
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
    ]
    summary = {OFFICIAL_STL_METHOD: {"method": OFFICIAL_STL_METHOD, "n": len(ok_rows)}}
    item = summary[OFFICIAL_STL_METHOD]
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in ok_rows if row.get(key) not in ("", None)], dtype=float)
        if values.size:
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_median"] = float(np.median(values))
            item[f"{key}_max"] = float(np.max(values))
    return summary


def write_config(path: Path, args: argparse.Namespace, vertices: np.ndarray, template_entries) -> None:
    payload = vars(args).copy()
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["official_stl_method"] = OFFICIAL_STL_METHOD
    payload["template_vertices"] = int(vertices.shape[0])
    payload["template_library_size"] = int(len(template_entries))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    priority = ["sample_set", "case", "stem", "method", "stl_method", "ok", "error"]
    ordered = [key for key in priority if key in fieldnames] + [key for key in fieldnames if key not in priority]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_outputs(sample_dir: Path, rows: list[dict]) -> None:
    write_csv(sample_dir / "metrics_by_case.csv", rows)
    summary = summarize_rows(rows)
    write_csv(sample_dir / "summary_metrics.csv", list(summary.values()))
    (sample_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.relative_to(data_dir)
        return "__".join(rel.parts)
    except ValueError:
        return case_path.name


if __name__ == "__main__":
    main()
