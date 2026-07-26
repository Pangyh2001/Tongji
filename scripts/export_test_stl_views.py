from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from export_prediction_stl import prediction_to_stl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export test split GT/pred STL files for visual inspection.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("predictions/m0_new_all"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/m0_new_test/stl"))
    parser.add_argument(
        "--method",
        choices=["alpha_clean", "alpha_raw", "poisson"],
        default="alpha_clean",
        help="Official experiment default is alpha_clean.",
    )
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.split_file.read_text(encoding="utf-8"))
    cases = payload[args.split]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in tqdm(cases, desc="export-stl"):
        case_path = Path(case)
        stem = safe_case_name(case_path, args.data_dir)
        out_case = args.output_dir / stem
        out_case.mkdir(parents=True, exist_ok=True)

        pred_npy = args.prediction_dir / f"{stem}.npy"
        gt_npy = case_path / "train" / "crown_points.npy"
        gt_stl = find_gt_stl(case_path)

        pred_out = out_case / f"{stem}_M0_pred.stl"
        gt_out = out_case / f"{stem}_GT_technician.stl"
        gt_points_out = out_case / f"{stem}_GT_points_alpha.stl"

        if pred_npy.exists():
            prediction_to_stl(
                pred_npy,
                pred_out,
                method=args.method,
                alpha=args.alpha,
                depth=args.depth,
                density_quantile=0.02,
            )
        if gt_stl and gt_stl.exists():
            mesh = o3d.io.read_triangle_mesh(str(gt_stl))
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            o3d.io.write_triangle_mesh(str(gt_out), mesh, write_ascii=False)
        elif gt_npy.exists():
            temp_npy = out_case / "_gt_temp.npy"
            np.save(temp_npy, np.load(gt_npy).astype(np.float32))
            prediction_to_stl(
                temp_npy,
                gt_points_out,
                method=args.method,
                alpha=args.alpha,
                depth=args.depth,
                density_quantile=0.02,
            )
            temp_npy.unlink(missing_ok=True)

        rows.append(
            {
                "case": case,
                "pred_stl": str(pred_out) if pred_out.exists() else "",
                "gt_stl": str(gt_out) if gt_out.exists() else str(gt_points_out) if gt_points_out.exists() else "",
            }
        )

    (args.output_dir / "stl_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output_dir}")


def find_gt_stl(case_path: Path) -> Path | None:
    processed = case_path / "processed"
    crowns = sorted(processed.glob("crown_*.stl"))
    return crowns[0] if crowns else None


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        rel = case_path
    return "__".join(rel.parts)


if __name__ == "__main__":
    main()
