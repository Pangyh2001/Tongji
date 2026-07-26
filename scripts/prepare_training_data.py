from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.crown_m0.preprocess import PreprocessConfig, iter_processed_cases, preprocess_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-density training point clouds from processed STL cases.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--case", type=Path, default=None, help="Optional single case directory to process.")
    parser.add_argument("--roi-side-mm", type=float, default=20.0)
    parser.add_argument("--prep-points", type=int, default=8192)
    parser.add_argument("--antagonist-points", type=int, default=8192)
    parser.add_argument("--crown-points", type=int, default=16384)
    parser.add_argument("--margin-points", type=int, default=1024)
    parser.add_argument("--candidate-factor", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--summary", type=Path, default=Path("training_preprocess_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessConfig(
        roi_side_mm=args.roi_side_mm,
        prep_points=args.prep_points,
        antagonist_points=args.antagonist_points,
        crown_points=args.crown_points,
        margin_points=args.margin_points,
        candidate_factor=args.candidate_factor,
        seed=args.seed,
    )
    cases = [args.case] if args.case else list(iter_processed_cases(args.data_dir))
    rows = []
    for case_dir in tqdm(cases, desc="preprocess"):
        try:
            meta = preprocess_case(case_dir, config)
            rows.append(
                {
                    "case": str(case_dir),
                    "ok": True,
                    "tooth_id": meta.get("tooth_id"),
                    "prep_arch": meta["role_assignment"]["prep_arch"],
                    "margin_to_upper_mean": meta["role_assignment"]["margin_to_upper_mean"],
                    "margin_to_lower_mean": meta["role_assignment"]["margin_to_lower_mean"],
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append({"case": str(case_dir), "ok": False, "error": str(exc)})

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case", "ok", "tooth_id", "prep_arch", "margin_to_upper_mean", "margin_to_lower_mean", "error"]
    with args.summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["ok"])
    print(f"Processed {ok}/{len(rows)} cases. Summary: {args.summary}")


if __name__ == "__main__":
    main()
