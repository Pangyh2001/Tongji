from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .manifest import build_manifest
from .pipeline import run_cross_validation, run_full_pipeline, run_single_case_overfit
from .preprocess import preprocess_manifest


def _variants_argument(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crown generation validation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest", help="Generate cases_manifest.csv from raw STL folders.")
    manifest_parser.add_argument("--raw-dir", required=True)
    manifest_parser.add_argument("--output", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess raw STL files into normalized NPZ cases.")
    preprocess_parser.add_argument("--manifest", required=True)
    preprocess_parser.add_argument("--output-dir", required=True)
    preprocess_parser.add_argument("--config", default=None)
    preprocess_parser.add_argument("--include-unconfirmed", action="store_true")

    cv_parser = subparsers.add_parser("run-cv", help="Run leave-one-out cross-validation experiments.")
    cv_parser.add_argument("--manifest", required=True)
    cv_parser.add_argument("--processed-dir", required=True)
    cv_parser.add_argument("--output-dir", required=True)
    cv_parser.add_argument("--config", default=None)
    cv_parser.add_argument("--variants", default="baseline,improved")
    cv_parser.add_argument("--include-unconfirmed", action="store_true")

    full_parser = subparsers.add_parser("full-pipeline", help="Run manifest generation, preprocessing, and CV training.")
    full_parser.add_argument("--raw-dir", required=True)
    full_parser.add_argument("--work-dir", required=True)
    full_parser.add_argument("--config", default=None)
    full_parser.add_argument("--variants", default="baseline,improved")
    full_parser.add_argument("--include-unconfirmed", action="store_true")

    overfit_parser = subparsers.add_parser("overfit-one", help="Overfit a single case to test model capacity.")
    overfit_parser.add_argument("--raw-dir", required=True)
    overfit_parser.add_argument("--work-dir", required=True)
    overfit_parser.add_argument("--case-id", required=True)
    overfit_parser.add_argument("--variant", default="improved")
    overfit_parser.add_argument("--config", default=None)
    overfit_parser.add_argument("--device", default=None)
    overfit_parser.add_argument("--include-unconfirmed", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-manifest":
        build_manifest(raw_dir=args.raw_dir, output_csv=args.output)
        return

    if args.command == "preprocess":
        config = load_config(args.config)
        preprocess_manifest(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            prep_points=int(config["prep_points"]),
            opposing_points=int(config["opposing_points"]),
            crown_points=int(config["crown_points"]),
            margin_points=int(config["margin_points"]),
            seed=int(config["seed"]),
            include_unconfirmed=args.include_unconfirmed,
        )
        return

    if args.command == "run-cv":
        run_cross_validation(
            manifest_path=args.manifest,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            variants=_variants_argument(args.variants),
            include_unconfirmed=args.include_unconfirmed,
        )
        return

    if args.command == "full-pipeline":
        run_full_pipeline(
            raw_dir=args.raw_dir,
            work_dir=args.work_dir,
            config_path=args.config,
            variants=_variants_argument(args.variants),
            include_unconfirmed=args.include_unconfirmed,
        )
        return

    if args.command == "overfit-one":
        run_single_case_overfit(
            raw_dir=args.raw_dir,
            work_dir=args.work_dir,
            config_path=args.config,
            variant=args.variant,
            case_id=args.case_id,
            device_override=args.device,
            include_unconfirmed=args.include_unconfirmed,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
