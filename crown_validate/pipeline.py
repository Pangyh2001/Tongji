from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .config import load_config
from .manifest import build_manifest, load_manifest
from .model import CrownDeformationNet
from .preprocess import preprocess_manifest
from .template import build_open_cap_template
from .training import train_single_fold
from .utils import choose_device, ensure_dir, json_dump, set_seed


def _make_leave_one_out_splits(case_ids: list[str]) -> list[dict[str, list[str] | str]]:
    if len(case_ids) < 3:
        raise ValueError("At least 3 cases are required for leave-one-out validation.")

    folds: list[dict[str, list[str] | str]] = []
    for index, case_id in enumerate(case_ids):
        test_case = case_id
        val_case = case_ids[(index + 1) % len(case_ids)]
        train_cases = [item for item in case_ids if item not in {test_case, val_case}]
        folds.append(
            {
                "fold_name": f"fold_{index + 1:02d}",
                "train_case_ids": train_cases,
                "val_case_ids": [val_case],
                "test_case_ids": [test_case],
            }
        )
    return folds


def run_cross_validation(
    manifest_path: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None,
    variants: list[str],
    include_unconfirmed: bool = True,
) -> Path:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    output_root = ensure_dir(output_dir)
    device = choose_device(str(config["device"]))

    rows = load_manifest(manifest_path, include_unconfirmed=include_unconfirmed)
    case_ids = [row["case_id"] for row in rows]
    folds = _make_leave_one_out_splits(case_ids)

    template_vertices, template_faces, boundary_indices = build_open_cap_template(**config["template"])
    template_vertices_t = torch.from_numpy(template_vertices)
    template_faces_t = torch.from_numpy(template_faces)
    boundary_indices_t = torch.from_numpy(boundary_indices)

    variant_summaries: dict[str, list[dict[str, float | str]]] = {}
    for variant_name in variants:
        variant_dir = ensure_dir(output_root / variant_name)
        loss_cfg = config["losses"][variant_name]
        summaries: list[dict[str, float | str]] = []
        for fold in folds:
            fold_dir = ensure_dir(variant_dir / str(fold["fold_name"]))
            run_name = f"{variant_name}/{fold['fold_name']}"
            model = CrownDeformationNet(
                template_vertices=template_vertices_t,
                template_faces=template_faces_t,
                boundary_indices=boundary_indices_t,
                hidden_dim=int(config["model"]["hidden_dim"]),
                latent_dim=int(config["model"]["latent_dim"]),
                dropout=float(config["model"]["dropout"]),
                max_offset=float(config["model"]["max_offset"]),
                encoder_knn=int(config["model"].get("encoder_knn", 16)),
                decoder_knn=int(config["model"].get("decoder_knn", 16)),
            ).to(device)

            summary = train_single_fold(
                model=model,
                processed_dir=processed_dir,
                train_case_ids=list(fold["train_case_ids"]),
                val_case_ids=list(fold["val_case_ids"]),
                test_case_ids=list(fold["test_case_ids"]),
                device=device,
                loss_cfg=loss_cfg,
                training_cfg=config["training"],
                metric_cfg=config["metrics"],
                output_dir=fold_dir,
                run_name=run_name,
            )
            summary["fold_name"] = str(fold["fold_name"])
            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(variant_dir / "cv_results.csv", index=False)
        numeric_columns = summary_df.select_dtypes(include=["number"]).columns
        aggregate = {}
        for column in numeric_columns:
            aggregate[column] = {
                "mean": float(summary_df[column].mean()),
                "std": float(summary_df[column].std(ddof=0)),
            }
        json_dump({"variant": variant_name, "aggregate": aggregate}, variant_dir / "aggregate_metrics.json")
        variant_summaries[variant_name] = summaries

    comparison_rows: list[dict[str, float | str]] = []
    for variant_name, summaries in variant_summaries.items():
        frame = pd.DataFrame(summaries)
        comparison_rows.append(
            {
                "variant": variant_name,
                "mean_chamfer_l2_mm2": float(frame["chamfer_l2_mm2"].mean()),
                "mean_hd95_mm": float(frame["hd95_mm"].mean()),
                "mean_fscore": float(frame["fscore"].mean()),
                "mean_margin_chamfer_l2_mm2": float(frame["margin_chamfer_l2_mm2"].mean()),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(output_root / "variant_comparison.csv", index=False)
    return output_root


def run_full_pipeline(
    raw_dir: str | Path,
    work_dir: str | Path,
    config_path: str | Path | None,
    variants: list[str],
    include_unconfirmed: bool = True,
) -> Path:
    config = load_config(config_path)
    work_root = ensure_dir(work_dir)
    manifest_path = build_manifest(Path(raw_dir), work_root / "cases_manifest.csv")
    preprocess_manifest(
        manifest_path=manifest_path,
        output_dir=work_root / "processed_cases",
        prep_points=int(config["prep_points"]),
        opposing_points=int(config["opposing_points"]),
        crown_points=int(config["crown_points"]),
        margin_points=int(config["margin_points"]),
        seed=int(config["seed"]),
        include_unconfirmed=include_unconfirmed,
    )
    return run_cross_validation(
        manifest_path=manifest_path,
        processed_dir=work_root / "processed_cases",
        output_dir=work_root / "experiments",
        config_path=config_path,
        variants=variants,
        include_unconfirmed=include_unconfirmed,
    )


def run_single_case_overfit(
    raw_dir: str | Path,
    work_dir: str | Path,
    config_path: str | Path | None,
    variant: str,
    case_id: str,
    device_override: str | None = None,
    include_unconfirmed: bool = True,
) -> Path:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    work_root = ensure_dir(work_dir)
    device = choose_device(device_override if device_override is not None else str(config["device"]))

    manifest_path = build_manifest(Path(raw_dir), work_root / "cases_manifest.csv")
    preprocess_manifest(
        manifest_path=manifest_path,
        output_dir=work_root / "processed_cases",
        prep_points=int(config["prep_points"]),
        opposing_points=int(config["opposing_points"]),
        crown_points=int(config["crown_points"]),
        margin_points=int(config["margin_points"]),
        seed=int(config["seed"]),
        include_unconfirmed=include_unconfirmed,
    )

    rows = load_manifest(manifest_path, include_unconfirmed=include_unconfirmed)
    case_ids = [row["case_id"] for row in rows]
    if case_id not in case_ids:
        raise ValueError(f"Case '{case_id}' not found in manifest. Available case_ids: {', '.join(case_ids)}")
    if variant not in config["losses"]:
        raise ValueError(f"Variant '{variant}' not found in config losses: {', '.join(config['losses'].keys())}")

    template_vertices, template_faces, boundary_indices = build_open_cap_template(**config["template"])
    model = CrownDeformationNet(
        template_vertices=torch.from_numpy(template_vertices),
        template_faces=torch.from_numpy(template_faces),
        boundary_indices=torch.from_numpy(boundary_indices),
        hidden_dim=int(config["model"]["hidden_dim"]),
        latent_dim=int(config["model"]["latent_dim"]),
        dropout=float(config["model"]["dropout"]),
        max_offset=float(config["model"]["max_offset"]),
        encoder_knn=int(config["model"].get("encoder_knn", 16)),
        decoder_knn=int(config["model"].get("decoder_knn", 16)),
    ).to(device)

    output_dir = ensure_dir(work_root / "overfit" / variant / case_id)
    summary = train_single_fold(
        model=model,
        processed_dir=work_root / "processed_cases",
        train_case_ids=[case_id],
        val_case_ids=[case_id],
        test_case_ids=[case_id],
        device=device,
        loss_cfg=config["losses"][variant],
        training_cfg=config["training"],
        metric_cfg=config["metrics"],
        output_dir=output_dir,
        run_name=f"overfit/{variant}/{case_id}",
    )
    summary["mode"] = "single_case_overfit"
    summary["variant"] = variant
    summary["case_id"] = case_id
    json_dump(summary, output_dir / "overfit_summary.json")
    return output_dir
