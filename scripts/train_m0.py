from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.crown_m0.dataset import CrownDataset, discover_cases, split_by_patient
from src.crown_m0.losses import (
    coarse_to_fine_loss,
    dmc_dpsr_loss,
    m0_loss,
    tangent_coarse_to_fine_loss,
)
from src.crown_m0.model import (
    M0CoarseToFineNet,
    M0CrownNet,
    M0DMCDPSRNet,
    M0TangentCoarseToFineNet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M0 crown generation baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m0"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-file", type=Path, default=None, help="Optional fixed split JSON from make_m0_split.py.")
    parser.add_argument("--chamfer-points", type=int, default=2048)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--output-points", type=int, default=16384)
    parser.add_argument(
        "--decoder",
        choices=[
            "direct",
            "coarse_to_fine",
            "coarse_to_fine_tangent",
            "dmc_dpsr",
            "dmc_dpsr_m1",
            "dmc_dpsr_m2",
            "dmc_dpsr_m3",
        ],
        default="direct",
    )
    parser.add_argument("--coarse-points", type=int, default=8192)
    parser.add_argument("--first-factor", type=int, default=4)
    parser.add_argument("--second-factor", type=int, default=2)
    parser.add_argument("--repulsion-weight", type=float, default=0.10)
    parser.add_argument("--uniformity-weight", type=float, default=0.01)
    parser.add_argument("--offset-weight", type=float, default=0.001)
    parser.add_argument("--point-to-plane-weight", type=float, default=0.50)
    parser.add_argument("--local-plane-weight", type=float, default=0.20)
    parser.add_argument("--normal-drift-weight", type=float, default=0.50)
    parser.add_argument("--grid-weight", type=float, default=100.0)
    parser.add_argument("--dpsr-resolution", type=int, default=128)
    parser.add_argument("--dpsr-sigma", type=float, default=2.0)
    parser.add_argument("--roi-half-extent-mm", type=float, default=12.0)
    parser.add_argument("--dmc-model-dim", type=int, default=256)
    parser.add_argument("--dmc-context-tokens", type=int, default=256)
    parser.add_argument("--dmc-queries", type=int, default=256)
    parser.add_argument("--dmc-fold-step", type=int, default=8)
    parser.add_argument("--dmc-transformer-layers", type=int, default=3)
    parser.add_argument("--margin-anchor-queries", type=int, default=64)
    parser.add_argument("--ring-groups", type=int, default=6)
    parser.add_argument("--margin-anchor-weight", type=float, default=0.5)
    parser.add_argument("--margin-risk-weight", type=float, default=0.5)
    parser.add_argument("--margin-risk-alpha", type=float, default=3.0)
    parser.add_argument("--margin-risk-sigma-mm", type=float, default=1.0)
    parser.add_argument("--margin-zero-weight", type=float, default=0.0)
    parser.add_argument("--narrow-band-weight", type=float, default=0.0)
    parser.add_argument("--narrow-band-width", type=float, default=0.08)
    parser.add_argument("--multiscale-grid-weight", type=float, default=0.0)
    parser.add_argument("--grid-gradient-weight", type=float, default=0.0)
    parser.add_argument("--topology-weight", type=float, default=0.0)
    parser.add_argument("--topology-resolution", type=int, default=32)
    parser.add_argument("--topology-temperature", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_cases(args.data_dir)
    if not records:
        raise SystemExit(f"No training records found under {args.data_dir}")
    if args.split_file:
        train_records, val_records, test_records = load_split_records(records, args.split_file)
    else:
        train_records, val_records, test_records = split_by_patient(
            records, args.val_fraction, args.test_fraction, args.seed
        )
    split = {
        "train": [str(r.train_dir.parent) for r in train_records],
        "val": [str(r.train_dir.parent) for r in val_records],
        "test": [str(r.train_dir.parent) for r in test_records],
    }
    (args.output_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        CrownDataset(train_records, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        CrownDataset(val_records, seed=args.seed + 1000),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = build_model(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, optimizer)
        val_metrics = run_epoch(model, val_loader, args, None) if val_records else {}
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(row, ensure_ascii=False))

        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(latest, args.output_dir / "latest.pt")
        val_loss = val_metrics.get("loss", train_metrics["loss"])
        if val_loss < best_val:
            best_val = val_loss
            torch.save(latest, args.output_dir / "best.pt")


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.decoder.startswith("dmc_dpsr"):
        use_margin = args.decoder != "dmc_dpsr"
        use_anchor = args.decoder in {"dmc_dpsr_m2", "dmc_dpsr_m3"}
        return M0DMCDPSRNet(
            model_dim=args.dmc_model_dim,
            context_tokens_per_input=args.dmc_context_tokens,
            num_queries=args.dmc_queries,
            fold_step=args.dmc_fold_step,
            transformer_layers=args.dmc_transformer_layers,
            dpsr_resolution=args.dpsr_resolution,
            dpsr_sigma=args.dpsr_sigma,
            roi_half_extent_mm=args.roi_half_extent_mm,
            use_margin=use_margin,
            margin_anchor_queries=args.margin_anchor_queries if use_anchor else 0,
            ring_groups=args.ring_groups if use_anchor else 0,
        )
    if args.decoder == "coarse_to_fine_tangent":
        return M0TangentCoarseToFineNet(
            coarse_points=args.coarse_points,
            first_factor=args.first_factor,
            second_factor=args.second_factor,
        )
    if args.decoder == "coarse_to_fine":
        return M0CoarseToFineNet(
            coarse_points=args.coarse_points,
            first_factor=args.first_factor,
            second_factor=args.second_factor,
        )
    return M0CrownNet(output_points=args.output_points)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if args.decoder.startswith("dmc_dpsr"):
        sums = {
            "loss": 0.0,
            "chamfer": 0.0,
            "normal": 0.0,
            "grid_mse": 0.0,
            "grid_l1": 0.0,
            "margin_anchor": 0.0,
            "margin_risk": 0.0,
            "margin_zero": 0.0,
            "narrow_band": 0.0,
            "multiscale_grid": 0.0,
            "grid_gradient": 0.0,
            "topology": 0.0,
        }
    elif args.decoder == "coarse_to_fine_tangent":
        sums = {
            "loss": 0.0,
            "coarse_chamfer": 0.0,
            "middle_chamfer": 0.0,
            "fine_chamfer": 0.0,
            "fine_normal": 0.0,
            "fine_point_to_plane": 0.0,
            "repulsion": 0.0,
            "uniformity": 0.0,
            "local_plane": 0.0,
            "normal_drift": 0.0,
        }
    elif args.decoder == "coarse_to_fine":
        sums = {
            "loss": 0.0,
            "coarse_chamfer": 0.0,
            "middle_chamfer": 0.0,
            "fine_chamfer": 0.0,
            "fine_normal": 0.0,
            "repulsion": 0.0,
            "uniformity": 0.0,
            "offset": 0.0,
        }
    else:
        sums = {"loss": 0.0, "chamfer": 0.0, "normal": 0.0}
    count = 0

    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        prep = batch["prep"].to(args.device, non_blocking=True)
        antagonist = batch["antagonist"].to(args.device, non_blocking=True)
        crown = batch["crown"].to(args.device, non_blocking=True)
        margin = batch["margin"].to(args.device, non_blocking=True)
        tooth_index = batch["tooth_index"].to(args.device, non_blocking=True)
        prep_arch_index = batch["prep_arch_index"].to(args.device, non_blocking=True)

        with torch.set_grad_enabled(training):
            if args.decoder.startswith("dmc_dpsr"):
                outputs = model(
                    prep,
                    antagonist,
                    tooth_index,
                    prep_arch_index,
                    margin=margin if model.use_margin else None,
                    return_grid=True,
                )
                with torch.no_grad():
                    target_grid = model.points_to_grid(crown)
                loss, metrics = dmc_dpsr_loss(
                    outputs["points"],
                    outputs["psr_grid"],
                    crown,
                    target_grid,
                    chamfer_points=args.chamfer_points,
                    normal_weight=args.normal_weight,
                    grid_weight=args.grid_weight,
                    margin=margin if model.use_margin else None,
                    margin_anchor_weight=(
                        args.margin_anchor_weight
                        if args.decoder in {"dmc_dpsr_m2", "dmc_dpsr_m3"}
                        else 0.0
                    ),
                    margin_risk_weight=(
                        args.margin_risk_weight if args.decoder == "dmc_dpsr_m3" else 0.0
                    ),
                    margin_risk_alpha=args.margin_risk_alpha,
                    margin_risk_sigma_mm=args.margin_risk_sigma_mm,
                    roi_half_extent_mm=args.roi_half_extent_mm,
                    margin_zero_weight=args.margin_zero_weight,
                    narrow_band_weight=args.narrow_band_weight,
                    narrow_band_width=args.narrow_band_width,
                    multiscale_grid_weight=args.multiscale_grid_weight,
                    grid_gradient_weight=args.grid_gradient_weight,
                    topology_weight=args.topology_weight,
                    topology_resolution=args.topology_resolution,
                    topology_temperature=args.topology_temperature,
                )
            elif args.decoder == "coarse_to_fine_tangent":
                outputs = model(
                    prep,
                    antagonist,
                    tooth_index,
                    prep_arch_index,
                    return_stages=True,
                )
                loss, metrics = tangent_coarse_to_fine_loss(
                    outputs["stages"],
                    outputs["offsets"],
                    crown,
                    chamfer_points=args.chamfer_points,
                    normal_weight=args.normal_weight,
                    point_to_plane_weight=args.point_to_plane_weight,
                    local_plane_weight=args.local_plane_weight,
                    repulsion_weight=args.repulsion_weight,
                    uniformity_weight=args.uniformity_weight,
                    normal_drift_weight=args.normal_drift_weight,
                )
            elif args.decoder == "coarse_to_fine":
                outputs = model(
                    prep,
                    antagonist,
                    tooth_index,
                    prep_arch_index,
                    return_stages=True,
                )
                loss, metrics = coarse_to_fine_loss(
                    outputs["stages"],
                    outputs["offsets"],
                    crown,
                    chamfer_points=args.chamfer_points,
                    normal_weight=args.normal_weight,
                    repulsion_weight=args.repulsion_weight,
                    uniformity_weight=args.uniformity_weight,
                    offset_weight=args.offset_weight,
                )
            else:
                pred = model(prep, antagonist, tooth_index, prep_arch_index)
                loss, metrics = m0_loss(
                    pred,
                    crown,
                    chamfer_points=args.chamfer_points,
                    normal_weight=args.normal_weight,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = prep.shape[0]
        count += bs
        for key in sums:
            sums[key] += metrics[key] * bs

    return {key: value / max(count, 1) for key, value in sums.items()}


def load_split_records(records, split_file: Path):
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    by_case = {str(record.train_dir.parent): record for record in records}

    def collect(name: str):
        missing = []
        out = []
        for case_dir in payload[name]:
            record = by_case.get(case_dir)
            if record is None:
                missing.append(case_dir)
            else:
                out.append(record)
        if missing:
            raise SystemExit(f"{split_file} references {len(missing)} missing {name} cases, first: {missing[0]}")
        return out

    return collect("train"), collect("val"), collect("test")


if __name__ == "__main__":
    main()
