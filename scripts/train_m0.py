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
from src.crown_m0.losses import m0_loss
from src.crown_m0.model import M0CrownNet


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

    model = M0CrownNet(output_points=args.output_points).to(args.device)
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


def run_epoch(
    model: M0CrownNet,
    loader: DataLoader,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "chamfer": 0.0, "normal": 0.0}
    count = 0

    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        prep = batch["prep"].to(args.device, non_blocking=True)
        antagonist = batch["antagonist"].to(args.device, non_blocking=True)
        crown = batch["crown"].to(args.device, non_blocking=True)
        tooth_index = batch["tooth_index"].to(args.device, non_blocking=True)
        prep_arch_index = batch["prep_arch_index"].to(args.device, non_blocking=True)

        with torch.set_grad_enabled(training):
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
