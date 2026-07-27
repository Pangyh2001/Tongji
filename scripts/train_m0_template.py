from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crown_m0.dataset import CrownDataset, discover_cases
from src.crown_m0.losses import template_deform_loss
from src.crown_m0.model import M0TemplateDeformNet
from src.crown_m0.template_mesh import load_template_npz, mesh_edges_from_faces
from src.crown_m0.template_selection import case_feature, load_template_index, select_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M0 template-deformation baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/m0_patient_split_seed20260706.json"))
    parser.add_argument("--template-index", type=Path, default=Path("templates/m0_dynamic_library_4096/template_index.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m0_template"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--chamfer-points", type=int, default=4096)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--laplacian-weight", type=float, default=0.10)
    parser.add_argument("--displacement-weight", type=float, default=0.001)
    parser.add_argument("--max-displacement-mm", type=float, default=4.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template_entries = load_template_index(args.template_index)
    first_vertices, _ = load_template_npz(template_entries[0].template)

    records = discover_cases(args.data_dir)
    train_records, val_records, test_records = load_split_records(records, args.split_file)
    record_by_case = {str(record.train_dir.parent): record for record in train_records + val_records + test_records}
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

    model = M0TemplateDeformNet(
        template_vertices=first_vertices.shape[0],
        max_displacement_mm=args.max_displacement_mm,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, template_entries, record_by_case, args, optimizer)
        val_metrics = run_epoch(model, val_loader, template_entries, record_by_case, args, None) if val_records else {}
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}, ensure_ascii=False))

        latest = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": sanitize_args(args),
            "template_vertices": int(first_vertices.shape[0]),
            "template_library_size": int(len(template_entries)),
        }
        torch.save(latest, args.output_dir / "latest.pt")
        val_loss = val_metrics.get("loss", train_metrics["loss"])
        if val_loss < best_val:
            best_val = val_loss
            torch.save(latest, args.output_dir / "best.pt")


def run_epoch(model, loader, template_entries, record_by_case, args, optimizer) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "chamfer": 0.0, "edge": 0.0, "laplacian": 0.0, "displacement": 0.0}
    count = 0
    template_cache = {}

    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        if len(batch["case_id"]) != 1:
            raise ValueError("Dynamic template training currently requires --batch-size 1")
        case_path = Path(batch["case_id"][0])
        record = record_by_case[str(case_path)]
        entry = select_template(
            template_entries,
            tooth_id=record.tooth_id,
            prep_arch=record.prep_arch,
            feature=case_feature(case_path),
        )
        template_vertices, edges = load_training_template(entry, template_cache, args.device)
        prep = batch["prep"].to(args.device, non_blocking=True)
        antagonist = batch["antagonist"].to(args.device, non_blocking=True)
        crown = batch["crown"].to(args.device, non_blocking=True)
        tooth_index = batch["tooth_index"].to(args.device, non_blocking=True)
        prep_arch_index = batch["prep_arch_index"].to(args.device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred_vertices, delta = model(prep, antagonist, tooth_index, prep_arch_index, template_vertices)
            loss, metrics = template_deform_loss(
                pred_vertices,
                crown[..., :3],
                delta,
                template_vertices,
                edges,
                chamfer_points=args.chamfer_points,
                edge_weight=args.edge_weight,
                laplacian_weight=args.laplacian_weight,
                displacement_weight=args.displacement_weight,
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


def load_training_template(entry, cache, device):
    cached = cache.get(entry.template)
    if cached is None:
        vertices_np, faces_np = load_template_npz(entry.template)
        edges_np = mesh_edges_from_faces(faces_np)
        cached = (
            torch.from_numpy(vertices_np).to(device),
            torch.from_numpy(edges_np).to(device),
        )
        cache[entry.template] = cached
    return cached


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


def sanitize_args(args: argparse.Namespace) -> dict:
    out = vars(args).copy()
    for key, value in out.items():
        if isinstance(value, Path):
            out[key] = str(value)
    return out


if __name__ == "__main__":
    main()
