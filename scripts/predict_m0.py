from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.crown_m0.dataset import CrownDataset, discover_cases
from src.crown_m0.io import write_ply, write_xyz
from src.crown_m0.model import M0CrownNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M0 inference and export predicted crown points.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/m0"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_cases(args.data_dir)
    loader = DataLoader(CrownDataset(records), batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = M0CrownNet().to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["prep"].to(args.device),
                batch["antagonist"].to(args.device),
                batch["tooth_index"].to(args.device),
                batch["prep_arch_index"].to(args.device),
            )
            pred_np = pred.cpu().numpy()
            for i, case_id in enumerate(batch["case_id"]):
                safe_name = safe_case_name(Path(case_id), args.data_dir)
                out_base = args.output_dir / safe_name
                np.save(out_base.with_suffix(".npy"), pred_np[i].astype(np.float32))
                write_xyz(out_base.with_suffix(".xyz"), pred_np[i])
                write_ply(out_base.with_suffix(".ply"), pred_np[i])


def safe_case_name(case_path: Path, data_dir: Path) -> str:
    try:
        rel = case_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        rel = case_path
    return "__".join(rel.parts)


if __name__ == "__main__":
    main()
