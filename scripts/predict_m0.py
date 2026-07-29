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
from src.crown_m0.model import (
    M0CoarseToFineNet,
    M0CrownNet,
    M0DMCDPSRNet,
    M0TangentCoarseToFineNet,
)


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
    checkpoint_args = checkpoint.get("args", {})
    decoder = str(checkpoint_args.get("decoder", "direct"))
    if decoder.startswith("dmc_dpsr"):
        use_margin = decoder != "dmc_dpsr"
        use_anchor = decoder in {"dmc_dpsr_m2", "dmc_dpsr_m3"}
        model = M0DMCDPSRNet(
            model_dim=int(checkpoint_args.get("dmc_model_dim", 256)),
            context_tokens_per_input=int(checkpoint_args.get("dmc_context_tokens", 256)),
            num_queries=int(checkpoint_args.get("dmc_queries", 256)),
            fold_step=int(checkpoint_args.get("dmc_fold_step", 8)),
            transformer_layers=int(checkpoint_args.get("dmc_transformer_layers", 3)),
            dpsr_resolution=int(checkpoint_args.get("dpsr_resolution", 128)),
            dpsr_sigma=float(checkpoint_args.get("dpsr_sigma", 2.0)),
            roi_half_extent_mm=float(checkpoint_args.get("roi_half_extent_mm", 12.0)),
            use_margin=use_margin,
            margin_anchor_queries=(
                int(checkpoint_args.get("margin_anchor_queries", 64)) if use_anchor else 0
            ),
            ring_groups=int(checkpoint_args.get("ring_groups", 6)) if use_anchor else 0,
        ).to(args.device)
    elif decoder == "coarse_to_fine_tangent":
        model = M0TangentCoarseToFineNet(
            coarse_points=int(checkpoint_args.get("coarse_points", 8192)),
            first_factor=int(checkpoint_args.get("first_factor", 4)),
            second_factor=int(checkpoint_args.get("second_factor", 2)),
        ).to(args.device)
    elif checkpoint_args.get("decoder") == "coarse_to_fine":
        model = M0CoarseToFineNet(
            coarse_points=int(checkpoint_args.get("coarse_points", 8192)),
            first_factor=int(checkpoint_args.get("first_factor", 4)),
            second_factor=int(checkpoint_args.get("second_factor", 2)),
        ).to(args.device)
    else:
        model = M0CrownNet(output_points=int(checkpoint_args.get("output_points", 16384))).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        for batch in loader:
            model_kwargs = {}
            if isinstance(model, M0DMCDPSRNet) and model.use_margin:
                model_kwargs["margin"] = batch["margin"].to(args.device)
            pred = model(
                batch["prep"].to(args.device),
                batch["antagonist"].to(args.device),
                batch["tooth_index"].to(args.device),
                batch["prep_arch_index"].to(args.device),
                **model_kwargs,
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
