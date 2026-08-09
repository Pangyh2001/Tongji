from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crown_m0.model import M0DMCDPSRNet


def run_variant(detail_decoder: str) -> None:
    model = M0DMCDPSRNet(
        model_dim=64,
        context_tokens_per_input=16,
        num_queries=8,
        fold_step=4,
        transformer_layers=1,
        dpsr_resolution=16,
        dpsr_sigma=1.0,
        use_margin=True,
        margin_anchor_queries=2,
        detail_decoder=detail_decoder,
        spd_parent_step=2,
        spd_factor=4,
        spd_neighbors=4,
    )
    prep = torch.randn(1, 32, 6)
    antagonist = torch.randn(1, 32, 6)
    margin = torch.randn(1, 16, 3)
    output = model(
        prep,
        antagonist,
        torch.tensor([2]),
        torch.tensor([1]),
        margin=margin,
        return_grid=True,
    )
    assert output["points"].shape == (1, 128, 6)
    assert output["psr_grid"].shape == (1, 16, 16, 16)
    assert torch.isfinite(output["points"]).all()
    assert torch.isfinite(output["psr_grid"]).all()
    output["points"].square().mean().backward()
    assert model.detail_upsampler.offset_head.weight.grad is not None


if __name__ == "__main__":
    run_variant("spd")
    run_variant("spd_skip")
    print("E6 progressive decoder smoke test passed")
