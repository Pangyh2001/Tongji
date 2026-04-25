from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CrownCaseDataset(Dataset):
    def __init__(self, processed_dir: str | Path, case_ids: list[str]) -> None:
        self.processed_dir = Path(processed_dir)
        self.case_ids = list(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_id = self.case_ids[index]
        payload = np.load(self.processed_dir / case_id / "processed_case.npz")
        return {
            "case_id": case_id,
            "prep_points": torch.from_numpy(payload["prep_points"]).float(),
            "opposing_points": torch.from_numpy(payload["opposing_points"]).float(),
            "crown_points": torch.from_numpy(payload["crown_points"]).float(),
            "crown_normals": torch.from_numpy(payload["crown_normals"]).float(),
            "margin_points": torch.from_numpy(payload["margin_points"]).float(),
            "center": torch.from_numpy(payload["center"]).float(),
            "scale": torch.from_numpy(payload["scale"]).float(),
        }


def collate_cases(items: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "case_id": [item["case_id"] for item in items],
        "prep_points": torch.stack([item["prep_points"] for item in items], dim=0),
        "opposing_points": torch.stack([item["opposing_points"] for item in items], dim=0),
        "crown_points": torch.stack([item["crown_points"] for item in items], dim=0),
        "crown_normals": torch.stack([item["crown_normals"] for item in items], dim=0),
        "margin_points": torch.stack([item["margin_points"] for item in items], dim=0),
        "center": torch.stack([item["center"] for item in items], dim=0),
        "scale": torch.stack([item["scale"] for item in items], dim=0),
    }
