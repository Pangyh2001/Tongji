from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .sampling import ensure_2d_float32, resample_rows


TOOTH_IDS = [14, 15, 16, 17, 24, 25, 26, 27, 34, 35, 36, 37, 44, 45, 46, 47]
TOOTH_TO_INDEX = {tooth: i for i, tooth in enumerate(TOOTH_IDS)}
ARCH_TO_INDEX = {"upper": 0, "lower": 1, "unknown": 2}


@dataclass(frozen=True)
class CaseRecord:
    train_dir: Path
    patient_id: str
    tooth_id: int
    prep_arch: str
    antagonist_arch: str


def discover_cases(data_dir: Path) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for metadata_path in sorted(data_dir.rglob("train/metadata.json")):
        train_dir = metadata_path.parent
        required = ["prep_points.npy", "antagonist_points.npy", "crown_points.npy"]
        if not all((train_dir / name).exists() for name in required):
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        case_dir = train_dir.parent
        tooth_raw = meta.get("tooth_id") or _infer_tooth_id(case_dir)
        tooth_id = int(tooth_raw)
        role = meta.get("role_assignment", {})
        records.append(
            CaseRecord(
                train_dir=train_dir,
                patient_id=_infer_patient_id(case_dir, meta),
                tooth_id=tooth_id,
                prep_arch=str(role.get("prep_arch", "unknown")).lower(),
                antagonist_arch=str(role.get("antagonist_arch", "unknown")).lower(),
            )
        )
    return records


def split_by_patient(
    records: list[CaseRecord],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[CaseRecord], list[CaseRecord], list[CaseRecord]]:
    rng = np.random.default_rng(seed)
    patients = sorted({r.patient_id for r in records})
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, int(round(n * test_fraction))) if test_fraction > 0 else 0
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test : n_test + n_val])

    train, val, test = [], [], []
    for record in records:
        if record.patient_id in test_patients:
            test.append(record)
        elif record.patient_id in val_patients:
            val.append(record)
        else:
            train.append(record)
    return train, val, test


class CrownDataset(Dataset):
    def __init__(
        self,
        records: list[CaseRecord],
        *,
        prep_points: int = 8192,
        antagonist_points: int = 8192,
        crown_points: int = 16384,
        seed: int = 20260621,
    ) -> None:
        self.records = records
        self.prep_points = prep_points
        self.antagonist_points = antagonist_points
        self.crown_points = crown_points
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        rng = np.random.default_rng(self.seed + index)

        prep = ensure_2d_float32(np.load(record.train_dir / "prep_points.npy"), 6, "prep_points")
        antagonist = ensure_2d_float32(np.load(record.train_dir / "antagonist_points.npy"), 6, "antagonist_points")
        crown = ensure_2d_float32(np.load(record.train_dir / "crown_points.npy"), 6, "crown_points")

        prep = resample_rows(prep, self.prep_points, rng)
        antagonist = resample_rows(antagonist, self.antagonist_points, rng)
        crown = resample_rows(crown, self.crown_points, rng)

        tooth_index = TOOTH_TO_INDEX.get(record.tooth_id)
        if tooth_index is None:
            raise ValueError(f"Unsupported tooth id {record.tooth_id} in {record.train_dir}")

        return {
            "prep": torch.from_numpy(prep),
            "antagonist": torch.from_numpy(antagonist),
            "crown": torch.from_numpy(crown),
            "tooth_index": torch.tensor(tooth_index, dtype=torch.long),
            "prep_arch_index": torch.tensor(ARCH_TO_INDEX.get(record.prep_arch, 2), dtype=torch.long),
            "case_id": str(record.train_dir.parent),
        }


def _infer_tooth_id(case_dir: Path) -> int:
    if case_dir.name.isdigit():
        return int(case_dir.name)
    match = re.search(r"_(\d{2})$", case_dir.name)
    if not match:
        raise ValueError(f"Cannot infer tooth id from {case_dir}")
    return int(match.group(1))


def _infer_patient_id(case_dir: Path, meta: dict) -> str:
    if case_dir.name.isdigit():
        return case_dir.parent.name
    case_name = str(meta.get("case") or case_dir.name)
    match = re.match(r"(.+)_\d{2}$", case_name)
    return match.group(1) if match else case_name

