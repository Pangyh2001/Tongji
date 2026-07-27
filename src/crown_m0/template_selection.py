from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TemplateEntry:
    template: Path
    preview_stl: Path
    source_case: str
    tooth_id: int
    prep_arch: str
    vertices: int
    faces: int
    feature: np.ndarray


def case_feature(case_path: Path) -> np.ndarray:
    prep = np.load(case_path / "train" / "prep_points.npy").astype(np.float32)[:, :3]
    bbox = prep.max(axis=0) - prep.min(axis=0)
    std = prep.std(axis=0)
    return np.concatenate([bbox, std]).astype(np.float32)


def load_template_index(path: Path) -> list[TemplateEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    entries = []
    for item in payload["templates"]:
        entries.append(
            TemplateEntry(
                template=(base / item["template"]).resolve(),
                preview_stl=(base / item["preview_stl"]).resolve(),
                source_case=str(item["source_case"]),
                tooth_id=int(item["tooth_id"]),
                prep_arch=str(item["prep_arch"]),
                vertices=int(item["vertices"]),
                faces=int(item["faces"]),
                feature=np.asarray(item["feature"], dtype=np.float32),
            )
        )
    if not entries:
        raise ValueError(f"No templates in {path}")
    vertex_counts = {entry.vertices for entry in entries}
    if len(vertex_counts) != 1:
        raise ValueError(f"Template library must use one vertex count, got {sorted(vertex_counts)}")
    return entries


def select_template(
    entries: list[TemplateEntry],
    *,
    tooth_id: int,
    prep_arch: str,
    feature: np.ndarray,
) -> TemplateEntry:
    candidates = [entry for entry in entries if entry.tooth_id == tooth_id and entry.prep_arch == prep_arch]
    if not candidates:
        candidates = [entry for entry in entries if entry.tooth_id == tooth_id]
    if not candidates:
        candidates = [entry for entry in entries if entry.prep_arch == prep_arch]
    if not candidates:
        candidates = entries

    scale = np.maximum(np.std(np.stack([entry.feature for entry in entries], axis=0), axis=0), 1e-3)
    feature = np.asarray(feature, dtype=np.float32)
    return min(candidates, key=lambda entry: float(np.linalg.norm((entry.feature - feature) / scale)))

