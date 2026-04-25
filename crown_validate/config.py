from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "prep_points": 2048,
    "opposing_points": 2048,
    "crown_points": 2048,
    "margin_points": 256,
    "template": {
        "num_latitude_rings": 31,
        "num_longitude_segments": 32,
        "theta_max_radians": 2.65,
    },
    "model": {
        "hidden_dim": 128,
        "latent_dim": 256,
        "dropout": 0.1,
        "max_offset": 1.35,
        "encoder_knn": 16,
        "decoder_knn": 16,
    },
    "training": {
        "epochs": 180,
        "batch_size": 2,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "patience": 25,
        "grad_clip_norm": 1.0,
        "num_workers": 0,
    },
    "losses": {
        "baseline": {
            "use_margin_weighting": False,
            "margin_alpha": 2.5,
            "margin_sigma": 0.08,
            "curvature_weight": 0.0,
            "normal_weight": 0.0,
            "roughness_knn": 8,
        },
        "improved": {
            "use_margin_weighting": True,
            "margin_alpha": 2.5,
            "margin_sigma": 0.08,
            "curvature_weight": 0.25,
            "normal_weight": 0.15,
            "roughness_knn": 8,
        },
    },
    "metrics": {
        "fscore_threshold_mm": 0.3,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    return _deep_update(DEFAULT_CONFIG, user_config)
