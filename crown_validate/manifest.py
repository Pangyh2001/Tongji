from __future__ import annotations

import csv
import re
from pathlib import Path

from .utils import as_bool, ensure_dir


CASE_FIELDS = [
    "case_id",
    "patient_name",
    "tooth_position",
    "prep_path",
    "opposing_path",
    "crown_path",
    "include",
    "needs_manual_confirmation",
    "notes",
]


def _extract_tooth_position(text: str) -> str:
    matches = re.findall(r"(?<!\d)(1[1-7]|2[1-7]|3[1-7]|4[1-7])(?!\d)", text)
    return matches[0] if matches else ""


def _pick_files(case_dir: Path) -> dict[str, str]:
    files = sorted([path for path in case_dir.iterdir() if path.is_file() and path.suffix.lower() == ".stl"])
    crown = ""
    opposing = ""
    prep = ""
    notes: list[str] = []

    crown_candidates = [path for path in files if any(token in path.name.lower() for token in ["crown", "牙冠", "coping"])]
    if crown_candidates:
        crown = str(crown_candidates[0])

    opposing_candidates = [
        path
        for path in files
        if any(token in path.name.lower() for token in ["对颌", "对合", "upperjaw"])
    ]
    if opposing_candidates:
        opposing = str(opposing_candidates[0])

    remaining = [path for path in files if str(path) not in {crown, opposing}]
    prep_candidates = [
        path
        for path in remaining
        if any(token in path.name.lower() for token in ["工作模", "lowerjaw"])
    ]
    if prep_candidates:
        prep = str(prep_candidates[0])
    elif remaining:
        prep = str(remaining[0])

    if case_dir.name == "吴丰荷":
        notes.append("根据 tooth_position=45 和文件名推断 LowerJaw 为预备体所在颌，建议人工复核。")
    if not crown or not opposing or not prep:
        notes.append("存在文件角色未自动识别的风险，请人工确认。")

    return {
        "prep_path": prep,
        "opposing_path": opposing,
        "crown_path": crown,
        "notes": " ".join(notes),
    }


def build_manifest(raw_dir: str | Path, output_csv: str | Path) -> Path:
    source_dir = Path(raw_dir)
    output_path = Path(output_csv)
    ensure_dir(output_path.parent)

    rows: list[dict[str, str]] = []
    case_dirs = sorted([path for path in source_dir.iterdir() if path.is_dir()])
    for case_index, case_dir in enumerate(case_dirs, start=1):
        picks = _pick_files(case_dir)
        tooth_position = _extract_tooth_position(Path(picks["crown_path"]).name if picks["crown_path"] else "")
        rows.append(
            {
                "case_id": f"case_{case_index:03d}",
                "patient_name": case_dir.name,
                "tooth_position": tooth_position,
                "prep_path": picks["prep_path"],
                "opposing_path": picks["opposing_path"],
                "crown_path": picks["crown_path"],
                "include": "true",
                "needs_manual_confirmation": "true" if case_dir.name == "吴丰荷" else "false",
                "notes": picks["notes"],
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def load_manifest(path: str | Path, include_unconfirmed: bool = True) -> list[dict[str, str]]:
    manifest_path = Path(path)
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not as_bool(row.get("include", "true")):
                continue
            if not include_unconfirmed and as_bool(row.get("needs_manual_confirmation", "false")):
                continue
            rows.append(dict(row))
    return rows
