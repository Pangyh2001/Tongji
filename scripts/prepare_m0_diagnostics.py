from __future__ import annotations

import csv
import json
import re
from pathlib import Path


DATA_DIR = Path("data")
SPLIT_FILE = Path("splits/m0_patient_split_seed20260706.json")


def main() -> None:
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    train_cases = set(split["train"])
    test_cases = set(split["test"])

    records = []
    for meta_path in sorted(DATA_DIR.rglob("train/metadata.json")):
        case_dir = meta_path.parent.parent
        case = str(case_dir)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tooth_id = str(meta.get("tooth_id") or infer_tooth_id(case_dir))
        prep_arch = str(meta.get("role_assignment", {}).get("prep_arch", "unknown")).lower()
        expected_arch = "upper" if int(tooth_id) < 30 else "lower"
        records.append(
            {
                "case": case,
                "tooth_id": tooth_id,
                "prep_arch": prep_arch,
                "expected_arch": expected_arch,
                "arch_matches": prep_arch == expected_arch,
                "split": "train" if case in train_cases else "test" if case in test_cases else "val",
            }
        )

    overfit_46 = [
        r["case"]
        for r in records
        if r["tooth_id"] == "46" and r["split"] == "train" and r["arch_matches"]
    ][:20]
    if len(overfit_46) < 10:
        raise SystemExit(f"Only found {len(overfit_46)} clean train cases for tooth 46")

    overfit_payload = {
        "description": "Tooth 46 clean train cases for M0-simple overfit diagnostic.",
        "train": overfit_46,
        "val": overfit_46,
        "test": overfit_46,
    }
    Path("splits").mkdir(exist_ok=True)
    Path("splits/m0_overfit_46_20.json").write_text(
        json.dumps(overfit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path("splits/m0_overfit_46_20.txt").write_text("\n".join(overfit_46) + "\n", encoding="utf-8")

    metrics_path = Path("visualizations/m0_new_test/metrics.csv")
    rows = []
    with metrics_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ok") == "True":
                row["pred_to_gt_rms"] = float(row["pred_to_gt_rms"])
                rows.append(row)
    rows.sort(key=lambda r: r["pred_to_gt_rms"])
    selected = []
    seen = set()
    for row in rows[-5:] + rows[len(rows) // 2 - 1 : len(rows) // 2 + 1] + rows[:3]:
        if row["case"] not in seen:
            selected.append(row["case"])
            seen.add(row["case"])
    Path("splits/m0_gt_reconstruction_representative10.txt").write_text(
        "\n".join(selected) + "\n",
        encoding="utf-8",
    )

    summary = {
        "records": len(records),
        "overfit_46_cases": len(overfit_46),
        "overfit_46_split": "splits/m0_overfit_46_20.json",
        "gt_reconstruction_representative_cases": len(selected),
        "gt_reconstruction_representative_list": "splits/m0_gt_reconstruction_representative10.txt",
        "arch_mismatches": [r for r in records if not r["arch_matches"]],
    }
    Path("splits/m0_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def infer_tooth_id(case_dir: Path) -> str:
    if case_dir.name.isdigit():
        return case_dir.name
    match = re.search(r"_(\d{2})$", case_dir.name)
    if not match:
        raise ValueError(f"Cannot infer tooth id from {case_dir}")
    return match.group(1)


if __name__ == "__main__":
    main()
