from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    src = Path("visualizations/m0_stl_representative10")
    out = Path("visualizations/m3_implicit_poisson_representative10")
    if out.exists():
        shutil.rmtree(out)
    (out / "m3_implicit_poisson").mkdir(parents=True)
    (out / "gt_stl").mkdir(parents=True)

    rows = []
    with (src / "metrics_by_case.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") != "poisson_clean_taubin" or row.get("ok") != "True":
                continue
            stem = row["stem"]
            case_dir = out / "m3_implicit_poisson" / stem
            case_dir.mkdir(parents=True, exist_ok=True)
            pred_src = Path(row["stl"])
            gt_src = Path(row["gt_stl"])
            pred_dst = case_dir / f"{stem}_m3_implicit_poisson.stl"
            gt_dst = out / "gt_stl" / gt_src.name
            shutil.copy2(pred_src, pred_dst)
            if not gt_dst.exists():
                shutil.copy2(gt_src, gt_dst)
            row["method"] = "m3_implicit_poisson"
            row["stl"] = str(pred_dst)
            row["gt_stl"] = str(gt_dst)
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with (out / "metrics_by_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "pred_to_gt_rms",
        "gt_to_pred_rms",
        "symmetric_rms",
        "pred_to_gt_hd95",
        "gt_to_pred_hd95",
        "triangles",
        "surface_area",
    ]
    summary = []
    if rows:
        item = {"method": "m3_implicit_poisson", "n": len(rows)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in rows if row.get(key) not in ("", None)])
            if values.size:
                item[f"{key}_mean"] = float(values.mean())
                item[f"{key}_median"] = float(np.median(values))
                item[f"{key}_max"] = float(values.max())
        summary = [item]

    (out / "summary_by_method.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({key for row in summary for key in row})
    fieldnames = ["method", "n"] + [key for key in keys if key not in ("method", "n")]
    with (out / "summary_by_method.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
