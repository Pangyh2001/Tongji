#!/usr/bin/env bash
set -euo pipefail

cd /home/yhpang/maketooth

PY=/home/yhpang/miniconda3/envs/ensemble/bin/python
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=visualizations/m0_stl_reconstruction_logs
mkdir -p "$LOG_DIR"

CASE_LIST=splits/m0_stl_representative_10.txt

"$PY" - <<'PY'
import csv
from pathlib import Path

metrics = Path("visualizations/m0_new_test/metrics.csv")
rows = []
with metrics.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("ok") == "True":
            row["pred_to_gt_rms"] = float(row["pred_to_gt_rms"])
            rows.append(row)

rows.sort(key=lambda r: r["pred_to_gt_rms"])
best = rows[:3]
worst = rows[-5:]
mid_start = max(0, len(rows) // 2 - 1)
middle = rows[mid_start : mid_start + 2]

selected = []
seen = set()
for row in worst + middle + best:
    case = row["case"]
    if case not in seen:
        selected.append(case)
        seen.add(case)

out = Path("splits/m0_stl_representative_10.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(selected)} cases")
for case in selected:
    print(case)
PY

echo "[$(date)] Stage 1/2: full test official alpha_clean STL"
"$PY" scripts/run_stl_reconstruction_experiments.py \
  --methods alpha_clean \
  --output-dir visualizations/m0_stl_alpha70

echo "[$(date)] Stage 2/2: representative 10 official alpha_clean STL"
"$PY" scripts/run_stl_reconstruction_experiments.py \
  --case-list "$CASE_LIST" \
  --methods alpha_clean \
  --output-dir visualizations/m0_stl_representative10

echo "[$(date)] done" | tee "$LOG_DIR/done_${RUN_ID}.txt"
