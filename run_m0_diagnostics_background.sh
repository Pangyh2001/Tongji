#!/usr/bin/env bash
set -euo pipefail

cd /home/yhpang/maketooth

PY=/home/yhpang/miniconda3/envs/ensemble/bin/python
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=diagnostics/logs
mkdir -p "$LOG_DIR"

echo "[$(date)] prepare diagnostic splits"
"$PY" scripts/prepare_m0_diagnostics.py

echo "[$(date)] GT reconstruction alpha70 with official alpha_clean STL"
"$PY" scripts/run_stl_reconstruction_experiments.py \
  --source gt-crown-points \
  --methods alpha_clean \
  --output-dir diagnostics/gt_reconstruction_alpha70

echo "[$(date)] GT reconstruction representative10 with official alpha_clean STL"
"$PY" scripts/run_stl_reconstruction_experiments.py \
  --source gt-crown-points \
  --case-list splits/m0_gt_reconstruction_representative10.txt \
  --methods alpha_clean \
  --output-dir diagnostics/gt_reconstruction_representative10

echo "[$(date)] M0-simple overfit diagnostic on 20 clean tooth-46 train cases"
"$PY" scripts/train_m0_overfit_diagnostic.py \
  --split-file splits/m0_overfit_46_20.json \
  --output-dir runs/m0_overfit_46_20_e300 \
  --epochs 300 \
  --batch-size 2 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --chamfer-points 2048 \
  --save-every 25

echo "[$(date)] export overfit predictions"
"$PY" scripts/predict_m0_selected.py \
  --checkpoint runs/m0_overfit_46_20_e300/best.pt \
  --split-file splits/m0_overfit_46_20.json \
  --split train \
  --output-dir predictions/m0_overfit_46_20_train

echo "[$(date)] reconstruct overfit predictions with alpha_clean"
"$PY" scripts/run_stl_reconstruction_experiments.py \
  --prediction-dir predictions/m0_overfit_46_20_train \
  --case-list splits/m0_overfit_46_20.txt \
  --methods alpha_clean \
  --output-dir diagnostics/m0_overfit_46_20_alpha_clean_stl

echo "[$(date)] done" | tee "$LOG_DIR/done_${RUN_ID}.txt"
