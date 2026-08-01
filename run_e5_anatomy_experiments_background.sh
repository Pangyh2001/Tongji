#!/usr/bin/env bash
set -euo pipefail

cd /home/yhpang/maketooth
PYTHON=/home/yhpang/miniconda3/envs/ensemble/bin/python
GPU="${GPU:-6}"
DATE="${DATE:-20260801}"
EPOCHS="${EPOCHS:-20}"
TRAIN_FREE_MB="${TRAIN_FREE_MB:-6000}"

wait_for_gpu() {
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    if [ "$free_mb" -ge "$TRAIN_FREE_MB" ]; then return; fi
    echo "[$(date '+%F %T')] GPU $GPU free=${free_mb}MB; waiting"
    sleep 60
  done
}

run_one() {
  name="$1"
  curvature_lambda="$2"
  queries="$3"
  fold_step="$4"
  anchor_queries="$5"
  wait_for_gpu
  mkdir -p "runs/$name" "result/$DATE"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/train_m0.py \
    --data-dir data \
    --split-file splits/m0_patient_split_seed20260706.json \
    --output-dir "runs/$name" \
    --init-checkpoint runs/e2_m2_detail/best.pt \
    --freeze-batch-norm \
    --epochs "$EPOCHS" \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --num-workers 2 \
    --chamfer-points 4096 \
    --decoder dmc_dpsr_m2 \
    --grid-weight 100 \
    --dpsr-sigma 1 \
    --dmc-queries "$queries" \
    --dmc-fold-step "$fold_step" \
    --margin-anchor-queries "$anchor_queries" \
    --margin-anchor-weight 0.5 \
    --margin-zero-weight 5 \
    --narrow-band-weight 10 \
    --multiscale-grid-weight 10 \
    --grid-gradient-weight 1 \
    --curvature-penalty-weight 1 \
    --curvature-lambda "$curvature_lambda" \
    --curvature-points 1024 \
    --curvature-neighbors 16 \
    --device cuda > "runs/$name/train.log" 2>&1

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/run_m0_official_experiment.py \
    --checkpoint "runs/$name/best.pt" \
    --data-dir data \
    --split-file splits/m0_patient_split_seed20260706.json \
    --output-root result \
    --date "$DATE" \
    --experiment-name "$name" \
    --stl-method dmc_dpsr_marching_cubes \
    --batch-size 1 \
    --device cuda > "result/$DATE/${name}_evaluation.log" 2>&1
}

run_one e5a_cpl_lambda05 0.5 256 8 64
run_one e5b_cpl_lambda10 1.0 256 8 64
run_one e5c_cpl_lambda10_q1024f4 1.0 1024 4 128

echo "[$(date '+%F %T')] E5 anatomy experiments completed"
