#!/usr/bin/env bash
set -euo pipefail

cd /home/yhpang/maketooth
PYTHON=/home/yhpang/miniconda3/envs/ensemble/bin/python
GPU="${GPU:-6}"
DATE="${DATE:-20260729}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START_STAGE="${START_STAGE:-1}"
TRAIN_FREE_MB="${TRAIN_FREE_MB:-6000}"
EVAL_FREE_MB="${EVAL_FREE_MB:-4000}"
COMMON=(
  --data-dir data
  --split-file splits/m0_patient_split_seed20260706.json
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps 16
  --init-checkpoint runs/m2_mla_dmc_dpsr128_grid100/best.pt
  --freeze-batch-norm
  --num-workers 2
  --chamfer-points 4096
  --decoder dmc_dpsr_m2
  --grid-weight 100
  --margin-anchor-weight 0.5
  --device cuda
)

wait_for_gpu() {
  required_mb="${1:-14000}"
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    if [ "$free_mb" -ge "$required_mb" ]; then
      return
    fi
    echo "[$(date '+%F %T')] GPU $GPU free=${free_mb}MB; waiting for ${required_mb}MB"
    sleep 60
  done
}

run_training() {
  name="$1"
  shift
  wait_for_gpu "$TRAIN_FREE_MB"
  mkdir -p "runs/$name"
  echo "[$(date '+%F %T')] starting $name on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/train_m0.py \
    "${COMMON[@]}" \
    --output-dir "runs/$name" \
    "$@" > "runs/$name/train.log" 2>&1
}

run_evaluation() {
  name="$1"
  wait_for_gpu "$EVAL_FREE_MB"
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

if [ "$START_STAGE" -le 1 ]; then
  run_training e1_m2_margin_zero \
    --margin-zero-weight 5
  run_evaluation e1_m2_margin_zero
fi

if [ "$START_STAGE" -le 2 ]; then
  run_training e2_m2_detail \
    --margin-zero-weight 5 \
    --dpsr-sigma 1 \
    --narrow-band-weight 10 \
    --multiscale-grid-weight 10 \
    --grid-gradient-weight 1
  run_evaluation e2_m2_detail
fi

if [ "$START_STAGE" -le 3 ]; then
  run_training e3_m2_topology \
    --margin-zero-weight 5 \
    --topology-weight 0.01 \
    --topology-resolution 32 \
    --topology-temperature 0.05
  run_evaluation e3_m2_topology
fi

if [ "$START_STAGE" -le 4 ]; then
  run_training e4_m2_combined \
    --margin-zero-weight 5 \
    --dpsr-sigma 1 \
    --narrow-band-weight 10 \
    --multiscale-grid-weight 10 \
    --grid-gradient-weight 1 \
    --topology-weight 0.01 \
    --topology-resolution 32 \
    --topology-temperature 0.05
  run_evaluation e4_m2_combined
fi

echo "[$(date '+%F %T')] all DPSR improvement experiments completed"
