#!/bin/bash
#SBATCH --job-name=et-alpha-ts-sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-11

set -euo pipefail

# --- User config (edit as needed) ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="et"                 # change to your env name
PYTHON_BIN="python"            # or full path to python

# Training knobs you may want to adjust once for all runs
EPOCHS=100
BATCH_SIZE=128
LR=8e-5
DATA_PATH="./"                 # uses repo-provided CIFAR by default
DATA_NAME="cifar10"            # or cifar100

# Grid to sweep
ALPHAS=(0.5 1 5 10)
TSS=(8 12 16)

# --- Environment setup (modules / conda) ---
module purge || true
# module load cuda/12.1            # uncomment and adjust for your cluster
# module load anaconda/2024.06     # uncomment and adjust for your cluster

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || true
fi

mkdir -p "$SLURM_SUBMIT_DIR/logs" || mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# --- Map array index to (alpha, time_steps) ---
NA=${#ALPHAS[@]}
NT=${#TSS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}

ai=$(( IDX % NA ))
ti=$(( IDX / NA ))

ALPHA=${ALPHAS[$ai]}
TS=${TSS[$ti]}

RESULT_DIR="results/sweep/alpha_${ALPHA}_ts_${TS}"
mkdir -p "$RESULT_DIR"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> alpha=$ALPHA, time_steps=$TS"
echo "Writing to $RESULT_DIR"

# Prefer running without accelerate CLI (single-process); Accelerator still works
$PYTHON_BIN train.py \
  --tkn-dim 128 \
  --qk-dim 64 \
  --nheads 12 \
  --hn-mult 4.0 \
  --attn-beta 0.125 \
  --attn-bias False \
  --hn-bias False \
  --time-steps "$TS" \
  --alpha "$ALPHA" \
  --blocks 1 \
  --epochs "$EPOCHS" \
  --result-dir "$RESULT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --b1 0.9 \
  --b2 0.999 \
  --mask-ratio 0.85 \
  --weight-decay 0.0001 \
  --data-path "$DATA_PATH" \
  --data-name "$DATA_NAME"
