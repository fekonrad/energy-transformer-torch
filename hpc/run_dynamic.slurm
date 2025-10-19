#!/bin/bash
#SBATCH --job-name=et-dynamic
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

# --- User config (edit as needed) ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="et"                 # your conda env
PYTHON_BIN="python"            # path to python if not in env

# Training args
RESULT_DIR="results_dynamic/run_alpha_trainable"
PATCH_SIZE=4
TKN_DIM=256
QK_DIM=64
NHEADS=12
HN_MULT=4.0
ATTN_BETA=0.125
ATTN_BIAS=False
HN_BIAS=False
TIME_STEPS=12
BLOCKS=1
ALPHA_INIT=1.0                 # initial value for trainable per-block alphas
EPOCHS=100
BATCH_SIZE=128
LR=8e-5
B1=0.9
B2=0.999
MASK_RATIO=0.85
WEIGHT_DECAY=0.0001
DATA_PATH="./"                 # point to shared CIFAR location on cluster
DATA_NAME="cifar10"            # or cifar100

# --- Environment setup (modules / conda) ---
module purge || true
# module load cuda/12.1            # uncomment/adjust for your cluster
# module load anaconda/2024.06     # uncomment/adjust for your cluster

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || true
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

$PYTHON_BIN train_dynamic.py \
  --patch-size "$PATCH_SIZE" \
  --tkn-dim "$TKN_DIM" \
  --qk-dim "$QK_DIM" \
  --nheads "$NHEADS" \
  --hn-mult "$HN_MULT" \
  --attn-beta "$ATTN_BETA" \
  --attn-bias "$ATTN_BIAS" \
  --hn-bias "$HN_BIAS" \
  --time-steps "$TIME_STEPS" \
  --alpha "$ALPHA_INIT" \
  --blocks "$BLOCKS" \
  --epochs "$EPOCHS" \
  --result-dir "$RESULT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --b1 "$B1" \
  --b2 "$B2" \
  --mask-ratio "$MASK_RATIO" \
  --weight-decay "$WEIGHT_DECAY" \
  --data-path "$DATA_PATH" \
  --data-name "$DATA_NAME"

