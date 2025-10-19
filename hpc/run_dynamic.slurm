#!/bin/bash -l
#SBATCH --job-name=et-dynamic
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=A40devel
#SBATCH --gpus=1
# SBATCH --gres=gpu:1                 # ← uncomment if your cluster uses --gres instead of --gpus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

set -euo pipefail

# --------------------------
# Paths / project
# --------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
echo "== Node: $(hostname)"
echo "== When: $(date)"
echo "== PROJECT_DIR: $PROJECT_DIR"

REQ_FILE="$PROJECT_DIR/requirements.txt"
CONDA_HOME="$HOME/miniforge3"
ENV_NAME="et"

# --------------------------
# Miniforge bootstrap (user-local, no admin)
# --------------------------
if ! [ -x "$CONDA_HOME/bin/conda" ]; then
  echo "== Miniforge not found; installing to $CONDA_HOME"
  mkdir -p "$HOME/.cache/miniforge"
  INSTALLER="$HOME/.cache/miniforge/Miniforge3-Linux-x86_64.sh"
  if ! [ -f "$INSTALLER" ]; then
    if command -v wget >/dev/null 2>&1; then
      wget -O "$INSTALLER" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    else
      curl -L -o "$INSTALLER" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    fi
  fi
  bash "$INSTALLER" -b -p "$CONDA_HOME"
fi

# Enable `conda activate` in batch shells
source "$CONDA_HOME/etc/profile.d/conda.sh"

# --------------------------
# Env create/activate (modern Python)
# --------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "== Creating conda env '$ENV_NAME' (python=3.11)"
  conda create -y -n "$ENV_NAME" python=3.11
fi

echo "== Activating env '$ENV_NAME'"
conda activate "$ENV_NAME"
PYTHON_BIN="$(command -v python)"
PIP_BIN="$(command -v pip)"
echo "== PYTHON_BIN: $PYTHON_BIN"
"$PYTHON_BIN" -V

# --------------------------
# One-pass requirements install (Option B)
# - Use CUDA wheels if a GPU is visible; else CPU wheels
# - requirements.txt MUST include torch/torchvision pins that match each other
#   (e.g., torch==2.1.2 with torchvision==0.16.2), or omit pins to let pip resolve.
# --------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"   # A40 → cu121 wheels
  echo "== GPU detected; using PyTorch index: $TORCH_INDEX_URL"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  echo "== No GPU detected; using PyTorch index: $TORCH_INDEX_URL"
fi

if [ ! -f "$REQ_FILE" ]; then
  echo "!! requirements.txt not found at $REQ_FILE"
  exit 1
fi

echo "== Upgrading pip and installing requirements..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --no-cache-dir -r "$REQ_FILE" --extra-index-url "$TORCH_INDEX_URL"

# Quick diagnostics
"$PYTHON_BIN" - <<'PY'
import sys
print("== python:", sys.version.replace("\n"," "))
try:
    import torch
    print("== torch:", torch.__version__)
    print("== cuda available:", torch.cuda.is_available())
except Exception as e:
    print("!! torch import failed:", e)
try:
    import torchvision
    print("== torchvision:", torchvision.__version__)
except Exception as e:
    print("!! torchvision import failed:", e)
PY

# --------------------------
# Training args
# --------------------------
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
ALPHA_INIT=1.0
EPOCHS=100
BATCH_SIZE=128
LR=8e-5
B1=0.9
B2=0.999
MASK_RATIO=0.85
WEIGHT_DECAY=0.0001
DATA_PATH="./"
DATA_NAME="cifar10"

echo "== Launching training..."
srun "$PYTHON_BIN" train_dynamic.py \
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

echo "== Done: $(date)"
