#!/usr/bin/env bash
#SBATCH --job-name=act_screwdriver
#SBATCH --partition=gpu.24h
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load eth_proxy || true

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
UV="${UV:-uv}"

export HF_HOME="${HF_HOME:-$TMPDIR/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$TMPDIR/hf_datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$TMPDIR/hf_hub}"
export WANDB_DIR="${WANDB_DIR:-$TMPDIR/wandb}"
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Project root: $PROJECT_ROOT"
nvidia-smi

if [ ! -x "$VENV/bin/lerobot-train" ]; then
  echo "Creating LeRobot environment at $VENV"
  "$UV" venv --python 3.12 "$VENV"
  "$UV" pip install --python "$VENV/bin/python" -e "$PROJECT_ROOT/../lerobot[training]"
  "$UV" pip install --python "$VENV/bin/python" -r "$PROJECT_ROOT/requirements.txt"
fi

cd "$PROJECT_ROOT"
mkdir -p logs

LEROBOT_TRAIN="$VENV/bin/lerobot-train" \
DEVICE=cuda \
VIDEO_BACKEND=pyav \
./act/train_act.sh
