#!/usr/bin/env bash
#SBATCH --partition=gpu.24h
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CAMERA="${1:?usage: sbatch eval_checkpoint.sbatch <angle|top_wrist> <checkpoint_step>}"
CKPT_STEP="${2:?usage: sbatch eval_checkpoint.sbatch <angle|top_wrist> <checkpoint_step>}"
if [ "$CAMERA" != "angle" ] && [ "$CAMERA" != "top_wrist" ]; then
  echo "Unsupported camera '$CAMERA'. Expected angle or top_wrist."
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/cluster/work/rsl/$USER/hw3_camera_ablation}"
SRC="${SRC:-$PROJECT_ROOT/src}"
DATA_ROOT="${DATA_ROOT:-/cluster/scratch/$USER/hw3_camera_ablation/datasets_rand_goal_h6/$CAMERA}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/rand_goal_h6/$CAMERA}"
REPO_ID="${REPO_ID:-local/hw3-act-$CAMERA-randgoal-h6}"
VENV="${VENV:-/cluster/project/rsl/$USER/hw3_camera_ablation/.venv}"

CUBE_POS_STD="${CUBE_POS_STD:-0.006}"
GOAL_POS_STD="${GOAL_POS_STD:-0.006}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-1042}"
IMAGE_SIZE="${IMAGE_SIZE:-96}"
MAX_STEPS="${MAX_STEPS:-800}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-3600}"

export HF_HOME="$TMPDIR/hf_home"
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"
export HF_HUB_CACHE="$TMPDIR/hf_hub"
export WANDB_DIR="$TMPDIR/wandb"
export MUJOCO_GL=egl
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CKPT="$RUN_ROOT/checkpoints/$CKPT_STEP/pretrained_model"
deadline=$((SECONDS + WAIT_TIMEOUT_SEC))
while [ ! -d "$CKPT" ]; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "Timed out waiting for checkpoint: $CKPT"
    exit 3
  fi
  echo "Waiting for checkpoint: $CKPT"
  sleep 60
done

for MODE in train adversarial; do
  "$VENV/bin/python" "$SRC/scripts/eval_lerobot_act.py" \
    --checkpoint="$CKPT" \
    --dataset-root="$DATA_ROOT" \
    --repo-id="$REPO_ID" \
    --camera="$CAMERA" \
    --episodes="$EPISODES" \
    --seed="$SEED" \
    --image-size="$IMAGE_SIZE" \
    --max-steps="$MAX_STEPS" \
    --obstacle-mode="$MODE" \
    --cube-pos-std="$CUBE_POS_STD" \
    --goal-pos-std="$GOAL_POS_STD" \
    --output="$RUN_ROOT/eval_${MODE}_seed${SEED}_${CKPT_STEP}.json" \
    --device=cuda \
    --n-action-steps="$N_ACTION_STEPS"
done
