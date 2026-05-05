#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-carmensc/record-test-screwdriver}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_screwdriver}"
JOB_NAME="${JOB_NAME:-act_screwdriver}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-lerobot-train}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

if ! command -v "$LEROBOT_TRAIN" >/dev/null 2>&1; then
  echo "Missing '$LEROBOT_TRAIN'. Activate your environment or set LEROBOT_TRAIN=/path/to/lerobot-train." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

args=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.revision="$DATASET_REVISION"
  --dataset.video_backend="$VIDEO_BACKEND"
  --policy.type=act
  --policy.device="$DEVICE"
  --policy.push_to_hub=false
  --wandb.enable="$WANDB_ENABLE"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --num_workers="$NUM_WORKERS"
  --batch_size="$BATCH_SIZE"
)

if [ -n "$DATASET_ROOT" ]; then
  args+=(--dataset.root="$DATASET_ROOT")
fi

exec "$LEROBOT_TRAIN" "${args[@]}" "$@"
