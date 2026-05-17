#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-carmensc/record-test-screwdriver}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla_screwdriver}"
JOB_NAME="${JOB_NAME:-smolvla_screwdriver}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
RENAME_MAP="${RENAME_MAP:-{\"observation.images.front\":\"observation.images.camera1\"}}"
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
  --policy.path="$POLICY_PATH"
  --policy.device="$DEVICE"
  --policy.push_to_hub=false
  --rename_map="$RENAME_MAP"
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
