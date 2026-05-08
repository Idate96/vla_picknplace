#!/usr/bin/env bash
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

CAMERA="${1:?usage: sbatch train_and_eval.sbatch <angle|top_wrist>}"
if [ "$CAMERA" != "angle" ] && [ "$CAMERA" != "top_wrist" ]; then
  echo "Unsupported camera '$CAMERA'. Expected angle or top_wrist."
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/cluster/work/rsl/$USER/hw3_camera_ablation}"
SRC="${SRC:-$PROJECT_ROOT/src}"
LERO="${LERO:-$PROJECT_ROOT/lerobot}"
DATA_ROOT="${DATA_ROOT:-/cluster/scratch/$USER/hw3_camera_ablation/datasets_rand_goal_h6/$CAMERA}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/rand_goal_h6/$CAMERA}"
VENV="${VENV:-/cluster/project/rsl/$USER/hw3_camera_ablation/.venv}"
UV="${UV:-/cluster/project/rsl/$USER/bin/uv}"
REPO_ID="${REPO_ID:-local/hw3-act-$CAMERA-randgoal-h6}"

CUBE_POS_STD="${CUBE_POS_STD:-0.006}"
GOAL_POS_STD="${GOAL_POS_STD:-0.006}"
EPISODES="${EPISODES:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-96}"
MAX_STEPS="${MAX_STEPS:-800}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
TEMPORAL_ENSEMBLE_COEFF="${TEMPORAL_ENSEMBLE_COEFF:-0.01}"

export HF_HOME="$TMPDIR/hf_home"
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"
export HF_HUB_CACHE="$TMPDIR/hf_hub"
export WANDB_DIR="$TMPDIR/wandb"
export UV_CACHE_DIR="/cluster/project/rsl/$USER/hw3_camera_ablation/uv-cache"
export MUJOCO_GL=egl
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_ROOT/logs" "$(dirname "$DATA_ROOT")" "$(dirname "$RUN_ROOT")" "$(dirname "$VENV")"

echo "Job ID: $SLURM_JOB_ID"
echo "Camera: $CAMERA"
echo "Source: $SRC"
echo "Dataset: $DATA_ROOT"
echo "Run: $RUN_ROOT"
echo "Started: $(date)"
nvidia-smi

if [ ! -x "$UV" ]; then
  echo "Missing uv at $UV"
  exit 3
fi

(
  flock 9
  if [ ! -x "$VENV/bin/lerobot-train" ]; then
    "$UV" venv --python 3.12 "$VENV"
    "$UV" pip install --python "$VENV/bin/python" -e "$SRC" -e "$LERO[training]"
  fi
) 9>"/cluster/project/rsl/$USER/hw3_camera_ablation/venv.lock"

if [ ! -f "$DATA_ROOT/meta/info.json" ]; then
  "$VENV/bin/python" "$SRC/scripts/export_expert_lerobot.py" \
    --repo-id="$REPO_ID" \
    --root="$DATA_ROOT" \
    --camera="$CAMERA" \
    --episodes="$EPISODES" \
    --seed=42 \
    --image-size="$IMAGE_SIZE" \
    --max-steps="$MAX_STEPS" \
    --obstacle-mode=train \
    --cube-pos-std="$CUBE_POS_STD" \
    --goal-pos-std="$GOAL_POS_STD" \
    --stationary-keep-head=3 \
    --stationary-keep-tail=1 \
    --overwrite \
    --use-videos \
    --vcodec=h264
fi

"$VENV/bin/python" - <<PY
import json
from pathlib import Path

summary = json.loads((Path("$DATA_ROOT") / "export_summary.json").read_text())
assert summary["camera"] == "$CAMERA"
assert summary["saved_episodes"] == int("$EPISODES")
assert summary["successes"] == int("$EPISODES")
assert summary["cube_pos_std"] == float("$CUBE_POS_STD")
assert summary["goal_pos_std"] == float("$GOAL_POS_STD")
assert summary["stationary_keep_head"] == 3
assert summary["stationary_keep_tail"] == 1
print("dataset_summary", {
    "camera": summary["camera"],
    "saved_episodes": summary["saved_episodes"],
    "attempts": summary["attempts"],
    "discarded_failures": summary["discarded_failures"],
    "attempt_success_rate": summary["attempt_success_rate"],
})
PY

if [ -f "$RUN_ROOT/checkpoints/last/pretrained_model/train_config.json" ]; then
  TRAIN_ARGS=(
    --config_path="$RUN_ROOT/checkpoints/last/pretrained_model/train_config.json"
    --resume=true
  )
else
  if [ -d "$RUN_ROOT" ] && [ -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 ! -name '*.json' -print -quit)" ]; then
    echo "Refusing to overwrite non-empty output dir without a resumable checkpoint: $RUN_ROOT"
    exit 4
  fi
  TRAIN_ARGS=(
    --dataset.repo_id="$REPO_ID"
    --dataset.root="$DATA_ROOT"
    --dataset.video_backend=pyav
    --policy.type=act
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.chunk_size="$CHUNK_SIZE"
    --policy.n_action_steps="$N_ACTION_STEPS"
    --policy.temporal_ensemble_coeff="$TEMPORAL_ENSEMBLE_COEFF"
    --wandb.enable=false
    --output_dir="$RUN_ROOT"
    --job_name="act_${CAMERA}_randgoal_h6"
    --steps="$TRAIN_STEPS"
    --save_freq="$SAVE_FREQ"
    --log_freq=1000
    --num_workers=4
    --batch_size=8
  )
fi

"$VENV/bin/lerobot-train" "${TRAIN_ARGS[@]}"

CKPT="$RUN_ROOT/checkpoints/last/pretrained_model"
for MODE in train adversarial; do
  "$VENV/bin/python" "$SRC/scripts/eval_lerobot_act.py" \
    --checkpoint="$CKPT" \
    --dataset-root="$DATA_ROOT" \
    --repo-id="$REPO_ID" \
    --camera="$CAMERA" \
    --episodes=100 \
    --seed=1042 \
    --image-size="$IMAGE_SIZE" \
    --max-steps="$MAX_STEPS" \
    --obstacle-mode="$MODE" \
    --cube-pos-std="$CUBE_POS_STD" \
    --goal-pos-std="$GOAL_POS_STD" \
    --output="$RUN_ROOT/eval_${MODE}_seed1042.json" \
    --device=cuda \
    --n-action-steps="$N_ACTION_STEPS"
done

echo "Finished: $(date)"
