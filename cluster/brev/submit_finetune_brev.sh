#!/usr/bin/env bash
# Submit a guarded MolmoAct2 fine-tune command to a Brev VM over SSH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.brev"
if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: cluster/brev/.env.brev not found. Copy cluster/brev/.env.brev.template and set BREV_INSTANCE_NAME." >&2
    exit 1
fi
# shellcheck source=cluster/brev/.env.brev
# shellcheck disable=SC1091
source "${ENV_FILE}"

if [ "${CLUSTER_TYPE:-}" != "brev" ]; then
    echo "CLUSTER_TYPE must be 'brev' for this script." >&2
    exit 1
fi

required_vars=(BREV_INSTANCE_NAME BREV_CODE_DIR BREV_LOGS_DIR)
for var_name in "${required_vars[@]}"; do
    if [ -z "${!var_name:-}" ]; then
        echo "${var_name} is required in cluster/brev/.env.brev." >&2
        exit 1
    fi
done
if [ "${BREV_CODE_DIR##*/}" != "vla_picknplace" ]; then
    echo "BREV_CODE_DIR must end with /vla_picknplace." >&2
    exit 1
fi
for remote_path in "${BREV_CODE_DIR}" "${BREV_LOGS_DIR}"; do
    case "${remote_path}" in
        "" | [!/]*)
            echo "Remote paths must be absolute: ${remote_path}" >&2
            exit 1
            ;;
        *[!A-Za-z0-9_./-]*)
            echo "Remote paths must be absolute and contain only letters, numbers, _, ., /, or -: ${remote_path}" >&2
            exit 1
            ;;
    esac
done

show_help() {
    cat <<'EOF'
Usage: submit_finetune_brev.sh [options] --train-command CMD

Options:
  --dataset-repo-id ID     Dataset repo/id for local readiness gate.
  --dataset-root PATH      Optional local LeRobot dataset root for readiness.
  --gpu-list LIST          Comma-separated GPU IDs on the Brev VM (default: 0).
  --gpus N                 Use GPUs 0..N-1 if --gpu-list is not set (default: 1).
  --time TIME              GNU timeout duration, e.g. 24h or 120m (default: 24h).
  --run-id ID              Stable run id for log naming.
  --skip-sync              Do not sync code before launching.
  --dry-run                Print resolved launch plan without SSH launch.
  --allow-blocked-dry-run  With --dry-run only, print the dry-run plan even if readiness blocks.
  --follow                 Tail remote log after launch.
  --train-command CMD      Required remote command, run under ${BREV_CODE_DIR}.

Example:
  ./submit_finetune_brev.sh --dataset-repo-id <hf_user>/<dataset> \
    --train-command '.venv/bin/python -m your.official.train ...'
EOF
    exit 0
}

DATASET_REPO_ID=""
DATASET_ROOT=""
GPU_LIST=""
NUM_GPUS=1
JOB_TIME="${JOB_TIME:-24h}"
RUN_ID=""
DO_SYNC=1
DRY_RUN=0
ALLOW_BLOCKED_DRY_RUN=0
FOLLOW_LOG=0
TRAIN_COMMAND=""

if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-repo-id) DATASET_REPO_ID="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --time) JOB_TIME="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --skip-sync) DO_SYNC=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --allow-blocked-dry-run) ALLOW_BLOCKED_DRY_RUN=1; shift ;;
        --follow) FOLLOW_LOG=1; shift ;;
        --train-command) TRAIN_COMMAND="$2"; shift 2 ;;
        -h|--help) show_help ;;
        *) echo "Unknown argument: $1" >&2; show_help ;;
    esac
done

if [ -z "${TRAIN_COMMAND}" ]; then
    echo "--train-command is required. The official MolmoAct2 train command is not public yet." >&2
    exit 1
fi
if [ -z "${DATASET_REPO_ID}" ]; then
    echo "--dataset-repo-id is required." >&2
    exit 1
fi

if [ -z "${GPU_LIST}" ]; then
    if [ "${NUM_GPUS}" -le 0 ]; then
        echo "--gpus must be >= 1 when --gpu-list is not set." >&2
        exit 1
    fi
    GPU_LIST="$(seq 0 $((NUM_GPUS - 1)) | paste -sd, -)"
else
    NUM_GPUS="$(printf "%s" "${GPU_LIST}" | awk -F',' '{print NF}')"
fi
case "${GPU_LIST}" in
    *[!0-9,]* | "" | *, | ,* | *,,*)
        echo "--gpu-list must be comma-separated GPU IDs, e.g. 0 or 0,1." >&2
        exit 1
        ;;
esac
case "${JOB_TIME}" in
    *[!0-9smhd.]*)
        echo "--time must be a simple GNU timeout duration, e.g. 24h or 120m." >&2
        exit 1
        ;;
esac

RUN_ID="${RUN_ID:-$(date +%F_%H-%M-%S)_molmoact2}"
LOG_FILE="${BREV_LOGS_DIR}/brev-${RUN_ID}.log"

READINESS_CMD=("${PYTHON_BIN}" "${PROJECT_ROOT}/molmoact2/check_finetune_readiness.py" "--dataset-repo-id" "${DATASET_REPO_ID}")
if [ -n "${DATASET_ROOT}" ]; then
    READINESS_CMD+=("--dataset-root" "${DATASET_ROOT}")
fi

echo "Brev host: ${BREV_INSTANCE_NAME}"
echo "Remote code: ${BREV_CODE_DIR}"
echo "Remote log: ${LOG_FILE}"
echo "GPUs: ${GPU_LIST}"
echo "Train command: ${TRAIN_COMMAND}"

echo "Running local readiness gate..."
READINESS_STATUS=0
if "${READINESS_CMD[@]}"; then
    READINESS_STATUS=0
else
    READINESS_STATUS=$?
    if [ "${DRY_RUN}" -eq 1 ] && [ "${ALLOW_BLOCKED_DRY_RUN}" -eq 1 ]; then
        echo "Readiness gate blocked; continuing only because --allow-blocked-dry-run was set." >&2
    else
        exit "${READINESS_STATUS}"
    fi
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    if [ "${READINESS_STATUS}" -eq 0 ]; then
        echo "Dry run only; readiness passed; not syncing or launching."
    else
        echo "Dry run only; readiness blocked; not syncing or launching."
    fi
    echo "Remote command would run under ${BREV_CODE_DIR} with CUDA_VISIBLE_DEVICES=${GPU_LIST}."
    echo "Remote log would be ${LOG_FILE}."
    exit 0
fi

if [ "${DO_SYNC}" -eq 1 ]; then
    "${SCRIPT_DIR}/sync_code_brev.sh"
fi

GPU_PIDS="$(ssh "${BREV_INSTANCE_NAME}" bash -s -- "${GPU_LIST}" <<'REMOTE'
set -euo pipefail
nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader
REMOTE
)"
if [ -n "$(echo "${GPU_PIDS}" | tr -d '[:space:]')" ]; then
    echo "GPUs ${GPU_LIST} are busy (compute PIDs: ${GPU_PIDS}). Aborting." >&2
    exit 1
fi

escape_squotes() {
    printf "%s" "$1" | sed "s/'/'\"'\"'/g"
}

TRAIN_COMMAND_ESCAPED="$(escape_squotes "${TRAIN_COMMAND}")"
HF_TOKEN_ESCAPED="$(escape_squotes "${HF_TOKEN:-}")"
WANDB_API_KEY_ESCAPED="$(escape_squotes "${WANDB_API_KEY:-}")"
WANDB_USERNAME_ESCAPED="$(escape_squotes "${WANDB_USERNAME:-}")"
WANDB_MODE_ESCAPED="$(escape_squotes "${WANDB_MODE:-}")"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
mkdir -p "${BREV_LOGS_DIR}"
cd "${BREV_CODE_DIR}"
export PATH="/home/nvidia/.local/bin:\${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONUNBUFFERED=1
export HF_TOKEN='${HF_TOKEN_ESCAPED}'
export WANDB_API_KEY='${WANDB_API_KEY_ESCAPED}'
export WANDB_USERNAME='${WANDB_USERNAME_ESCAPED}'
export WANDB_MODE='${WANDB_MODE_ESCAPED}'
TIMEOUT_PREFIX=""
if [ -n "${JOB_TIME}" ]; then
    TIMEOUT_PREFIX="timeout ${JOB_TIME}"
fi
nohup \${TIMEOUT_PREFIX} stdbuf -oL -eL bash -lc '${TRAIN_COMMAND_ESCAPED}' > "${LOG_FILE}" 2>&1 &
PID=\$!
sleep 1
if ! ps -p "\${PID}" >/dev/null; then
    echo "Process exited early; last log lines:" >&2
    tail -n 200 "${LOG_FILE}" >&2 || true
    exit 1
fi
echo "PID: \${PID}"
echo "Log: ${LOG_FILE}"
EOF
)

REMOTE_CMD_ESCAPED="$(escape_squotes "${REMOTE_CMD}")"
# shellcheck disable=SC2029
REMOTE_OUTPUT="$(ssh "${BREV_INSTANCE_NAME}" "bash -lc '${REMOTE_CMD_ESCAPED}'")"
printf "%s\n" "${REMOTE_OUTPUT}"

REMOTE_PID="$(printf "%s\n" "${REMOTE_OUTPUT}" | tr -d '\r' | grep -m 1 '^PID:' | awk '{print $2}')"
REMOTE_LOG="$(printf "%s\n" "${REMOTE_OUTPUT}" | tr -d '\r' | grep -m 1 '^Log:' | awk '{print $2}')"
if [ -z "${REMOTE_PID}" ] || [ -z "${REMOTE_LOG}" ]; then
    echo "Failed to parse remote PID/log path from SSH output." >&2
    exit 1
fi

SUBMITTED_LOG_PATH="${SCRIPT_DIR}/submitted_jobs.txt"
touch "${SUBMITTED_LOG_PATH}"
TAIL_CMD="ssh ${BREV_INSTANCE_NAME} \"tail -f ${REMOTE_LOG}\""
KILL_CMD="ssh ${BREV_INSTANCE_NAME} \"kill ${REMOTE_PID}\""
{
    printf "=== %s ===\n" "${RUN_ID}"
    printf "host: %s\n" "${BREV_INSTANCE_NAME}"
    printf "pid: %s\n" "${REMOTE_PID}"
    printf "log: %s\n" "${REMOTE_LOG}"
    printf "train_command: %s\n" "${TRAIN_COMMAND}"
    printf "%s\n" "${TAIL_CMD}"
    printf "%s\n\n" "${KILL_CMD}"
} >> "${SUBMITTED_LOG_PATH}"

echo "Training started on ${BREV_INSTANCE_NAME}."
echo "Monitor with:"
echo "  ${TAIL_CMD}"

if [ "${FOLLOW_LOG}" -eq 1 ]; then
    ssh -tt "${BREV_INSTANCE_NAME}" "tail -n +1 -F ${REMOTE_LOG}"
fi
