#!/usr/bin/env bash
# Sync vla_picknplace to a Brev VM using rsync over the Brev SSH alias.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

if [ -z "${BREV_INSTANCE_NAME:-}" ]; then
    echo "BREV_INSTANCE_NAME is required in cluster/brev/.env.brev." >&2
    exit 1
fi
if [ -z "${BREV_CODE_DIR:-}" ] || [ -z "${BREV_LOGS_DIR:-}" ] || [ -z "${BREV_DATA_DIR:-}" ]; then
    echo "BREV_CODE_DIR, BREV_LOGS_DIR, and BREV_DATA_DIR must be set in cluster/brev/.env.brev." >&2
    exit 1
fi
if [ "${BREV_CODE_DIR##*/}" != "vla_picknplace" ]; then
    echo "BREV_CODE_DIR must end with /vla_picknplace before rsync --delete is allowed." >&2
    exit 1
fi
for remote_path in "${BREV_CODE_DIR}" "${BREV_LOGS_DIR}" "${BREV_DATA_DIR}"; do
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

LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SSH_TARGET="${BREV_INSTANCE_NAME}"
RSYNC_RSH_CMD="${RSYNC_RSH:-ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=10}"
RSYNC_RETRIES="${RSYNC_RETRIES:-3}"
RSYNC_TIMEOUT="${RSYNC_TIMEOUT:-300}"

echo "Syncing ${LOCAL_PROJECT_DIR} to Brev instance ${BREV_INSTANCE_NAME}:${BREV_CODE_DIR}"

ssh "${SSH_TARGET}" bash -s -- "${BREV_CODE_DIR}" "${BREV_LOGS_DIR}" "${BREV_DATA_DIR}" <<'REMOTE'
set -euo pipefail
mkdir -p "$1" "$2" "$3"
REMOTE

EXCLUDES=(
    --exclude=.git
    --exclude=.venv
    --exclude=__pycache__
    --exclude='*.pyc'
    --exclude=.pytest_cache
    --exclude=.ruff_cache
    --exclude=.cache
    --exclude=.uv-cache
    --exclude=data
    --exclude=hf_cache
    --exclude=hf_meta
    --exclude=wandb
    --exclude=logs
    --exclude=runs
    --exclude=checkpoints
    --exclude='*.arrow'
    --exclude='*.parquet'
    --exclude='*.safetensors'
    --exclude='*.ckpt'
    --exclude='*.pt'
    --exclude='*.pth'
    --exclude='*.onnx'
    --exclude='.DS_Store'
    --exclude='*.swp'
    --exclude='*.swo'
)

if [ "${BREV_SYNC_OUTPUTS:-0}" != "1" ]; then
    EXCLUDES+=(--exclude=outputs)
fi

attempt=1
while true; do
    if rsync -az --delete -e "${RSYNC_RSH_CMD}" --timeout="${RSYNC_TIMEOUT}" \
        "${EXCLUDES[@]}" \
        "${LOCAL_PROJECT_DIR}/" "${SSH_TARGET}:${BREV_CODE_DIR}/"; then
        echo "Synced to Brev."
        exit 0
    fi
    if [ "${attempt}" -ge "${RSYNC_RETRIES}" ]; then
        echo "Sync failed after ${RSYNC_RETRIES} attempts." >&2
        exit 1
    fi
    echo "Retrying rsync (${attempt}/${RSYNC_RETRIES})..." >&2
    attempt=$((attempt + 1))
    sleep 2
done
