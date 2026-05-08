#!/usr/bin/env bash
# Prepare a Brev VM for vla_picknplace MolmoAct2 work.
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

required_vars=(BREV_INSTANCE_NAME BREV_CODE_DIR BREV_LOGS_DIR BREV_DATA_DIR)
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

show_help() {
    cat <<'EOF'
Usage: setup_brev_env.sh [--skip-sync] [--skip-import-check]

Options:
  --skip-sync          Skip rsync code sync step.
  --skip-import-check  Skip import smoke check after install.

This script:
  1) Syncs vla_picknplace to Brev unless --skip-sync is set.
  2) Ensures uv is installed on the Brev VM.
  3) Ensures FFmpeg shared libraries are installed for LeRobot video decoding.
  4) Creates ${BREV_CODE_DIR}/.venv.
  5) Installs LeRobot and requirements.txt.
  6) Verifies imports and visible GPUs.
EOF
    exit 0
}

DO_SYNC=1
RUN_IMPORT_CHECK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sync)
            DO_SYNC=0
            shift
            ;;
        --skip-import-check)
            RUN_IMPORT_CHECK=0
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown argument: $1" >&2
            show_help
            ;;
    esac
done

if [ "${DO_SYNC}" -eq 1 ]; then
    "${SCRIPT_DIR}/sync_code_brev.sh"
fi

echo "Bootstrapping Python environment on ${BREV_INSTANCE_NAME}..."

ssh "${BREV_INSTANCE_NAME}" bash -s -- \
    "${BREV_CODE_DIR}" \
    "${BREV_LOGS_DIR}" \
    "${BREV_DATA_DIR}" \
    "${RUN_IMPORT_CHECK}" <<'REMOTE'
set -euo pipefail

BREV_CODE_DIR="$1"
BREV_LOGS_DIR="$2"
BREV_DATA_DIR="$3"
RUN_IMPORT_CHECK="$4"

mkdir -p "${BREV_CODE_DIR}" "${BREV_LOGS_DIR}" "${BREV_DATA_DIR}"

if ! ldconfig -p 2>/dev/null | grep -Eq 'libavutil\.so\.(56|57|58|59|60)'; then
    if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true >/dev/null 2>&1; then
        echo "[setup] Missing FFmpeg shared libraries and passwordless sudo is unavailable." >&2
        exit 1
    fi
    echo "[setup] Installing FFmpeg shared libraries..."
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
fi

if [ ! -x "/home/nvidia/.local/bin/uv" ]; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="/home/nvidia/.local/bin:${PATH}"
cd "${BREV_CODE_DIR}"

/home/nvidia/.local/bin/uv venv --python 3.12 .venv
/home/nvidia/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

if [ "${RUN_IMPORT_CHECK}" -eq 1 ]; then
    .venv/bin/python - <<'PY'
import importlib

for module_name in ("datasets", "huggingface_hub", "lerobot.datasets", "mujoco", "torch", "transformers"):
    importlib.import_module(module_name)
print("[setup] Import check passed.")
PY
fi

echo "[setup] Remote code path: ${BREV_CODE_DIR}"
echo "[setup] Remote logs path: ${BREV_LOGS_DIR}"
echo "[setup] Remote data path: ${BREV_DATA_DIR}"
echo "[setup] Visible GPUs:"
nvidia-smi -L
REMOTE

echo "Brev environment ready on ${BREV_INSTANCE_NAME}."
