#!/usr/bin/env bash
# Run the toy EBM experiment on a Stanford SNAP host.
#
# Defaults are chosen for the currently working SNAP GPU host:
#   HOST=skampere2.stanford.edu
#   DEVICE=cuda
#   GPU=0
#
# Usage:
#   ./experiments/01_toy_ebm_training/run_on_snap.sh
#   HOST=skampere1.stanford.edu GPU=4 ./experiments/01_toy_ebm_training/run_on_snap.sh
#   TAG=my_run DEVICE=cpu EMAIL_TO=me@example.com ./experiments/01_toy_ebm_training/run_on_snap.sh
set -euo pipefail

HOST="${HOST:-skampere2.stanford.edu}"
USER_REMOTE="${USER_REMOTE:-brando9}"
TAG="${TAG:-snap_exact}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
EMAIL_TO="${EMAIL_TO:-brandojazz@gmail.com}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3.11}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-}"

ssh_cmd() {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${USER_REMOTE}@${HOST}" "$@"
}

if [[ -z "${REMOTE_ROOT}" ]]; then
    REMOTE_ROOT="$(ssh_cmd 'printf "%s/free-energy" "$HOME"')"
fi

REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/.venv-toy-ebm}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${REMOTE_ROOT}/experiments/01_toy_ebm_training/results/${TAG}}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_ROOT}/experiments/01_toy_ebm_training/results/logs}"

echo "[snap] target: ${USER_REMOTE}@${HOST}:${REMOTE_ROOT}"
echo "[snap] tag=${TAG} device=${DEVICE} gpu=${GPU} email=${EMAIL_TO}"

echo "[snap] ensuring remote root exists"
ssh_cmd "mkdir -p '${REMOTE_ROOT}'"

echo "[snap] rsyncing repo subset"
rsync -azP \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    --exclude='*.pt' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='experiments/00_start_off/results/' \
    --exclude='experiments/02_ar_pros_cons/' \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
    "${LOCAL_ROOT}/" "${USER_REMOTE}@${HOST}:${REMOTE_ROOT}/"

echo "[snap] running experiment on remote"
set +e
ssh_cmd bash -s -- \
    "${REMOTE_ROOT}" \
    "${REMOTE_VENV}" \
    "${REMOTE_OUTPUT_DIR}" \
    "${REMOTE_LOG_DIR}" \
    "${REMOTE_PYTHON}" \
    "${TAG}" \
    "${DEVICE}" \
    "${GPU}" \
    "${EMAIL_TO}" <<'REMOTE'
set -uo pipefail

REMOTE_ROOT="$1"
REMOTE_VENV="$2"
REMOTE_OUTPUT_DIR="$3"
REMOTE_LOG_DIR="$4"
REMOTE_PYTHON="$5"
TAG="$6"
DEVICE="$7"
GPU="$8"
EMAIL_TO="$9"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "${REMOTE_OUTPUT_DIR}" "${REMOTE_LOG_DIR}"
LOG="${REMOTE_LOG_DIR}/run_${TAG}_$(date -u +%Y%m%dT%H%M%SZ).log"

{
    set -e
    cd "${REMOTE_ROOT}"

    echo "[remote] host=$(hostname -f 2>/dev/null || hostname)"
    echo "[remote] date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[remote] root=${REMOTE_ROOT}"
    echo "[remote] output_dir=${REMOTE_OUTPUT_DIR}"
    echo "[remote] python=${REMOTE_PYTHON}"
    if [[ "${REMOTE_PYTHON}" == */* ]]; then
        REMOTE_PYTHON_PATH="${REMOTE_PYTHON}"
    else
        REMOTE_PYTHON_PATH="$(command -v "${REMOTE_PYTHON}")"
    fi
    echo "[remote] python_path=${REMOTE_PYTHON_PATH}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    fi

    if [[ ! -x "${REMOTE_VENV}/bin/python" ]]; then
        uv venv --python "${REMOTE_PYTHON_PATH}" --system-site-packages "${REMOTE_VENV}"
    fi
    if ! "${REMOTE_VENV}/bin/python" -c 'import torch' >/dev/null 2>&1; then
        rm -rf "${REMOTE_VENV}"
        uv venv --python "${REMOTE_PYTHON_PATH}" --system-site-packages "${REMOTE_VENV}"
    fi
    uv pip install --python "${REMOTE_VENV}/bin/python" --quiet pytest

    "${REMOTE_VENV}/bin/python" - <<'PY'
import sys
import torch
print(f"[remote] executable={sys.executable}")
print(f"[remote] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

    echo "[remote] running pytest"
    "${REMOTE_VENV}/bin/python" -m pytest experiments/01_toy_ebm_training/test_toy_ebm.py

    echo "[remote] running smoke"
    PYTHON_BIN="${REMOTE_VENV}/bin/python" ./experiments/01_toy_ebm_training/run_smoke_test.sh

    echo "[remote] running real exact experiment"
    if [[ "${DEVICE}" == "cuda" && -n "${GPU}" ]]; then
        export CUDA_VISIBLE_DEVICES="${GPU}"
    fi
    "${REMOTE_VENV}/bin/python" experiments/01_toy_ebm_training/run_toy_ebm.py \
        --tag "${TAG}" \
        --output-dir "${REMOTE_OUTPUT_DIR}" \
        --models linear mlp cnn resnet transformer \
        --seq-len 9 \
        --num-train-tasks 48 \
        --num-test-tasks 16 \
        --epochs 80 \
        --batch-size 8 \
        --hidden-dim 64 \
        --lr 0.003 \
        --device "${DEVICE}" \
        --require-improvement

    REPORT="${REMOTE_OUTPUT_DIR}/snap_run_report.md"
    {
        echo "# SNAP Toy EBM Run"
        echo
        echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
        echo "Host: \`$(hostname -f 2>/dev/null || hostname)\`"
        echo
        echo "Output directory on SNAP: \`${REMOTE_OUTPUT_DIR}\`"
        echo
        echo "Tag: \`${TAG}\`"
        echo
        echo "Device: \`${DEVICE}\`"
        if [[ "${DEVICE}" == "cuda" ]]; then
            echo
            echo "CUDA_VISIBLE_DEVICES: \`${GPU}\`"
        fi
        echo
        echo "Log: \`${LOG}\`"
        echo
        echo "## Command"
        echo
        echo '```bash'
        echo "${REMOTE_VENV}/bin/python experiments/01_toy_ebm_training/run_toy_ebm.py \\"
        echo "  --tag ${TAG} \\"
        echo "  --output-dir ${REMOTE_OUTPUT_DIR} \\"
        echo "  --models linear mlp cnn resnet transformer \\"
        echo "  --seq-len 9 --num-train-tasks 48 --num-test-tasks 16 \\"
        echo "  --epochs 80 --batch-size 8 --hidden-dim 64 --lr 0.003 \\"
        echo "  --device ${DEVICE} --require-improvement"
        echo '```'
        echo
        echo "## Experiment Report"
        echo
        cat "${REMOTE_OUTPUT_DIR}/${TAG}_report.md"
    } > "${REPORT}"
} 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

subject="[free-energy] toy EBM SNAP ${TAG}"
if [[ "${status}" -eq 0 ]]; then
    subject="${subject} completed"
else
    subject="${subject} failed"
fi

if command -v mail >/dev/null 2>&1 && [[ -n "${EMAIL_TO}" ]]; then
    {
        echo "SNAP toy EBM run status: ${status}"
        echo "Host: $(hostname -f 2>/dev/null || hostname)"
        echo "Output: ${REMOTE_OUTPUT_DIR}"
        echo "Log: ${LOG}"
        echo
        if [[ -f "${REMOTE_OUTPUT_DIR}/${TAG}_report.md" ]]; then
            cat "${REMOTE_OUTPUT_DIR}/${TAG}_report.md"
        else
            tail -n 120 "${LOG}"
        fi
    } | mail -s "${subject}" "${EMAIL_TO}" || true
fi

exit "${status}"
REMOTE
remote_status=$?
set -e

echo "[snap] pulling results back"
rsync -azP \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
    "${USER_REMOTE}@${HOST}:${REMOTE_ROOT}/experiments/01_toy_ebm_training/results/" \
    "${LOCAL_ROOT}/experiments/01_toy_ebm_training/results/"

if [[ "${remote_status}" -ne 0 ]]; then
    echo "[snap] remote run failed with status ${remote_status}" >&2
    exit "${remote_status}"
fi

echo "[snap] done"
