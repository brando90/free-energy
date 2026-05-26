#!/usr/bin/env bash
# Run the vision derisking experiment on SNAP (default: skampere2).
#
# Usage:
#   ./experiments/04_vision_energy_comparison/run_on_snap.sh smoke
#   ./experiments/04_vision_energy_comparison/run_on_snap.sh real
#   ./experiments/04_vision_energy_comparison/run_on_snap.sh both
set -euo pipefail

MODE="${1:-both}"
HOST="${HOST:-skampere2.stanford.edu}"
USER_REMOTE="${USER_REMOTE:-brando9}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
PASS_FILE="${PASS_FILE:-${HOME}/keys/skampere_password.txt}"
REMOTE_BASE="${REMOTE_BASE:-/dfs/scratch0/${USER_REMOTE}/free-energy/experiments/04_vision_energy_comparison}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_BASE}/.venv}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/lfs/skampere2/0/brando9/miniconda/bin/python3}"

if [[ ! -f "${PASS_FILE}" ]]; then
    echo "ERROR: SNAP password file not found: ${PASS_FILE}" >&2
    exit 2
fi
if ! command -v sshpass >/dev/null 2>&1; then
    echo "ERROR: sshpass not found. Install with: brew install hudochenkov/sshpass/sshpass" >&2
    exit 2
fi

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

ssh_cmd() {
    sshpass -f "${PASS_FILE}" ssh \
        -o ControlMaster=no \
        -o ControlPath=none \
        -o ConnectTimeout=30 \
        -o StrictHostKeyChecking=accept-new \
        "${USER_REMOTE}@${HOST}" "$@"
}

rsync_cmd() {
    sshpass -f "${PASS_FILE}" rsync -azP \
        --exclude='.venv' --exclude='__pycache__' --exclude='logs/*.log' \
        -e "ssh -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new" \
        "$@"
}

echo "[snap] target=${USER_REMOTE}@${HOST}:${REMOTE_BASE} mode=${MODE} device=${DEVICE} gpu=${GPU}"
ssh_cmd "mkdir -p '${REMOTE_BASE}'"

echo "[snap] rsyncing experiment"
rsync_cmd "${LOCAL_DIR}/" "${USER_REMOTE}@${HOST}:${REMOTE_BASE}/"

echo "[snap] checking remote python"
REMOTE_RUNNER="$(ssh_cmd bash -s -- "${REMOTE_PYTHON}" "${REMOTE_VENV}" <<'EOF'
set -euo pipefail
REMOTE_PYTHON="$1"
REMOTE_VENV="$2"
export PATH="$HOME/.local/bin:$PATH"
if [[ -x "${REMOTE_PYTHON}" ]] && "${REMOTE_PYTHON}" - <<'PY' >/dev/null 2>&1
import torch, sklearn, matplotlib
PY
then
    printf "%s" "${REMOTE_PYTHON}"
    exit 0
fi

if [[ ! -x "${REMOTE_VENV}/bin/python" ]]; then
    uv venv --python 3.11 --system-site-packages "${REMOTE_VENV}" >/dev/null
fi
uv pip install --python "${REMOTE_VENV}/bin/python" --quiet -r requirements.txt
if ! "${REMOTE_VENV}/bin/python" - <<'PY' >/dev/null 2>&1
import torch, sklearn, matplotlib
PY
then
    uv pip install --python "${REMOTE_VENV}/bin/python" --quiet torch torchvision
fi
printf "%s" "${REMOTE_VENV}/bin/python"
EOF
)"
echo "[snap] remote python: ${REMOTE_RUNNER}"

echo "[snap] running remote benchmark"
ssh_cmd bash -s -- "${REMOTE_BASE}" "${REMOTE_RUNNER}" "${MODE}" "${DEVICE}" "${GPU}" "${SEED}" <<'EOF'
set -euo pipefail
REMOTE_BASE="$1"
REMOTE_RUNNER="$2"
MODE="$3"
DEVICE="$4"
GPU="$5"
SEED="$6"
cd "${REMOTE_BASE}"
mkdir -p logs results
export CUDA_VISIBLE_DEVICES="${GPU}"
LOG="logs/run_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"
{
    echo "[remote] host=$(hostname -f 2>/dev/null || hostname)"
    echo "[remote] date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[remote] cwd=$PWD"
    "${REMOTE_RUNNER}" - <<'PY'
import torch, sys
print(f"[remote] python={sys.executable}")
print(f"[remote] torch={torch.__version__} cuda={torch.cuda.is_available()} n={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[remote] gpu0={torch.cuda.get_device_name(0)}")
PY

    if [[ "${MODE}" == "smoke" || "${MODE}" == "both" ]]; then
        "${REMOTE_RUNNER}" run_vision_benchmark.py \
            --smoke --tag snap_smoke --dataset digits --device "${DEVICE}" --seed "${SEED}" \
            --output-dir results
    fi
    if [[ "${MODE}" == "real" || "${MODE}" == "both" ]]; then
        "${REMOTE_RUNNER}" run_vision_benchmark.py \
            --tag snap_real --dataset digits --device "${DEVICE}" --seed "${SEED}" \
            --epochs 6 --diffusion-epochs 4 --max-train 1400 --max-test 450 --batch-size 128 \
            --output-dir results
    fi
} 2>&1 | tee "${LOG}"
echo "[remote] log=${LOG}"
EOF

echo "[snap] pulling results"
rsync_cmd "${USER_REMOTE}@${HOST}:${REMOTE_BASE}/results/" "${LOCAL_DIR}/results/"
rsync_cmd "${USER_REMOTE}@${HOST}:${REMOTE_BASE}/logs/" "${LOCAL_DIR}/logs/"

echo "[snap] done"
