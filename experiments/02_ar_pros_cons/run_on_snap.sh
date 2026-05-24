#!/usr/bin/env bash
# Run probes 01/03/05 on a snap cluster GPU host (default: skampere2.stanford.edu).
#
# Requirements on the local box:
#   - ~/keys/skampere_password.txt           (snap NFS password)
#   - sshpass (brew install hudochenkov/sshpass/sshpass)
#   - uv installed on the remote (it is already on /lfs/skampere2/0/brando9/.local/bin/uv)
#
# What it does:
#   1. rsyncs experiments/02_ar_pros_cons/ to the remote scratch directory
#   2. creates / refreshes a uv venv with torch + numpy
#   3. runs `python -m probes.run_all --tag $TAG` on the chosen device (cuda by default)
#   4. rsyncs mnt/user-data/outputs/ back to the local repo
#
# Usage:
#   ./run_on_snap.sh                       # smoke on skampere2, cuda, tag=smoke
#   HOST=skampere1.stanford.edu ./run_on_snap.sh full      # full run on skampere1
#   TAG=mytest ./run_on_snap.sh smoke
#   GPU=4 ./run_on_snap.sh smoke           # pin CUDA_VISIBLE_DEVICES=4
#
set -euo pipefail

HOST="${HOST:-skampere2.stanford.edu}"
USER_REMOTE="${USER_REMOTE:-brando9}"
TAG="${1:-${TAG:-smoke}}"
REMOTE_BASE="${REMOTE_BASE:-/dfs/scratch0/${USER_REMOTE}/free-energy/experiments/02_ar_pros_cons}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_BASE}/.venv}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-}"
SEED="${SEED:-0}"
PASS_FILE="${PASS_FILE:-${HOME}/keys/skampere_password.txt}"

if [[ ! -f "${PASS_FILE}" ]]; then
    echo "ERROR: snap password not found at ${PASS_FILE}" >&2
    exit 2
fi
if ! command -v sshpass >/dev/null 2>&1; then
    echo "ERROR: sshpass not found. Install with: brew install hudochenkov/sshpass/sshpass" >&2
    exit 2
fi

EXTRA_SMOKE=""
if [[ "${TAG}" == "smoke" ]]; then
    EXTRA_SMOKE="--smoke"
fi

EXPORT_GPU=""
if [[ -n "${GPU}" ]]; then
    EXPORT_GPU="export CUDA_VISIBLE_DEVICES=${GPU};"
fi

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SSHPASS_FILE="${PASS_FILE}"

ssh_cmd() {
    sshpass -f "${SSHPASS_FILE}" ssh -o StrictHostKeyChecking=accept-new "${USER_REMOTE}@${HOST}" "$@"
}

rsync_cmd() {
    sshpass -f "${SSHPASS_FILE}" rsync -azP \
        --exclude='.venv' --exclude='__pycache__' --exclude='mnt/user-data/outputs' \
        -e "ssh -o StrictHostKeyChecking=accept-new" \
        "$@"
}

echo "[snap] target: ${USER_REMOTE}@${HOST}:${REMOTE_BASE}  tag=${TAG} device=${DEVICE} gpu=${GPU:-default}"

echo "[snap] ensuring remote base exists"
ssh_cmd "mkdir -p '${REMOTE_BASE}'"

echo "[snap] rsyncing experiment dir"
rsync_cmd "${LOCAL_DIR}/" "${USER_REMOTE}@${HOST}:${REMOTE_BASE}/"

echo "[snap] preparing uv venv on remote"
sshpass -f "${SSHPASS_FILE}" ssh -o StrictHostKeyChecking=accept-new "${USER_REMOTE}@${HOST}" bash -s <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd '${REMOTE_BASE}'
if [[ ! -d '${REMOTE_VENV}' ]]; then
    uv venv --python 3.11 '${REMOTE_VENV}'
fi
uv pip install --python '${REMOTE_VENV}/bin/python' --quiet -r requirements.txt
EOF

echo "[snap] running probes on remote"
sshpass -f "${SSHPASS_FILE}" ssh -o StrictHostKeyChecking=accept-new "${USER_REMOTE}@${HOST}" bash -s <<EOF
set -euo pipefail
cd '${REMOTE_BASE}'
${EXPORT_GPU}
mkdir -p logs
LOG=logs/run_${TAG}_\$(date -u +%Y%m%dT%H%M%SZ).log
'${REMOTE_VENV}/bin/python' -m probes.run_all \\
    --tag '${TAG}' \\
    --device '${DEVICE}' \\
    --seed '${SEED}' \\
    ${EXTRA_SMOKE} 2>&1 | tee "\$LOG"
echo "[snap] log: \$LOG"
EOF

echo "[snap] pulling outputs back"
sshpass -f "${SSHPASS_FILE}" rsync -azP \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    "${USER_REMOTE}@${HOST}:${REMOTE_BASE}/mnt/user-data/outputs/" \
    "${LOCAL_DIR}/mnt/user-data/outputs/"

echo "[snap] done."
