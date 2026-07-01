#!/usr/bin/env bash
# Run the Q02 dynamic weighted CD experiment on a SNAP GPU host.

set -euo pipefail

HOST="${HOST:-skampere2.stanford.edu}"
USER_REMOTE="${USER_REMOTE:-brando9}"
TAG="${1:-${TAG:-snap_5seed}}"
REMOTE_BASE="${REMOTE_BASE:-/dfs/scratch0/${USER_REMOTE}/free-energy/questions/02_ebm_dynamic_cd_langevin}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_BASE}/.venv}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-800}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRID_SIZE="${GRID_SIZE:-160}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2048}"
EVAL_LANGEVIN_STEPS="${EVAL_LANGEVIN_STEPS:-300}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa}"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="expt_v1/results/${TAG}"

ssh_cmd() {
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "${USER_REMOTE}@${HOST}" "$@"
}

rsync_cmd() {
    # shellcheck disable=SC2086
    rsync -azP \
        --exclude='.venv' --exclude='__pycache__' --exclude='expt_v1/results' \
        -e "ssh ${SSH_OPTS}" \
        "$@"
}

echo "[snap] target: ${USER_REMOTE}@${HOST}:${REMOTE_BASE} tag=${TAG} gpu=${GPU}"

echo "[snap] ensuring remote base exists"
ssh_cmd "mkdir -p '${REMOTE_BASE}'"

echo "[snap] rsyncing Q02 packet"
rsync_cmd "${LOCAL_DIR}/" "${USER_REMOTE}@${HOST}:${REMOTE_BASE}/"

echo "[snap] preparing uv venv"
ssh_cmd "bash -lc \"set -euo pipefail; export PATH=\\\"\$HOME/.local/bin:\$PATH\\\"; cd '${REMOTE_BASE}'; if [[ ! -d '${REMOTE_VENV}' ]]; then uv venv --python 3.11 '${REMOTE_VENV}'; fi; uv pip install --python '${REMOTE_VENV}/bin/python' --quiet torch numpy matplotlib tqdm\""

echo "[snap] running seeds: ${SEEDS}"
ssh_cmd "bash -lc \"set -euo pipefail; cd '${REMOTE_BASE}'; mkdir -p 'expt_v1/results/${TAG}/logs'; export CUDA_VISIBLE_DEVICES='${GPU}'; for SEED in ${SEEDS}; do OUT='expt_v1/results/${TAG}/seed_'\\\$SEED; LOG='expt_v1/results/${TAG}/logs/seed_'\\\$SEED'.log'; echo '[snap] seed='\\\$SEED 'out='\\\$OUT; '${REMOTE_VENV}/bin/python' expt_v1/src/dynamic_weighted_cd.py --steps '${STEPS}' --batch-size '${BATCH_SIZE}' --grid-size '${GRID_SIZE}' --eval-samples '${EVAL_SAMPLES}' --eval-langevin-steps '${EVAL_LANGEVIN_STEPS}' --device '${DEVICE}' --seed \\\$SEED --out-dir \\\$OUT 2>&1 | tee \\\$LOG; done; '${REMOTE_VENV}/bin/python' expt_v1/src/aggregate_reports.py --input-dir 'expt_v1/results/${TAG}' --out-json 'expt_v1/results/${TAG}/aggregate.json' --out-md 'expt_v1/results/${TAG}/aggregate.md'\""

echo "[snap] pulling results"
mkdir -p "${LOCAL_DIR}/expt_v1/results/${TAG}"
rsync -azP \
    -e "ssh ${SSH_OPTS}" \
    "${USER_REMOTE}@${HOST}:${REMOTE_BASE}/expt_v1/results/${TAG}/" \
    "${LOCAL_DIR}/expt_v1/results/${TAG}/"

echo "[snap] done: ${LOCAL_DIR}/expt_v1/results/${TAG}"
