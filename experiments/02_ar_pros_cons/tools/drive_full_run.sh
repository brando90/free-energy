#!/usr/bin/env bash
# Drive a full-tag probe run on the snap cluster, trying multiple hosts until
# one succeeds, then email + push the results.
#
# Strategy:
#   1. Walk HOSTS in order. For each host call ../run_on_snap.sh with TAG=full.
#   2. The first host whose run exits 0 wins. Skip the rest.
#   3. After a success, rsync outputs back (already done by run_on_snap.sh),
#      send email with the summary, commit + push results to main.
#
# Requirements: ~/keys/skampere_password.txt, ~/keys/gmail_app_password.txt,
# sshpass, git push access to main.
#
# Env overrides:
#   HOSTS="skampere2.stanford.edu skampere1.stanford.edu ..."
#   TAG=full
#   DEVICE=cuda
#   GPU=             (passed through to run_on_snap.sh)
#   SEED=0
#   SKIP_EMAIL=1 SKIP_PUSH=1   for dry runs

set -euo pipefail

HOSTS_DEFAULT="skampere2.stanford.edu skampere1.stanford.edu skampere3.stanford.edu mercury1.stanford.edu mercury2.stanford.edu"
HOSTS="${HOSTS:-$HOSTS_DEFAULT}"
TAG="${TAG:-full}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
GPU="${GPU:-}"
SKIP_EMAIL="${SKIP_EMAIL:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
RUN_SCRIPT="${EXP_DIR}/run_on_snap.sh"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
DRIVER_LOG="${LOG_DIR}/drive_full_${TAG}_${RUN_TS}.log"

echo "[drive] hosts: ${HOSTS}" | tee -a "${DRIVER_LOG}"
echo "[drive] tag=${TAG} device=${DEVICE} seed=${SEED} gpu=${GPU:-default}" | tee -a "${DRIVER_LOG}"

SUCCESS_HOST=""
for HOST in ${HOSTS}; do
    echo "[drive] === attempting host: ${HOST} ===" | tee -a "${DRIVER_LOG}"
    if HOST="${HOST}" TAG="${TAG}" DEVICE="${DEVICE}" GPU="${GPU}" SEED="${SEED}" "${RUN_SCRIPT}" "${TAG}" 2>&1 | tee -a "${DRIVER_LOG}"; then
        SUCCESS_HOST="${HOST}"
        echo "[drive] === host ${HOST} succeeded ===" | tee -a "${DRIVER_LOG}"
        break
    else
        echo "[drive] !!! host ${HOST} failed; trying next" | tee -a "${DRIVER_LOG}"
    fi
done

if [[ -z "${SUCCESS_HOST}" ]]; then
    echo "[drive] FATAL: no host succeeded for tag=${TAG}" | tee -a "${DRIVER_LOG}"
    if [[ "${SKIP_EMAIL}" != "1" ]]; then
        cd "${EXP_DIR}" && python3 -m tools.send_email \
            --subject "[02_ar_pros_cons] ${TAG} run FAILED on all hosts" \
            --note "Driver log: ${DRIVER_LOG}. Hosts tried: ${HOSTS}." || true
    fi
    exit 1
fi

SUMMARY_PATH="${EXP_DIR}/mnt/user-data/outputs/summary/${TAG}/summary.json"
echo "[drive] summary: ${SUMMARY_PATH}" | tee -a "${DRIVER_LOG}"

if [[ "${SKIP_EMAIL}" != "1" ]]; then
    echo "[drive] sending email" | tee -a "${DRIVER_LOG}"
    cd "${EXP_DIR}" && python3 -m tools.send_email \
        --subject "[02_ar_pros_cons] ${TAG} run done on ${SUCCESS_HOST}" \
        --note "Driver log: ${DRIVER_LOG}." \
        --attach "${SUMMARY_PATH}" || echo "[drive] WARN: email failed" | tee -a "${DRIVER_LOG}"
fi

if [[ "${SKIP_PUSH}" != "1" ]]; then
    echo "[drive] committing + pushing outputs" | tee -a "${DRIVER_LOG}"
    cd "${REPO_ROOT}"
    git add experiments/02_ar_pros_cons/mnt/user-data/outputs experiments/02_ar_pros_cons/logs 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "experiments/02_ar_pros_cons: record ${TAG} run on ${SUCCESS_HOST}" \
            -m "" \
            -m "Driver log: experiments/02_ar_pros_cons/logs/$(basename "${DRIVER_LOG}")" \
            -m "" \
            -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tee -a "${DRIVER_LOG}"
        git push origin main 2>&1 | tee -a "${DRIVER_LOG}"
    else
        echo "[drive] no new output files to commit" | tee -a "${DRIVER_LOG}"
    fi
fi

echo "[drive] done. winner: ${SUCCESS_HOST}" | tee -a "${DRIVER_LOG}"
