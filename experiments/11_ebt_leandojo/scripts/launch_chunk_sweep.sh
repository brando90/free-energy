#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_API_KEY="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"
export WANDB_PROJECT="${WANDB_PROJECT:-free-energy-11-ebt-leandojo}"

mkdir -p runs/launch_logs

chunks=(1 4 8 16)
gpus=(0 1 2 3)
batches=(16 8 4 2 1)

launch_one() {
  local chunk="$1"
  local gpu="$2"
  for batch in "${batches[@]}"; do
    local name="leandojo_lwb_chunk${chunk}_token256_l2_heads16_bs${batch}"
    local log="runs/launch_logs/${name}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date -Is)] START chunk=${chunk} gpu=${gpu} batch=${batch} log=${log}" | tee -a runs/launch_logs/chunk_sweep_master.log
    CUDA_VISIBLE_DEVICES="${gpu}" python train_ebt.py \
      data.chunk_size="${chunk}" \
      loader.batch_size="${batch}" \
      loader.num_workers=0 \
      loader.persistent_workers=false \
      wandb.project="${WANDB_PROJECT}" \
      wandb.name="${name}" \
      hydra.run.dir="runs/${name}_\${now:%Y%m%d_%H%M%S}" \
      > "${log}" 2>&1
    local rc=$?
    echo "[$(date -Is)] END chunk=${chunk} gpu=${gpu} batch=${batch} rc=${rc} log=${log}" | tee -a runs/launch_logs/chunk_sweep_master.log
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    if grep -qiE "out of memory|cuda.*oom|CUDA out of memory" "${log}"; then
      echo "[$(date -Is)] OOM chunk=${chunk} batch=${batch}; retrying smaller batch" | tee -a runs/launch_logs/chunk_sweep_master.log
      continue
    fi
    return "${rc}"
  done
  return 1
}

for i in "${!chunks[@]}"; do
  launch_one "${chunks[$i]}" "${gpus[$i]}" &
done

wait
