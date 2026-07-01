#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_API_KEY="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"
export WANDB_PROJECT="${WANDB_PROJECT:-free-energy-11-ebt-leandojo}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p runs/launch_logs

# Match the chunk=16 baseline model size while preserving the concatenated
# token scheme: hidden_dim = 256 * chunk_size.
chunks=(1 4 8)
gpus=(0 1 2)
layers=(86 16 6)
heads=(1 4 8)
batches=(2 1)

launch_one() {
  local chunk="$1"
  local gpu="$2"
  local num_layers="$3"
  local num_heads="$4"
  for batch in "${batches[@]}"; do
    local name="leandojo_lwb_chunk${chunk}_token256_l${num_layers}_heads${num_heads}_fair_bs${batch}"
    local log="runs/launch_logs/${name}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date -Is)] FAIR_START chunk=${chunk} gpu=${gpu} layers=${num_layers} heads=${num_heads} batch=${batch} log=${log}" | tee -a runs/launch_logs/fair_chunk_sweep_master.log
    CUDA_VISIBLE_DEVICES="${gpu}" python train_ebt.py \
      data.chunk_size="${chunk}" \
      model.num_layers="${num_layers}" \
      model.num_heads="${num_heads}" \
      loader.batch_size="${batch}" \
      loader.num_workers=0 \
      loader.persistent_workers=false \
      wandb.project="${WANDB_PROJECT}" \
      wandb.name="${name}" \
      hydra.run.dir="runs/${name}_\${now:%Y%m%d_%H%M%S}" \
      > "${log}" 2>&1
    local rc=$?
    echo "[$(date -Is)] FAIR_END chunk=${chunk} gpu=${gpu} layers=${num_layers} heads=${num_heads} batch=${batch} rc=${rc} log=${log}" | tee -a runs/launch_logs/fair_chunk_sweep_master.log
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    if grep -qiE "out of memory|cuda.*oom|CUDA out of memory|torch.OutOfMemoryError" "${log}"; then
      echo "[$(date -Is)] FAIR_OOM chunk=${chunk} layers=${num_layers} heads=${num_heads} batch=${batch}; retrying smaller batch" | tee -a runs/launch_logs/fair_chunk_sweep_master.log
      continue
    fi
    return "${rc}"
  done
  return 1
}

for i in "${!chunks[@]}"; do
  launch_one "${chunks[$i]}" "${gpus[$i]}" "${layers[$i]}" "${heads[$i]}" &
done

wait
