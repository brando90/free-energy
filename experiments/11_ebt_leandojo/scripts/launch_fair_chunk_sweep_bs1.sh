#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_API_KEY="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"
export WANDB_PROJECT="${WANDB_PROJECT:-free-energy-11-ebt-leandojo}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p runs/launch_logs

# Fair-size chunk sweep:
# - chunk 16 keeps the original baseline architecture.
# - chunks 1/4/8 use layer/head schedules with approximately matching params.
chunks=(1 4 8 16)
gpus=(0 1 2 3)
layers=(86 16 6 2)
heads=(1 4 8 16)
batch=1
grad_accum=4

launch_one() {
  local chunk="$1"
  local gpu="$2"
  local num_layers="$3"
  local num_heads="$4"
  local name="leandojo_lwb_chunk${chunk}_token256_l${num_layers}_heads${num_heads}_fair_bs${batch}_ga${grad_accum}"
  local log="runs/launch_logs/${name}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date -Is)] FAIR_BS1_START chunk=${chunk} gpu=${gpu} layers=${num_layers} heads=${num_heads} batch=${batch} grad_accum=${grad_accum} log=${log}" | tee -a runs/launch_logs/fair_chunk_sweep_bs1_master.log
  CUDA_VISIBLE_DEVICES="${gpu}" python train_ebt.py \
    data.chunk_size="${chunk}" \
    model.num_layers="${num_layers}" \
    model.num_heads="${num_heads}" \
    loader.batch_size="${batch}" \
    train.grad_accum_steps="${grad_accum}" \
    loader.num_workers=0 \
    loader.persistent_workers=false \
    wandb.project="${WANDB_PROJECT}" \
    wandb.name="${name}" \
    hydra.run.dir="runs/${name}_\${now:%Y%m%d_%H%M%S}" \
    > "${log}" 2>&1
  local rc=$?
  echo "[$(date -Is)] FAIR_BS1_END chunk=${chunk} gpu=${gpu} layers=${num_layers} heads=${num_heads} batch=${batch} grad_accum=${grad_accum} rc=${rc} log=${log}" | tee -a runs/launch_logs/fair_chunk_sweep_bs1_master.log
  return "${rc}"
}

for i in "${!chunks[@]}"; do
  launch_one "${chunks[$i]}" "${gpus[$i]}" "${layers[$i]}" "${heads[$i]}" &
done

wait
