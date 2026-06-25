### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4 

### LOG INFO ###
#SBATCH --job-name=baseline_transformer-small-lr=
#SBATCH --output=logs/slurm/vid/baseline_transformer-small-lr=%A-%a.log
export RUN_NAME="baseline_transformer-small-lr="
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/vid/
module purge

lr=(0.0002)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "VID" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--backbone_type "dinov2" \
--vit_backbone_size "base" \
--custom_image_normalization \
--time_between_frames 0.25 \
\
--context_length 16 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 64 \
--accumulate_grad_batches 1 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 100000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "something" \
--dataset_dir ${SSV2_DIR} \
--num_workers 12 \
--image_dims 224 224 \
\
--wandb_project 'vid_dino_pretrain' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 20 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}