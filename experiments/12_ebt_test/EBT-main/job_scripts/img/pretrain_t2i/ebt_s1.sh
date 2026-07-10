### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

### LOG INFO ###
#SBATCH --job-name=ebt-large-rand_cap_coco_med_bs=256_lr=
#SBATCH --output=logs/slurm/img/ebt-large-rand_cap_coco_med_bs=256_lr=%A-%a.log
export RUN_NAME="ebt-large-rand_cap_coco_med_bs=256_lr="
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/img/
module purge

lr=(0.0001)
patch_size=(2)
alpha=(3000)
alpha_lr=(9000)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]}_ps=${patch_size[${SLURM_ARRAY_TASK_ID}]}_alpha=${alpha[${SLURM_ARRAY_TASK_ID}]} \
--modality "IMG" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--patch_size ${patch_size[${SLURM_ARRAY_TASK_ID}]} \
--clip_text_encoder_size "large" \
--log_image_every_n_steps 3000 \
--ffn_dim_multiplier 1 \
--check_val_every_n_epoch 3 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 16 \
--accumulate_grad_batches 4 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 100000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "coco_medium" \
--num_workers 12 \
--image_dims 256 256 \
\
--wandb_project 'img_t2i' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 20 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}