### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

### LOG INFO ###
#SBATCH --job-name=ebt-xxs-all_s2_lngvn=1_truncate_lr=
#SBATCH --output=logs/slurm/vid/ebt-xxs-all_s2_lngvn=1_truncate_lr=%A-%a.log
export RUN_NAME="ebt-xxs-all_s2_lngvn=1_truncate_lr="
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/vid/
module purge

lr=(0.0004)
alpha=(3000)
# NOTE may need to remove truncate as that makes things more unstable

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]}_alpha=${alpha[${SLURM_ARRAY_TASK_ID}]} \
--modality "VID" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--no_mcmc_detach \
--truncate_mcmc \
--mcmc_replay_buffer \
--langevin_dynamics_noise 1 \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--randomize_mcmc_step_size_scale 2 \
--randomize_mcmc_num_steps 2 \
--randomize_mcmc_num_steps_min 2 \
--mcmc_num_steps 1 \
\
--backbone_type "vae" \
--vae_normalization \
--time_between_frames 0.25 \
--sdxl_vae_standardization \
\
--context_length 16 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 60000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "something" \
--dataset_dir ${SSV2_DIR} \
--num_workers 12 \
--image_dims 224 224 \
\
--wandb_project 'vid_vae_pretrain' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 20 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}