### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

### LOG INFO ###
#SBATCH --job-name=ebt-xxs-s2_no_replay_coco_med_bs=256_pix=128_rand=4_lngvn=3_lr=
#SBATCH --output=logs/slurm/img/ebt-xxs-s2_no_replay_coco_med_bs=256_pix=128_rand=4_lngvn=3_lr=%A-%a.log
export RUN_NAME="ebt-xxs-s2_no_replay_coco_med_bs=256_pix=128_rand=4_lngvn=3_lr="
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/img/
module purge

# NOTE did not really test/build out functionality for this script

lr=(0.0004)
patch_size=(2)
alpha=(3000)
alpha_lr=(9000)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]}_ps=${patch_size[${SLURM_ARRAY_TASK_ID}]}_alpha=${alpha[${SLURM_ARRAY_TASK_ID}]} \
--modality "IMG" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--no_mcmc_detach \
--truncate_mcmc \
--langevin_dynamics_noise 3.0 \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--randomize_mcmc_step_size_scale 2 \
--randomize_mcmc_num_steps 4 \
--randomize_mcmc_num_steps_min 4 \
--mcmc_num_steps 1 \
\
--patch_size ${patch_size[${SLURM_ARRAY_TASK_ID}]} \
--clip_text_encoder_size "large" \
--log_image_every_n_steps 1500 \
--ffn_dim_multiplier 1 \
--check_val_every_n_epoch 5 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 256 \
--accumulate_grad_batches 1 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 200000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "coco_medium" \
--num_workers 12 \
--image_dims 128 128 \
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