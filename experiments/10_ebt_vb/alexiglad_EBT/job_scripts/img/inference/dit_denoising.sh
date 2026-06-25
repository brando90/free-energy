### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

### LOG INFO ###
#SBATCH --job-name=dit-large-dns=0.2_recurse=2_time
#SBATCH --output=logs/slurm/img/dit-large-dns=0.2_recurse=2_time%A-%a.log
export RUN_NAME="dit-large-dns=0.2_recurse=2_time"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/img/
module purge

SELECTED_CKPT="your/model/ckpt"

python train_model.py \
--run_name ${RUN_NAME} \
--modality "IMG" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--diffusion_steps 100 \
--use_deterministic_reverse \
\
--patch_size 16 \
--image_task "denoising" \
--log_image_every_n_steps 10 \
--ffn_dim_multiplier 1 \
\
--gpus "-1" \
\
--peak_learning_rate 0.0001 \
--batch_size_per_device 128 \
--accumulate_grad_batches 1 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.0 \
--min_lr_scale 10 \
--max_steps 100000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "coco_medium" \
--num_workers 12 \
--image_dims 128 128 \
\
--wandb_project 'img_denoising_inference' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 20 \
\
--execution_mode "inference" \
--infer_recurse_diffusion_n_times 2 \
--denoise_images_noise_level 0.2 \
--only_test \
--only_test_model_ckpt ${SELECTED_CKPT} \
--override_slurm_checks \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}