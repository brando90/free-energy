### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 

### LOG INFO ###
#SBATCH --job-name=baseline_transformer-medium-something_default
#SBATCH --output=logs/slurm/vid_inference/baseline_transformer-medium-something_default%A-%a.log
export RUN_NAME="baseline_transformer-medium-something_default"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/vid_inference/
module purge

lr=(0.0003)




python train_model.py \
--run_name ${RUN_NAME} \
--modality "VID" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--backbone_type "vae" \
--vae_normalization \
--time_between_frames 0.25 \
\
--context_length 16 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 8 \
--accumulate_grad_batches 1 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 30000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "k400" \
--dataset_dir ${K400_DIR} \
--num_workers 12 \
--image_dims 224 224 \
\
--wandb_project 'vid_inference' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 20 \
\
--execution_mode "inference" \
--only_test \
--only_test_model_ckpt "your/model/ckpt" \
--infer_max_gen_len 16 \
--infer_video_condition_frames 5 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}