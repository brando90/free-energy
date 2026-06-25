### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
# NOTE if use less than full node gpus set ntasks to same value and comment out exclusive directive

### LOG INFO ###
#SBATCH --job-name=ebt-xxs-bs=256_s1_lr=
#SBATCH --output=logs/slurm/nlp/ebt-xxs-bs=256_s1_lr=%A-%a.log
export RUN_NAME="ebt-xxs-bs=256_s1_lr="
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/nlp/
module purge

echo Node list: $SLURM_JOB_NODELIST

lr=(0.0012)
alpha=(0.5)
alpha_lr=(1.5)

${SLURM_ARRAY_TASK_ID:+srun} python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
--tokenizer "EleutherAI/gpt-neox-20b" \
\
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--context_length 256 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \
--accumulate_grad_batches 1 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 1000000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "fineweb" \
--num_workers 12 \
--validation_split_pct 0.0005 \
--val_check_interval 15000 \
\
--wandb_project 'nlp_pretrain' \
\
--log_model_archi \
--log_gradients \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}