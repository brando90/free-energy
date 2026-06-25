### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 

### LOG INFO ###
#SBATCH --job-name=baseline_transformer-small-dataloader_run_through_data_test
#SBATCH --output=logs/slurm/debug/baseline_transformer-small-dataloader_run_through_data_test%A-%a.log
export RUN_NAME="baseline_transformer-small-dataloader_run_through_data_test"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/debug/
module purge

lr=(0.0006)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--tokenizer "EleutherAI/gpt-neox-20b" \
\
--context_length 256 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 64 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 1000000 \
--warm_up_steps 10 \
\
--dataset_name "fineweb" \
--num_workers 12 \
--validation_split_pct 0.0005 \
\
--wandb_project 'debug' \
\
--log_model_archi \
--log_gradients \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run} \
\
--debug_dataloader \