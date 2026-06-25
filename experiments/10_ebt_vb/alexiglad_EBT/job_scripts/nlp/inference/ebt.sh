### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

### LOG INFO ###
#SBATCH --job-name=ebt-xxs-steps=2_langevin=1
#SBATCH --output=logs/slurm/nlp_inference/ebt-xxs-steps=2_langevin=1%A-%a.log
export RUN_NAME="ebt-xxs-steps=2_langevin=1"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/nlp_inference/
module purge


BENCHMARKS=("lambada") # "gsm8k" "ai2arc" "bigbench_matrixshapes" "squad" "bigbench_elementary_math_qa" "bigbench_dyck_languages" 
DATASET=${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}
export RUN_NAME="${RUN_NAME}_${DATASET}"

python train_model.py \
--run_name ${RUN_NAME} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--tokenizer "EleutherAI/gpt-neox-20b" \
\
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--norm_pred \
--scale_alpha_with_energy \
\
--context_length 256 \
\
--gpus "-1" \
\
--peak_learning_rate 0.0012 \
--batch_size_per_device 8 \
--accumulate_grad_batches 4 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 211000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name ${DATASET} \
--num_workers 12 \
--validation_split_pct 0.0005 \
--val_check_interval 15000 \
\
--wandb_project "nlp_inference_accuracy" \
\
--log_model_archi \
--log_gradients \
\
--execution_mode "inference" \
--infer_ebt_advanced \
--infer_langevin_dynamics_noise 1 \
--infer_ebt_num_steps 2 \
--only_test \
--only_test_model_ckpt "your/model/ckpt" \
--infer_max_gen_len 2 \
--infer_topp 0.1 \
--infer_temp 0.0 \
--override_slurm_checks \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}