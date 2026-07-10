# EBT Synthetic Chain Results

## Some notes about the codebase

### Main: Uses lightning trainer (see train_model.py set_trainer)
* `distributed_strategy: auto`
  * only applies when you have multiple GPUs (aka data parallel etc.)
    * `ddp` is the most common; data split in GPU, so fraction of batch, calculate locally, sync, and update model weights simulate
  * `accumulate_grad_batches` = 1 (gradient accumulation) 
  * there is gradient clipping during trainig: 1.0

### ModelTrainer (`base_model_trainer.py`):
* there is a "test_step" that generates text --> logs data
* `target_autoregressive_metrics`; tracks the autoregressive metrics
  * the atuoregressive metrics is always `torch.argmax`... no temperature it appears so.
  * kind of redundant function...
* We use AdamW optimizer!! 
* LR stuff:
  * We do CosineAnnealingLR; T_max is usually max_steps (1M)
  * goes from peak lr --> peak_lr/min_lr_scale (10x smaller!) in 1M 
    * This is cosine annealing; "safe landing"
    * when model aproaches local minimum, gradients too small; 
    * if lr too high, optmizer will overstep. 
    * Has several good properties about Cosine Annealing for optimization
      * flat top; explore the hgih-dimensional landscape over and over again
      * steep drop; middle training, forces model to transition to exploration --> exploitation on one basin
      * long tail; convergence. Extended ultra fine tuning. Settling on the lowest sharpest point

  * Furthermore, we do WarmUpCosineAnnealing 
    * starts from 0 --> peak_lr in 10k steps!
    * AdamW; learning rate IS ALMOST MADATORY
    * AdamW will make very poor and highly skewed updates (variance is completely new)
      * furthermore, loss is really high --> extremely high gradients.
    * for stablility, we eventually want to warm up the cosine annealing.
    * "safe takeoff"

* We also enable weight decay!
  * use smaller weights $\to$ reduce overfitting: 0.01

### generate_text (`inference/nlp/generate_text.py`): 
1. given model, batch, hparams
2. Note that context length is on the entire thing; prompt + generation; EBT returns logits.
3. There is an "advanced infer mode" for ebt that is not being done
  * more mcmc steps, override alpha, generate more samples (Best-of-N)
  * by default false
4. There is temperature done! 0.6 on inference (on `inference`)
  * this is the `infer_temp`; likely may want to change this!
5. on training, we have `get_ppl` which is for advanced inference
  * quite different; might want to change this later. Seems ot be similar to forward_loss...

### EBT (`EBT-main/model/nlp/ebt.py`):
* what is the langevin dynamic shape (what is it on):
* step size can be learnable; however by default it is set to False
* langevin_noise_std is 0.0 by default

----- TODO: really important ----- 
* vocab_to_embed_uses_prob_dist --> makes initial condition a normalized probability distribution if modality is NLP (TRUE!)

* it DOESN'T do any randomization of step size
    parser.add_argument("--randomize_mcmc_step_size_scale", help="randomize the value of mcmc_step_size by a factor specified, i.e. if is 2 will mult by 2 and div by 2 and thats the range to sample from uniformly", type=float, default=1)

* mcmc num steps is 4 and step size is 0.75

* Tried contrastive learning loss --> doesn't appear to work according to benchmark
  * rollout --> contrastive rollout..... interesting
  * still want to try to see if there's anything there...

* `forward`: Performs Langevin Dynamics stuff
  1. take x and embed
  2. calculate alpha based on randomize_mcmc_step_size
  3. corrupt_embeddings (aka the initial noise N(0,I ))
  4. randomize num steps
  5. for each step 
    1. by default false, but if true, attaches entire computation graph if `no_mcmc_detach=True` (TODO: Usually S2 learning); save
    2. we normalize EVERY STEP with softmax (from randn N(0,I)) via softmax
    3. then go through an embedding via `self.embedding.weight`
    4. concatenate w/ orig context --> transformer feed forward 
    5. by default false, but if `truncate_mcmc=True`, only builds computation graph on the last step (TODO: Usually for S2) 
    6. with `scale_alpha_with_energy`, sometimes varies alpha via exp or something, but by default it updates the predicted tokens again.

* randomization of num steps:
  * randomization happens on `randomize_mcmc_num_steps_final_landscape` only applies on the last step. 
  * `randomize_mcmc_num_steps` is applied (even if no_randomness flag). It seems that for ebt_tyoe=default, each landscape is effectively the same (we kinda want this)
    * TODO: See if you want to implement this parameter

* `forward_loss_wrapper` is the actually loss function
  * if use replay buffer, it will sample from replay buffer and do predicted_dist and predicted_energies. (by default, false, although seems interesting to play around with)
  * else, it would just have x/y
  * trains the "reconstruction_loss", and iterates over all the steps:
    1. creates nll_loss (after log_softmax) of the predicted distribution and next token indicies.
      * basically cross entropy loss.
    2. if `truncate_mcmc`(TODO: Usually for S2 learning) only applies final loss at the end of the very last step! ELSE IT JUST AVERAGES!!!
    3. Logs final prediction energies gaps as well.
  * ALSO CREATES CONTRASTIVE LOSS (model predicted; push up, pushing down on energy of true samples; doesn't work apparently)
    * uses the already generated as samples...

### DataLoader:
N/A (didn't take notes)

## 2026-07-08: EBT-main retrain with slash-named W&B metrics

### Code changes

- `base_model_trainer.py`
  - Uses the parent experiment dataloader from `/lfs/skampere1/0/eshaanb/free-energy/experiments/12_ebt_test/dataloader.py` for `dataset_name=synthetic_chain`.
  - Adapts parent dataloader batches by mapping `full_input_ids` to `input_ids` and building `target_metric_mask` from `target_start` and `target_len`.
  - Logs metrics with W&B/Lightning keys like `train/loss` and `val/loss` instead of old flat keys like `train_loss` and `valid_loss`.
  - Runs train autoregressive metrics every train step by default through `synthetic_chain_train_ar_interval=1`.

- `model/nlp/ebt.py`
  - Logs `reconstruction_loss` in addition to `loss`, `perplexity`, teacher-forcing accuracy, and autoregressive accuracy.
  - Logs `contrastive_rollout_accuracy=0.0` and `contrastive_rollout_loss=0.0` placeholders for this EBT-main trainer path, since contrastive rollout is not enabled in this model implementation.

- `train_model.py`
  - Sets `context_length=64` automatically for `dataset_name=synthetic_chain` when no context length is provided.
  - Sanitizes checkpoint filenames when monitoring slash-named metrics.

### W&B metrics expected

- `train/teacher_forcing_token_accuracy`
- `train/perplexity`
- `train/loss`
- `train/autoregressive_token_accuracy`
- `val/teacher_forcing_token_accuracy`
- `val/reconstruction_loss`
- `val/perplexity`
- `val/loss`
- `val/autoregressive_token_accuracy`
- `val/autoregressive_exact_accuracy`
- `val/contrastive_rollout_accuracy`

## Dataset

Current implementation files:

- `generate_dataset.py`: creates the JSONL dataset.
- `tokenizer.py`: defines the generator-side synthetic tokenizer.
- `EBT-main/data/nlp/synthetic_chain_dataset.py`: defines the EBT-main dataset wrapper/collator used by `dataset_name=synthetic_chain`.
- `EBT-main/data/nlp/synthetic_chain_dataset/synthetic_chain_dataset.jsonl`: active dataset file.

### Dataset generator mechanics

The active synthetic-chain generator creates goal-directed proof traces over 128 fact symbols:

- Fact symbols are generated by `fact_symbols(128)`: `A` through `Z`, then `AA`, `AB`, ..., up to 128 symbols.
- Rule types are exactly `unary`, `or`, and `and`.
- A unary rule renders as `A -> B`.
- An `or` rule renders as `A or B -> C`; either premise is sufficient.
- An `and` rule renders as `A and B -> C`; both premises must already be known.
- Prompts render as `Start [ ... ] <SEP> Rules [ ... ] <SEP> Goal X`.
- Targets render as fact-only canonical traces: `A B C D`, not `A -> B -> C -> D`.
- The generator verifies that the target trace ends at the goal and that the goal is in the forward-chaining closure of the starts and rules.

Proof construction:

- Each sample chooses a random proof depth.
- It samples 1-5 starting facts for the current 128-token dataset.
- It chooses one current known fact as the proof cursor.
- At each proof step it creates a new consequent and one proof rule.
- For `or`, the unused branch is usually an unavailable alternate premise; `proof_premise` records which branch was actually used.
- For `and`, the second premise is chosen from already known facts when possible; if no second known fact exists, the generator adds a new start fact so the conjunction is solvable.
- The target trace records required facts in canonical order without duplicate facts.

Distractor construction:

- Distractors are sampled from the same rule language as proof rules.
- Distractor consequents avoid the goal and normally avoid proof facts.
- The generator enforces unique rule keys `(premises, op, consequent)`.
- If a generated example is too long, distractors are popped until the full sequence fits under `max_full_tokens`.

Current active JSONL stats:

- Splits: `300000` train, `75000` val, `1000` test.
- Max full sequence length: `128` tokens including prompt, BOS, target, EOS.
- Proof depth range: `1-5`.
- Prompt length: min `38`, max `124`, mean `92.33`.
- Target length: min `2`, max `10`, mean `4.47`.
- Full sequence length: min `42`, max `128`, mean `98.8`.
- Distractor rules: min `6`, max `18`, mean `11.77`.
- Proof-path rule histogram: `unary=475712`, `or=332795`, `and=319973`.

Tokenizer:

- Vocabulary size is `147`.
- Special tokens: `<pad>`, `<eos>`, `<bos>`, `<unk>`.
- Fact tokens: 128 symbols from `A` onward, including multi-character facts such as `AA`, `AB`, etc.
- Structural/task tokens include `->`, `and`, `or`, `Start`, `Rules`, `Goal`, `Prove`, `<SEP>`, `[`, `]`, `,`, `|`, `fact:`, `rule:`, `query:`.
- `->`, `and`, `or`, `Start`, `Rules`, `Goal`, and `<SEP>` are all single tokens.

EBT-main dataset wrapper behavior:

- The JSONL row fields used for training are `prompt_tokens`, `target_tokens`, `split`, and `id`.
- `input_ids = prompt_ids + [<bos>] + target_ids + [<eos>]`.
- `labels` are initialized to `-100` everywhere, then filled only from the target region through EOS.
- `target_metric_mask` is true only for the target facts, not the prompt and not EOS.
- The collator pads `input_ids`, `labels`, prompt ids, and target ids separately.
- `attention_mask` marks non-pad input tokens.
- `target_attention_mask` marks non-pad target tokens.
- `synthetic_chain_max_items=0` means use the full split.

## Best EBT run on dataset + Transformer run + command + hyperparameters changes

## 2026-07-09: synthetic-chain-128-hop1-5 live run summary

### Dataset currently used by `dataset_name=synthetic_chain`

Code paths:

- Generator: `/lfs/skampere1/0/eshaanb/free-energy/experiments/12_ebt_test/generate_dataset.py`
- EBT-main dataset wrapper: `EBT-main/data/nlp/synthetic_chain_dataset.py`
- JSONL data: `EBT-main/data/nlp/synthetic_chain_dataset/synthetic_chain_dataset.jsonl`

Task format:

- Prompt tokens are `Start [ ... ] <SEP> Rules [ ... ] <SEP> Goal X`.
- Target tokens are the canonical proof trace with facts only; arrows are not included in the answer.
- Training sequence is `prompt_tokens + <bos> + target_tokens + <eos>`.
- Labels are masked over the prompt, so loss/accuracy are computed only on the generated proof trace plus EOS.
- The prompt can contain unary rules, `and` rules, `or` rules, multiple starts, many distractors, and exactly one goal.
- `or` means either premise is enough; the target trace includes the branch actually used.
- `and` requires both premises; when one required premise is a separate start/known fact, it can appear in the target trace before the consequent.

Current data summary:

- Splits: `300000` train, `75000` val, `1000` test.
- Max full sequence length: `128` tokens including prompt, BOS, target, EOS.
- Proof depth range: `1-5`; observed mean depth `3.0`.
- Prompt length: min `38`, max `124`, mean `92.33`.
- Target length: min `2`, max `10`, mean `4.47`.
- Full sequence length: min `42`, max `128`, mean `98.8`.
- Distractor rules: min `6`, max `18`, mean `11.77`.
- Proof-path rule histogram over the full JSONL: `unary=475712`, `or=332795`, `and=319973`.

Examples of what the model sees and must predict:

```text
Prompt:
Start [ AE , AZ , CW , CK , BW ] <SEP> Rules [ Z or DE -> C , BY or AO -> BI , BW or N -> S , DN -> BX , S or Z -> DN , DM -> P , AZ or S -> C , DS -> BK , BH and AB -> BE , AX -> BB , AB or CF -> DJ , DW -> T ] <SEP> Goal S

Target:
BW S
```

```text
Prompt:
Start [ DP ] <SEP> Rules [ DP -> BF , AG and DR -> AH , DO and DI -> V , CN and BM -> BS , BV -> BD , AG -> K , R and DF -> BJ , BW or K -> CC , AC or AZ -> DD , DD -> AN , CD -> AH , AN and C -> DT , DL or E -> BN , BS -> BD , L or T -> AN , BT -> D , CZ or BX -> BG , R -> BI ] <SEP> Goal BF

Target:
DP BF
```

```text
Prompt:
Start [ BF ] <SEP> Rules [ CP or DN -> BE , DH -> CB , I -> AZ , AJ -> CS , DR and BF -> BP , CO and X -> G , F -> DR , AB and X -> CS , CG -> BG , BF -> F , DQ -> W , CP and L -> CU , BP or CN -> BJ , DL and DB -> N , BS or W -> AI ] <SEP> Goal BJ

Target:
BF F DR BP BJ
```

### EBT 2xs run

Run name: `ebt-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu7-v4`

Status at summary time:

- Still running in tmux session `ebt_2xs_chain128_gpu7`.
- Uses GPU 7.
- W&B project: `synthetic-chain-128-hop1-5`.
- Log file: `EBT-main/logs/synthetic_chain_128_hops1_5_2xs/ebt-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu7-v4.log`.

Important config:

- Model: `ebt`, size `2xs`.
- Context length: `128`.
- Batch size per device: `64`; accumulation `1`.
- Max steps: `100000`.
- Peak LR: `0.0005`; warmup `1000`.
- Validation before training enabled; then `val_check_interval=200`.
- `limit_val_batches=32`.
- Train autoregressive metrics interval: `synthetic_chain_train_ar_interval=2000`.
- EBT/Langevin settings: `mcmc_num_steps=4`, `mcmc_step_size=0.75`, random-noise initial condition, normalized initial condition, `vocab_to_embed_uses_prob_dist`.

Launch command used, with the W&B key intentionally omitted:

```bash
cd /lfs/skampere1/0/eshaanb/free-energy/experiments/12_ebt_test/EBT-main
export CUDA_VISIBLE_DEVICES=7
export WANDB_API_KEY="$WANDB_API_KEY"
../.venv/bin/python train_model.py \
  --run_name ebt-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu7-v4 \
  --modality NLP \
  --model_name ebt \
  --model_size 2xs \
  --dataset_name synthetic_chain \
  --tokenizer synthetic_chain \
  --context_length 128 \
  --mcmc_num_steps 4 \
  --mcmc_step_size 0.75 \
  --denoising_initial_condition random_noise \
  --gaussian_random_noise_scaling 1.0 \
  --normalize_initial_condition \
  --vocab_to_embed_uses_prob_dist \
  --gpus 1 \
  --distributed_strategy auto \
  --batch_size_per_device 64 \
  --accumulate_grad_batches 1 \
  --train_metric_log_batch_size 64 \
  --num_workers 8 \
  --synthetic_chain_max_items 0 \
  --synthetic_chain_train_ar_interval 2000 \
  --peak_learning_rate 0.0005 \
  --warm_up_steps 1000 \
  --max_steps 100000 \
  --val_check_interval 200 \
  --limit_val_batches 32 \
  --validate_before_training \
  --gradient_clip_val 1.0 \
  --checkpoint_monitor_string val/loss \
  --checkpoint_monitor_mode min \
  --save_top_k_ckpts 2 \
  --wandb_project synthetic-chain-128-hop1-5 \
  --wandb_tags synthetic_chain ebt 2xs chain128 hops1-5 bs64 gpu7 v4 lr0.0005 \
  > logs/synthetic_chain_128_hops1_5_2xs/ebt-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu7-v4.log 2>&1
```

Latest observed metrics from the log:

- Around step `~2125`, latest train batch:
  - `train/loss=1.43`
  - `train/teacher_forcing_token_accuracy=0.572`
  - `train/teacher_forcing_exact_accuracy=0.156`
- Last completed validation:
  - `val/loss=1.67`
  - `val/teacher_forcing_token_accuracy=0.489`
  - `val/teacher_forcing_exact_accuracy=0.159`
  - `val/autoregressive_token_accuracy=0.374`
  - `val/autoregressive_exact_accuracy=0.156`

Interpretation: this is the best EBT run observed so far on the current synthetic-chain task. It is learning the task, but still substantially trails the transformer baseline on exact proof generation.

### Transformer 2xs run

Run name: `transformer-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu6-v4`

Status at summary time:

- Still running in tmux session `transformer_2xs_chain128_gpu6`.
- Uses GPU 6.
- W&B project: `synthetic-chain-128-hop1-5`.
- Log file: `EBT-main/logs/synthetic_chain_128_hops1_5_2xs/transformer-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu6-v4.log`.

Important config:

- Model: `baseline_transformer`, size `2xs`.
- Context length: `128`.
- Batch size per device: `64`; accumulation `1`.
- Max steps: `100000`.
- Peak LR: `0.0005`; warmup `1000`.
- Validation before training enabled; then `val_check_interval=200`.
- `limit_val_batches=32`.
- Train autoregressive metrics interval: `synthetic_chain_train_ar_interval=2000`.
- Tied token embeddings enabled.

Launch command used, with the W&B key intentionally omitted:

```bash
cd /lfs/skampere1/0/eshaanb/free-energy/experiments/12_ebt_test/EBT-main
export CUDA_VISIBLE_DEVICES=6
export WANDB_API_KEY="$WANDB_API_KEY"
../.venv/bin/python train_model.py \
  --run_name transformer-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu6-v4 \
  --modality NLP \
  --model_name baseline_transformer \
  --model_size 2xs \
  --dataset_name synthetic_chain \
  --tokenizer synthetic_chain \
  --context_length 128 \
  --tie_token_embeddings \
  --gpus 1 \
  --distributed_strategy auto \
  --batch_size_per_device 64 \
  --accumulate_grad_batches 1 \
  --train_metric_log_batch_size 64 \
  --num_workers 8 \
  --synthetic_chain_max_items 0 \
  --synthetic_chain_train_ar_interval 2000 \
  --peak_learning_rate 0.0005 \
  --warm_up_steps 1000 \
  --max_steps 100000 \
  --val_check_interval 200 \
  --limit_val_batches 32 \
  --validate_before_training \
  --gradient_clip_val 1.0 \
  --checkpoint_monitor_string val/loss \
  --checkpoint_monitor_mode min \
  --save_top_k_ckpts 2 \
  --wandb_project synthetic-chain-128-hop1-5 \
  --wandb_tags synthetic_chain transformer 2xs chain128 hops1-5 bs64 gpu6 v4 lr0.0005 \
  > logs/synthetic_chain_128_hops1_5_2xs/transformer-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu6-v4.log 2>&1
```

Latest observed metrics from the log:

- Latest train batch:
  - `train/loss=0.0787`
  - `train/teacher_forcing_token_accuracy=0.974`
  - `train/teacher_forcing_exact_accuracy=0.906`
- Last completed validation:
  - `val/loss=0.134`
  - `val/teacher_forcing_token_accuracy=0.945`
  - `val/teacher_forcing_exact_accuracy=0.780`
  - `val/autoregressive_token_accuracy=0.832`
  - `val/autoregressive_exact_accuracy=0.702`

Interpretation: the transformer baseline is healthy and mostly solves the 128-token, 1-5 hop dataset. It is much faster than EBT and currently much stronger on both teacher-forcing and autoregressive exact accuracy.
