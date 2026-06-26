# Results

## Goedel-Prover-V2-8B evaluation (parallel compile, 32 workers)

- Run: `evaluate_veribench_outputs_from_safetensors.py`
- Run date: `2026-06-24`
- Workers: `32`
- Input manifest: `experiments/10_ebt_vb/data/context_gold/manifest.jsonl`
- Source outputs: `experiments/09_vb_testing_ipynb/results/goedel_prover_v2_8b_sglang_896_hidden_states_gpus0_5_full_884_bs16`
- Output directory: `experiments/10_ebt_vb/results/veribench_output_eval_workers32`

The input manifest currently contains **689** rows (train/val/test split counts are 550/65/74).

| Split | Rows | IC1 | IC2 | TE1 | D1 | D2 | S_tilde |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 550 | 0.120000 | 0.005455 | 0.000000 | 0.970909 | 0.926571 | 0.000000 |
| val | 65 | 0.076923 | 0.000000 | 0.000000 | 0.984615 | 0.936190 | 0.000000 |
| test | 74 | 0.094595 | 0.000000 | 0.000000 | 0.986486 | 0.921686 | 0.000000 |
| all | 689 | 0.113208 | 0.004354 | 0.000000 | 0.973875 | 0.926954 | 0.000000 |

## Output files

- `experiments/10_ebt_vb/results/veribench_output_eval_workers32/veribench_output_scores.csv`
- `experiments/10_ebt_vb/results/veribench_output_eval_workers32/veribench_output_aggregate.csv`

## EBT validation-overfit trainer

- Run date: `2026-06-24`
- Trainer: `experiments/10_ebt_vb/train_ebt.py`
- Config: `experiments/10_ebt_vb/configs/train_config.yaml`
- Run directory: `experiments/10_ebt_vb/runs/val_overfit/20260624_154937`
- Dataset split: `val`
- Validation rows: `65`
- Vocab: compact VeriBench dataset vocab, `6724` tokens
- Target cap: `2048` tokens per sample
- Steps: `1000`
- GPU: one A100 via `CUDA_VISIBLE_DEVICES=0`

The initial full-Goedel-vocab setup used dense EBT states shaped like `[batch, target_tokens, 151669]`.
That OOMed on one 80GB A100, including with batch size 1, because validation contains long gold Lean targets
up to `8100` tokens. The trainer now defaults to the compact VeriBench token vocabulary and trains on capped
gold-prefix targets so every validation task is still represented.

| Metric | Value |
|---|---:|
| Peak CUDA memory | 45.55 GB |
| Sampled GPU utilization | 89% |
| Best logged loss | 9.1331 |
| Best-loss step | 840 |
| Best logged final-step loss | 9.0984 |
| Final logged loss | 9.3183 |
| Final logged final-step loss | 9.3241 |
| Final logged perplexity | 11205.24 |
| Final logged throughput | 19501.16 target tokens/sec |

Artifacts:

- `experiments/10_ebt_vb/runs/val_overfit/20260624_154937/metrics.jsonl`
- `experiments/10_ebt_vb/runs/val_overfit/20260624_154937/summary.json`
- `experiments/10_ebt_vb/runs/val_overfit/20260624_154937/checkpoint_final.pt`

Conclusion: the trainer is functional and stable on one A100, but this default run did not strongly overfit
the validation split. Loss improved modestly around the best checkpoint and then regressed by the final logged
batch.

## How the EBT overfit experiments are run

The recent overfit checks use the cloned Alexi EBT code under:

- `experiments/10_ebt_vb/alexiglad_EBT/train_veribench_overfit.py`
- `experiments/10_ebt_vb/alexiglad_EBT/sample_veribench_overfit.py`

Common setup:

- Dataset: first `5` examples from the VeriBench `val` split.
- Compact target vocabulary: `6724` local VeriBench/Goedel tokens from `data/context_gold/vocab.json`.
- Target format: decoder input is `BOS + gold Lean prefix`; labels are the corresponding next gold Lean tokens.
- Sampling: autoregressive greedy decoding with `temperature=0`, using local token ids and decoding back through the Goedel tokenizer mapping.
- Goedel context ablation: `--no-context` zeros the projected Goedel hidden-state conditioning, so the model sees only the decoder sequence.
- Architecture unless otherwise stated: `6` EBT transformer layers, `6` heads, `hidden_dim=384`.
- Optimizer settings unless otherwise stated: `AdamW`, `lr=0.0012`, alpha lr multiplier `1.5`, learnable MCMC step size, `mcmc_steps=2`.
- Stochasticity: training uses random model initialization and random dense-vocab MCMC initial conditions; greedy sampling is deterministic apart from CUDA nondeterminism.

## Alexi EBT 5-example overfit and sampling results

These runs were intended as sanity checks: if the EBT setup is healthy, it should memorize five validation examples, especially when the gold proof is truncated to five tokens.

### Full 256-token targets with Goedel context

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps ... --max-items 5 --max-target-tokens 256 --batch-size 5 \
  --hidden-dim 384 --num-layers 6 --num-heads 6
```

| Run | Variant | Best logged loss | Sampled checkpoint | Exact recovered | Token matches |
|---|---|---:|---:|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_124131` | normal context-conditioned | 9.2092 | 600 | 0/5 | 1/1280 |
| `runs/alexiglad_ebt_veribench5/20260625_124614` | no MCMC detach, larger MCMC step | 9.2355 | 300 | 0/5 | 0/1280 |

Conclusion: the context-conditioned EBT did not recover the 5 examples. Generations were random-looking tokenizer fragments, not Lean code.

### Larger hidden dimension

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps 1000 --max-items 5 --max-target-tokens 256 --batch-size 5 \
  --hidden-dim 1024 --num-layers 6 --num-heads 8
```

`1024` is not divisible by `6`, so this run used `8` heads.

| Run | Best logged loss | Sampled checkpoints | Exact recovered | Token matches |
|---|---:|---|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_132907` | 9.2288 | 200, 600 | 0/5, 0/5 | 0/1280, 0/1280 |

Conclusion: increasing hidden size to `1024` did not improve recovery.

### Full 256-token targets without Goedel context

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps 1000 --max-items 5 --max-target-tokens 256 --batch-size 5 \
  --hidden-dim 384 --num-layers 6 --num-heads 6 --no-context
```

| Run | Best logged loss | Sampled checkpoints | Exact recovered | Token matches |
|---|---:|---|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_134438` | 9.2006 | 100, 400 | 0/5, 0/5 | 0/1280, 0/1280 |

Conclusion: removing Goedel activations did not fix the failure to memorize.

### Five-token target truncation without Goedel context

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --max-items 5 --max-target-tokens 5 --batch-size 5 \
  --hidden-dim 384 --num-layers 6 --num-heads 6 --no-context
```

| Run | Steps | Best logged loss | Sampled checkpoints | Exact recovered | Token matches |
|---|---:|---:|---|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_135002` | 368 interrupted | 8.7603 | 200, 300 | 0/5, 0/5 | 0/25, 0/25 |
| `runs/alexiglad_ebt_veribench5/20260625_135231` | 10000 | 4.4294 at step 9965 | 7000, 10000 | 0/5, 0/5 | 4/25, 6/25 |

The 10k-step run used exactly:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps 10000 --max-items 5 --max-target-tokens 5 --batch-size 5 \
  --log-every 100 --save-every 1000 \
  --hidden-dim 384 --num-layers 6 --num-heads 6 \
  --warmup-steps 50 --lr 0.0012 --no-context
```

10k-step details:

| Metric | Value |
|---|---:|
| Final loss | 7.7497 |
| Final final-step loss | 7.7151 |
| Best observed loss | 4.4294 |
| Best observed final-step loss | 4.1443 |
| Best observed step | 9965 |
| Best saved checkpoint by loss | 7000 |
| Peak CUDA memory | 0.38 GB |

Saved sampling outputs:

- `runs/alexiglad_ebt_veribench5/20260625_135231/sample_checkpoint_step_007000.jsonl`
- `runs/alexiglad_ebt_veribench5/20260625_135231/sample_checkpoint_step_010000.jsonl`

Conclusion: longer training improved token-level recovery from `0/25` to `6/25`, but still did not recover any complete 5-token example. Outputs mostly repeated frequent Lean-ish tokens such as `import`, `Std`, and `_option`.

## Alexi EBT limited language-modeling reproduction

This checks whether the upstream Alexi EBT language-modeling code path works in this environment on a tiny local subset. It is not a paper-scale reproduction of FineWeb results.

Setup:

- Script: `experiments/10_ebt_vb/alexiglad_EBT/train_lm_subset.py`
- Run directory: `experiments/10_ebt_vb/runs/alexiglad_lm_subset/20260625_171232`
- Models: `Baseline_Transformer_NLP` and `EBT_NLP` from the cloned Alexi repo.
- Tokenizer: `EleutherAI/gpt-neox-20b`
- Data: 8 local English sentences, tokenized into sliding windows.
- Train subset: 64 windows from the first 6 sentences.
- Validation subset: 16 windows from 2 held-out sentences.
- Context length: 32.
- Steps: 80.
- Batch size: 4.
- GPU: one A100 via `CUDA_VISIBLE_DEVICES=0`.
- EBT S1 settings: `hidden_dim=384`, `6` layers, `6` heads, `mcmc_steps=2`, `mcmc_step_size=0.5`, learnable alpha, `lr=0.0012`, alpha LR multiplier `1.5`.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python alexiglad_EBT/train_lm_subset.py \
  --steps 80 --train-samples 64 --val-samples 16 \
  --context-length 32 --batch-size 4 \
  --models baseline_transformer ebt \
  --log-every 10 --eval-every 20 --eval-batches 4
```

Results:

| Model | Train loss step 1 | Final train loss | Final train PPL | Best val final-step loss | Final val final-step loss |
|---|---:|---:|---:|---:|---:|
| Baseline Transformer | 10.8231 | 0.0051 | 1.0051 | 10.8053 | 13.1697 |
| EBT | 11.9499 | 0.1294 | 1.2081 | 10.9928 | 14.1864 |

Artifacts:

- `runs/alexiglad_lm_subset/20260625_171232/baseline_transformer_metrics.jsonl`
- `runs/alexiglad_lm_subset/20260625_171232/ebt_metrics.jsonl`
- `runs/alexiglad_lm_subset/20260625_171232/baseline_transformer_final.pt`
- `runs/alexiglad_lm_subset/20260625_171232/ebt_final.pt`
- `runs/alexiglad_lm_subset/20260625_171232/summary.json`

Conclusion: on a deliberately tiny language-modeling subset, both Alexi LM models learn the training windows quickly. The EBT path drops from train loss `11.9499` to `0.1294` by step 80, confirming that the upstream `EBT_NLP` language-modeling objective and MCMC inference loop are functional in this environment. Validation loss rises because the validation sentences are held-out tiny text snippets and the run is intentionally an overfit/mechanics check, not a generalization benchmark.

### Sampling from the LM subset checkpoints

Script:

- `experiments/10_ebt_vb/alexiglad_EBT/sample_lm_subset.py`

Greedy command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python alexiglad_EBT/sample_lm_subset.py \
  --run-dir runs/alexiglad_lm_subset/20260625_171232 \
  --max-new-tokens 32 --temperature 0 \
  --models baseline_transformer ebt
```

Top-p command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python alexiglad_EBT/sample_lm_subset.py \
  --run-dir runs/alexiglad_lm_subset/20260625_171232 \
  --max-new-tokens 32 --temperature 0.8 --top-p 0.95 \
  --models baseline_transformer ebt \
  --output runs/alexiglad_lm_subset/20260625_171232/samples_t0.8_top0.95.json
```

Greedy sampling examples:

| Model | Prompt | Continuation summary |
|---|---|---|
| Baseline Transformer | `Lean programs` | Reproduces the training sentence: `describe definitions, theorems, and proofs...`, then continues into the next training sentence. |
| Baseline Transformer | `Energy based transformers` | Reproduces the training sentence about refining dense token predictions, then continues into the small-reproduction sentence. |
| EBT | `Lean programs` | Starts with the correct training continuation, then becomes repetitive/noisy after roughly one sentence. |
| EBT | `Energy based transformers` | Reproduces the full training continuation cleanly through the small-reproduction sentence. |
| EBT | `Formal verification` | Noisy and repetitive; does not cleanly reproduce the intended training sentence. |

Saved sampling outputs:

- `runs/alexiglad_lm_subset/20260625_171232/samples_t0.0_top0.95.json`
- `runs/alexiglad_lm_subset/20260625_171232/samples_t0.8_top0.95.json`

Conclusion: sampling confirms the train-overfit result. The baseline transformer memorizes the tiny corpus almost verbatim. The EBT checkpoint has learned recognizable training continuations for some prompts, but generations are less stable and can become repetitive or off-distribution under both greedy and top-p sampling.

## Local EBT language-modeling wrapper check

This checks whether the local `experiments/10_ebt_vb/ebt.py` implementation can reproduce the same tiny language-modeling overfit behavior as Alexi's upstream `EBT_NLP`.

Initial diagnosis:

- Alexi's `EBT_NLP` performs the MCMC update on the normalized dense token state after `softmax`.
- The local `LocalLanguageEBT` was computing gradients through `softmax` back to pre-softmax logits, so the MCMC update was effectively zero.
- One-batch local gradient norm before the fix was about `5.5e-4`; after matching Alexi's normalized-state update it was about `24.8`.
- The MCMC state update changed from roughly `1.9e-5` mean absolute delta to roughly `1.07`, matching the Alexi code path's scale.

Patch:

- `experiments/10_ebt_vb/ebt.py`
- `LocalLanguageEBT.forward` now applies `softmax` to the dense state before the energy call and updates that normalized state, matching Alexi `EBT_NLP.forward`.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python train_local_lm_subset.py \
  --steps 80 --train-samples 64 --val-samples 16 \
  --context-length 32 --batch-size 4 \
  --hidden-dim 384 --layers 6 --heads 6 \
  --mcmc-steps 2 --mcmc-step-size 0.5 \
  --lr 0.0012 --log-every 10 --eval-every 20 --eval-batches 4
```

Run directory:

- `runs/local_lm_subset/20260625_173612`

Results:

| Metric | Value |
|---|---:|
| Step 1 train loss | 11.5587 |
| Step 1 final-step train loss | 11.3410 |
| Final train loss | 0.2929 |
| Final final-step train loss | 0.4813 |
| Final train perplexity | 1.6182 |
| Final validation final-step loss | 15.8020 |
| Peak CUDA memory | 1.56 GB |

Sampling examples from the fixed local wrapper:

| Prompt | Continuation summary |
|---|---|
| `Lean programs` | Reproduces the training sentence about definitions/theorems/proofs, then continues into the language-model sentence. |
| `Energy based transformers` | Reproduces the training sentence and continues into the small-reproduction sentence cleanly. |
| `The baseline transformer` | Partially memorized but noisy and repetitive. |
| `Formal verification` | Starts close to the target sentence but becomes repetitive around `pass`/`EBT` tokens. |

Artifacts:

- `runs/local_lm_subset/20260625_173612/checkpoint_final.pt`
- `runs/local_lm_subset/20260625_173612/metrics.jsonl`
- `runs/local_lm_subset/20260625_173612/samples.json`
- `runs/local_lm_subset/20260625_173612/summary.json`

Conclusion: after matching Alexi's MCMC update semantics, the local EBT wrapper overfits the tiny language-modeling subset. This confirms the earlier local failure was an implementation mismatch, not evidence that the EBT objective cannot overfit this toy LM setup.

## GoedelVocabEBT 5-example VeriBench overfit sanity checks

This is the active local VeriBench EBT path after removing the temporary `LocalLanguageEBT` class/script. The normalized-state MCMC fix was moved into `GoedelVocabEBT`.

Setup:

- Trainer: `experiments/10_ebt_vb/train_ebt.py`
- Dataset wrapper: `experiments/10_ebt_vb/veribench_embedding_dataloader.py`
- Task wrapper: `experiments/10_ebt_vb/veribench_task.py`
- Split: `val`
- Examples: first 5 validation tasks
- EBT hidden size: `384`
- Heads/layers: `6` heads, `6` layers
- Goedel activations: disabled for this sanity check
- Context used instead: learned per-task embedding from `task_index`
- Dense-state initialization: `zeros`
- MCMC: `2` steps, learnable step size initialized at `0.5`
- LR: `0.0012`, alpha LR multiplier `1.5`

Important implementation changes:

- `GoedelVocabEBT` now updates the normalized dense token state after `softmax`, matching Alexi `EBT_NLP` semantics.
- `GoedelVocabEBT` can ignore Goedel activations via `model.use_context_activations=false`.
- When activations are disabled, it conditions on a learned task embedding, otherwise five different targets would be indistinguishable.
- Added learned target-position embeddings to the PyTorch decoder path. Without this, the 50-token run plateaued around `65-70%` token accuracy.
- `train_ebt.py` logs final token/exact accuracy and supports early stopping on exact accuracy.

Results:

| Target cap | Early stop step | Token acc | Exact acc | Final-step loss | Peak mem |
|---:|---:|---:|---:|---:|---:|
| 5 tokens | 887 | 1.000 | 1.000 | 0.5090 | 0.27 GB |
| 50 tokens | 224 | 1.000 | 1.000 | 0.0574 | 0.42 GB |
| Full targets | 484 | 1.000 | 1.000 | 0.0025 | 8.24 GB |

Run directories:

- `runs/goedel_vocab_taskctx_384_overfit5_tok5_zero/20260625_195231`
- `runs/goedel_vocab_taskctx_pos_384_overfit5_tok50_zero/20260625_200443`
- `runs/goedel_vocab_taskctx_pos_384_overfit5_full_zero/20260625_200540`

Conclusion: with `384`-d EBT state, no Goedel activation conditioning, learned task context, normalized-state MCMC updates, and target-position embeddings, the active `GoedelVocabEBT` training path overfits the 5-example VeriBench validation subset at 5 tokens, 50 tokens, and full target length.

## GoedelVocabEBT 4096-d Goedel-activation overfit checks

This repeats the 5-example VeriBench overfit ladder with real Goedel prompt activations enabled and `hidden_dim=4096`. No learned task-index conditioning is used in these runs; conditioning comes from the stored Goedel V2 prompt hidden states through cross-attention.

Shared setup:

- Split: `val`
- Examples: first 5 validation tasks
- `model.use_context_activations=true`
- `model.hidden_dim=4096`
- `model.context_dim=4096`
- `model.num_heads=32`
- `model.denoising_initial_condition=zeros`
- `model.mcmc_num_steps=2`
- `model.mcmc_step_size=0.5`
- Dense-state MCMC update: normalized-state update in `GoedelVocabEBT`
- Target positions: learned target-position embeddings in `GoedelVocabEBT`

What improved convergence:

- **Target-position embeddings**: the earlier PyTorch decoder path had no target positional signal. Adding learned target positions made 50-token and full-target memorization possible.
- **Normalized-state MCMC update**: the dense token state is softmax-normalized before the energy gradient update, matching the Alexi EBT NLP behavior.
- **For 50/full targets, one decoder layer**: the 4-layer 4096-d decoder was unstable or slow for this tiny overfit. A 1-layer decoder retained real Goedel conditioning while making the optimization tractable.
- **Final-step-only loss for 50/full targets**: optimizing the final MCMC prediction directly worked better than averaging both MCMC-step losses.
- **Full target stabilization**: the full-target run needed `lr=1e-4`, alpha LR multiplier `1.0`, and grad clip `0.25`; the higher-LR full run reached about `86%` token accuracy, then collapsed.

Results:

| Target cap | Early stop step | Token acc | Exact acc | Final-step loss | Peak mem |
|---:|---:|---:|---:|---:|---:|
| 5 tokens | 917 | 1.000 | 1.000 | 0.0403 | 14.98 GB |
| 50 tokens | 4165 | 1.000 | 1.000 | 0.0666 | 3.94 GB |
| Full targets | 3548 | 1.000 | 1.000 | 0.0013 | 14.15 GB |

### 5-token command

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python train_ebt.py \
  data.split=val data.max_items=5 data.max_target_tokens=5 \
  loader.batch_size=5 loader.max_target_tokens_per_batch=null \
  loader.shuffle=true loader.num_workers=0 loader.persistent_workers=false \
  model.hidden_dim=4096 model.context_dim=4096 model.num_heads=32 \
  model.num_layers=4 model.dim_feedforward=8192 \
  model.use_context_activations=true model.denoising_initial_condition=zeros \
  model.mcmc_num_steps=2 model.mcmc_step_size=0.5 \
  model.mcmc_step_size_learnable=true \
  optim.use_scheduler=false optim.lr=0.0003 optim.weight_decay=0.01 \
  train.max_steps=20000 train.grad_accum_steps=1 train.log_every=50 \
  train.save_every=1000 train.keep_last_checkpoints=1 \
  train.stop_on_exact_accuracy=1.0 \
  hydra.run.dir='runs/goedel_acts_4096_overfit5_tok5_zero/${now:%Y%m%d_%H%M%S}'
```

Run directory:

- `runs/goedel_acts_4096_overfit5_tok5_zero/20260625_201044`

### 50-token command

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python train_ebt.py \
  data.split=val data.max_items=5 data.max_target_tokens=50 \
  loader.batch_size=5 loader.max_target_tokens_per_batch=null \
  loader.shuffle=true loader.num_workers=0 loader.persistent_workers=false \
  model.hidden_dim=4096 model.context_dim=4096 model.num_heads=32 \
  model.num_layers=1 model.dim_feedforward=8192 \
  model.use_context_activations=true model.denoising_initial_condition=zeros \
  model.mcmc_num_steps=2 model.mcmc_step_size=0.5 \
  model.mcmc_step_size_learnable=true model.loss_on_final_step_only=true \
  model.truncate_mcmc=false \
  optim.use_scheduler=false optim.lr=0.0003 optim.weight_decay=0.01 \
  train.max_steps=30000 train.grad_accum_steps=1 train.log_every=100 \
  train.save_every=5000 train.keep_last_checkpoints=1 \
  train.stop_on_exact_accuracy=1.0 \
  hydra.run.dir='runs/goedel_acts_4096_l1_overfit5_tok50_zero_finalonly/${now:%Y%m%d_%H%M%S}'
```

Run directory:

- `runs/goedel_acts_4096_l1_overfit5_tok50_zero_finalonly/20260625_203141`

### Full-target command

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python train_ebt.py \
  data.split=val data.max_items=5 data.max_target_tokens=null \
  loader.batch_size=5 loader.max_target_tokens_per_batch=null \
  loader.shuffle=true loader.num_workers=0 loader.persistent_workers=false \
  model.hidden_dim=4096 model.context_dim=4096 model.num_heads=32 \
  model.num_layers=1 model.dim_feedforward=8192 \
  model.use_context_activations=true model.denoising_initial_condition=zeros \
  model.mcmc_num_steps=2 model.mcmc_step_size=0.5 \
  model.mcmc_step_size_learnable=true model.loss_on_final_step_only=true \
  model.truncate_mcmc=false \
  optim.use_scheduler=false optim.lr=0.0001 \
  optim.mcmc_step_size_lr_multiplier=1.0 optim.weight_decay=0.01 \
  train.grad_clip_norm=0.25 train.max_steps=80000 \
  train.grad_accum_steps=1 train.log_every=100 \
  train.save_every=5000 train.keep_last_checkpoints=1 \
  train.stop_on_exact_accuracy=1.0 \
  hydra.run.dir='runs/goedel_acts_4096_l1_overfit5_full_zero_finalonly_lr1e4_clip025/${now:%Y%m%d_%H%M%S}'
```

Run directory:

- `runs/goedel_acts_4096_l1_overfit5_full_zero_finalonly_lr1e4_clip025/20260625_205124`

Conclusion: with real Goedel prompt activations and `4096`-d EBT state, `GoedelVocabEBT` overfits the 5-example validation subset at 5 tokens, 50 tokens, and full target length.

## EBT validation sampling and VeriBench metrics

Note that I compacted + cleaned the veribench repo a lot to ~3k unique tokens (as opposed to ~105k unique tokens!)

Evaluated the two most recent 7k-token, 4-head EBT checkpoints by sampling on the `val` split and scoring generated Lean with `VeriBenchTask.evaluate_lean_output(..., skip_te1=true)`. TE1 was intentionally skipped, so TE1 and `S_tilde` are reported as `0.0` by the evaluator.

Sampling/evaluation command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python helper/sample_and_evaluate_ebt_val.py \
  --workers 16 \
  --compile-timeout 300 \
  --out-dir results/ebt_val_sampling_eval
```

Checkpoints:

- `val_train`: `runs/target7k_4h_bs2_val_1m_20260625_232609/checkpoint_step_9500.pt`
- `train_test_train`: `runs/target7k_4h_bs2_train_test_1m_20260625_232609/checkpoint_step_10000.pt`

Artifacts:

- `results/ebt_val_sampling_eval/aggregate.csv`
- `results/ebt_val_sampling_eval/scores.csv`
- `results/ebt_val_sampling_eval/val_train/samples.jsonl`
- `results/ebt_val_sampling_eval/train_test_train/samples.jsonl`

Results on `val`:

| Run | Rows | Exact tokens | Token acc | IC1 | IC2 | TE1 | D1 | D2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `val_train` | 65 | 65/65 | 1.0000 | 1.0000 | 0.8429 | 0.0000 | 1.0000 | 0.9362 |
| `train_test_train` | 65 | 0/65 | 0.0551 | 0.5538 | 0.0000 | 0.0000 | 1.0000 | 0.9362 |

Interpretation: `val_train` exactly reproduces the cleaned/anonymized validation target tokens, but only 57/65 compile because the cleaned validation targets themselves are not all compile-valid under the current Lean validation setup. Thus token exactness and Lean compile pass measure different things here.
