# AGENTS.md — 11_ebt_leandojo

## What this repo is
Lean Workbook Plus pipeline for two things:
- prepare compact target data and vocab;
- precompute Goedel hidden states;
- run Goedel-Prover baseline + compile-based validation;
- train and validate EBT (`train_ebt.py`) using those hidden states.

## Top-level layout
- `configs/train_config.yaml` – Hydra defaults for EBT.
- `leanworkbook_dataloader.py` – dataset + dataloader utilities (includes `--split`, `--chunk-size`, `--batch-size` smoke test).
- `train_ebt.py` – Hydra trainer for EBT.
- `precompute_leanworkbook_hidden_states_sglang.py` – generate and save `*.safetensors` context activations.
- `run_goedel_leanworkbook_plus_sglang.py` + `leanworkbook_plus_benchmark.py` – benchmark generation + compile-eval helper.
- `data/clean_leanworkbook_targets.py` – clean raw formal statements into compact token ids.
- `data/validate_cleaned_lean_targets.py` – validate cleaned `target_text` by compiling with Lean.
- `lean_compile.py` + `lean_repl.py` – shared Lean REPL compile primitives.
- `results/`, `runs/` – existing artifacts and experiment outputs.

## Environment
- Python: >=3.11,<3.13.
- Current deps are in `pyproject.toml` (`uv` + `sglang`, `torch`, `transformers`, `datasets`, `hydra-core`, etc.).
- `.venv` exists in repo root; activate before running:
  - `source .venv/bin/activate`
- Common bootstrap with uv (if needed):
  - `uv sync`
- Fallback bootstrap with pip:
  - `pip install torch==2.6.* transformers==4.51.* datasets==2.20.* hydra-core==1.3.* sglang pexpect tqdm safetensors jsonlines wandb`

## Data pipeline (already prepared in this checkout)
- Raw data: `data/leanworkbook_plus_train.jsonl`
- Cleaned manifest: `data/context_gold/manifest.jsonl`
- Vocab: `data/context_gold/vocab.json`
- Val indices: `data/leanworkbook_plus_val500_indices.json` (500-row val split)
- Precomputed hidden states dir currently used by train: `results/leanworkbook_plus_goedel_hidden_states_gpus0_3_contextonly/hidden_states_safetensors`

## Important command set

### Quick sanity checks
- Test dataset/dataloader integration:
  - `python leanworkbook_dataloader.py`
- Preview cleaned rows:
  - `python data/clean_leanworkbook_targets.py --preview 3`

### Regenerate cleaned targets
- Full clean:
  - `python data/clean_leanworkbook_targets.py --input data/leanworkbook_plus_train.jsonl --out-dir data/context_gold --tokenizer Goedel-LM/Goedel-Prover-V2-8B`

### Validate cleaned targets in Lean
- Validate all (or subset):
  - `python data/validate_cleaned_lean_targets.py --data-dir data/context_gold --workers 16 --limit 0`
  - outputs: `results/cleaned_leanworkbook_validation_full/summary.json` and `compile_results.csv`

### Missing hidden-state bookkeeping
- Identify missing rows:
  - `python write_missing_hidden_state_indices.py --out-dir results/leanworkbook_plus_goedel_hidden_states_gpus0_3_contextonly --total-rows 25214`

### Precompute hidden states
- Full pass (default GPUs 0,1,2,3):
  - `python precompute_leanworkbook_hidden_states_sglang.py`
- Resume from missing-row list:
  - `python precompute_leanworkbook_hidden_states_sglang.py --out-dir results/leanworkbook_plus_goedel_hidden_states_gpus0_3_contextonly --row-indices-file results/leanworkbook_plus_goedel_hidden_states_gpus0_3_contextonly/missing_row_indices.json`
- Force rewrite:
  - `python precompute_leanworkbook_hidden_states_sglang.py --out-dir ... --overwrite`

### Baseline benchmark / compile pass
- Default 500-item val generation (and compile eval):
  - `python run_goedel_leanworkbook_plus_sglang.py`
- Just fetch/download dataset+indices:
  - `python run_goedel_leanworkbook_plus_sglang.py --download-only`
- Sample new validation subset from another seed:
  - `python run_goedel_leanworkbook_plus_sglang.py --sample-validation --seed 3407 --val-size 500`

### EBT training
- Default run (uses Hydra config in `configs/train_config.yaml`):
  - `python train_ebt.py`
- Exact run-equivalent overrides from existing sweep history:
  - `python train_ebt.py data.chunk_size=8 loader.batch_size=64 validation.every_steps=100 train.save_every_steps=1000 validation.progress_every=1 validation.random_sample_size=50 validation.random_sample_seed=17 wandb.name=leanworkbook_chunk8_h4096_l3_bs64_val50`
- Common alternative batch/chunk points used in this repo:
  - `... data.chunk_size=1 loader.batch_size=8`
  - `... data.chunk_size=32 loader.batch_size=96`
  - `... data.chunk_size=64 loader.batch_size=96`
- Outputs per run:
  - `runs/<generated_name>/config_resolved.yaml`
  - `runs/<generated_name>/metrics.jsonl`
  - `runs/<generated_name>/validation_metrics.jsonl`
  - `runs/<generated_name>/checkpoint_step_*.pt` and `checkpoint_final.pt`

## Run gotchas
- `train_ebt.py` currently does not implement resume-from-checkpoint loading; runs start fresh.
- `allow_cpu` defaults to false. Set `allow_cpu=true` only for debug CPU runs.
- Validation compiles Lean proofs during `validation.every_steps`; this is expensive and can add long stalls.
- Use a real `HF_TOKEN` to avoid anonymous Hub rate-limits during model/tokenizer fetch.
- OOM is plausible on larger `data.chunk_size`/`loader.batch_size` combinations; logs show memory-sensitive failures in earlier sweeps.
