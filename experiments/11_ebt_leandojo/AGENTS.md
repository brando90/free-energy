# AGENTS.md — 11_ebt_leandojo

## What this repo is
Lean Workbook Plus pipeline for two things:
- precompute Goedel hidden states;
- run Goedel-Prover baseline + compile-based validation;
- train and validate EBT (`train_ebt.py`) using those hidden states.

## Top-level layout
- `configs/train_config.yaml` – Hydra defaults for EBT.
- `dataloader.py` – dataset + dataloader utilities (includes `--split`, `--chunk-size`, `--batch-size` smoke test).
- `train_ebt.py` – Hydra trainer for EBT.
- `precompute_states.py` – generate and save `*.safetensors` context activations.
- `run_goedel.py` + `helpers/benchmark.py` – benchmark generation + compile-eval helper.
- `helpers/lean_compile.py` + `helpers/lean_repl.py` – shared Lean REPL compile primitives.
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
- Precomputed hidden states dir currently used by train: `results/leandojo_hidden_states/hidden_states_safetensors`

## Important command set

### Quick sanity checks
- Test dataset/dataloader integration:
  - `python dataloader.py`

### Missing hidden-state bookkeeping
- Identify missing rows:
  - `python helpers/missing_states.py --out-dir results/leandojo_hidden_states --total-rows 25214`

### Precompute hidden states
- Full pass (default GPUs 0,1,2,3):
  - `python precompute_states.py`
- Resume from missing-row list:
  - `python precompute_states.py --out-dir results/leandojo_hidden_states --row-indices-file results/leandojo_hidden_states/missing_row_indices.json`
- Force rewrite:
  - `python precompute_states.py --out-dir ... --overwrite`

### Baseline benchmark / compile pass
- Default 500-item val generation (and compile eval):
  - `python run_goedel.py`
- Just fetch/download dataset+indices:
  - `python run_goedel.py --download-only`
- Sample new validation subset from another seed:
  - `python run_goedel.py --sample-validation --seed 3407 --val-size 500`

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
