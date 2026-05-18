# SNAP Three-Example EBM Pilot Report

TL;DR: local smoke-test plumbing passes. SNAP transformer run is the next step.

## Local Smoke Test

Command:

```bash
cd /Users/brandomiranda/free-energy
./experiments/00_start_off/run_smoke_test.sh
```

Outcome:

```text
easy_set/1_MyAdd: gold_rank=1 num_candidates=7 top=gold
easy_set/21_is_palindrome: gold_rank=1 num_candidates=7 top=gold
cs_set/binary_search: gold_rank=1 num_candidates=7 top=gold
status=pass
```

Artifact:

```text
experiments/00_start_off/results/smoke_test_rankings.json
```

Important caveat: this is only a path/candidate-pool smoke test. The toy energy uses a source prior, so it is not a scientific result.

## Local Transformer Test

Command:

```bash
cd /Users/brandomiranda/free-energy
.venv/bin/python experiments/00_start_off/train_transformer_energy.py \
  --veribench-root "$HOME/veribench" \
  --output-dir experiments/00_start_off/results/local_codebert_cpu \
  --model-name microsoft/codebert-base \
  --epochs 1 \
  --batch-size 1 \
  --max-length 128 \
  --device cpu
```

Outcome:

```text
model=microsoft/codebert-base
device=cpu
epochs=1
num_pairs=18
loss=0.6112
mean_reciprocal_rank=0.8333
all_gold_top=false
```

Final local ranking summary:

| Task | Gold rank | Top candidate | Notes |
|---|---:|---|---|
| `easy_set/1_MyAdd` | 1 | `gold` | learned model ranks gold first |
| `easy_set/21_is_palindrome` | 1 | `gold` | learned model ranks gold first |
| `cs_set/binary_search` | 2 | `corrupt_no_imports` | useful failure: model is not yet verifier-aware and can prefer an invalid import-stripped file |

Artifact kept locally but ignored by git:

```text
experiments/00_start_off/results/local_codebert_cpu/training_report.json
```

## SNAP Transformer Run

Status: completed on `skampere1.stanford.edu`.

Command:

```bash
cd ~/free-energy
source .venv/bin/activate
export VERIBENCH_ROOT=${VERIBENCH_ROOT:-$HOME/veribench}

python experiments/00_start_off/train_transformer_energy.py \
  --veribench-root "$VERIBENCH_ROOT" \
  --model-name microsoft/codebert-base \
  --epochs 5 \
  --batch-size 2 \
  --max-length 512 \
  --device cuda
```

Outcome:

```text
host=skampere1.stanford.edu
gpu=A100 via CUDA_VISIBLE_DEVICES=0
model=microsoft/codebert-base
device=cuda
epochs=5
num_pairs=18
loss=0.2386
mean_reciprocal_rank=1.0000
all_gold_top=true
```

Artifact on SNAP:

```text
experiments/00_start_off/results/transformer_energy/training_report.json
```

## Final Ranking Table

| Task | Gold rank | Top candidate | Notes |
|---|---:|---|---|
| `easy_set/1_MyAdd` | 1 | `gold` | gold ranks first |
| `easy_set/21_is_palindrome` | 1 | `gold` | gold ranks first |
| `cs_set/binary_search` | 1 | `gold` | gold ranks first on SNAP after 5 epochs |

## Next Step

After the transformer run works, add short-run local-edit or LLM-proposal negatives and compare whether they produce harder negatives than static corruptions.

Bottom TL;DR: the experiment is wired; now run the transformer energy script on SNAP and inspect whether learned energy ranks gold/verifier-passing artifacts above generated and corrupted candidates.
