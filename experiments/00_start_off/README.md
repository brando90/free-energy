# Start Off

Initial workspace for free-energy project notes, prompts, plans, literature traces, and rough experiment design.

Current notes:

- `ebm_mcmc_veribench_notes.md` - transcription and research plan from handwritten notes on Song and Kingma's EBM tutorial.
- `veribench_three_example_manifest.json` - first three VeriBench tasks for a tiny EBM/MCMC pilot.
- `pilot_ebm_ranking.py` - local smoke test that builds candidate pools and verifies the ranking/evaluation plumbing.
- `train_transformer_energy.py` - cluster starter script for fitting a transformer scalar energy model on the three-example pool.
- `run_mle_mcmc_experiment.py` - finite-support exact MLE vs MCMC-estimated EBM training on VeriBench candidate pools.
- `run_ebm_mle_sanity_check.py` - gradient-level check that the EBM negative phase matches finite-support MLE as sampling improves.
- `SNAP_RUNBOOK.md` - commands and workflow for running the pilot on the SNAP cluster.
- `agent_prompt_snap_cluster.md` - prompt to hand to a cluster agent.

Local smoke test:

```bash
cd /Users/brandomiranda/free-energy
./experiments/00_start_off/run_smoke_test.sh
```

Expected output: a pass message and JSON results in `experiments/00_start_off/results/smoke_test_rankings.json`.

SNAP full finite-support MLE/MCMC experiment:

```bash
cd ~/free-energy
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/00_start_off/run_mle_mcmc_experiment.py \
  --veribench-root ~/veribench \
  --output-dir experiments/00_start_off/results/mle_mcmc_full \
  --model-name microsoft/codebert-base \
  --subsets easy_set cs_set humaneval_set \
  --epochs 2 \
  --task-batch-size 4 \
  --max-length 256 \
  --mcmc-steps 10 \
  --device cuda
```

SNAP gradient-level EBM/MLE sanity check:

```bash
cd ~/free-energy
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/00_start_off/run_ebm_mle_sanity_check.py \
  --veribench-root ~/veribench \
  --output-dir experiments/00_start_off/results/ebm_mle_sanity \
  --model-name microsoft/codebert-base \
  --subsets easy_set cs_set humaneval_set \
  --task-batch-size 4 \
  --negative-batch-size 64 \
  --max-length 256 \
  --exact-sample-counts 1 4 16 64 \
  --mcmc-steps-list 0 1 5 25 100 \
  --mcmc-samples 64 \
  --device cuda
```

Use numbered folders under `experiments/` for concrete research threads:

- `00_start_off/` - project bootstrapping and first notes
- `01_<topic>/` - first concrete experiment or writeup
- `02_<topic>/` - next experiment or writeup

Keep polished, durable material in the repo root or `docs/`; keep scratch work, agent prompts, intermediate drafts, and exploratory runs here.
