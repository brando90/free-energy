# Start Off

Initial workspace for free-energy project notes, prompts, plans, literature traces, and rough experiment design.

Current notes:

- `ebm_mcmc_veribench_notes.md` - transcription and research plan from handwritten notes on Song and Kingma's EBM tutorial.
- `veribench_three_example_manifest.json` - first three VeriBench tasks for a tiny EBM/MCMC pilot.
- `pilot_ebm_ranking.py` - local smoke test that builds candidate pools and verifies the ranking/evaluation plumbing.
- `train_transformer_energy.py` - cluster starter script for fitting a transformer scalar energy model on the three-example pool.
- `SNAP_RUNBOOK.md` - commands and workflow for running the pilot on the SNAP cluster.
- `agent_prompt_snap_cluster.md` - prompt to hand to a cluster agent.

Local smoke test:

```bash
cd /Users/brandomiranda/free-energy
./experiments/00_start_off/run_smoke_test.sh
```

Expected output: a pass message and JSON results in `experiments/00_start_off/results/smoke_test_rankings.json`.

Use numbered folders under `experiments/` for concrete research threads:

- `00_start_off/` - project bootstrapping and first notes
- `01_<topic>/` - first concrete experiment or writeup
- `02_<topic>/` - next experiment or writeup

Keep polished, durable material in the repo root or `docs/`; keep scratch work, agent prompts, intermediate drafts, and exploratory runs here.
