# Toy EBM Training

Finite-support conditional EBM toy from Brando's Lean AI Club meeting notes.

This experiment trains scalar energy models over binary task/candidate pairs:

```text
p_theta(x | c) = exp(-E_theta(c, x)) / Z_theta(c).
```

Because the candidate space is finite and small, the partition function is
enumerated exactly. That makes the standard EBM positive-phase/negative-phase
training update directly testable before introducing approximate MCMC.

## Files

- `PLAN.md` - experiment plan and success criteria.
- `TRANSCRIPTION.md` - best-effort transcription of the handwritten notes.
- `assets/` - raw uploaded note photos.
- `run_toy_ebm.py` - PyTorch experiment runner.
- `test_toy_ebm.py` - smoke/unit tests.
- `run_smoke_test.sh` - CLI smoke experiment.
- `results/` - generated JSON/markdown reports.

## Run

```bash
cd /Users/brandomiranda/free-energy
.venv/bin/python -m pytest experiments/01_toy_ebm_training/test_toy_ebm.py
./experiments/01_toy_ebm_training/run_smoke_test.sh
.venv/bin/python experiments/01_toy_ebm_training/run_toy_ebm.py --tag real_exact
```

## SNAP

Run the full check on a SNAP host from the repo root:

```bash
./experiments/01_toy_ebm_training/run_on_snap.sh
```

Defaults target `skampere2.stanford.edu`, pin `CUDA_VISIBLE_DEVICES=0`, run
pytest, run the smoke check, run the real exact finite-support experiment, pull
`results/` back to the local repo, and email `brandojazz@gmail.com` when the
remote job finishes.

Useful overrides:

```bash
HOST=skampere1.stanford.edu GPU=4 ./experiments/01_toy_ebm_training/run_on_snap.sh
TAG=my_run DEVICE=cpu EMAIL_TO=me@example.com ./experiments/01_toy_ebm_training/run_on_snap.sh
```
