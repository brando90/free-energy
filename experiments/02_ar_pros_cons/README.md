# `ar_claims` — Do the theoretical objections to autoregressive LLMs actually hold?

ref: https://claude.ai/chat/8e0e2e24-18f9-4eff-948e-9a5340fc0fce

A two-level empirical suite that tests *which* of the standard objections to
autoregressive / softmax / MLE language models are real, how strong each one is,
and — critically — which ones still **bite when everything is trained together**
end-to-end on real Lean 4 sequences (VeriBench / VeriBench-FTP).

The point is not to argue against LLMs. It is to separate claims that are
**theorems** or **robust empirical findings** from claims that are popular but
weak, and to measure realized effect sizes in the setting we actually care about
(formal verification with a hard verifier).

---

## The core idea: isolate, then integrate

Each objection lives at a different layer of the system. The single most common
way people fool themselves is to test a claim at the wrong layer — e.g. "proving"
rank collapse on a trained model where residuals have already masked it, or
"disproving" the softmax bottleneck on data that happens to be low-rank.

So the suite has two levels:

1. **Isolated probes** (`probes/`) — measure each mechanism's *latent strength*
   at its correct layer, with a positive control where the effect **must** appear
   if the theory is true. If the control fails, the probe is broken, not the theory.

2. **Integrated harness** (`integrated/`) — train real autoregressive models
   end-to-end on VeriBench, log every probe as an *online metric*, and run a
   **factorial ablation grid** so each mechanism's *realized downstream effect* is
   causally attributable rather than merely correlated.

The deliverable that ties them together is the **bridge analysis**
(`integrated/BRIDGE.md`): plot each mechanism's isolated strength against its
realized effect. Three outcomes, all informative:

| Outcome | Meaning |
|---|---|
| strength predicts effect | mechanism is **load-bearing** |
| strong but no effect | **masked** in practice (say by what, e.g. residuals) |
| weak but large effect | an **interaction** is doing the work (identify the pair) |

---

## Layer discipline (read this before running anything)

Every probe is tagged with the layer it tests. Never mix layers in one claim.

| Layer | Needs | Example claims |
|---|---|---|
| **architecture** | random-init model, no training, no data | softmax bottleneck, rank collapse, fixed-compute ceiling |
| **objective** | a controlled training run | mode-covering / forward-KL |
| **trained behavior** | trained model + Lean verifier | error compounding, reversal curse, brittleness |
| **external** | a resource curve, not a model | data wall |

---

## The claims under test

Full detail and strength ratings in [`docs/CLAIMS.md`](docs/CLAIMS.md).

| # | Claim | Layer | Prior strength |
|---|---|---|---|
| 1 | Softmax bottleneck (rank ≤ d+1) | architecture + data | **theorem** |
| 2 | Mode-covering (forward-KL is zero-avoiding) | objective | **theorem** |
| 3 | Rank collapse with depth | architecture | **theorem** (pure attn) |
| 4 | Partition function is a removable per-step tax | architecture (ablation) | strong |
| 5 | Fixed compute per token is a representational ceiling | architecture + complexity | strong |
| 6 | Error compounding, the (1−e)ⁿ argument | trained behavior + verifier | **weak — under test** |
| 7 | Reversal curse | trained behavior | empirical, reproduced |
| 8 | Brittleness / Lipschitz–margin | trained behavior | empirical + theory |
| — | Data wall | external | empirical |

Claims explicitly **excluded** as weak/non-operationalizable (and why) are listed
in `docs/CLAIMS.md` so the exclusion is part of the argument, not an omission.

---

## Repository layout

```
ar_claims/
├── README.md                  ← you are here
├── FINDINGS.md                ← living results log; update as runs complete
├── docs/
│   ├── CLAIMS.md              ← the claims, strength ratings, what each motivates
│   └── METHODOLOGY.md         ← layer separation, controls, stats protocol
├── probes/
│   ├── README.md              ← isolated-probe suite overview + run order
│   └── PROBE_SPECS.md         ← all 8 probes: prediction / null / measurement
├── integrated/
│   ├── README.md              ← integrated harness + factorial ablation grid
│   └── BRIDGE.md              ← isolated-strength ↔ realized-effect analysis
└── data/
    └── VERIBENCH.md           ← dataset + Lean verifier setup, smoke subset
```

---

## Quickstart

> The architecture-only probes (01, 03, 05) are written first and run end-to-end
> on **CPU in seconds** or on **a single H200 in well under a minute**. Validate
> their positive controls before scaling to the integrated grid.

### Local (CPU smoke)

```bash
# 0. environment (uv or venv, your call)
cd experiments/02_ar_pros_cons
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. probe controls (no data, no verifier)
python -m probes.probe_01_softmax_bottleneck --smoke
python -m probes.probe_03_rank_collapse      --smoke
python -m probes.probe_05_fixed_compute      --smoke
# or all at once:
python -m probes.run_all --smoke

# 2. results land in mnt/user-data/outputs/<probe>/smoke/result.json
```

### Snap cluster (skampere2 by default)

The helper script `run_on_snap.sh` rsyncs the experiment to a snap host, sets up
a uv venv, runs the probes on a GPU, and rsyncs the JSON outputs back.

It expects `~/keys/skampere_password.txt` (snap NFS password — see
[ilwiki snap-servers](https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers))
and `sshpass` on the local box.

```bash
# smoke run on skampere2 GPU 0
./run_on_snap.sh smoke

# full run on skampere1, pinning to GPU 4
HOST=skampere1.stanford.edu GPU=4 ./run_on_snap.sh full

# explicit env overrides
HOST=skampere3.stanford.edu DEVICE=cuda TAG=mytest SEED=7 ./run_on_snap.sh
```

Outputs land back at
`experiments/02_ar_pros_cons/mnt/user-data/outputs/<probe>/<tag>/result.json`
and a combined `summary/<tag>/summary.json`.

### What is *not* wired yet

`integrated/run_grid.py`, `data.setup --smoke`, and probes 02/04/06/07/08 are
still TODO — see `PROBE_SPECS.md`. The integrated harness section below is the
target shape once the data + verifier are in place.

```bash
# planned, not yet implemented
python integrated/run_grid.py --config integrated/grid.yaml --smoke
python integrated/bridge.py              # the headline plot
python integrated/interactions.py
python integrated/dashboard.py           # one figure: per-claim verdict + effect size + CI
```

---

## What "done" looks like

A single dashboard figure rating each claim **CONFIRMED / PARTIAL / MASKED /
INTERACTION-DRIVEN / NOT-SUPPORTED in this setup**, each with a realized effect
size, a bootstrap CI, and a caveats column (sample size, model scale, what was
not controlled). Plus a `FINDINGS.md` narrative written as runs complete.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the statistical protocol
(seeds, bootstrap CIs, compute- vs param-matched comparisons, pre-registration of
predicted signs).

---

## Scope decision: from-scratch vs fine-tuned

The integrated grid trains **from scratch** by default — tiny models, so absolute
VeriBench numbers are weak, but mechanism attribution is clean. One **fine-tuned**
small-pretrained baseline is included as a reality check. They answer different
questions: from-scratch tests the *mechanisms*; fine-tuning tests the *deployment*.
See `integrated/README.md` for why this matters and how to switch.
