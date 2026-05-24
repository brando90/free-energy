# Isolated probes

Each probe measures **one mechanism's latent strength** at its correct layer, in
isolation, with a positive control. These are the cleanest signals in the suite:
the architecture-only probes (1, 3, 5) need no training and their positive controls
are essentially proofs.

Full per-probe specs (prediction / null / measurement / control) are in
[`PROBE_SPECS.md`](PROBE_SPECS.md).

---

## Run order (fastest-to-ground-truth first)

1. **Architecture-only, no training** — `probe_01`, `probe_03`, `probe_05`.
   Validate their positive controls. If a control fails, the probe is broken; stop
   and fix before trusting anything else.
2. **Objective** — `probe_02` (controlled training run).
3. **Trained behavior + verifier** — `probe_06`, `probe_07`, `probe_08`.
4. **External** — `probe_data_wall` (scaling-curve extrapolation).
5. **Ablation** — `probe_04` (swap the attention nonlinearity, match the rest).

Each probe is `probes/probe_NN_name.py`, emits **one figure** + a **stats JSON**,
and writes a line to `FINDINGS.md`.

---

## File map

| File | Claim | Layer | Needs training? |
|---|---|---|---|
| `probe_01_softmax_bottleneck.py` | 1 | architecture + data | no |
| `probe_02_mode_covering.py` | 2 | objective | yes (small) |
| `probe_03_rank_collapse.py` | 3 | architecture | no |
| `probe_04_partition_removable.py` | 4 | architecture (ablation) | yes (small) |
| `probe_05_fixed_compute.py` | 5 | architecture + complexity | no |
| `probe_06_error_compounding.py` | 6 | trained behavior + verifier | yes |
| `probe_07_reversal_curse.py` | 7 | trained behavior | yes |
| `probe_08_lipschitz_margin.py` | 8 | trained behavior | yes |
| `probe_data_wall.py` | data wall | external | no (uses runs) |
| `run_all.py` | — | — | orchestrates all, `--smoke` flag |

---

## Output contract (every probe)

Every probe writes:

- `probes/out/probe_NN/figure.png` — one figure, layer in the title, CIs shown.
- `probes/out/probe_NN/stats.json` — machine-readable:
  ```json
  {
    "probe": "01_softmax_bottleneck",
    "layer": "architecture+data",
    "control_passed": true,
    "prediction": "fit error >= truncated-SVD tail when d < r",
    "effect_size": 0.37,
    "ci": [0.31, 0.42],
    "seeds": [0, 1, 2],
    "verdict": "CONFIRMED",
    "caveats": "real-data rank estimated at 99% energy; model = <name>"
  }
  ```

`run_all.py --smoke` runs every probe on the CPU smoke config and asserts every
`control_passed == true`, failing loudly otherwise.

---

## The two probes most worth watching

- **probe_01 (softmax bottleneck)** — the strongest claim (exact rank inequality).
  Expect a crisp CONFIRMED on the toy; the real question is whether the *measured*
  VeriBench data rank actually exceeds the head's `d`.
- **probe_06 (error compounding)** — the claim most likely to be **falsified** in
  our domain. With a Lean verifier, errors are recoverable by construction, so the
  recoverable-Markov model should beat geometric `(1−e)ⁿ`. If it does, that is a
  genuine negative result against LeCun's argument *in the verifier setting* — see
  `PROBE_SPECS.md` §6.
