# Data & verifier — VeriBench / VeriBench-FTP + Lean 4

"Real sequences" in this suite means tokenized Lean 4 program/proof text, and the
verifier is the **Lean compiler itself** — a hard pass/fail oracle. That hard oracle
is exactly what makes the error-compounding (#6) and fixed-compute (#5) probes
meaningful: you get ground-truth validity, not a soft proxy metric.

---

## The benchmarks

- **VeriBench** — a Lean-4 benchmark requiring generation of complete Lean 4
  programs: implementations, unit tests, correctness theorems, and formal proofs,
  derived from real (human-curated) Python code, including security-critical
  programs. Targets the full verification pipeline, not just proof success.
- **VeriBench-FTP** — a formal-theorem-proving sibling: 857 theorems from 140
  problems across five difficulty levels (HumanEval puzzles, foundational exercises,
  classical algorithms, real-world security vulnerabilities, stdlib programs).
  Reported baseline: Goedel-Prover V2-8B ≈ 39.6% Pass@32 — i.e. hard, unsaturated.

Use **VeriBench** when the task is end-to-end program+spec+proof generation; use
**VeriBench-FTP** when you want isolated theorem-proving at controlled difficulty
(useful for the proof-depth axis in probe 05 and the bidirectional-lemma scan in
probe 07).

> Confirm the exact dataset access (HF dataset id / repo) and license at setup time;
> the loader is intentionally pluggable so the suite doesn't hard-code a path that
> may change.

---

## What each probe needs from the data

| Probe | Needs from VeriBench |
|---|---|
| 01 bottleneck | next-token log-prob matrices (contexts × vocab) from a small model |
| 02 mode-covering | per-token labels of verifier-accepted vs rejected next tokens |
| 05 fixed compute | proofs labeled by **proof depth** (tactic-step count) |
| 06 error compounding | **per-step validity** via the Lean verifier along generated proofs |
| 07 reversal curse | lemmas usable in **both directions** (for asymmetry measurement) |
| 08 brittleness | sequences amenable to **minimal token edits** + re-verification |
| integrated | pass@1 / pass@k via the verifier as the downstream metric |

---

## The verifier interface

A thin wrapper exposing a single contract the probes rely on:

```python
class LeanVerifier:
    def check(self, lean_src: str, timeout_s: float = 30.0) -> VerifyResult: ...

@dataclass
class VerifyResult:
    compiles: bool          # did `lake build` / the checker accept it
    error_step: int | None  # first failing tactic/line index, for per-step validity (#6)
    messages: list[str]     # compiler diagnostics
    elapsed_s: float
```

Implementation options (pick per environment, keep behind the same interface):
- **`lean-dojo`** for programmatic interaction with a built Lean repo (good for
  per-step / tactic-level state, which probe 06 needs).
- **direct `lake build`** on a generated file for whole-artifact pass/fail (simplest
  for VeriBench end-to-end pass@k).

Per-step validity for probe 06 requires tactic-level checking: verify the proof
prefix after each tactic so you can record where (if anywhere) it leaves the valid
state and whether a later tactic **recovers** it — the recovery events are the crux
of the (1−e)ⁿ-vs-recoverable-Markov test.

---

## Train / validation / test split (run this first)

```bash
cd experiments/00_ar_pros_cons

# Gold Lean files only. Writes train/val/test plus a 20-row smoke manifest.
python -m data.setup --smoke

# Include generated agent candidates as additional rows, split by the same task id.
python -m data.setup --include-generated-agents --smoke
```

`data.setup` scans `~/veribench/veribench_dataset` by default and writes:

```text
data/splits/veribench_manifest.jsonl
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test.jsonl
data/splits/smoke.jsonl
data/splits/summary.json
```

The split is deterministic and task-level: all generated variants of the same
task stay in the same split. That avoids a subtle but fatal leakage mode where
`agent0` for a task lands in train and `agent1` for the same task lands in test.

The current manifest rows include:

- `split`, `task_id`, `variant_id`, `source_kind`, and `family`;
- absolute local `lean_path` and optional `py_path`;
- line/character counts;
- theorem count, `sorry`/`admit` count, and a tactic-count proxy;
- SHA-256 of the Lean source for cache keys.

## Smoke subset requirements

The smoke subset should:
- contain a spread of proof depths (so probe 05's depth axis is non-trivial),
- include at least a few known-recoverable proofs (so probe 06 can detect recovery),
- run the *entire* probe suite + integrated smoke grid end-to-end on CPU in minutes.

If the Lean toolchain isn't callable, the verifier-dependent probes (02 real-proxy,
05 real, 06, 08 real) must **skip with a clear message**, not silently pass.

---

## Caching & determinism

- Cache verifier results keyed by `sha256(lean_src)` — verification is the slowest
  step; never re-verify identical source.
- Record the Lean toolchain version and benchmark commit/version in every output
  JSON so results are reproducible and comparable across runs.
- Generation for probes 06/08 is seeded; log the seed and decoding params
  (temperature, top-p) with each result.

---

## Honesty notes specific to the verifier

- **Pass@k is not pass@1.** Report both; pass@k partly *defeats* the
  error-compounding argument by construction (resampling is recovery), which is the
  whole point of probe 06 — so keep them clearly separated in plots.
- **Timeouts are not failures of the model.** Log timeouts as a third outcome
  (`compiles=False, reason=timeout`) and report their rate; a high timeout rate
  silently biases every downstream metric.
- **Verifier ≠ correctness of the spec.** VeriBench checks that the proof matches
  the stated theorem; a vacuous or mis-stated theorem can "pass." For the security
  set especially, note this caveat — passing the verifier is necessary, not
  sufficient, for the property you care about.
