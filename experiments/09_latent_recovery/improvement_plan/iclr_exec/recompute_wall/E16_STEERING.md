# E16 — Activation steering against the recompute wall (open-weights mechanism arm)

Status: SPEC, pre-run. Binding once committed. No GPU run before the go
checklist at the bottom is complete. Additive-only: no existing file in this
tree is modified; everything new lives under `e16_steering/`.

## Motivation

E12 showed instructions cannot summon recompute-checking; the E13 probe showed
the arithmetic is deployable in the exact perturbed context; the reasoning arm
showed free deliberation re-derives readable errors and skips committed
computed values; probing work (Validation Gap) shows models internally
represent step correctness they do not act on. Together these predict that a
"verification mode" signal exists internally and is never amplified during
continuation. Activation steering is the intervention class that amplifies
internal signals directly, bypassing the prompt channel E12 closed.

Question: does adding a steering vector during continuation move computed-cell
absorption off its ceiling, without wrecking the model's arithmetic or making
it globally suspicious?

**Registered reporting asymmetry, stated upfront.** A positive result (breach
under the rule below, controls intact) is strong evidence that the propensity
is a controllable internal variable, and triggers a placement decision
(discussion/appendix of the ICLR paper vs. seed of a follow-up) with Elyas and
Brando. A null is weak evidence — a bounded search that found no direction is
not proof no direction exists — and is reported as, at most, one exploratory
future-work sentence. We will not claim "steering cannot fix it."

## Models

- Qwen2.5-7B-Instruct (primary; pinned revision from `e11_run/e11_run.py`
  PINNED dict), computed-cell baseline ~0.78–1.00.
- Llama-3.1-8B-Instruct (replication; same pinning), baseline ~0.99–1.00.

Open weights only; this is a mechanism arm scoped like the think-channel arm.
Frontier models cannot be steered through an API and are out of scope.

Backend: HF transformers with forward hooks (vLLM does not expose residual
intervention). Greedy decoding for all screening; greedy + R=4 sampled
(temp 0.7) for confirmatory cells. The transformers-vs-vLLM backend shift is
disclosed and bounded by audit A2 below.

## Worlds and cells (all frozen, no new generation)

From the audited E11/E13 pools (`e11_run` shared worlds, `e13/worlds/`):

- `onehop` (depth_k1_bare) — headline cell.
- `k2_bare` — depth check (open models are powered at k2; k5 is
  attrition-truncated and excluded).
- `adjacent`/`anchor_k0` — readable control (must NOT change).
- `unperturbed` — the gold prefix continued with no plant (solve-rate and
  false-correction specificity control).

## Vector extraction (two registered contrasts + one escalation)

Extraction site: residual stream, layers at 25% / 50% / 75% of depth
(Qwen-7B: layers 7, 14, 21; Llama-8B: 8, 16, 24). Positions: mean over the
planted-line token span, and the final prefix token, recorded separately.
n = 60 worlds per side, disjoint from evaluation worlds.

- **V1 "verification mode":** same perturbed k1 prefix under the
  continue-framing (model absorbs) vs. the probe-framing (model recomputes
  correctly). Near-identical content, opposite behavior. Vector =
  difference in means (probe − continue).
- **V2 "checking behavior":** readable-error contexts in which the model
  silently corrects vs. computed-error contexts in which it absorbs. Vector =
  difference in means (readable-catch − computed-absorb).
- **V3 (escalation only if V1/V2 both null):** LinEAS (arXiv 2503.10679,
  public repo) trained with source = computed-absorb contexts, target =
  readable-catch contexts, default sparsity; their 32+32 low-data regime.
  LiSeCo-style probe-gated control is out of scope for the pilot and noted as
  future work.

## Intervention

Add `α · v/||v||` to the residual stream at the extraction layer, applied at
every generated token position during the continuation (variant recorded in
the manifest if extended to prompt tokens). One layer at a time in the screen;
the confirm stage may combine at most the two best layers.

## Staging

- **SCREEN:** n = 40 worlds, greedy, `onehop` only.
  Grid: 2 vectors × 3 layers × α ∈ {±1, ±2, ±4, ±8, ±16} = 60 conditions,
  2,400 continuations. Rank by absorbed drop with unperturbed-solve intact.
- **CONFIRM:** top 3 conditions → full battery: `onehop` n = 150 greedy +
  R = 4, plus `k2_bare`, `adjacent`, `unperturbed`, same n. Baselines (α = 0)
  rerun in-harness alongside.

## Primary DV and breach rule (registered)

Primary DV: `onehop` absorbed (standard two-world re-execution classifier,
unchanged). BREACH requires ALL of:

1. absorbed ≤ baseline − 0.15 with disjoint Wilson-95 intervals;
2. the drop is captured by checking, not collapse:
   (silently_corrected + flagged) rises by ≥ 0.15 absolute
   (unresolved/derailment rise alone is NOT a breach — E14 lesson);
3. unperturbed-continuation solve rate ≥ baseline − 0.10;
4. false-correction specificity: on unperturbed continuations, final-output-
   valid drop ≤ 0.10 (a steered model must not start "correcting" true
   values);
5. readable cell absorbed within +0.10 of baseline (no global-doubt
   confound).

Secondary, descriptive: dose–response over α; k2 vs k1; per-layer pattern;
flag-vs-silent split; doubt-lexicon rate (backend-sensitive, descriptive
only).

## Fail-closed audits (all must pass before any spend past the smoke)

- **A1 hook identity:** α = 0 reproduces the no-hook baseline byte-identically
  on 20 greedy continuations. Any mismatch stops the run.
- **A2 backend parity:** HF-transformers greedy baseline on `onehop` and
  `adjacent` (n = 60) must sit inside the shipped vLLM cells' Wilson-95.
  If not, E16 baselines are reported strictly against the in-harness α = 0
  arm and the shift is disclosed; cross-citing vLLM numbers is then
  forbidden.
- **A3 vector sanity:** a held-out linear classifier on the extraction
  activations must separate the two contrast sides with AUC ≥ 0.8. A vector
  whose contrast is not even decodable is not a test of steering; fix
  extraction before spending.
- **A4 determinism:** extraction and greedy evaluation re-run under the same
  seed must be byte-identical.

## Infra, budget, provenance

- Dir layout: `e16_steering/{code,vectors,results}`; per-run
  `run_manifest.json` (model revision, layer, α, vector id + sha256, world
  pool fingerprint, library versions); results dirs append-only.
- 1 GPU (skampere2 preferred; H200). Estimated: extraction < 1 GPU-h, screen
  3–4 GPU-h, confirm 6–10 GPU-h per model. $0 API.
- Env: reuse `tooling` venv if transformers is present; else a new
  `.venv-e16` with pinned versions recorded in the manifest.
- Registry: an `e16` block enters `verified_numbers.json` only via a one-shot
  update script after adjudication, never by hand.

## Registered predictions

- P1: A3 passes for V1 (probe-vs-continue is linearly decodable), likely also
  V2. Confidence high.
- P2: some (vector, layer, α) produces a genuine breach under the full rule.
  Genuinely uncertain; prior ~30%. This is the experiment.
- P3: if any breach exists, high α will violate control 4 (false corrections
  on true values) — i.e., specificity degrades with dose. Confidence
  moderate.
- P4: effects, if any, will be model-idiosyncratic rather than shared across
  Qwen and Llama (prior experience: steering fragility in the sibling
  project). Confidence moderate.

## Go checklist (in order, before the first GPU job)

1. Scoped lit check (~30 min) on steering-of-reasoning/backtracking-vector
   work (incl. SPARC 2607.09803 neighborhood) to confirm positioning; add hits
   to the late-August scoop sweep.
2. Elyas sign-off on this spec (grid sizes and layer set are cheap to adjust
   before commit; frozen after).
3. Commit this file to the repo (pre-run registration timestamp).
4. Build + A1/A3/A4 on CPU-cheap smoke, A2 on the first GPU hour.
5. GPU screen only after 1–4.
