# E9 — Implementation Deviations

Companion to `E9_REGISTRATION.md` and `E9_REGISTRATION_AMENDMENT.md`. Records
non-substantive implementation fixes made during the confirmatory launch. Each
entry states the bug, the fix, why it does not touch any registered design
decision or any outcome data, and the test evidence.

---

## 2026-07-24 — IMPLEMENTATION_DEVIATION #1: nonce-noun pool exhaustion at n=750

**Where:** `gen_worlds_e9.py` (nonce-noun stream).

**Bug.** The first confirmatory `prepare` at the registered scale
(`--seed 20260724 --n-families 750 --surprise-control`) aborted with
`StopIteration` at `gen_worlds_e9.py:111`
(`filler = [next(nouns) for _ in range(FILLER_PAD + 1)]`). Root cause:
`gen_worlds_e9.py` reused `gen_worlds_expd.noun_stream`, a **finite** without-
replacement pool of `-pus` pseudowords — 1,620 `cvc` + 8,100 `cvcv` prefixes =
9,716 tokens after the 4 in-pool fewshot nouns are removed. E9 draws ~18 nouns
per family under `--surprise-control`
(trunk 5 + a_mid 2 + b_mid 2 + z 1 + bridge 1 + spur 1 + filler 4 + bridge_b 1 +
b_dead 1). The pool therefore supports only `9,716 // 18 = 539` families;
generation ran dry mid-family at ~n=540 and raised `StopIteration`. The smoke
tests never exposed this because they used only n=20 families (~360 draws). The
failure happened **before any world was written** — the exception propagated out
of `generate()` before `write_outputs()`, so zero rows were produced.

**Fix.** Replaced the module-level alias `noun_stream = genw.noun_stream` with a
local, unbounded, deterministic `noun_stream(rng)` in `gen_worlds_e9.py`:

* **Tiers 1–2 are byte-for-byte identical** to `gen_worlds_expd.noun_stream`:
  the same `cvc` + `cvcv` prefixes are concatenated in the same nested-loop
  order and passed through a single `rng.shuffle`. `random.shuffle` permutes by
  index (independent of element strings) and the `"pus"` suffix is appended at
  yield time, so for any draw count the original pool could satisfy the yielded
  sequence is unchanged (verified: first 9,716 yields under seed 20260724 are
  equal to the old stream, element-for-element).
* **Beyond the pool the SAME PrOntoQA phonotactic family is continued**:
  alternating consonant/vowel prefixes starting with a consonant (positions
  0,2,4,…=`CONSONANTS`, 1,3,5,…=`VOWELS`), grown one position at a time —
  `cvcvc` (length 5, 145,800 forms), `cvcvcv`, … — each tier shuffled with the
  same run-seeded `rng`, then `"pus"` appended. Effectively unlimited (>145k in
  tier 3 alone, far past any planned n) and deterministic under the run seed.

**Global uniqueness (registration §1.2) preserved.** Uniqueness is structural:
distinct prefixes give distinct tokens within a tier, and prefixes of different
length give tokens of different length across tiers, so no two yields collide.
An explicit `emitted` guard is kept as a fail-closed backstop and to skip
`FEWSHOT_NOUNS`. The `audit_global_uniqueness` check (regex `\b[a-z]+pus\b`)
still matches every tier and reported zero cross-world collisions at n=750.

**Why this is non-substantive (not a design change).**
1. It is a pure **implementation detail** of nonce-token supply. No registered
   structural size changes (`TRUNK_LEN=4`, `BRANCH_LEN=3`, `PRIMARY_D=1`,
   `FILLER_PAD=3` unchanged), no change to family geometry, audits, arms,
   surprise-control construction, eligibility criteria, or the DV.
2. The tokens are opaque nonce categories with no semantics; lengthening a
   prefix by one CV position keeps the same `-pus` phonotactic class the
   registration and smoke tests already used. Per-family invariants
   (constant rule-count, word-count, one-token structure) are unaffected.
3. **Zero outcome data existed** — and none existed anywhere — when the bug
   fired: it aborted during CPU world *construction*, before any world row was
   written and long before any GPU rollout. There is no result to bias.
4. Small-n behavior is provably unchanged (byte-identical tiers 1–2), so all
   prior smoke artifacts remain reproducible.

**Test evidence.**
* `e9_tests.py`: all **11 fixture groups pass** (75 individual `ok` checks),
  including determinism, global-nonce-uniqueness, and the surprise-control
  certificate — unchanged from before the fix.
* Equivalence check under seed 20260724: first 9,716 yields of the new stream
  equal the old finite pool element-for-element; 4,000 further draws are all
  valid `-pus` tokens, all unique, all length-8 `cvcvc+pus`.
* Full CPU `prepare` dry-run at the registered scale
  (`--seed 20260724 --n-families 750 --surprise-control`) into a throwaway dir
  (deleted after recording): **750/750 worlds generated, 750/750 per-world
  audits passed, 0 audit failures**, global-uniqueness clean. Constant across
  all worlds: `rule_count=18`, `word_count=97`, `planted_token_freq=3`. All 750:
  plant audited false, plant complement distance == 1, zero plant/goal overlap
  on both arms, surprise-control S-a inertness certificate passing. CPU
  structural eligibility (G1–G3, outcome-independent) = **750 candidate pairs**
  (750 usable-arm + 750 inert-arm = 1,500 arm prompts); the registered ~0.4–0.7
  yield to ≥300 *final* eligible pairs additionally depends on GPU-time
  gold-trace solvability, which this CPU generator does not evaluate.
  `worlds_sha256 = fdd49f81abdc7c7f52e810663c0a25f484e2fb9f9d87c78c79afab39e370f1b9`.
