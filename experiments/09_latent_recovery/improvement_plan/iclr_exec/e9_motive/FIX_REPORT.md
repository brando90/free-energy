# E9 — nonce-noun pool exhaustion fix + relaunch (2026-07-24)

## Bug
Confirmatory `prepare` at registered scale (`--seed 20260724 --n-families 750
--surprise-control`) aborted with `StopIteration` at `gen_worlds_e9.py:111`
(`filler = [next(nouns) ...]`). E9 reused `gen_worlds_expd.noun_stream`, a finite
`-pus` pool (1,620 cvc + 8,100 cvcv = 9,716 after fewshot removal). E9 draws ~18
nouns/family under `--surprise-control`, so the pool supports only 9,716//18 =
**539 families**; it ran dry at ~n=540. Smoke tests (n=20, ~360 draws) never hit
it. Failure occurred inside `generate()` before `write_outputs()` — **zero rows
written**, no outcome data existed.

## Fix
Replaced the `noun_stream = genw.noun_stream` alias with a local unbounded
deterministic `noun_stream(rng)` in `gen_worlds_e9.py`:
- **Tiers 1-2 byte-for-byte identical** to the original (same cvc+cvcv prefixes,
  same nested-loop order, single `rng.shuffle`; `"pus"` appended at yield —
  `shuffle` permutes by index so this is provably equivalent). Verified: first
  9,716 yields under seed 20260724 equal the old stream element-for-element.
- **Beyond the pool**, continues the SAME PrOntoQA phonotactic family:
  alternating C/V prefixes starting with a consonant, grown one position/tier
  (cvcvc=145,800 forms, cvcvcv, ...), each tier shuffled with the same seeded
  rng, `"pus"` appended. Effectively unlimited, deterministic under the seed.
- **Global uniqueness (§1.2) preserved** structurally (distinct prefixes /
  distinct lengths ⇒ no collisions) + explicit `emitted` backstop + FEWSHOT skip.

## Fix commit
`f447bc5436ffe26f1584f57829646cb1da4295e8` (branch `paper-framing-polish`)
message: `E9: fix nonce-noun pool exhaustion at n=750 (pre-data implementation fix)`
files: `gen_worlds_e9.py`, `DEVIATIONS.md` (new). Deviation documented in
`$E9/DEVIATIONS.md` (IMPLEMENTATION_DEVIATION #1).

## Test evidence
- `e9_tests.py`: **11/11 fixture groups pass, 75 `ok` checks** (determinism,
  global-uniqueness, surprise-control cert included).
- Full CPU dry-run (throwaway dir, deleted after recording): `--n-families 750
  --surprise-control`, runtime 0.47s:
  - **750/750 worlds generated, 750/750 per-world audits passed, 0 failures**,
    global-uniqueness clean.
  - Constant across all worlds: `rule_count=18`, `word_count=97`,
    `planted_token_freq=3`.
  - All 750: plant audited false; plant complement distance == 1; zero
    plant/goal overlap both arms; surprise-control S-a inertness cert passing.
  - **CPU structural eligibility (G1-G3, outcome-independent) = 750 candidate
    pairs** (750 usable + 750 inert = 1,500 arm prompts). Final ≥300 eligible
    depends additionally on GPU gold-trace solvability (not evaluated on CPU).
  - `worlds_sha256 = fdd49f81abdc7c7f52e810663c0a25f484e2fb9f9d87c78c79afab39e370f1b9`

## run_confirmatory reset
Failed attempt archived to `run_confirmatory/failed_attempt_1/`
(queue_status.json [state=failed], waiter.out, empty worlds/) — not deleted.
`E9_HUMAN_GO` kept in place.

## Relaunch + current state
Gates at relaunch: E1-terminal=yes (completed), GPU-free=yes (GPU 0 = 0 MiB),
human-GO=yes. Launched `run_e9_queue.sh` directly via nohup pinned to the free
GPU (E9_POLL_SEC=120); gates cleared on the first iteration and it fired
immediately on **GPU 0**.

**State: RUNNING on GPU 0.** The fix carried the pipeline past the previously
failing `prepare` and through `gold` and `pair`; it is now in the **rollout**
stage: 746 pair-eligible worlds × R=8 = 5,968 prompts processing (GPU 0 ~124 GB).
`queue_status.json`: `state=running detail=rollout pid=1716503`. (746/750
worlds entered rollout; 4 dropped at GPU gold-trace pair-eligibility.)

## Progress-check one-liner
```
ssh skampere2 'D=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/e9_motive/run_confirmatory; cat $D/queue_status.json; echo; tail -1 $D/waiter.out'
```
