# E17: Statement-boundary test on the Yee et al. stimuli (pre-registration draft, 2026-09-14)

Status: DRAFT for Elyas sign-off. Nothing has run. Commit this file to
`improvement_plan/iclr_exec/recompute_wall/` before the first API call; the commit hash is the
registration timestamp.

## Question

The paper explains the gap between Yee et al.'s high recovery from calculation errors (78% for
GPT-4 on GSM8K) and our near-zero recovery on committed computed values by where the cut lands:
their perturbed prefix ends right after the corrupted numeral, so the model resumes mid-statement
(measured on their release: 516 of 516 GPT-4 calculation-error prefixes end on the numeral, 83% right
after an equals sign; registry block `yee_prefix_endings`); ours ends after a completed line. That the
boundary CAUSES the recovery gap is asserted, not tested. E17 tests it directly:
holding everything else fixed, does completing the statement that contains the false value
suppress recovery?

## Hypotheses (registered)

- H1 (boundary): recovery is lower when the prefix ends after the completed statement than when it
  ends right after the corrupted numeral, on the same items, same model, same decoding.
- H0: no difference beyond a pre-set margin.

Prediction stated for the record: the authors expect H1. A null is publishable and would demote
the committed-versus-in-flight sentence in Section 8 to a hypothesis.

## Design

Substrate: the 106 audited original-number pairs from the renumbering replication
(`scratch/yee_memo/pairs_v3.json`, the ORIG arm; GSM8K, gpt-4-0314 published-denominator items
with position-calc, perturbation-random). Original numbers only; renumbered twins are not used.

Model: gpt-4o (the one model in the replication with an intact repair paradigm; gpt-4.1 restarts
from scratch on every item and cannot show a boundary effect). Same protocol as the replication:
two-message chat exactly as Yee et al.'s `build_messages`, temperature 0, max_tokens 500, single
call, no extraction call.

Two prefix boundaries per item, identical bytes up to the corrupted numeral:

- **numeral** (their boundary): prefix ends immediately after the corrupted numeral.
- **statement** (our boundary): prefix ends after the closing suffix of the statement containing
  the numeral. Suffix = the original text from the numeral to the first sentence terminator
  (`.`, `!`, `?` followed by whitespace or newline) or to the closing `>>` of a calculator
  annotation followed by the sentence terminator, whichever the original contains. Suffix is
  taken verbatim from the model's own original chain.

Fail-closed item filters (applied before any call; every rejection logged):
- the suffix must not contain another number (no new computation, no second occurrence of the
  planted or true value);
- the suffix must not contain a corrective phrase (regex over: wait, actually, mistake, error,
  should be, correct, recheck);
- the suffix must be non-empty (the numeral is not already the last token of its sentence);
- suffix length at most 40 tokens.

Controls, same two boundaries, uncorrupted prefix (the true numeral), to separate boundary effects
on recovery from boundary effects on ordinary continuation (malformed output, restart rate).

Calls: 106 items x 2 boundaries x 2 (corrupted, uncorrupted) = 424 calls, minus filtered items.
Estimated cost at the replication's rate ($1.24 for 424 calls): about $1.50. Cap $5.

## Outcomes (single pass, continuation only, same scorer as the replication)

Per continuation, one of: **recovered** (concluding answer = true target), **propagated**
(concluding answer = naive-propagation target, mechanically derived), **other** (a different
number), **no_answer**. Separately annotated, not exclusive with the above: **restart** (the
continuation re-solves from the question rather than continuing the chain) and **malformed**
(truncated at max_tokens or unparseable). Restarts are reported, never excluded.

## Pre-specified analysis

Primary: paired comparison of recovery, numeral vs statement boundary, on the corrupted arm, over
items that pass the filters. Exact McNemar test; report the 2x2 flip table in full (all four
cells), as the replication should have.

Equivalence margin: if the paired recovery difference is under 5 points with McNemar p > 0.05,
report H0 supported (no boundary effect at this margin).

Secondary, all pre-specified: (a) propagation rate by boundary; (b) restart rate by boundary,
corrupted and uncorrupted; (c) malformed rate by boundary; (d) recovery difference restricted to
continuation-style (non-restart) responses; (e) recovery difference on the uncorrupted arm, which
must be near zero for the corrupted-arm difference to be interpretable.

Decision rule stated in advance:
- Recovery falls by 15 points or more at the statement boundary, with uncorrupted performance and
  malformed rate stable (within 5 points): H1 supported; the Section 8 sentence stands and gains a
  citation to this appendix.
- Difference under 5 points: H0; rewrite the Section 8 sentence as a hypothesis and drop the
  "more fundamentally" framing.
- Difference between 5 and 15 points, or driven by malformed/restart changes: report as partial
  and say which.

## What this does not test

Whether the boundary explains the full gap to our trace substrate (different task, grader, and
models); only whether the boundary matters on their substrate with their model.

## Files

- Code: `scratch/e17_boundary/` (build_prefixes.py, run_api.py reusing `yee_memo/run_api.py`
  request logic, score.py reusing `yee_memo/score.py`), copied to
  `iclr_exec/recompute_wall/e17_boundary/` with all raw responses on completion.
- Ledger: `usage_ledger.json`, hard stop at $5.
- Registry: block `e17_boundary` added by one-shot script on completion.

## Sign-off needed

1. Approve the design and the $5 cap (API spend).
2. Confirm gpt-4o as the model, or add a second (gpt-4o-mini would roughly double the cost).
3. Approve committing this spec as the registration.
