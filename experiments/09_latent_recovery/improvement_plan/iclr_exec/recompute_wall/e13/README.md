# E13 matched-construction controls — generator + worlds

Spec: `../E13_MATCHED_CONTROLS.md` (pre-run, binding). Everything in this
directory is ADDITIVE: no existing file is modified; `gen_depth`,
`gen_programs`, `inject`, `audit_depth`, `trace_format` are reused by import.

Files:

* `e13_gen.py` — builders + fail-closed audits for the three new world sets;
  pinned E13b-inc A4 instruction string (`E13B_INC_ARM_A4`).
* `e13_dryrun.py` — CPU dry-run CLI: `python e13_dryrun.py [N]` (default 200).
  Writes `worlds/*.jsonl`, `dryrun/DRYRUN_REPORT.md`,
  `dryrun/dryrun_summary.json`.
* `worlds/line_k0.jsonl` — E13a worlds (families `anchor_k0` kept +
  `line_k0`, `line_k0_swap`). Base seed 910000.
* `worlds/grid_kr1.jsonl` — E13a-grid worlds (existing corners
  `adjacent_contradiction`/`opfree_kr1` + 4 new corners + 3 dose families,
  all paired on the same worlds). Base seed 920000.
* `worlds/k1_deference.jsonl` — E13b worlds (standard
  `depth_k1_{bare,full,partial}` + `k1_full_sum`, `k1_tentative`,
  `k1_probe`). Base seed 930000.

E13c uses NO new worlds (existing regime2 `programs.jsonl`; fingerprint in the
dry-run report). E13b-inc uses NO new worlds (A4 string + e12_api mechanics).

## World-row schema

Each line is an e11-style program dict — the exact fields of
`../e11_run/e11_haiku_programs_shared.jsonl` — so the e11/e12 harness code
(`tf.make_user_message`, `inject.build_injection`, `validate.classify_run`,
`e11_run.build_injection_e11`) consumes it unchanged:

```
program_id k seed knob_hash stmt_texts L site_line site_var site_kind
root_var root_line k_r k_c r1 r1_var r2 r2_var position position_frac
true_values_by_line out_true site_true_value total_ops listing_ws_tokens
families audit_metrics
```

plus `e13_cell` (`line_k0` | `grid_kr1` | `k1_deference`). Grid worlds are
EXPG kr1 dicts (they additionally carry `shape`, `audit`; they lack
`operand_names/operands/operand_lines`) padded with `k=0`, `total_ops`,
`listing_ws_tokens` for consumer compatibility.

Family-spec fields (superset of e11): `planted_var planted_value true_value
force_after_line delta_policy transform opacity out_cf` plus, where relevant,
`site_body site_body_true` (value transforms; the runner splices
`"line %d: %s" % (site_line, site_body)` over the model's own gold site line —
e11 semantics), `note_var note_value splice_line grid naturalness_flag dose
prefix_family probe_question probe_answer e13_arm`.

## New transforms (runner contract, Build 2/2)

Let j = `site_line` (grid: j=6, def line of u at j-1=5, copy line at j).
"Value-spliced" always means `inject.replace_final_value` on the model's OWN
gold trace line (replace the final integer, never insert), fail-closed on
var/value mismatch exactly like `build_injection`.

* `value` — existing e11 semantics (splice `site_body` over gold line j).
  Used by `line_k0`, `line_k0_swap`, `k1_full_sum`, `k1_tentative` (and the
  standard opacity arms).
* `note` — existing EXPG semantics (prefix = gold 1..j-1 + note line; used by
  `adjacent_contradiction` and the three `dose_*` families).
* `line_false_note_true_after` — prefix = gold 1..j-1 with line j-1
  value-spliced to `planted_value`, then `note: <u> = <true_value>`;
  continuation from line j.
* `note_false_before_line` — prefix = gold 1..j-2, `note: <u> =
  <planted_value>`, then gold line j-1 UNCHANGED; continuation from line j.
* `note_true_before_line_false` — prefix = gold 1..j-2, `note: <u> =
  <true_value>`, then gold line j-1 value-spliced to `planted_value`;
  continuation from line j.
* `line_false_copy_true` — prefix = gold 1..j with line j-1 value-spliced to
  `planted_value` and copy line j kept UNCHANGED; continuation from line j+1.
* `probe` — build the perturbed prefix of `prefix_family`
  (= `depth_k1_bare`), then REPLACE the trace-continuation request with the
  direct question `probe_question` ("What is 32 + 58?"); score the reply
  against `probe_answer`. Interpretation note: bare (not full_sum) prefix so
  the probe measures whether the arithmetic is deployable in the HEADLINE k1
  context; a full_sum prefix would print the answer and reduce the probe to
  reading. Change `prefix_family` at the runner if the Review agent overrules.

All new grid families share one counterfactual world (`u := planted_value`
forced after line j-1) and one planted value with `adjacent_contradiction`,
so `validate.classify_run` scores them with the standard 2-world machinery
(`inj["cf"]=True`, `force_after_line=j-1`; for `line_false_copy_true` use
`expected_lines_from=j+1`, else `j`).

## Audit stack (fail-closed; see DRYRUN_REPORT.md for results)

* line_k0 / k1 worlds: full `audit_depth` stack (min-verification-path incl.
  k=0, plant validity/collision, cross-cell L/ops/ws-token/nearest-ancestor
  targets, filler independence, opacity byte-identity) + e13 extras
  (canonical-body form, value-splice byte mechanics, delta-policy checks,
  full_sum/tentative body relations, probe integrity, shared-plant checks).
* NO k=0 distance exemption: `line_k0` pins nearest-ancestor distance == 6
  exactly (k>=1 pins 4; the 2-token gap is the MAJOR-5 floor — computed root
  line vs bare const operand line — disclosed, not exempted).
* grid worlds: EXPG build+family audits at build, then a fail-closed
  re-measure of EVERY planted family (collision, MOD-10 washout,
  discriminating reads), dose-magnitude windows (1 / 10–20 / 100–400),
  shared-plant pairing, def-splice mechanics, geometry pin (L=12, j=6,
  root=5). Cross-corner matching is within-world (paired) by construction.
* Determinism: every builder is re-run on probe seeds; accepted worlds must
  rebuild byte-identically, rejected seeds must re-reject identically.

## Corner accounting (E13a-grid)

Existing (full 13-model data, same construction): `adjacent_contradiction`
(note, false-in-note, false second), `opfree_kr1` (official copy,
false-in-copy, false second). New: `grid_line_false_note_true`,
`grid_note_false_before`, `grid_note_true_before_line_false`,
`grid_line_false_copy_true`. Impossible (dropped, disclosed): the two
copy-line-before-definition corners (undefined-variable read; invalid SSA
program). `grid_note_false_before` / `grid_note_true_before_line_false` are
CONDITIONAL on a pilot/gold parse-sanity gate (note precedes the variable's
trace-time definition; adjacent displacement only).

## Scale-up note for the runner

`worlds/*.jsonl` hold the N=200 dry-run-audited pools. The builders are
deterministic in seed; to extend a pool, continue seeds upward from the base
seed with the same builder+audit (e.g. `eg.generate(builder, audit, n,
base_seed)` regenerates a superset whose first 200 worlds are byte-identical
to the shipped files). Recommended pool ~350/cell before the gold
double-filter for 150-world cells (e11 practice). NO API/GPU spend before the
Review agent issues GO.
