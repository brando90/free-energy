# E13 dry-run audit report (CPU only; fail-closed)

N target per cell: 200. Worlds: `worlds/*.jsonl` (e11-style program dicts; see README.md for schema).

## Cell set `line_k0` -- PASS

* accepted 200/200 worlds (attempts 299, accept rate 0.669, starved=False); determinism: ok
* world file: `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e13/worlds/line_k0.jsonl`
* L=[20] ops=[14] ws_tokens=[106] nearest_anc_dist=[6] measured_min_k=[0]
* reject funnel: {'build': 48, 'program_error': 23, 'fam': 23, 'line_k0_swap_fam': 5}
* per-family verbatim planted lines (world 1, `k0_43d06a_00910000`):

  * `line_k0`: `line 10: g = q; g = 149`
  * `line_k0_swap`: `line 10: g = q; g = 193`
  * `anchor_k0`: `note: q = 319`

## Cell set `grid_kr1` -- PASS

* accepted 200/200 worlds (attempts 230, accept rate 0.870, starved=False); determinism: ok
* world file: `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e13/worlds/grid_kr1.jsonl`
* L=[12] ops=[4, 5, 6, 7, 8] ws_tokens=[54, 56, 58, 60, 62] nearest_anc_dist=['within-world (paired families)'] measured_min_k=['n/a (kr1 substrate, k_c=0)']
* grid ws-token spread (diagnostic; corners are within-world paired so cross-corner matching is exact): {'mean': 58.84, 'sd': 1.85, 'min': 54, 'max': 62}
* reject funnel: {'true_values_not_distinct': 20, 'true_values_not_distinct;major5_site_value_in_listing': 3, 'major5_site_value_in_listing': 5, 'dose_off1_starved': 2}
* per-family verbatim planted lines (world 1, `kr1_507786_00920000`):

  * `adjacent_contradiction`: `line 5: r = w + g; r = 51  ||  note: r = 15`
  * `opfree_kr1`: `line 5: r = w + g; r = 51  ||  line 6: q = r; q = 15`
  * `grid_line_false_note_true`: `line 5: r = w + g; r = 15  ||  note: r = 51`
  * `grid_note_false_before`: `note: r = 15  ||  line 5: r = w + g; r = 51`
  * `grid_note_true_before_line_false`: `note: r = 51  ||  line 5: r = w + g; r = 15`
  * `grid_line_false_copy_true`: `line 5: r = w + g; r = 15  ||  line 6: q = r; q = 51`
  * `dose_off1`: `line 5: r = w + g; r = 51  ||  note: r = 50`
  * `dose_off10_20`: `line 5: r = w + g; r = 51  ||  note: r = 71`
  * `dose_large`: `line 5: r = w + g; r = 51  ||  note: r = 397`

## Cell set `k1_deference` -- PASS

* accepted 200/200 worlds (attempts 254, accept rate 0.787, starved=False); determinism: ok
* world file: `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e13/worlds/k1_deference.jsonl`
* L=[20] ops=[14] ws_tokens=[106] nearest_anc_dist=[4] measured_min_k=[1]
* reject funnel: {'build': 43, 'program_error': 11}
* per-family verbatim planted lines (world 1, `k1_43d06a_00930000`):

  * `depth_k1_bare`: `line 10: a = v + c; a = 67`
  * `k1_full_sum`: `line 10: a = v + c = 27 + 20 = 47; a = 67`
  * `k1_tentative`: `line 10: a = v + c = 27 + 20; a = 67 (unverified)`
  * `k1_probe`: `[prefix=depth_k1_bare plant] What is 27 + 20?  (true answer 47)`

## Distance-audit disclosure (line_k0, no k=0 exemption)

nearest-stated-ancestor listing-token distance is PINNED at exactly 6 for every line_k0 world and audited fail-closed (target in LINE_K0_TARGETS). k>=1 E11 cells pin 4. The 2-token difference is a hard floor: the k>=1 ancestor is a bare operand const line ('9: q = 47' = 4 ws-tokens) while the k=0 copy root must be a COMPUTED line ('9: u = a + b' = 6 ws-tokens) because MAJOR-5 forbids the site's true value appearing verbatim in the listing. Pinned + disclosed, not exempted.

## E13a-grid corner accounting

Existing corners already covered (full 13-model data, same kr1 construction, carried on every new world for paired reruns):
  * `adjacent_contradiction`: format=note, false_in=note, false_position=second
  * `opfree_kr1`: format=official, false_in=official_copy, false_position=second

New corners generated: `grid_line_false_copy_true`, `grid_line_false_note_true`, `grid_note_false_before`, `grid_note_true_before_line_false`

Dropped corners (fail-closed disclosure):
  * official_copy_before_definition (x2 truth values): IMPOSSIBLE: an official copy line 't = u' placed before u's definition reads an undefined variable -- not a valid program in the SSA grammar (interp rejects it). The official-format face of the cube is therefore only half-realizable; dropped and disclosed.

Naturalness check (structural; CPU dry-run cannot test model parse sanity -- CONDITIONAL corners must pass a pilot/gold parse gate before scaling, per spec):
  * `grid_note_false_before`: note about u appears one trace line BEFORE u's trace-time definition. Minimal (adjacent) displacement only; the full program listing (always in the prompt) already defines u, so the note is odd but not incoherent. CONDITIONAL: runner must gate on prefix parse sanity at the pilot/gold stage before scaling.
  * `grid_note_true_before_line_false`: same note-before-definition placement as grid_note_false_before; same CONDITIONAL gate.

## E13b-inc (no worlds)

A4 instruction string pinned in e13_gen.E13B_INC_ARM_A4: `Before continuing, check this trace for inconsistencies.` -- apply with e12_api mechanics (prepended to user message, golds regenerated per arm, prefill unchanged).

## E13c (no new worlds) -- regime2 world fingerprint

`{"path": "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/results/EXPH2_FRONTIER_FOLLOWUPS/regime2/programs.jsonl", "status": "ok", "sha256": "8505ee8564ff6b3b896e56f6aba7470880a8afedb74ef531aafefcb2b5cc2c7a", "n_programs": 250, "shapes": {"kr1": 70, "kr8": 60, "kc1": 60, "kc5": 60}}`

## Overall: ALL CELLS PASS
