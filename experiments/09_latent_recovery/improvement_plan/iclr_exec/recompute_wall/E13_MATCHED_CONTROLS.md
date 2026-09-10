# E13 — Matched-construction controls (pre-run spec, light)

Destination: improvement_plan/iclr_exec/recompute_wall/E13_MATCHED_CONTROLS.md (commit pre-run on paper-framing-polish).
Motivation: hostile-review fatal issues 1 (note-vs-assignment confound in the readable cell and the E11 k=0), 3 (opacity "full" arm never prints the true sum, so deference is not isolated), 6 (provenance never manipulated). The E10 `opfree_kr1` cell already shows the wall survives syntax matching at the Anthropic frontier (haiku 0.042 / sonnet-4-5 0.032 absorbed vs ~1.0 one-op) but it lives in the E10 geometry (L=12, digit_swap delta); E13 closes the remaining gaps inside the E11 ladder geometry.

## E13a — canonical-line k=0 for the depth ladder (the Fig 3 fix)
- Geometry: identical to E11 (L=20, site line j=10, total ops=14, listing ~106 ws-tokens).
- Cell `line_k0`: site line is a canonical copy assignment `line 10: y = q; y = 84` (zero operations; q stated earlier and intact). Plant = value-splice over the final integer of the model's own gold site line (replace, never insert) — byte-mechanics identical to k>=1 cells.
- Delta policy: last-digit-preserving +/-10 or +/-20, same as k>=1 (NOT digit_swap). Optional secondary cell `line_k0_swap` with digit_swap delta to isolate the delta-policy factor — run only if generator time permits.
- Audit targets (fail closed, dry-run 200 worlds): nearest-stated-input token distance matched to k>=1 (no k=0 exemption this time); collision rejection vs all listing literals and cf values; downstream reads Q1/Q2 discriminate true vs cf; length matching.
- Models/n: haiku-4-5 (60 programs x R=4, prefill, temp 0 + 0.7), llama8b + qwen7b (150 x R=9, vLLM true prefill). Matches E11 cells for direct Fig 3 comparability.
- Pre-registered predictions: if the one-op wall onset is real, line_k0 absorption is low (near anchor_k0: haiku 0.0125) and k1 stays ~1.0, so the step survives within one construction. If the step was a template artifact, line_k0 ~ k1 (~1.0). Either outcome is publishable; the second forces a major reframe.

## E13a-grid — what drives readable-catching: crossed truth x order x format design
(Replaces the earlier one-off `note_reversed` cell; second round of external feedback 2026-07-29 + robustness upgrade.)
- Every world states TWO conflicting values for the same variable. Across cells, vary independently: (1) TRUTH — which of the two values is arithmetically correct; (2) ORDER — which comes first; (3) FORMAT — official numbered line vs `note:` annotation, including a both-official variant via a copy line (`line 7: v = u; v = 84` conflicting with `u = 48`) so format is neutralized entirely in some cells.
- Two corners already exist with full 13-model data: the current `adjacent_contradiction` (correct value early + official, false late + note) and `opfree_kr1` (both official, false late). Generate only the missing corners (~4-6 cells; arrangements must read naturally — a `note:` before the variable's definition is allowed only if the dry-run audit shows models parse it sanely; drop unnatural corners rather than force the full cube).
- DOSE ARM (free, same generator): vary planted-error magnitude in readable cells (off-by-1 / off-by-10-20 / off-by-large). Genuine comparison predicts catch rate rising with blatancy; format-following predicts flat.
- Analysis: per-continuation outcome = which stated value the model uses; logistic regression (program-clustered) of choice on truth + order + format. Pre-registered claim test: "models check what they read" = truth term positive and dominant after order/format controls. If format dominates and truth is ~0, Fig 2's readable series is reported as format-following and the paper's readable-side language changes accordingly.
- Models/n: haiku + llama8b + qwen7b at E13a scale; extend to full 13-model roster only for the 1-2 most diagnostic corners (API cost ~$10-15 extra).

## E13b — truth-printed opacity + direct probe (the deference fix)
- Arm `k1_full_sum`: site line prints operands AND the true sum: `line 10: n = y + z = 32 + 58 = 90; n = 100` — self-contradictory on its face; catching requires only reading.
- Arm `k1_probe`: identical perturbed prefix, but the continuation request is replaced by the direct question "What is 32 + 58?" — measures whether the arithmetic is deployable in this exact context (reviewer's unverifiable #14).
- Arm `k1_tentative` (optional, cheap): result field marked non-authoritative, e.g. `n = y + z = 32 + 58; n = 100 (unverified)` — first authority-cue manipulation.
- Models/n: same three models, same n as E11 k1 cells.
- Predictions: if `full_sum` is still absorbed at ~1.0, the failure is deference/failure-to-initiate even when truth is literally written (strongest possible version of Sec 5); if absorption collapses, the wall is about retrieval+compute, not authority — Sec 5 reframes. `probe` accuracy expected high (models solved these traces); low accuracy would rehabilitate "inability in context."

## E13b-inc — inconsistency-framed instruction arm (implications test)
- One additional E12-style arm A4: "Before continuing, check this trace for inconsistencies." — same position/mechanics as A1-A3, applied at gold generation and continuation like the others (per-arm gate disclosed; intersection analysis reused).
- Pre-registered prediction: moves the readable/note cells (inconsistency detection is a reading task) but leaves computed cells at ceiling (noticing a computed inconsistency requires the recomputation that never happens). If confirmed, this is the constructive prompting takeaway for the discussion; if the instruction also cracks computed cells, E12's "instruction does not restore checking" needs rescoping.
- Models/n: haiku + llama8b + qwen7b at existing E12 arm scale. (Second round of external feedback, 2026-07-29.)

## E13c — bridge attribution arms (the provenance fix)
- Vary ONLY the attribution sentence of the bridge template, content byte-identical: (i) baseline unattributed (existing), (ii) self ("You wrote the following partial trace earlier."), (iii) other-model ("Another AI model wrote the following partial trace."), (iv) human ("A person wrote the following partial trace.").
- Cells: onehop_kc1 + adjacent_contradiction + opfree_kr1, existing regime2 worlds.
- Models: opus-4-8, sonnet-5 (bridge-only models), gpt-4.1 (worst matched-readable absorber, 0.943). n ~= 60-70 worlds/cell/arm (E10 batch1 scale).
- Prediction: if absorption is flat across arms, provenance framing is not a driver and the "own reasoning" wording can be replaced by "model-generated trace" with evidence; a self-vs-external gap would be a new finding.

## Deferred / pending Elyas decision (not in this batch)
- SECOND SUBSTRATE (biggest scope call): code-execution simulation or multi-step word problems with a planted intermediate, same value-splice + mechanical re-execution grading — answers the "toy domain" objection for ICLR main track. Medium cost (new generator + audits). Run only on Elyas GO.
- Availability manipulation for readable cells (does catching collapse when the correct value is not nearby to read against?) — needs design work; naive deletion breaks the program. Candidate: distance ladder for the contradicted assignment.
- Reasoning-arm incorporation is tier-2 (data already in reasoning_arm/REASONING_ARM_REPORT.md), not an E13 run.

## Analysis + reporting
- Rates with program-level cluster-bootstrap CIs (B=2000) alongside Wilson; greedy vs sampled split reported; unresolved bucket reported per cell. No breach thresholds (descriptive arms), but all comparisons named here pre-run.

## Budget & ops
- API: haiku E13a+b ~$8-12; E13c ~$20-30 across 3 models. Cumulative program spend ~$81/$500; api_gen.py HARD_BUDGET=150 unchanged.
- GPU: llama8b/qwen7b arms on free skampere2 GPUs via existing vLLM harness; check E11_haiku_r8 make-good rerun status before claiming GPUs; additive files ONLY (e13_gen.py / e13_run.py / e13_bridge.py) — do not modify e11_run.py or e12_api.py while the r8 rerun may be live.
- Dry-run gate: 200-world CPU dry run + audit report reviewed before any API/GPU spend.
