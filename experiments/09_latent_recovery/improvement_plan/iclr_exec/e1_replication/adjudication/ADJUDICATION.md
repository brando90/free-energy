# E1 Adjudication — multi-model replication of the three boundary conditions (EXPD-R)

**Adjudicated:** 2026-07-24 · **Registration:** `iclr_exec/registrations/E1_REGISTRATION.md` (commit `bdb098a1d85b33063274481e53cb9d7130dc32c6`) · **Runs:** `iclr_exec/e1_replication/results/{qwen1p5b,llama8b,olmo7b,qwen32b}` (queue rc=0, 2026-07-24 18:50 UTC) · **Adjudicator:** locked `improvement_plan/expd/expd_confirmatory_analysis.py` machinery (stated-complement DV, cluster unit `family_id`, positions pooled, cluster-bootstrap CIs) driven by `compute_e1.py` (archived in scratchpad).

## Bottom line

**The program-level claim does NOT replicate. 0 of 4 models fully replicate (rule requires ≥3/4).** No single boundary condition reaches the ≥3/4 per-condition bar either (B1 0/4, B2 2/4, B3 0/4). The most important finding is not merely a null: on the d=0 visibility cliff (B1) **three of four models show a statistically significant REVERSAL of the Qwen2.5-7B effect** — explicit rejection is *lower* at d=0 than at d=1, the opposite of the anchor. The adjudicator reproduces the Qwen2.5-7B anchor exactly (B1 +0.216, B2 +0.799, B3 pass → fully replicates), so the pipeline is validated and the non-replication is a property of the new models, not the analysis.

---

## 1. Registered criteria (quoted verbatim before application)

From §6 (THE confirmatory family — per-model boundary slots, Holm within model, m = 3, α = 0.05):

> | **B1** | d=0 cliff (sign + magnitude) | stated-complement(`aff_false_attr_d0`) − stated-complement(`aff_false_attr_d1`), one-sided > 0, cluster bootstrap | p < Holm-α **AND** point diff ≥ **+0.10** |
> | **B2** | usability gap (sign + magnitude) | derivational inj-dep(`cat_false_usable`, pooled d{1,3,∞}) − (`cat_false_inert`, pooled), one-sided > 0 | p < Holm-α **AND** point diff ≥ **+0.30** |
> | **B3** | flat-d (conjunctive TOST) | (i) stated-complement `aff_false_attr` d1 vs d5 TOST ±0.10; (ii) derivational inj-dep `cat_false_usable` d1 vs d∞ TOST ±0.15 | both TOSTs pass at the slot's Holm-α |

> Within each subject model Mi, Holm–Bonferroni over the three slots (conjunctive slots are intersection-union; no alpha splitting inside a slot). Cluster unit = family_id (mirrored-quadruple/base-chain), cluster-bootstrap CIs; primary contrasts pooled over positions.

From §6.1 (Decision rules):

> **Model-level replication:** model Mi *fully replicates* iff B1 ∧ B2 ∧ B3 all pass under Mi's Holm family.
> **Program-level claim** … : iff **≥3 of the 4 scored models fully replicate** … ≥3/4, absolute floor 3. The absolute floor of 3 fully-replicating models binds even if a model is excluded under §8.
> **Per-condition claims** … : iff ≥3/4 of scored models pass that slot.
> **Pre-stated failure reading:** any model that FAILS B3 because it genuinely checks at d≥1 (stated-complement rising in d) is a *positive scientific finding* … it still counts as a non-replication for the claim rule.

From §8.2 (Exclusion — cell-level): 100 ≤ n < 150 scored+flagged; 50 ≤ n < 100 scored, CI-widened, flagged; **n < 50 → the affected slot is INVALID-underpowered — reported, not scored, counts as "not replicating" (conservative direction).**

Anchor (Qwen2.5-7B, §6, NOT re-tested): d=0 cliff **+0.209 [0.153, 0.264]** (stated-comp 0.247 vs 0.031); usability gap **≈ +0.80** (usable 0.863/0.768/0.787 vs inert ≤0.023); flat-d stated-comp d1..d5 = 0.031/0.050/0.021.

---

## 2. Integrity (Step 1) — all clean

| Model | Oversample (reg) | Gold attempted | Gold solve | Gold eligible (double-filter) | Validated rows / grid target | world_audit_failures | measured_d / truth / uniq run_id |
|---|---|---|---|---|---|---|---|
| qwen1p5b | 4× (4×) ✓ | 11,520 | 0.883 | 3,259 (0.283) | **6,477 / 6,600** (−123, in d0 cells) | 0 | all True |
| llama8b | 2× (2×) ✓ | 5,760 | 0.982 | 3,115 (0.541) | 6,600 / 6,600 | 0 | all True |
| olmo7b | 4× (4×) ✓ | 11,520 | **0.199** | **506 (0.044)** | **893 / 6,600 (13.5%)** | 0 | all True |
| qwen32b | 2× (2×) ✓ | 5,760 | 1.000 | 5,732 (0.995) | 6,600 / 6,600 | 0 | all True |

- **Oversampling** matches the registered §3 budget exactly (seeds 0-3 for the 4× models, 0-1 for 2×; generator reused unchanged, `worlds_json_sha256` recorded per model).
- **Zero world-audit failures**, all four models. Integrity block True on every flag: `all_measured_d_match`, `all_planted_false_cells_audited_false`, `all_anchor_cells_audited_true`, `unique_run_ids`, `duplicate_run_id_count=0`, single registered `prompt_sha256 = f668299…`.
- **Gold-attrition** (the two 4× models the registration flagged):
  - **qwen1p5b:** solve 0.883, double-filter yield 0.283 → 3,259 eligible. Grid filled to 6,477/6,600; the 123-row shortfall is concentrated in the d0 attribute cells (`aff_false_attr_d0` n=441, `neg_false_attr_d0` n=336) where joint-solve is intrinsically lowest (mirrored-pair joint-solve d0 = 0.126). Per-position n stays ≥100 for the scored B1 cells → scored, flagged, not invalid.
  - **olmo7b:** solve **0.199**, double-filter yield **0.044** → only **506 eligible**. The 4× oversample was insufficient; the grid reached only 893/6,600 rows and most cells fall below n=50. See §4 (olmo is underpowered/exclusion-candidate).
- **Cell spot-check (3 cells/model vs manifest):** all 12 sampled cells join to `manifest.jsonl` with zero misses; `designed_d` and `world_kind` match row-for-row; `prompt_sha256` = registered hash. (qwen1p5b: true_interruption, aff_false_attr_d3, cat_false_usable_dinf · llama8b: aff_false_attr_d5, aff_true_attr_d1, neg_true_attr_d1 · olmo7b: cat_false_usable_dinf, cat_false_inert_dinf, neg_false_attr_d5 · qwen32b: benign_paraphrase, neg_false_attr_d0, aff_true_attr_d1.)
- **Manipulation checks:** MC-R2 anchor sanity passes on all four (benign `closure_valid` ≥ 0.80 and stated-complement = 0.000 everywhere — even olmo benign valid = 0.878). MC-R3 true-cell false-positive floor = **0.000** on `aff_true_attr_d1` and `neg_true_attr_d1` for every model, so no floor subtraction is needed and the B1 results below are not floor artifacts.

---

## 3. Verdict table (Step 2/3) — per model × per boundary condition

Point diffs are pooled-rate differences (registered "point diff"); CIs are family-cluster bootstrap 95%; p-values are the registered paired-family one-sided (B1/B2) or intersection-union max-TOST (B3), scored under per-model Holm (m=3, α=0.05). **stated-comp = strict stated-complement (primary rejection DV); deriv = derivational injection-dependence (primary reuse DV).**

### B1 — d=0 visibility cliff  (pass iff Holm-sig one-sided>0 AND diff ≥ +0.10)
| Model | stated-comp d0 | stated-comp d1 | point diff (d0−d1) | boot 95% CI | one-sided p | Holm thr | verdict |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B (anchor)** | 0.247 | 0.031 | **+0.216** | [+0.164, +0.269] | 0.0 | 0.0167 | **PASS** |
| qwen1p5b | 0.227 | 0.369 | **−0.142** | [−0.215, −0.072] | 0.99998 | 0.05 | **FAIL — reversed** |
| llama8b | 0.051 | 0.196 | **−0.144** | [−0.189, −0.100] | 1.0 | 0.05 | **FAIL — reversed** |
| olmo7b | 0.470 | 0.278 | +0.192 | [0.000, +0.369] | 0.113 | 0.025 | **FAIL — INVALID (n<50) / n.s.** |
| qwen32b | 0.013 | 0.082 | **−0.069** | [−0.104, −0.036] | 0.99996 | 0.05 | **FAIL — reversed** |

The reversal on qwen1p5b/llama8b/qwen32b is **consistent across all three injection positions** (e.g. llama early d0=0.04 vs d1=0.327; qwen1p5b mid d0=0.184 vs d1=0.360; qwen32b every position d1>d0). olmo shows the anchor sign but on n=66/72 cells with only 3 paired families → INVALID and non-significant.

### B2 — usability gap  (pass iff Holm-sig one-sided>0 AND diff ≥ +0.30)
| Model | deriv usable (pooled) | deriv inert (pooled) | point diff | boot 95% CI | one-sided p | Holm thr | verdict |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B (anchor)** | 0.807 | 0.008 | **+0.799** | [+0.762, +0.834] | 0.0 | 0.025 | **PASS** |
| qwen1p5b | 0.711 | 0.011 | **+0.700** | [+0.659, +0.738] | 0.0 | 0.0167 | **PASS** |
| llama8b | 0.938 | 0.014 | **+0.923** | [+0.896, +0.948] | 0.0 | 0.0167 | **PASS** |
| olmo7b | 0.442 | 0.000 | +0.442 | [+0.305, +0.588] | 0.0003 | 0.0167 | **FAIL — INVALID (n<50)** (descriptively large) |
| qwen32b | 0.013 | 0.000 | **+0.013** | [+0.006, +0.022] | 0.0012 | 0.0167 | **FAIL — magnitude (≪ +0.30)** |

B2 is the only condition with any cross-model support (qwen1p5b, llama pass; olmo directionally large but underpowered). **qwen32b is a genuine null**: near-zero derivational absorption (0.013) despite being significant — the effect is real but ~60× below threshold.

### B3 — flat-d  (conjunctive TOST; pass iff BOTH equivalence tests significant at slot's Holm-α)
| Model | (i) stated-comp d1 vs d5 diff [ci90] p (±0.10) | (ii) deriv usable d1 vs d∞ diff [ci90] p (±0.15) | slot p | verdict |
|---|---|---|---|---|
| **Qwen2.5-7B (anchor)** | +0.013 [−0.008,+0.034] p=0.0 | +0.084 [+0.035,+0.133] p=0.014 | 0.014 | **PASS** |
| qwen1p5b | **+0.118 [+0.063,+0.173]** p=0.71 (outside margin) | +0.074 p=0.009 | 0.71 | **FAIL — (i) not flat** |
| llama8b | +0.061 [+0.017,+0.106] p=0.079 | −0.012 p=0.0 | 0.079 | **FAIL — (i) n.s.** |
| olmo7b | (i) no paired families / INVALID | +0.167 [−0.320,+0.653] p=0.54 | — | **FAIL — INVALID** |
| qwen32b | +0.078 [+0.049,+0.108] p=0.112 | +0.007 p=0.0 | 0.112 | **FAIL — (i) n.s.** |

B3 fails everywhere. The failure is driven by component (i): stated-complement is **not flat** across d in the new models — it is elevated at d1 and declines toward d5 (the same d1 elevation that drives the B1 reversal), whereas the anchor is genuinely flat (0.031/…/0.021).

### Per-model roll-up
| Model | B1 | B2 | B3 | Holm slots passed | **Fully replicates?** |
|---|---|---|---|---|---|
| **Qwen2.5-7B (anchor, reference)** | PASS | PASS | PASS | 3/3 | **YES (reference)** |
| qwen1p5b | FAIL (reversed) | **PASS** | FAIL | 1/3 | **NO** |
| llama8b | FAIL (reversed) | **PASS** | FAIL | 1/3 | **NO** |
| olmo7b | FAIL (invalid) | FAIL (invalid) | FAIL (invalid) | 0/3 | **NO** |
| qwen32b | FAIL (reversed) | FAIL (magnitude) | FAIL | 0/3 | **NO** |

### Program-level verdict
- **Fully-replicating models: 0 / 4.** Rule = ≥3/4 with absolute floor 3 → **NOT MET. The boundary conditions do NOT replicate across open-weights models.**
- **Per-condition:** B1 0/4, B2 2/4, B3 0/4 — **no condition reaches ≥3/4.** No per-condition generality claim survives.
- olmo7b is additionally a §8.3 **model-exclusion candidate** (gold-eligible 506, double-filter yield 0.044, grid at 13.5% — far below any 25%-of-target reading; it passes MC-R2 so exclusion rests on the eligibility floor alone). Per §6.1 this changes nothing: with olmo excluded the denominator drops to 3 and the absolute floor of 3 would require **all three** of qwen1p5b/llama/qwen32b to fully replicate — none do. The verdict is identical under either reading.

---

## 4. Notable descriptive differences (Step 4) — paper-worthy

1. **The d=0 cliff is Qwen-7B-specific and reverses elsewhere (headline).** "Rejection is a d=0 visibility event" does not generalize. On qwen1p5b, llama8b, and qwen32b, explicit verbalized rejection (stated-complement) is *lower* at d=0 than at d=1, position-consistently and significantly. Mechanistically: at d=0 these models tend to silently re-derive the true attribute (high `valid_recovery`) without ever emitting "X is not <false>", whereas one hop away they more often verbalize the contradiction. The anchor's large d0 spike (0.247) is not reproduced by any model (new-model d0 stated-comp: 0.013–0.47).

2. **Usability-driven absorption collapses at 32B scale (clean scale-within-family result, the §10 M3-vs-M4 reading).** Derivational injection-dependence in the usable cells: **0.711 (Qwen-1.5B) → 0.807 (Qwen-7B) → 0.013 (Qwen-32B).** The 1.5B and 7B absorb a usable false categorical premise into their derivation; the 32B almost never does — it recovers correctly (usable-d1 valid = 0.617) or expresses doubt (0.317). Absorption via derivational usability appears to be a small/mid-scale phenomenon that vanishes with capability.

3. **stated-complement carries a real distance signal (opposite of "flat").** For qwen1p5b it rises then falls across d (d1=0.369, d5=0.224); this is the pre-registered "checking model" reading (§6.1) — a positive scientific finding that still counts as non-replication. The boundary story's "flat in refutation distance" claim is Qwen-7B-specific.

4. **Polarity is mixed, not uniformly exonerated.** B-P (secondary, `closure_valid` aff vs neg, ±0.10 TOST, pooled d{1,3}): **qwen32b reproduces polarity exoneration** (diff −0.022, TOST p=0.0, like the anchor). But **llama8b shows a large aff≫neg asymmetry** (aff valid 0.777 vs neg 0.474, diff +0.32 — it recovers false *affirmatives* far better than false *negations*), and **olmo7b shows the classic neg≫aff confound direction** (diff −0.167, underpowered, nfam=9) — a scoped, low-power revival of the polarity confound for olmo per §6. qwen1p5b is marginal (+0.060, TOST p=0.056).

5. **OLMo-2-7B-Instruct cannot sustain the base task.** Greedy gold solve 0.199 and a 4.4% double-filter yield (vs 0.88–1.0 solve on the others) mean the 4× oversample still left the grid at 13.5%. This is itself a reportable limitation on task portability, not a harness fault (world audits, prompt hash, and anchors all pass).

---

## 5. Housekeeping (Step 5) — done

`results/queue_status.json` top-level `"state"` corrected **`NOT_LAUNCHED` → `queue_complete`** after integrity verification, with an added one-line `note` recording the fix. Only those two keys changed; `current`, `history` (61 entries, terminal `queue_complete` rc=0), `queue`, `e1_registration_commit`, `reason`, `updated_at` are untouched. This unblocks the E9 waiter's gate 1; **E9 remains human-gated** regardless.

---

## Method notes / caveats

- Adjudicator = the locked `expd_confirmatory_analysis.py` functions (`stated_complement`, `mval`, `paired_family_stats`, `one_sided_greater`, `tost`, `cluster_boot`) applied to each model's `validated_outputs.jsonl` + `manifest.jsonl`. Cluster unit = `family_id` per registered §6 (the task brief's "problem_id" wording was read as the registered family cluster, which the locked machinery uses). Positions pooled per §6. B1/B2 significance uses the registered paired-family one-sided test with a corroborating unpaired family-cluster bootstrap of the point diff (both agree in sign/significance). B3 slot p = max of the two TOST p's (intersection-union), scored at the slot's Holm-α.
- Pipeline validation: running the identical adjudicator on the Qwen2.5-7B EXPD grid reproduces the registered anchor (B1 +0.216 vs registered +0.209; stated-comp d0 0.2467 / d1 0.0311 exactly match `EXPD_FULL_REPORT.md`; B2 +0.799; B3 pass) → fully replicates, confirming the negative results are model properties.
- Backend/doubt confound is disclosed upstream and does not touch the structural DVs used here (stated-complement, derivational inj-dep, closure-valid are cell-level backend-robust per the replay report); the anchor TOST gate on `benign_paraphrase`/`true_interruption` valid_recovery is a runner diagnostic, not a boundary slot.
