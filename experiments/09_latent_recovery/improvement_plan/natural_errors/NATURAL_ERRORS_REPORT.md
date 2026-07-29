# Natural-Error Characterization: How Qwen2.5-7B Actually Errs on PrOntoQA, vs. How We Plant Errors

Generated: 2026-07-06. Zero-GPU analysis; script `extract_natural_errors.py` in this directory.
Answers reviewer objection (Rylan): *"maybe your insights apply to how YOU inject falsehoods but not how MODELS actually err."*

## 1. Data

Every stored **gold (unperturbed) rollout** of Qwen2.5-7B-Instruct (rev `a09a3545`, greedy, shared
INSTR+FEWSHOT prompt) across all runs, re-validated with the paper's own validator:

| source | rollouts | valid | failed | failed w/ auditable world | notes |
|---|---|---|---|---|---|
| legacy pilot (`results/gold` + `data/pilot.jsonl`) | 219 | 168 | 51 | 51 | ProofsOnly grammar |
| EXPA gold screen (`EXPA_GLOBAL_EXPANSION`) | 2,810* | 0* | 2,810 | 106 | only screen-failures loaded; 2,704 from Composed-grammar worlds |
| EXPE gold expansion (`EXPE_GOLD_EXPANSION`) | 386* | 318 | 68 | 68 | *after cross-source problem-id dedup |
| EXPD synthetic worlds (`EXPD_MATCHED_GRADIENT`) | 300* | 0* | 300 | 300 | only gold_not_solved / not_valid loaded; in-grammar by construction |
| **total** | **3,715** | **486** | **3,229** | **525** | |

**Validator-blind mass (excluded, reported honestly):** 2,704 EXPA failures come from *Composed*
PrOntoQA files whose premises ("Everything that is a brimpus, a tumpus, or a yumpus is a jompus.")
and often targets are outside the validator grammar. There, "gold failure" conflates model error
with instrument limits (re-validated classes: 1,489 unparsed, 989 parroted, 226 derailed), so no
truth-level claims are made about them. All property analysis below uses the 525 failures whose
worlds fully parse (225 PrOntoQA + 300 EXPD).

**Control (walk soundness):** the same step-walk applied to all 486 valid rollouts produced
**0 fresh errors** (false-positive rate 0.000). Nine "failures" re-validate as valid_rederivation
(pipeline `solved()` did not strip discourse markers; footnote-level).

## 2. What a natural error is (extraction semantics)

Each failed rollout is walked sentence-by-sentence against the true closure (validator
`parse_world`/`closure`/`derivable`; multi-error generalization of the poisoned logic: an
underivable-even-given-prior-errors entity fact is a **fresh natural error**; later facts derivable
*only* through an error atom are **downstream use**; re-assertions are **echo**). Rule sentences are
checked against the question's premise set; unentailed novel rules are **fabricated rules** (rules
entailed as implications under the closure are counted separately as true derived generalizations,
not errors). Refutation distance of the error's complement is `shortest_rule_distance` (EXPC, Alg.
5 semantics) from the *visible prefix* state at the point of error; truth status follows
`audit_truth_status` (CWA for affirmative categories, open-world for negated categories,
undecidable attributes).

## 3. Natural error inventory (525 auditable failed rollouts)

| error object | PrOntoQA (225 rollouts) | EXPD (300 rollouts) | combined |
|---|---|---|---|
| fresh invalid entity-fact steps | **135** | **101** | **236** |
| fabricated rule statements (unentailed) | 369 | 749 | 1,118 |
| — of which load-bearing (license later facts) | 112 (30%) | 346 (46%) | 458 (41%) |
| — few-shot rule leakage / reversed real rule | 36 / 45 | 0 / 72 | 36 / 117 |
| entailed-true "derived rule" restatements (not errors) | 59 | 7 | 66 |
| unparsed entity-fact sentences (compound etc.) | 32 | 53 | 85 |
| off-entity facts | 1 | 0 | 1 |

Rollout-level failure taxonomy (PrOntoQA): 107/225 contain >=1 fresh entity-fact error; 81/225 fail
through other objects only (mostly fabricated-rule rambling; 55 of those have a load-bearing
fabricated rule); 21/225 are format-only (unparsed compound facts, all steps true); 16/225 are pure
truncation/derailment with **no falsehood at all**. Looping is common in failures (56/225 rollouts
repeat a sentence >=3 times).

## 4. Property distribution of fresh entity-fact errors vs. planted design points

### 4a. Natural distribution

| property | PrOntoQA (n=135) | EXPD (n=101) |
|---|---|---|
| category-affirmative | 115 (85%) | 33 (33%) |
| category-negated | 12 (9%) | 49 (49%) |
| attribute-affirmative / -negated | 3 / 5 (6%) | 12 / 7 (19%) |
| truth: false, CWA-unentailed category | 115 (85%) | 33 (33%) |
| truth: false, contradicted (opposite entailed) | 11 (8%) | 11 (11%) |
| truth: unknown (open-world neg-cat / undecidable attr) | 3 + 6 (7%) | 44 + 13 (56%) |
| refutation distance of complement from visible prefix | unreachable(>8): 124 (92%); contradicted ones at d = 0:1, 1:5, 2:3, 3:2 | unreachable: 90 (89%); contradicted at d = 5–8 |
| position (early/mid/late thirds) | 40 / 46 / 49 | 2 / 35 / 64 |
| claimed token appears in question vocab | 67 (50%) | 87 (86%) |
| design-usable toward target (atom feeds goal under closure) | 28 (21%) | 2 (2%) |
| goal-jump target assertions | 2 | 0 |

### 4b. Planted-family design points (for reference)

| family (where) | object | type/polarity | audited truth | refut. dist | usability | positions |
|---|---|---|---|---|---|---|
| benign paraphrase (legacy/EXPA/EXPD) | entity fact | as gold step | true | n/a | on-path | e/m/l |
| true interruption (EXPA/EXPD) | entity fact | affirmative | true | n/a | inert | e/m/l |
| distractor (legacy) | novel rule | cat->adj | unentailed | n/a | inert by design | e/m/l |
| contradiction (legacy) | entity fact | negated | false-contradicted | 0 | inert | e/m/l |
| one-hop / negstep (legacy/EXPA) | entity fact | negated | false-contradicted | 1 | inert | e/m/l |
| neghop2–5 (legacy) | entity fact | negated | false-contradicted | 2–5 | inert | e/m/l |
| global falsehood / wrong (legacy/EXPA) | entity fact | cat-affirmative | false-CWA | unreachable | often usable | e/m/l |
| EXPD attr cells (aff/neg x true/false) | entity fact | attribute | true or false-contradicted | 0,1,2,3,5 | inert | e/m/l |
| EXPD cat_false usable/inert | entity fact | cat-affirmative | contradicted@d / CWA@inf | 1,3,inf | usable vs inert | 2 positions |

### 4c. Coverage verdict

**On the axes the paper manipulates — object x type x polarity x audited truth x refutation
distance — the planted families cover 126/135 = 93% of natural invalid entity-facts on PrOntoQA,
and the modal natural error (affirmative unentailed category with no reachable refutation, 115/135
= 85%) is exactly the global-falsehood / `cat_false_*_dinf` design point.** Locally refutable
falsehoods (d<=5), the informative end of the paper's distance gradient, are *rare in nature*
(11/135 = 8%) — they are the counterfactual probe, not the natural mode; natural errors live at
the far end of the gradient, where the paper finds silent absorption. Position is covered (natural
errors occur at all thirds; planted at e/m/l).

Not spanned by any planted family:

1. **Fabricated rules** — the single most frequent natural error object (369 unentailed rule
   assertions vs 135 false entity-facts on PrOntoQA, 2.7x), including misquotes that reverse a
   real rule (45) and few-shot leakage (36). The legacy distractor family injects an unentailed
   rule but only in its *inert* form; the load-bearing 30% (which license later false facts) have
   no planted analog.
2. **Truth-undecidable claims** (9/135 = 7% on PrOntoQA; 57/101 = 56% on EXPD's grammar with
   negative rules): open-world negated categories and undecidable attributes. Planted families
   deliberately plant audited-false or audited-true statements only. Note these natural claims are
   unlicensed but not false — arguably outside a "falsehood-processing" study's scope, but the
   EXPD-grammar share is large enough to deserve a scoping sentence.
3. **Surface deviations**: half of natural false categories use tokens *not in the question*
   (novel blends like "vampus", few-shot vocabulary); planted errors always reuse in-world tokens.
   On the truth/distance axes these are still global falsehoods (no counterevidence anywhere), so
   this is a lexical, not logical, gap. Compound/unparsed formats (32 sentences) are likewise
   outside the planted (and validator) grammar.

## 5. Natural absorption vs. the planted-usable ~0.8 finding

Planted reference (EXPD `cat_false_usable`, derivational injection-dependence, positions pooled):
0.863 (d1), 0.768 (d3), 0.787 (dinf) — pooled **0.81** [0.78, 0.83]. Planted inert: 0.00–0.023.

Natural (per fresh entity-fact error, PrOntoQA):

| | used downstream (derivational) | n | rate |
|---|---|---|---|
| design-usable toward target | 15 | 28 | **0.54** [0.36, 0.70] |
| design-inert | 5 | 107 | **0.047** [0.02, 0.10] |
| all | 20 (+4 joint-only, +11 echo-only) | 135 | 0.15 |

Rollout-level (the "ignore vs use" question): among 107 PrOntoQA failed rollouts containing an
invalid step, the model **uses one downstream in 24 (22%)** (30/107 = 28% counting echoes) and
ignores them entirely in 78%. Combined with EXPD: 39/196 = 20% used.

**The usable-vs-inert absorption asymmetry reproduces in natural errors (54% vs 5%, ~11x), in the
same direction as planted (81% vs <=2%).** The natural usable rate sits below the planted 0.81 —
expected attenuation: natural errors co-occur with derailment/looping (so downstream derivation
often never happens for any atom), and natural usable errors are not position-controlled; the
small cell (n=28) makes 0.54 vs 0.81 a qualitative, not sharp, comparison. Fabricated rules show
the same gradient: 30% of unentailed rules are consumed downstream on PrOntoQA (46% on EXPD).

## 6. Recommended paper text

**Defense paragraph (supported by the data; recommended):**

> To check that our planted families resemble how the model actually errs, we audited every failed
> unperturbed rollout across our runs with the same validator (3,715 gold rollouts; 525 failures on
> fully-parseable worlds; the step-walk yields zero false positives on the 486 valid rollouts).
> The natural invalid entity-facts we extract (n=236, of which 135 on PrOntoQA) concentrate on
> exactly the design point our global/unentailed cells probe: 85% are affirmative category claims,
> false under the closed-world audit, whose complement is unreachable from the visible prefix
> (>8 rule hops, 92%) — locally refutable falsehoods, the informative end of our distance gradient,
> are rare in nature (8%), which is precisely why they must be planted to be studied. The
> absorption asymmetry we report for planted errors reproduces naturally: derivationally usable
> natural falsehoods are consumed downstream 54% of the time (15/28) versus 5% (5/107) for inert
> ones (planted: 81% vs <=2%). One natural mode our families do not emulate — and we scope our
> claims accordingly — is that the model's most frequent error *object* is a fabricated or
> misspliced rule rather than a false entity-fact (369 vs 135 on PrOntoQA), 30% of which license
> later false facts; our distractor family covers only the inert version of this mode.

**Honest scoping sentence (if a single sentence is preferred):**

> Our planted families span 93% of the naturally occurring invalid entity-facts in the model's own
> failed rollouts (dominated, 85%, by the unentailed-category/no-local-refutation design point),
> but do not cover the model's frequent fabricated-*rule* errors or truth-undecidable claims, so
> our conclusions are scoped to entity-fact falsehoods.

## 7. Caveats

- Single model (Qwen2.5-7B-Instruct), greedy decoding, shared prompt template; EXPE used vLLM.
- 2,704 Composed-grammar failures are excluded as validator-blind; the natural-error distribution
  is therefore conditional on the ProofsOnly-style grammar the paper's experiments use.
- EXPD worlds are procedurally generated (nonce vocabulary, negative rules); their natural errors
  skew to open-world negated-category claims (56% truth-undecidable) — quoted separately throughout.
- Downstream-use attribution for multi-error rollouts credits any single error atom that suffices;
  4 PrOntoQA steps needed a joint set (reported separately, not in the 15/28).
- Natural usable-absorption (0.54) vs planted (0.81) is not position- or length-matched.

## 8. Artifacts

- `extract_natural_errors.py` — extraction/analysis script (imports `src/` read-only; no GPU).
- `natural_errors.jsonl` — one row per natural error object (entity-fact, fabricated-rule,
  off-entity, unparsed) with all properties.
- `natural_rollouts.jsonl` — one row per auditable failed rollout (taxonomy + absorption flags).
- `summary.json` — all aggregates; `run.log` — full run output.
- Mirror: local scratchpad `latent_recovery/natural_errors/`.
