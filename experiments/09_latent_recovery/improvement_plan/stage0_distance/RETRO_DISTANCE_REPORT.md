# Stage-0 M5 — Retroactive Refutation-Distance Audit

Generated: 2026-07-02T13:49:26.320695+00:00

Distance owner: `shortest_rule_distance` (src/expc_polarity_control.py:340-356), imported
unmodified; prefix state built by code-native `world_state(question, entity, prefix_steps)`;
planted-claim complement via `opposite_pred(statement_predicate(...))`.
d is reported from a depth-64 BFS (identical algorithm, larger cap); rows where the
code-native default cap of 8 would disagree are counted per family
(`native_max8_disagreements`).

## d-distribution per family x position

| family | position | designed d | n | d=0 | d=1 | d=2 | d=3 | d=4 | d=5+ | d=inf | unparseable | finite-d frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXPA:benign_paraphrase | pooled | true_control(no design) | 450 | 0 | 0 | 0 | 0 | 0 | 0 | 450 | 0 | 0.0 |
| EXPA:benign_paraphrase | early | true_control(no design) | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:benign_paraphrase | late | true_control(no design) | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:benign_paraphrase | mid | true_control(no design) | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:global_falsehood | pooled | inf | 450 | 0 | 0 | 0 | 0 | 0 | 0 | 450 | 0 | 0.0 |
| EXPA:global_falsehood | early | inf | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:global_falsehood | late | inf | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:global_falsehood | mid | inf | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:one_hop_falsehood | pooled | 1 | 450 | 67 | 345 | 28 | 9 | 1 | 0 | 0 | 0 | 1.0 |
| EXPA:one_hop_falsehood | early | 1 | 150 | 29 | 119 | 1 | 1 | 0 | 0 | 0 | 0 | 1.0 |
| EXPA:one_hop_falsehood | late | 1 | 150 | 13 | 113 | 16 | 7 | 1 | 0 | 0 | 0 | 1.0 |
| EXPA:one_hop_falsehood | mid | 1 | 150 | 25 | 113 | 11 | 1 | 0 | 0 | 0 | 0 | 1.0 |
| EXPA:true_interruption | pooled | true_control(no design) | 450 | 0 | 0 | 0 | 0 | 0 | 0 | 450 | 0 | 0.0 |
| EXPA:true_interruption | early | true_control(no design) | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:true_interruption | late | true_control(no design) | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| EXPA:true_interruption | mid | true_control(no design) | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 | 0 | 0.0 |
| legacy:perturbed | pooled | none(wrong-category, v1; often entailed-true) | 390 | 0 | 0 | 0 | 0 | 0 | 0 | 390 | 0 | 0.0 |
| legacy:perturbed | early | none(wrong-category, v1; often entailed-true) | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0 | 0.0 |
| legacy:perturbed | late | none(wrong-category, v1; often entailed-true) | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0 | 0.0 |
| legacy:perturbed | mid | none(wrong-category, v1; often entailed-true) | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0 | 0.0 |
| legacy:perturbed_contradiction | pooled | 0 | 390 | 390 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| legacy:perturbed_contradiction | early | 0 | 130 | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| legacy:perturbed_contradiction | late | 0 | 130 | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| legacy:perturbed_contradiction | mid | 0 | 130 | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| legacy:perturbed_distractor | pooled | rule-injection(unparseable as entity fact) | 390 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 390 | 0.0 |
| legacy:perturbed_distractor | early | rule-injection(unparseable as entity fact) | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0.0 |
| legacy:perturbed_distractor | late | rule-injection(unparseable as entity fact) | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0.0 |
| legacy:perturbed_distractor | mid | rule-injection(unparseable as entity fact) | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0.0 |
| legacy:perturbed_falsehood | pooled | inf | 105 | 0 | 0 | 0 | 0 | 0 | 0 | 105 | 0 | 0.0 |
| legacy:perturbed_falsehood | early | inf | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 0 | 0.0 |
| legacy:perturbed_falsehood | late | inf | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 0 | 0.0 |
| legacy:perturbed_falsehood | mid | inf | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 0 | 0.0 |
| legacy:perturbed_neghop2 | pooled | 2 | 260 | 56 | 54 | 116 | 23 | 9 | 2 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop2 | early | 2 | 130 | 41 | 22 | 50 | 13 | 2 | 2 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop2 | mid | 2 | 130 | 15 | 32 | 66 | 10 | 7 | 0 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop3 | pooled | 3 | 199 | 29 | 54 | 45 | 55 | 13 | 3 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop3 | early | 3 | 130 | 18 | 39 | 26 | 36 | 9 | 2 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop3 | mid | 3 | 69 | 11 | 15 | 19 | 19 | 4 | 1 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop4 | pooled | 4 | 134 | 20 | 30 | 43 | 20 | 18 | 3 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop4 | early | 4 | 96 | 13 | 22 | 31 | 15 | 14 | 1 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop4 | mid | 4 | 38 | 7 | 8 | 12 | 5 | 4 | 2 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop5 | pooled | 5 | 91 | 10 | 20 | 22 | 24 | 13 | 2 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop5 | early | 5 | 69 | 10 | 14 | 17 | 16 | 10 | 2 | 0 | 0 | 1.0 |
| legacy:perturbed_neghop5 | mid | 5 | 22 | 0 | 6 | 5 | 8 | 3 | 0 | 0 | 0 | 1.0 |
| legacy:perturbed_negstep | pooled | 1 | 390 | 61 | 290 | 15 | 19 | 3 | 2 | 0 | 0 | 1.0 |
| legacy:perturbed_negstep | early | 1 | 130 | 14 | 110 | 2 | 3 | 1 | 0 | 0 | 0 | 1.0 |
| legacy:perturbed_negstep | late | 1 | 130 | 21 | 91 | 9 | 6 | 2 | 1 | 0 | 0 | 1.0 |
| legacy:perturbed_negstep | mid | 1 | 130 | 26 | 89 | 4 | 10 | 0 | 1 | 0 | 0 | 1.0 |
| legacy:perturbed_paraphrase | pooled | unknown | 390 | 0 | 0 | 0 | 0 | 0 | 0 | 390 | 0 | 0.0 |
| legacy:perturbed_paraphrase | early | unknown | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0 | 0.0 |
| legacy:perturbed_paraphrase | late | unknown | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0 | 0.0 |
| legacy:perturbed_paraphrase | mid | unknown | 130 | 0 | 0 | 0 | 0 | 0 | 0 | 130 | 0 | 0.0 |

## Key question (a): do designed distances hold?

- **EXPA:benign_paraphrase**: true_control(no design); finite-d fraction = 0.0, unparseable = 0/450.
- **EXPA:global_falsehood** (designed d=inf): 1.0 of 450 rows measure d=inf; finite-d fraction = 0.0.
- **EXPA:one_hop_falsehood** (designed d=1): 0.7667 of 450 rows measure exactly d=1.
- **EXPA:true_interruption**: true_control(no design); finite-d fraction = 0.0, unparseable = 0/450.
- **legacy:perturbed**: none(wrong-category, v1; often entailed-true); finite-d fraction = 0.0, unparseable = 0/390.
- **legacy:perturbed_contradiction** (designed d=0): 1.0 of 390 rows measure exactly d=0.
- **legacy:perturbed_distractor**: rule-injection(unparseable as entity fact); finite-d fraction = 0.0, unparseable = 390/390.
- **legacy:perturbed_falsehood** (designed d=inf): 1.0 of 105 rows measure d=inf; finite-d fraction = 0.0.
- **legacy:perturbed_neghop2** (designed d=2): 0.4462 of 260 rows measure exactly d=2.
- **legacy:perturbed_neghop3** (designed d=3): 0.2764 of 199 rows measure exactly d=3.
- **legacy:perturbed_neghop4** (designed d=4): 0.1343 of 134 rows measure exactly d=4.
- **legacy:perturbed_neghop5** (designed d=5): 0.022 of 91 rows measure exactly d=5.
- **legacy:perturbed_negstep** (designed d=1): 0.7436 of 390 rows measure exactly d=1.
- **legacy:perturbed_paraphrase**: unknown; finite-d fraction = 0.0, unparseable = 0/390.

## Key question (b): EXPA global-cell contamination

- Globally-checkable-only (`global_falsehood`) rows audited: **450**
- Rows with FINITE refutation distance (mis-binned): **0**
- Contaminated fraction: **0.0** (Wilson95 [0.0, 0.0085])
- Unparseable planted claims in this cell: 0

Contamination is at or below the 2% threshold; re-binned rates reported anyway for completeness.

### Headline rates: full cell vs strictly-d=inf subset (pooled)

| subset | n | closure_valid | poisoned (inj-dep) | doubt | parroted | derailed |
|---|---|---|---|---|---|---|
| full cell | 450 | 0.3933 [0.3493, 0.4392] | 0.3422 [0.2999, 0.3872] | 0.0133 [0.0061, 0.0288] | 0.2 [0.1657, 0.2394] | 0.0467 [0.0307, 0.0703] |
| strict d=inf | 450 | 0.3933 [0.3493, 0.4392] | 0.3422 [0.2999, 0.3872] | 0.0133 [0.0061, 0.0288] | 0.2 [0.1657, 0.2394] | 0.0467 [0.0307, 0.0703] |
| finite-d (contaminated) | 0 | - | - | - | - | - |

### By position

| position | subset | n | closure_valid | poisoned (inj-dep) | doubt | parroted | derailed |
|---|---|---|---|---|---|---|---|
| early  full | 150 | 0.4467 [0.3694, 0.5266] | 0.2333 [0.1728, 0.3072] | 0.0067 [0.0012, 0.0368] | 0.2333 [0.1728, 0.3072] | 0.06 [0.0319, 0.1101] |
| early  strict d=inf | 150 | 0.4467 [0.3694, 0.5266] | 0.2333 [0.1728, 0.3072] | 0.0067 [0.0012, 0.0368] | 0.2333 [0.1728, 0.3072] | 0.06 [0.0319, 0.1101] |
| late  full | 150 | 0.3733 [0.3, 0.453] | 0.42 [0.344, 0.5] | 0.02 [0.0068, 0.0571] | 0.1733 [0.1211, 0.2419] | 0.02 [0.0068, 0.0571] |
| late  strict d=inf | 150 | 0.3733 [0.3, 0.453] | 0.42 [0.344, 0.5] | 0.02 [0.0068, 0.0571] | 0.1733 [0.1211, 0.2419] | 0.02 [0.0068, 0.0571] |
| mid  full | 150 | 0.36 [0.2876, 0.4394] | 0.3733 [0.3, 0.453] | 0.0133 [0.0037, 0.0473] | 0.1933 [0.1381, 0.2639] | 0.06 [0.0319, 0.1101] |
| mid  strict d=inf | 150 | 0.36 [0.2876, 0.4394] | 0.3733 [0.3, 0.453] | 0.0133 [0.0037, 0.0473] | 0.1933 [0.1381, 0.2639] | 0.06 [0.0319, 0.1101] |

## Coverage / schema notes

- legacy family perturbed: rows=522 audited=390 skip-marked=39 not-in-gold-validated-cohort=93 schema-missing=0
- legacy family perturbed_contradiction: rows=522 audited=390 skip-marked=39 not-in-gold-validated-cohort=93 schema-missing=0
- legacy family perturbed_distractor: rows=522 audited=390 skip-marked=39 not-in-gold-validated-cohort=93 schema-missing=0
- legacy family perturbed_falsehood: rows=522 audited=105 skip-marked=378 not-in-gold-validated-cohort=39 schema-missing=0
- legacy family perturbed_neghop2: rows=522 audited=260 skip-marked=204 not-in-gold-validated-cohort=58 schema-missing=0
- legacy family perturbed_neghop3: rows=522 audited=199 skip-marked=269 not-in-gold-validated-cohort=54 schema-missing=0
- legacy family perturbed_neghop4: rows=522 audited=134 skip-marked=347 not-in-gold-validated-cohort=41 schema-missing=0
- legacy family perturbed_neghop5: rows=522 audited=91 skip-marked=402 not-in-gold-validated-cohort=29 schema-missing=0
- legacy family perturbed_negstep: rows=522 audited=390 skip-marked=40 not-in-gold-validated-cohort=92 schema-missing=0
- legacy family perturbed_paraphrase: rows=522 audited=390 skip-marked=39 not-in-gold-validated-cohort=93 schema-missing=0

## Semantics notes

- `d` uses only DIRECT rule applications from cat-atoms in the prefix state, exactly as
  `shortest_rule_distance` implements (BFS over `direct_rule_map`); `d=0` means the
  complement is literally an atom of the prefix state.
- `unparseable` = the planted statement does not parse as an entity fact for the proof
  entity (`statement_predicate` returned None). Rule injections (distractor family) land
  here by construction; refutation distance is not defined for them in this fragment.
- `complement_closure_derivable_from_prefix` (row-level output) cross-checks the BFS:
  it must agree with finite d in this fragment (closure == unbounded BFS).
- Legacy rows are restricted to the gold-validated cohort (gold rollout solved AND
  `valid_rederivation`), mirroring `src/validator.py:main`, so distances correspond to
  the locked, analyzed rows.

## Addendum: EXPA one_hop_falsehood re-binned by MEASURED d (exploratory)

The designed d=1 cell contains 14.9% d=0 rows (the planted negation contradicts a fact
already visible in the prefix state) and 8.4% d>=2 rows. Stored-outcome rates by measured d:

| measured d | n | doubt | closure_valid | poisoned |
|---|---|---|---|---|
| 0 | 67 | 0.284 | 0.552 | 0.000 |
| 1 | 345 | 0.368 | 0.690 | 0.014 |
| 2 | 28 | 0.179 | 0.750 | 0.071 |
| 3 | 9 | 0.000 | 0.556 | 0.000 |
| 4 | 1 | 0.000 | 1.000 | 0.000 |

Within this single locked condition, doubt at measured d=0 (0.284) is LOWER than at d=1
(0.368) — replicating the cross-family d=0<d=1 non-monotonicity (MASTER_PLAN MAJOR-4 note)
inside one cell. The legacy neghop-k ladder is heavily smeared (e.g., neghop2: only 44.6%
of rows measure d=2), because hop count in the model's own proof is not rule-BFS distance
from the prefix state; any dose-response claim over k should be re-binned by measured d
using row_level_distances.jsonl.
