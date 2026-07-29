# E1 Adjudication — Adversarial Verification

**Verifier:** independent recompute from RAW per-model `validated_outputs.jsonl` + `manifest.jsonl` (no adjudicator intermediate file read).
**Method:** own script `/tmp/verify_e1.py` — re-reads raw jsonl, recomputes `stated_complement` from raw `continuation` via the *registered* DV parser only, reads `inj_derivational` straight off the raw rows, and **reimplements from scratch** (scipy) the paired-family one-sided t, the TOST, Holm-m3, and the §8.2 exclusion classifier. Ran against all 4 subject models + the Qwen2.5-7B anchor grid.

## VERDICT: adjudication CONFIRMED. No discrepancy that changes any call.

Every headline number reproduces to 3 decimals; every pass/fail follows mechanically; exclusion/attrition handling matches the registration; integrity figures reproduce exactly. Three disclosed method notes are re-examined below and all are verdict-immaterial.

---

## 1. Headline numbers — independent recompute vs adjudicator (3-dp)

### B1 — d0−d1 stated-complement (rejection DV)
| model | my sc_d0 | my sc_d1 | my diff | adj diff | match |
|---|---|---|---|---|---|
| qwen1p5b | 0.2268 | 0.3689 | **−0.1421** | −0.142 | ✓ |
| llama8b | 0.0511 | 0.1956 | **−0.1444** | −0.144 | ✓ |
| olmo7b | 0.4697 | 0.2778 | +0.1919 | +0.192 | ✓ |
| qwen32b | 0.0133 | 0.0822 | **−0.0689** | −0.069 | ✓ |
| anchor 7B | 0.2467 | 0.0311 | **+0.2156** | +0.216 | ✓ |

Reversal on qwen1p5b/llama8b/qwen32b confirmed: point diff negative, one-sided(>0) p ≈ 1.0 (0.99998 / 1.0 / 0.99996) ⇒ the *reverse* direction is significant. Anchor d0/d1 rates match the registration's cited 0.247/0.031 exactly.

### B2 — usable−inert derivational injection-dependence (reuse DV, pooled d{1,3,∞})
| model | my usable | my inert | my diff | adj diff | match |
|---|---|---|---|---|---|
| qwen1p5b | 0.7111 | 0.0111 | **+0.7000** | +0.700 | ✓ |
| llama8b | 0.9378 | 0.0144 | **+0.9233** | +0.923 | ✓ |
| olmo7b | 0.4419 | 0.0000 | +0.4419 | +0.442 | ✓ |
| qwen32b | 0.0133 | 0.0000 | **+0.0133** | +0.013 | ✓ |
| anchor 7B | 0.8068 | 0.0079 | **+0.7989** | +0.799 | ✓ |

Scale-within-family (usable derivational absorption) reproduced: **0.7111 (1.5B) → 0.8068 (7B) → 0.0133 (32B)** — matches the reported 0.711/0.807/0.013.

### B3 — TOST bounds (≥1 bound independently recomputed per model; all shown)
| model | (i) sc d1 vs d5 diff [ci90] p (m=.10) | (ii) deriv d1 vs d∞ diff [ci90] p (m=.15) | slot p |
|---|---|---|---|
| qwen1p5b | +0.1182 [+0.063,+0.173] p=0.7073 | +0.0741 [+0.022,+0.126] p=0.0087 | 0.7073 |
| llama8b | +0.0614 [+0.017,+0.106] p=0.0786 | −0.0117 [−0.041,+0.018] p=0.0000 | 0.0786 |
| qwen32b | +0.0783 [+0.049,+0.108] p=0.1122 | +0.0067 [−0.009,+0.022] p=0.0000 | 0.1122 |
| olmo7b | INVALID (0 paired fam d5) | +0.1667 [−0.320,+0.653] p=0.5353 | 0.5353 |
| anchor 7B | +0.0132 [−0.008,+0.034] p=0.0000 | +0.0839 [+0.035,+0.133] p=0.0135 | 0.0135 |

All match the adjudicator's B3 table to 3 dp (qwen1p5b (i) +0.118 p=0.71; llama (i) +0.061 p=0.079; qwen32b (i) +0.078 p=0.112; anchor (i) +0.013 / (ii) +0.084 p=0.014). B3 fails 4/4, driven by component (i) not being flat.

---

## 2. Registered criteria applied as written — checks

- **B1 gate** `p<Holm-α AND diff≥+0.10` — applied verbatim; magnitude on the pooled point diff, sign one-sided>0. ✓
- **B2 gate** `p<Holm-α AND diff≥+0.30`, derivational inj-dep pooled over d{1,3,∞}. ✓ (qwen32b fails on magnitude 0.013≪0.30 exactly as registered; a *significant* but trivially small effect.)
- **B3** conjunctive TOST, margins ±0.10 (i) / ±0.15 (ii), slot p = intersection-union max of the two TOST p's. ✓
- **Holm m=3, α=0.05** per model, no alpha-splitting inside a slot. Reproduced (thresholds 0.0167/0.025/0.05 assigned by ascending p). ✓
- **DV definitions** unchanged from the locked instrument: strict stated-complement (string parser) for rejection, `inj_derivational` for reuse. My independent read of the raw `continuation` through the registered parser reproduces the rejection rates exactly, so the DV was not silently altered. ✓

## 3. Exclusion / attrition handling — matches registration

- §8.2 grain is **per (cell × position)**, target 150/cell/pos as literally registered (§4 table + §8.2). My min-cell-pos n: qwen1p5b B1 **147** → flag(100-149), scored (not invalid) ✓; llama/qwen32b all cells 150 (full) ✓.
- **olmo7b n<50 ⇒ INVALID**: my min-cell-pos n = 22 (B1) / 11 (B2) / 4 (B3) — all `INVALID<50`, scored as "not replicating" (conservative). Matches. ✓
- §8.3 model-exclusion candidacy: olmo gold-eligible 506 / grid 893 = **13.5%** of the 6,600 target, far below 25%. As the adjudicator states, the program verdict is identical whether olmo is scored-as-fail or excluded (floor of 3 unmet either way). ✓

## 4. Integrity figures — reproduced from raw

| model | oversample | world_audits_passed / world_count | gold solve | eligible (rate) | grid rows |
|---|---|---|---|---|---|
| qwen1p5b | 4× | 11520/11520 (0 fail) | 10170/11520 = **0.883** | 3259 = **0.283** | **6477** |
| llama8b | 2× | 5760/5760 (0 fail) | 5654/5760 = **0.982** | 3115 = **0.541** | **6600** |
| olmo7b | 4× | 11520/11520 (0 fail) | 2292/11520 = **0.199** | 506 = **0.044** | **893** |
| qwen32b | 2× | 5760/5760 (0 fail) | 5760/5760 = **1.000** | 5732 = **0.995** | **6600** |

Every figure matches the adjudicator's integrity table and the registered §3 oversample plan (M1 Llama 2×, M2 OLMo 4×, M3 Qwen32B 2×, M4 Qwen1.5B 4×). Zero manifest-join misses on all four models (my loader dropped 0 rows for missing run_id). ✓

## 5. Pass/fail follows mechanically

- Per model: qwen1p5b B2 only (1/3) · llama8b B2 only (1/3) · olmo7b 0/3 · qwen32b 0/3 ⇒ **0/4 fully replicate**.
- Program rule ≥3/4 (floor 3) ⇒ **NOT MET**. Per-condition B1 0/4, B2 2/4, B3 0/4 ⇒ **none reach ≥3/4**. All reproduced by my independent roll-up. ✓
- Anchor recompute fully replicates (B1+B2+B3 pass) ⇒ pipeline validated; the negative is a model property. ✓

## 6. Method notes examined — all verdict-immaterial

1. **B1/B2 significance test.** Registration's B1/B2 test column literally reads "one-sided > 0, **cluster bootstrap**"; the adjudicator used the paired-family one-sided **t** as the primary p and the cluster bootstrap only as corroboration. My independent check used the paired-family t and reproduces every call. This substitution is **immaterial to all eight B1/B2 verdicts**: B1 fails on sign/magnitude/validity for all 4 models (no p can rescue a reversed or invalid slot); B2 passes (p→0, huge effect) or fails on magnitude (qwen32b 0.013) / validity (olmo) independent of the p mechanism. Disclosed by the adjudicator. **No impact.**
2. **Anchor B1 +0.216 vs registered +0.209.** The registration cites +0.209 [0.153,0.264] for the anchor cliff; my recompute (and the adjudicator's) gives +0.2156, while the underlying d0/d1 stated-comp rates (0.2467/0.0311) match the registration exactly. The ~0.007 gap is a pre-existing anchor-estimator nuance in `EXPD_FULL_REPORT`, disclosed, and immaterial (anchor passes decisively either way).
3. **`family_id` cluster unit ("problem_id" wording).** The brief said "problem_id"; the locked machinery and registration §6 use `family_id` (mirrored-quadruple). Disclosed; this is the registered cluster unit, so correct.

## 7. Housekeeping edit (queue_status.json) — validated

`state` = `queue_complete`, `note` added, `history` = 61 entries with terminal `{event: done, model: ALL, rc: 0, stage: queue_complete}`, `e1_registration_commit` = `bdb098a1…` intact. The prior `NOT_LAUNCHED` was genuinely a stale status-writer artifact (history terminal is queue_complete rc=0), so the correction is justified and scoped to the two claimed keys. ✓

---

## Bottom line

The E1 adjudication is **CONFIRMED on independent recompute from raw data**. 0/4 models replicate; the boundary conditions do NOT replicate; B1 is significantly reversed on 3/4 models; the 32B usability collapse (0.013) and olmo catastrophic attrition (13.5%) are real. No discrepancy alters any headline number or any pass/fail call. The three method notes are all disclosed and verdict-immaterial.
