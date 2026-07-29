# MASTER REVISION PLAN — Workshop Reject → ICLR 2027

Bundle root: `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/3b21ecc2-2fca-4bc4-94a4-df05162f5e6f/scratchpad/latent_recovery_review` (all relative paths below are relative to this root; the canonical copies live on skampere2 at `/lfs/skampere2/0/eobbad/free-energy`).

Process note: the orchestrator passed the bundle root as the literal string `undefined`; this plan is written to the scratchpad bundle root above. Two verifier agents could not find `CONTEXT.md`/`REVIEW.md` locally and one confirmed `REVIEW.md` at this root — before finalizing any rebuttal wording, re-check the reviewer's exact sentences against `REVIEW.md` here.

Authoritative facts used throughout (per STATUS.md, which overrides the draft):
- The draft's headline (falsifiability gradient) is **dead**. Pre-registered family PLAN2 v1.1 (commit `545b35db`, sha256 `112da1a5...`, timestamped 2026-07-02T14:05:29-07:00, filed before generation): **1 of 6 hypotheses passed (Holm m=6)** — the polarity-equivalence deconfound H2' (`experiments/09_latent_recovery/improvement_plan/STATUS.md`, 2026-07-02 entries).
- What replaced it: (i) rejection is a **d=0 visibility** event (stated-complement 0.247 at d=0 vs 0.031 at d=1, flat d≥1; `results/EXPD_MATCHED_GRADIENT/EXPD_FULL_REPORT.md`, C-0 — registered, direction-open, **non-family/no-verdict**, disclose as such); (ii) absorption tracks **derivational usability**, not distance (usable reuse 0.863/0.768/0.787 at d=1/3/∞ vs inert 0.000–0.023; registered H4 trend FAILED p=0.996, exploratory reversal p=0.004 — disclose exploratory status); (iii) verbal rejection is **lexically cued** (EXPE H5': REFUTING ≈ FREQ_MATCHED, p=0.629; both ≫ TEMPLATE, p=3e-8; `results/EXPE_EVIDENCE_MOVER/EXPE_FULL_REPORT.md`); (iv) absorption is **in-stream-production-bound** and **training-recipe-bound**, with a **universal recompute boundary** (`results/EXPH_API_MODELS/EXPH_REPORT.md`, `results/EXPH2_FRONTIER_FOLLOWUPS/EXPH2_REPORT.md`).

---

## 1. Verdict table

| Point | Reviewer concern | Verdict | One-line evidence (file) |
|---|---|---|---|
| **W1** | Locality confounded with polarity/syntax/lexical content | **Answered** | EXPD matched design pins all axes (parse gaps ≤0.67pp), gradient did not survive; polarity exonerated by H2' TOST pass (+0.054 within ±0.10, p=0.0012) — `results/EXPD_MATCHED_GRADIENT/EXPD_FULL_REPORT.md` |
| **Q1** | "How much of one-hop vs global survives identical measurement?" | **Substantially answered; one zero-GPU decomposition outstanding** | Graded effect: none (H4 flat-to-reversed). Survives: 8.7pp strict hop-sound gap w/ non-overlapping CIs (0.300 vs 0.213) + d=0 cliff + usability dichotomy — `improvement_plan/stage0_strict/STRICT_METRICS_REPORT.md`. Caution: dropping the 14.9% d=0-contaminated rows **widens** the legacy gap (d=0 rows have LOWER doubt 0.284 vs 0.368) — `improvement_plan/stage0_distance/RETRO_DISTANCE_REPORT.md`; the "manufactured signal" account rests on the EXPE lexical-priming leg, not d=0 contamination |
| **W2/Q2** | Strict stepwise validator not used for main results | **Answered (with honest reframe)** | A genuinely stepwise hop-sound + strict-repair standard was built post-submission and run over 6,147 stored rows (report tables; STATUS says 4,857 — reconcile before citing a total); gradient survives compressed (0.671→0.300 vs 0.393→0.213), SRR 0.251 vs 0.002 — `improvement_plan/stage0_strict/STRICT_METRICS_REPORT.md`. Must disclose: the draft's "strict replay" (main.tex 350-365) was final-match, NOT stepwise; EXPB's 900 rows were skipped (schema) |
| **W3** | Certificate result is a parse artifact | **Partially** | Reviewer confirmed: unparsed 4.0%→31.7%, closure-valid bounds [0.467,0.783] vs [0.500,0.537], ranking p=0.46 — `improvement_plan/stage0/STAGE0_REPORT.md`, `results/EXPB_LOCAL_CERT_FLIP/summary_tables.json`. Missing: judge pass over the 118 unparsed rows; stale plans (storyPlan.md L97, MASTER_PLAN.md §5/§8) still route EXPB to main text and must be superseded. Robust survivor: doubt contrasts (+0.370, discordant 111-0) — report as contrasts only |
| **W4/Q4** | Only solved/parseable proofs perturbed; test natural errors | **Partially** | Natural-error audit of 3,715 gold rollouts: 93% of natural invalid entity-facts inside planted design space; modal error (85%) IS the global cell; usability asymmetry reproduces (54% vs 5%) — `improvement_plan/natural_errors/NATURAL_ERRORS_REPORT.md`. Missing: protocol actually RUN on natural-error prefixes; load-bearing fabricated-rule plant family (the modal error OBJECT, 1,118 combined, 41% load-bearing) |
| **W5** | Doubt regex is a weak lexical proxy | **Answered — converted into a finding** | Prereg demoted doubt to descriptive before review (PLAN2 v1.1 DV registry); FREQ arm proves rejection is lexically cued (mid-matched judge-doubt 0.141 vs 0.073); doubt fires on TRUE plants (0.47-0.71) and collapses 0.273→0.01 on format change (EXPH bridge); backend-sensitive −8 to −15pp — `results/EXPE_EVIDENCE_MOVER/EXPE_FULL_REPORT.md`, `improvement_plan/tooling/judge_v2/JUDGE_V2_REPORT.md` (floor = 0/150, pilot/smoke-scoped), `improvement_plan/tooling/replay_full/REPLAY_FULL_REPORT.md`. Open: pre-registered human-kappa audit (PLAN2 L98-99) not yet run |
| **W6** | Controls only on Qwen2.5-7B | **Partially — now aimed at the paper's core** | All confirmatory verdicts are Qwen2.5-7B-only (`EXPD_FULL_REPORT.md` L4, `EXPE_FULL_REPORT.md` L4). Breadth exists descriptively: 9 API models, lab-level split + universal recompute boundary — `results/EXPH2_FRONTIER_FOLLOWUPS/EXPH2_REPORT.md`. Fix = replication matrix (E1 below). Legacy cross-model cells are too thin to cite (falsehood mid n=13/30/9; Llama has negstep only) |
| **W7** | Only 11 references, thin related work | **Partially** | Bundle main.tex already has 23 refs, ALL cited (no orphans — the four "orphans" are cited via \citet at line 507); ~45-candidate verified sweep exists (WP7). To fix: rebuild the 2-paragraph section around the new story; add PrOntoQA-OOD (arXiv:2305.15269, mandatory — used at main.tex L105); fix neumann2025cascade venue (it is REAL: HEAL Workshop @ CHI 2025, not "unpublished"); verify all 2026 arXiv IDs |
| **Q3** | GSM8K/arithmetic corruptions undescribed; metrics incommensurable | **Partially** | Fair hit (metric split conceded only in a footnote, main.tex 400-404). Replacement exists at pilot scale: EXPG interpreter-validated program-trace regime (parse 1.00; adjacent conflict 25.7% resistance; re-read vs recompute split) — `results/EXPG_PROGTRACE_PILOT/REPORT.md`; EXPG is PARKED pending re-anchoring (STATUS.md L65). Must add construction paragraph for legacy arms (verify constructor script on cluster first — not in bundle) |

---

## 2. The chosen story

**Thesis (one law, merged from framings A+C; B's recipe-split becomes the stakes section, not the headline):**

> Under a pre-registered, audited perturb-and-validate protocol, LLM in-stream verification is **zero-hop and incentive-blind**: models reject a planted falsehood only when its contradiction is directly visible in the prefix (stated-complement 0.247 at d=0 vs 0.031 at d=1, flat for d≥1), never perform even one hop of proof search, and silently reuse derivationally useful falsehoods at ~0.8 no matter how close the refuting evidence sits. This holds only under in-stream production (0.863 → 0.15 → 0.00 under chat re-presentation), is training-recipe-bound at the re-read level (Anthropic-2025 models reject essentially every audited plant; gpt-4.1 absorbs at 0.58, above the open-weights anchor), and is universal at the recompute level — **no model tested re-verifies a computed value** (deep_kc5 absorption 1.00; scoped to the 3 models run on those cells until E2 lands).

**Honest presentation of the reversal (non-negotiable):** the intro states that we pre-registered the intuitive nearby-evidence-gradient account (print commit `545b35db` + CONSORT funnel: 11,577 validated rows, 0 world-audit failures, 57 fail-closed rejections) and it survived 1/6 tests; a scoreboard figure of all 6 registered verdicts appears in the body; every replacement claim is labeled **registered-confirmatory / registered-non-family / exploratory** (C-0 is non-family direction-open; the H4 reversal is exploratory). The multi-model replication (E1) exists precisely to give the boundary conditions their own confirmatory pass.

**Why not headline the lab split (Strategy B):** red-team is right that it is descriptive, one-seed, 2-lab, anchor-metric-linked, and optically a vendor scoreboard. It stays in the paper as one scoped "stakes" section + appendix panel, with the universal recompute column as its non-partisan center, the within-format sonnet-5 (0.010) vs gpt-4.1 (0.580) comparison stated prominently, and the contamination rebuttal (nonce generated worlds; same models absorb 1.00 on recomputable plants) as a named subsection. Never say Anthropic models check "every time" (supported: reject every audited falsehood, valid 0.97-1.00, reuse 0.000; the d=1-check transcript is Haiku-attributed) and never say gpt-4.1 "never checks" (unmeasured; only reuse 0.58 is on record).

**Cuts (professor's sprawl warning — executed, not promised):**
| Cut | Where it goes |
|---|---|
| "Falsifiability gradient" terminology, Fig-1 dose-response, neghop ladder | Gone; one appendix paragraph: parse artifact (parsed-only validity flat 0.730-0.788) + d-smearing (76.7%/44.6%/2.2% at designed d — cite RETRO report, not STATUS's wrong "14%") |
| Legacy EXPA locality gradient as main result | Appendix: "the confounded observation that motivated the pre-registered test", with strict-vs-permissive accounting table |
| EXPB certificate flip | Appendix instrument-lesson + joint-outcome figure (data already in `summary_tables.json`); **supersedes** storyPlan.md L97/L129 and MASTER_PLAN.md §5 L78, §8 L100 which still route EXPB to main text |
| EXPC-127 polarity table | Appendix; keep one footnote (affirmative-local doubt 0.436 is the HIGHEST cell — kills the negation-token account) |
| GSM8K + chained arithmetic as standalone table | EXPG regime map replaces them; arithmetic = degenerate recompute corner (locked n=99); GSM8K = flagged non-audited corroboration with the construction paragraph (answers Q3) |
| Latent signatures (§3.6), attention knockout, QA-reframing, verification-prompting pilots | One sentence each + appendix |
| Scale/lineage tables | One lineage row per model; Mistral gets an explicit exclusion sentence; R1-Distill exclusion justified (deliberation channel) |
| Absolute doubt rates anywhere | Contrasts only (house rule, STATUS.md replay entry) |
| "Recovery/repair" vocabulary; the names "latent recovery" and "Nearby Evidence" | Strict-repair (SRR) sense only; retitle the paper |

Title short-list (decide with Elyas): "Zero-Hop Verifiers: Language Models Check Planted Errors Only When the Contradiction Is Visible" / "Seen, Not Sought: Usability, Not Local Falsifiability, Governs Whether LLMs Absorb Errors in Their Own Reasoning" / "Re-Read, Never Recompute" (if E2+panel land strongly).

---

## 3. Experiment queue (deduplicated across WPs; cluster = idle 8xH200, vLLM port done ~2,074 tok/s)

### MUST (blockers for the ICLR variant)

| # | What | Spec | Models | Cost | Answers |
|---|---|---|---|---|---|
| **E1** | **Pre-registered replication of the boundary conditions** (EXPD core grid: cat usable/inert × d{1,3,∞}; aff/neg attr × d{0,1,3,5}; anchors; + H2' TOST per model) | Fresh timestamped mini-registration; same generator/harness; strict stated-complement primary; report d=0 cliff, usability gap, flat-d per model with cluster CIs | Llama-3.1-8B, OLMo-2-7B, Qwen2.5-32B, Qwen2.5-1.5B — **note 3/4 Qwen-family; ask Elyas about adding one more non-Qwen (e.g. Qwen3→ no; Gemma/DeepSeek-7B-class)** | <8 GPU-h total (EXPD full grid was 0.09 GPU-h); 2-4 person-days QA; budget 3-4× gold attempts for 1.5B/OLMo attrition | W6 + red-team "single-model confirmatory base" (the #1 fixable rejection reason) |
| **E2** | **EXPG-FULL second regime** (un-park with STATUS L65 fixes: re-anchored plants, kc5 2.5× oversample, deep-cell grammar disclosure; MAJOR-5/MOD-9/MOD-10) | Register family-level contrast (re-readable vs computed) as the ONE confirmatory pair — NOT the k_r gradient (known flat for Qwen: kr1=kr8=1.00); k_r secondary (frontier-only gradient 0.043→0.217); run 32B judge pass; ~13k gens | Qwen2.5-7B + spot cells on E1 models | ~3-6 GPU-h gen + 1-2 h judge + 4-6 days eng (`improvement_plan/reports/regime2.md` §2-5) | Q3, professor's "second serious regime", red-team toy-domain kill-shot |
| **E3** | **Forced-check mediator** (powered EXPF n=150×3 incl. forced-Wait-on-true 2×2, + EXPG_FORCE in deep/kr8 cells) | Does forcing initiation rescue validity/reuse at d=1? Failure branch (rescue only where re-readable) confirms recompute boundary causally — publishable either way; one Holm slot in amendment | Qwen2.5-7B (+1 E1 model if cheap) | ~0.5-1 GPU-day + 2-3 days plumbing (legacy STAGE=force code exists) | verdict.md's "single addition that most raises ICLR probability"; turns taxonomy into mechanism |
| **E4** | **Human validation package** (blinded 2-rater audit per validatorPlan.md Part 4: ~380-430 rows incl. 20 SRR positives; + ~200-row rejection/doubt precision-recall sample) | Gates: κ≥0.8/axis, ≥95% overall, ≥90%/class; resolves the M3 stated-reliance all-zeros check. **This is a pre-registered commitment (PLAN2 L98-99, 303-304: second independent non-author rater)** — SRR must not ship in reviewer-facing text before it, or be caveated "pending human validation" | — | 2-4 person-days; **needs Elyas + one labmate (approval/recruiting)** | W2, W5, professor weakness 3 ("validator is a black box"); Qwen-judging-Qwen objection |
| **E5** | **Licensed-vs-unlicensed stated-complement split + inert-d=1 residual adjudication** | Classify every stated-complement row (EXPE full arms; EXPD d=0 + cat cells; inert d=1 bump 0.193 vs 0.023) as licensed (closure-valid derivation of complement) vs lexically-cued; the flagged exploratory follow-up in STATUS.md | analysis of stored rows | 0 GPU, ~1-2 days; **cluster-side** (row JSONLs not in bundle) | W1 (is any one-hop checking real?), W5, red-team "lexical-confound regress on the surviving claim" |
| **E6** | **Legacy-gap decomposition** (re-bin EXPA one-hop vs global by measured d on strict + permissive DVs, cluster-bootstrap CIs) | Produces Q1's literal number. Frame correctly: expect the gap to persist/slightly WIDEN after d=0 exclusion (d=0 rows are low-doubt); the "manufactured" leg is lexical priming, not contamination | analysis | 0 GPU, 0.5-1 day; needs cluster (`row_level_distances.jsonl` on skampere2) | Q1, W1 |
| **E7** | **Judge_v2 completion pass**: EXPD full grid (11,577 rows) + EXPB's 118 unparsed rows; + EXPB joint-outcome stacked figure (data already in `summary_tables.json` — figure task only) | Closes "registered secondary missing on the flagship experiment" and converts EXPB bounds to judge-informed imputation | Qwen2.5-32B judge | ~15-30 GPU-min (EXPE pass: 153 GPU-s/2,104 rows) + 0.5 day | W3, W5, prereg completeness |
| **E8** | **Natural-error continuation + fabricated-rule plant family** | (a) resume generation from 236 natural-error prefixes, unchanged harness, same DVs; (b) new plant family whose object is an unentailed RULE (reversed/spliced/leakage subtypes × usable/inert × position) | Qwen2.5-7B (+1 more if cheap) | (a) 1-2 days + few GPU-h; (b) 2-4 days + GPU-h | W4/Q4 literal ask + the audit's biggest honest gap; (b) is a genuinely new result |

### SHOULD
- **S1** Strict/hop-sound pass over EXPD+EXPE stored rows, **explicitly re-running H2' under hop-soundness** (the family's only pass is currently adjudicated on closure-valid — the very DV the paper demotes; EXPE closure-valid is 0.31-0.46, so this is substantive, not a formality). 0 GPU, ~1 day.
- **S2** EXPB schema adaptation for strict metrics (900 rows; "adaptable later" per STATUS). 0 GPU, 0.5-1 day.
- **S3** EXPE-lite H5' three-arm on Llama-3.1-8B (lexical cueing not a Qwen quirk). ~0.2 GPU-h + 1 day.
- **S4** Second-model natural-error audit (Llama or OLMo golds + rerun extractor). ~1 day.
- **S5** Position/length-matched natural-absorption re-analysis (0.54 vs 0.81 currently unmatched). ~1 day; superseded if E8a lands.
- **S6** Cross-regime matched-strictness table (PrOntoQA hop-sound beside EXPG trace-valid). 0 GPU, ~1 day.
- **S7** solved() discourse-marker one-line fix + re-validate the 9 misfiled rollouts + disclosure. Hours.
- **S8** Legacy Llama falsehood-cell backfill OR delete the unsupported main.tex 457-461 sentence; retire Mistral remnants explicitly. 0.5 day.
- **S9** Lineage-panel hardening: +1-2 labs (Gemini-tier), n→300 on anchor cells, 3 temp-0.7 seeds; and/or the **open-weights post-training ladder** (OLMo-2/Tulu-3 stages, base-vs-instruct) which turns the black-box split into an inspectable training-stage result. GPU near-free; **API ~$150-400 — NEEDS ELYAS PRE-APPROVAL** (cap $100, $30.81 spent, ~$69 headroom).
- **S10** Frontier check on ~50 natural-error prefixes via prefill (EXPH channel). **~$5-10 API — needs approval.**

### NICE
CRUXEval externally-authored arm (~1-2 days, <1 GPU-h; answers "author-written grammar" phrasing); gsm8k_gradient MAJOR-3 fix (appendix only); cross-family judge spot-check (Llama-70B local or ~$5 API); reasoning-dose arm (gpt-5.1 effort sweep, ~$50-150 — approval); contamination-proof API re-panel (~$30 — approval); citation adds (Huang/Tyen/Schaeffer/Wang etc. — largely subsumed by §4).

---

## 4. Writing plan

House rules first: **all LaTeX is written together with Elyas in chat — never autonomously** (collab rule, CONTEXT.md + memory). Numbers bind to locked JSONs via `src/paper_artifacts.py`; backend rule stated once (never mix HF/vLLM; doubt as contrasts); one metrology box (DV definitions from `improvement_plan/reports/validatorPlan.md` Part 2, incl. the 18-item disclosure checklist and the "recovery" ban).

| Section | Content | Evidence files |
|---|---|---|
| Abstract + §1 Intro | One-law thesis; prereg-reversal-as-rigor paragraph (commit hash printed); scope-guard sentence (non-reasoning conditions); stakes = recipe-bound + recompute boundary | STATUS.md; EXPD/EXPE reports |
| §2 Protocol | Audited perturb-and-validate instrument: truth audits (355/390 inversion), measured-d, fail-closed integrity, design-matrix table (PLAN2 §9.7 — the visual answer to W1); natural-error grounding syllogism (85% of natural errors live where checking is impossible → planting is the design's justification) | `stage0_distance/RETRO_DISTANCE_REPORT.md`, PLAN2.md, `natural_errors/NATURAL_ERRORS_REPORT.md` §4-6 |
| §3 Registered reversal | Scoreboard figure (6 hypotheses, verdicts, effect sizes); H2' polarity pass; funnel | `EXPD_FULL_REPORT.md`, `EXPE_FULL_REPORT.md` |
| §4 Boundary condition 1: d=0 visibility | C-0 step function as the psychophysics figure (labeled registered/non-family); EXPE H1' flat; EXPG adjacent 25.7%; E5 licensed-split hardening | EXPD, EXPE, EXPG reports |
| §5 Boundary condition 2: usability | Lead with FLATNESS not the asymmetry (anti-tautology framing); natural reproduction 54% vs 5%; E1 replication matrix table | EXPD report; natural_errors §5 |
| §6 Boundary condition 3: production + recompute boundary | EXPH bridge (0.863→0.15→0.00; doubt 0.273→0.01); EXPG/EXPH2 regime map (arithmetic = degenerate corner; GSM8K flagged non-audited + construction paragraph = Q3 answer); note draft's own §3.4 conjecture is hereby confirmed (continuity asset, main.tex 392-395) | EXPH, EXPH2, EXPG reports; regime2.md |
| §7 Instrument findings | Lexically-cued rejection (H5', mid-matched numbers); doubt demotion evidence (backend, true-plant firing); strict-vs-permissive accounting; human-kappa table (E4) | EXPE report, JUDGE_V2_REPORT.md, REPLAY_FULL_REPORT.md, STRICT_METRICS_REPORT.md |
| §8 Stakes: lineage panel (scoped, descriptive) | Lab split + universal recompute column; within-format comparison; contamination subsection; claim-by-model **evidence-tier table** (tier 1 confirmatory / 2 replication / 3 descriptive API / 4 legacy-appendix) | EXPH2 report |
| §9 Related work (~0.5-0.75 pp — reconcile with storyPlan.md L176's 0.5pp budget, Q7 below) | WP7's verified ~35-45 set in clusters A-I; explicit one-line contrasts for the 5 scoop-adjacent papers (von Recum 2602.07470, Self-Correction Bench 2507.02778, Rethinking Reflection 2504.04022, RFEval 2602.17053, Amjith 2512.17079); Validation Gap 2502.11771 as corroboration; appendix protocol-property contrast table | WP7 sweep |
| §10 Limitations | Upper-bound selection (retained); fabricated-rule scoping until E8b; single-model confirmatory base until E1; grammar-conditionality; EXPD worlds easier than natural cohort (anchor TOST +0.149, MOD-12 scoping); cat-cell freq==2 deviation | reports as cited |
| Appendices | Legacy EXPA/EXPC/EXPB necropsies; strict tables; replay defense; API panel detail; response-to-reviewer map (W1-W7/Q1-Q4 → experiment) | — |

Bib hygiene (can be prepped now, wired in chat): add PrOntoQA-OOD `2305.15269` (mandatory); fix neumann2025cascade venue to HEAL@CHI-2025 (real paper — do NOT delete as hallucinated); un-lump the line-507 \citet into individual discussions; verify every 2026 arXiv ID against abs pages; execute STATUS.md L86 Rylan citations (Schaeffer 2307.10573, Wang 2212.10001 — pick one year convention, Mirage beside Dziri). No orphan-fixing needed (refuted).

Old draft keeps: protocol description bones, validator definitions, PrOntoQA setup, the six-model cohort list (recast per evidence-tier table), the dissociation subsection's stance (now evidence-backed). Old draft loses: everything in the cuts table, the abstract, the title.

---

## 5. Timeline (today = 2026-07-22; ICLR 2027 abstract ~mid-Sept, full ~late-Sept 2026)

| Window | Work |
|---|---|
| **Jul 22 – Aug 1** | Zero-GPU/cheap: E5, E6, E7, S1, S2, S7, S8; EXPB joint-outcome figure; bib verification pass; draft mini-registrations for E1/E2/E3; **get Elyas's approvals (Q2, Q3 below)** |
| **Aug 1 – 15** | E1 replication matrix (register → run → analyze); E2 EXPG-FULL (re-anchor, register, run); E8a natural-error continuation; writing sessions with Elyas begin (skeleton + §2-3) |
| **Aug 15 – 31** | E3 forced-check mediator; E4 human audit (both raters); E8b fabricated-rule family; S3/S4/S6; optional S9/S10 if approved; writing §4-8 |
| **Sept 1** | **GO/NO-GO GATE (professor's rule: TMLR unless causal isolation substantially strengthened + one clean story + second regime).** GO to ICLR iff: (a) E1 boundary conditions replicate on ≥3/4 models, (b) E2 registered contrast lands, (c) E3 done (either branch), (d) E4 gates pass. Else: **default TMLR**, submit Oct 2026 with current package + appendices. Dual submission prohibited — TMLR waits for this gate either way |
| **Sept 1 – 15** | Full draft assembly with Elyas; scoop-watch re-sweep; freeze numbers/locks |
| **~Sept 15-19** | ICLR abstract deadline |
| **~Sept 24-28** | ICLR full paper deadline; response-to-reviewer appendix finalized |

Standing item: monthly arXiv scoop-watch (WP7 spec) + full re-sweep in submission week. Consensus venue read across all three strategies: current package = strong TMLR accept; with E1+E2+E3+E4 landed, ICLR is a genuine ~40-50% shot.

---

## 6. Open questions for Elyas/Brando (decision-ready)

1. **Story sign-off**: adopt the one-law thesis (§2) with the recipe split as a scoped stakes section rather than the headline? (Strategy B's split-as-headline is the alternative; red-team rates it ~25-35% at ICLR vs ~40-50% for the one-law + replication package.)
2. **API budget**: raise the $100 cap? Needed for S9 panel expansion (~$150-400), S10 natural-prefix frontier check (~$5-10), reasoning-dose arm (~$50-150), contamination re-panel (~$30). Current spend $30.81. A "no" only drops SHOULD/NICE items, not the ICLR gate.
3. **Second human rater**: who? (E4 requires an independent non-author rater per the prereg — a labmate; ~2-3 hours of their time.)
4. **Replication-matrix composition**: 3 of 4 planned models are Qwen-family. Add one more non-Qwen lineage (which?) to harden the W6 answer, at ~0.1 GPU-h marginal cost?
5. **Title + rename**: pick from the short-list in §2; also, do we rename the repo/paper away from "latent recovery"? (Terminology debt invites "framing disagrees with results" reviews.)
6. **EXPB fate**: confirm demotion to appendix, superseding storyPlan.md's recorded MAIN-§4.3 routing and MASTER_PLAN's Fig-1 inset. (Recommendation: yes; EXPD C-0 + EXPE H6 own the causal role now.)
7. **Related-work length**: storyPlan.md budgets 0.5pp; WP7 proposes ~1pp + appendix table. Recommendation: 0.75pp main + appendix contrast table. OK?
8. **Sept 1 gate criteria**: sign off on the four GO conditions in §5, and on TMLR-in-October as the default on a NO.
9. **Reasoning-dose arm**: run it (pre-empts "just turn on reasoning") or accept the scope-guard sentence alone?
10. **EXPG registration**: confirm the family-level contrast (re-readable vs computed) as the single registered primary — the originally planned k_r gradient is known-dead for Qwen and must not be registered.
