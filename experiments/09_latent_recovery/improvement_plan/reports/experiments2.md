# Complete Experiment Inventory — latent_recovery codebase

Mirror root: `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/ad633660-4a3e-4cbc-99dd-093a0feae95e/scratchpad/latent_recovery/exp` (canonical: `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`). All numbers below read from artifacts on disk in the mirror. Primary model everywhere unless stated: **Qwen/Qwen2.5-7B-Instruct** (rev `a09a35458c...`), greedy decode, max_new_tokens=192, PrOntoQA-OOD.

## 1. Locked full-schema experiments (RESULTS_LOCK.md, git `8894a05c`, locked 2026-06-23)

| Exp | Tests | N | Headline | Status |
|---|---|---|---|---|
| **EXPA_GLOBAL_EXPANSION** (`results/EXPA_GLOBAL_EXPANSION/`) | Expands global-falsehood cell from n=35→150/position with matched benign/one-hop/true-interruption conditions | 1800 validated rows (4 cond × 3 pos × 150); 4832 raw gens; eligibility audit 22,132 rows | global_falsehood: valid .393, poisoned .342, doubt .013 (n=450) vs benign .849/.000/.080, one-hop .671/.016/.336, true-interruption .856/.000/.020 | **Locked/final** (in paper; PAPER_CHANGELOG notes a stale-writer late-append caveat: unsummarized rows retained, result treated as partial/disclosed) |
| **EXPB_LOCAL_CERT_FLIP** (`results/EXPB_LOCAL_CERT_FLIP/`) | Adds a true local certificate C that falsifies the planted global falsehood F, vs irrelevant-certificate control | 900 rows (3 cond × 300), 183 problem clusters, 100 paired triples/position | LOCAL_CERT: poisoned .027 vs .153 baseline / .240 irrelevant (McNemar p≈0); doubt .38 vs .01/.02; **valid re-derivation .467 does NOT beat irrelevant control .50 (p=.46)** — supportive for detection/anti-poisoning, not for clean recovery | **Locked/final** (in paper as null-preserved) |
| **EXPC_POLARITY_CONTROL** (`results/EXPC_POLARITY_CONTROL/`) | 2×2 locality (local/global) × polarity (pos/neg), fully paired at all 3 positions, identical prompt | **7 problems, 84 validated rows (n=21/cell)** | doubt local−global = +.238 CI [.071,.405]; valid_recovery contrasts null (−.071 CI [−.214,.071]); poisoning insufficient data | **Locked but explicitly a noisy pilot / failed strict control** (paper reports it as such) |
| Smoke/dev variants | `EXPA_GLOBAL_EXPANSION_SMOKE` (24 rows), `EXPB_..._SMOKE`, `_SMOKE_FIXED`, `_SMOKE_MASK`, `_FAILED_PROMPTIDS`, `_PREP`, `_TEST` (empty) | — | — | **Smoke/abandoned**, kept for audit trail |

## 2. Legacy Qwen2.5-7B primary sweep (`results/validated_summary*.json`, n=130/cell except falsehood, cohort 168 gold-validated of ~500)

valid_rederivation_rate by early/mid/late (poisoned in parens where nonzero):
- **wrong (pre-audit, 92% entailed-true)**: .823/.754/.862 — `validated_summary.json`
- **paraphrase (benign)**: .769/.815/.915
- **distractor**: .731/.715/.762
- **contradiction**: .731/.792/.854, ack 30/35/23
- **negstep (k=1 falsehood)**: .646/.638/.746, ack 51/33/24, poisoned 0
- **falsehood (global category, n=35/cell)**: .343/.229/.286, **poisoned .286/.600/.429, ack ≈0** — silent absorption
- naive final-answer recovery (`stats.json`): .938/.969/.963 (n=161) — the parroting-inflation finding

## 3. Distance-k dose-response (item b) — exact N per k per position (Qwen-7B, `results/validated_summary_neghop*.json`)

| k | early n | mid n | late n | valid (early) | ack (early) |
|---|---|---|---|---|---|
| 1 (negstep) | 130 | 130 | 130 | .646 | 51/130 (.392) |
| 2 | 130 | 130 | **absent** | .562 | 42/130 (.323) |
| 3 | 130 | 69 | absent | .531 | 30/130 (.231) |
| 4 | 96 | 38 | absent | .594 | 22/96 (.229) |
| 5 | 69 | 22 | absent | .594 | 17/69 (.246) |

Note: **k=4,5 exist but plateau (~.59 valid) and were not in the UPDATE.md headline curve (which stops at k=3 + global)**; mid-position Ns collapse with k (availability), no late cells for k≥2. 32B replication: neghop2 early/mid 93/93 (valid .753/.774), neghop3 93/38 (.710/.842). Plus **`results_ctx/summary.json` (context_check.py, Batch-4)**: G-arm (graph distance k=1,2,3 fixed early, n=130 each: valid .654/.569/.577, doubt .400/.331/.238) vs T-arm (token distance of licensing rule front/mid/back + 520-run permutation control: valid .646/.654/.708/.671, doubt flat ~.39–.45) — **decay is in G, flat in T**; not in the paper.

## 4. Model families beyond Qwen (item a)

| Dir | Model | Sweep | N/cell | Headline | Status |
|---|---|---|---|---|---|
| `results_1p5b` | Qwen2.5-1.5B | wrong+negstep+falsehood, 3 pos | 55; falsehood 13 | negstep valid .56/.53/.67; falsehood poisoned .31/.62/.54 | final, low power |
| `results_32b` | Qwen2.5-32B | wrong+negstep+falsehood+neghop2,3 | 93; falsehood 30 | negstep .70/.74/.71; falsehood valid .73/.70/.60, poisoned .10/.20/.37 (absorption attenuates) | final |
| `results_olmo` | OLMo-2-1124-7B-Instruct | **full 3-family behavioral** (paraphrase/negstep/falsehood), 3 pos | 76; falsehood 9 | negstep .61/.53/.62 with **ack≈0** (recovery replicates, verbalization doesn't) | final |
| `results_llama` | Llama-3.1-8B-Instruct | **negstep only** | 107 | valid .55/.53/.72, ack=0, high unparsed (19–38) | **partial** (no paraphrase/falsehood summaries) |
| `results_mistral` | Mistral-7B-Instruct-v0.2 | 3 families but **n=4/4/1 per cell** despite 181 gold-validated | 4 (falsehood 1) | uninterpretable | **effectively abandoned/failed sweep — do not cite** |
| `results/r1` | DeepSeek-R1-Distill-Qwen-7B | paraphrase+negstep, visible-body injection | 122/cell; gold 147/500 solved | negstep valid .689, **ack_visible .672** vs paraphrase .008; gold think-channel doubt 1.0; reopened_think 0 | final (in paper) |

Paper (main.tex line 112) uses: Qwen 1.5B/7B/32B, OLMo-2, Llama-3.1, R1-Distill. Mistral is nowhere.

## 5. Asset check for follow-ups (item c)

- **Matched perturbations**: `src/matched_pert.py` + `results/matched_rho.json` — perturbed continuations resampled at null's T=0.8 n=4, mid injection, paraphrase vs negstep; h1–h4 medians: paraphrase 22.4→1.36, negstep 16.3→1.58 (n≈53→41). Benign≈falsehood persists → latent-metric demotion is final. Also `rho_stability.json` (rho medians 1.9–8.2 across 7 null configs, IQRs span 2 orders — unstable), `functional_rho.json` (per-family per-position h1–h8 with CIs, n=130), `geometry.json` (per-layer L0/7/14/21/28 raw+projected medians, n≈700/family), `shuffle_control.json` (own-gold JSD .464 vs other-gold .693, ratio 1.49, n=1272), `probes/` (kill-gated: best .68–.78 < .80 threshold; layer-0 mean-pool ≈ deep layers = lexical echo).
- **Polarity**: EXPC above (n=7 pilot) + full eligibility audit already on disk (23,539 rows).
- **Anchor knockout**: `src/anchor_knockout.py` (attention-mask=0 on "Prove: <target>" span, paraphrase+negstep, mid) exists but **`results/anchor_knockout/` is EMPTY in the mirror — no output; treat as not-run/not-synced**. The QA-format variant DID run: `results/anchor_qa/summary.json` — goal-truth-unknown QA format, mid injection: negstep recovered .571, ack .571 (n=35); paraphrase .639/.361 (n=36); gold 72/130 solved. Not in paper.
- **Doubt suppression/forcing** (`results_doubt/`, Batch-4, not in paper): probe.json — doubt direction linearly decodable, balanced acc .75–.77 at layers 7/14/21 (n=965, passes kill-gate). suppress_summary.json (negstep mid, n=130): projection ablation cuts doubt .246→.046 while valid falls only .654→.615 (token-ban control .615) — **recovery survives doubt suppression**. force_summary.json (global falsehood mid, n=35): forcing "Wait," flips valid .286→.714 and poisoned .429→.029 (neutral-token control .229/.400; wait-on-true control .629 valid, 0 poison). Strong causal asset, unused.
- **Doubt judge**: `results/doubt_judge.json` — Qwen-32B judge vs regex at mid, agreement .76–1.00, contrast sharpened (falsehood judge-doubt .000, negstep .285).
- **GSM8K/arith**: `results/gsm8k/summary.json` n=57 (gold 57/250): recovered .088, poisoned .877. `results/arith/summary.json` n=99/position: recovered .00–.02, poisoned-next-step .99–1.0. Both in paper (regime map). `src/gsm8k_gradient.py` (early/mid/late × k=1..3 GSM8K sweep) exists but **no `results/gsm8k_gradient/` output — never run**.
- **Verify-instruction prompt** (`results_verify/`, n=130 negstep / 35 falsehood): "verify each fact" instruction raises ack (negstep early 76/130) but **lowers** valid negstep (.554/.469/.692) and only partially helps falsehood (valid .457/.343/.286; poisoned .229/.371/.486 — worse late). Cited via `paper_artifacts.py` macros (mid cells).
- **Cluster stats**: `results/cluster_stats.json` — fixed-effect logistic ORs w/ problem-cluster bootstrap (global_false valid OR .078 [.045,.128] vs benign; one-hop doubt OR 3.59).

## 6. Raw text availability for GPU-free recomputation (item d)

**The mirror contains NO raw generations** — all `hs/` dirs (gold, perturbed*, null*) are empty and no `runs.jsonl`/`rollouts.jsonl`/`raw_generations.jsonl`/`validated_outputs.jsonl` were synced; the mirror has only summaries, reports, run_metadata, code, and the PrOntoQA datasets (47 MB). On the **cluster canonical copy** (per RESULTS_LOCK hashes + code paths) full raw text exists and is sufficient to recompute stricter validators CPU-only:
- `results/EXPA_GLOBAL_EXPANSION/raw_generations.jsonl` (4832) + `validated_outputs.jsonl` (1800); EXPB raw 900 + validated 900; EXPC raw 91 + validated 84 (all hash-locked).
- Legacy: `results*/gold/rollouts.jsonl` (`gen_text`), `results*/perturbed*/runs.jsonl` (stores full `continuation` string — `src/perturb.py:122`), `results/{gsm8k,arith,r1,anchor_qa}/runs.jsonl`, `results/validated.jsonl`.
Any stricter-metric reanalysis must run on skampere2 (or after rsync), not from this mirror.

## 7. KNOWN FINDING verified (item e): EXPC --target 7 vs 127 eligible

**Confirmed.** `src/expc_polarity_control.py:1476` → `p.add_argument("--target", type=int, default=7)`; selection is `selected = candidates[: args.target]` (line 776). Eligibility audit (both `RESULTS_LOCK.md:104` and `EXPC/summary_tables.json`): `fully_matched_2x2_available: 127`, of which only 7 were generated (84 validated rows). **A --target 127 re-run is ~18× the pilot: 127 × (4 conditions × 3 positions) = 1524 perturbed generations + 127 original-proof generations ≈ 1651 greedy 7B generations at ≤192 new tokens.** The candidate pool and 23,539-row eligibility audit are cached (prepare() skipped if pool exists) and the generate loop resumes past existing run_ids. Pilot timing (run_metadata 20:04:03 → report 20:05:39 for 91 generations) implies roughly 1–4 s/generation on one GPU → **~30 min–2 h single-GPU, well under half a day even conservatively; n/cell rises 21→381**, which would convert the professor-flagged "noisy pilot" polarity control into a powered experiment at trivial cost. Caveat: EXPC availability shows local-positive cells are grammatically constrained to attribute templates (token_mean 4.8 vs 7.2–8.5) — the grammar imbalance persists at any n and should be reported as in the pilot.

## 8. Status map vs paper

- **In paper (locked)**: primary 7B family sweep, neghop dose-response (k≤3 + global), EXPA/EXPB/EXPC, size sweep (1.5B/32B), OLMo, Llama, R1, GSM8K+arith regime map, verify-instruction mid cells, functional-rho descriptive mention, cluster stats.
- **Ran but unused (paper-ready ammunition for the causal-isolation critique)**: `results_ctx` G-vs-T decomposition, `results_doubt` suppression/forcing causal test, `anchor_qa`, `doubt_judge`, neghop4/5, matched_rho/rho_stability/geometry/probes/shuffle_control (demotion evidence).
- **Code-only, never run**: `gsm8k_gradient.py`, `anchor_knockout.py` (empty results dir).
- **Abandoned/failed**: Mistral arm (n≤4/cell), all EXPA/EXPB SMOKE dirs, EXPB_LOCAL_CERT_FLIP_TEST (empty), false-ontology arm (dataset `prontoqa/4hop_ProofsOnly_falseontology.json` exists, no results dir).

Key docs: `exp/PLAN.md` (pre-registrations incl. Batch-2/4), `exp/UPDATE.md` (v1 truth-audit inversion — original "wrong" family was 92% entailed-TRUE; addenda with all batch outcomes), `exp/RESULTS_LOCK.md` (hashes, row counts, immutability; note EXPB/EXPC/EXPA listed WRITABLE at lock in mirror copy), `exp/PAPER_CHANGELOG.md` (claim-calibration history), `PROFESSOR_FEEDBACK.md` (TMLR-first recommendation; wants causal isolation, validator spec, "recovery" language tightened, less sprawl, one more task regime).