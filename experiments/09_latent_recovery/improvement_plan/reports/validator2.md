All questions are now answerable from source. Composing the report.

# Validator & Metrics Pipeline — Reverse-Engineered Specification

Source root: `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/ad633660-4a3e-4cbc-99dd-093a0feae95e/scratchpad/latent_recovery/exp/src/` (canonical: `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/src`). All citations `file:line` relative to that dir.

---

## 1. Proof grammar (what parses)

**Sentence segmentation**: split on `(?<=\.)\s+` — whitespace after a literal period (`validator.py:72,130`; identical `common.py:52`). A generation lacking `". "` separators lumps into one "sentence".

**Discourse-marker stripping** (`validator.py:23,25-28`): leading `Therefore|Thus|Hence|So|Then|It follows that|We know that|This means( that)?` + optional comma is removed and the next char uppercased, *before* parsing. Note the strip list does **not** include `However`, `But`, `Wait`, `Actually` — those sentences fail to parse (see §9).

**Rule syntax** (`parse_rule`, `validator.py:33-54`), anchored regexes, period stripped:
- `(Every|Each|All) X is (a|an) Y` → `(x, ("cat", y))`
- `(Every|Each|All) X is not (a|an) Y` → `(x, ("not_cat", y))`
- `(Every|Each|All) X is [not] ADJ` → `(x, ("adj", adj, polarity))`
- Plural forms: `Xes are Yes`, `Xes are not Yes`, `Xes are [not] ADJ` (lhs must end in `-es`)

**Fact syntax** (`parse_fact`, `validator.py:56-68`): subject must match `[A-Z]\w*` (and equal the tracked entity when passed):
- `E is (a|an) C` → `("cat", c)`; `E is not (a|an) C` → `("not_cat", c)`; `E is [not] ADJ` → `("adj", a, pol)`

**Singularization** (`_sg`, `validator.py:30-31`): strip trailing `es` only — works for PrOntoQA's `-pus/-puses` nonce nouns, nothing else. **Fragile**: `E is wumpus` (dropped article) parses as an *adjective* fact, not a category fact (`validator.py:65-67` catches anything the article patterns missed).

Anything else (multi-clause sentences, "Wait, that contradicts...", cross-entity statements) does not parse; see §6 for its fate.

## 2. Representation

- **Predicate atoms**: `("cat", c)` / `("not_cat", c)` / `("adj", a, bool_polarity)`. Negation is a *distinct atom type or flipped polarity flag*, not logical ¬ — there is **no inconsistency check**: the state can simultaneously hold `("cat","wumpus")` and `("not_cat","wumpus")`.
- **Rules**: pairs `(lhs_category, rhs_atom)` — a single unary-Horn fragment, one category antecedent (`validator.py:91`).
- **World**: `parse_world(question)` (`validator.py:70-85`) splits the question and buckets each sentence into rules / facts / `unparsed` (unparsed world sentences are silently discarded in `validator.py`; expc at least counts them, `expc_polarity_control.py:391`).
- **Entity**: the **first word of the first sentence of the model's own gold rollout** (`validator.py:184,200,231`; `perturb.py:49-53`). Facts about other entities are not tracked.

## 3. Closure computation

`closure(rules)` (`validator.py:87-105`): build `direct: lhs → {rhs atoms}`; then a **fixpoint loop**: for each category `c`, for each reachable `("cat", p)` in `reach[c]`, union in `direct[p]`; repeat until no change. Result: transitive closure of cat→cat edges, with cat→adj and cat→not_cat atoms propagated along the chains. `not_cat` and `adj` atoms never chain (logically correct in this fragment). Monotone, terminates; no cycle/depth issues.

`derivable(pred, state, reach)` (`validator.py:107-114`): true iff `pred ∈ state` **or** some `("cat", c) ∈ state` with `pred ∈ reach[c]`. Only positive category atoms fire rules; a `not_cat`/`adj` atom in the state entails nothing further.

**Key consequence**: derivability is *k-hop for any k* — a continuation may jump from the premise straight to the target and still be "derivable" (see §7).

## 4. Closed-world assumption — yes, asymmetrically, and only in the *audit*, not the validator

- The **validator** never assigns truth values: an unentailed step is merely *invalid* (`validator.py:144`), not "false".
- The **truth-status audit** uses CWA for **positive category claims only**: `paper_artifacts.py:111` (`"true" if derivable(...) else "false"`), and explicitly labeled in `expa_global_expansion.py:206-210` (`audit_basis="unentailed_category_closed_world"`) and `expc_polarity_control.py:375-376`.
- **Negated** unentailed categories are **open-world "unknown"** (`expa_global_expansion.py:211-215`, basis `negated_unentailed_category_open_world`; `expc_polarity_control.py:377-378`). Attribute claims: true if entailed, false if the opposite polarity is entailed, else "unknown" (`expa_global_expansion.py:216-225`).
- CWA also drives **stimulus construction**: `make_falsehood` (`common.py:115-135`) calls a category "GUARANTEED false" iff not derivable from premises under closure, with an assert at `common.py:134`.

So: "false" for positive category injections = closed world; negations are never called false unless they negate an entailed atom. This asymmetry between perturbation families must be disclosed.

## 5. Injection-dependence ("poisoned") detection

`validator.py:125-128,140-142`: parse the corrupted step into `corrupted_atom` (None if unparseable). For each continuation fact not derivable from the TRUE state, retry with `state ∪ {corrupted_atom}`; if now derivable, the step is **poisoned** and — critically — **the poisoned atom is added to the state, so poison propagates** (`validator.py:142`). Merely *repeating* the injected lie is poisoned (pred equals corrupted_atom).

**Structural hole**: if the corrupted step is a *rule* (the distractor family injects `Every X is Y`), `parse_fact` returns None → `corrupted_atom = None` → **injection-dependence is silently undetectable for the entire distractor family** (`validator.py:126-128`). `paper_artifacts.py:178-185` papers over this with daggers/`---` in the falsifiability table. The corrupted rule is also never added to the rule set, so continuations that *use* the distractor rule land in `invalid` → parroted/derailed, never poisoned.

## 6. Unparsed continuations and denominators

Inside `validate_continuation` (`validator.py:148-149`): only sentences matching `^[A-Z]\w*\s+is\b` that fail all parses increment `n_unparsed`. **Any** `n_unparsed > 0` forces `class="unparsed"` with top precedence (`validator.py:155-156`).

**Denominator fact**: in `main()`, unparsed rows are counted in `counts[point]["n"]` and headline rates divide by that full `n` (`validator.py:204-206,213-216`) — so `valid_rederivation_rate = valid / (valid+poisoned+parroted+derailed+unparsed)`. Unparsed rows **deflate** all rates rather than being removed. **The module docstring (`validator.py:16-17`) claims unparsed instances are "excluded from headline stats" — the code contradicts this.** Same convention in expa (`expa_global_expansion.py:790-799`, `unparsed` and `generation_failed` both inside `n`) and in `audit.py:53-59`.

**Silent drops** (no counting at all): sentences that don't start `Capitalized + is` — doubt commentary ("Wait, that contradicts the rules."), questions, multi-clause reasoning — are skipped entirely for validity (`validator.py:146-149` falls through) and surface only via the doubt flag. Rule restatements are skipped as vacuously fine (`validator.py:146-147`), never checked against the actual rule set — a continuation could assert a *fabricated rule* without penalty as long as it never uses it in an entity fact.

## 7. Parroted vs. closure-valid

Precedence ladder (`validator.py:155-164`): `unparsed > derailed (final≠target) > poisoned > parroted > valid_rederivation`.
- **parroted** = final sentence normalizes to target (`_norm`: lowercase, strip non-alphanumerics, `validator.py:151-153`) **and** ≥1 intermediate entity fact was in `invalid` (underivable even with the corrupted atom).
- **valid_rederivation** = final = target, zero poisoned, zero invalid. Because `derivable` is closure-based (any-hop), **a continuation consisting solely of the target sentence is closure-valid** if the target is entailed — goal jumps and skipped derivations pass. This is exactly the professor's §4 objection. It is measured, not fixed, by `audit.py:135-171` ("A. GOAL-JUMP AUDIT"): counts `jump0` (zero intermediate entity facts) and `rederive_correct` (continuation restates the correct version of the corrupted step).
- Distinguishing feature of parroted: at least one *asserted* intermediate is bogus; a model that asserts nothing bogus but skips everything is still "valid".

## 8. After an invalid step

The loop **never stops** (`validator.py:134-149`). An invalid (non-poisoned) fact is appended to `invalid` and **not** added to the state (`validator.py:144`), so later steps depending on it are themselves invalid — errors cascade in classification but scoring continues to the final sentence. Poisoned facts *are* added to the state (`validator.py:142`), so downstream consequences of the planted error are also labeled poisoned rather than invalid. Classification is aggregated only at the end.

## 9. Doubt detection vs. marker stripping

- **Order**: `strip_marker` is applied per-sentence *for parsing only* (`validator.py:130-131`); the doubt regex runs on the **raw, unstripped full continuation text** (`validator.py:167`). So stripping cannot mask doubt tokens; conversely, doubt-marked sentences are typically *invisible to the validity check* (a sentence starting "However, Sam is..." fails the `^[A-Z]\w*\s+is\b` gate — the word after "However" is "Sam", not "is" — so it's silently skipped, not unparsed).
- **Detector** (`validator.py:22`, duplicated verbatim `doubt_judge.py:16`): case-insensitive word-boundary regex `\b(wait|however|but|actually|mistake|error|contradict|incorrect|cannot|inconsistent)\b`. Exact-word only: **"contradiction", "contradicts", "errors" do NOT match** (trailing `\b` after `contradict`/`error` fails); plain conjunction "but" is a false positive. One hit anywhere in the continuation sets `acknowledged=True` — it is an independent flag, never affecting the validity class.
- **Calibration**: `doubt_judge.py` runs Qwen2.5-32B-Instruct as a YES/NO judge on continuations (truncated to 1500 chars, `doubt_judge.py:47`) for **mid-position runs only**, cohort-filtered, and reports lexical-vs-judge agreement per family (`doubt_judge.py:41-60`). `audit.py:61-73,184-196` re-runs the lexical regex fresh on stored continuations and computes the contradiction-vs-wrong doubt contrast with a two-proportion z-test.

## 10. Edge cases, silent drops, asymmetries

1. **Cohort gating**: only instances whose *gold* rollout is itself `valid_rederivation` enter headline stats (`validator.py:177-187`); perturbed runs on other instances are dropped with only an aggregate counter (`validator.py:194-195,209-210`). `sanity_on_gold` (`validator.py:221-239`) exists to check the gold pass rate.
2. **Denominator mismatch across artifacts**: `metrics.py:35-39` computes behavioral "recovered" over **all** non-skip runs (no cohort filter), where `recovered` is the looser `solved(prefix+continuation)` final-answer check from `perturb.py:114-115`; validator rates use the gold-validated cohort. Two different populations and two different success criteria coexist.
3. **`recovered` counts poisoned wins**: `solved()` only checks the final sentence equals the target — a poisoned derivation that lands on the right target string counts as behaviorally "recovered".
4. **Distractor-family hole** (§5): rule-type injections can't be poisoned; injected rule never checked or usable.
5. **Family asymmetry in derivability of negations**: contradiction/negstep inject `not_cat` facts; continuations *re-asserting* negated facts are rarely derivable (negative rules are rare) → they skew toward parroted/derailed rather than poisoned, while wrong-category injections skew poisoned. Validation difficulty is not family-uniform.
6. **No contradiction detection**: state may hold `p` and `¬p`; a continuation asserting both an entailed fact and (later, via poison) its negation raises no flag beyond the poison label.
7. **Entity heuristic**: first word of first gold sentence (`validator.py:184`); a gold rollout opening with a rule restatement yields entity="Every" — such instances fail gold validation and exit via the cohort filter (silent, uncounted per-cause).
8. **Injection sites** (`perturb.py:48-61`): entity-subject sentences minus first (premise restatement) and last (conclusion); needs ≥3 intermediates or the instance is skipped (`too_few_intermediate_steps`); early/mid/late = first/middle/last intermediate. Skips also for `tokenization_not_prefix_stable` and `no_corruption_candidate` (`perturb.py:94,108-110`).
9. **Truth audit CWA asymmetry** (§4): positive-category injections auditable "false" under CWA; negations only "false" when they negate an entailed atom; attributes may be "unknown". expa *excludes* conditions whose injection audits `unknown/unparsed/unavailable` (`expa_global_expansion.py:638-661`) and requires all three global-falsehood positions to survive (`expa_global_expansion.py:631-634,676-678`) — a per-family eligibility filter that changes the instance mix across conditions.
10. **`generation_failed` retained in denominators** in expa (`expa_global_expansion.py:716-717,790`).
11. **Doubt regex misses inflections** ("contradiction"), fires on plain "but"; only lexical layer feeds `acknowledged` in headline tables; the LLM judge is a separate calibration table at mid-position only.
12. **Segmentation dependence**: everything (validation, sentence horizons in `functional_metrics.py`, boundary states) inherits the `". "` split; abbreviation-free PrOntoQA makes this safe in practice but it's an assumption, not a check.
13. **`functional_metrics.py` cohort** = any id present in `validated.jsonl` regardless of class (`functional_metrics.py:71`) — includes poisoned/derailed instances in ρ-func.
14. **Cross-model harness `val3.py`** swallows all exceptions into a text file (`val3.py:8-12,25`) — fine for ops, worth noting Llama/Mistral numbers came through this path.
15. **`strip_marker` upper-cases the first char** after stripping (`validator.py:28`) so `parse_fact`'s `[A-Z]` anchor still matches paraphrase-family injections ("Therefore, sam..." case) — a load-bearing cosmetic fix.

---

## Pipeline pseudocode (faithful)

```
# STAGE 1 — gold (collect_gold.py)
for instance in pilot.jsonl (until 200 solved):
    gen = greedy(chat_prompt(INSTR + 2-shot FEWSHOT + Q + "Prove: target"))   # common.py:17-48,63-66
    solved := last_sentence_norm(gen) == norm(target)                          # common.py:58-60
    if solved: save hidden states; write rollout row

# STAGE 2 — perturb (perturb.py)
for solved instance:
    steps = split_sentences(gold_gen)
    ents  = [i : steps[i] starts with entity]; inter = ents[1:-1]
    if len(inter) < 3: skip
    points = {early: inter[0], mid: inter[mid], late: inter[-1]}
    assert prefix-tokenization stability, else skip
    for point, si:
        corrupted = family_maker(...)                # wrong/distractor/contradiction/paraphrase/falsehood/negstep/neghop-k
        if None: skip
        continuation = greedy(prompt + steps[:si] + [corrupted])   # no feedback
        recovered = solved(prefix+continuation)      # loose behavioral flag
        write run row + hidden states

# STAGE 3 — validate (validator.py)
rules, facts = parse_world(question); reach = closure(rules)      # fixpoint, §3
cohort = {id : validate(gold_gen) == valid_rederivation}
for run in runs (skip non-cohort, count n_excluded):
    state = {premise atoms for entity} ∪ {atoms of correct prefix steps[:si]}
    corrupted_atom = parse_fact(strip_marker(corrupted))          # None if rule/unparseable
    for s in strip_marker(split(continuation)):
        if fact f := parse_fact(s, entity):
            if derivable(f, state, reach):            state += f
            elif corrupted_atom and derivable(f, state+{corrupted_atom}, reach):
                                                       poisoned += s; state += f   # poison propagates
            else:                                      invalid += s               # NOT added to state; keep going
        elif parse_rule(s): pass                       # never checked
        elif s matches "^[A-Z]\w+ is": n_unparsed += 1
        else: pass                                     # silently ignored (doubt prose etc.)
    final_ok = norm(last sentence) == norm(target)
    class = unparsed if n_unparsed>0
            else derailed if not final_ok
            else poisoned if poisoned
            else parroted if invalid
            else valid_rederivation
    acknowledged = DOUBT.search(RAW continuation)      # independent flag
    rates per (family, point): count(class)/n, n includes unparsed

# STAGE 4 — audit/metrics
audit.py : recompute rates + Wilson CIs from raw rows; re-run doubt regex; goal-jump audit;
           two-proportion z-tests; per-horizon effective n for ρ
metrics.py / functional_metrics.py : null-calibrated latent (cosine) and functional (JSD of
           next-token dists via verified lm_head) divergence ρ = D_pert/D_null, medians + bootstrap CIs
doubt_judge.py : Qwen2.5-32B YES/NO doubt judge at mid, agreement vs lexical
paper_artifacts.py / expa/expb/expc : truth-status audit (CWA for positive categories),
           local-refutability distance (expc shortest_rule_distance), LaTeX tables
```

---

## Disclosures a formal spec section MUST make (including the embarrassing ones)

1. **Full grammar**: the exact regex-defined fragment (unary Horn: one category antecedent; cat/not_cat/adj consequents), the `-es`-only singularizer, the missing-article → adjective misparse, sentence split on `". "`.
2. **Closure semantics**: fixpoint transitive closure; `derivable` is any-hop → **"closure-valid" permits derivation-free goal jumps**; report the goal-jump audit numbers (jump0 rate, rederive-correct rate from `audit.py`) alongside every closure-valid rate, and adopt the professor's vocabulary (closure-valid continuation ≠ repair).
3. **CWA asymmetry**: unentailed positive category = false (closed world) in the *audit and stimulus construction*; negated unentailed category = unknown (open world); attributes may be undecidable. The validator itself assigns no truth values — invalid ≠ false.
4. **Injection-dependence blind spot**: rule-type injections (distractor family) cannot be classified poisoned by construction; injected rules are never added to the rule set nor checked. State this instead of hiding it behind table daggers.
5. **Denominators**: unparsed (and, in expa, generation_failed) rows remain in `n` for all reported rates; fix or justify, and **fix the docstring at `validator.py:16-17`, which currently claims the opposite of what the code does**.
6. **Silent drops**: non-entity-subject prose (including all doubt/self-correction commentary) is invisible to validity scoring; rule restatements are accepted unchecked; sentences about other entities count as unparsed.
7. **Poison propagation and continue-on-invalid semantics**: poisoned atoms enter the state, invalid ones don't; scoring never halts; class precedence unparsed > derailed > poisoned > parroted > valid.
8. **Doubt detector**: exact 10-word lexical regex on raw text; misses "contradiction"/"contradicts"/"errors"; false-positives on conjunction "but"; independent of validity class; LLM-judge calibration exists only at mid-position — report agreement and both rates.
9. **Marker stripping**: applied before parsing, per sentence, leading markers only; doubt regex sees unstripped text; `However/But/Wait` sentences bypass parsing entirely.
10. **Two coexisting success metrics**: behavioral `recovered` (final-answer string match, all runs, counts poisoned wins) vs. validator classes (gold-validated cohort only). `metrics.py`/figure recovery rates and validator tables use different populations *and* different criteria.
11. **Cohort/eligibility filters**: gold-must-validate filter with only aggregate exclusion counts; expa's per-condition truth-audit eligibility filter changes instance composition across perturbation families; injection-site requirement (≥3 intermediate entity steps); tokenization-stability skips.
12. **Entity identification heuristic** (first word of first gold sentence) and its failure mode.
13. **No consistency checking** in the state (p and ¬p can coexist).
14. **Family-level validation asymmetries**: negation-family reassertions are rarely derivable (skew to parroted/derailed), category-family skew to poisoned — the classifier's error surface differs by family, which interacts directly with the paper's causal claim about perturbation families.
15. **Sanity harness**: `sanity_on_gold` gold pass rate should be reported as validator calibration; cross-model runs went through `val3.py`, which suppresses per-run exceptions.