# Professor feedback on latent-recovery workshop paper (verbatim, 2026-07-01)

Recommendation: I would target TMLR first, unless you can substantially strengthen the causal isolation and simplify the story for ICLR.

The paper has a strong empirical idea and a valuable protocol, but in its current form it is better matched to TMLR's tolerance for nuanced, artifact-heavy, carefully validated empirical work than to ICLR's preference for a crisp, high-impact main claim with clean evidence. If you go to ICLR, I think you need a stronger, cleaner version than the current draft. If you go to TMLR, the current contribution can be made credible through thorough exposition, appendices, validator detail, and careful claim calibration.

The paper's core contribution is not just "LLMs recover from some errors." The stronger contribution is:
"You introduce an audited perturb-and-validate protocol for testing whether language models absorb, reject, or route around planted errors in their own proof traces, and you show that local counterevidence strongly modulates this behavior." This is especially valuable because you avoid the common weakness in CoT-intervention work, i.e., treating arbitrary perturbations as "errors" without proving they are actually false or checking whether the continuation is logically valid.

However, for a strong venue, the question is not only whether the result is interesting. It is whether the paper establishes a clean, generalizable phenomenon with enough methodological rigor that reviewers believe the conclusion. Plausible for TMLR, risky for ICLR.

## Main Weaknesses for TMLR / ICLR

### 1: The causal variable is still not cleanly isolated

This is the biggest scientific weakness.

You claim the important factor is nearby evidence / local falsifiability. But the perturbation families differ along several axes:
Benign paraphrase is confounded by stylistic change, no falsity; True interruption is confounded by true but off-path; Distractor rule is confounded by rule-like, not entity-fact-like; Contradiction is confounded by explicit lexical contradiction; 1-hop falsehood is often negated; and Global falsehood is confounded by affirmative categorical claim, off-path, different usability.

So reviewers can reasonably ask: "Is the model responding to inferential distance, or to negation, lexical contradiction, grammatical type, relevance, category frequency, or derivational usefulness?"
Your distance-(k) experiment and local-certificate flip help, but in the current draft they are not presented with enough force or detail to fully neutralize this concern.

High risk for ICLR, but should work for TMLR if you frame claims carefully.

### 2: The paper is too sprawling

e.g., the current draft has (surprisingly for me to say) too many experiments. For TMLR, this breadth can be acceptable if the manuscript is long and well-organized. For ICLR, this is dangerous. ICLR reviewers may come away thinking: There is a lot here, but I am not sure what the main claim is or which evidence is decisive. ICLR papers usually need a cleaner "one big thing" story.

### 3: The validator is under-specified relative to its importance

For both venues, but especially TMLR, you need much more detail on the validator.
You need to define exactly: what the proof grammar allows; how facts, rules, categories, and negations are represented; how closure is computed; whether "unentailed" equals false under a closed-world assumption; how injection-dependence is detected; how unparsed continuations affect denominators; how parroted outputs are distinguished from closure-valid ones; what happens after an invalid step; whether discourse markers are stripped before or after doubt detection.

Right now the validator is a black box. That is not acceptable for TMLR or ICLR.

### 4: "Recovery" is not always recovery

Your closure-valid metric is useful, but it permits skipped derivations and goal jumps. So when you say the model "recovers," reviewers may object: "Did the model actually detect and repair the planted error, or did it merely avoid using it, skip to the target, or exploit the visible goal?" You partly address this, but the paper still uses "recovery" somewhat loosely.

For both venues, replace broad "recovery" language with more precise terms e.g., closure-valid continuation, explicit flagging, injection-dependence, routing around the error, semantic validity under closure. Use "repair" only when the output actually indicates correction or logically routes around the false step.

### 5: Generality is still limited

The core phenomenon is shown primarily in PrOntoQA. GSM8K and arithmetic are useful sanity checks, but they are not deeply integrated or equivalently validated. A skeptical ICLR reviewer may say: "This is an interesting artifact of a synthetic ontology grammar." TMLR reviewers may also raise this, but they may be more willing to accept a careful, bounded empirical study if the claims are narrow.

To strengthen for ICLR, you likely need one more serious task regime, e.g., Lean / miniF2F-style formal proof traces; theorem-proving traces; synthetic but richer logic; program-execution traces; controlled symbolic math with formal validation.

The main TMLR risk is that reviewers may judge the contribution as too narrow or too synthetic unless you make the methodological contribution very explicit.
Could also try for ICLR, but would need to rewrite much more aggressively and come up with a cleaner and clearer causal story.

## Proposed paper structure for TMLR

- Introduction: compounding-error assumption; recovery depends on local refutability; audited perturb-and-validate protocol.
- Protocol: PrOntoQA setup; original successful proof filtering; injection procedure; perturbation families; truth-status audit.
- Validator: formal closure definition; closure-valid; injection-dependent; parroted; derailed; unparsed; doubt detector.
- Main results: primary Qwen2.5-7B sweep; distance-(k) dose-response; local-certificate flip.
- Alternative explanations: target copying; polarity; task structure; scale/lineage.
- Limitations: synthetic domain; closure vs strict validation; perturbation confounds; Qwen-heavy model set; deterministic decoding.
- Appendices: prompt templates; validator pseudocode; all tables; all confidence intervals; filtering pipeline; examples; model details; regression details.
