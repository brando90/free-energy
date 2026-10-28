# Readability batch (item 37), proposed 2026-09-04. Nothing applied.

Scope: sentence-level only. No claim changes, no number changes. Every number below is verbatim
from the current text. Where a sentence I split also contains the phrase "the frontier model", the
replacement says "Haiku 4.5" (that is batch item 28; approving this batch approves it). Item 17 (the
two bibliography TODOs) is included at the end because it prints in the PDF.

Not in this batch, still pending your separate decision: the "chance is zero" scoping (item 32), the
"reliably solves" definition (item 31), captions.

How to read: **C** = current text, **P** = proposed. Replacements are exact LaTeX so the apply script
can anchor on them.

---

## Part 1. Abstract (319 words to 246)

**P (whole abstract):**

> When a language model continues a chain of reasoning that already contains a false step, it can
> catch the error or build on it. We measure which, with no instruction to check. A model solves a
> short program step by step, we replace one line of its own trace with a false value, and we let it
> continue. The continuation is graded by re-executing the program. One property of the error decides
> the outcome. A false value that contradicts a number written earlier in the trace is rejected by
> strong models. A false value that could only be exposed by redoing a computation is reused. Across
> thirteen models from five developers, computed falsehoods are absorbed in 88 to 100 percent of
> continuations by every model that reliably solves the base task, while the strongest models reject
> readable ones in at least 99 percent. Reading checks improve sharply with model quality;
> recomputation checks do not improve at all. For Claude Haiku 4.5 the boundary is a step: absorption
> jumps from 2 percent to 100 percent as soon as one operation separates the model from the truth.
> Showing the operands beside the false value, telling the model who wrote the trace, and four
> escalating instructions to verify all leave absorption at its ceiling. Test-time reasoning clears
> the readable side and dents only the deepest computed cells. Agent pipelines and chain-of-thought
> monitors assume errors have some chance of being caught downstream; for committed computed values,
> that chance is zero.

What changed: "capable models" -> "strong models" (undefined term, same meaning as Sec 3's usage);
dropped "97 to 100 percent for ten of the eleven" and "the two weakest models solve too few base
tasks" (both stay in Sec 3); "99 percent" -> "at least 99 percent" (item 27); "the frontier model" ->
"Claude Haiku 4.5"; "Showing the full derivation" -> "Showing the operands" (the full condition prints
operands, not the derivation's result; this is the A5 overclaim from the review); attribution added
to the null list since it is in the paper; "leave absorption at 100 percent" -> "at its ceiling"
(true for all three models; 100 percent was Haiku only, item 33); reasoning sentence shortened,
"passes at 99 percent" moved to Sec 7 where it already is. The last sentence is unchanged pending
item 32.

## Part 2. Section 7 rewrite (paragraphs 2 to 4; paragraph 1 and 5 keep their content, split below)

The current middle three paragraphs carry 27 numbers under three denominators. Proposed replacement
for the three paragraphs starting "Enabling reasoning on GPT-5.1..." through "...the model's own fresh
derivation of the contrary.":

> Enabling reasoning on GPT-5.1 gives a within-model contrast on the same worlds. With reasoning off,
> the model absorbs the readable error at 0.06 and both computed cells at 1.00. With reasoning on, the
> readable side goes to the floor at 0.004. The one-operation computed cell does not move: 0.988. Only
> the five-operation cell falls, from 1.00 to 0.875. The model spends its reasoning budget where the
> problem looks hard, about 360 reasoning tokens on five-operation errors against about 150 on
> one-operation errors, and absorption falls only where the budget goes. A value one operation from
> its inputs does not look difficult, so it is not checked.
>
> A reasoning model whose deliberation we can read shows the same default with no breach at all.
> DeepSeek-R1-Distill-7B \citep{deepseekr1} absorbs computed errors at 0.98 and 0.95 on the one- and
> five-operation cells when continuing directly, and at 0.93 on both with its thinking channel on.
> Reading the channel explains why. On readable errors it routinely re-derives the contradicted value
> and corrects it. On computed errors it almost never revisits the planted value: no re-derivation
> appears in 96 percent of deep-cell rollouts for R1-Distill and in 85 percent for GPT-5.1 (among
> rollouts with a non-empty reasoning summary; 92 percent if empty summaries are counted as
> containing none). The deliberation that was trained to check work rereads; it does not recompute.
>
> The channel data also contains the most direct evidence in the paper that the failure is deference.
> When the true value does surface in the model's own reasoning on a deep cell, the final answer
> still builds on the planted value in a fifth to a third of cases: 2 of 9 such rollouts for
> R1-Distill, 7 of 20 for GPT-5.1. The counts are small but the direction is the same in both arms.
> The model derives the truth, writes it in its deliberation, and continues from the value formatted
> as settled computation.

What changed: sentences shortened; "consistent with Section~\ref{sec:wall}, which gives 0.09 under
the chat presentation" dropped from the body (it is a cross-check, not a result; it can live in the
appendix reasoning-arm paragraph if you want it kept); "22 to 35 percent" -> "a fifth to a third"
(2/9 = 22 percent, 7/20 = 35 percent; the counts are given right after); "Whatever priority governs
the final channel, a value that looks already computed outranks the model's own fresh derivation of
the contrary" cut as a restatement.

## Part 3. Sentence splits, by section

### 01_intro.tex

[1] **C** The bet grows riskier as the values in context stop being the model's own: tool calls, retrieved documents, and other agents write computed results into the stream, and those sources can be wrong or even adversarial.
**P** The bet grows riskier as the values in context stop being the model's own. Tool calls, retrieved documents, and other agents write computed results into the stream, and those sources can be wrong or even adversarial.

[2] **C** The self-correction literature is more skeptical \citep{huang2024,stechly2024selfverification,tsui2025}, but it measures whether models can fix errors when asked to look for them, which is a different question from whether they catch errors they were never told were there.
**P** The self-correction literature is more skeptical \citep{huang2024,stechly2024selfverification,tsui2025}. But it measures whether models can fix errors when asked to look for them. Whether they catch errors they were never told about is a different question.

[3] **C** A model solves a short program by writing its execution trace one line at a time; we take its own correct trace, replace a single line with a false value, and let it continue with no indication that anything changed.
**P** A model solves a short program by writing its execution trace one line at a time. We take its own correct trace, replace a single line with a false value, and let it continue with no indication that anything changed.

[4] **C** This contrast is the paper's central observation: the ability to check a value one can reread improves sharply with scale and training, and the ability to check a value one must recompute does not improve at all.
**P** This contrast is the paper's central observation. The ability to check a value one can reread improves sharply with scale and training. The ability to check a value one must recompute does not improve at all.

[5] **C** Varying how many operations separate the planted value from a re-readable source, we find that for the frontier model absorption is a step function: 2 percent when the contradiction is directly readable, and 100 percent as soon as a single operation stands between the model and the truth.
**P** We vary how many operations separate the planted value from a re-readable source. For Claude Haiku 4.5 absorption is a step function: 2 percent when the contradiction is directly readable, and 100 percent as soon as a single operation stands between the model and the truth.

Also intro line 20 (contribution bullet): "for the frontier model, flips" -> "for Haiku 4.5, flips".

### 02_setup.tex

[6] **C** This section defines the protocol: how a model produces the reasoning we perturb, how we plant a false step, how we separate an error that can be re-read from one that must be recomputed, and how we score what the model does next.
**P** This section defines the protocol. It covers how a model produces the reasoning we perturb, how we plant a false step, how we separate an error that can be re-read from one that must be recomputed, and how we score what the model does next.

[7] **C** In the wall cells the readable error is realized as an annotation line beside the model's intact computation, while the computed error replaces the model's own line; matched controls that plant both by the identical splice show the frontier contrast is unchanged by this choice, while weaker models' readable catching is partly bound to the annotation format (Section~\ref{sec:depth}, Appendix~\ref{sec:appendix}).
**P** In the wall cells the readable error is realized as an annotation line beside the model's intact computation, while the computed error replaces the model's own line. Matched controls plant both by the identical splice. They show that the frontier contrast is unchanged by this choice, while weaker models' readable catching is partly bound to the annotation format (Section~\ref{sec:depth}, Appendix~\ref{sec:appendix}).

[8] **C** A synthetic task makes it easy to plant a value that is wrong by construction yet happens to equal another true value in the trace, which would make absorption ambiguous, so we reject any perturbation whose planted value coincides with a re-readable true value and verify by re-execution that the planted line is false under the program.
**P** A synthetic task makes it easy to plant a value that is wrong by construction yet happens to equal another true value in the trace, which would make absorption ambiguous. We therefore reject any perturbation whose planted value coincides with a re-readable true value, and we verify by re-execution that the planted line is false under the program.

[9] **C** The gold-generation stage records every failed attempt at executing a program, and the failures are overwhelmingly wrong computed values rather than mis-copies or malformed traces: 853 of 960 attempts for Qwen2.5-1.5B and 600 of 960 for OLMo-2-7B end in an incorrectly computed intermediate, while malformed traces are rare (five of 960 for the former).
**P** The gold-generation stage records every failed attempt at executing a program. The failures are overwhelmingly wrong computed values rather than mis-copies or malformed traces: 853 of 960 attempts for Qwen2.5-1.5B and 600 of 960 for OLMo-2-7B end in an incorrectly computed intermediate. Malformed traces are rare (five of 960 for the former).

[10] **C** These self-generated slips also share the fate of our plants: across the five open-weight gold runs, 1{,}775 traces contain a self-produced computed slip, and in 1{,}396 of them the slip changes the final answer if carried forward.
**P** These self-generated slips also share the fate of our plants. Across the five open-weight gold runs, 1{,}775 traces contain a self-produced computed slip, and in 1{,}396 of them the slip changes the final answer if carried forward.

[11] **C** Finally, the contrast between families is internal to the protocol: whatever is artificial about planting is present in both, and a matched control that plants the same false value by the same splice in a line with no operation to check drops the frontier model's absorption from 1.00 to 0.14 (Section~\ref{sec:depth}).
**P** Finally, the contrast between families is internal to the protocol, since whatever is artificial about planting is present in both. A matched control plants the same false value by the same splice in a line with no operation to check; it drops Haiku 4.5's absorption from 1.00 to 0.14 (Section~\ref{sec:depth}).

[12] **C** A continuation is \emph{absorbed} if its downstream values follow from the planted value, \emph{silently corrected} if they follow from the true value with no comment, and \emph{flagged} if the model explicitly notes the error, detected by a fixed lexical scan frozen before the runs.
**P** A continuation is \emph{absorbed} if its downstream values follow from the planted value, and \emph{silently corrected} if they follow from the true value with no comment. It is \emph{flagged} if the model explicitly notes the error, detected by a fixed lexical scan frozen before the runs.

### 03_wall.tex

[13] **C** On the one-operation computed cell, absorption is 1.00 for three of the four frontier Anthropic models (0.97 for Sonnet 4.5), all four OpenAI models, Llama-3.1-8B, and Qwen2.5-32B, and 0.88 for Qwen2.5-7B; on the five-operation cell it is 0.95 or above for every adequately powered cell.
**P** On the one-operation computed cell, absorption is 1.00 for three of the four frontier Anthropic models (0.97 for Sonnet 4.5), all four OpenAI models, Llama-3.1-8B, and Qwen2.5-32B, and 0.88 for Qwen2.5-7B. On the five-operation cell it is 0.95 or above for every adequately powered cell.

[14] **C** This is the paper's central observation stated as a contrast: the ability to check a value one can reread improves sharply across the model set, and the ability to check a value one must recompute does not improve at all.
**P** This is the paper's central observation stated as a contrast. The ability to check a value one can reread improves sharply across the model set. The ability to check a value one must recompute does not improve at all.

[15] **C** For the frontier Anthropic models, absorption of a readable error is at most 0.01 while absorption of a computed error is at least 0.97, so the same model that catches essentially every re-readable contradiction reuses essentially every computed one.
**P** For the frontier Anthropic models, absorption of a readable error is at most 0.01 while absorption of a computed error is at least 0.97. The same model that catches essentially every re-readable contradiction reuses essentially every computed one.

[16] **C** Two of the open-weight models, Qwen2.5-1.5B and OLMo-2-7B, solve the base task too rarely to fill the hardest cells, so their deepest computed cells rest on small samples and are reported with widened intervals rather than as settled points.
**P** Two of the open-weight models, Qwen2.5-1.5B and OLMo-2-7B, solve the base task too rarely to fill the hardest cells. Their deepest computed cells rest on small samples and are reported with widened intervals rather than as settled points.

[17] **C** The two newest Anthropic models do not permit prefilling and are measured through the user-turn presentation; on models that support both methods the computed-cell rates coincide, so this does not affect the computed-side result, though it inflates absorption on the pure-copy cell, which we therefore report separately.
**P** The two newest Anthropic models do not permit prefilling and are measured through the user-turn presentation. On models that support both methods the computed-cell rates coincide, so this does not affect the computed-side result. It does inflate absorption on the pure-copy cell, which we therefore report separately.

### 04_depth.tex

[18] **C** We hold the trace length and the number of tokens between the planted line and its nearest stated input fixed across depths, so depth measures the amount of computation a check requires and not the length of the trace or the distance the model must look back.
**P** We hold the trace length and the number of tokens between the planted line and its nearest stated input fixed across depths. Depth therefore measures the amount of computation a check requires, not the length of the trace or the distance the model must look back.

[19] **C** Part of Qwen's ramp is output failure rather than rejection: at depth one 21 percent of its continuations state no output and count against absorption, and among continuations that produce an output both open models absorb at 0.99 or above at depth one.
**P** Part of Qwen's ramp is output failure rather than rejection. At depth one, 21 percent of its continuations state no output and count against absorption. Among continuations that produce an output, both open models absorb at 0.99 or above at depth one.

[20] **C** Planting the false value in an ordinary trace line with zero operations to check, Llama absorbs 0.98 and Qwen 0.79, already at their depth-one levels, so their onset largely reflects the format of the planted line rather than the cost of checking it.
**P** When the false value is planted in an ordinary trace line with zero operations to check, Llama absorbs 0.98 and Qwen 0.79, already at their depth-one levels. Their onset therefore largely reflects the format of the planted line rather than the cost of checking it.

Also Sec 4 line 8 "For the frontier model the result is a step function." -> "For Haiku 4.5 the result is a step function."; line 10 "The frontier model absorbs the same matched plant at 0.14" -> "Haiku 4.5 absorbs the same matched plant at 0.14"; "the frontier model begins near zero" -> "Haiku 4.5 begins near zero". And the defining sentence (item 29) after "...and so on." in line 6: "These arms, and the opacity and instruction arms that follow, run on Claude Haiku 4.5, the frontier-developer model we could run at this volume with true prefilling, and on Llama-3.1-8B and Qwen2.5-7B." (Confirm the reason clause.)

### 05_opacity.tex

[21] **C** In a perturbed trace the printed operands are the true ones, so when the stated result is false the line contradicts itself: the operands sum to the true value while the result asserts the planted one, and catching the error requires only adding the numbers already on the line.
**P** In a perturbed trace the printed operands are the true ones, so when the stated result is false the line contradicts itself. The operands sum to the true value while the result asserts the planted one. Catching the error requires only adding the numbers already on the line.

[22] **C** And replacing the continuation request with the direct question---what is the sum of the planted line's operands?---yields the correct value in essentially every reply on the same perturbed context; the arithmetic is deployable at the moment of reading, and the continuation objective simply does not call it.
**P** And replacing the continuation request with the direct question, what is the sum of the planted line's operands, yields the correct value in essentially every reply on the same perturbed context. The arithmetic is deployable at the moment of reading; the continuation objective simply does not call it.

Also line 8 "For the frontier model absorption stays above 0.97" -> "For Haiku 4.5 absorption stays above 0.97"; line 10 "0.57 for the frontier model" -> "0.57 for Haiku 4.5".

### 06_fixes.tex

[23] **C** ...If a value is wrong, correct it and continue from the corrected value.'' The demonstrated condition adds the targeted text plus a worked example in which a wrong intermediate value is caught by recomputation, corrected, and the continuation proceeds from the corrected value.
**P** (second sentence only) The demonstrated condition adds the targeted text plus a worked example. In the example a wrong intermediate value is caught by recomputation and corrected, and the continuation proceeds from the corrected value.

[24] **C** The demonstrated condition is the strongest intervention prompting can offer, an explicit rule plus an in-context example of exactly the desired behavior, and it changes nothing: the model that has just been shown how to catch a planted computed error does not catch the next one.
**P** The demonstrated condition is the strongest intervention prompting can offer: an explicit rule plus an in-context example of exactly the desired behavior. It changes nothing. The model that has just been shown how to catch a planted computed error does not catch the next one.

[25] **C** For the frontier model, continuations are the same length in every condition, about 160 tokens, so the instruction does not even produce the surface form of checking, and a mechanical scan for re-derivation of the planted value fires in under a tenth of continuations regardless of condition.
**P** For Haiku 4.5, continuations are the same length in every condition, about 160 tokens, so the instruction does not even produce the surface form of checking. A mechanical scan for re-derivation of the planted value fires in about a tenth of continuations regardless of condition.
(This carries item 24: "under a tenth" -> "about a tenth", baseline 0.117.)

Also line 8 "The frontier model absorbs at 1.00 in all four conditions" -> "Haiku 4.5 absorbs at 1.00 in all four conditions".

### 07_reasoning.tex

[26] **C** This section tests that bet on two reasoning-channel models and finds the wall largely intact: deliberation completes the readable side and reaches the deepest computed cells, but a one-operation computed error still passes at 99 percent.
**P** This section tests that bet on two reasoning-channel models and finds the wall largely intact. Deliberation completes the readable side and reaches the deepest computed cells, but a one-operation computed error still passes at 99 percent.

[27], [28]: covered by the Part 2 rewrite.

[29] **C** These runs use each model's native deliberation interface rather than the roster's continuation protocols, so they are reported as their own arm and not pooled into Section~\ref{sec:wall}; GPT-5.1's reasoning is available only in summarized form, which makes its channel statistics a lower bound on re-derivation; 43 percent of its reasoning-on rollouts return an empty summary, and channel statistics condition on a non-empty summary except where stated.
**P** These runs use each model's native deliberation interface rather than the roster's continuation protocols, so they are reported as their own arm and not pooled into Section~\ref{sec:wall}. GPT-5.1's reasoning is available only in summarized form, which makes its channel statistics a lower bound on re-derivation. Pooled over the three cells, 43 percent of its reasoning-on rollouts return an empty summary, and channel statistics condition on a non-empty summary except where stated.
(Carries item 25.)

### 08_related.tex

[30] **C** Models asked to review their own answers rarely improve them \citep{huang2024}, struggle to locate errors even when told one exists \citep{tyen2024}, fail to improve plans by self-critique \citep{valmeekam2023selfcritique,stechly2024selfverification}, and remain blind to injected errors in their own generations even in benchmarks built to elicit correction \citep{tsui2025}.
**P** Models asked to review their own answers rarely improve them \citep{huang2024}, and they struggle to locate errors even when told one exists \citep{tyen2024}. They fail to improve plans by self-critique \citep{valmeekam2023selfcritique,stechly2024selfverification}, and they remain blind to injected errors in their own generations even in benchmarks built to elicit correction \citep{tsui2025}.

[31] **C** A related line perturbs reasoning to measure dependence rather than detection: corrupting a chain of thought to see whether the final answer changes tests whether the model uses its stated reasoning \citep{lanham2023}, and invalid reasoning in few-shot demonstrations barely hurts downstream accuracy \citep{wang2022,schaeffer2023invalid}.
**P** A related line perturbs reasoning to measure dependence rather than detection. Corrupting a chain of thought to see whether the final answer changes tests whether the model uses its stated reasoning \citep{lanham2023}, and invalid reasoning in few-shot demonstrations barely hurts downstream accuracy \citep{wang2022,schaeffer2023invalid}.

[32] **C** Our question is upstream of all of these: not whether a model can find an error when asked, nor whether its answer depends on its stated reasoning, but whether it notices an error it was never told about, in reasoning it believes is its own.
**P** Our question is upstream of all of these. It is not whether a model can find an error when asked, nor whether its answer depends on its stated reasoning. It is whether the model notices an error it was never told about, in reasoning it believes is its own.

[33] **C** Hallucinations snowball into confident elaboration \citep{zhang2023snowball}, single wrong steps compound in compositional tasks \citep{dziri2023} and in next-token training itself \citep{bachmann2024pitfalls}, and reasoning accuracy moves under surface perturbations of the problem: irrelevant context \citep{shi2023distracted}, premise order \citep{chen2024premise}, and entity or number substitutions \citep{mirzadeh2025gsmsymbolic}.
**P** Hallucinations snowball into confident elaboration \citep{zhang2023snowball}, and single wrong steps compound in compositional tasks \citep{dziri2023} and in next-token training itself \citep{bachmann2024pitfalls}. Reasoning accuracy also moves under surface perturbations of the problem: irrelevant context \citep{shi2023distracted}, premise order \citep{chen2024premise}, and entity or number substitutions \citep{mirzadeh2025gsmsymbolic}.

[34] **C** We plant audited errors rather than relying on natural ones, which fixes the ground truth; we separate readable from computed content, which locates the failure precisely; and we test frontier systems, which shows the readable side improving while the computed side does not.
**P** We plant audited errors rather than relying on natural ones, which fixes the ground truth. We separate readable from computed content, which locates the failure precisely. And we test frontier systems, which shows the readable side improving while the computed side does not.

[35] **C** Concurrent work reaches the deference conclusion by other routes: Self-Correction Bench attributes uncorrected errors to a capability that exists but is not activated, and the Self-Correction Illusion shows that role and addressability manipulations gate correction \citep{tsui2025,sci2026}.
**P** Concurrent work reaches the deference conclusion by other routes. Self-Correction Bench attributes uncorrected errors to a capability that exists but is not activated, and the Self-Correction Illusion shows that role and addressability manipulations gate correction \citep{tsui2025,sci2026}.

[36] **C** First, part of the reported recovery reflects stimulus memorization rather than verification: re-running their protocol with renumbered problems, in the style of GSM-Symbolic \citep{mirzadeh2025gsmsymbolic}, GPT-4o's recovery falls from 66\% to 52\% (paired $p=0.004$), with roughly a quarter of its original recoveries flipping to error propagation---on those items the model was restoring remembered numbers, not recomputing (Appendix~\ref{app:renumber}).
**P** First, part of the reported recovery reflects stimulus memorization rather than verification. Re-running their protocol with renumbered problems, in the style of GSM-Symbolic \citep{mirzadeh2025gsmsymbolic}, GPT-4o's recovery falls from 66\% to 52\% (paired $p=0.004$), with roughly a quarter of its original recoveries flipping to error propagation. On those items the model was restoring remembered numbers, not recomputing (Appendix~\ref{app:renumber}).

[37] **C** Probing studies find that models internally represent whether an arithmetic step is correct even while their behavior ignores it \citep{validationgap2025}, which matches what we observe behaviorally: the failure is not missing information but an unused check.
**P** Probing studies find that models internally represent whether an arithmetic step is correct even while their behavior ignores it \citep{validationgap2025}. This matches what we observe behaviorally: the failure is not missing information but an unused check.

[38] **C** Process reward models supervise individual reasoning steps and readily learn to catch computation errors \citep{cobbe2021verifiers,uesato2022process,lightman2023,wang2024mathshepherd}, trained critics catch bugs that reviewers miss \citep{saunders2022selfcritique,mcaleese2024critics}, and models can be trained to revise their own output \citep{welleck2023selfcorrect}, so the checking ability is plainly learnable.
**P** Process reward models supervise individual reasoning steps and readily learn to catch computation errors \citep{cobbe2021verifiers,uesato2022process,lightman2023,wang2024mathshepherd}. Trained critics catch bugs that reviewers miss \citep{saunders2022selfcritique,mcaleese2024critics}, and models can be trained to revise their own output \citep{welleck2023selfcorrect}. The checking ability is plainly learnable.

[39] **C** We use the same mechanism for measurement rather than attack: continuation from the model's own perturbed trace is what lets us observe the unprompted default, and error compounding in multi-step reasoning \citep{dziri2023} is why that default matters downstream.
**P** We use the same mechanism for measurement rather than attack. Continuation from the model's own perturbed trace is what lets us observe the unprompted default, and error compounding in multi-step reasoning \citep{dziri2023} is why that default matters downstream.

### 09_discussion.tex

[40] **C** Agent pipelines pass computed results from step to step \citep{yao2023react,kapoor2024agents}, and chain-of-thought monitors watch for signs of doubt \citep{baker2025monitoring,korbak2025monitorability}; our measurements say that a corrupted computed value crosses both defenses silently, since no model rechecks it and almost none remarks on it.
**P** Agent pipelines pass computed results from step to step \citep{yao2023react,kapoor2024agents}, and chain-of-thought monitors watch for signs of doubt \citep{baker2025monitoring,korbak2025monitorability}. Our measurements say that a corrupted computed value crosses both defenses silently: no model rechecks it and almost none remarks on it.

[41] **C** Nor is the trust self-trust that accurate provenance labels might break: absorption is unchanged when the trace is attributed to another model or to a human (Appendix~\ref{app:attribution}), so a wrong value entering from a tool call or a collaborating agent is carried exactly as the model's own would be.
**P** Nor is the trust self-trust that accurate provenance labels might break. Computed-cell absorption is unchanged when the trace is attributed to another model or to a human (Appendix~\ref{app:attribution}), so a wrong value entering from a tool call or a collaborating agent is carried exactly as the model's own would be.
(Carries item 26.)

[42] **C** The failure mode this predicts is specific: not noisy degradation but confident, arithmetically consistent propagation of a single wrong intermediate, invisible to any monitor that reads the model's own commentary; it is the within-trace analog of hallucinations snowballing into confident elaboration \citep{zhang2023snowball}.
**P** The failure mode this predicts is specific: not noisy degradation but confident, arithmetically consistent propagation of a single wrong intermediate, invisible to any monitor that reads the model's own commentary. It is the within-trace analog of hallucinations snowballing into confident elaboration \citep{zhang2023snowball}.

[43] **C** In ordinary text, a value written as the result of a computation is almost always correct, so trusting it is the right prediction on the training distribution; text rarely shows an author stopping mid-derivation to re-verify a line \citep{mccoy2024embers,prystawski2023why}.
**P** In ordinary text, a value written as the result of a computation is almost always correct, so trusting it is the right prediction on the training distribution. Text rarely shows an author stopping mid-derivation to re-verify a line \citep{mccoy2024embers,prystawski2023why}.

[44] **C** A prior installed by the distribution and untouched by the objective would look exactly like the flat ceiling we observe, including its indifference to instruction: the fix ladder shows the model reads the command to recompute and continues predicting text in which computed values are settled.
**P** A prior installed by the distribution and untouched by the objective would look exactly like the flat ceiling we observe, including its indifference to instruction. The fix ladder shows the model reads the command to recompute and continues predicting text in which computed values are settled.

[45] **C** On Qwen2.5-7B we trained linear probes on minimal pairs of traces that differ only in the planted value, reading the residual stream at the first position after the planted line, where the tokens are identical in both members of the pair.
**P** On Qwen2.5-7B we trained linear probes on minimal pairs of traces that differ only in the planted value. We read the residual stream at the first position after the planted line, where the tokens are identical in both members of the pair.

[46] **C** The probe is at chance through 25 of the model's 28 layers and separates wrong from right values only in the last three (held-out AUC 0.80, 0.77, and 0.98 at layers 26, 27, and 28); at the planted value itself it reaches 1.00 at layer 27.
**P** The probe is at chance through 25 of the model's 28 layers. It separates wrong from right values only in the last three (held-out AUC 0.80, 0.77, and 0.98 at layers 26, 27, and 28), and at the planted value itself it reaches 1.00 at layer 27.

[47] **C** Across 150 conditions spanning single layers, layer bands, and the late layers where the signal lives, both signs, and doses up to the direction's natural magnitude, absorption stayed between 0.78 and 0.95 against a 0.90 baseline in every condition that left the model able to continue unperturbed traces, and explicit checking never exceeded 5 percent of continuations; the only doses that produced more checking language had also destroyed ordinary continuation.
**P** We tried 150 conditions: single layers, layer bands, and the late layers where the signal lives, both signs, and doses up to the direction's natural magnitude. In every condition that left the model able to continue unperturbed traces, absorption stayed between 0.78 and 0.95 against a 0.90 baseline, and explicit checking never exceeded 5 percent of continuations. The only doses that produced more checking language had also destroyed ordinary continuation.

[48] **C** Training against planted errors is the untested lever on this substrate, though not untried elsewhere: reinforcement learning on corrupted-prefix rollouts improves outcome-level recovery \citep{rlflawed2025}, while supervised fine-tuning on correction traces can fail outright, with corrected models parroting the original mistake \citep{sfterr2025}.
**P** Training against planted errors is the untested lever on this substrate, though not untried elsewhere. Reinforcement learning on corrupted-prefix rollouts improves outcome-level recovery \citep{rlflawed2025}, while supervised fine-tuning on correction traces can fail outright, with corrected models parroting the original mistake \citep{sfterr2025}.

### 10_conclusion.tex

[49] **C** A value the model would have to recompute is kept: absorbed into the continuation at 88 to 100 percent by every model with enough solved programs to measure, unmoved by scale, by showing the operands beside the wrong result, or by four escalating instructions to check, and switched on, for the frontier model, by a single operation of verification depth.
**P** A value the model would have to recompute is kept. It is absorbed into the continuation at 88 to 100 percent by every model with enough solved programs to measure, unmoved by scale, by showing the operands beside the wrong result, or by four escalating instructions to check. For Haiku 4.5 it is switched on by a single operation of verification depth.

Also conclusion "from almost never for a 1.5-billion-parameter model" -> "from less than half the time for the small open-weight models" (item 20; "almost never" is false, catch rate 46 percent).

## Part 4. Bibliography (item 17)

`main.bib`: `note = {arXiv:2605.05737. TODO verify full author list}` -> `note = {arXiv:2605.05737}`;
`note = {arXiv:2606.25449. TODO verify full author list}` -> `note = {arXiv:2606.25449}`.

## Part 5. Left alone on purpose

- The appendix (79 sentences, 23 over 35 words, 7.7 numbers per 100 words). It is a provenance
  register and reviewers are told they need not read it; splitting it buys little for him.
- Semicolons inside sentences under 35 words (about 25 remain in the body). Harmless at that length.
- Intro line 10 "near zero for the smallest model" (item 19) and Sec 3 line 6 "by every model" (item
  21): these are claim corrections, not readability; they stay in batch 17-35 for your decision.

## Effect if applied

Body sentences over 35 words: 49 -> 0 outside the appendix. Body mean sentence length drops from
about 26 words to about 19. Abstract 319 -> 246 words. No number changes anywhere; three wording
items from batch 17-35 ride along and are marked where they do (24, 25, 26, 27, 28, 29, 33 partial).
