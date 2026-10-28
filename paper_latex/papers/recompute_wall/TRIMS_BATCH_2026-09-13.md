# Text trims (item 42), proposed 2026-09-13. Nothing applied.

Seven paragraph-level trims. Numbers are verbatim from the current text; nothing dropped from the body
disappears from the paper, it moves to the appendix (T3, T7) or is already there (T1, T4). Estimated
saving about 450 words, roughly 0.9 page. **C** = current (opening words), **P** = proposed full text.

## T1. Section 5, paragraph 4 (243 -> about 160 words)

**C** "The reading rules out the two innocent explanations. ..." (whole paragraph)

**P**
> The reading rules out the two innocent explanations. The inputs are not inaccessible, since they sit on the same line as the result, and the arithmetic is not hard, since it is the single addition the model just declined to perform. A matched control sharpens what remains. When the line also prints the operands' true sum, so the correct value stands finished on the planted line itself, absorption falls to 0.57 for Haiku 4.5, 0.47 for Llama-3.1-8B, and 0.38 for Qwen2.5-7B. The models can use the truth when it appears as a completed result; they do not compute it from operands in the same position. Two further controls point the same way (Appendix~\ref{sec:appendix}): tagging the planted value as ``(unverified)'' leaves absorption at 0.90--0.97 across three models, and asking the direct question, what is the sum of the planted line's operands, yields the correct value in essentially every reply on the same perturbed context. The arithmetic is deployable at the moment of reading; the continuation objective simply does not call it.

Dropped: the two interpretive sentences ("What protects a false value is that checking it requires arithmetic..." and "The more the line looks like finished computation..."), which restate the paragraph's claim. Both extra controls stay named with their numbers, per the flag I raised.

## T2. Section 6, paragraph 5 (113 -> about 60 words)

**C** "Together with Section~\ref{sec:opacity} this closes the two obvious repair paths. ..." (whole paragraph)

**P**
> Together with Section~\ref{sec:opacity} this closes the two obvious repair paths: the model reads past operands that refute the result beside them, and it declines to recompute when told, even when shown. These are prompting results. Trained fixes are a different lever, and Self-Correction Bench reports that fine-tuning on correction traces restores readable-error corrections \citep{tsui2025}; we return to training-side interventions in the discussion.

## T3. Section 7, paragraphs 3 and 4 (211 -> about 110 words in the body; details to the appendix)

**C** the two paragraphs beginning "A reasoning model whose deliberation we can read..." and "The channel data also contains the most direct evidence..."

**P** (one paragraph replacing both)
> A reasoning model whose deliberation we can read shows the same default with no breach at all. DeepSeek-R1-Distill-7B \citep{deepseekr1} absorbs computed errors at 0.98 and 0.95 on the one- and five-operation errors when continuing directly, and at 0.93 on both with its thinking channel on. Reading the channel explains why: on readable errors it routinely re-derives the contradicted value and corrects it, while on computed errors it almost never revisits the planted value (no re-derivation in 96 percent of five-operation continuations, and 85 percent for GPT-5.1). Even when the true value does surface in the deliberation, the final answer still builds on the planted value in a fifth to a third of such cases (Appendix~\ref{sec:appendix}). The deliberation that was trained to check work rereads; it does not recompute.

**Appendix addition**, appended to the "Reasoning-model protocol" paragraph:
> The 85 percent figure for GPT-5.1 is among continuations with a non-empty reasoning summary; it is 92 percent if empty summaries are counted as containing no re-derivation. When the true value surfaces in a five-operation deliberation, the final answer still builds on the planted value in 2 of 9 such continuations for R1-Distill and 7 of 20 for GPT-5.1; the counts are small but the direction is the same for both models.

## T4. Related work, the Yee paragraph (170 -> about 105 words)

**C** "Yee et al.\ \citep{yee2024faithful} report that models frequently recover..." (whole paragraph)

**P**
> Yee et al.\ \citep{yee2024faithful} report that models frequently recover from corrupted intermediate values in GSM8K reasoning chains, in apparent contradiction with the wall we document. Two factors close the gap. Part of the reported recovery is stimulus memorization: with renumbered problems in the style of GSM-Symbolic \citep{mirzadeh2025gsmsymbolic}, GPT-4o's recovery falls from 66\% to 52\% (Appendix~\ref{app:renumber}). More fundamentally, their truncation cuts inside an unfinished calculator annotation, so the model resumes mid-expression and completing it entails recomputing it, whereas our injections sit after completed statements. We read their results as characterizing in-flight values and ours as characterizing committed ones.

Dropped from the body: the paired $p$ value and "roughly a quarter flipping to propagation" (both in Appendix A.5), and the sentence about their propagated condition.

## T5. The trained-verifiers point, said twice (related work paragraph 8 and the discussion)

Related work, **C** "Trained verifiers are the deliberate foil for our result. ..." (100 words) -> **P**
> Trained verifiers are the deliberate foil for our result: process reward models and trained critics readily learn to catch computation errors and bugs \citep{cobbe2021verifiers,uesato2022process,lightman2023,wang2024mathshepherd,saunders2022selfcritique,mcaleese2024critics}, and models can be trained to revise their own output \citep{welleck2023selfcorrect}. Our finding is that the generating model does not deploy this ability unprompted, at any scale we test, and that neither instruction nor demonstration turns it on at inference time.

Discussion, **C** "The same reading says what should fix it, and our protocol supplies the test. Dedicated step verifiers already learn to catch computation errors \citep{...}, so the checking behavior is learnable; what remains unmeasured is whether training installs the generating model's propensity to deploy it mid-trace." -> **P**
> The same reading says what should fix it, and our protocol supplies the test. The checking behavior is learnable (Section~\ref{sec:related}); what remains unmeasured is whether training installs the generating model's propensity to deploy it mid-trace.

(The rest of that discussion paragraph is unchanged.)

## T6. Discussion, limitations paragraph (121 -> about 85 words)

**C** "Four limitations bound the claims. ..." (whole paragraph)

**P**
> Three limitations bound the claims beyond the caveats of Section~\ref{sec:wall}. Our task is straight-line integer programs, chosen because ground truth is mechanical; computed content in freer forms, such as code with control flow, is untested beyond the word-problem check in Appendix~\ref{app:nl}. Reasoning-channel models are tested under their own deliberation interfaces rather than the roster protocol. And our claim is behavioral: models may represent step correctness they do not act on, as our probes and prior work suggest, so the wall describes what models do, not what they encode.

Dropped: the Fable 5 refusal and the user-turn/completion-endpoint caveats, both already stated in Section 3's caveat paragraph.

## T7. Discussion, the probe paragraph (171 -> about 120 words; full version to the appendix)

**C** "We also looked for the missing check inside one open model, and found it present but unused. ..." (whole paragraph)

**P**
> We also looked for the missing check inside one open model, and found it present but unused. On Qwen2.5-7B, linear probes on minimal pairs of traces that differ only in the planted value are at chance through 25 of 28 layers and separate wrong from right values only in the last three (held-out AUC 0.80, 0.77, and 0.98), read at a position whose tokens are identical in both members of the pair. Steering along these directions in 150 conditions, spanning single layers, layer bands, both signs, and doses up to the direction's natural magnitude, never produced a check: absorption stayed between 0.78 and 0.95 against a 0.90 baseline wherever the model could still continue unperturbed traces, and explicit checking never exceeded 5 percent. The information exists, in a form the generator does not consult, and amplifying it does not make the generator consult it (Appendix~\ref{sec:appendix}).

**Appendix addition**, new paragraph in "Bookkeeping and provenance" after "Matched controls (E13)":
> \paragraph{Probes and steering (E16).} The probes of Section~\ref{sec:discussion} read the residual stream at the first position after the planted line; at the planted value itself the linear probe reaches held-out AUC 1.00 at layer 27. The 150 steering conditions comprise 73 single-layer, 53 multi-layer, and 24 generation-position conditions, each at 40 greedy continuations plus 20 unperturbed controls. The only conditions that produced more checking language than baseline had also destroyed ordinary continuation (control validity 0.0), and the rare ``checked'' labels elsewhere proved on reading to be classifier noise. A fuller account of this representation is future work.

## What this does not touch
Section 2 (the instrument), Section 4, the first half of related work, the conclusion, and the abstract.
