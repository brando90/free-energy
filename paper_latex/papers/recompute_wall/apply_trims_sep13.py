#!/usr/bin/env python3
"""Item 42 (approved 2026-09-13): seven paragraph trims T1-T7 from TRIMS_BATCH_2026-09-13.md, with
appendix additions for T3 and T7. All-or-nothing, .bak_sep13f backups."""
import os, re, shutil, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); a = ap.parse_args()
files = {}
def get(rel):
    if rel not in files: files[rel] = open(os.path.join(a.root, rel)).read()
    return files[rel]
def replace_para(rel, startswith, new, count=1):
    ps = get(rel).split("\n\n"); hits = [i for i, p in enumerate(ps) if p.strip().startswith(startswith)]
    if len(hits) != 1: sys.exit(f"ABORT {rel}: {len(hits)} paragraphs start with {startswith[:50]!r}")
    i = hits[0]; ps[i:i + count] = [new]; files[rel] = "\n\n".join(ps)
def replace_once(rel, old, new):
    s = get(rel); pat = re.compile(r"\s+".join(re.escape(w) for w in old.split())); ms = list(pat.finditer(s))
    if len(ms) != 1: sys.exit(f"ABORT {rel}: {len(ms)} matches for {old[:60]!r}")
    m = ms[0]; files[rel] = s[:m.start()] + new + s[m.end():]
O, F, R, T, X, A = "sections/05_opacity.tex", "sections/06_fixes.tex", "sections/07_reasoning.tex", "sections/08_related.tex", "sections/09_discussion.tex", "sections/A_appendix.tex"
# T1
replace_para(O, "The reading rules out the two innocent explanations.",
 "The reading rules out the two innocent explanations. The inputs are not inaccessible, since they sit on the same line as the result, and the arithmetic is not hard, since it is the single addition the model just declined to perform. A matched control sharpens what remains. When the line also prints the operands' true sum, so the correct value stands finished on the planted line itself, absorption falls to 0.57 for Haiku 4.5, 0.47 for Llama-3.1-8B, and 0.38 for Qwen2.5-7B. The models can use the truth when it appears as a completed result; they do not compute it from operands in the same position. Two further controls point the same way (Appendix~\\ref{sec:appendix}): tagging the planted value as ``(unverified)'' leaves absorption at 0.90--0.97 across three models, and asking the direct question, what is the sum of the planted line's operands, yields the correct value in essentially every reply on the same perturbed context. The arithmetic is deployable at the moment of reading; the continuation objective simply does not call it.")
# T2
replace_para(F, "Together with Section~\\ref{sec:opacity} this closes the two obvious repair paths.",
 "Together with Section~\\ref{sec:opacity} this closes the two obvious repair paths: the model reads past operands that refute the result beside them, and it declines to recompute when told, even when shown. These are prompting results. Trained fixes are a different lever, and Self-Correction Bench reports that fine-tuning on correction traces restores readable-error corrections \\citep{tsui2025}; we return to training-side interventions in the discussion.")
# T3 (two paragraphs -> one)
ps = get(R).split("\n\n"); i3 = [i for i, p in enumerate(ps) if p.strip().startswith("A reasoning model whose deliberation we can read")]; i4 = [i for i, p in enumerate(ps) if p.strip().startswith("The channel data also contains the most direct evidence")]
if len(i3) != 1 or len(i4) != 1 or i4[0] != i3[0] + 1: sys.exit(f"ABORT Sec 7 paragraphs: {i3} {i4}")
ps[i3[0]:i3[0] + 2] = ["A reasoning model whose deliberation we can read shows the same default with no breach at all. DeepSeek-R1-Distill-7B \\citep{deepseekr1} absorbs computed errors at 0.98 and 0.95 on the one- and five-operation errors when continuing directly, and at 0.93 on both with its thinking channel on. Reading the channel explains why: on readable errors it routinely re-derives the contradicted value and corrects it, while on computed errors it almost never revisits the planted value (no re-derivation in 96 percent of five-operation continuations, and 85 percent for GPT-5.1). Even when the true value does surface in the deliberation, the final answer still builds on the planted value in a fifth to a third of such cases (Appendix~\\ref{sec:appendix}). The deliberation that was trained to check work rereads; it does not recompute."]
files[R] = "\n\n".join(ps)
replace_once(A, "and five-operation conditions at the sample sizes recorded in the results registry.",
 "and five-operation conditions at the sample sizes recorded in the results registry. The 85 percent figure for GPT-5.1 is among continuations with a non-empty reasoning summary; it is 92 percent if empty summaries are counted as containing no re-derivation. When the true value surfaces in a five-operation deliberation, the final answer still builds on the planted value in 2 of 9 such continuations for R1-Distill and 7 of 20 for GPT-5.1; the counts are small but the direction is the same for both models.")
# T4
replace_para(T, "Yee et al.\\ \\citep{yee2024faithful} report that models frequently recover",
 "Yee et al.\\ \\citep{yee2024faithful} report that models frequently recover from corrupted intermediate values in GSM8K reasoning chains, in apparent contradiction with the wall we document. Two factors close the gap. Part of the reported recovery is stimulus memorization: with renumbered problems in the style of GSM-Symbolic \\citep{mirzadeh2025gsmsymbolic}, GPT-4o's recovery falls from 66\\% to 52\\% (Appendix~\\ref{app:renumber}). More fundamentally, their truncation cuts inside an unfinished calculator annotation, so the model resumes mid-expression and completing it entails recomputing it, whereas our injections sit after completed statements. We read their results as characterizing in-flight values and ours as characterizing committed ones.")
# T5
replace_para(T, "Trained verifiers are the deliberate foil for our result.",
 "Trained verifiers are the deliberate foil for our result: process reward models and trained critics readily learn to catch computation errors and bugs \\citep{cobbe2021verifiers,uesato2022process,lightman2023,wang2024mathshepherd,saunders2022selfcritique,mcaleese2024critics}, and models can be trained to revise their own output \\citep{welleck2023selfcorrect}. Our finding is that the generating model does not deploy this ability unprompted, at any scale we test, and that neither instruction nor demonstration turns it on at inference time.")
replace_once(X, "The same reading says what should fix it, and our protocol supplies the test. Dedicated step verifiers already learn to catch computation errors \\citep{cobbe2021verifiers,uesato2022process,lightman2023,wang2024mathshepherd,mcaleese2024critics}, so the checking behavior is learnable; what remains unmeasured is whether training installs the generating model's propensity to deploy it mid-trace.",
 "The same reading says what should fix it, and our protocol supplies the test. The checking behavior is learnable (Section~\\ref{sec:related}); what remains unmeasured is whether training installs the generating model's propensity to deploy it mid-trace.")
# T6
replace_para(X, "Four limitations bound the claims.",
 "Three limitations bound the claims beyond the caveats of Section~\\ref{sec:wall}. Our task is straight-line integer programs, chosen because ground truth is mechanical; computed content in freer forms, such as code with control flow, is untested beyond the word-problem check in Appendix~\\ref{app:nl}. Reasoning-channel models are tested under their own deliberation interfaces rather than the roster protocol. And our claim is behavioral: models may represent step correctness they do not act on, as our probes and prior work suggest, so the wall describes what models do, not what they encode.")
# T7
replace_para(X, "We also looked for the missing check inside one open model, and found it present but unused.",
 "We also looked for the missing check inside one open model, and found it present but unused. On Qwen2.5-7B, linear probes on minimal pairs of traces that differ only in the planted value are at chance through 25 of 28 layers and separate wrong from right values only in the last three (held-out AUC 0.80, 0.77, and 0.98), read at a position whose tokens are identical in both members of the pair. Steering along these directions in 150 conditions, spanning single layers, layer bands, both signs, and doses up to the direction's natural magnitude, never produced a check: absorption stayed between 0.78 and 0.95 against a 0.90 baseline wherever the model could still continue unperturbed traces, and explicit checking never exceeded 5 percent. The information exists, in a form the generator does not consult, and amplifying it does not make the generator consult it (Appendix~\\ref{sec:appendix}).")
# appendix paragraph for T7: insert before \paragraph{Reasoning-model protocol.}
s = get(A); anchor = "\\paragraph{Reasoning-model protocol.}"
if s.count(anchor) != 1: sys.exit("ABORT: reasoning-model protocol anchor")
files[A] = s.replace(anchor, "\\paragraph{Probes and steering (E16).} The probes of Section~\\ref{sec:discussion} read the residual stream at the first position after the planted line; at the planted value itself the linear probe reaches held-out AUC 1.00 at layer 27. The 150 steering conditions comprise 73 single-layer, 53 multi-layer, and 24 generation-position conditions, each at 40 greedy continuations plus 20 unperturbed controls. The only conditions that produced more checking language than baseline had also destroyed ordinary continuation (control validity 0.0), and the rare ``checked'' labels elsewhere proved on reading to be classifier noise. A fuller account of this representation is future work.\n\n" + anchor, 1)
for rel in files:
    if not files[rel].endswith("\n"): files[rel] += "\n"
print("verified 7 trims + 2 appendix additions; files:", sorted(files))
if a.dry: sys.exit(0)
for rel in files: shutil.copy2(os.path.join(a.root, rel), os.path.join(a.root, rel) + ".bak_sep13f")
for rel, s in files.items(): open(os.path.join(a.root, rel), "w").write(s); print("wrote", rel)
