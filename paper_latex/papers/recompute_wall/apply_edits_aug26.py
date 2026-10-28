#!/usr/bin/env python3
"""Apply the approved 2026-08-26 edit batch (items 1-13; 14 omitted by Elyas).

All-or-nothing anchored replace: every anchor must occur EXACTLY ONCE in its
file or nothing is written. Backups: <file>.bak_aug26. Bib additions appended;
tsui2025 gets a COLM 2026 note.
"""
import os
import shutil
import sys

P = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall"
S = os.path.join(P, "sections")

ATTR_SUBSECTION = r"""\subsection{Attribution does not move the wall}
\label{app:attribution}

Planting raises the objection that the model did not produce the false step itself, so absorption might reflect trust in its own past work rather than a general failure to check. We test this by varying who the trace is said to come from while holding its bytes fixed. Each perturbed trace from the wall substrate's readable, one-operation, and copy cells is presented under four framings: the standard continuation with no attribution, and prefaces stating that the model itself produced the trace, that a different language model produced it, or that a human wrote it by hand. The test runs on three models measured under the user-turn presentation, where a preface about the trace's origin is natural to add (Claude Opus 4.8, Claude Sonnet 5, GPT-4.1; 60 programs per cell, 59 in Sonnet 5's one-operation cell).

Attribution changes nothing at the wall. The one-operation computed cell is absorbed at 0.98--1.00 in every arm on every model, and the readable cell is caught in every arm (absorbed $\leq 0.02$). The framing is not inert as an instrument: the operation-free copy cell, already reported separately because it is presentation-sensitive (Appendix~\ref{sec:appendix}), moves with attribution for Opus 4.8 (0.27 unattributed, 0.13 self, 0.15 other model, 0.32 human), so where provenance can matter, the protocol registers it; GPT-4.1 absorbs the copy cell at 0.88--0.97 in all arms, in line with the copy-cell inflation under this presentation recorded there. On committed computed values, attribution is irrelevant: a model told outright that the trace is not its own work absorbs the computed falsehood at the same ceiling.

"""

EDITS = [
    # (file, old, new) --- item numbers in comments
    # 1: Five checks
    ("A_appendix.tex",
     "Four checks probe the wall outside the main protocol: whether error salience rescues checking, whether the dissociation transfers to natural-language word problems, how models treat computed errors they produced themselves, and a renumbered replication of the nearest prior protocol.",
     "Five checks probe the wall outside the main protocol: whether error salience rescues checking, whether the dissociation transfers to natural-language word problems, how models treat computed errors they produced themselves, whether telling the model who wrote the trace changes absorption, and a renumbered replication of the nearest prior protocol."),
    # 2: attribution subsection before the renumber subsection
    ("A_appendix.tex",
     "\\subsection{Renumbered replication of the Yee et al.\\ protocol}",
     ATTR_SUBSECTION + "\\subsection{Renumbered replication of the Yee et al.\\ protocol}"),
    # 8b: appendix matched-controls extension
    ("A_appendix.tex",
     "Cells pool greedy and sampled rollouts as in the main arms; per-cell records are under e13/results in the experiment archive and in the paper registry.",
     "Cells pool greedy and sampled rollouts as in the main arms; per-cell records are under e13/results in the experiment archive and in the paper registry. Two further E13 cells probe the same boundary. A tentative-marker cell appends ``(unverified)'' to the planted line's value: absorption is 0.92 for Haiku 4.5 ($n = 240$), 0.97 for Llama-3.1-8B ($n = 396$), and 0.90 for Qwen2.5-7B ($n = 324$). A direct-probe cell replaces the continuation request with the question ``What is $a + b$?'' for the planted line's operands: the correct sum appears in 100, 100, and 96 percent of replies respectively, and the planted value is echoed in at most 10 percent; replies typically open by restating the operands, so we report the answer-anywhere rate rather than a first-token match."),
    # 4: setup design-defense sentence
    ("02_setup.tex",
     "The continuation recovers the correct output in 10 of the 1{,}396 (0.7\\%), and an explicit correction remark appears in 20 (Appendix~\\ref{app:persistence}).",
     "The continuation recovers the correct output in 10 of the 1{,}396 (0.7\\%), and an explicit correction remark appears in 20 (Appendix~\\ref{app:persistence}). Finally, the contrast between families is internal to the protocol: whatever is artificial about planting is present in both, and a matched control that plants the same false value by the same splice in a line with no operation to check drops the frontier model's absorption from 1.00 to 0.14 (Section~\\ref{sec:depth})."),
    # 5: intro tool-call sentence
    ("01_intro.tex",
     "Both bet that a wrong step has some chance of being caught before it propagates.",
     "Both bet that a wrong step has some chance of being caught before it propagates. The bet grows riskier as the values in context stop being the model's own: tool calls, retrieved documents, and other agents write computed results into the stream, and those sources can be wrong or even adversarial."),
    # 8a + 3: opacity additions
    ("05_opacity.tex",
     "which is why the full condition raises absorption for the one model it moves at all.",
     "which is why the full condition raises absorption for the one model it moves at all. Two further controls sharpen the point. Tagging the planted value itself as ``(unverified)'' leaves absorption at 0.90--0.97 across three models, so a cue of doubt on the very line does not summon the check. And replacing the continuation request with the direct question---what is the sum of the planted line's operands?---yields the correct value in essentially every reply on the same perturbed context; the arithmetic is deployable at the moment of reading, and the continuation objective simply does not call it.\n\nThe deference is also not self-trust. Presenting byte-identical perturbed traces attributed to the model itself, to a different model, or to a human leaves computed-cell absorption at 0.98--1.00 on the three frontier models tested (Appendix~\\ref{app:attribution})."),
    # 10: fixes scope sentence
    ("06_fixes.tex",
     "whatever installs it during training, neither context nor command uninstalls it at inference.",
     "whatever installs it during training, neither context nor command uninstalls it at inference. These are prompting results: they show instruction does not summon the check at inference time on committed computed values. Trained fixes are a different lever---Self-Correction Bench reports that fine-tuning on correction traces restores its readable-error corrections \\citep{tsui2025}---and we return to training-side interventions in the discussion."),
    # 7: reasoning in-flight scope
    ("07_reasoning.tex",
     "stated reasoning can diverge from the computation that produces the answer \\citep{lanham2023,turpin2023unfaithful,arcuschin2025wild,chen2025unfaithful}.",
     "stated reasoning can diverge from the computation that produces the answer \\citep{lanham2023,turpin2023unfaithful,arcuschin2025wild,chen2025unfaithful}. In both arms the planted error sits in the committed trace, as in every experiment in this paper; the deliberation channel itself is never corrupted, only read. Corrupting values while they are still being written is the in-flight regime discussed in Section~\\ref{sec:related}."),
    # 9: related-work convergence paragraph after the SVD paragraph
    ("08_related.tex",
     "our result is the within-trace complement, where the same deference that is prudent across sessions becomes total within one.",
     "our result is the within-trace complement, where the same deference that is prudent across sessions becomes total within one.\n\nConcurrent work reaches the deference conclusion by other routes: Self-Correction Bench attributes uncorrected errors to a capability that exists but is not activated, and the Self-Correction Illusion shows that role and addressability manipulations gate correction \\citep{tsui2025,sci2026}. Our contribution to this converging picture is localization: the dissociation is specific to committed computed values, survives provenance and instruction manipulations, and persists when the capability is demonstrably deployable in the same context."),
    # 6: discussion provenance insert
    ("09_discussion.tex",
     "since no model rechecks it and almost none remarks on it.",
     "since no model rechecks it and almost none remarks on it. Nor is the trust self-trust that accurate provenance labels might break: absorption is unchanged when the trace is attributed to another model or to a human (Appendix~\\ref{app:attribution}), so a wrong value entering from a tool call or a collaborating agent is carried exactly as the model's own would be."),
    # 11: discussion training passage
    ("09_discussion.tex",
     "so the checking behavior is learnable; what has never been trained is the generating model's propensity to deploy it mid-trace. Training against planted computed errors, whether by supervised traces that model correction or by rewarding caught plants, is the direct route, and system-side re-execution",
     "so the checking behavior is learnable; what remains unmeasured is whether training installs the generating model's propensity to deploy it mid-trace. Training against planted errors is the untested lever on this substrate, though not untried elsewhere: reinforcement learning on corrupted-prefix rollouts improves outcome-level recovery \\citep{rlflawed2025}, while supervised fine-tuning on correction traces can fail outright, with corrected models parroting the original mistake \\citep{sfterr2025}. Neither has been evaluated against unprompted verification of committed values under an audited instrument, which is the experiment our protocol makes possible. System-side re-execution"),
    # 13: bridge count fix
    ("09_discussion.tex",
     "Two of our thirteen models are measured through a user-turn presentation rather than prefilling, calibrated but not identical.",
     "Five of our thirteen models are measured through a user-turn presentation rather than prefilling, and one through a legacy completion endpoint, calibrated but not identical."),
]

BIB_ADD = r"""
@misc{sci2026,
  title={The Self-Correction Illusion: Role Relabeling Gates Explicit Error Flagging in Large Language Models},
  author={Chen, Kuan-Yen and Su, Fang-Yi and Lin, Shih-Yen and Li, Bao and Chiang, Jung-Hsien},
  year={2026},
  eprint={2606.05976},
  archivePrefix={arXiv}
}

@misc{rlflawed2025,
  title={Can Large Reasoning Models Improve Accuracy on Mathematical Tasks Using Flawed Thinking?},
  author={Amjith, Saraswathy and Dusad, Mihika and Muramalla, Neha and Shah, Shweta},
  year={2025},
  eprint={2512.17079},
  archivePrefix={arXiv}
}

@misc{sfterr2025,
  title={Synthetic Error Injection Fails to Elicit Self-Correction In Language Models},
  author={Wu, David X. and Kapur, Shreyas and Sahai, Anant and Russell, Stuart},
  year={2025},
  eprint={2512.02389},
  archivePrefix={arXiv}
}
"""

# ---- verify phase ----
problems = []
texts = {}
for fn, old, new in EDITS:
    path = os.path.join(S, fn)
    if fn not in texts:
        texts[fn] = open(path).read()
    n = texts[fn].count(old)
    if n != 1:
        problems.append("%s: anchor occurs %d times: %r..." % (fn, n, old[:70]))
bib = open(os.path.join(P, "main.bib")).read()
for key in ("sci2026", "rlflawed2025", "sfterr2025"):
    if "@misc{%s," % key in bib:
        problems.append("main.bib already has %s" % key)
if bib.count("@misc{tsui2025,") != 1:
    problems.append("tsui2025 entry not found exactly once")
if problems:
    print("ABORT — nothing written:")
    for p in problems:
        print("  ", p)
    sys.exit(1)

# ---- apply phase (sequential per file so later anchors see earlier edits) ----
for fn, old, new in EDITS:
    texts[fn] = texts[fn].replace(old, new, 1)
for fn in set(f for f, _, _ in EDITS):
    path = os.path.join(S, fn)
    shutil.copy2(path, path + ".bak_aug26")
    open(path, "w").write(texts[fn])
    print("edited:", fn)
shutil.copy2(os.path.join(P, "main.bib"), os.path.join(P, "main.bib.bak_aug26"))
if "COLM" not in bib.split("@misc{tsui2025,")[1].split("}")[0:6][-1]:
    bib = bib.replace("@misc{tsui2025,", "@misc{tsui2025,\n  note={COLM 2026},", 1)
bib = bib.rstrip() + "\n" + BIB_ADD
open(os.path.join(P, "main.bib"), "w").write(bib)
print("edited: main.bib (+3 entries, tsui2025 COLM note)")
print("ALL EDITS APPLIED")
