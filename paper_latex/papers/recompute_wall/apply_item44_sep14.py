#!/usr/bin/env python3
"""Item 44 (approved 2026-09-14): review-response batch A1-A6, B1-B10 (caption only; footer via registry), C1-C5.
All-or-nothing, .bak_sep14b backups. Whitespace-flexible anchors (appendix/main.tex are hard-wrapped)."""
import os, re, shutil, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); a = ap.parse_args()
files = {}
def get(rel):
    if rel not in files: files[rel] = open(os.path.join(a.root, rel)).read()
    return files[rel]
def rep(rel, old, new):
    s = get(rel); pat = re.compile(r"\s+".join(re.escape(w) for w in old.split())); ms = list(pat.finditer(s))
    if len(ms) != 1: sys.exit(f"ABORT {rel}: {len(ms)} matches for {old[:70]!r}")
    m = ms[0]; files[rel] = s[:m.start()] + new + s[m.end():]
def rep_para(rel, startswith, new):
    ps = get(rel).split("\n\n"); hits = [i for i, p in enumerate(ps) if p.strip().startswith(startswith)]
    if len(hits) != 1: sys.exit(f"ABORT {rel}: {len(hits)} paragraphs start with {startswith[:50]!r}")
    ps[hits[0]] = new; files[rel] = "\n\n".join(ps)
AB, I, S, D, O, F, R, T, X, C, A, M = ("sections/00_abstract.tex", "sections/01_intro.tex", "sections/02_setup.tex", "sections/04_depth.tex", "sections/05_opacity.tex", "sections/06_fixes.tex", "sections/07_reasoning.tex", "sections/08_related.tex", "sections/09_discussion.tex", "sections/10_conclusion.tex", "sections/A_appendix.tex", "main.tex")
# ---- A. literature
rep(T, "The Self-Verification Dilemma \\citep{svd2026} finds that models rarely re-verify steps in naturally produced traces, framed as a general token-budget tradeoff, on open models.",
       "The Self-Verification Dilemma \\citep{svd2026} finds that open models recheck their own steps often in naturally produced traces, but that the vast majority of rechecks are confirmatory rather than corrective, and frames the pattern as a token-budget tradeoff.")
rep(T, "Yee et al.\\ \\citep{yee2024faithful} report that models frequently recover from corrupted intermediate values in GSM8K reasoning chains, in apparent contradiction with the wall we document.",
       "The closest precursor of our readable/computed split is Experiment 3 of \\citet{yee2024faithful}, which finds that GPT-4 recovers more often from copying errors than from calculation errors on GSM8K and reads the difference as a gradient in evidence for the correct value; recovery is high in every condition there, whereas on committed values under our grader it collapses at a single operation. Their overall recovery rates stand in apparent contradiction with the wall we document.")
rep(T, "Part of the reported recovery is stimulus memorization: with renumbered problems", "Part of the reported recovery depends on the original numbers, which is consistent with memorization: with renumbered problems")
rep(T, "More fundamentally, their truncation cuts inside an unfinished calculator annotation, so the model resumes mid-expression and completing it entails recomputing it, whereas our injections sit after completed statements. We read their results as characterizing in-flight values and ours as characterizing committed ones.",
       "More fundamentally, the two designs probe different moments of commitment. In their released GPT-4 stimuli every perturbed prefix ends immediately after the corrupted numeral, never at the end of a sentence, and for calculation errors 83 percent end right after an equals sign, so the model resumes mid-statement with the expression still open; our injections sit after completed statements. We read their results as characterizing in-flight values and ours as characterizing committed ones, and we treat the boundary as the hypothesis for the gap rather than an established cause.")
rep_para(T, "Concurrent work reaches the deference conclusion by other routes.",
 "Prior work has seen pieces of this picture: the copying-versus-calculation gradient of \\citet{yee2024faithful}, the capability-without-activation account of Self-Correction Bench and the role and addressability effects of the Self-Correction Illusion \\citep{tsui2025,sci2026}, and the Validation Gap's separation of computing a result from validating it inside the network \\citep{validationgap2025}. What this paper adds is an instrument that grades continuations by re-executing the program, a controlled verification-depth construction with matched-format controls, and a set of interventions, visibility, attribution, instruction, and steering, under which the computed-error rate does not move while the arithmetic is demonstrably deployable in the same context.")
rep(A, "(McNemar $p=0.004$; 19 items flip from recovered to propagated against 2 in reverse), so roughly a quarter (19 of 70) of its original-numbers recoveries do not survive renumbering.",
       "(McNemar $p=0.004$). Of its 70 original-number recoveries, 50 survive renumbering, 19 flip to propagation, and 1 to another wrong answer; in the other direction, 2 propagated and 3 otherwise-wrong items become recoveries, giving the 55 renumbered recoveries. Roughly a quarter (19 of 70) of the original recoveries flip to propagation.")
# ---- B. scope
rep(AB, "For Claude Haiku 4.5, the boundary is absolute: absorption jumps from 2\\% to 100\\% as soon as a single operation separates the model from the truth.",
        "In a matched-format control, Claude Haiku 4.5's absorption rises from 14\\% when the correct value can be read directly to 100\\% when checking requires one operation; the corresponding open-model contrasts are more sensitive to how the false value is presented.")
rep(AB, "our results demonstrate that for committed computed values, that probability is effectively zero.", "our results show that for committed computed values on this task, that probability is effectively zero.")
rep(I, "Printing the value's full derivation beside it, so the error is again visible by reading, does not restore checking on any model tested; models defer to content formatted as already computed.",
       "Printing the operands beside the false value does not restore checking on any model tested, although catching the error still requires one addition; models defer to content formatted as already computed.")
rep(I, "\\textbf{Two mechanism results}: the failure is deference to computed-looking content, since showing the derivation does not help, and it resists instruction, since four escalating prompts leave it unchanged.",
       "\\textbf{Intervention results}: showing the operands beside the false value does not restore checking, and neither do attribution of the trace to another author, four escalating instructions, or steering along the model's own error representation.")
rep(D, "The failure is not graded effort that decays as checking grows harder; it is a categorical refusal to recompute that the first operation already triggers in full.",
       "The failure is not graded effort that decays as checking grows harder; the first operation already produces the full effect.")
rep(O, "\\section{The failure is deference, not inability}", "\\section{Arithmetic capability is available but does not repair the continuation}")
rep(F, "The instruction is read and ignored rather than attempted and failed.", "The tested instructions do not increase measured correction, continuation length, or the frequency of detected re-derivation.")
rep(R, "The deliberation that was trained to check work rereads; it does not recompute.", "Observable re-derivation of the planted computed value is uncommon, and when it does occur it does not always change the final answer.")
rep(C, "Five years of model progress built near-perfect rereaders and did not produce a recomputer.", "Across the tested conditions, stronger readable-error performance is not accompanied by lower propagation of computed errors.")
rep(M, "even \\texttt{n = y + z = 32 + 58; n = 100}, where \\texttt{32 + 58 = 90} is written on the line, is copied as \\texttt{100}.",
       "even \\texttt{n = y + z = 32 + 58; n = 100}, whose operands sum to 90 on the same line, is copied as \\texttt{100}. The correct sum is not written; catching the error still requires the addition.")
# ---- C. instrument
rep(S, "We replace exactly one line of the correct trace with a false value and leave the rest unchanged; the planted line keeps its variable and position and differs only in the value it asserts.",
       "We keep the model's own trace up to a chosen completed line, replace that line's asserted value with a false one, discard the original continuation, and have the model generate a new continuation from the next line; the planted line keeps its variable and position and differs only in the value it asserts.")
rep(S, "A continuation that yields no gradeable output is retained in the denominator as a fourth outcome, \\emph{unresolved}, so absorption rates are computed against everything the model produced.",
       "A continuation that yields no gradeable output, or an output matching neither the true execution nor the planted one, is retained in the denominator as a fourth outcome, \\emph{unresolved}, so absorption rates are computed against everything the model produced. The labels are assigned in a fixed order: a continuation that remarks on the error counts as flagged even if it also propagates the value; otherwise it is absorbed if its final output equals the output the planted value implies, silently corrected if its output is true or a later line restores the true value, and unresolved otherwise.")
PROMPTS = ("\\paragraph{Prompts.} Every run uses the same user message: the instruction below, two worked exemplar programs with their traces, then the target program and the word ``Trace:''. The instruction reads, verbatim: ``You will be given a short Python program with numbered lines. Trace its execution line by line. For each line, write exactly one trace line in the format \\texttt{line N: <code>; <var> = <value>}, where \\texttt{<value>} is the value the variable has after that line runs. For the final print line, write \\texttt{line N: print(<var>); output = <value>}. Write one trace line per program line, in order, and nothing else.'' The first exemplar is a seven-line program (\\texttt{a = 7; b = 12; x = a + b; t = x; q = t - 4; out = q + a; print(out)}) with its trace ending \\texttt{output = 22}; the second is a ten-line program with a six-operand sum. Under prefilling, the perturbed trace is placed in the assistant turn and the model continues it; under the user-turn method, the same message continues after the program with, verbatim: ``A partial trace of this program has already been written; it is shown below under `Trace:'. Continue it. Write only the remaining trace lines, in order, in the exact same format, picking up immediately after the last line shown. Do not repeat lines that are already written.'' followed by the perturbed trace. The instruction arms of Section~\\ref{sec:fixes} prepend one of the following to the user message, with the prefill unchanged: generic, ``Work carefully and verify your work as you go.''; targeted, ``Before using any already-computed value in a later step, re-verify it by redoing its computation. If a value is wrong, correct it and continue from the corrected value.''; demonstrated, the targeted text followed by ``Worked example (illustration only; not part of the task below and unrelated to it): an earlier step recorded total\\_apples = 6 + 5 = 12. Before reusing total\\_apples, re-add: 6 + 5 = 11, not 12, so the recorded value is wrong. Correct it to total\\_apples = 11, then continue every later step from 11 (e.g. if the next step is baskets = total\\_apples - 3, compute 11 - 3 = 8, not 9).'' The direct-probe condition replaces the continuation request with the question ``What is $a + b$?'' for the planted line's operands. Readable errors in the thirteen-model comparison are planted as a line \\texttt{note: <var> = <value>} following the model's own line.\n\n")
s = get(A); anc = "\\paragraph{Sampling depth.}"
if s.count(anc) != 1: sys.exit("ABORT Sampling depth anchor")
files[A] = s.replace(anc, PROMPTS + anc, 1)
TABLE = ("\\begin{table}[h]\\centering\\small\n\\begin{tabular}{llrr}\\hline\nexperiment & condition & programs & continuations \\\\ \\hline\n"
 "thirteen-model comparison (Figs 1, 2) & readable, computed, copy; per model & 30--190 & one per program \\\\\n"
 "depth and opacity, Haiku 4.5 & each depth $\\times$ opacity & 60 & 540 \\\\\n"
 "depth and opacity, Llama-3.1-8B & depths 0--3, each opacity & 150 & 1350 \\\\\n"
 "depth and opacity, Llama-3.1-8B & depth 5 & 56 & 504 \\\\\n"
 "depth and opacity, Qwen2.5-7B & depths 0--3, each opacity & 150 & 1350 \\\\\n"
 "depth and opacity, Qwen2.5-7B & depth 5 & 22 & 198 \\\\\n"
 "instruction arms, Haiku 4.5 & each arm $\\times$ depth & 60 & 540 \\\\\n"
 "instruction arms, open models & each arm, depths 0--1 & 100 & 700 \\\\\n"
 "matched controls (E13) & per model & 225--279 & 225--1350 \\\\\n"
 "attribution (E13c) & each framing $\\times$ condition & 59--60 & one per program \\\\\n"
 "reasoning arm, GPT-5.1 & readable / computed & 70 / 60 & 280 / 240 \\\\\n"
 "reasoning arm, R1-Distill & readable / computed & 70 / 60 & 280 / 240 \\\\ \\hline\n\\end{tabular}\n"
 "\\caption{Programs and continuations behind each reported condition; per-model values for the thirteen-model comparison are in the registry's \\texttt{fig2} block. Continuations pool one greedy and $R$ sampled rollouts per program where $R>0$ (a designed mixture, reported as such).}\n\\label{tab:counts}\n\\end{table}\n\n")
rep(A, "The wall figure uses Wilson intervals, since each program contributes one continuation per condition.",
       "The wall figure uses Wilson intervals, since each program contributes one continuation per condition. Wilson intervals, where used, are over continuations.")
s = get(A); anc2 = "\\paragraph{Matched controls (E13).}"
if s.count(anc2) != 1: sys.exit("ABORT matched-controls anchor")
files[A] = s.replace(anc2, TABLE + anc2, 1)
rep(A, "and the E10, E11, and E12 specifications.",
       "and the E10, E11, and E12 specifications. API models are called by these identifiers: claude-haiku-4-5, claude-sonnet-4-5, claude-sonnet-5, claude-opus-4-8, gpt-4o, gpt-4.1, gpt-5.1, gpt-3.5-turbo-instruct; open models are Qwen2.5-1.5B/7B/32B-Instruct, OLMo-2-1124-7B-Instruct, Meta-Llama-3.1-8B-Instruct, and DeepSeek-R1-Distill-Qwen-7B, at the Hugging Face revisions recorded in the run manifests.")
print("verified all edits; files:", sorted(files))
if a.dry: sys.exit(0)
for rel in files: shutil.copy2(os.path.join(a.root, rel), os.path.join(a.root, rel) + ".bak_sep14b")
for rel, s in files.items(): open(os.path.join(a.root, rel), "w").write(s); print("wrote", rel)
