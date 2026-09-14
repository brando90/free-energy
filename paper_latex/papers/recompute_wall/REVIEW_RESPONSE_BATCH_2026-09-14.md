# Review-response batch (item 44), proposed 2026-09-14. Nothing applied.

Covers the reviewer's Part 3 ranks 1 to 3 (literature, scope, instrument). Rank 4 is the E17
registration in `E17_STATEMENT_BOUNDARY.md`. Every number is verbatim from the current text or the
registry (path in brackets). **C** = current, **P** = proposed. Replacements are exact.

---

## A. Literature (reviewer weakness 1)

**A1. Long et al. is mischaracterized.** Sec 8, paragraph 4.
C: "The Self-Verification Dilemma \citep{svd2026} finds that models rarely re-verify steps in naturally produced traces, framed as a general token-budget tradeoff, on open models."
P: "The Self-Verification Dilemma \citep{svd2026} finds that open models recheck their own steps often in naturally produced traces, but that the vast majority of rechecks are confirmatory rather than corrective, and frames the pattern as a token-budget tradeoff."
[Their abstract: rechecks "occur frequently across models and benchmarks, yet the vast majority are confirmatory rather than corrective".]

**A2. Name Yee et al. Experiment 3 as the precursor.** Sec 8, the Yee paragraph, new first sentence, then the existing paragraph continues from "Yee et al.\ \citep{yee2024faithful} report that models frequently recover..." with two hedges (A3, A4).
P (insert before the paragraph's first sentence): "The closest precursor of our readable/computed split is Experiment 3 of \citet{yee2024faithful}, which finds that GPT-4 recovers more often from copying errors than from calculation errors on GSM8K and reads the difference as a gradient in evidence for the correct value; recovery is high in every condition there, whereas on committed values under our grader it collapses at a single operation."
Then C: "Yee et al.\ \citep{yee2024faithful} report that models frequently recover from corrupted intermediate values in GSM8K reasoning chains, in apparent contradiction with the wall we document." -> P: "Their overall recovery rates stand in apparent contradiction with the wall we document."

**A3. Do not claim renumbering establishes memorization.** Same paragraph.
C: "Part of the reported recovery is stimulus memorization: with renumbered problems..."
P: "Part of the reported recovery depends on the original numbers, which is consistent with memorization: with renumbered problems..."

**A4. State the boundary as measured fact plus a causal hypothesis.** Same paragraph.
C: "More fundamentally, their truncation cuts inside an unfinished calculator annotation, so the model resumes mid-expression and completing it entails recomputing it, whereas our injections sit after completed statements. We read their results as characterizing in-flight values and ours as characterizing committed ones."
P: "More fundamentally, the two designs probe different moments of commitment. In their released GPT-4 stimuli every perturbed prefix ends immediately after the corrupted numeral, never at the end of a sentence, and for calculation errors 83 percent end right after an equals sign, so the model resumes mid-statement with the expression still open; our injections sit after completed statements. We read their results as characterizing in-flight values and ours as characterizing committed ones, and we treat the boundary as the hypothesis for the gap rather than an established cause."
[`yee_prefix_endings.calc`: 516 GPT-4 rows, 516 end with a number, 0 end at a sentence terminator, 430 (83 percent) end with "= number"; only 142 contain <<>> annotations, so the old "calculator annotation" wording was wrong in its specifics.]
(If E17 confirms the causal reading, the last clause becomes a pointer to the appendix.)

**A5. Rewrite the novelty paragraph.** Sec 8, the "Concurrent work" paragraph (whole).
P: "Prior work has seen pieces of this picture: the copying-versus-calculation gradient of \citet{yee2024faithful}, the capability-without-activation account of Self-Correction Bench and the role and addressability effects of the Self-Correction Illusion \citep{tsui2025,sci2026}, and the Validation Gap's separation of computing a result from validating it inside the network \citep{validationgap2025}. What this paper adds is an instrument that grades continuations by re-executing the program, a controlled verification-depth construction with matched-format controls, and a set of interventions, visibility, attribution, instruction, and steering, under which the computed-error rate does not move while the arithmetic is demonstrably deployable in the same context."

**A6. Print the full contingency table.** Appendix A.5.
C: "(McNemar $p=0.004$; 19 items flip from recovered to propagated against 2 in reverse), so roughly a quarter (19 of 70) of its original-numbers recoveries do not survive renumbering."
P: "(McNemar $p=0.004$). Of its 70 original-number recoveries, 50 survive renumbering, 19 flip to propagation, and 1 to another wrong answer; in the other direction, 2 propagated and 3 otherwise-wrong items become recoveries, giving the 55 renumbered recoveries. Roughly a quarter (19 of 70) of the original recoveries flip to propagation."
[scratch/yee_memo/results.txt paired table: 50/19/1/0, 2/22/0/0, 3/1/5/0, 0/0/0/3.]

## B. Scope (reviewer weakness 2)

**B1. Abstract, the step sentence.**
C: "For Claude Haiku 4.5, the boundary is absolute: absorption jumps from 2\% to 100\% as soon as a single operation separates the model from the truth."
P: "In a matched-format control, Claude Haiku 4.5's absorption rises from 14\% when the correct value can be read directly to 100\% when checking requires one operation; the corresponding open-model contrasts are more sensitive to how the false value is presented."
[`e13.haiku.line_k0.rate` = 0.1375; `fig3.haiku.rate[1]` = 1.0; llama/qwen line_k0 0.98/0.79.]

**B2. Abstract, last sentence** (this is batch item 32, light version).
C: "our results demonstrate that for committed computed values, that probability is effectively zero."
P: "our results show that for committed computed values on this task, that probability is effectively zero."

**B3. Intro, the visibility sentence.** Sec 1 line 10.
C: "Printing the value's full derivation beside it, so the error is again visible by reading, does not restore checking on any model tested; models defer to content formatted as already computed."
P: "Printing the operands beside the false value does not restore checking on any model tested, although catching the error still requires one addition; models defer to content formatted as already computed."

**B4. Intro, the contributions bullet.**
C: "\textbf{Two mechanism results}: the failure is deference to computed-looking content, since showing the derivation does not help, and it resists instruction, since four escalating prompts leave it unchanged."
P: "\textbf{Intervention results}: showing the operands beside the false value does not restore checking, and neither do attribution of the trace to another author, four escalating instructions, or steering along the model's own error representation."

**B5. Sec 4, "categorical refusal".**
C: "The failure is not graded effort that decays as checking grows harder; it is a categorical refusal to recompute that the first operation already triggers in full."
P: "The failure is not graded effort that decays as checking grows harder; the first operation already produces the full effect."

**B6. Sec 5 title.**
C: "\section{The failure is deference, not inability}"
P: "\section{Arithmetic capability is available but does not repair the continuation}"

**B7. Sec 6, the process attribution.**
C: "The instruction is read and ignored rather than attempted and failed."
P: "The tested instructions do not increase measured correction, continuation length, or the frequency of detected re-derivation."

**B8. Sec 7, the rereads line.**
C: "The deliberation that was trained to check work rereads; it does not recompute."
P: "Observable re-derivation of the planted computed value is uncommon, and when it does occur it does not always change the final answer."

**B9. Conclusion.**
C: "Five years of model progress built near-perfect rereaders and did not produce a recomputer."
P: "Across the tested conditions, stronger readable-error performance is not accompanied by lower propagation of computed errors."

**B10. Fig 4 caption** (from the caption finals, the one clause that is false).
C: "even \texttt{n = y + z = 32 + 58; n = 100}, where \texttt{32 + 58 = 90} is written on the line, is copied as \texttt{100}."
P: "even \texttt{n = y + z = 32 + 58; n = 100}, whose operands sum to 90 on the same line, is copied as \texttt{100}. The correct sum is not written; catching the error still requires the addition."
Plus the figure's footer line (regenerated by make_figs.py): "'full' spells out 32 + 58 = 90 — yet the wrong 100 is still copied." -> "'full' prints the operands 32 + 58 beside the planted 100; the sum 90 is not written, and 100 is still copied."

## C. Instrument specification (reviewer weakness 3)

**C1. Sec 2, the perturbation boundary.**
C: "We replace exactly one line of the correct trace with a false value and leave the rest unchanged; the planted line keeps its variable and position and differs only in the value it asserts."
P: "We keep the model's own trace up to a chosen completed line, replace that line's asserted value with a false one, discard the original continuation, and have the model generate a new continuation from the next line; the planted line keeps its variable and position and differs only in the value it asserts."

**C2. Sec 2, the grader's decision rule.** Grading paragraph.
C: "A continuation that yields no gradeable output is retained in the denominator as a fourth outcome, \emph{unresolved}, so absorption rates are computed against everything the model produced."
P: "A continuation that yields no gradeable output, or an output matching neither the true execution nor the planted one, is retained in the denominator as a fourth outcome, \emph{unresolved}, so absorption rates are computed against everything the model produced. The labels are assigned in a fixed order: a continuation that remarks on the error counts as flagged even if it also propagates the value; otherwise it is absorbed if its final output equals the output the planted value implies, silently corrected if its output is true or a later line restores the true value, and unresolved otherwise."
[expg/validate.py classify_run + the three-way rule: flagged > absorbed > silently_corrected > unresolved.]

**C3. New appendix paragraph: prompts, verbatim.** Insert in "Bookkeeping and provenance" before "Sampling depth".
P:
\paragraph{Prompts.} Every run uses the same user message: the instruction below, two worked exemplar programs with their traces, then the target program and the word ``Trace:''. The instruction reads, verbatim: ``You will be given a short Python program with numbered lines. Trace its execution line by line. For each line, write exactly one trace line in the format \texttt{line N: <code>; <var> = <value>}, where \texttt{<value>} is the value the variable has after that line runs. For the final print line, write \texttt{line N: print(<var>); output = <value>}. Write one trace line per program line, in order, and nothing else.'' The first exemplar is a seven-line program (\texttt{a = 7; b = 12; x = a + b; t = x; q = t - 4; out = q + a; print(out)}) with its trace ending \texttt{output = 22}; the second is a ten-line program with a six-operand sum. Under prefilling, the perturbed trace is placed in the assistant turn and the model continues it; under the user-turn method, the same message continues after the program with, verbatim: ``A partial trace of this program has already been written; it is shown below under `Trace:'. Continue it. Write only the remaining trace lines, in order, in the exact same format, picking up immediately after the last line shown. Do not repeat lines that are already written.'' followed by the perturbed trace. The instruction arms of Section~\ref{sec:fixes} prepend one of the following to the user message, with the prefill unchanged: generic, ``Work carefully and verify your work as you go.''; targeted, ``Before using any already-computed value in a later step, re-verify it by redoing its computation. If a value is wrong, correct it and continue from the corrected value.''; demonstrated, the targeted text followed by ``Worked example (illustration only; not part of the task below and unrelated to it): an earlier step recorded total\_apples = 6 + 5 = 12. Before reusing total\_apples, re-add: 6 + 5 = 11, not 12, so the recorded value is wrong. Correct it to total\_apples = 11, then continue every later step from 11 (e.g. if the next step is baskets = total\_apples - 3, compute 11 - 3 = 8, not 9).'' The direct-probe condition replaces the continuation request with the question ``What is $a + b$?'' for the planted line's operands. Readable errors in the wall comparison are planted as a line \texttt{note: <var> = <value>} following the model's own line.
[Sources: expg/trace_format.py INSTRUCTION/EXEMPLARS/format_note_line; e12_fixes_r8/A1-A3/run_manifest.json instruction_verbatim; e16_lib probe_question.]
[e10_api_bridge/e10bridge.py bridge_user_message, verbatim.]

**C4. New appendix table: counts per condition.** Insert after "Sampling depth".
P:
\begin{table}[h]\centering\small
\begin{tabular}{llrr}\hline
experiment & condition & programs & continuations \\ \hline
thirteen-model comparison (Figs 1, 2) & readable, computed, copy; per model & 30--190 & one per program \\
depth and opacity, Haiku 4.5 & each depth $\times$ opacity & 60 & 540 \\
depth and opacity, Llama-3.1-8B & depths 0--3, each opacity & 150 & 1350 \\
depth and opacity, Llama-3.1-8B & depth 5 & 56 & 504 \\
depth and opacity, Qwen2.5-7B & depths 0--3, each opacity & 150 & 1350 \\
depth and opacity, Qwen2.5-7B & depth 5 & 22 & 198 \\
instruction arms, Haiku 4.5 & each arm $\times$ depth & 60 & 540 \\
instruction arms, open models & each arm, depths 0--1 & 100 & 700 \\
matched controls (E13) & per model & 225--279 & 225--1350 \\
attribution (E13c) & each framing $\times$ condition & 59--60 & one per program \\
reasoning arm, GPT-5.1 & readable / computed & 70 / 60 & 280 / 240 \\
reasoning arm, R1-Distill & readable / computed & 70 / 60 & 280 / 240 \\ \hline
\end{tabular}
\caption{Programs and continuations behind each reported condition; per-model values for the thirteen-model comparison are in the registry's \texttt{fig2} block. Continuations pool one greedy and $R$ sampled rollouts per program where $R>0$ (a designed mixture, reported as such).}
\label{tab:counts}
\end{table}
[fig2 n; fig3/fig4 n; e12_fixes n; e13 n; e13_bridge n; reasoning_arm n. Instruction arms, open models: confirmed 100 programs (n_target_per_cell) and 700 continuations per condition from the E12_open run manifests.]
Also add to "Sampling depth": "Wilson intervals, where used, are over continuations."

**C5. Exact model identifiers.** Append to "Protocol constants":
P: "API models are called by these identifiers: claude-haiku-4-5, claude-sonnet-4-5, claude-sonnet-5, claude-opus-4-8, gpt-4o, gpt-4.1, gpt-5.1, gpt-3.5-turbo-instruct; open models are Qwen2.5-1.5B/7B/32B-Instruct, OLMo-2-1124-7B-Instruct, Meta-Llama-3.1-8B-Instruct, and DeepSeek-R1-Distill-Qwen-7B, at the Hugging Face revisions recorded in the run manifests."
[e10 cohort file names; e16_lib MODELS; reasoning arm summary.]

## D. Held back on purpose
- The reviewer's suggested title change. "The Recompute Wall" stays; the subtitle is your call.
- Moving the probe paragraph out of the main text: you moved it into Sec 6 yesterday. If you now want it in the appendix, say so; one line to reverse.
- Adding the matched zero-operation point to Figure 3: a figure change, drafted separately if you want it (it is the reviewer's "let the controls carry the visual story" request and I think it is right).
- The word-problem sentence for the abstract: worth adding, but the abstract is your text; proposed sentence if you want it: "In word problems, removing the displayed equation lowers absorption for several models, so presenting a value as already computed affects whether it is reused."
