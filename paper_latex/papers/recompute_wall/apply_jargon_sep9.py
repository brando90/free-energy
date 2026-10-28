#!/usr/bin/env python3
"""Item 40: replace lab vocabulary (arm, rollout, world, gold, site, ladder, bridge, substrate,
presentation, plus appendix one-offs). Anchors match with flexible whitespace (appendix is hard-wrapped).
All-or-nothing, .bak_sep9d, post-check asserts none of the terms remain. --list prints the C/P table."""
import argparse, os, re, shutil, sys
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); ap.add_argument("--list", action="store_true"); a = ap.parse_args()
S, W, M, D, O, F, R, A, T, X = "sections/02_setup.tex", "sections/03_wall.tex", "main.tex", "sections/04_depth.tex", "sections/05_opacity.tex", "sections/06_fixes.tex", "sections/07_reasoning.tex", "sections/A_appendix.tex", "sections/08_related.tex", "sections/09_discussion.tex"
E = [
 # substrate
 (S, "as the reasoning substrate.", "as the reasoning task."),
 (S, "also match how models err naturally on this substrate.", "also match how models err naturally on this task."),
 (W, "The dissociation is not specific to the trace substrate.", "The dissociation is not specific to the program-trace task."),
 (X, "Training against planted errors is the untested lever on this substrate,", "Training against planted errors is the untested lever on this task,"),
 (X, "Our substrate is straight-line integer programs,", "Our task is straight-line integer programs,"),
 (A, "\\caption{Word-problem substrate: absorption", "\\caption{Word-problem task: absorption"),
 (A, "Each perturbed trace from the wall substrate's readable, one-operation, and copy conditions", "Each perturbed trace from the thirteen-model comparison's readable, one-operation, and copy conditions"),
 (A, "reproduced the substrate-gap bridge row, so the on-versus-off contrast is a pure reasoning effect.", "reproduced its user-turn row from the thirteen-model comparison, so the on-versus-off contrast is a pure reasoning effect."),
 (A, "The protocol constants are fixed across conditions within each substrate.", "The protocol constants are fixed across conditions within each task format."),
 (A, "The wall substrate behind Figures 1 and 2 uses 12-line programs with the planted site at line 6,", "The programs behind Figures 1 and 2 have 12 lines, with the planted line at line 6,"),
 # presentation / bridge
 (S, "so the presentation does not drive the result.", "so the method does not drive the result."),
 (W, "are measured through the user-turn presentation.", "are measured through the user-turn method."),
 (O, "shows absorption under bare, full, and partial presentation for the three models.", "shows absorption under the bare, full, and partial conditions for the three models."),
 (X, "measured through a user-turn presentation rather than prefilling,", "measured through the user-turn method rather than prefilling,"),
 (A, "measured under the user-turn presentation, where a preface", "measured under the user-turn method, where a preface"),
 (A, "already reported separately because it is presentation-sensitive", "already reported separately because it is method-sensitive"),
 (A, "in line with the copy-condition inflation under this presentation recorded there.", "in line with the copy-condition inflation under this method recorded there."),
 (A, "\\paragraph{Bridge versus prefill calibration.}", "\\paragraph{User-turn versus prefill calibration.}"),
 (A, "The bridge (user-turn) presentation and the prefill presentation were cross-calibrated on the models that support both.", "The user-turn method and the prefill method were cross-calibrated on the models that support both."),
 (A, "so the computed-side result does not depend on presentation.", "so the computed-side result does not depend on the method."),
 (A, "The pure-copy conditions inflate under the bridge method relative to prefill, so we report the copy conditions separately rather than pool the two presentations.", "The pure-copy conditions inflate under the user-turn method relative to prefill, so we report the copy conditions separately rather than pool the two methods."),
 (A, "showed the copy conditions measure presentation rather than recomputation.", "showed the copy conditions measure the method rather than recomputation."),
 # arms
 (D, "These arms, and the opacity and instruction arms that follow, run on", "These experiments, and the opacity and instruction experiments that follow, run on"),
 (R, "The counts are small but the direction is the same in both arms.", "The counts are small but the direction is the same for both models."),
 (R, "so they are reported as their own arm and not pooled into", "so they are reported separately and not pooled into"),
 (R, "In both arms the planted error sits in the committed trace,", "For both models the planted error sits in the committed trace,"),
 (A, "is absorbed at 0.98--1.00 in every arm on every model, and the readable error is caught in every arm", "is absorbed at 0.98--1.00 under every framing on every model, and the readable error is caught under every framing"),
 (A, "absorbs the copy condition at 0.88--0.97 in all arms,", "absorbs the copy condition at 0.88--0.97 under all framings,"),
 (A, "on its GSM8K random-perturbation arm with all quantities renumbered", "on its GSM8K random-perturbation condition with all quantities renumbered"),
 (A, "restarts the solution from scratch on all 106 items in both arms rather than", "restarts the solution from scratch on all 106 items, original and renumbered, rather than"),
 (A, "and the frontier arms are reported at that registered depth.", "and the Haiku 4.5 runs are reported at that registered depth."),
 (A, "The open-model depth arms run at $R = 8$ plus greedy and contribute 1350 rollouts", "The open-model depth runs use $R = 8$ plus greedy and contribute 1350 continuations"),
 (A, "run on the same worlds with the same gold gates and grader as the depth arms,", "run on the same programs with the same correctness gates and grader as the depth experiments,"),
 (A, "Conditions pool greedy and sampled rollouts as in the main arms;", "Conditions pool greedy and sampled continuations as in the main experiments;"),
 (A, "\\paragraph{Reasoning-arm protocol.}", "\\paragraph{Reasoning-model protocol.}"),
 (A, "come from two arms held apart from the roster.", "come from two runs kept separate from the thirteen-model roster."),
 (A, "The frontier arm runs GPT-5.1 through the OpenAI Responses API", "The GPT-5.1 run uses the OpenAI Responses API"),
 (A, "The open-weight arm runs DeepSeek-R1-Distill-7B, whose gold traces are generated through its own think channel rather than by raw completion, because raw-completion gold collapses on the five-operation errors the model can only solve with that channel engaged.", "The open-weight run uses DeepSeek-R1-Distill-7B, whose correct base traces are generated through its own think channel rather than by raw completion, because raw completion rarely solves the five-operation programs, which the model solves only with that channel engaged."),
 (A, "Both arms are evaluated on the readable, one-operation, and five-operation conditions at the sample sizes recorded in the reasoning-arm report.", "Both are evaluated on the readable, one-operation, and five-operation conditions at the sample sizes recorded in the results registry."),
 (A, "was not flat across the instruction arms but drifted", "was not flat across the instruction conditions but drifted"),
 (A, "In the depth and instruction arms (E11 and E12),", "In the depth and instruction experiments (E11 and E12),"),
 # rollouts
 (R, "no re-derivation appears in 96 percent of five-operation rollouts for R1-Distill and in 85 percent for GPT-5.1 (among rollouts with", "no re-derivation appears in 96 percent of five-operation continuations for R1-Distill and in 85 percent for GPT-5.1 (among continuations with"),
 (R, "2 of 9 such rollouts for R1-Distill,", "2 of 9 such continuations for R1-Distill,"),
 (R, "43 percent of its reasoning-on rollouts return an empty summary,", "43 percent of its reasoning-on continuations return an empty summary,"),
 (X, "Reinforcement learning on corrupted-prefix rollouts improves", "Reinforcement learning on corrupted-prefix samples improves"),
 (A, "The fix-ladder specification (E12) called for $R = 8$ sampled rollouts per world plus one greedy rollout,", "The instruction-experiment specification (E12) called for $R = 8$ sampled continuations per program plus one greedy continuation,"),
 (A, "so each depth-ladder condition contributes $n = 540$ rollouts over 60 programs, nine per program, and the fix ladder was likewise rerun", "so each depth condition contributes $n = 540$ continuations over 60 programs, nine per program, and the instruction experiment was likewise rerun"),
 (A, "The depth-five conditions are attrition-limited: 504 rollouts over 56 programs", "The depth-five conditions are limited by how few programs are solved: 504 continuations over 56 programs"),
 (A, "since rollouts of the same program are correlated;", "since continuations of the same program are correlated;"),
 (A, "on all 360 of 360 attempted rollouts,", "on all 360 of 360 attempted continuations,"),
 (A, "Sampling uses one greedy rollout at temperature 0 followed by $R$ sampled rollouts at temperature 0.7.", "Sampling uses one greedy continuation at temperature 0 followed by $R$ sampled continuations at temperature 0.7."),
 # worlds
 (R, "gives a within-model contrast on the same worlds.", "gives a within-model contrast on the same programs."),
 (A, "Worlds are generated deterministically from an integer seed;", "Programs are generated deterministically from an integer seed;"),
 # gold
 (S, "The gold-generation stage records every failed attempt", "The solving stage records every failed attempt"),
 (S, "Across the five open-weight gold runs, 1{,}775 traces contain a self-produced computed slip,", "Across the five open-weight solving runs, 1{,}775 traces contain a self-produced computed slip,"),
 (A, "and it states $V$ (the gold gate).", "and it states $V$ (the correctness gate)."),
 (A, "In more than 2{,}700 gold solutions,", "In more than 2{,}700 correct solutions,"),
 (A, "The gold-generation stage also yields natural errors:", "The solving stage also yields natural errors:"),
 (A, "Across the five open-weight gold runs, 1{,}775 traces contain one,", "Across the five open-weight solving runs, 1{,}775 traces contain one,"),
 # site
 (T, "our planted contradictions sit adjacent to the site, the easy case.", "our planted contradictions sit adjacent to the planted line, the easy case."),
 (A, "with the planted site at line 10 (fractional position 0.5)", "with the planted line at line 10 (fractional position 0.5)"),
 (A, "Site operands are two-digit integers drawn from 11 to 99,", "Operands of the planted line are two-digit integers drawn from 11 to 99,"),
 (A, "and site opacity in", "and the planted line's opacity in"),
 (M, "under three site opacities: bare", "under three ways of showing the planted line: bare"),
 # ladder
 (F, "stays at 0.99 across the ladder,", "stays at 0.99 across the four instruction conditions,"),
 (X, "The fix ladder shows the model reads the command", "The instruction experiment shows the model reads the command"),
 # one-offs
 (A, "and the audited near-miss bound caps possible label mis-assignment at 0.07.", "and an audit of values landing within a few units of a wrong label caps possible mislabeling at 0.07."),
 (A, "a solution enters the cohort only if", "a solution enters the sample only if"),
 (A, "so its readable condition is empty by honest starvation;", "so its readable condition is empty because no such solutions exist;"),
 (A, "The framing is not inert as an instrument:", "The framing does register where it can:"),
 (A, "against same-run readable anchors of 0.69, 0.67, and 0.03.", "against readable-error absorption of 0.69, 0.67, and 0.03 in the same runs."),
]
if a.list:
    for i, (f, c, p) in enumerate(E, 1): print(f"[{i}] {f}\n  C: {c}\n  P: {p}\n")
    sys.exit(0)
files = {}
def get(rel):
    if rel not in files: files[rel] = open(os.path.join(a.root, rel)).read()
    return files[rel]
for rel, c, p in E:
    s = get(rel); pat = re.compile(r"\s+".join(re.escape(w) for w in c.split()))
    ms = list(pat.finditer(s))
    if len(ms) != 1: sys.exit(f"ABORT {rel}: {len(ms)} matches for: {c[:70]}")
    m = ms[0]; files[rel] = s[:m.start()] + p + s[m.end():]
terms = re.compile(r"\b(arms?|rollouts?|worlds?|gold|sites?|ladder|bridge|substrates?|presentations?|anchors?|starvation|attrition|inert|near-miss|cohort)\b", re.I)
left = []
for rel in sorted(set(f for f, _, _ in E) | {"sections/" + x for x in os.listdir(os.path.join(a.root, "sections")) if x.endswith(".tex")} | {"main.tex"}):
    s = files.get(rel) or open(os.path.join(a.root, rel)).read()
    for m in terms.finditer(s):
        ctx = s[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")
        left.append((rel, m.group(0), ctx))
if left: sys.exit("ABORT: terms remain:\n" + "\n".join(f"  {r}: {t} | {c}" for r, t, c in left))
print(f"verified {len(E)} edits, no target terms remain, files: {sorted(files)}")
if a.dry: sys.exit(0)
for rel in files: shutil.copy2(os.path.join(a.root, rel), os.path.join(a.root, rel) + ".bak_sep9d")
for rel, s in files.items(): open(os.path.join(a.root, rel), "w").write(s); print("wrote", rel)
