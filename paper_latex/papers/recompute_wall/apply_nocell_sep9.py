#!/usr/bin/env python3
"""Item 38: remove the internal word 'cell' from the paper. All-or-nothing anchored replace, .bak_sep9b
backups, refuses on missing/non-unique anchor; post-check asserts zero remaining 'cell(s)'.
Usage: apply_nocell_sep9.py [--dry] [--root DIR] [--list]"""
import argparse, os, re, shutil, sys
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); ap.add_argument("--list", action="store_true")
a = ap.parse_args()
E = [
 ("sections/00_abstract.tex", "dents only the deepest computed cells.", "dents only the deepest computed errors."),
 ("sections/02_setup.tex", "In the wall cells the readable error is realized as an annotation line beside the model's intact computation, while the computed error replaces the model's own line.",
  "In the thirteen-model comparison of Section~\\ref{sec:wall}, the readable error is written as a note beside the model's own line, which stays intact, while the computed error overwrites the model's own line."),
 ("sections/03_wall.tex", "On the one-operation computed cell, absorption is 1.00", "For a one-operation computed error, absorption is 1.00"),
 ("sections/03_wall.tex", "On the five-operation cell it is 0.95 or above for every adequately powered cell.", "For a five-operation computed error it is 0.95 or above in every adequately powered measurement."),
 ("sections/03_wall.tex", "solve the base task too rarely to fill the hardest cells.", "solve the base task too rarely to fill the hardest conditions."),
 ("sections/03_wall.tex", "Their deepest computed cells rest on small samples", "Their deepest computed conditions rest on small samples"),
 ("sections/03_wall.tex", "the computed-cell rates coincide", "the computed-error rates coincide"),
 ("sections/03_wall.tex", "It does inflate absorption on the pure-copy cell,", "It does inflate absorption on the pure-copy condition,"),
 ("sections/04_depth.tex", "though the depth-five cells rest on few solved programs", "though the depth-five measurements rest on few solved programs"),
 ("sections/05_opacity.tex", "leaves computed-cell absorption at 0.98--1.00", "leaves computed-error absorption at 0.98--1.00"),
 ("sections/07_reasoning.tex", "reaches the deepest computed cells, but", "reaches the deepest computed errors, but"),
 ("sections/07_reasoning.tex", "and both computed cells at 1.00.", "and both computed errors at 1.00."),
 ("sections/07_reasoning.tex", "The one-operation computed cell does not move: 0.988.", "The one-operation computed error does not move: 0.988."),
 ("sections/07_reasoning.tex", "the five-operation cell falls, from 1.00 to 0.875.", "the five-operation error falls, from 1.00 to 0.875."),
 ("sections/07_reasoning.tex", "five-operation cells when continuing directly,", "five-operation errors when continuing directly,"),
 ("sections/07_reasoning.tex", "96 percent of deep-cell rollouts for R1-Distill", "96 percent of five-operation rollouts for R1-Distill"),
 ("sections/07_reasoning.tex", "in the model's own reasoning on a deep cell,", "in the model's own reasoning on a five-operation error,"),
 ("sections/07_reasoning.tex", "Pooled over the three cells,", "Pooled over the three error conditions,"),
 ("sections/09_discussion.tex", "Computed-cell absorption is unchanged when the trace", "Absorption of computed errors is unchanged when the trace"),
 ("sections/09_discussion.tex", "when its computed-cell absorption leaves the ceiling,", "when its absorption of computed errors leaves the ceiling,"),
 ("main.tex", "The readable cell uses an annotation-line construction;", "The readable error is planted as a note beside the model's own line;"),
 ("main.tex", "Bands are program-cluster bootstrap 95\\% intervals (cluster = program); Wilson at degenerate ceiling cells.", "Bands are program-cluster bootstrap 95\\% intervals (cluster = program), with Wilson intervals where a condition sits at the ceiling."),
 ("main.tex", "The depth-five Qwen cell rests on 22 solved programs,", "The depth-five Qwen measurement rests on 22 solved programs,"),
 ("main.tex", "the depth-five Llama cell (56 programs) carries", "the depth-five Llama measurement (56 programs) carries"),
 ("main.tex", "Whiskers are program-cluster bootstrap 95\\% intervals (cluster = program); Wilson at degenerate ceiling cells.", "Whiskers are program-cluster bootstrap 95\\% intervals (cluster = program), with Wilson intervals where a condition sits at the ceiling."),
 ("sections/A_appendix.tex", "We re-ran the computed cell on two open-weight models", "We re-ran the computed-error condition on two open-weight models"),
 ("sections/A_appendix.tex", "1{,}350 rows per cell.", "1{,}350 rows per condition."),
 ("sections/A_appendix.tex", "in all twelve cells) and silent correction stays at or below 1.4\\% in every cell.", "in all twelve conditions) and silent correction stays at or below 1.4\\% in every condition."),
 ("sections/A_appendix.tex", "across the six cells.", "across the six conditions."),
 ("sections/A_appendix.tex", "the large-offset cells show lower raw absorption", "the large-offset conditions show lower raw absorption"),
 ("sections/A_appendix.tex", "let it continue, in four cells:", "let it continue, in four conditions:"),
 ("sections/A_appendix.tex", "so the bare cell replaces the model's arithmetic sentence", "so the bare condition replaces the model's arithmetic sentence"),
 ("sections/A_appendix.tex", "(the worked cell's rewrite sentence with the equation removed)", "(the worked condition's rewrite sentence with the equation removed)"),
 ("sections/A_appendix.tex", "absorption of planted errors by cell ($n=150$ per cell for API models", "absorption of planted errors by condition ($n=150$ per condition for API models"),
 ("sections/A_appendix.tex", "so its readable cell is empty by honest starvation", "so its readable condition is empty by honest starvation"),
 ("sections/A_appendix.tex", "readable, one-operation, and copy cells is presented", "readable, one-operation, and copy conditions is presented"),
 ("sections/A_appendix.tex", "60 programs per cell, 59 in Sonnet 5's one-operation cell", "60 programs per condition, 59 in Sonnet 5's one-operation condition"),
 ("sections/A_appendix.tex", "The one-operation computed cell is absorbed at 0.98--1.00 in every arm on every model, and the readable cell is caught", "The one-operation computed error is absorbed at 0.98--1.00 in every arm on every model, and the readable error is caught"),
 ("sections/A_appendix.tex", "the operation-free copy cell, already reported separately", "the operation-free copy condition, already reported separately"),
 ("sections/A_appendix.tex", "so each depth-ladder cell", "so each depth-ladder condition"),
 ("sections/A_appendix.tex", "they reproduce every cell within its confidence interval", "they reproduce every condition within its confidence interval"),
 ("sections/A_appendix.tex", "The depth-five cells are attrition-limited:", "The depth-five conditions are attrition-limited:"),
 ("sections/A_appendix.tex", "where the bootstrap degenerates at a ceiling cell we report Wilson intervals.", "where the bootstrap degenerates for a condition at the ceiling we report Wilson intervals."),
 ("sections/A_appendix.tex", "since each program contributes one continuation per cell.", "since each program contributes one continuation per condition."),
 ("sections/A_appendix.tex", "Cells pool greedy and sampled rollouts as in the main arms; per-cell records", "Conditions pool greedy and sampled rollouts as in the main arms; per-condition records"),
 ("sections/A_appendix.tex", "Two further E13 cells probe the same boundary.", "Two further E13 conditions probe the same boundary."),
 ("sections/A_appendix.tex", "A tentative-marker cell appends", "A tentative-marker condition appends"),
 ("sections/A_appendix.tex", "A direct-probe cell replaces the continuation request", "A direct-probe condition replaces the continuation request"),
 ("sections/A_appendix.tex", "raw-completion gold collapses on the deep cells the model", "raw-completion gold collapses on the five-operation errors the model"),
 ("sections/A_appendix.tex", "and five-operation cells at the sample sizes", "and five-operation conditions at the sample sizes"),
 ("sections/A_appendix.tex", "failed the solvability gate on the hardest cells.", "failed the solvability gate on the hardest conditions."),
 ("sections/A_appendix.tex", "their deepest computed cells\n", "their deepest computed conditions\n"),
 ("sections/A_appendix.tex", "We report those cells with widened Wilson intervals", "We report those conditions with widened Wilson intervals"),
 ("sections/A_appendix.tex", "The computed cells coincide\n", "The computed-error conditions coincide\n"),
 ("sections/A_appendix.tex", "The pure-copy cells inflate under the bridge method", "The pure-copy conditions inflate under the bridge method"),
 ("sections/A_appendix.tex", "so we report the copy cells separately rather than pool", "so we report the copy conditions separately rather than pool"),
 ("sections/A_appendix.tex", "as the one-operation cell alone, with the copy cells separate,", "as the one-operation error alone, with the copy conditions separate,"),
 ("sections/A_appendix.tex", "which pooled copy and computed cells into one recompute-side rate", "which pooled copy and computed conditions into one recompute-side rate"),
 ("sections/A_appendix.tex", "the copy cells measure presentation rather than recomputation.", "the copy conditions measure presentation rather than recomputation."),
 ("sections/A_appendix.tex", "The protocol constants are fixed across cells within each substrate.", "The protocol constants are fixed across conditions within each substrate."),
 ("sections/A_appendix.tex", "likewise fixed across its own cells.", "likewise fixed across its own conditions."),
 ("sections/A_appendix.tex", "GPT-4.1 absorbs the copy cell at", "GPT-4.1 absorbs the copy condition at"),
 ("sections/A_appendix.tex", "in line with the copy-cell inflation under this presentation", "in line with the copy-condition inflation under this presentation"),
]
if a.list:
    for i, (f, c, p) in enumerate(E, 1): print(f"[{i}] {f}\n  C: {c.strip()}\n  P: {p.strip()}\n")
    sys.exit(0)
files = {}
def get(rel):
    if rel not in files: files[rel] = open(os.path.join(a.root, rel)).read()
    return files[rel]
for rel, c, p in E:
    s = get(rel); n = s.count(c)
    if n != 1: sys.exit(f"ABORT {rel}: anchor count {n}: {c[:70]}")
    files[rel] = s.replace(c, p, 1)
left = []
for rel in sorted(set(f for f, _, _ in E) | {"sections/" + x for x in os.listdir(os.path.join(a.root, "sections")) if x.endswith(".tex")}):
    s = files.get(rel) or open(os.path.join(a.root, rel)).read()
    for m in re.finditer(r"[^.\n]*\bcells?\b[^.\n]*", s, re.I): left.append((rel, m.group(0)[:120]))
if left: sys.exit("ABORT: 'cell' remains: " + str(left))
print(f"verified {len(E)} edits, zero 'cell' remaining in {sorted(files)}")
if a.dry: print("dry run"); sys.exit(0)
for rel in files: shutil.copy2(os.path.join(a.root, rel), os.path.join(a.root, rel) + ".bak_sep9b")
for rel, s in files.items(): open(os.path.join(a.root, rel), "w").write(s); print("wrote", rel)
