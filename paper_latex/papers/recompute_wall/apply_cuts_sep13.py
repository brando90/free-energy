#!/usr/bin/env python3
"""Item 41 (approved 2026-09-13): the 'genuine improvement' cuts. Paragraph-level deletions + one collapsed
intro paragraph + figure references folded into claim sentences. All-or-nothing, .bak_sep13d."""
import os, re, shutil, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); a = ap.parse_args()
def paras(s): return s.split("\n\n")
files = {}
def get(rel):
    if rel not in files: files[rel] = open(os.path.join(a.root, rel)).read()
    return files[rel]
def drop_para(rel, startswith):
    ps = paras(get(rel)); hits = [i for i, p in enumerate(ps) if p.strip().startswith(startswith)]
    if len(hits) != 1: sys.exit(f"ABORT {rel}: {len(hits)} paragraphs start with {startswith[:50]!r}")
    del ps[hits[0]]; files[rel] = "\n\n".join(ps)
def replace_once(rel, old, new):
    s = get(rel); pat = re.compile(r"\s+".join(re.escape(w) for w in old.split())); ms = list(pat.finditer(s))
    if len(ms) != 1: sys.exit(f"ABORT {rel}: {len(ms)} matches for {old[:60]!r}")
    m = ms[0]; files[rel] = s[:m.start()] + new + s[m.end():]
I, S, W, D, O = "sections/01_intro.tex", "sections/02_setup.tex", "sections/03_wall.tex", "sections/04_depth.tex", "sections/05_opacity.tex"
# 1. intro: replace paragraphs 4-6 with one
NEW_INTRO = ("We measure the unprompted behavior directly. A model solves a short program by writing its execution trace one line at a time. "
 "We replace a single line of its own correct trace with a false value, let it continue with no indication that anything changed, and grade the continuation by re-executing the program, so whether the model reused the false value is an arithmetic fact rather than a judgment (Section~\\ref{sec:setup}). "
 "One property of the planted error governs the outcome. A false value that contradicts a number written earlier in the trace is rejected by strong models. "
 "A false value that is itself a computed result is taken at face value and carried through the rest of the trace, by every model we test, and the boundary between the two is a single operation.")
ps = paras(get(I)); idx = [i for i, p in enumerate(ps) if p.strip().startswith(("We measure the unprompted behavior directly.", "This pattern holds across thirteen models", "The boundary between the two regimes is sharp."))]
if len(idx) != 3 or idx != list(range(idx[0], idx[0] + 3)): sys.exit(f"ABORT intro paragraphs not found consecutively: {idx}")
ps[idx[0]:idx[0] + 3] = [NEW_INTRO]; files[I] = "\n\n".join(ps)
# 2. setup: cut the walkthrough sentences
replace_once(S, "Figure~\\ref{fig:protocol} shows the protocol on one program. The model has computed \\texttt{u = 48} earlier in its own trace; a readable error inserts \\texttt{u = 84} on the next line, and the model ignores it and continues from 48. A computed error instead changes a later step from its correct \\texttt{j = 107} to \\texttt{j = 97}, with the inputs \\texttt{k = 70} and \\texttt{c = 37} standing two lines above; the model takes 97 and carries it through every remaining step. We run this protocol",
             "Figure~\\ref{fig:protocol} shows the protocol on one program. We run this protocol")
# 3./4. wall
drop_para(W, "The asymmetry is sharpest exactly where models are strongest.")
drop_para(W, "Figure~\\ref{fig:wall} gives the full per-model picture")
replace_once(W, "A computed error is absorbed in almost every continuation, by every model.", "A computed error is absorbed in almost every continuation, by every model (Figure~\\ref{fig:wall}).")
# 5. depth
drop_para(D, "Figure~\\ref{fig:depth} plots absorption against verification depth")
replace_once(D, "For Haiku 4.5 the result is a step function.", "For Haiku 4.5 the result is a step function (Figure~\\ref{fig:depth}).")
# 6. opacity
drop_para(O, "Figure~\\ref{fig:opacity} shows absorption under the bare, full, and partial conditions")
replace_once(O, "Showing the operands does not restore checking.", "Showing the operands does not restore checking (Figure~\\ref{fig:opacity}).")
for rel in files:
    if not files[rel].endswith("\n"): files[rel] += "\n"
for ref in ("fig:protocol", "fig:wall", "fig:depth", "fig:opacity"):
    n = sum(files.get(r, "").count(ref) for r in files)
    assert n >= 1, f"figure reference {ref} lost"
print("verified; files:", sorted(files))
if a.dry: sys.exit(0)
for rel in files: shutil.copy2(os.path.join(a.root, rel), os.path.join(a.root, rel) + ".bak_sep13d")
for rel, s in files.items(): open(os.path.join(a.root, rel), "w").write(s); print("wrote", rel)
