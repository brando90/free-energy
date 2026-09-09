#!/usr/bin/env python3
"""Item 39 (approved 2026-09-09): remove 'family' (as error type), 'modal', 'ecological'. All-or-nothing, .bak_sep9c."""
import os, shutil, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); a = ap.parse_args()
E = [
 ("sections/02_setup.tex", "The planted families also match how models err naturally on this substrate.", "The two kinds of planted error also match how models err naturally on this substrate."),
 ("sections/02_setup.tex", "The computed errors we plant are therefore the modal error these models produce on their own; the readable family is the contrast case, not the ecological one.", "The computed errors we plant are therefore the kind of error these models most often make on their own. The readable error is the comparison case, not the natural one."),
 ("sections/02_setup.tex", "Finally, the contrast between families is internal to the protocol, since whatever is artificial about planting is present in both.", "Finally, the contrast between the two kinds of error is internal to the protocol, since whatever is artificial about planting is present in both."),
 ("sections/A_appendix.tex", "The modal fate of a model's own computed slip is silent carriage into the answer, matching", "The usual fate of a model's own computed slip is to be carried silently into the answer, matching"),
 ("main.tex", "Haiku 4.5 continuations from the same program family (seed 507786).", "Claude Haiku 4.5 continuations on programs generated from the same seed (507786)."),
]
files = {}
for rel, c, p in E:
    s = files.get(rel) or open(os.path.join(a.root, rel)).read(); n = s.count(c)
    if n != 1: sys.exit(f"ABORT {rel}: count {n}: {c[:60]}")
    files[rel] = s.replace(c, p, 1)
print("verified", len(E), "edits in", sorted(files))
if a.dry: sys.exit(0)
for rel in files: shutil.copy2(os.path.join(a.root, rel), os.path.join(a.root, rel) + ".bak_sep9c")
for rel, s in files.items(): open(os.path.join(a.root, rel), "w").write(s); print("wrote", rel)
