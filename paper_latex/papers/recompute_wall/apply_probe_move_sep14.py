#!/usr/bin/env python3
"""Item 43 (approved 2026-09-14): move the probe/steering paragraph from Sec 9 to the end of Sec 6 with the
blunter wording; widen Sec 6 title; repoint the appendix cross-reference. All-or-nothing, .bak_sep14."""
import os, re, shutil, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); a = ap.parse_args()
F, X, A = "sections/06_fixes.tex", "sections/09_discussion.tex", "sections/A_appendix.tex"
files = {r: open(os.path.join(a.root, r)).read() for r in (F, X, A)}
NEW = ("Steering the model's own representation does not restore checking either. On Qwen2.5-7B, a linear probe on minimal pairs that differ only in the planted value can tell a wrong committed value from a right one, but only in the last three of 28 layers (held-out AUC 0.80, 0.77, and 0.98), at a position whose tokens are identical across the pair. Amplifying those directions in 150 steering conditions, across single layers, layer bands, both signs, and doses up to the direction's natural magnitude, changed nothing: absorption stayed between 0.78 and 0.95 against a 0.90 baseline wherever the model could still continue unperturbed traces, and explicit checking never exceeded 5 percent. The model represents that the value is wrong, late, and does not act on it (Appendix~\\ref{sec:appendix}).")
# 1. remove from discussion
ps = files[X].split("\n\n"); hits = [i for i, p in enumerate(ps) if p.strip().startswith("We also looked for the missing check inside one open model")]
if len(hits) != 1: sys.exit(f"ABORT discussion paragraph: {hits}")
del ps[hits[0]]; files[X] = "\n\n".join(ps)
# 2. append to Sec 6 + retitle
s = files[F].rstrip("\n") + "\n\n" + NEW + "\n"
old_t = "\\section{Instruction does not restore checking}"
if s.count(old_t) != 1: sys.exit("ABORT Sec 6 title")
s = s.replace(old_t, "\\section{Neither instruction nor steering restores checking}"); files[F] = s
# 3. appendix cross-ref
old_x = "The probes of Section~\\ref{sec:discussion} read the residual stream"
if files[A].count(old_x) != 1: sys.exit("ABORT appendix xref")
files[A] = files[A].replace(old_x, "The probes of Section~\\ref{sec:fixes} read the residual stream")
print("verified: paragraph moved, title widened, appendix xref repointed")
if a.dry: sys.exit(0)
for r in files: shutil.copy2(os.path.join(a.root, r), os.path.join(a.root, r) + ".bak_sep14")
for r, s in files.items(): open(os.path.join(a.root, r), "w").write(s); print("wrote", r)
