#!/usr/bin/env python3
"""Approved 2026-09-04: items 15-16 (note-construction scoping) + item 36 (probe/steering paragraph).
All-or-nothing anchored replace; .bak_sep4 backups; refuses on missing/non-unique anchor or if already applied.
Usage: python3 apply_edits_sep4.py [--dry] [--root PAPER_DIR]"""
import argparse, os, shutil, sys
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default=".")
a = ap.parse_args()
P36 = ("We also looked for the missing check inside one open model, and found it present but unused. On Qwen2.5-7B we trained linear probes on minimal pairs of traces that differ only in the planted value, reading the residual stream at the first position after the planted line, where the tokens are identical in both members of the pair. The probe is at chance through 25 of the model's 28 layers and separates wrong from right values only in the last three (held-out AUC 0.80, 0.77, and 0.98 at layers 26, 27, and 28); at the planted value itself it reaches 1.00 at layer 27. The model computes that the committed value is wrong, but late. Steering along these directions did not turn the computation into a check. Across 150 conditions spanning single layers, layer bands, and the late layers where the signal lives, both signs, and doses up to the direction's natural magnitude, absorption stayed between 0.78 and 0.95 against a 0.90 baseline in every condition that left the model able to continue unperturbed traces, and explicit checking never exceeded 5 percent of continuations; the only doses that produced more checking language had also destroyed ordinary continuation. We read this as the representational side of deference: the information exists, in a form the generator does not consult, and amplifying it does not make the generator consult it. A fuller account of this representation is future work.")
EDITS = [
 ("sections/02_setup.tex",
  "the only difference is whether the truth can be read off or must be recomputed.",
  "the only difference is whether the truth can be read off or must be recomputed. In the wall cells the readable error is realized as an annotation line beside the model's intact computation, while the computed error replaces the model's own line; matched controls that plant both by the identical splice show the frontier contrast is unchanged by this choice, while weaker models' readable catching is partly bound to the annotation format (Section~\\ref{sec:depth}, Appendix~\\ref{sec:appendix})."),
 ("main.tex",
  "absorption is a flat ceiling at $\\approx 1.0$ for every capable model. Bars",
  "absorption is a flat ceiling at $\\approx 1.0$ for every capable model. The readable cell uses an annotation-line construction; matched-construction controls (Section~\\ref{sec:depth}) show the frontier's catching is format-robust, while open-weight models' catching is partly annotation-bound. Bars"),
 ("sections/09_discussion.tex",
  "continues predicting text in which computed values are settled.\n\nThe same reading says what should fix it",
  "continues predicting text in which computed values are settled.\n\n" + P36 + "\n\nThe same reading says what should fix it"),
]
plan = []
for rel, anchor, new in EDITS:
    p = os.path.join(a.root, rel); s = open(p).read(); c = s.count(anchor)
    if c != 1: sys.exit(f"ABORT {rel}: anchor count {c}")
    if new in s: sys.exit(f"ABORT {rel}: already applied")
    plan.append((p, s.replace(anchor, new, 1)))
print("anchors verified:", [os.path.relpath(p, a.root) for p, _ in plan])
if a.dry: print("dry run"); sys.exit(0)
for p, _ in plan: shutil.copy2(p, p + ".bak_sep4")
for p, new in plan: open(p, "w").write(new); print("wrote", os.path.relpath(p, a.root))
