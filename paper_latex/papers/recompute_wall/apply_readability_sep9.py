#!/usr/bin/env python3
"""Approved 2026-09-09: readability batch (item 37) from READABILITY_BATCH_2026-09-04.md.
All-or-nothing anchored replace; .bak_sep9 backups; refuses on missing/non-unique anchor.
Usage: apply_readability_sep9.py --batch FILE [--dry] [--root PAPER_DIR]"""
import argparse, os, re, shutil, sys
ap = argparse.ArgumentParser(); ap.add_argument("--batch", required=True); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default=".")
a = ap.parse_args(); md = open(a.batch).read()
def unquote(block): return "\n".join(l[2:] if l.startswith("> ") else ("" if l.strip() == ">" else l) for l in block.strip().split("\n")).strip()
# ---- sentence pairs from the markdown
pairs = []  # (file, C, P)
cur_file = None
for line in md.split("\n"):
    m = re.match(r"^### (\d\d_\w+\.tex)", line)
    if m: cur_file = m.group(1)
for m in re.finditer(r"\[(\d+)\] \*\*C\*\* (.+?)\n\*\*P\*\* (.+?)\n", md, flags=re.S):
    n, C, P = int(m.group(1)), m.group(2).strip(), m.group(3).strip()
    if n == 23: continue  # handled explicitly below
    pairs.append((n, C, P))
# locate file for each pair by searching sections
sections = {f: open(os.path.join(a.root, "sections", f)).read() for f in os.listdir(os.path.join(a.root, "sections")) if f.endswith(".tex")}
edits = []  # (relpath, anchor, replacement)
for n, C, P in pairs:
    hits = [f for f, s in sections.items() if C in s]
    if len(hits) != 1: sys.exit(f"ABORT pair [{n}]: found in {hits}")
    edits.append((f"sections/{hits[0]}", C, P))
edits.append(("sections/06_fixes.tex",
  "The demonstrated condition adds the targeted text plus a worked example in which a wrong intermediate value is caught by recomputation, corrected, and the continuation proceeds from the corrected value.",
  "The demonstrated condition adds the targeted text plus a worked example. In the example a wrong intermediate value is caught by recomputation and corrected, and the continuation proceeds from the corrected value."))
# ---- abstract
ab = unquote(md[md.index("> When a language model"):md.index('What changed: "capable')])
edits.append(("sections/00_abstract.tex", "__ABSTRACT__", ab))
# ---- Sec 7 rewrite (paragraphs 2-4)
s7new = unquote(md[md.index("> Enabling reasoning on GPT-5.1"):md.index("What changed: sentences shortened")])
s7 = sections["07_reasoning.tex"]; i = s7.index("Enabling reasoning on GPT-5.1"); j = s7.index("fresh derivation of the contrary.") + len("fresh derivation of the contrary.")
old7 = s7[i:j]; assert "22 to 35 percent" in old7 and old7.count("\n\n") == 2, "Sec 7 span unexpected"
edits.append(("sections/07_reasoning.tex", old7, s7new))
# ---- naming lines + item 29 + conclusion item 20
edits += [
 ("sections/01_intro.tex", "for the frontier model, flips", "for Haiku 4.5, flips"),
 ("sections/04_depth.tex", "For the frontier model the result is a step function.", "For Haiku 4.5 the result is a step function."),
 ("sections/04_depth.tex", "The frontier model absorbs the same matched plant at 0.14", "Haiku 4.5 absorbs the same matched plant at 0.14"),
 ("sections/04_depth.tex", "the frontier model begins near zero", "Haiku 4.5 begins near zero"),
 ("sections/04_depth.tex", "depth two places two, and so on.", "depth two places two, and so on. These arms, and the opacity and instruction arms that follow, run on Claude Haiku 4.5, the frontier-developer model we could run at this volume with true prefilling, and on Llama-3.1-8B and Qwen2.5-7B."),
 ("sections/05_opacity.tex", "For the frontier model absorption stays above 0.97", "For Haiku 4.5 absorption stays above 0.97"),
 ("sections/05_opacity.tex", "0.57 for the frontier model", "0.57 for Haiku 4.5"),
 ("sections/06_fixes.tex", "The frontier model absorbs at 1.00 in all four conditions", "Haiku 4.5 absorbs at 1.00 in all four conditions"),
 ("sections/10_conclusion.tex", "from almost never for a 1.5-billion-parameter model to almost always at the frontier", "from less than half the time for the small open-weight models to almost always at the frontier"),
 ("main.bib", "note   = {arXiv:2605.05737. TODO verify full author list}", "note   = {arXiv:2605.05737}"),
 ("main.bib", "note   = {arXiv:2606.25449. TODO verify full author list}", "note   = {arXiv:2606.25449}"),
]
# ---- apply in memory, all-or-nothing
files = {}
def get(rel):
    if rel not in files: files[rel] = open(os.path.join(a.root, rel)).read()
    return files[rel]
for rel, anchor, new in edits:
    s = get(rel)
    if anchor == "__ABSTRACT__":
        i = s.index("\\begin{abstract}") + len("\\begin{abstract}"); j = s.index("\\end{abstract}")
        files[rel] = s[:i] + "\n" + new + "\n" + s[j:]; continue
    c = s.count(anchor)
    if c != 1: sys.exit(f"ABORT {rel}: anchor count {c} for: {anchor[:80]}")
    files[rel] = s.replace(anchor, new, 1)
# post-checks
body = "".join(v for k, v in files.items() if k.startswith("sections/") and not k.endswith("A_appendix.tex"))
left = re.findall(r"[^.]*\bthe frontier model\b[^.]*\.", body)
if left: sys.exit("ABORT: 'the frontier model' still present: " + str(left[:3]))
if "TODO" in files.get("main.bib", ""): sys.exit("ABORT: TODO remains in bib")
print(f"verified {len(edits)} edits across {len(files)} files: {sorted(files)}")
if a.dry: print("dry run"); sys.exit(0)
for rel in files: shutil.copy2(os.path.join(a.root, rel), os.path.join(a.root, rel) + ".bak_sep9")
for rel, s in files.items(): open(os.path.join(a.root, rel), "w").write(s); print("wrote", rel)
