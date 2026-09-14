#!/usr/bin/env python3
"""Fig 3 + Fig 4 side by side in one float (captions verbatim, each keeps its own number/label); float placed
after Sec 4 input. .bak_sep13e backup of main.tex."""
import re, shutil, sys, os
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
p = os.path.join(ROOT, "main.tex"); s = open(p).read()
def block(figfile):
    pat = r"(% CAPTION DRAFT[^\n]*\n)?\\begin\{figure\}\[t\]\n\\centering\n\\includegraphics\[width=0\.7\\textwidth\]\{figures/FIGFILE\}\n(\\caption\{.*?\})\n(\\label\{[^}]*\})\n\\end\{figure\}\n".replace("FIGFILE", re.escape(figfile))
    m = re.search(pat, s, flags=re.S)
    assert m, figfile; return m
b3, b4 = block("fig3.pdf"), block("fig4.pdf")
cap3, lab3 = b3.group(2), b3.group(3); cap4, lab4 = b4.group(2), b4.group(3)
combined = ("% CAPTION DRAFT -- final wording with author (both captions verbatim; floats merged 2026-09-13 for page limit)\n"
 "\\begin{figure}[t]\n\\centering\n"
 "\\begin{minipage}[t]{0.48\\textwidth}\n\\centering\n\\includegraphics[width=\\linewidth]{figures/fig3.pdf}\n" + cap3 + "\n" + lab3 + "\n\\end{minipage}\\hfill\n"
 "\\begin{minipage}[t]{0.48\\textwidth}\n\\centering\n\\includegraphics[width=\\linewidth]{figures/fig4.pdf}\n" + cap4 + "\n" + lab4 + "\n\\end{minipage}\n"
 "\\end{figure}\n")
s2 = s[:b3.start()] + combined + s[b3.end():]
m4 = re.search(r"(% CAPTION DRAFT[^\n]*\n)?\\begin\{figure\}\[t\]\n\\centering\n\\includegraphics\[width=0\.7\\textwidth\]\{figures/fig4\.pdf\}\n\\caption\{.*?\}\n\\label\{[^}]*\}\n\\end\{figure\}\n\n?", s2, flags=re.S)
assert m4; s2 = s2[:m4.start()] + s2[m4.end():]
assert s2.count("fig3.pdf") == 1 and s2.count("fig4.pdf") == 1 and s2.count("fig:depth") == 1 and s2.count("fig:opacity") == 1
if "--dry" in sys.argv: print("ok"); sys.exit(0)
shutil.copy2(p, p + ".bak_sep13e"); open(p, "w").write(s2); print("main.tex: figs 3+4 merged")
