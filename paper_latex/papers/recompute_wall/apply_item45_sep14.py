#!/usr/bin/env python3
"""Item 45 (approved 2026-09-14): statement-boundary test into Sec 8 + Appendix A.5. All-or-nothing, .bak_sep14c."""
import os, re, shutil, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--root", default="."); a = ap.parse_args()
T, A = "sections/08_related.tex", "sections/A_appendix.tex"
files = {r: open(os.path.join(a.root, r)).read() for r in (T, A)}
def rep(rel, old, new):
    s = files[rel]; pat = re.compile(r"\s+".join(re.escape(w) for w in old.split())); ms = list(pat.finditer(s))
    if len(ms) != 1: sys.exit(f"ABORT {rel}: {len(ms)} matches for {old[:60]!r}")
    m = ms[0]; files[rel] = s[:m.start()] + new + s[m.end():]
rep(T, "We read their results as characterizing in-flight values and ours as characterizing committed ones, and we treat the boundary as the hypothesis for the gap rather than an established cause.",
       "We read their results as characterizing in-flight values and ours as characterizing committed ones. A direct test on their own items supports the direction but not the size: completing the statement after the corrupted numeral lowers GPT-4o's recovery by about 9 points in each of two numberings (pooled paired $p=0.08$ over 90 items; Appendix~\\ref{app:renumber}), so the boundary accounts for part of the gap and the rest lies in the task and the grader.")
PARA = ("\\paragraph{Statement-boundary test.} To test whether resuming mid-statement is what lets their models recover, we took the 45 original-number items from the renumbering replication whose original chain continues past the corrupted numeral without another number before the end of the sentence, and their 45 renumbered twins. For each we built two prefixes identical up to the corrupted numeral: one ending on the numeral, as in their protocol, and one ending after the sentence's own closing words taken verbatim from the model's original chain (median one word). Same model, prompt, and decoding as the replication; uncorrupted prefixes at both boundaries as controls; 360 calls. GPT-4o recovers 55.6 percent of original items at the numeral boundary and 46.7 percent at the statement boundary; on renumbered items, 40.0 and 31.1 percent. The paired difference is 8.9 points in both numberings (pooled over 90 items: 31 recovered at both boundaries, 12 only at the numeral, 4 only after the statement, 43 at neither; exact McNemar $p=0.08$). Propagation rises by the same amount. Uncorrupted prefixes are unaffected (93.3 and 97.8 percent correct). Completing the statement therefore moves recovery toward propagation, in the registered direction, but by far less than the gap between their recovery rates and ours; the boundary is part of the explanation, not the whole of it. Reading the continuations shows the in-flight mechanism directly: at the numeral boundary the model sometimes finishes the number itself, writing the correct value's remaining digits after the planted one, which no statement-boundary continuation can do. The test was registered before the run (specification and amendment in the experiment archive).")
anchor = "not as a full decomposition of it."
if files[A].count(anchor) != 1: sys.exit("ABORT appendix anchor")
files[A] = files[A].replace(anchor, anchor + "\n\n" + PARA, 1)
print("verified")
if a.dry: sys.exit(0)
for r in files: shutil.copy2(os.path.join(a.root, r), os.path.join(a.root, r) + ".bak_sep14c")
for r, s in files.items(): open(os.path.join(a.root, r), "w").write(s); print("wrote", r)
