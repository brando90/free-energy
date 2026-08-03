#!/usr/bin/env python3
import csv
from collections import Counter
csv.field_size_limit(10**9)
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/grading"
r = csv.DictReader(open(OUT + "/regrade_full.csv"))
tot = 0; kept = 0; annot = 0; true_norb = Counter(); false_rb = 0
cont_fail = 0; styles = Counter(); dup_rows = 0
by_style_rb = Counter()
for row in r:
    tot += 1
    if row['is_kept_after_dedupe'] == '1':
        kept += 1
        if row['correct_col'] != '':
            annot += 1
        if row['correct_col'] == 'True' and row['recovery_behavior'] == '':
            true_norb[row['style']] += 1
        if row['correct_col'] == 'False' and row['recovery_behavior'] != '':
            false_rb += 1
    else:
        dup_rows += 1
    if row['cont_ok'] == '0':
        cont_fail += 1
    styles[row['style']] += 1
    if row['recovery_behavior'] and row['is_kept_after_dedupe'] == '1':
        by_style_rb[row['style']] += 1
print("total rows:", tot, "kept(dedup):", kept, "dup rows dropped:", dup_rows)
print("kept annotated:", annot)
print("Correct?=True kept rows with BLANK Recovery Behavior, by style:", dict(true_norb))
print("Correct?=False kept rows WITH a Recovery Behavior label:", false_rb)
print("continuation extraction failures (Full Prompt !startswith Question):", cont_fail)
print("style counts (all rows):", dict(styles))
print("labeled (RB) kept rows by style:", dict(by_style_rb))
