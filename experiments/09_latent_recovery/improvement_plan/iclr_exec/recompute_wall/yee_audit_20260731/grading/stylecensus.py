#!/usr/bin/env python3
"""Census of Prompt Style values per file/model + annotation status by style."""
import csv, glob, os
from collections import Counter
csv.field_size_limit(10**9)
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
files = sorted(glob.glob(os.path.join(REPO, 'results', '*', '*_annotated.csv')))
grand = Counter()
for path in files:
    base = os.path.basename(path)
    c = Counter()
    ann = Counter()
    for row in csv.DictReader(open(path, encoding='utf-8', newline='')):
        st = row['Prompt Style']
        c[st] += 1
        grand[st] += 1
        if row['Correct?'].strip() != '':
            ann[st] += 1
    interesting = {k: v for k, v in c.items() if k != 'sbs'}
    print(f"{base[:70]:70s} styles={dict(c)} annotated_by_style={dict(ann)}")
print("\nGRAND:", dict(grand))
