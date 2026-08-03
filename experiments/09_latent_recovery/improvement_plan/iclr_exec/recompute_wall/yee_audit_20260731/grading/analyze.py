#!/usr/bin/env python3
import csv, json, sys
from collections import Counter, defaultdict
csv.field_size_limit(10**9)
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/grading"

rows = list(csv.DictReader(open(OUT + "/cells_regrade.csv")))

def grid(model):
    print(f"\n===== {model} =====")
    print(f"{'ds':6s} {'pos':8s} {'pert':7s} {'ty':2s} | {'pipe_n':>6s} {'pipe%':>6s} | {'strict_n':>8s} {'strict%':>7s} | {'blank_n':>7s} {'blank%':>6s} | {'cc_n':>5s} {'cc%':>6s} | rows uniq annot")
    for r in rows:
        if r["model"] == model:
            print(f'{r["dataset"]:6s} {r["position"]:8s} {r["perturbation"]:7s} {r["typo10"]:2s} | '
                  f'{r["pipe_n"]:>6s} {r["pipe_rate"]:>6s} | {r["strict_n"]:>8s} {r["strict_rate"]:>7s} | '
                  f'{r["blankok_n"]:>7s} {r["blankok_rate"]:>6s} | {r["cc_n"]:>5s} {r["cc_rate"]:>6s} | '
                  f'{r["rows"]:>4s} {r["unique"]:>4s} {r["unique_annotated"]:>5s}')

for m in ['gpt-4-0314', 'gpt-3.5-turbo-0301', 'claude-3-opus-20240229', 'meta-llama/Llama-3-70b-chat-hf']:
    grid(m)

# paper comparison: GPT-4 Exp3 cells
paper = {  # (dataset,position): (rate, n)  -- perturbation unknown, test all
    ('awps', 'calc'): (77.73, 247), ('asdiv', 'calc'): (74.34, 226),
    ('svamp', 'calc'): (76.54, 243), ('gsm8k', 'calc'): (78.57, 224),
    ('awps', 'copy'): (99.32, None), ('asdiv', 'copy'): (97.45, None),
    ('svamp', 'copy'): (97.71, None), ('gsm8k', 'copy'): (94.98, None),
    ('awps', 'propcalc'): (32.54, None), ('asdiv', 'propcalc'): (52.71, None),
    ('svamp', 'propcalc'): (54.47, None), ('gsm8k', 'propcalc'): (26.45, None),
}
print("\n===== PAPER MATCH SEARCH (gpt-4-0314) =====")
print("paper cell -> candidates (pert, metric, n, rate) with |rate-paper|<0.6 or exact n match")
for (ds, pos), (prate, pn) in paper.items():
    cands = []
    for r in rows:
        if r["model"] != "gpt-4-0314" or r["dataset"] != ds or r["position"] != pos:
            continue
        for metric in ["pipe", "strict", "blankok", "cc"]:
            n = r[metric + "_n"] if metric != "cc" else r["cc_n"]
            rate = r[metric + "_rate"] if metric != "cc" else r["cc_rate"]
            if not rate:
                continue
            n, rate = int(n), float(rate)
            tag = f'{r["perturbation"]}{"+typo" if r["typo10"]=="1" else ""}/{metric}: n={n} rate={rate}'
            if abs(rate - prate) < 0.6 or (pn and n == pn):
                cands.append(tag)
    print(f"  {ds:6s} {pos:8s} paper={prate}/{pn}: " + ("; ".join(cands) if cands else "NO MATCH <0.6"))

# disagreement direction breakdown
dis = list(csv.DictReader(open(OUT + "/disagreements_correct_col.csv")))
c = Counter((d["correct_col"], d["regrade_scorer"], d["alt_first"], d["alt_any"]) for d in dis)
print("\n===== DISAGREEMENT BREAKDOWN (correct_col, scorer, alt_first, alt_any) =====")
for k, v in sorted(c.items()):
    print("  ", k, v)

# ack crosstab detail
ct = json.load(open(OUT + "/ack_crosstab.json"))
print("\n===== ACK CROSSTAB tier1 or tier2 =====")
for k, v in ct["tier1or2"].items():
    print("  ", k, v)
print("\n--- unfaithful_with_ack examples (first 8) ---")
for e in ct["examples"].get("unfaithful_with_ack", [])[:8]:
    print(json.dumps(e)[:500])
print("\n--- explicit_without_ack examples (first 8) ---")
for e in ct["examples"].get("explicit_without_ack", [])[:8]:
    print(json.dumps(e)[:500])
