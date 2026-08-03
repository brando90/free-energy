import json, csv
from collections import Counter
csv.field_size_limit(10**8)
R="/lfs/skampere2/0/eobbad/scratch/yee_audit/repo/results"
jp=f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add1_gpt-4-0314.json"
cp=f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add1_annotated.csv"
d=json.load(open(jp))
rows=list(csv.DictReader(open(cp, encoding="utf-8")))
SUF=" Therefore, the answer (arabic numerals) is"
def norm(s): return s.replace("\r\n","\n").replace("\r","\n")
# crosstab
ct=Counter((r["Model Name"], r["Correct?"], bool(r["Recovery Behavior"])) for r in rows)
for k,v in sorted(ct.items()): print(k,v)
print()
et=Counter((r["Model Name"], r["Error Type"]) for r in rows)
for k,v in sorted(et.items()): print(k,v)
print()
# cross-reference 3 examples for gpt-4-0314
checked=0
for k,(prefix,target) in d.items():
    kk=norm(k).strip(); pf=norm(prefix).strip()
    matches=[r for r in rows if r["Model Name"]=="gpt-4-0314" and kk[:80] in norm(r["Question"])]
    if not matches: continue
    r=matches[0]
    q=norm(r["Question"]); fp=norm(r["Full Prompt"]); ans=r["Answer"]
    print("=== EXAMPLE", checked)
    print("num CSV rows matching this question (gpt4):", len(matches))
    print("Question col starts with Q: ?", q.startswith("Q: "))
    print("Question col contains sbs prompt?", "Let's think step by step." in q)
    print("Question col ends with prefix?", q.rstrip().endswith(pf[-60:].rstrip()))
    print("FullPrompt starts with Question col (first 200 chars)?", fp.startswith(q.rstrip()[:200]))
    print("FullPrompt ends with suffix?", fp.rstrip().endswith(SUF.strip()))
    tail=pf[-60:]
    i=fp.find(tail)
    print("prefix end idx in FullPrompt:", i)
    if i>=0:
        cont=fp[i+60:]
        print("CONTINUATION (first 200):", repr(cont[:200]))
        print("CONTINUATION (last 100):", repr(cont[-100:]))
    print("ANSWER col:", repr(ans[:200]))
    print("Target:", target, "Correct?:", r["Correct?"], "RB:", repr(r["Recovery Behavior"]), "Notes:", repr(r["Notes"][:100]))
    checked+=1
    if checked==3: break
