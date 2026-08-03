import json, csv, re
from collections import Counter, defaultdict
csv.field_size_limit(10**8)
R="/lfs/skampere2/0/eobbad/scratch/yee_audit/repo/results"
def norm(s): return (s or "").replace("\r\n","\n").replace("\r","\n")

# 1. all gpt-4 dup pairs identical?
cp=f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add1_annotated.csv"
rows=list(csv.DictReader(open(cp, encoding="utf-8")))
by_q=defaultdict(list)
for r in rows:
    if r["Model Name"]=="gpt-4-0314": by_q[norm(r["Question"]).strip()].append(r)
same_fp=same_ans=diff_fp=diff_ans=0
anno_pattern=Counter()
for q,rs in by_q.items():
    if len(rs)==2:
        a,b=rs
        (same_fp,diff_fp)[norm(a["Full Prompt"])!=norm(b["Full Prompt"])]  # noqa
        if norm(a["Full Prompt"])==norm(b["Full Prompt"]): same_fp+=1
        else: diff_fp+=1
        if a["Answer"]==b["Answer"]: same_ans+=1
        else: diff_ans+=1
        anno_pattern[(bool(a["Correct?"]), bool(b["Correct?"]))]+=1
print("dup pairs: same FP", same_fp, "diff FP", diff_fp, "same Ans", same_ans, "diff Ans", diff_ans)
print("annotation pattern (row1 annotated, row2 annotated):", dict(anno_pattern))

# 2. Correct? vs number_scorer(Answer, Target)
GSM8K_SUFFIX=" Therefore, the answer (arabic numerals) is"
def number_scorer(generation, target):
    generation=str(generation)
    if GSM8K_SUFFIX in generation: generation=generation.split(GSM8K_SUFFIX)[-1]
    generation=generation.replace(',','')
    m=re.search(r'-?\d+\.?\d*', generation)
    try: t=float(str(target).replace(',',''))
    except: return None
    if m:
        try: return t==float(m.group(0))
        except: return None
    return False
agree=disagree=0; examples=[]
for r in rows:
    if not r["Correct?"]: continue
    s=number_scorer(r["Answer"], r["Target Answer"])
    if s is None: continue
    manual = (r["Correct?"]=="True")
    if s==manual: agree+=1
    else:
        disagree+=1
        if len(examples)<5: examples.append((r["Model Name"], r["Answer"][:80], r["Target Answer"], r["Correct?"]))
print("Correct? vs scorer: agree", agree, "disagree", disagree)
for e in examples: print("  disagree:", e)

# 3. add1 vs add101 JSON matched?
d1=json.load(open(f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add1_gpt-4-0314.json"))
d101=json.load(open(f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add101_gpt-4-0314.json"))
common=set(d1)&set(d101)
print("add1 keys:", len(d1), "add101 keys:", len(d101), "common:", len(common))
k=sorted(common)[0]
p1,p101=d1[k][0],d101[k][0]
i=next((j for j in range(min(len(p1),len(p101))) if p1[j]!=p101[j]), None)
print("first divergence at char", i)
print(" add1 around:", repr(p1[max(0,i-40):i+10]) if i is not None else "identical")
print(" add101 around:", repr(p101[max(0,i-40):i+10]) if i is not None else "")

# 4. Answer col length distribution (extraction output)
lens=[len(r["Answer"]) for r in rows if r["Answer"]]
lens.sort()
print("Answer len: median", lens[len(lens)//2], "p90", lens[int(len(lens)*0.9)], "max", lens[-1])
long=[r["Answer"] for r in rows if len(r["Answer"])>200][:2]
for a in long: print("LONG ANSWER sample:", repr(a[:300]))

# 5. random-perturbation JSON for gpt-3.5 sample
d35=json.load(open(f"{R}/gsm8k/test_adjusted_position-calc_perturbation-random_gpt-3-5-turbo-0301.json"))
print("gpt-3.5 random stimuli:", len(d35))
v=list(d35.values())[0]
print(" value types:", type(v), len(v), repr(v[0][-80:]), repr(v[1]))
