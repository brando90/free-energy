import json, csv
from collections import Counter, defaultdict
csv.field_size_limit(10**8)
R="/lfs/skampere2/0/eobbad/scratch/yee_audit/repo/results"
def norm(s): return (s or "").replace("\r\n","\n").replace("\r","\n")

cp=f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add1_annotated.csv"
rows=list(csv.DictReader(open(cp, encoding="utf-8")))

# 1. Do claude/llama share gpt-4's perturbed prefixes (Question col)?
qsets=defaultdict(set)
for r in rows: qsets[r["Model Name"]].add(norm(r["Question"]).strip())
g=qsets["gpt-4-0314"]; c=qsets["claude-3-opus-20240229"]; l=qsets["meta-llama/Llama-3-70b-chat-hf"]
print("unique Question per model:", {k:len(v) for k,v in qsets.items()})
print("claude & gpt4 overlap:", len(c & g), "claude only:", len(c-g))
print("llama & gpt4 overlap:", len(l & g), "llama only:", len(l-g))
# compare on bare question text (before A:)
def bare(q): return q.split("\n\nA:")[0]
gb={bare(q) for q in g}; cb={bare(q) for q in c}; lb={bare(q) for q in l}
print("bare-question overlap gpt4 vs claude:", len(gb & cb), "of", len(cb))
print("bare-question overlap gpt4 vs llama:", len(gb & lb), "of", len(lb))
# for one shared bare question, compare the prefix part
shared=sorted(gb & cb)
if shared:
    q0=shared[0]
    gq=[q for q in g if bare(q)==q0][0]; cq=[q for q in c if bare(q)==q0][0]
    print("identical full Question (incl prefix)?", gq==cq)
    print("gpt4 prefix tail:", repr(gq[-100:]))
    print("claude prefix tail:", repr(cq[-100:]))
print()

# 2. duplicate structure for gpt-4 rows
gq_rows=[r for r in rows if r["Model Name"]=="gpt-4-0314"]
qc=Counter(norm(r["Question"]).strip() for r in gq_rows)
print("gpt-4 rows:", len(gq_rows), "unique questions:", len(qc), "dup dist:", Counter(qc.values()))
# pick a duplicated question, compare rows
dupq=[q for q,n in qc.items() if n==2][0]
pair=[r for r in gq_rows if norm(r["Question"]).strip()==dupq]
print("pair Answers:", [repr(r["Answer"][:60]) for r in pair])
print("pair Correct?:", [r["Correct?"] for r in pair])
print("pair FullPrompt equal?", norm(pair[0]["Full Prompt"])==norm(pair[1]["Full Prompt"]))
print()

# 3. baseline CSV structure
for f in ["gsm8k/test.csv","asdiv/ASDiv.csv"]:
    rws=list(csv.DictReader(open(f"{R}/{f}", encoding="utf-8")))
    print(f, "rows:", len(rws))
    print(" cols:", list(rws[0].keys()))
    print(" models:", Counter(r.get("Model Name") for r in rws).most_common(8))
    print(" styles:", Counter(r.get("Prompt Style") for r in rws).most_common(8))
    r0=rws[0]
    print(" sample FullPrompt head:", repr(norm(r0.get("Full Prompt",""))[:150]))
    print(" sample FullPrompt tail:", repr(norm(r0.get("Full Prompt",""))[-120:]))
    print(" sample Answer:", repr(norm(r0.get("Answer",""))[:80]))
    print()

# 4. error_audit.csv
rws=list(csv.reader(open(f"{R}/error_audit.csv", encoding="utf-8")))
print("error_audit.csv rows:", len(rws))
for row in rws[:5]: print([c[:60] for c in row])
