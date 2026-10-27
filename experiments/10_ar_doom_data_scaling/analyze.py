import json, sys, glob, os, math
d = sys.argv[1]
order = ["base","d1_n1000_s0","d123_n1000_s0","d123_n1000_s1","d123_n1000_s2","d123_n1000_s3",
         "d12345_n1000_s0","d12345_n1000_s1","d12345_n1000_s2","d12345_n1000_s3","clean_d123_n1000_s0","clean_d12345_n1000_s0"]
lines=[]
def pct(x,n): return "%.2f"%(x/n)
for name in order:
    f=os.path.join(d,name+".json")
    if not os.path.exists(f): continue
    items=json.load(open(f))["items"]
    P={};C={}
    for it in items:
        (P if it["kind"]=="perturbed" else C)[(it["depth"],it["i"])]=it
    row=[]
    for dep in range(1,6):
        keys=[k for k in P if k[0]==dep]
        n=len(keys)
        tpr=sum(P[k]["recheck"] for k in keys)
        fpr=sum(C[k]["recheck"] for k in keys)
        both=sum(P[k]["recheck"] and C[k]["recheck"] for k in keys)
        pnc=sum(P[k]["recheck"] and not C[k]["recheck"] for k in keys)   # discriminative hit
        cnp=sum((not P[k]["recheck"]) and C[k]["recheck"] for k in keys) # flagged only the correct trace
        hit_true=sum(P[k]["recheck"] and P[k]["names_true"] for k in keys)
        row.append((dep,n,tpr/n,fpr/n,(tpr-fpr)/n,pnc,cnp,both,hit_true))
    lines.append((name,row))
out=[]
out.append("trace-level: perturbed re-check rate (TPR), same-program control re-check rate (FPR), J=TPR-FPR (0 = blanket / indiscriminate, 1 = perfect), discriminative = flagged perturbed but not its own control, reversed = flagged control only")
for name,row in lines:
    out.append("\n"+name)
    out.append("depth  n  TPR   FPR   J      discrim  reversed  both  recheck_names_true_value")
    for dep,n,t,f,j,pnc,cnp,both,ht in row:
        out.append("%d  %d  %.2f  %.2f  %+.2f   %d  %d  %d  %d"%(dep,n,t,f,j,pnc,cnp,both,ht))
print("\n".join(out))
