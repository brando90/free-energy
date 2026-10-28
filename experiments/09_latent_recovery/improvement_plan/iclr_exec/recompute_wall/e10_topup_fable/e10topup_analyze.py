"""E10 TOPUP + FABLE analyzer.

Pools batch1 (original) + batch2 (superset) per (model, cell), reports both
SEPARATELY and POOLED with Wilson-95 CIs, and a two-proportion batch-effect test.
Fable-5-on adds a thinking-summary re-derivation scan (raw CoT is never returned;
only the display=summarized summary is scannable).

batch1 sources (same re-execution analyzer, matching each model's protocol):
  opus-4-8, sonnet-5  -> e10_api_bridge/validated_<id>.jsonl   (bridge)
  haiku-4-5, sonnet-4-5 -> regime2/validated_<id>.jsonl        (prefill)
"""
import json, math, os, re
from collections import defaultdict

EXP="/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
RW=os.path.join(EXP,"improvement_plan","iclr_exec","recompute_wall")
TOP=os.path.join(RW,"e10_topup_fable")
BRIDGE=os.path.join(RW,"e10_api_bridge")
REG2=os.path.join(EXP,"results","EXPH2_FRONTIER_FOLLOWUPS","regime2")
CELLS=["adjacent_contradiction","opfree_kr1","onehop_kc1","deep_kc5"]

def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None

def wilson(k,n,z=1.96):
    if n==0: return [None,None]
    ph=k/n; d=1+z*z/n
    c=(ph+z*z/(2*n))/d; h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return [round(c-h,4), round(c+h,4)]

def two_prop(k1,n1,k2,n2):
    if n1==0 or n2==0: return None
    p1,p2=k1/n1,k2/n2; p=(k1+k2)/(n1+n2)
    se=math.sqrt(p*(1-p)*(1/n1+1/n2))
    if se==0: return {"delta":round(p1-p2,4),"z":0.0,"p":1.0}
    z=(p1-p2)/se; pv=math.erfc(abs(z)/math.sqrt(2))
    return {"delta":round(p1-p2,4),"z":round(z,3),"p":round(pv,4)}

def cell_stats(rows, cell):
    sub=[r for r in rows if r.get("condition")==cell and not r.get("generation_failed")]
    cf=[r for r in sub if r.get("cf")]
    k=sum(1 for r in cf if r.get("final_output_absorbed"))
    dl=sum(1 for r in cf if r.get("doubt_lex"))
    return {"n_cf":len(cf),"absorbed":k,"rate":round(k/len(cf),4) if cf else None,
            "wilson95":wilson(k,len(cf)),"doubt_lex":dl}

MODELS=[
 ("claude-opus-4-8","bridge","e10_api_bridge"),
 ("claude-sonnet-5","bridge","e10_api_bridge"),
 ("claude-haiku-4-5","prefill","regime2"),
 ("claude-sonnet-4-5","prefill","regime2"),
]
def b1_path(mid,src):
    return os.path.join(BRIDGE if src=="e10_api_bridge" else REG2, "validated_%s.jsonl"%mid)

out={"job1_pooled":{}, "fable":{}}
print("="*78); print("JOB 1 -- POOLED (batch1 + batch2) absorbed rates, four key cells"); print("="*78)
for mid,mode,src in MODELS:
    b1=load(b1_path(mid,src)); b2=load(os.path.join(TOP,"validated_%s.jsonl"%mid))
    if b1 is None or b2 is None:
        print(f"  {mid}: MISSING (b1={b1 is not None} b2={b2 is not None})"); continue
    out["job1_pooled"][mid]={"mode":mode,"batch1_src":src,"cells":{}}
    print(f"\n{mid} ({mode}; batch1={src})")
    for c in CELLS:
        s1=cell_stats(b1,c); s2=cell_stats(b2,c)
        pk=s1["absorbed"]+s2["absorbed"]; pn=s1["n_cf"]+s2["n_cf"]
        pooled={"n_cf":pn,"absorbed":pk,"rate":round(pk/pn,4) if pn else None,"wilson95":wilson(pk,pn)}
        be=two_prop(s1["absorbed"],s1["n_cf"],s2["absorbed"],s2["n_cf"])
        flag = "BATCH-EFFECT" if (be and be["p"]<0.05 and abs(be["delta"])>=0.1) else "consistent"
        out["job1_pooled"][mid]["cells"][c]={"batch1":s1,"batch2":s2,"pooled":pooled,"batch_effect":be,"flag":flag}
        print(f"  {c:24s} b1={s1['rate']} (n={s1['n_cf']})  b2={s2['rate']} (n={s2['n_cf']})  "
              f"POOLED={pooled['rate']} n={pn} CI={pooled['wilson95']}  Δ={be['delta'] if be else None} p={be['p'] if be else None} [{flag}]")

print("\n"+"="*78); print("JOB 2 -- FABLE-5"); print("="*78)
# Fable-off (main table)
foff=load(os.path.join(TOP,"validated_claude-fable-5-off.jsonl"))
if foff:
    out["fable"]["fable-5-off (effort=low, thinking floor; MAIN TABLE)"]={}
    print("\nclaude-fable-5-off  (bridge, effort=low; thinking always-on floor)")
    for c in CELLS:
        s=cell_stats(foff,c)
        out["fable"]["fable-5-off (effort=low, thinking floor; MAIN TABLE)"][c]=s
        print(f"  {c:24s} absorbed={s['rate']} (n={s['n_cf']}) CI={s['wilson95']} doubt_lex={s['doubt_lex']}")
# Fable-on satellite + thinking scan
fon=load(os.path.join(TOP,"validated_claude-fable-5-on.jsonl"))
if fon:
    out["fable"]["fable-5-on (thinking adaptive, summarized; SATELLITE)"]={}
    print("\nclaude-fable-5-on  (bridge, thinking adaptive display=summarized, effort=high)")
    for c in CELLS:
        s=cell_stats(fon,c)
        sub=[r for r in fon if r.get("condition")==c and not r.get("generation_failed") and r.get("cf")]
        # thinking-summary re-derivation scan
        with_think=[r for r in sub if (r.get("thinking_text") or "").strip()]
        def nums(t): return set(re.findall(r"-?\d+", t or ""))
        rederive_planted=sum(1 for r in sub if str(r.get("planted_value")) in nums(r.get("thinking_text")))
        rederive_true=sum(1 for r in sub if str(r.get("true_value")) in nums(r.get("thinking_text")))
        rec={"final_absorbed":s,"n_cf":len(sub),"n_with_thinking_text":len(with_think),
             "thinking_mentions_planted":rederive_planted,"thinking_mentions_true":rederive_true}
        out["fable"]["fable-5-on (thinking adaptive, summarized; SATELLITE)"][c]=rec
        print(f"  {c:24s} FINAL absorbed={s['rate']} (n={s['n_cf']}) CI={s['wilson95']} | "
              f"thinking_text present {len(with_think)}/{len(sub)}; "
              f"summary contains planted={rederive_planted} true={rederive_true}")

led=os.path.join(TOP,"cost_ledger.json")
if os.path.exists(led):
    d=json.load(open(led)); out["this_run_usd"]=d.get("total_usd"); out["usd_by_model"]=d.get("usd_by_model")
    print(f"\nthis_run_usd (e10_topup ledger) = ${d.get('total_usd')}")
    print("per-model:", json.dumps(d.get("usd_by_model")))
json.dump(out, open(os.path.join(TOP,"topup_analysis.json"),"w"), indent=2)
print("\nwrote topup_analysis.json")
