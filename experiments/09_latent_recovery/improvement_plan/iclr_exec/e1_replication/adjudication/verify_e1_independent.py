"""INDEPENDENT E1 verification. Reads RAW validated_outputs.jsonl + manifest.jsonl.
Reuses ONLY the registered DV parser (stated_complement string machinery); reimplements
all stats (paired-family one-sided t, TOST, Holm, exclusion) from scratch with scipy.
Does NOT read any adjudicator intermediate file.
"""
import json, os, math, sys
from collections import defaultdict
import numpy as np
from scipy import stats as sps

EXPD = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/expd"
sys.path.insert(0, EXPD)
# registered DV instrument ONLY:
from expd_confirmatory_analysis import stated_complement as REG_stated_complement

BASE = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/e1_replication/results"
ANCHOR = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/results/EXPD_MATCHED_GRADIENT"
MODELS = [("qwen1p5b", BASE+"/qwen1p5b"), ("llama8b", BASE+"/llama8b"),
          ("olmo7b", BASE+"/olmo7b"), ("qwen32b", BASE+"/qwen32b"),
          ("qwen7b_anchor", ANCHOR)]
ALPHA = 0.05

def readjsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def load(d):
    manifest = {r["run_id"]: r for r in readjsonl(os.path.join(d, "manifest.jsonl"))}
    rows = []
    miss = 0
    for r in readjsonl(os.path.join(d, "validated_outputs.jsonl")):
        m = manifest.get(r["run_id"])
        if not m:
            miss += 1
            continue
        # recompute stated_complement from raw continuation via registered parser
        if r.get("failed_generation"):
            sc = False
        else:
            sc = bool(REG_stated_complement(r.get("continuation",""), m["injected_statement"], m["entity"]))
        r["_sc"] = int(sc)
        r["_deriv"] = int(bool(r.get("inj_derivational")))
        rows.append(r)
    return rows, manifest, miss

def by_cond(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["condition"]].append(r)
    return d

def pooled(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None

def fam_means(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r["family_id"]].append(r[key])
    return {f: float(np.mean(v)) for f,v in by.items()}

def paired(rows_a, rows_b, key):
    ma, mb = fam_means(rows_a, key), fam_means(rows_b, key)
    common = sorted(set(ma)&set(mb))
    diffs = np.array([ma[f]-mb[f] for f in common], float)
    n = len(diffs)
    mean = float(diffs.mean()) if n else None
    sd = float(diffs.std(ddof=1)) if n>1 else None
    se = sd/math.sqrt(n) if sd is not None else None
    return dict(nfam=n, mean=mean, se=se, df=n-1,
                rate_a=pooled(rows_a,key), rate_b=pooled(rows_b,key),
                na=len(rows_a), nb=len(rows_b))

def one_sided_gt(ps):
    mean, se, df = ps["mean"], ps["se"], ps["df"]
    if mean is None: return None
    if se is None or se==0:
        return 0.0 if mean>0 else 1.0
    return float(sps.t.sf(mean/se, df))

def tost(ps, margin):
    mean, se, df = ps["mean"], ps["se"], ps["df"]
    if mean is None: return None, None
    if se is None or se==0:
        p = 0.0 if abs(mean)<margin else 1.0
        return p, (mean, mean)
    p1 = float(sps.t.sf((mean+margin)/se, df))
    p2 = float(sps.t.sf((margin-mean)/se, df))
    tq = sps.t.ppf(0.95, df)  # 90% CI
    ci = (mean-tq*se, mean+tq*se)
    return max(p1,p2), ci

def cell_n_by_pos(rows, *conds):
    d = defaultdict(int)
    for r in rows:
        if r["condition"] in conds:
            d[(r["condition"], r["injection_position"])] += 1
    return d

def classify(minn):
    if minn>=150: return "full"
    if minn>=100: return "flag100-149"
    if minn>=50: return "CIwide50-99"
    return "INVALID<50"

def holm3(pv):
    items = sorted(pv.items(), key=lambda kv:(kv[1] if kv[1] is not None else 1.0))
    m=len(items); out={}; still=True
    for i,(s,p) in enumerate(items):
        thr = ALPHA/(m-i); pp = p if p is not None else 1.0
        rej = still and pp<=thr
        if not rej: still=False
        out[s]=dict(p=p, thr=thr, reject=rej)
    return out

OUT={}
for name,d in MODELS:
    rows,man,miss = load(d)
    c = by_cond(rows)
    # B1
    a0,a1 = c.get("aff_false_attr_d0",[]), c.get("aff_false_attr_d1",[])
    b1p = paired(a0,a1,"_sc"); b1_p = one_sided_gt(b1p)
    b1_diff = (b1p["rate_a"]-b1p["rate_b"]) if b1p["rate_a"] is not None and b1p["rate_b"] is not None else None
    b1n = cell_n_by_pos(rows,"aff_false_attr_d0","aff_false_attr_d1")
    b1_minn = min(b1n.values()) if b1n else 0
    b1_cls = classify(b1_minn)
    # B2
    us = c.get("cat_false_usable_d1",[])+c.get("cat_false_usable_d3",[])+c.get("cat_false_usable_dinf",[])
    ir = c.get("cat_false_inert_d1",[])+c.get("cat_false_inert_d3",[])+c.get("cat_false_inert_dinf",[])
    b2p = paired(us,ir,"_deriv"); b2_p = one_sided_gt(b2p)
    b2_diff = (b2p["rate_a"]-b2p["rate_b"]) if b2p["rate_a"] is not None and b2p["rate_b"] is not None else None
    b2n = cell_n_by_pos(rows,"cat_false_usable_d1","cat_false_usable_d3","cat_false_usable_dinf",
                        "cat_false_inert_d1","cat_false_inert_d3","cat_false_inert_dinf")
    b2_minn = min(b2n.values()) if b2n else 0
    b2_cls = classify(b2_minn)
    # B3
    a5 = c.get("aff_false_attr_d5",[])
    b3ip = paired(a1,a5,"_sc"); b3i_p, b3i_ci = tost(b3ip,0.10)
    ud1,udinf = c.get("cat_false_usable_d1",[]), c.get("cat_false_usable_dinf",[])
    b3iip = paired(ud1,udinf,"_deriv"); b3ii_p, b3ii_ci = tost(b3iip,0.15)
    b3_slotp = max([p for p in [b3i_p,b3ii_p] if p is not None], default=None)
    b3n = cell_n_by_pos(rows,"aff_false_attr_d1","aff_false_attr_d5","cat_false_usable_d1","cat_false_usable_dinf")
    b3_minn = min(b3n.values()) if b3n else 0
    b3_cls = classify(b3_minn)
    # Holm
    H = holm3({"B1":b1_p,"B2":b2_p,"B3":b3_slotp})
    def inv(cls): return cls.startswith("INVALID")
    b1_pass = H["B1"]["reject"] and (b1_diff is not None and b1_diff>=0.10) and not inv(b1_cls)
    b2_pass = H["B2"]["reject"] and (b2_diff is not None and b2_diff>=0.30) and not inv(b2_cls)
    b3_pass = H["B3"]["reject"] and not inv(b3_cls)
    repl = bool(b1_pass and b2_pass and b3_pass)
    OUT[name]=dict(nrows=len(rows),miss=miss,
        B1=dict(d0=b1p["rate_a"],d1=b1p["rate_b"],diff=b1_diff,p=b1_p,nfam=b1p["nfam"],
                minn=b1_minn,cls=b1_cls,thr=H["B1"]["thr"],reject=H["B1"]["reject"],PASS=b1_pass),
        B2=dict(us=b2p["rate_a"],inert=b2p["rate_b"],diff=b2_diff,p=b2_p,nfam=b2p["nfam"],
                minn=b2_minn,cls=b2_cls,thr=H["B2"]["thr"],reject=H["B2"]["reject"],PASS=b2_pass),
        B3=dict(i_d1=b3ip["rate_a"],i_d5=b3ip["rate_b"],i_diff=b3ip["mean"],i_ci90=b3i_ci,i_p=b3i_p,i_nfam=b3ip["nfam"],
                ii_d1=b3iip["rate_a"],ii_dinf=b3iip["rate_b"],ii_diff=b3iip["mean"],ii_ci90=b3ii_ci,ii_p=b3ii_p,ii_nfam=b3iip["nfam"],
                slotp=b3_slotp,minn=b3_minn,cls=b3_cls,thr=H["B3"]["thr"],reject=H["B3"]["reject"],PASS=b3_pass),
        repl=repl)
    def f(x,s="{:+.4f}"):
        return "None" if x is None else s.format(x)
    print(f"\n===== {name} (n={len(rows)}, miss={miss}) =====")
    b=OUT[name]
    print(f" B1 sc_d0={f(b['B1']['d0'],'{:.4f}')} sc_d1={f(b['B1']['d1'],'{:.4f}')} diff={f(b['B1']['diff'])} p={f(b['B1']['p'],'{:.5f}')} nfam={b['B1']['nfam']} minn={b['B1']['minn']}({b['B1']['cls']}) thr={b['B1']['thr']:.4f} rej={b['B1']['reject']} PASS={b['B1']['PASS']}")
    print(f" B2 us={f(b['B2']['us'],'{:.4f}')} inert={f(b['B2']['inert'],'{:.4f}')} diff={f(b['B2']['diff'])} p={f(b['B2']['p'],'{:.5f}')} nfam={b['B2']['nfam']} minn={b['B2']['minn']}({b['B2']['cls']}) thr={b['B2']['thr']:.4f} rej={b['B2']['reject']} PASS={b['B2']['PASS']}")
    ci_i = b['B3']['i_ci90']; ci_ii=b['B3']['ii_ci90']
    print(f" B3i sc d1={f(b['B3']['i_d1'],'{:.4f}')} d5={f(b['B3']['i_d5'],'{:.4f}')} diff={f(b['B3']['i_diff'])} ci90=[{f(ci_i[0]) if ci_i else 'NA'},{f(ci_i[1]) if ci_i else 'NA'}] p={f(b['B3']['i_p'],'{:.4f}')} nfam={b['B3']['i_nfam']} (m=.10)")
    print(f" B3ii deriv d1={f(b['B3']['ii_d1'],'{:.4f}')} dinf={f(b['B3']['ii_dinf'],'{:.4f}')} diff={f(b['B3']['ii_diff'])} ci90=[{f(ci_ii[0]) if ci_ii else 'NA'},{f(ci_ii[1]) if ci_ii else 'NA'}] p={f(b['B3']['ii_p'],'{:.4f}')} nfam={b['B3']['ii_nfam']} (m=.15)")
    print(f" B3 slotp={f(b['B3']['slotp'],'{:.4f}')} thr={b['B3']['thr']:.4f} minn={b['B3']['minn']}({b['B3']['cls']}) rej={b['B3']['reject']} PASS={b['B3']['PASS']}")
    print(f" >>> FULLY REPLICATES={repl}")

scored=[m for m,_ in MODELS if m!="qwen7b_anchor"]
nrep=sum(OUT[m]["repl"] for m in scored)
pc={s:sum(OUT[m][s]["PASS"] for m in scored) for s in ["B1","B2","B3"]}
print("\n########## PROGRAM ##########")
print(f" fully-replicating: {nrep}/4 (rule >=3, floor 3) -> PROGRAM REPLICATES={nrep>=3}")
print(f" per-condition passes: B1={pc['B1']}/4 B2={pc['B2']}/4 B3={pc['B3']}/4")
json.dump(OUT, open("/tmp/verify_e1_out.json","w"), indent=1, default=str)
