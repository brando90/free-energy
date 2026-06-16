"""Adversarial audit: recompute every contested number from raw rows.

0a. Row counts per family/point + per-horizon effective n behind rho CIs
0b. Recompute headline valid-rederivation rates from validated*.jsonl raw rows
0c. Re-run the doubt regex fresh on stored continuations
A.  Goal-jump audit: intermediate-fact counts pert vs gold; derivation-free rate;
    corrected-fact re-derivation rate
F.  Wilson CIs for behavioral cells; two-proportion tests for claimed orderings
"""
import os, json, glob, re, math
import numpy as np
from collections import defaultdict
from common import RESULTS, DATA, MODEL, split_sentences
from validator import parse_fact, DOUBT
from probes import sentence_token_offsets

FAMS = [("wrong", "perturbed"), ("distractor", "perturbed_distractor"),
        ("contradiction", "perturbed_contradiction")]

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def two_prop_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (k1 / n1 - k2 / n2) / se
    from math import erf
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
    return z, pval

def main():
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}
    out = {}

    # ---------- 0b: recompute headline rates from raw validated rows ----------
    print("=" * 70)
    print("0b. HEADLINE RATES RECOMPUTED FROM RAW validated*.jsonl ROWS")
    behav = {}
    for fam, d in FAMS:
        suf = "" if fam == "wrong" else f"_{fam}"
        path = os.path.join(RESULTS, f"validated{suf}.jsonl")
        rows = [json.loads(l) for l in open(path)]
        for p in ["early", "mid", "late"]:
            sub = [r for r in rows if r["point"] == p]
            k = sum(1 for r in sub if r["class"] == "valid_rederivation")
            n = len(sub)
            lo, hi = wilson(k, n)
            behav[(fam, p)] = (k, n)
            print(f"  {fam:14s} {p:6s}: {k}/{n} = {k/n:.4f}  Wilson95=[{lo:.3f},{hi:.3f}]")

    # ---------- 0c: doubt regex re-run on stored continuations ----------
    print("=" * 70)
    print("0c. DOUBT REGEX RE-RUN FRESH ON STORED CONTINUATIONS (cohort rows)")
    for fam, d in FAMS:
        suf = "" if fam == "wrong" else f"_{fam}"
        vrows = {(json.loads(l)["id"], json.loads(l)["point"])
                 for l in open(os.path.join(RESULTS, f"validated{suf}.jsonl"))}
        runs = [json.loads(l) for l in open(os.path.join(RESULTS, d, "runs.jsonl"))
                if "skip" not in json.loads(l)]
        for p in ["early", "mid", "late"]:
            sub = [r for r in runs if r["point"] == p and (r["id"], p) in vrows]
            k = sum(1 for r in sub if DOUBT.search(r["continuation"]))
            print(f"  {fam:14s} {p:6s}: doubt {k}/{len(sub)}")

    # ---------- 0a: per-horizon effective n behind rho ----------
    print("=" * 70)
    print("0a. EFFECTIVE n PER HORIZON behind rho_func (same gates as the metric)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    cohort = {json.loads(l)["id"] for l in open(os.path.join(RESULTS, "validated.jsonl"))}
    null_cache = {}
    for fam, d in FAMS:
        runs = [json.loads(l) for l in open(os.path.join(RESULTS, d, "runs.jsonl"))
                if "skip" not in json.loads(l)]
        n_h = defaultdict(lambda: np.zeros(8, dtype=int))
        for r in runs:
            if r["id"] not in cohort:
                continue
            safe = r["id"].replace("/", "_").replace("::", "__")
            gpath = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
            ppath = os.path.join(RESULTS, d, "hs", f"{safe}__{r['point']}.npz")
            if not (os.path.exists(gpath) and os.path.exists(ppath)):
                continue
            steps = split_sentences(gold[r["id"]]["gen_text"])
            offs_g = sentence_token_offsets(tok, steps)
            gz = np.load(gpath, allow_pickle=True)
            gT = gz[[k for k in gz.files if k.startswith("layer_")][0]].shape[0]
            pz = np.load(ppath, allow_pickle=True)
            pT = pz[[k for k in pz.files if k.startswith("layer_")][0]].shape[0]
            cont = split_sentences(r["continuation"])
            if not cont:
                continue
            offs_p = sentence_token_offsets(tok, cont)
            key = (safe, r["point"])
            if key not in null_cache:
                noffs = []
                for npz in sorted(glob.glob(os.path.join(RESULTS, "null", "hs",
                                                         f"{safe}__{r['point']}__s*.npz"))):
                    nz = np.load(npz, allow_pickle=True)
                    nm = json.loads(str(nz["meta"]))
                    if not nm.get("correct") or "text" not in nm:
                        continue
                    ns = split_sentences(nm["text"])
                    if ns:
                        nT = nz[[k for k in nz.files if k.startswith("layer_")][0]].shape[0]
                        noffs.append(([o for o in sentence_token_offsets(tok, ns)], nT))
                null_cache[key] = noffs
            noffs = null_cache[key]
            if len(noffs) < 2:
                continue
            si = r["sent_idx"]
            for h in range(1, 9):
                gi = si + h
                if gi >= len(offs_g) or offs_g[gi] - 1 >= gT:
                    break
                if h - 1 >= len(offs_p) or offs_p[h - 1] - 1 >= pT:
                    break
                n_ok = sum(1 for no, nT in noffs if h - 1 < len(no) and no[h - 1] - 1 < nT)
                if n_ok < 2:
                    break
                n_h[r["point"]][h - 1] += 1
        for p in ["early", "mid", "late"]:
            print(f"  {fam:14s} {p:6s}: n per h=1..8 -> {list(n_h[p])}")

    # ---------- A: goal-jump audit ----------
    print("=" * 70)
    print("A. GOAL-JUMP AUDIT (wrong family, valid_rederivation rows only)")
    vrows = [json.loads(l) for l in open(os.path.join(RESULTS, "validated.jsonl"))]
    vmap = {(r["id"], r["point"]): r for r in vrows}
    runs = {(json.loads(l)["id"], json.loads(l)["point"]): json.loads(l)
            for l in open(os.path.join(RESULTS, "perturbed", "runs.jsonl"))
            if "skip" not in json.loads(l)}
    def _norm(x):
        return re.sub(r"[^a-z0-9 ]", "", x.lower()).strip()
    stats = defaultdict(lambda: {"n": 0, "jump0": 0, "rederive_correct": 0,
                                 "pert_facts": [], "gold_facts": []})
    for (gid, p), v in vmap.items():
        if v["class"] != "valid_rederivation" or (gid, p) not in runs:
            continue
        r = runs[(gid, p)]
        steps = split_sentences(gold[gid]["gen_text"])
        entity = steps[0].split()[0]
        si = r["sent_idx"]
        target = data[gid]["target"]
        cont = split_sentences(r["continuation"])
        inter = [s for s in cont[:-1] if parse_fact(s, entity)]      # intermediate entity facts
        gold_inter = [s for s in steps[si + 1: -1] if parse_fact(s, entity)]
        st = stats[p]
        st["n"] += 1
        st["jump0"] += int(len(inter) == 0)
        st["rederive_correct"] += int(any(_norm(s) == _norm(steps[si]) for s in cont))
        st["pert_facts"].append(len(inter))
        st["gold_facts"].append(len(gold_inter))
    for p in ["early", "mid", "late"]:
        st = stats[p]
        print(f"  {p:6s}: n={st['n']}  derivation-free goal-jumps={st['jump0']} "
              f"({st['jump0']/st['n']:.1%})  re-derives CORRECT version of lied-about fact="
              f"{st['rederive_correct']} ({st['rederive_correct']/st['n']:.1%})")
        print(f"          intermediate entity-facts: pert median={np.median(st['pert_facts']):.0f} "
              f"mean={np.mean(st['pert_facts']):.2f} | gold-remaining median={np.median(st['gold_facts']):.0f} "
              f"mean={np.mean(st['gold_facts']):.2f}")

    # ---------- F: significance tests for claimed orderings ----------
    print("=" * 70)
    print("F. TWO-PROPORTION TESTS (z, p) FOR CLAIMED ORDERINGS")
    tests = [
        ("wrong early vs mid (valid)", behav[("wrong", "early")], behav[("wrong", "mid")]),
        ("wrong late vs mid (valid)", behav[("wrong", "late")], behav[("wrong", "mid")]),
        ("wrong vs distractor pooled-early (valid)", behav[("wrong", "early")], behav[("distractor", "early")]),
    ]
    for name, (k1, n1), (k2, n2) in tests:
        z, pv = two_prop_z(k1, n1, k2, n2)
        print(f"  {name}: {k1}/{n1} vs {k2}/{n2}  z={z:.2f} p={pv:.4f}")
    # doubt contrast: contradiction-early vs wrong-early, fresh counts
    suf_rows = {}
    for fam, d in [("wrong", "perturbed"), ("contradiction", "perturbed_contradiction")]:
        suf = "" if fam == "wrong" else f"_{fam}"
        vr = {(json.loads(l)["id"], json.loads(l)["point"])
              for l in open(os.path.join(RESULTS, f"validated{suf}.jsonl"))}
        rr = [json.loads(l) for l in open(os.path.join(RESULTS, d, "runs.jsonl"))
              if "skip" not in json.loads(l)]
        sub = [r for r in rr if r["point"] == "early" and (r["id"], "early") in vr]
        suf_rows[fam] = (sum(1 for r in sub if DOUBT.search(r["continuation"])), len(sub))
    (k1, n1), (k2, n2) = suf_rows["contradiction"], suf_rows["wrong"]
    z, pv = two_prop_z(k1, n1, k2, n2)
    print(f"  doubt: contradiction-early {k1}/{n1} vs wrong-early {k2}/{n2}  z={z:.2f} p={pv:.6f}")

if __name__ == "__main__":
    main()
