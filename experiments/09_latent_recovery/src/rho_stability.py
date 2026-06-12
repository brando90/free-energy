"""Null-ensemble stability: how sensitive is rho_func to ensemble size and temperature?

On the 40-instance ablation subset (wrong family, mid injection, h=2):
  - temperature: n=8 at T in {0.6, 0.8, 1.0}
  - size: subsamples of {4, 8, 16, 32} from the n=32 T=0.8 pool
Output: results/rho_stability.json
"""
import os, json, glob
import numpy as np
from collections import defaultdict
from common import load_model, split_sentences, LAYERS, RESULTS, DATA
from probes import sentence_token_offsets
from functional_metrics import jsd, boundary_state

FINAL = str(LAYERS[-1])
H = 2

def load_nulls(dirname, safe, point, tok):
    out = []
    for npz in sorted(glob.glob(os.path.join(RESULTS, dirname, "hs", f"{safe}__{point}__s*.npz"))):
        z = np.load(npz, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m.get("correct") and "text" in m:
            ns = split_sentences(m["text"])
            if ns:
                out.append((z[f"layer_{FINAL}"], sentence_token_offsets(tok, ns)))
    return out

def main():
    tok, model = load_model()
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}
    runs = [json.loads(l) for l in open(os.path.join(RESULTS, "perturbed", "runs.jsonl"))
            if "skip" not in json.loads(l)]
    runs = [r for r in runs if r["point"] == "mid"]
    subset_ids = {json.loads(l)["id"] for l in
                  open(os.path.join(RESULTS, "null_n32", "runs.jsonl"))}
    rng = np.random.RandomState(0)
    rho = defaultdict(list)
    for r in runs:
        if r["id"] not in subset_ids:
            continue
        safe = r["id"].replace("/", "_").replace("::", "__")
        gpath = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
        ppath = os.path.join(RESULTS, "perturbed", "hs", f"{safe}__mid.npz")
        if not (os.path.exists(gpath) and os.path.exists(ppath)):
            continue
        gz = np.load(gpath, allow_pickle=True)
        g_hs = gz[f"layer_{FINAL}"]
        steps = split_sentences(gold[r["id"]]["gen_text"])
        offs_g = sentence_token_offsets(tok, steps)
        gi = r["sent_idx"] + H
        if gi >= len(offs_g) or offs_g[gi] - 1 >= g_hs.shape[0]:
            continue
        g_state = g_hs[offs_g[gi] - 1]
        pz = np.load(ppath, allow_pickle=True)
        cont = split_sentences(r["continuation"])
        if not cont:
            continue
        offs_p = sentence_token_offsets(tok, cont)
        p_state = boundary_state(pz[f"layer_{FINAL}"], offs_p, H - 1)
        if p_state is None:
            continue
        d_pert = float(jsd(model, p_state[None], g_state[None])[0])

        def dnull(nulls):
            ds = []
            for nv, no in nulls:
                st = boundary_state(nv, no, H - 1)
                if st is not None:
                    ds.append(float(jsd(model, st[None], g_state[None])[0]))
            return float(np.mean(ds)) if len(ds) >= 2 else None

        for cond, dirname in [("t06_n8", "null_t06"), ("t08_n8_orig", "null"),
                              ("t10_n8", "null_t10")]:
            nulls = load_nulls(dirname, safe, "mid", tok)
            dn = dnull(nulls[:8])
            if dn and dn > 1e-6:
                rho[cond].append(d_pert / dn)
        pool = load_nulls("null_n32", safe, "mid", tok)
        for sz in [4, 8, 16, 32]:
            if len(pool) >= sz:
                idx = rng.choice(len(pool), sz, replace=False)
                dn = dnull([pool[i] for i in idx])
                if dn and dn > 1e-6:
                    rho[f"t08_n{sz}_pool"].append(d_pert / dn)
    out = {c: {"median": round(float(np.median(v)), 3), "iqr": [round(float(np.percentile(v, 25)), 3),
               round(float(np.percentile(v, 75)), 3)], "n": len(v)}
           for c, v in sorted(rho.items())}
    with open(os.path.join(RESULTS, "rho_stability.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("DONE stability", flush=True)

if __name__ == "__main__":
    main()
