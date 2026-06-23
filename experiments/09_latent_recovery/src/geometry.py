"""Layer-wise geometric divergence with the lexical control built in.

For each (instance, point, horizon h<=4) and each cached layer L:
  rho_geo(L) = cosdist(pert_L, gold_L) / mean_nulls cosdist(null_L, gold_L)
at matched sentence boundaries (same alignment contract as rho_func).

Lexical control: layer 0 IS the token-identity layer. If rho_geo at deep layers
tracks rho_geo at layer 0, geometric "re-entry" is token echo. Genuine
representational re-entry = deep-layer rho below layer-0 rho.

Projection variant: deep states orthogonalized against the layer-0 direction at the
same position (removes the component linearly predictable from token identity);
rho_geo recomputed on the residual.

Output: results/geometry.json + per-family deep-vs-L0 summary.
"""
import os, json, glob
import numpy as np
from collections import defaultdict
from common import LAYERS, RESULTS, DATA, MODEL, split_sentences
from probes import sentence_token_offsets

H_MAX = 4
FAMS = [("wrong", "perturbed"), ("paraphrase", "perturbed_paraphrase"),
        ("negstep", "perturbed_negstep"), ("falsehood", "perturbed_falsehood"),
        ("contradiction", "perturbed_contradiction")]

def cosdist(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def proj_out(h, h0):
    """Remove from h its component along the layer-0 direction h0."""
    h = h.astype(np.float32); h0 = h0.astype(np.float32)
    u = h0 / (np.linalg.norm(h0) + 1e-8)
    return h - (h @ u) * u

def boundary(hs, offs, idx):
    if idx >= len(offs):
        return None
    pos = offs[idx] - 1
    return pos if pos < hs.shape[0] else None

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}
    cohort = {json.loads(l)["id"] for l in open(os.path.join(RESULTS, "validated.jsonl"))}
    out = {}
    for fam, d in FAMS:
        rpath = os.path.join(RESULTS, d, "runs.jsonl")
        if not os.path.exists(rpath):
            continue
        runs = [json.loads(l) for l in open(rpath) if "skip" not in json.loads(l)]
        acc = defaultdict(list)   # (layer, kind) -> rho values pooled over inst/point/h
        for r in runs:
            if r["id"] not in cohort or r["point"] == "late":   # well-powered only
                continue
            safe = r["id"].replace("/", "_").replace("::", "__")
            gpath = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
            ppath = os.path.join(RESULTS, d, "hs", f"{safe}__{r['point']}.npz")
            if not (os.path.exists(gpath) and os.path.exists(ppath)):
                continue
            gz = np.load(gpath, allow_pickle=True)
            pz = np.load(ppath, allow_pickle=True)
            steps = split_sentences(gold[r["id"]]["gen_text"])
            offs_g = sentence_token_offsets(tok, steps)
            cont = split_sentences(r["continuation"])
            if not cont:
                continue
            offs_p = sentence_token_offsets(tok, cont)
            nulls = []
            for npz in sorted(glob.glob(os.path.join(RESULTS, "null", "hs",
                                                     f"{safe}__{r['point']}__s*.npz"))):
                nz = np.load(npz, allow_pickle=True)
                nm = json.loads(str(nz["meta"]))
                if nm.get("correct") and "text" in nm:
                    ns = split_sentences(nm["text"])
                    if ns:
                        nulls.append((nz, sentence_token_offsets(tok, ns)))
            if len(nulls) < 2:
                continue
            si = r["sent_idx"]
            for h in range(1, H_MAX + 1):
                gi = si + h
                gpos = boundary(gz[f"layer_{LAYERS[0]}"], offs_g, gi) if gi < len(offs_g) else None
                ppos = boundary(pz[f"layer_{LAYERS[0]}"], offs_p, h - 1)
                if gpos is None or ppos is None:
                    break
                npos = [(nz, boundary(nz[f"layer_{LAYERS[0]}"], no, h - 1)) for nz, no in nulls]
                npos = [(nz, p_) for nz, p_ in npos if p_ is not None]
                if len(npos) < 2:
                    break
                g0 = gz["layer_0"][gpos]; p0 = pz["layer_0"][ppos]
                for L in LAYERS:
                    Ls = f"layer_{L}"
                    gL = gz[Ls][gpos]; pL = pz[Ls][ppos]
                    dn = np.mean([cosdist(nz[Ls][p_], gL) for nz, p_ in npos])
                    if dn > 1e-6:
                        acc[(L, "raw")].append(cosdist(pL, gL) / dn)
                    if L != 0:
                        gLp = proj_out(gL, g0); pLp = proj_out(pL, p0)
                        dnp = np.mean([cosdist(proj_out(nz[Ls][p_], nz["layer_0"][p_]), gLp)
                                       for nz, p_ in npos])
                        if dnp > 1e-6:
                            acc[(L, "proj")].append(cosdist(pLp, gLp) / dnp)
        fam_out = {}
        for (L, kind), vals in sorted(acc.items()):
            v = np.array(vals)
            fam_out[f"L{L}_{kind}"] = {"median": round(float(np.median(v)), 3),
                                       "n": int(len(v))}
        out[fam] = fam_out
        l0 = fam_out.get("L0_raw", {}).get("median")
        deep = fam_out.get(f"L{LAYERS[-1]}_raw", {}).get("median")
        deep_p = fam_out.get(f"L{LAYERS[-1]}_proj", {}).get("median")
        print(f"{fam:14s} L0={l0}  L{LAYERS[-1]}_raw={deep}  L{LAYERS[-1]}_proj={deep_p}  "
              f"(deep<L0 => representational re-entry beyond echo)")
    with open(os.path.join(RESULTS, "geometry.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote geometry.json")

if __name__ == "__main__":
    main()
