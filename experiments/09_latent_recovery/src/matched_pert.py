"""Matched-decoding calibration: perturbed continuations sampled at the null's
temperature (T=0.8, n=4), so rho_func's numerator and denominator share a decoding
distribution. Families: paraphrase (benign) vs negstep (falsehood), mid injection.

rho_matched(h) = mean_i JSD(pert_sample_i, gold) / mean_j JSD(null_j, gold)
Output: results/matched_rho.json
"""
import os, json, glob
import numpy as np
import torch
from collections import defaultdict
from common import (load_model, make_prompt_ids, sample_n, split_sentences,
                    make_paraphrase, make_negstep, LAYERS, RESULTS, DATA)
from probes import sentence_token_offsets
from perturb import injection_points
from functional_metrics import jsd, boundary_state

FINAL = str(LAYERS[-1])
H_MAX = 4
N_SAMP = 4

@torch.no_grad()
def final_states(model, full_ids, from_pos):
    out = model.forward(input_ids=full_ids.unsqueeze(0).to("cuda:0"), output_hidden_states=True)
    return out.hidden_states[-1][0, from_pos:, :].to(torch.float16).cpu().numpy()

def main():
    tok, model = load_model()
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}
    cohort = sorted({json.loads(l)["id"] for l in open(os.path.join(RESULTS, "validated.jsonl"))})
    acc = defaultdict(lambda: [[] for _ in range(H_MAX)])
    n_done = 0
    for gid in cohort:
        g = gold[gid]
        inst = data[gid]
        steps = split_sentences(g["gen_text"])
        pts = injection_points(steps)
        if pts is None:
            continue
        si = pts["mid"]
        safe = gid.replace("/", "_").replace("::", "__")
        gpath = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
        if not os.path.exists(gpath):
            continue
        gz = np.load(gpath, allow_pickle=True)
        g_hs = gz[f"layer_{FINAL}"]
        offs_g = sentence_token_offsets(tok, steps)
        nulls = []
        for npz in sorted(glob.glob(os.path.join(RESULTS, "null", "hs", f"{safe}__mid__s*.npz"))):
            nz = np.load(npz, allow_pickle=True)
            nm = json.loads(str(nz["meta"]))
            if nm.get("correct") and "text" in nm:
                ns = split_sentences(nm["text"])
                if ns:
                    nulls.append((nz[f"layer_{FINAL}"], sentence_token_offsets(tok, ns)))
        if len(nulls) < 2:
            continue
        for fam, mk in [("paraphrase", lambda: make_paraphrase(steps, si, 1)),
                        ("negstep", lambda: make_negstep(steps, si))]:
            corrupted = mk()
            if corrupted is None:
                continue
            prefix = " " + " ".join(steps[:si] + [corrupted])
            ids = make_prompt_ids(tok, inst["question"], inst["target"], answer_prefix=prefix)
            texts, outs = sample_n(tok, model, ids, n=N_SAMP, temp=0.8, max_new=192)
            for h in range(1, H_MAX + 1):
                gi = si + h
                if gi >= len(offs_g) or offs_g[gi] - 1 >= g_hs.shape[0]:
                    break
                g_state = g_hs[offs_g[gi] - 1]
                d_nulls = []
                for nv, no in nulls:
                    st = boundary_state(nv, no, h - 1)
                    if st is not None:
                        d_nulls.append(float(jsd(model, st[None], g_state[None])[0]))
                if len(d_nulls) < 2:
                    break
                dn = float(np.mean(d_nulls))
                if dn <= 1e-6:
                    break
                d_perts = []
                for txt, full in zip(texts, outs):
                    cs = split_sentences(txt)
                    if not cs:
                        continue
                    offs_p = sentence_token_offsets(tok, cs)
                    if h - 1 >= len(offs_p):
                        continue
                    p_hs = final_states(model, full, ids.shape[1])
                    st = boundary_state(p_hs, offs_p, h - 1)
                    if st is not None:
                        d_perts.append(float(jsd(model, st[None], g_state[None])[0]))
                if d_perts:
                    acc[fam][h - 1].append(float(np.mean(d_perts)) / dn)
        n_done += 1
        if n_done % 20 == 0:
            print(f"[{n_done}] instances", flush=True)
    out = {}
    for fam, byh in acc.items():
        out[fam] = {f"h{h+1}": {"median": round(float(np.median(v)), 3), "n": len(v)}
                    for h, v in enumerate(byh) if v}
    with open(os.path.join(RESULTS, "matched_rho.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print("DONE matched", flush=True)

if __name__ == "__main__":
    main()
