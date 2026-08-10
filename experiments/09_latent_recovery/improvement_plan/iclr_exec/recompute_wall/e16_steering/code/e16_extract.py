#!/usr/bin/env python3
"""E16 vector extraction + audit A3 (spec E16_STEERING.md, binding).

V1 "verification mode": continue-framing vs probe-framing on the SAME
perturbed k1 prefix; representation = final-token residual.
V2 "checking behavior": readable (anchor_k0) vs computed (depth_k1_bare)
perturbed prefixes under continue-framing; representation = mean over the
planted-line token span.

Both at layers LAYERS[model]. Diff-in-means (positive side = the checking
side). A3: vector from the first 40 extraction programs must separate the
held-out last 20 with AUC >= 0.8, else that vector FAILS and may not be used.

Usage: e16_extract.py --model qwen7b [--device cuda:0]
Writes: vectors/<model>/{V1,V2}_L<L>.pt + extract_manifest.json
"""
import argparse
import json
import os

import e16_lib as L


def collect(model, tok, ids_list, span_list):
    """Forward each ids; return dict layer -> {'final': [h], 'span': [h]}."""
    import torch
    out = {l: {"final": [], "span": []} for l in L.LAYERS[ARGS.model]}
    dev = next(model.parameters()).device
    for ids, span in zip(ids_list, span_list):
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            hs = model(x, output_hidden_states=True).hidden_states
        for l in out:
            h = hs[l][0].float().cpu()          # layer l == hidden_states[l]
            out[l]["final"].append(h[-1])
            if span is not None:
                a, b = span
                out[l]["span"].append(h[a:b].mean(dim=0))
            else:
                out[l]["span"].append(h[-1])
    return out


def auc(scores_pos, scores_neg):
    wins = ties = 0
    for p in scores_pos:
        for n in scores_neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    tot = len(scores_pos) * len(scores_neg)
    return (wins + 0.5 * ties) / tot if tot else None


def diff_and_auc(pos, neg):
    """pos/neg: list of 1-D tensors (aligned by program). Vector from first
    40 pairs, AUC on the last 20 held out."""
    import torch
    n_tr = 40
    v = (torch.stack(pos[:n_tr]).mean(0) - torch.stack(neg[:n_tr]).mean(0))
    u = v / (v.norm() + 1e-8)
    sp = [float(u @ h) for h in pos[n_tr:]]
    sn = [float(u @ h) for h in neg[n_tr:]]
    return v, auc(sp, sn)


def main():
    pool = L.load_pool(ARGS.model)
    pids_k1 = pool.part["depth_k1_bare"]["extract"]
    pids_k0 = pool.part["anchor_k0"]["extract"]
    assert len(pids_k1) >= 50 and len(pids_k0) >= 50, \
        ("extraction pools too small", len(pids_k1), len(pids_k0))
    model, tok = L.load_model(ARGS.model, ARGS.device)

    # V1 sides are paired on the k1 programs; V2 sides are UNPAIRED
    # (readable side from the anchor_k0 cohort) -- diff-in-means needs no
    # pairing, and train/holdout splits are within-side.
    cont_ids, cont_span, probe_ids, read_ids, read_span = [], [], [], [], []
    for pid in pids_k1:
        prog = pool.programs[pid]
        um = L.user_msg_for(prog)
        m1 = pool.by_fam["depth_k1_bare"][pid]
        cont_ids.append(L.render_continue(tok, um, m1["prefix_text"]))
        cont_span.append(L.planted_line_span(tok, um, m1["prefix_text"]))
        probe_ids.append(L.render_probe(tok, um, m1["prefix_text"],
                                        L.probe_question(prog, m1)))
    for pid in pids_k0:
        prog = pool.programs[pid]
        um = L.user_msg_for(prog)
        m0 = pool.by_fam["anchor_k0"][pid]
        read_ids.append(L.render_continue(tok, um, m0["prefix_text"]))
        read_span.append(L.planted_line_span(tok, um, m0["prefix_text"]))

    acts_cont = collect(model, tok, cont_ids, cont_span)
    acts_probe = collect(model, tok, probe_ids, [None] * len(probe_ids))
    acts_read = collect(model, tok, read_ids, read_span)

    import torch
    vdir = os.path.join(L.VEC_DIR, ARGS.model)
    os.makedirs(vdir, exist_ok=True)
    man = {"model": ARGS.model, "n_extract_k1": len(pids_k1),
           "n_extract_k0": len(pids_k0), "pids_k1": pids_k1, "pids_k0": pids_k0,
           "seed": L.SEED, "vectors": {}}
    for l in L.LAYERS[ARGS.model]:
        # V1: probe(final) - continue(final); positive side = checking
        v1, a1 = diff_and_auc(acts_probe[l]["final"], acts_cont[l]["final"])
        # V2: readable(span) - computed(span)
        v2, a2 = diff_and_auc(acts_read[l]["span"], acts_cont[l]["span"])
        for name, v, a in (("V1", v1, a1), ("V2", v2, a2)):
            path = os.path.join(vdir, "%s_L%d.pt" % (name, l))
            torch.save(v, path)
            man["vectors"]["%s_L%d" % (name, l)] = {
                "path": path, "auc_holdout": a, "a3_pass": bool(a is not None and a >= 0.8),
                "norm": float(v.norm()),
                "sha256": L.sha256_bytes(v.numpy().tobytes())}
            print("%s_L%-3d AUC=%.3f  %s" % (name, l, a, "PASS" if a >= 0.8 else "FAIL"))
    with open(os.path.join(vdir, "extract_manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    n_pass = sum(1 for v in man["vectors"].values() if v["a3_pass"])
    print("A3: %d/%d vector-layer combos pass AUC>=0.8" % (n_pass, len(man["vectors"])))
    if n_pass == 0:
        raise SystemExit("A3 FAILED for every vector: fix extraction before spending")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ARGS = ap.parse_args()
    main()
