#!/usr/bin/env python3
"""E16 Amendment 2, stage 1: all-layer extraction + the V3 error-presence probe.

V3 contrast (the gate): prefixes identical except the site line's value —
planted (error-present) vs the model's own true line (error-absent), both
under continue-framing. Representations at EVERY layer: final token and
site-line span mean. Held-out AUC per layer answers "is the corrupted value
linearly detectable at all?"

Also harvests all-layer V1 (probe-vs-continue, final token) and V2
(readable-vs-computed contexts, span mean) vectors for the multi-layer arm.

Writes vectors/<model>/ml/{V1,V2,V3}_L<l>.pt + ml_manifest.json (AUC curves).
Usage: e16_extract_ml.py --model qwen7b [--device cuda:0]
"""
import argparse
import json
import os

import e16_lib as L


def auc(pos, neg):
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    t = len(pos) * len(neg)
    return (wins + 0.5 * ties) / t if t else None


def collect_all(model, tok, ids_list, span_list, n_layers):
    import torch
    out = {l: {"final": [], "span": []} for l in range(1, n_layers + 1)}
    dev = next(model.parameters()).device
    for ids, span in zip(ids_list, span_list):
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            hs = model(x, output_hidden_states=True).hidden_states
        for l in range(1, n_layers + 1):
            h = hs[l][0].float().cpu()
            out[l]["final"].append(h[-1])
            if span is not None:
                a, b = span
                out[l]["span"].append(h[a:b].mean(dim=0))
            else:
                out[l]["span"].append(h[-1])
    return out


def diff_auc(pos, neg, n_tr=40):
    import torch
    v = torch.stack(pos[:n_tr]).mean(0) - torch.stack(neg[:n_tr]).mean(0)
    u = v / (v.norm() + 1e-8)
    return v, auc([float(u @ h) for h in pos[n_tr:]],
                  [float(u @ h) for h in neg[n_tr:]])


def main():
    import torch
    pool = L.load_pool(ARGS.model)
    pids_k1 = pool.part["depth_k1_bare"]["extract"]
    pids_k0 = pool.part["anchor_k0"]["extract"]
    model, tok = L.load_model(ARGS.model, ARGS.device)
    n_layers = model.config.num_hidden_layers

    cont_i, cont_s, clean_i, clean_s, probe_i = [], [], [], [], []
    for pid in pids_k1:
        prog = pool.programs[pid]
        um = L.user_msg_for(prog)
        m = pool.by_fam["depth_k1_bare"][pid]
        pert = m["prefix_text"]
        clean = L.unperturbed_prefix(m)
        cont_i.append(L.render_continue(tok, um, pert))
        cont_s.append(L.planted_line_span(tok, um, pert))
        clean_i.append(L.render_continue(tok, um, clean))
        clean_s.append(L.planted_line_span(tok, um, clean))
        probe_i.append(L.render_probe(tok, um, pert, L.probe_question(prog, m)))
    read_i, read_s = [], []
    for pid in pids_k0:
        prog = pool.programs[pid]
        um = L.user_msg_for(prog)
        m = pool.by_fam["anchor_k0"][pid]
        read_i.append(L.render_continue(tok, um, m["prefix_text"]))
        read_s.append(L.planted_line_span(tok, um, m["prefix_text"]))

    A_cont = collect_all(model, tok, cont_i, cont_s, n_layers)
    A_clean = collect_all(model, tok, clean_i, clean_s, n_layers)
    A_probe = collect_all(model, tok, probe_i, [None] * len(probe_i), n_layers)
    A_read = collect_all(model, tok, read_i, read_s, n_layers)

    vdir = os.path.join(L.VEC_DIR, ARGS.model, "ml")
    os.makedirs(vdir, exist_ok=True)
    man = {"model": ARGS.model, "n_layers": n_layers, "seed": L.SEED,
           "pids_k1": pids_k1, "pids_k0": pids_k0, "vectors": {}}
    print("layer |  V3(final)  V3(span)  |  V1(final)  V2(span)")
    for l in range(1, n_layers + 1):
        v3f, a3f = diff_auc(A_cont[l]["final"], A_clean[l]["final"])
        v3s, a3s = diff_auc(A_cont[l]["span"], A_clean[l]["span"])
        v1, a1 = diff_auc(A_probe[l]["final"], A_cont[l]["final"])
        v2, a2 = diff_auc(A_read[l]["span"], A_cont[l]["span"])
        # V3 steering vector = the span variant (site-local signal); sign:
        # +alpha amplifies error-presence
        for name, v, a in (("V1", v1, a1), ("V2", v2, a2), ("V3", v3s, a3s)):
            p = os.path.join(vdir, "%s_L%d.pt" % (name, l))
            torch.save(v, p)
            man["vectors"]["%s_L%d" % (name, l)] = {
                "auc": a, "norm": float(v.norm()),
                "sha256": L.sha256_bytes(v.numpy().tobytes())}
        man["vectors"]["V3final_L%d" % l] = {"auc": a3f}
        print("%5d |   %.3f      %.3f    |   %.3f      %.3f" % (l, a3f, a3s, a1, a2))
    v3_max = max(man["vectors"]["V3_L%d" % l]["auc"] for l in range(1, n_layers + 1))
    v3f_max = max(man["vectors"]["V3final_L%d" % l]["auc"] for l in range(1, n_layers + 1))
    man["v3_gate"] = {"span_max_auc": v3_max, "final_max_auc": v3f_max,
                      "pass": bool(max(v3_max, v3f_max) >= 0.8)}
    with open(os.path.join(vdir, "ml_manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print("\nV3 GATE:", "PASS" if man["v3_gate"]["pass"] else "FAIL",
          "| max span AUC %.3f | max final AUC %.3f" % (v3_max, v3f_max))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ARGS = ap.parse_args()
    main()
