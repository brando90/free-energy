#!/usr/bin/env python3
"""E16 probe integrity check (post-hoc, prompted by the delta-sign skew).

The extraction set has a 2:1 skew toward planted < true (40 neg / 20 pos),
so the late-layer "error-presence" AUC could in principle lean on numeric
MAGNITUDE rather than wrongness. Decisive tests, layers {14, 25, 26, 27, 28}:

  1. Reproduce the original full-sample diff-in-means AUC.
  2. Sign-BALANCED train (14 pos-delta + 14 neg-delta programs), AUC on the
     held-out rest.
  3. The signature test: with the balanced direction, AUC computed separately
     on positive-delta vs negative-delta holdout pairs. A WRONGNESS direction
     separates both subsets the same way; a MAGNITUDE direction separates
     them in OPPOSITE directions (AUC ~high on one, ~low/inverted on the other).

Also: MultiSteer alpha=0 byte-identity audit (full hook set vs no hooks,
10 worlds) — the multi-layer class never got the A1-style audit.

Writes results/<model>/probe_check.json. Usage: --model qwen7b [--device]
"""
import argparse
import json
import os

import e16_lib as L
from e16_screen_ml import MultiSteer

LAYERS = [14, 25, 26, 27, 28]


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


def main():
    import torch
    pool = L.load_pool(ARGS.model)
    pids = pool.part["depth_k1_bare"]["extract"]
    fam = pool.by_fam["depth_k1_bare"]
    deltas = {p: fam[p]["planted_value"] - fam[p]["true_value"] for p in pids}
    model, tok = L.load_model(ARGS.model, ARGS.device)
    dev = next(model.parameters()).device

    reps = {l: {"pert": {}, "clean": {}} for l in LAYERS}
    for pid in pids:
        prog = pool.programs[pid]
        um = L.user_msg_for(prog)
        m = fam[pid]
        for side, prefix in (("pert", m["prefix_text"]),
                             ("clean", L.unperturbed_prefix(m))):
            ids = L.render_continue(tok, um, prefix)
            a, b = L.planted_line_span(tok, um, prefix)
            x = torch.tensor([ids], device=dev)
            with torch.no_grad():
                hs = model(x, output_hidden_states=True).hidden_states
            for l in LAYERS:
                reps[l][side][pid] = hs[l][0][a:b].float().cpu().mean(dim=0)

    pos_pids = [p for p in pids if deltas[p] > 0]
    neg_pids = [p for p in pids if deltas[p] < 0]
    tr = pos_pids[:14] + neg_pids[:14]
    ho = [p for p in pids if p not in set(tr)]
    ho_pos = [p for p in ho if deltas[p] > 0]
    ho_neg = [p for p in ho if deltas[p] < 0]
    print(f"balanced train: {len(tr)} (14+14) | holdout: {len(ho)} "
          f"({len(ho_pos)} pos-delta, {len(ho_neg)} neg-delta)")

    out = {"model": ARGS.model, "layers": {}, "deltas": deltas,
           "train": tr, "holdout": ho}
    print("layer | orig-AUC | balanced-AUC | AUC(pos-delta) | AUC(neg-delta)")
    for l in LAYERS:
        R = reps[l]
        import torch as T
        # 1. original recipe: first-40-pids train
        v0 = (T.stack([R["pert"][p] for p in pids[:40]]).mean(0)
              - T.stack([R["clean"][p] for p in pids[:40]]).mean(0))
        u0 = v0 / (v0.norm() + 1e-8)
        a_orig = auc([float(u0 @ R["pert"][p]) for p in pids[40:]],
                     [float(u0 @ R["clean"][p]) for p in pids[40:]])
        # 2. balanced train
        v = (T.stack([R["pert"][p] for p in tr]).mean(0)
             - T.stack([R["clean"][p] for p in tr]).mean(0))
        u = v / (v.norm() + 1e-8)
        sc = {p: (float(u @ R["pert"][p]), float(u @ R["clean"][p])) for p in ho}
        a_bal = auc([sc[p][0] for p in ho], [sc[p][1] for p in ho])
        a_pos = auc([sc[p][0] for p in ho_pos], [sc[p][1] for p in ho_pos])
        a_neg = auc([sc[p][0] for p in ho_neg], [sc[p][1] for p in ho_neg])
        out["layers"][l] = {"auc_orig": a_orig, "auc_balanced": a_bal,
                            "auc_holdout_posdelta": a_pos,
                            "auc_holdout_negdelta": a_neg,
                            "scores": {p: sc[p] for p in ho}}
        print("%5d |  %.3f   |    %.3f     |     %.3f      |     %.3f"
              % (l, a_orig, a_bal, a_pos, a_neg))

    # ---- MultiSteer alpha=0 byte-identity audit ----
    vdir = os.path.join(L.VEC_DIR, ARGS.model, "ml")
    vecs = {l: torch.load(os.path.join(vdir, "V3_L%d.pt" % l), map_location="cpu")
            for l in range(1, model.config.num_hidden_layers + 1)}
    mism = 0
    k1_screen = pool.part["depth_k1_bare"]["screen"][:10]
    for pid in k1_screen:
        prog = pool.programs[pid]
        m = fam.get(pid) or pool.by_fam["depth_k1_bare"][pid]
        ids = L.render_continue(tok, L.user_msg_for(prog), m["prefix_text"])
        t0 = L.greedy(model, tok, ids)
        with MultiSteer(model, vecs, alpha=0.0):
            t1 = L.greedy(model, tok, ids)
        mism += (t0 != t1)
    out["multisteer_alpha0_identity"] = "PASS" if mism == 0 else "FAIL: %d/10" % mism
    print("MultiSteer alpha=0 identity:", out["multisteer_alpha0_identity"])

    d = os.path.join(L.RES_DIR, ARGS.model)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "probe_check.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("VERDICT:",
          "WRONGNESS (both subsets separate same direction)"
          if all(v["auc_holdout_posdelta"] >= 0.8 and v["auc_holdout_negdelta"] >= 0.8
                 for l, v in out["layers"].items() if l >= 27)
          else "SUSPECT — inspect probe_check.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ARGS = ap.parse_args()
    main()
