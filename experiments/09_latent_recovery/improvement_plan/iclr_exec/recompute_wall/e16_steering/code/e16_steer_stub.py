#!/usr/bin/env python3
"""E16 Amendment 3: stub-position steering arm (the last rung).

Directions from saved stub representations, extraction partition only,
sign-balanced group weighting. Steers during continuation at L26/L27 (+S28
exploratory). Standard screen: n=40 onehop + 20 controls, greedy,
resume-safe. Usage: --model qwen7b [--device cuda:0]
"""
import argparse
import json
import os

import torch

import e16_lib as L
from e16_screen_ml import MultiSteer

ALPHAS = [1.0, 2.0, 4.0, 8.0, -1.0, -2.0, -4.0, -8.0]
FRACS = [0.5, 1.0]
S28_ALPHAS = [2.0, 8.0, -2.0, -8.0]


def balanced_direction(reps, layer, pids, deltas):
    pos = [p for p in pids if deltas[p] > 0]
    neg = [p for p in pids if deltas[p] < 0]
    def grp(g):
        return (torch.stack([reps[layer][p][0] for p in g]).mean(0)
                - torch.stack([reps[layer][p][1] for p in g]).mean(0))
    v = 0.5 * grp(pos) + 0.5 * grp(neg)
    return v


def main():
    pool = L.load_pool(ARGS.model)
    blob = torch.load(os.path.join(L.RES_DIR, ARGS.model, "probe_neutral_reps.pt"),
                      map_location="cpu")
    reps, deltas, kept = blob["reps"], blob["deltas"], blob["kept"]
    ext = [p for p in pool.part["depth_k1_bare"]["extract"] if p in set(kept)]
    n_pos = sum(1 for p in ext if deltas[p] > 0)
    print(f"direction extraction: {len(ext)} pairs ({n_pos} pos-delta)")
    V = {l: balanced_direction(reps, l, ext, deltas) for l in (26, 27, 28)}
    for l, v in V.items():
        print(f"  S{l}: norm={float(v.norm()):.1f}")

    model, tok = L.load_model(ARGS.model, ARGS.device)
    fam = pool.by_fam["depth_k1_bare"]

    conds = [{"name": "baseline", "vecs": {}, "alpha": 0.0, "frac": None}]
    for nm, vecs in (("S26", {26: V[26]}), ("S2627", {26: V[26], 27: V[27]})):
        for a in ALPHAS:
            conds.append({"name": nm, "vecs": vecs, "alpha": a, "frac": None})
        for fr in FRACS:
            conds.append({"name": nm, "vecs": vecs, "alpha": None, "frac": fr})
    for a in S28_ALPHAS:
        conds.append({"name": "S28x", "vecs": {28: V[28]}, "alpha": a, "frac": None})

    d = os.path.join(L.RES_DIR, ARGS.model, "steer_stub")
    os.makedirs(d, exist_ok=True)
    rows_path = os.path.join(d, "rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        done = {r["cond"] for r in L.read_jsonl(rows_path) if r.get("cond_done")}

    k1_screen = pool.part["depth_k1_bare"]["screen"]
    onehop = [(pool.programs[p], fam[p]) for p in k1_screen if p in fam]
    ctrl = onehop[:20]

    for c in conds:
        ck = "%s_a%s_f%s" % (c["name"], c["alpha"], c["frac"])
        if ck in done:
            continue
        with MultiSteer(model, c["vecs"], alpha=c["alpha"], frac=c["frac"]):
            for prog, m in onehop:
                cont = L.greedy(model, tok, L.render_continue(
                    tok, L.user_msg_for(prog), m["prefix_text"]))
                rec = L.classify(prog, m, cont)
                L.append_jsonl(rows_path, {"cond": ck, "cell": "onehop",
                                           "program_id": m["program_id"],
                                           "label": rec["label"],
                                           "continuation": cont})
            for prog, m in ctrl:
                pre = L.unperturbed_prefix(m)
                cont = L.greedy(model, tok, L.render_continue(
                    tok, L.user_msg_for(prog), pre))
                L.append_jsonl(rows_path, {"cond": ck, "cell": "control",
                                           "program_id": m["program_id"],
                                           "control_valid": L.control_valid(prog, pre, cont),
                                           "continuation": cont})
        L.append_jsonl(rows_path, {"cond": ck, "cond_done": True})
        print("done:", ck, flush=True)

    rows = [r for r in L.read_jsonl(rows_path) if not r.get("cond_done")]
    by = {}
    for r in rows:
        by.setdefault(r["cond"], {"onehop": [], "control": []})[r["cell"]].append(r)
    base = by.get("baseline_a0.0_fNone", {"onehop": [], "control": []})
    base_ctrl, _ = L.rate(base["control"], lambda r: r.get("control_valid"))
    summ = {}
    for ck, cells in by.items():
        ab, n = L.rate(cells["onehop"], lambda r: r["label"] == "absorbed")
        chk, _ = L.rate(cells["onehop"],
                        lambda r: r["label"] in ("silently_corrected", "flagged"))
        cv, _ = L.rate(cells["control"], lambda r: r.get("control_valid"))
        summ[ck] = {"absorbed": ab, "checked": chk, "n": n, "control_valid": cv,
                    "control_ok": (cv is not None and base_ctrl is not None
                                   and cv >= base_ctrl - 0.10)}
    with open(os.path.join(d, "steer_stub_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, sort_keys=True)
    print("BASELINE:", json.dumps(summ.get("baseline_a0.0_fNone")))
    ranked = sorted((kv for kv in summ.items()
                     if kv[0] != "baseline_a0.0_fNone" and kv[1]["control_ok"]),
                    key=lambda kv: (kv[1]["absorbed"] is None, kv[1]["absorbed"]))
    for k, v in ranked[:6]:
        print("TOP:", k, json.dumps(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ARGS = ap.parse_args()
    main()
