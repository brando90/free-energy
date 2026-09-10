#!/usr/bin/env python3
"""E16: nonlinear (MLP) probe sweep across layers — is the error-presence
signal present EARLY in a non-linear form?

Design guards for the small-n regime:
  * n expanded to ALL manifested k1 programs (~150 pairs, 300 examples) —
    probing is read-only diagnosis, so the extraction/eval split need not bind.
  * 5-fold cross-validation, folds stratified by delta sign (the skew lesson).
  * Small MLP (hidden 64, GELU), weight decay 1e-2, fixed epochs — capacity
    kept low on purpose; we are mapping an accessibility-vs-capacity frontier,
    not maximizing AUC.
  * 5 seeds per (layer, probe): report mean +/- sd.
  * Selectivity control: same MLP trained on label-SHUFFLED folds; its holdout
    AUC calibrates what this capacity can memorize (~0.5 expected).
  * Linear probe (logistic head) run on the identical folds for
    apples-to-apples.

Layers: 1, 4, 8, 10, 12, 14, 16, 18, 20, 24, 27 (span-mean representation).
Writes results/<model>/probe_mlp.json. Usage: --model qwen7b [--device]
"""
import argparse
import json
import os

import e16_lib as L

LAYERS = [1, 4, 8, 10, 12, 14, 16, 18, 20, 24, 27]
SEEDS = [0, 1, 2, 3, 4]
FOLDS = 5
HID = 64
EPOCHS = 200
WD = 1e-2
LR = 1e-3


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


def train_probe(Xtr, ytr, Xte, kind, seed):
    import torch
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if kind == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(d, HID), torch.nn.GELU(),
                                  torch.nn.Linear(HID, 1))
    else:
        net = torch.nn.Linear(d, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = lossf(net(Xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(Xte).squeeze(-1).tolist()


def main():
    import torch
    pool = L.load_pool(ARGS.model)
    fam = pool.by_fam["depth_k1_bare"]
    pids = sorted(fam)
    model, tok = L.load_model(ARGS.model, ARGS.device)
    dev = next(model.parameters()).device

    reps = {l: {} for l in LAYERS}
    deltas = {}
    kept = []
    for pid in pids:
        m = fam[pid]
        prog = pool.programs.get(pid)
        if not prog:
            continue
        um = L.user_msg_for(prog)
        pair = {}
        for side, prefix in (("pert", m["prefix_text"]),
                             ("clean", L.unperturbed_prefix(m))):
            ids = L.render_continue(tok, um, prefix)
            a, b = L.planted_line_span(tok, um, prefix)
            x = torch.tensor([ids], device=dev)
            with torch.no_grad():
                hs = model(x, output_hidden_states=True).hidden_states
            pair[side] = {l: hs[l][0][a:b].float().cpu().mean(dim=0) for l in LAYERS}
        for l in LAYERS:
            reps[l][pid] = (pair["pert"][l], pair["clean"][l])
        deltas[pid] = m["planted_value"] - m["true_value"]
        kept.append(pid)
    print("pairs collected:", len(kept))

    # stratified folds by delta sign
    pos = [p for p in kept if deltas[p] > 0]
    neg = [p for p in kept if deltas[p] < 0]
    folds = [[] for _ in range(FOLDS)]
    for i, p in enumerate(pos):
        folds[i % FOLDS].append(p)
    for i, p in enumerate(neg):
        folds[i % FOLDS].append(p)

    import statistics
    out = {"model": ARGS.model, "n_pairs": len(kept), "layers": {}}
    print("layer | linear AUC (cv)   | MLP AUC (cv)      | shuffled-MLP (ctrl)")
    for l in LAYERS:
        res = {"linear": [], "mlp": [], "shuffled": []}
        for seed in SEEDS:
            import random
            rng = random.Random(seed)
            scores = {"linear": {}, "mlp": {}, "shuffled": {}}
            for k in range(FOLDS):
                te = folds[k]
                tr = [p for j, f in enumerate(folds) if j != k for p in f]
                Xtr, ytr = [], []
                for p in tr:
                    Xtr.append(reps[l][p][0]); ytr.append(1.0)
                    Xtr.append(reps[l][p][1]); ytr.append(0.0)
                ysh = ytr[:]
                rng.shuffle(ysh)
                Xte = []
                for p in te:
                    Xte.append(reps[l][p][0])
                    Xte.append(reps[l][p][1])
                import torch as T
                Xtr_t = T.stack(Xtr); Xte_t = T.stack(Xte)
                mu, sd = Xtr_t.mean(0), Xtr_t.std(0) + 1e-6
                Xtr_t = (Xtr_t - mu) / sd
                Xte_t = (Xte_t - mu) / sd
                for kind, yy in (("linear", ytr), ("mlp", ytr), ("shuffled", ysh)):
                    sc = train_probe(Xtr_t, T.tensor(yy), Xte_t,
                                     "mlp" if kind != "linear" else "linear", seed)
                    for i, p in enumerate(te):
                        scores[kind][p] = (sc[2 * i], sc[2 * i + 1])
            for kind in res:
                res[kind].append(auc([scores[kind][p][0] for p in kept],
                                     [scores[kind][p][1] for p in kept]))
        st = {k: {"mean": statistics.mean(v), "sd": statistics.stdev(v)}
              for k, v in res.items()}
        out["layers"][l] = st
        print("%5d | %.3f +/- %.3f   | %.3f +/- %.3f   | %.3f +/- %.3f"
              % (l, st["linear"]["mean"], st["linear"]["sd"],
                 st["mlp"]["mean"], st["mlp"]["sd"],
                 st["shuffled"]["mean"], st["shuffled"]["sd"]))
    d = os.path.join(L.RES_DIR, ARGS.model)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "probe_mlp.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ARGS = ap.parse_args()
    main()
