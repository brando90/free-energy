#!/usr/bin/env python3
"""E16 gate experiment: neutral-position probe sweep, ALL layers.

Why not "span minus value tokens": pert/clean prefixes differ ONLY in the
value tokens, so pre-value positions have bit-identical activations by
causality — AUC 0.5 by construction. The valid control probes positions whose
OWN tokens are identical across conditions but which attend to the value: a
stub appended after the planted line, consisting of the next line's header
("line <j+1>: <code>;", identical text in both conditions, value not
included). Any signal there is computed, propagated wrongness — no surface
digit-reading is possible.

Probes: linear + MLP(64) + shuffled-label control, every layer 1..n_layers,
n=150 pairs, 5-fold CV stratified by delta sign, 5 seeds. Representations
(stub-mean) are SAVED for later MDL analysis.

Writes results/<model>/probe_neutral.json + probe_neutral_reps.pt
Usage: --model qwen7b [--device cuda:0]
"""
import argparse
import json
import os

import e16_lib as L

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


def train_probe(Xtr, ytr, Xte, kind, seed, dev):
    import torch
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if kind == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(d, HID), torch.nn.GELU(),
                                  torch.nn.Linear(HID, 1)).to(dev)
    else:
        net = torch.nn.Linear(d, 1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xtr, ytr, Xte = Xtr.to(dev), ytr.to(dev), Xte.to(dev)
    for _ in range(EPOCHS):
        opt.zero_grad()
        lossf(net(Xtr).squeeze(-1), ytr).backward()
        opt.step()
    with __import__("torch").no_grad():
        return net(Xte).squeeze(-1).cpu().tolist()


def main():
    import torch
    pool = L.load_pool(ARGS.model)
    fam = pool.by_fam["depth_k1_bare"]
    pids = sorted(fam)
    model, tok = L.load_model(ARGS.model, ARGS.device)
    dev = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    LAYERS = list(range(1, n_layers + 1))

    reps = {l: {} for l in LAYERS}
    deltas = {}
    kept = []
    for pid in pids:
        m = fam[pid]
        prog = pool.programs.get(pid)
        if not prog:
            continue
        j = m["site_line"]                      # 1-based planted line number
        if j >= len(prog["stmt_texts"]):
            continue
        stub = "line %d: %s;" % (j + 1, prog["stmt_texts"][j])
        um = L.user_msg_for(prog)
        pair = {}
        ok = True
        for side, prefix in (("pert", m["prefix_text"]),
                             ("clean", L.unperturbed_prefix(m))):
            base = prefix.rstrip("\n")
            ids_wo = L.render_continue(tok, um, base)
            ids_w = L.render_continue(tok, um, base + "\n" + stub)
            a, b = len(ids_wo), len(ids_w)
            if not (0 < a < b):
                ok = False
                break
            x = torch.tensor([ids_w], device=dev)
            with torch.no_grad():
                hs = model(x, output_hidden_states=True).hidden_states
            pair[side] = {l: hs[l][0][a:b].float().cpu().mean(dim=0) for l in LAYERS}
        if not ok:
            continue
        for l in LAYERS:
            reps[l][pid] = (pair["pert"][l], pair["clean"][l])
        deltas[pid] = m["planted_value"] - m["true_value"]
        kept.append(pid)
    print("pairs collected:", len(kept))
    torch.save({"reps": reps, "deltas": deltas, "kept": kept},
               os.path.join(L.RES_DIR, ARGS.model, "probe_neutral_reps.pt"))

    pos = [p for p in kept if deltas[p] > 0]
    neg = [p for p in kept if deltas[p] < 0]
    folds = [[] for _ in range(FOLDS)]
    for i, p in enumerate(pos):
        folds[i % FOLDS].append(p)
    for i, p in enumerate(neg):
        folds[i % FOLDS].append(p)

    import statistics, random
    out = {"model": ARGS.model, "n_pairs": len(kept), "design": "neutral-stub",
           "layers": {}}
    print("layer | linear AUC (cv)   | MLP AUC (cv)      | shuffled-MLP (ctrl)")
    for l in LAYERS:
        res = {"linear": [], "mlp": [], "shuffled": []}
        for seed in SEEDS:
            rng = random.Random(seed)
            scores = {"linear": {}, "mlp": {}, "shuffled": {}}
            for k in range(FOLDS):
                te = folds[k]
                tr = [p for jf, f in enumerate(folds) if jf != k for p in f]
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
                Xtr_t, Xte_t = T.stack(Xtr), T.stack(Xte)
                mu, sd = Xtr_t.mean(0), Xtr_t.std(0) + 1e-6
                Xtr_t, Xte_t = (Xtr_t - mu) / sd, (Xte_t - mu) / sd
                for kind, yy in (("linear", ytr), ("mlp", ytr), ("shuffled", ysh)):
                    sc = train_probe(Xtr_t, T.tensor(yy), Xte_t,
                                     "mlp" if kind != "linear" else "linear",
                                     seed, dev)
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
    with open(os.path.join(L.RES_DIR, ARGS.model, "probe_neutral.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ARGS = ap.parse_args()
    main()
