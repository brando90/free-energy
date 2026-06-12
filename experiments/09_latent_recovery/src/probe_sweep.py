"""Probe design sweep — gold data only (measurement iteration, no perturbed peeking).

Axes:
  position: period token | mean-pool over sentence tokens
  boundaries: all sentences | entity-fact sentences only
  epochs: 200 | 800
Reports held-out accuracy per layer per variant; picks the winner for probes.py.
"""
import os, json
import numpy as np
import torch
from collections import defaultdict
from common import LAYERS, RESULTS, DATA, split_sentences
from validator import parse_fact
from probes import sentence_token_offsets, labels_for_sentences, train_probe

def build(tok, pool, fact_only):
    data_ids = set()
    gold = [json.loads(l) for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))]
    gold = [g for g in gold if g.get("solved")]
    X = {str(L): [] for L in LAYERS}
    y, inst_ids = [], []
    for g in gold:
        safe = g["id"].replace("/", "_").replace("::", "__")
        path = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        hs = {k.replace("layer_", ""): z[k] for k in z.files if k.startswith("layer_")}
        T = hs[str(LAYERS[0])].shape[0]
        steps = split_sentences(g["gen_text"])
        offs = sentence_token_offsets(tok, steps)
        if not (offs[-1] <= T and T - offs[-1] <= 3):
            continue
        entity = steps[0].split()[0]
        lasts, _ = labels_for_sentences(steps, entity)
        start = 0
        for j, e in enumerate(offs):
            is_fact = parse_fact(steps[j], entity) is not None
            if fact_only and not is_fact:
                start = e
                continue
            for L in LAYERS:
                seg = hs[str(L)][start:e] if pool == "mean" else hs[str(L)][e - 1: e]
                X[str(L)].append(seg.mean(0))
            y.append(lasts[j])
            inst_ids.append(g["id"])
            start = e
    return ({k: np.stack(v).astype(np.float32) for k, v in X.items()},
            np.array(y), np.array(inst_ids))

def main():
    from transformers import AutoTokenizer
    from common import MODEL
    tok = AutoTokenizer.from_pretrained(MODEL)
    results = {}
    for pool in ["period", "mean"]:
        for fact_only in [False, True]:
            X, y, inst_ids = build(tok, pool, fact_only)
            classes = np.array(sorted(set(y)))
            cls = {c: i for i, c in enumerate(classes)}
            yi = np.array([cls[c] for c in y])
            rng = np.random.RandomState(0)
            uids = np.array(sorted(set(inst_ids)))
            rng.shuffle(uids)
            test_ids = set(uids[: max(1, len(uids) // 5)])
            te = np.array([i in test_ids for i in inst_ids]); tr = ~te
            for epochs in [200, 800]:
                accs = {}
                for L in LAYERS:
                    _, a, _ = train_probe(X[str(L)][tr], yi[tr], X[str(L)][te], yi[te],
                                          len(classes), epochs=epochs)
                    accs[str(L)] = round(a, 4)
                key = f"pool={pool},fact_only={fact_only},epochs={epochs}"
                results[key] = {"n_samples": int(len(yi)), "best": max(accs.values()), **accs}
                print(key, "->", results[key], flush=True)
    with open(os.path.join(RESULTS, "probes", "sweep.json"), "w") as fh:
        json.dump(results, fh, indent=2)

if __name__ == "__main__":
    main()
