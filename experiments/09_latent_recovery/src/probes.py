"""Probe suite: decode the running task state from hidden states.

Probe target (per sentence boundary, per layer): the category of the most recently
derived entity-fact ("last-derived category"), multi-class over the shared PrOntoQA
nonsense-category vocabulary + "none".

Outputs:
  results/probes/probe_accuracy.json      held-out accuracy per layer (kill-cond: <0.8)
  results/probes/probe_{layer}.pt         trained probe weights
  results/probes/agreement.json           perturbed-vs-gold decoded-state re-entry, by
                                          family x injection point x horizon (in sentences)
  results/probes/precommit.json           future-state (next-derived-category) decodability
                                          at the injection boundary — precommitment control

Alignment: sentence -> token offsets via incremental re-tokenization; instances with
token-length mismatch beyond tolerance are skipped (counted).
"""
import os, json, re, glob
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from common import LAYERS, RESULTS, DATA, split_sentences
from validator import parse_fact

PROBE_DIR = os.path.join(RESULTS, "probes")
TOL = 3
SEED = 0

def sentence_token_offsets(tok, sentences):
    """Cumulative token offsets (end-of-sentence positions) when sentences are
    tokenized with a leading space, matching ' ' + ' '.join(sentences)."""
    offs, cum = [], 0
    for s in sentences:
        n = len(tok(" " + s, add_special_tokens=False)["input_ids"])
        cum += n
        offs.append(cum)
    return offs

def labels_for_sentences(sentences, entity):
    """Per sentence: last-derived category so far ('none' if no cat fact yet),
    and next-derived category after this sentence (for the precommitment probe)."""
    last, lasts = "none", []
    cats = []
    for s in sentences:
        f = parse_fact(s, entity)
        if f and f[1][0] == "cat":
            last = f[1][1]
        lasts.append(last)
        cats.append(f[1][1] if (f and f[1][0] == "cat") else None)
    nexts = []
    for i in range(len(sentences)):
        nxt = "none"
        for j in range(i + 1, len(sentences)):
            if cats[j]:
                nxt = cats[j]
                break
        nexts.append(nxt)
    return lasts, nexts

def build_gold_dataset(tok):
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = [json.loads(l) for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))]
    gold = [g for g in gold if g.get("solved")]
    X = {str(L): [] for L in LAYERS}
    y_last, y_next, inst_ids = [], [], []
    skipped = 0
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
        if not (offs[-1] <= T and T - offs[-1] <= TOL):
            skipped += 1
            continue
        entity = steps[0].split()[0]
        lasts, nexts = labels_for_sentences(steps, entity)
        for j, e in enumerate(offs):
            for L in LAYERS:
                X[str(L)].append(hs[str(L)][e - 1])
            y_last.append(lasts[j])
            y_next.append(nexts[j])
            inst_ids.append(g["id"])
    return ({k: np.stack(v).astype(np.float32) for k, v in X.items()},
            np.array(y_last), np.array(y_next), np.array(inst_ids), skipped)

def train_probe(Xtr, ytr, Xte, yte, n_classes, epochs=200, lr=1e-3, wd=1e-4):
    torch.manual_seed(SEED)
    dev = "cuda:3" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, device=dev)
    Xte_t = torch.tensor((Xte - mu) / sd, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    probe = nn.Linear(Xtr.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(probe(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (probe(Xte_t).argmax(-1).cpu().numpy() == yte).mean()
    return probe.cpu(), float(acc), (mu, sd)

def decode(probe, norm_stats, vecs, classes):
    mu, sd = norm_stats
    with torch.no_grad():
        idx = probe(torch.tensor((vecs.astype(np.float32) - mu) / sd)).argmax(-1).numpy()
    return classes[idx]

def main():
    os.makedirs(PROBE_DIR, exist_ok=True)
    from transformers import AutoTokenizer
    from common import MODEL
    tok = AutoTokenizer.from_pretrained(MODEL)

    X, y_last, y_next, inst_ids, skipped = build_gold_dataset(tok)
    classes = np.array(sorted(set(y_last) | set(y_next)))
    cls_idx = {c: i for i, c in enumerate(classes)}
    yl = np.array([cls_idx[c] for c in y_last])
    yn = np.array([cls_idx[c] for c in y_next])

    # split by INSTANCE (no leakage across boundaries of the same rollout)
    rng = np.random.RandomState(SEED)
    uids = np.array(sorted(set(inst_ids)))
    rng.shuffle(uids)
    test_ids = set(uids[: max(1, len(uids) // 5)])
    te = np.array([i in test_ids for i in inst_ids])
    tr = ~te

    acc, acc_next, probes = {}, {}, {}
    for L in LAYERS:
        Ls = str(L)
        p, a, ns = train_probe(X[Ls][tr], yl[tr], X[Ls][te], yl[te], len(classes))
        probes[Ls] = (p, ns)
        acc[Ls] = round(a, 4)
        p2, a2, ns2 = train_probe(X[Ls][tr], yn[tr], X[Ls][te], yn[te], len(classes))
        acc_next[Ls] = round(a2, 4)
        torch.save({"last": p.state_dict(), "mu": ns[0], "sd": ns[1],
                    "next": p2.state_dict(), "mu2": ns2[0], "sd2": ns2[1],
                    "classes": classes.tolist()},
                   os.path.join(PROBE_DIR, f"probe_{Ls}.pt"))
    out = {"n_boundaries": int(len(yl)), "n_instances": int(len(uids)),
           "n_classes": int(len(classes)), "skipped_align": skipped,
           "majority_baseline": round(float(np.mean(yl[te] == np.bincount(yl[tr]).argmax())), 4),
           "last_derived_acc_by_layer": acc,
           "next_derived_acc_by_layer (precommit signal)": acc_next}
    with open(os.path.join(PROBE_DIR, "probe_accuracy.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))

    # ---- agreement / re-entry on perturbed rollouts, best layer ----
    bestL = max(acc, key=lambda k: acc[k])
    probe, ns = probes[bestL]
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}
    agreement = {}
    for fam, dirname in [("wrong", "perturbed"), ("distractor", "perturbed_distractor"),
                         ("contradiction", "perturbed_contradiction")]:
        rpath = os.path.join(RESULTS, dirname, "runs.jsonl")
        if not os.path.exists(rpath):
            continue
        runs = [json.loads(l) for l in open(rpath) if "skip" not in json.loads(l)]
        re_entry = defaultdict(list)   # point -> list of per-horizon binary re-entry
        for r in runs:
            safe = r["id"].replace("/", "_").replace("::", "__")
            ppath = os.path.join(RESULTS, dirname, "hs", f"{safe}__{r['point']}.npz")
            gpath = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
            if not (os.path.exists(ppath) and os.path.exists(gpath)):
                continue
            g = gold[r["id"]]
            steps = split_sentences(g["gen_text"])
            entity = steps[0].split()[0]
            # gold remaining decoded path (true labels suffice: probes trained to match them)
            lasts, _ = labels_for_sentences(steps, entity)
            si = r["sent_idx"]
            gold_remaining = set(lasts[si:]) - {"none"}
            # perturbed continuation decoded states
            z = np.load(ppath, allow_pickle=True)
            hs = z[f"layer_{bestL}"]
            cont_sents = split_sentences(r["continuation"])
            if not cont_sents:
                continue
            offs = sentence_token_offsets(tok, cont_sents)
            T = hs.shape[0]
            offs = [o for o in offs if o - 1 < T]
            if not offs:
                continue
            vecs = np.stack([hs[o - 1] for o in offs])
            dec = decode(probe, ns, vecs, classes)
            hor = [int(d in gold_remaining) for d in dec]
            re_entry[r["point"]].append(hor)
        agreement[fam] = {}
        for p, lists in re_entry.items():
            H = max(len(x) for x in lists)
            M = np.full((len(lists), H), np.nan)
            for i, x in enumerate(lists):
                M[i, : len(x)] = x
            agreement[fam][p] = {
                "n": len(lists),
                "re_entry_rate_by_sentence_horizon": [round(float(v), 4) for v in np.nanmean(M, 0)[:10]],
            }
    with open(os.path.join(PROBE_DIR, "agreement.json"), "w") as fh:
        json.dump({"best_layer": bestL, "agreement": agreement}, fh, indent=2)
    print(json.dumps({"best_layer": bestL, "agreement": agreement}, indent=2))

if __name__ == "__main__":
    main()
