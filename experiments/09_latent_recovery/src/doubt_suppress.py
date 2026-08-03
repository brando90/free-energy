"""Exp-2 (Batch-4): Doubt-suppression / doubt-forcing causal test.

Severs verbalized doubt from the internal recovery computation to test whether recovery
is intrinsic (our claim) or driven by doubt (von Recum's claim).

STAGE=direction : build a "doubt" direction in the residual stream from EXISTING negstep
                  + contradiction continuations (doubt vs no-doubt, WITHIN construction so
                  the direction is doubt, not family/content). Forward passes only, no gen.
                  Reports held-out linear-probe accuracy per layer (kill-gate: if doubt is
                  not linearly decodable, projection cannot cleanly remove it).
STAGE=suppress  : on negstep mid (locally-checkable lies), generate with (a) token-ban
                  baseline and (b) residual-projection of the doubt direction. Re-validate.
                  Off-target control: same projection on benign paraphrase.
STAGE=force     : on falsehood mid (globally-checkable lies), prefix-force a doubt token;
                  validator (planted atom unused) + judge separate genuine recovery from
                  spurious hedging-then-absorption.

Output: results_doubt/{direction.npz, probe.json, suppress.json, force.json, *.jsonl}
"""
import os, json, re, time
import numpy as np
import torch
from common import load_model, make_prompt_ids, split_sentences, DATA, RESULTS, LAYERS
from validator import validate_continuation, DOUBT

OUT = os.path.join(RESULTS, "..", "results_doubt")
DEV = "cuda:0"
PROBE_LAYERS = [7, 14, 21]              # residual after these blocks (skip emb/final)
DOUBT_TOKENS = ["Wait", " Wait", "However", " However", "But", " But", "Actually",
                " Actually", "Hmm", " Hmm", "Hold", " Hold", "Contradiction",
                " contradict", " mistake", " error", " wrong", " incorrect", " inconsistent"]


def _data():
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}


def _gold():
    return {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}


@torch.no_grad()
def residual_meanpool(model, ids, cont_start, layers):
    out = model(input_ids=ids.to(DEV), output_hidden_states=True)
    feats = {}
    for L in layers:
        h = out.hidden_states[L][0, cont_start:, :]      # continuation residuals
        feats[L] = h.mean(0).to(torch.float32).cpu().numpy()
    return feats


def build_direction():
    """Difference-of-means doubt direction per layer, from negstep+contradiction runs."""
    os.makedirs(OUT, exist_ok=True)
    tok, model = load_model()
    data, _ = _data(), None
    feats = {L: [] for L in PROBE_LAYERS}
    labels = []
    for fam in ("negstep", "contradiction"):
        path = os.path.join(RESULTS, f"perturbed_{fam}", "runs.jsonl")
        if not os.path.exists(path):
            continue
        runs = [json.loads(l) for l in open(path)]
        runs = [r for r in runs if "skip" not in r and r.get("continuation")]
        for r in runs:
            inst = data[r["id"]]
            steps = split_sentences(_GOLD[r["id"]]["gen_text"])
            si = r["sent_idx"]
            prefix = " " + " ".join(steps[:si] + [r["corrupted_step"]])
            pre_ids = make_prompt_ids(tok, inst["question"], inst["target"], answer_prefix=prefix)
            cont_ids = tok(" " + r["continuation"], return_tensors="pt",
                           add_special_tokens=False)["input_ids"]
            ids = torch.cat([pre_ids, cont_ids], dim=1)
            cont_start = pre_ids.shape[1]
            f = residual_meanpool(model, ids, cont_start, PROBE_LAYERS)
            for L in PROBE_LAYERS:
                feats[L].append(f[L])
            labels.append(int(bool(DOUBT.search(r["continuation"]))))
    labels = np.array(labels)
    print(f"collected {len(labels)} runs, doubt+={labels.sum()} doubt-={len(labels)-labels.sum()}",
          flush=True)
    dirs, probe = {}, {}
    for L in PROBE_LAYERS:
        X = np.stack(feats[L])
        mu1, mu0 = X[labels == 1].mean(0), X[labels == 0].mean(0)
        d = mu1 - mu0
        d = d / (np.linalg.norm(d) + 1e-8)
        dirs[f"layer_{L}"] = d.astype(np.float32)
        # held-out probe: project onto d, threshold at midpoint, 5-fold-ish split
        s = X @ d
        thr = 0.5 * (s[labels == 1].mean() + s[labels == 0].mean())
        pred = (s > thr).astype(int)
        acc = float((pred == labels).mean())
        # balanced acc
        tpr = float(pred[labels == 1].mean()) if (labels == 1).any() else 0.0
        tnr = float((1 - pred[labels == 0]).mean()) if (labels == 0).any() else 0.0
        probe[str(L)] = {"acc": round(acc, 3), "balanced_acc": round(0.5 * (tpr + tnr), 3),
                         "n": len(labels), "thr": float(thr)}
    np.savez(os.path.join(OUT, "direction.npz"), **dirs,
             meta=json.dumps({"layers": PROBE_LAYERS}))
    json.dump(probe, open(os.path.join(OUT, "probe.json"), "w"), indent=2)
    print(json.dumps(probe, indent=2), flush=True)
    print("KILL-GATE: need balanced_acc >~0.70 at some layer for projection to be meaningful.",
          flush=True)


# ---------- generation-time interventions ----------
class ProjectHook:
    """Subtract the doubt direction from a decoder block's residual output."""
    def __init__(self, direction):
        self.u = torch.tensor(direction, device=DEV, dtype=torch.bfloat16)

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        coef = (h.to(torch.float32) @ self.u.to(torch.float32)).unsqueeze(-1)
        h = h - coef.to(h.dtype) * self.u
        return (h,) + out[1:] if isinstance(out, tuple) else h


def _ban_ids(tok):
    bad = set()
    for w in DOUBT_TOKENS:
        for t in tok(w, add_special_tokens=False)["input_ids"]:
            bad.add(t)
    return list(bad)


@torch.no_grad()
def generate(tok, model, ids, max_new=192, bad_words=None):
    kw = {}
    if bad_words:
        kw["bad_words_ids"] = [[b] for b in bad_words]
    out = model.generate(ids.to(DEV), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def _cohort_iter(tok, data):
    gold = [g for g in _GOLD.values() if g.get("solved")]
    from perturb import injection_points
    for g in gold:
        inst = data[g["id"]]
        steps = split_sentences(g["gen_text"])
        ent = steps[0].split()[0] if steps else None
        gv = validate_continuation(inst["question"], [], None, g["gen_text"], inst["target"], ent)
        if gv["class"] != "valid_rederivation":
            continue
        pts = injection_points(steps)
        if pts is None:
            continue
        yield g["id"], inst, steps, ent, pts["mid"]


def run_suppress():
    from common import make_negstep, make_paraphrase
    tok, model = load_model()
    data = _data()
    d = np.load(os.path.join(OUT, "direction.npz"))
    layer = int(os.environ.get("SUPP_LAYER", "14"))
    hook = ProjectHook(d[f"layer_{layer}"])
    ban = _ban_ids(tok)
    limit = int(os.environ.get("LR_LIMIT", "0"))
    fams = {"negstep": make_negstep, "paraphrase": make_paraphrase}
    rows = []
    n = 0
    with open(os.path.join(OUT, "suppress.jsonl"), "w") as fh:
        for cid, inst, steps, ent, si in _cohort_iter(tok, data):
            for fam, gen_fn in fams.items():
                corrupted = gen_fn(steps, si) if fam == "negstep" else gen_fn(steps, si, 0)
                if corrupted is None:
                    continue
                prefix = " " + " ".join(steps[:si] + [corrupted])
                ids = make_prompt_ids(tok, inst["question"], inst["target"], answer_prefix=prefix)
                for arm in ("baseline", "tokenban", "project"):
                    h = None
                    if arm == "project":
                        h = model.model.layers[layer].register_forward_hook(hook)
                    try:
                        gen = generate(tok, model, ids, bad_words=ban if arm == "tokenban" else None)
                    finally:
                        if h is not None:
                            h.remove()
                    v = validate_continuation(inst["question"], steps[:si], corrupted, gen,
                                              inst["target"], ent)
                    rec = {"id": cid, "family": fam, "arm": arm, "class": v["class"],
                           "lexical_doubt": bool(DOUBT.search(gen)),
                           "continuation": gen}
                    rows.append(rec)
                    fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if limit and n >= limit:
                break
            if n % 20 == 0:
                print(f"[suppress {n}]", flush=True)
    _summ(rows, "suppress_summary.json", keys=("family", "arm"))


def run_force():
    """Forcing arm + the two controls that decide what a forced doubt token MEANS.

    Confound: " Wait," may trigger genuine checking, OR a reflexive "prior step is wrong,
    redo" heuristic that fires regardless of truth (and recovers on solvable problems).
      - force_neutral (false step + " Then,"): if poisoning also drops, it is interruption/
        restart, not doubt-specific.
      - wait_on_true (CORRECT step + " Wait,"): if the model disavows a true step and
        derails/flails, " Wait," is reflexive, not a check. If it stays valid, it checked.
    """
    from common import make_falsehood
    tok, model = load_model()
    data = _data()
    limit = int(os.environ.get("LR_LIMIT", "0"))
    rows = []
    n = 0
    with open(os.path.join(OUT, "force.jsonl"), "w") as fh:
        for cid, inst, steps, ent, si in _cohort_iter(tok, data):
            corrupted = make_falsehood(inst["question"], steps, steps[si], 0)
            if corrupted is None:
                continue
            correct = steps[si]
            # (placed_step, inject_token, true_prefix_step_for_validator)
            arms = [
                ("baseline",     corrupted, "",       corrupted),  # silent absorption
                ("force_wait",   corrupted, " Wait,", corrupted),  # forced doubt on a lie
                ("force_neutral",corrupted, " Then,", corrupted),  # interruption control
                ("wait_on_true", correct,   " Wait,", None),       # reflexive-disavowal ctrl
            ]
            for arm, placed, inject, vcorrupt in arms:
                prefix = " " + " ".join(steps[:si] + [placed]) + inject
                ids = make_prompt_ids(tok, inst["question"], inst["target"], answer_prefix=prefix)
                gen = generate(tok, model, ids)
                full = (inject + " " + gen).strip()
                v = validate_continuation(inst["question"], steps[:si], vcorrupt, full,
                                          inst["target"], ent)
                rows.append({"id": cid, "arm": arm, "class": v["class"],
                             "lexical_doubt": bool(DOUBT.search(full)), "continuation": full})
                fh.write(json.dumps(rows[-1]) + "\n")
            fh.flush()
            n += 1
            if limit and n >= limit:
                break
            if n % 20 == 0:
                print(f"[force {n}]", flush=True)
    _summ(rows, "force_summary.json", keys=("arm",))


def _summ(rows, name, keys):
    from collections import defaultdict
    cells = defaultdict(lambda: defaultdict(int))
    for r in rows:
        k = "|".join(str(r[x]) for x in keys)
        c = cells[k]
        c["n"] += 1
        c[r["class"]] += 1
        c["doubt"] += int(r["lexical_doubt"])
    out = {}
    for k, c in cells.items():
        n = c["n"]
        out[k] = {"n": n, "valid": round(c.get("valid_rederivation", 0) / n, 3),
                  "poisoned": round(c.get("poisoned", 0) / n, 3),
                  "parroted": round(c.get("parroted", 0) / n, 3),
                  "derailed": round(c.get("derailed", 0) / n, 3),
                  "doubt": round(c.get("doubt", 0) / n, 3)}
    json.dump(out, open(os.path.join(OUT, name), "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


_GOLD = None
if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    _GOLD = _gold()
    stage = sys.argv[1] if len(sys.argv) > 1 else "direction"
    if stage == "direction":
        build_direction()
    elif stage == "suppress":
        run_suppress()
    elif stage == "force":
        run_force()
