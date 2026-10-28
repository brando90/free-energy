#!/usr/bin/env python3
"""E16 Amendment 2, stage 2: multi-layer steering screen.

Layer sets {all, early third, mid third, late third}; per-layer diff-in-means
vectors; unit alpha in +/-{1,2,4}; plus per-layer norm-fractions {0.25, 0.5}
on the all-layer set. Vectors: V1 + (V3 if the probe gate passed). Same
worlds, DV, and controls as the single-layer screen; resume-safe.

Usage: e16_screen_ml.py --model qwen7b [--device cuda:0]
"""
import argparse
import json
import os

import e16_lib as L

ALPHAS = [1.0, 2.0, 4.0, -1.0, -2.0, -4.0]
FRACS = [0.25, 0.5]


class MultiSteer:
    def __init__(self, model, vecs, alpha=None, frac=None):
        import torch
        p = next(model.parameters())
        self.entries = []
        for l, v in vecs.items():
            v32 = v.to(dtype=torch.float32)
            n = float(v32.norm()) + 1e-8
            a = (frac * n) if frac is not None else alpha
            delta = (a * (v32 / n)).to(device=p.device, dtype=p.dtype)
            self.entries.append((model.model.layers[l - 1], delta))
        self.handles = []

    @staticmethod
    def _mk(delta):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] != 1:
                return output
            hs = hs + delta
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return hook

    def __enter__(self):
        for mod, delta in self.entries:
            self.handles.append(mod.register_forward_hook(self._mk(delta)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def main():
    import torch
    pool = L.load_pool(ARGS.model)
    model, tok = L.load_model(ARGS.model, ARGS.device)
    n_layers = model.config.num_hidden_layers
    vdir = os.path.join(L.VEC_DIR, ARGS.model, "ml")
    man = json.load(open(os.path.join(vdir, "ml_manifest.json")))
    vec_names = ["V1"] + (["V3"] if man["v3_gate"]["pass"] else [])
    print("vectors in play:", vec_names, "| v3_gate:", man["v3_gate"])

    third = n_layers // 3
    SETS = {"all": list(range(1, n_layers + 1)),
            "early": list(range(1, third + 1)),
            "mid": list(range(third + 1, 2 * third + 1)),
            "late": list(range(2 * third + 1, n_layers + 1))}

    def load_vecs(name, layers):
        return {l: torch.load(os.path.join(vdir, "%s_L%d.pt" % (name, l)),
                              map_location="cpu") for l in layers}

    conds = [{"vector": "none", "set": "none", "alpha": 0.0, "frac": None}]
    for vn in vec_names:
        for sn in SETS:
            for a in ALPHAS:
                conds.append({"vector": vn, "set": sn, "alpha": a, "frac": None})
        for fr in FRACS:
            conds.append({"vector": vn, "set": "all", "alpha": None, "frac": fr})

    d = os.path.join(L.RES_DIR, ARGS.model, "screen_ml")
    os.makedirs(d, exist_ok=True)
    rows_path = os.path.join(d, "rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        done = {r["cond"] for r in L.read_jsonl(rows_path) if r.get("cond_done")}

    k1 = pool.part["depth_k1_bare"]["screen"]
    onehop = [(pool.programs[p], pool.by_fam["depth_k1_bare"][p]) for p in k1
              if p in pool.by_fam["depth_k1_bare"]]
    ctrl = onehop[:20]

    for c in conds:
        ck = "%s_%s_a%s_f%s" % (c["vector"], c["set"], c["alpha"], c["frac"])
        if ck in done:
            continue
        if c["vector"] == "none":
            steer = MultiSteer(model, {}, alpha=0.0)
        else:
            steer = MultiSteer(model, load_vecs(c["vector"], SETS[c["set"]]),
                               alpha=c["alpha"], frac=c["frac"])
        with steer:
            for prog, m in onehop:
                cont = L.greedy(model, tok, L.render_continue(
                    tok, L.user_msg_for(prog), m["prefix_text"]))
                rec = L.classify(prog, m, cont)
                L.append_jsonl(rows_path, {"cond": ck, "cell": "onehop",
                                           "program_id": m["program_id"],
                                           "label": rec["label"],
                                           "continuation": cont, **c})
            for prog, m in ctrl:
                pre = L.unperturbed_prefix(m)
                cont = L.greedy(model, tok, L.render_continue(
                    tok, L.user_msg_for(prog), pre))
                L.append_jsonl(rows_path, {"cond": ck, "cell": "control",
                                           "program_id": m["program_id"],
                                           "control_valid": L.control_valid(prog, pre, cont),
                                           "continuation": cont, **c})
        L.append_jsonl(rows_path, {"cond": ck, "cond_done": True})
        print("done:", ck, flush=True)

    rows = [r for r in L.read_jsonl(rows_path) if not r.get("cond_done")]
    by = {}
    for r in rows:
        by.setdefault(r["cond"], {"onehop": [], "control": []})[r["cell"]].append(r)
    base_ck = "none_none_a0.0_fNone"
    base_ctrl, _ = L.rate(by.get(base_ck, {}).get("control", []),
                          lambda r: r.get("control_valid"))
    summ = {}
    for ck, cells in by.items():
        ab, n = L.rate(cells["onehop"], lambda r: r["label"] == "absorbed")
        chk, _ = L.rate(cells["onehop"],
                        lambda r: r["label"] in ("silently_corrected", "flagged"))
        cv, _ = L.rate(cells["control"], lambda r: r.get("control_valid"))
        summ[ck] = {"absorbed": ab, "checked": chk, "n": n, "control_valid": cv,
                    "control_ok": (cv is not None and base_ctrl is not None
                                   and cv >= base_ctrl - 0.10)}
    ranked = sorted((kv for kv in summ.items() if kv[0] != base_ck and kv[1]["control_ok"]),
                    key=lambda kv: (kv[1]["absorbed"] is None, kv[1]["absorbed"]))
    out = {"baseline": summ.get(base_ck), "conditions": summ,
           "top5": [k for k, _ in ranked[:5]]}
    with open(os.path.join(d, "screen_ml_summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("BASELINE:", json.dumps(summ.get(base_ck)))
    for k, v in ranked[:5]:
        print("TOP:", k, json.dumps(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ARGS = ap.parse_args()
    main()
