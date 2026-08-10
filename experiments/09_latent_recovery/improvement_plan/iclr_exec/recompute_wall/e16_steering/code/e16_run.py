#!/usr/bin/env python3
"""E16 stage runner (spec E16_STEERING.md + amendment, binding).

Stages:
  audit   -- A1 hook identity + chat-template parity, A2 backend parity,
             A4 determinism. Writes results/<model>/audit_report.json.
             Exits nonzero on A1/A4 failure. A2 failure sets
             in_harness_baselines_only (allowed by spec).
  screen  -- 61 conditions (baseline + 2 vectors x 3 layers x 10 alphas),
             n=40 onehop + n=20 unperturbed, greedy. Resume-safe.
  confirm -- top conditions (--conditions JSON) + alpha0 baseline, full
             battery in e11 schema, graded by the SHIPPED e11 stage_validate.

Usage: e16_run.py --model qwen7b --stage screen [--device cuda:0]
"""
import argparse
import json
import os
from types import SimpleNamespace

import e16_lib as L


def vec_path(model_key, vec, layer):
    return os.path.join(L.VEC_DIR, model_key, "%s_L%d.pt" % (vec, layer))


def load_vec(model_key, vec, layer):
    import torch
    return torch.load(vec_path(model_key, vec, layer), map_location="cpu")


def res_dir(model_key, *parts):
    d = os.path.join(L.RES_DIR, model_key, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def eval_rows(pool, fam, pids):
    out = []
    for pid in pids:
        m = pool.by_fam[fam].get(pid)
        if m:
            out.append((pool.programs[pid], m))
    return out


# ------------------------------------------------------------------- audit

def stage_audit(model, tok, pool):
    import torch
    rep = {"model": ARGS.model}
    screen_pids = pool.part["depth_k1_bare"]["screen"]

    # A1c: chat-template parity vs the shipped vLLM path
    try:
        import sys
        if L.TOOLING not in sys.path:
            sys.path.insert(0, L.TOOLING)
        import vllm_gen
        vtok = vllm_gen.get_tokenizer(*L.MODELS[ARGS.model][:2])
        ok = True
        for pid in screen_pids[:5]:
            um = L.user_msg_for(pool.programs[pid])
            a = L.chat_ids(tok, [{"role": "user", "content": um}])
            b = list(vllm_gen._chat_ids(vtok, [{"role": "user", "content": um}]))
            ok = ok and (a == b)
        rep["A1c_chat_parity"] = "PASS" if ok else "FAIL"
    except Exception as e:  # noqa: BLE001
        rep["A1c_chat_parity"] = "FAIL: %s" % e

    # A1a: hook identity at alpha=0 (and vec=None at alpha=8)
    v = load_vec(ARGS.model, "V1", L.LAYERS[ARGS.model][1])
    rows = eval_rows(pool, "depth_k1_bare", screen_pids[:20])
    mism = 0
    base_texts = []
    for prog, m in rows:
        ids = L.render_continue(tok, L.user_msg_for(prog), m["prefix_text"])
        t0 = L.greedy(model, tok, ids)
        with L.Steer(model, L.LAYERS[ARGS.model][1], v, 0.0):
            t1 = L.greedy(model, tok, ids)
        with L.Steer(model, L.LAYERS[ARGS.model][1], None, 8.0):
            t2 = L.greedy(model, tok, ids)
        base_texts.append((ids, t0))
        mism += (t0 != t1) + (t0 != t2)
    rep["A1a_hook_identity"] = "PASS" if mism == 0 else "FAIL: %d mismatches" % mism

    # A4: greedy + extraction determinism
    det = 0
    for ids, t0 in base_texts[:5]:
        det += (L.greedy(model, tok, ids) != t0)
    x = torch.tensor([base_texts[0][0]], device=next(model.parameters()).device)
    with torch.no_grad():
        h1 = model(x, output_hidden_states=True).hidden_states[L.LAYERS[ARGS.model][1]]
        h2 = model(x, output_hidden_states=True).hidden_states[L.LAYERS[ARGS.model][1]]
    det += 0 if torch.allclose(h1, h2) else 1
    rep["A4_determinism"] = "PASS" if det == 0 else "FAIL: %d" % det

    # A2: backend parity on n=60 eval worlds, onehop + readable, no hook
    shipped = json.load(open(os.path.join(pool.dir, "summary_tables.json")))
    a2 = {}
    for fam, cell_names in (("depth_k1_bare", ("k1_bare",)),
                            ("anchor_k0", ("k0_bare", "anchor_k0"))):
        rows = eval_rows(pool, fam, pool.part[fam]["eval"][:60])
        labs = []
        for prog, m in rows:
            ids = L.render_continue(tok, L.user_msg_for(prog), m["prefix_text"])
            labs.append(L.classify(prog, m, L.greedy(model, tok, ids))["label"])
        k = sum(1 for x in labs if x == "absorbed")
        mine, n = k / len(labs), len(labs)
        cell = None
        for cn in cell_names:
            cell = shipped.get("cells", {}).get(cn) or cell
        srate, slo, shi = None, None, None
        if cell:
            ab = cell.get("absorbed")
            if isinstance(ab, dict):
                srate = ab.get("rate")
                w = ab.get("wilson95") or [None, None]
                slo, shi = w[0], w[1]
            elif isinstance(ab, (int, float)):
                srate = ab
        inside = (slo is not None and slo <= mine <= shi)
        a2[fam] = {"mine": mine, "n": n, "shipped_rate": srate,
                   "shipped_wilson95": [slo, shi], "inside_shipped_ci": bool(inside)}
    rep["A2_backend_parity"] = a2
    rep["in_harness_baselines_only"] = not all(v["inside_shipped_ci"] for v in a2.values())

    d = res_dir(ARGS.model)
    with open(os.path.join(d, "audit_report.json"), "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps(rep, indent=1, sort_keys=True))
    hard_fail = ("FAIL" in rep["A1a_hook_identity"] or "FAIL" in rep["A4_determinism"]
                 or "FAIL" in str(rep["A1c_chat_parity"]))
    if hard_fail:
        raise SystemExit("AUDIT HARD FAIL -- do not proceed")


# ------------------------------------------------------------------- screen

def conditions(model_key):
    conds = [{"vector": "none", "layer": 0, "alpha": 0.0}]
    for vec in ("V1", "V2"):
        for layer in L.LAYERS[model_key]:
            for alpha in L.ALPHAS:
                conds.append({"vector": vec, "layer": layer, "alpha": float(alpha)})
    return conds


def cond_key(c):
    return "%s_L%d_a%g" % (c["vector"], c["layer"], c["alpha"])


def stage_screen(model, tok, pool):
    d = res_dir(ARGS.model, "screen")
    rows_path = os.path.join(d, "rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        done = {r["cond"] for r in L.read_jsonl(rows_path) if r.get("cond_done")}
    k1_screen = pool.part["depth_k1_bare"]["screen"]
    onehop = eval_rows(pool, "depth_k1_bare", k1_screen)
    ctrl = eval_rows(pool, "depth_k1_bare", k1_screen[:20])
    for c in conditions(ARGS.model):
        ck = cond_key(c)
        if ck in done:
            continue
        vec = None if c["vector"] == "none" else load_vec(ARGS.model, c["vector"], c["layer"])
        layer = c["layer"] if c["layer"] else L.LAYERS[ARGS.model][0]
        with L.Steer(model, layer, vec, c["alpha"]):
            for prog, m in onehop:
                cont = L.greedy(model, tok, L.render_continue(
                    tok, L.user_msg_for(prog), m["prefix_text"]))
                rec = L.classify(prog, m, cont)
                L.append_jsonl(rows_path, {
                    "cond": ck, "cell": "onehop", "program_id": m["program_id"],
                    "label": rec["label"], "continuation": cont, **c})
            for prog, m in ctrl:
                pre = L.unperturbed_prefix(m)
                cont = L.greedy(model, tok, L.render_continue(
                    tok, L.user_msg_for(prog), pre))
                L.append_jsonl(rows_path, {
                    "cond": ck, "cell": "control", "program_id": m["program_id"],
                    "control_valid": L.control_valid(prog, pre, cont),
                    "continuation": cont, **c})
        L.append_jsonl(rows_path, {"cond": ck, "cond_done": True})
        print("screen done:", ck, flush=True)
    summarize_screen(rows_path, d)


def summarize_screen(rows_path, d):
    rows = [r for r in L.read_jsonl(rows_path) if not r.get("cond_done")]
    by = {}
    for r in rows:
        by.setdefault(r["cond"], {"onehop": [], "control": []})[r["cell"]].append(r)
    base = by.get("none_L0_a0", {"onehop": [], "control": []})
    base_ctrl, _ = L.rate(base["control"], lambda r: r.get("control_valid"))
    summ = {}
    for ck, cells in by.items():
        ab, n = L.rate(cells["onehop"], lambda r: r["label"] == "absorbed")
        chk, _ = L.rate(cells["onehop"],
                        lambda r: r["label"] in ("silently_corrected", "flagged"))
        cv, ncv = L.rate(cells["control"], lambda r: r.get("control_valid"))
        summ[ck] = {"absorbed": ab, "checked": chk, "n": n,
                    "control_valid": cv, "n_control": ncv,
                    "control_ok": (cv is not None and base_ctrl is not None
                                   and cv >= base_ctrl - 0.10)}
    ranked = sorted((c for c in summ.items()
                     if c[0] != "none_L0_a0" and c[1]["control_ok"]),
                    key=lambda kv: (kv[1]["absorbed"] is None, kv[1]["absorbed"]))
    out = {"baseline": summ.get("none_L0_a0"), "conditions": summ,
           "top3": [k for k, _ in ranked[:3]]}
    with open(os.path.join(d, "screen_summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("BASELINE:", json.dumps(summ.get("none_L0_a0")))
    for k, v in ranked[:5]:
        print("TOP:", k, json.dumps(v))


# ------------------------------------------------------------------- confirm

def stage_confirm(model, tok, pool):
    conds = json.loads(ARGS.conditions)
    conds = [{"vector": "none", "layer": 0, "alpha": 0.0}] + conds
    fams = ("depth_k1_bare", "depth_k2_bare", "anchor_k0")
    for c in conds:
        ck = cond_key(c)
        d = res_dir(ARGS.model, "confirm", ck)
        P = L.e11.paths(d)
        man = [pool.by_fam[f0][pid] for f0 in fams
               for pid in pool.part[f0]["eval"]]
        used_pids = sorted({m["program_id"] for m in man})
        with open(P["programs"], "w") as f:
            for pid in used_pids:
                f.write(json.dumps(pool.programs[pid]) + "\n")
        with open(P["manifest"], "w") as f:
            for m in man:
                f.write(json.dumps(m) + "\n")
        if os.path.exists(P["raw"]):
            os.remove(P["raw"])
        vec = None if c["vector"] == "none" else load_vec(ARGS.model, c["vector"], c["layer"])
        layer = c["layer"] if c["layer"] else L.LAYERS[ARGS.model][0]
        ctrl_path = os.path.join(d, "control_rows.jsonl")
        if os.path.exists(ctrl_path):
            os.remove(ctrl_path)
        with L.Steer(model, layer, vec, c["alpha"]):
            for m in man:
                prog = pool.programs[m["program_id"]]
                ids = L.render_continue(tok, L.user_msg_for(prog), m["prefix_text"])
                for roll in range(0, 5):
                    cont = (L.greedy(model, tok, ids) if roll == 0
                            else L.sampled(model, tok, ids, roll))
                    L.append_jsonl(P["raw"], {
                        "run_id": L.e11.cont_run_id(m["program_id"], m["family"],
                                                    roll, m["seed"]),
                        "program_id": m["program_id"], "condition": m["family"],
                        "k": m["k"], "rollout": roll, "continuation": cont,
                        "failed_generation": False, "created_at": ""})
            for pid in pool.part["depth_k1_bare"]["eval"]:
                m = pool.by_fam["depth_k1_bare"][pid]
                prog = pool.programs[pid]
                pre = L.unperturbed_prefix(m)
                cont = L.greedy(model, tok, L.render_continue(
                    tok, L.user_msg_for(prog), pre))
                L.append_jsonl(ctrl_path, {
                    "program_id": pid,
                    "control_valid": L.control_valid(prog, pre, cont),
                    "continuation": cont})
        L.e11.stage_validate(SimpleNamespace(out_dir=d, R=4))
        print("confirm generated+validated:", ck, flush=True)
    summarize_confirm(conds)


def summarize_confirm(conds):
    out = {}
    for c in conds:
        ck = cond_key(c)
        d = res_dir(ARGS.model, "confirm", ck)
        val = L.read_jsonl(L.e11.paths(d)["validated"])
        ctrl = L.read_jsonl(os.path.join(d, "control_rows.jsonl"))
        cells = {}
        for fam in ("depth_k1_bare", "depth_k2_bare", "anchor_k0"):
            rows = [r for r in val if r["family"] == fam]
            ab, n = L.rate(rows, lambda r: r["label"] == "absorbed")
            chk, _ = L.rate(rows, lambda r: r["label"] in ("silently_corrected", "flagged"))
            unres, _ = L.rate(rows, lambda r: r["label"] == "unresolved")
            k = sum(1 for r in rows if r["label"] == "absorbed")
            cells[fam] = {"absorbed": ab, "checked": chk, "unresolved": unres,
                          "n": n, "wilson95": L.wilson(k, n)}
        cv, ncv = L.rate(ctrl, lambda r: r.get("control_valid"))
        out[ck] = {"cells": cells, "control_valid": cv, "n_control": ncv}
    base = out.get("none_L0_a0")
    for ck, v in out.items():
        if ck == "none_L0_a0" or not base:
            continue
        b1, c1 = base["cells"]["depth_k1_bare"], v["cells"]["depth_k1_bare"]
        breach = (
            c1["absorbed"] is not None and b1["absorbed"] is not None
            and c1["absorbed"] <= b1["absorbed"] - 0.15
            and c1["wilson95"][1] < b1["wilson95"][0]
            and (c1["checked"] - b1["checked"]) >= 0.15
            and v["control_valid"] >= base["control_valid"] - 0.10
            and v["cells"]["anchor_k0"]["absorbed"]
                <= base["cells"]["anchor_k0"]["absorbed"] + 0.10)
        v["breach_all_conjuncts"] = bool(breach)
    d = res_dir(ARGS.model, "confirm")
    with open(os.path.join(d, "confirm_summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))


# ------------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--stage", required=True, choices=["audit", "screen", "confirm"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conditions", default="[]",
                    help='confirm only: JSON list of {"vector","layer","alpha"}')
    ARGS = ap.parse_args()
    pool = L.load_pool(ARGS.model)
    model, tok = L.load_model(ARGS.model, ARGS.device)
    {"audit": stage_audit, "screen": stage_screen,
     "confirm": stage_confirm}[ARGS.stage](model, tok, pool)
