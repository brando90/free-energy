"""Decoding-robustness satellite: replay ALL EXPC_POLARITY_CONTROL_FULL rows
under vLLM and measure cell-level rate deltas vs the stored HF outputs.

Read-only w.r.t. src/ and results/. Writes only under improvement_plan/tooling/replay_full/.
Reuses the experiment's own validator + scoring (validator.validate_continuation,
the validate() field mapping, metric_value, summarize_cell, wilson) so the vLLM
rows are scored byte-for-byte the same way the HF rows were.
"""
import json
import math
import os
import sys
import time
from collections import defaultdict, Counter

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
SRC = os.path.join(EXP, "src")
TOOLING = os.path.join(EXP, "improvement_plan", "tooling")
RESULTS = os.path.join(EXP, "results", "EXPC_POLARITY_CONTROL_FULL")
OUTDIR = os.path.join(TOOLING, "replay_full")
os.makedirs(OUTDIR, exist_ok=True)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_NEW_TOKENS = 192

sys.path.insert(0, SRC)
sys.path.insert(0, TOOLING)

import validator as V                     # noqa: E402
import expc_polarity_control as E         # noqa: E402  (stdlib-only import, safe)
from vllm_gen import (                     # noqa: E402
    render_prompt_ids, generate_continuations, get_tokenizer,
    eos_token_ids, expc_answer_prefix, load_prompt_block,
)

CONDITIONS = list(E.CONDITIONS)
POINTS = list(E.POINTS)
METRICS = ["valid_recovery", "doubt", "poisoning", "unparsed"]


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def wilson_lohi(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def paired_delta_ci(pairs, z=1.96):
    """Newcombe (1998) paired score interval for p2 - p1 (vLLM - HF).
    pairs: list of (x, y) with x=HF 0/1, y=vLLM 0/1."""
    e = f = g = h = 0
    for x, y in pairs:
        if x and y:
            e += 1
        elif x and not y:
            f += 1
        elif (not x) and y:
            g += 1
        else:
            h += 1
    n = e + f + g + h
    if n == 0:
        return None
    p1 = (e + f) / n          # HF rate
    p2 = (e + g) / n          # vLLM rate
    diff = p2 - p1
    l1, u1 = wilson_lohi(e + f, n, z)
    l2, u2 = wilson_lohi(e + g, n, z)
    A = (e + f) * (g + h) * (e + g) * (f + h)
    phi = (e * h - f * g) / math.sqrt(A) if A > 0 else 0.0
    lower = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    upper = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return {
        "hf_rate": round(p1, 4), "vllm_rate": round(p2, 4), "delta": round(diff, 4),
        "delta_wilson95": [round(lower, 4), round(upper, 4)],
        "n": n, "hf_count": e + f, "vllm_count": e + g,
        "hf_wilson95": [round(x, 4) if x is not None else None for x in wilson_lohi(e + f, n, z)],
        "vllm_wilson95": [round(x, 4) if x is not None else None for x in wilson_lohi(e + g, n, z)],
        "mcnemar_table": {"both": e, "hf_only": f, "vllm_only": g, "neither": h},
        "discordant": f + g,
    }


def score_vllm_row(m, continuation):
    """Reproduce validate() field mapping exactly for a vLLM continuation."""
    closure_v = V.validate_continuation(
        m["question"], m["prefix_steps"], m["injected_statement"],
        continuation, m["target"], m["entity"],
    )
    pm = bool(m["poisoning_measurable"])
    return {
        "run_id": m["run_id"],
        "problem_id": m["problem_id"],
        "condition": m["condition"],
        "locality": m["locality"],
        "polarity": m["polarity"],
        "injection_position": m["injection_position"],
        "poisoning_measurable": pm,
        "class": closure_v["class"],
        "closure_class": closure_v["class"],
        "valid_recovery": closure_v["class"] == "valid_rederivation",
        "doubt": bool(closure_v.get("acknowledged")),
        "verbalized_doubt": bool(closure_v.get("acknowledged")),
        "poisoning": (closure_v["class"] == "poisoned") if pm else None,
        "n_poisoned": closure_v.get("n_poisoned"),
        "n_invalid": closure_v.get("n_invalid"),
        "n_unparsed": closure_v.get("n_unparsed"),
        "final_ok": closure_v.get("final_ok"),
    }


def main():
    t_start = time.time()
    validated = read_jsonl(os.path.join(RESULTS, "validated_outputs.jsonl"))
    manifest = {r["run_id"]: r for r in read_jsonl(os.path.join(RESULTS, "manifest.jsonl"))}
    run_ids = [r["run_id"] for r in validated]           # preserve order
    hf_by_id = {r["run_id"]: r for r in validated}
    assert all(rid in manifest for rid in run_ids)

    instruction, fewshot = load_prompt_block(os.path.join(RESULTS, "run_metadata.json"))
    tok = get_tokenizer(MODEL, REV)

    # --- reconstruct prompts exactly as the 20-row pilot / EXPC generate stage did ---
    rows, order = [], []
    for rid in run_ids:
        m = manifest[rid]
        pre = expc_answer_prefix(m)                       # " " + " ".join(prefix_steps + [injected_statement])
        ids = render_prompt_ids(tok, m["question"], m["target"], pre, instruction, fewshot)
        rows.append({"prompt_token_ids": ids})
        order.append(rid)

    print(f"[replay] reconstructed {len(rows)} prompts; launching vLLM ...", flush=True)
    t_gen = time.time()
    outs = generate_continuations(MODEL, rows, max_new_tokens=MAX_NEW_TOKENS, revision=REV)
    gen_secs = time.time() - t_gen
    assert len(outs) == len(order)
    print(f"[replay] vLLM generation done in {gen_secs:.1f}s", flush=True)

    # --- validate every vLLM continuation with the experiment's own scorer ---
    vllm_by_id = {}
    exact_text = 0
    cont_records = []
    for rid, out in zip(order, outs):
        m = manifest[rid]
        text = out["text"]
        vrow = score_vllm_row(m, text)
        vllm_by_id[rid] = vrow
        hf = hf_by_id[rid]
        exact = (text == hf.get("continuation", ""))
        exact_text += int(exact)
        cont_records.append({
            "run_id": rid, "condition": m["condition"], "injection_position": m["injection_position"],
            "problem_id": m["problem_id"], "poisoning_measurable": vrow["poisoning_measurable"],
            "vllm_continuation": text, "finish_reason": out["finish_reason"],
            "vllm_class": vrow["class"], "hf_class": hf["class"],
            "vllm_valid_recovery": vrow["valid_recovery"], "hf_valid_recovery": bool(hf.get("valid_recovery")),
            "vllm_doubt": vrow["doubt"], "hf_doubt": bool(hf.get("doubt")),
            "vllm_poisoning": vrow["poisoning"], "hf_poisoning": hf.get("poisoning"),
            "exact_text_match": exact, "class_agree": vrow["class"] == hf["class"],
        })

    # --- HF-pipeline self-check: reproduce summary_tables pooled rates from validated_outputs ---
    st = json.load(open(os.path.join(RESULTS, "summary_tables.json")))
    st_pc = st["metrics"]["pooled_by_condition"]
    selfcheck = {}
    for cond in CONDITIONS:
        crows = [hf_by_id[r] for r in run_ids if hf_by_id[r]["condition"] == cond]
        cell = E.summarize_cell(crows, 0)
        selfcheck[cond] = {}
        for metric in METRICS:
            got = cell[metric]["rate"]
            want = st_pc[cond][metric]["rate"]
            selfcheck[cond][metric] = {"recomputed": got, "summary_tables": want,
                                       "match": got == want}

    # --- cell-level HF vs vLLM ---
    def cell_rows(pred):
        ids = [r for r in run_ids if pred(manifest[r])]
        return ids

    def build_cell(ids):
        cell = {}
        for metric in METRICS:
            pairs = []
            for rid in ids:
                hf = hf_by_id[rid]; vl = vllm_by_id[rid]
                xv = E.metric_value(hf, metric)
                yv = E.metric_value(vl, metric)
                if metric == "poisoning":
                    if xv is None or yv is None:      # non-measurable rows excluded from denom
                        continue
                pairs.append((int(bool(xv)), int(bool(yv))))
            cell[metric] = paired_delta_ci(pairs) if pairs else None
        return cell

    pooled = {c: build_cell(cell_rows(lambda m, c=c: m["condition"] == c)) for c in CONDITIONS}
    bycp = {}
    for c in CONDITIONS:
        for p in POINTS:
            bycp[f"{c}|{p}"] = build_cell(
                cell_rows(lambda m, c=c, p=p: m["condition"] == c and m["injection_position"] == p))

    # --- row-level agreement ---
    n = len(run_ids)
    class_agree = sum(1 for rid in run_ids if hf_by_id[rid]["class"] == vllm_by_id[rid]["class"])
    vr_agree = sum(1 for rid in run_ids if bool(hf_by_id[rid].get("valid_recovery")) == vllm_by_id[rid]["valid_recovery"])
    agree_by_cond = {}
    for c in CONDITIONS:
        ids = [r for r in run_ids if manifest[r]["condition"] == c]
        agree_by_cond[c] = {
            "n": len(ids),
            "class_agreement": round(sum(1 for r in ids if hf_by_id[r]["class"] == vllm_by_id[r]["class"]) / len(ids), 4),
            "exact_text_match": round(sum(1 for r in ids if next(cr for cr in cont_records if cr["run_id"] == r)["exact_text_match"]) / len(ids), 4),
        }
    class_confusion = Counter((hf_by_id[r]["class"], vllm_by_id[r]["class"]) for r in run_ids)

    # --- max abs cell-level delta ---
    def max_delta(cells):
        best = {"abs_delta": -1}
        for key, cell in cells.items():
            for metric, cc in cell.items():
                if cc is None:
                    continue
                ad = abs(cc["delta"])
                if ad > best["abs_delta"]:
                    best = {"abs_delta": round(ad, 4), "cell": key, "metric": metric,
                            "hf_rate": cc["hf_rate"], "vllm_rate": cc["vllm_rate"],
                            "delta": cc["delta"], "n": cc["n"], "delta_wilson95": cc["delta_wilson95"]}
        return best

    report = {
        "meta": {
            "model": MODEL, "revision": REV, "max_new_tokens": MAX_NEW_TOKENS,
            "n_rows": n, "gen_seconds": round(gen_secs, 1),
            "total_seconds": round(time.time() - t_start, 1),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "backend": "vLLM (greedy, temperature=0, prefix-cache off, seed=0)",
            "hf_source": "results/EXPC_POLARITY_CONTROL_FULL/validated_outputs.jsonl",
            "delta_ci_method": "Newcombe 1998 paired score interval (z=1.96)",
        },
        "hf_pipeline_selfcheck": selfcheck,
        "row_level": {
            "n": n,
            "class_agreement_rate": round(class_agree / n, 4),
            "class_agreement_count": class_agree,
            "valid_recovery_agreement_rate": round(vr_agree / n, 4),
            "exact_text_match_rate": round(exact_text / n, 4),
            "exact_text_match_count": exact_text,
            "class_agreement_by_condition": agree_by_cond,
            "class_confusion_hf_to_vllm": {f"{a}->{b}": c for (a, b), c in sorted(class_confusion.items(), key=lambda kv: -kv[1])},
        },
        "pooled_by_condition": pooled,
        "by_condition_position": bycp,
        "max_abs_cell_delta": {
            "pooled_by_condition": max_delta(pooled),
            "by_condition_position": max_delta(bycp),
        },
    }
    with open(os.path.join(OUTDIR, "replay_full.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(OUTDIR, "vllm_continuations.jsonl"), "w") as fh:
        for cr in cont_records:
            fh.write(json.dumps(cr, sort_keys=True) + "\n")

    sc_ok = all(v2["match"] for v1 in selfcheck.values() for v2 in v1.values())
    print(f"[replay] HF self-check all-match: {sc_ok}")
    print(f"[replay] class agreement {class_agree}/{n} = {class_agree/n:.4f}; exact text {exact_text}/{n}")
    print(f"[replay] max |delta| pooled: {report['max_abs_cell_delta']['pooled_by_condition']}")
    print(f"[replay] max |delta| cond x pos: {report['max_abs_cell_delta']['by_condition_position']}")
    print(f"[replay] wrote {OUTDIR}/replay_full.json + vllm_continuations.jsonl")


if __name__ == "__main__":
    main()
