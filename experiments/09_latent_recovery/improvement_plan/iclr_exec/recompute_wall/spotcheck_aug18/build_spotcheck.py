#!/usr/bin/env python3
"""Build the 50-row human spot-check packet for the re-execution grader.

Stratified (absorbed 20 / silently_corrected 12 / flagged 8 / unresolved 10,
shortfalls rebalanced), seeded (50915), BLIND (the HTML shows no labels, no
model names, no cell names; the key stays server-side). Sources: canonical
E11 dirs (haiku_r8, qwen7b, llama8b) + E10 topup (Anthropic) + E10 bridge
(GPT rows). Read-only over results; writes only into e16-adjacent spotcheck/.
"""
import html
import json
import os
import random

EXP = ("/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/"
       "improvement_plan/iclr_exec/recompute_wall")
OUT = os.path.join(EXP, "spotcheck_aug18")
SEED = 50915
ALLOC = {"absorbed": 20, "silently_corrected": 12, "flagged": 8, "unresolved": 10}


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def three_way(rec):
    if rec.get("judge_reject") if rec.get("judge_reject") is not None else rec.get("doubt_lex"):
        return "flagged"
    if rec.get("final_output_absorbed"):
        return "absorbed"
    if rec.get("repair_event") or rec.get("final_output_valid"):
        return "silently_corrected"
    return "unresolved"


pool = []
skipped = {}


def add(src, label, stmts, prefix, planted, true_v, cont, rid, cell):
    if not (stmts and prefix and cont is not None and planted is not None):
        skipped[src] = skipped.get(src, 0) + 1
        return
    pool.append(dict(src=src, label=label, stmts=stmts, prefix=prefix,
                     planted=planted, true=true_v, cont=cont, run_id=rid,
                     cell=cell))


# ---- E11-style dirs (raw joined by run_id; haiku_r8 borrows the manifest
# of its R=3 sibling, which used the identical shared worlds and golds) ----
for name, man_dir in (("E11_haiku_r8", "E11_haiku"),
                      ("E11_qwen7b", "E11_qwen7b"),
                      ("E11_llama8b", "E11_llama8b")):
    d = os.path.join(EXP, "e11_run", "results", name)
    md = os.path.join(EXP, "e11_run", "results", man_dir)
    try:
        programs = {p["program_id"]: p for p in jl(os.path.join(d, "programs.jsonl"))}
        man = {(m["program_id"], m["family"]): m for m in jl(os.path.join(md, "manifest.jsonl"))}
        raw = {r["run_id"]: r for r in jl(os.path.join(d, "raw_generations.jsonl"))}
        for v in jl(os.path.join(d, "validated_outputs.jsonl")):
            if v.get("generation_failed"):
                continue
            m = man.get((v["program_id"], v["family"]))
            r = raw.get(v.get("run_id"))
            if not m or not r:
                skipped[name] = skipped.get(name, 0) + 1
                continue
            p = programs.get(v["program_id"])
            add(name, v["label"], p["stmt_texts"] if p else None,
                m.get("prefix_text"), m.get("planted_value"), m.get("true_value"),
                r.get("continuation"), f'{name}:{v["program_id"]}:{v["family"]}:{v.get("rollout")}',
                v["family"])
    except FileNotFoundError as e:
        skipped[name] = str(e)

# ---- E10 dirs (join defensively) ----
E10 = [
    (os.path.join(EXP, "e10_topup_fable"),
     os.path.join(EXP, "e10_topup_fable", "programs_batch2.jsonl"),
     ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-8", "claude-sonnet-5"]),
    (os.path.join(EXP, "e10_api_bridge"),
     "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/"
     "results/EXPH2_FRONTIER_FOLLOWUPS/regime2/programs.jsonl",
     ["gpt-4.1", "gpt-4o", "gpt-5.1", "gpt-3.5-turbo-instruct"]),
]
KEEP_CELLS = {"onehop_kc1", "deep_kc5", "adjacent_contradiction"}
for d, progpath, models in E10:
    try:
        programs = {p["program_id"]: p for p in jl(progpath)}
    except FileNotFoundError:
        continue
    for mk in models:
        try:
            man = {(m["program_id"], m["cell"]): m
                   for m in jl(os.path.join(d, "manifest_%s.jsonl" % mk))}
            raws = jl(os.path.join(d, "raw_pert_%s.jsonl" % mk))
            raw = {}
            for r in raws:
                raw.setdefault(r["run_id"], []).append(r)
            for v in jl(os.path.join(d, "validated_%s.jsonl" % mk)):
                cell = v.get("cell")
                if cell not in KEEP_CELLS:
                    continue
                m = man.get((v["program_id"], cell))
                cont = v.get("continuation")
                if cont is None:
                    rr = raw.get(v.get("run_id"), [])
                    cont = rr[v.get("rollout", 0)].get("continuation") if rr else None
                p = programs.get(v["program_id"])
                add("e10:" + mk, three_way(v), p["stmt_texts"] if p else None,
                    m.get("prefix_text") if m else None,
                    m.get("planted_value") if m else v.get("planted_value"),
                    m.get("true_value") if m else v.get("true_value"),
                    cont, "e10:%s:%s:%s:%s" % (mk, v["program_id"], cell, v.get("rollout", 0)),
                    cell)
        except FileNotFoundError as e:
            skipped["e10:" + mk] = str(e)

from collections import Counter, defaultdict
print("pool:", len(pool), "| by label:", dict(Counter(r["label"] for r in pool)))
print("skipped:", skipped)

# ---- stratified draw: within each label, round-robin over (src, cell) ----
rng = random.Random(SEED)
sample = []
alloc = dict(ALLOC)
for label in ("flagged", "silently_corrected", "unresolved", "absorbed"):
    want = alloc[label]
    groups = defaultdict(list)
    for r in pool:
        if r["label"] == label:
            groups[(r["src"], r["cell"])].append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    rng.shuffle(keys)
    got, i = [], 0
    while len(got) < want and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            got.append(groups[k].pop())
        i += 1
    if len(got) < want:  # rebalance shortfall into the next scarce stratum
        short = want - len(got)
        for tgt in ("unresolved", "silently_corrected", "absorbed"):
            if tgt != label and short > 0:
                alloc[tgt] = alloc.get(tgt, 0) + short
                print("shortfall in", label, "->", short, "moved to", tgt)
                break
    sample.extend(got)

rng.shuffle(sample)
for i, r in enumerate(sample):
    r["row"] = i + 1

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "answer_key.json"), "w") as f:
    json.dump([{k: r[k] for k in ("row", "run_id", "src", "cell", "label")}
               for r in sample], f, indent=1)

# ---- blind HTML ----
CSS = ("body{font-family:Georgia,serif;max-width:900px;margin:2em auto;color:#222}"
       "pre{background:#f4f4f4;padding:10px;overflow-x:auto;font-size:13px;line-height:1.45}"
       "mark{background:#ffd9d0}h2{border-top:2px solid #ccc;padding-top:1em}"
       ".meta{color:#555;font-size:14px}.opts{background:#eef3ee;padding:8px 12px;font-size:14px}")
rows_html = []
for r in sample:
    listing = "\n".join("%d: %s" % (i + 1, s) for i, s in enumerate(r["stmts"]))
    pre_lines = r["prefix"].rstrip("\n").split("\n")
    marked = "\n".join(html.escape(l) for l in pre_lines[:-1])
    marked += ("\n<mark>" + html.escape(pre_lines[-1]) + "</mark>") if pre_lines else ""
    rows_html.append(
        "<h2>Row %d</h2>"
        "<p class='meta'>Planted value: <b>%s</b> &nbsp;·&nbsp; true value: <b>%s</b>"
        " (in the highlighted line)</p>"
        "<p><b>Program</b></p><pre>%s</pre>"
        "<p><b>What the model was continuing</b> (last line contains the plant)</p><pre>%s</pre>"
        "<p><b>The model's continuation</b></p><pre>%s</pre>"
        "<p class='opts'><b>Your label:</b> A = absorbed (downstream values build on the planted value) ·"
        " C = silently corrected (downstream follows the TRUE value, no comment) ·"
        " F = flagged (explicitly remarks on the error/doubt — F wins over A/C) ·"
        " U = unresolved (no usable/gradable output)</p>"
        % (r["row"], r["planted"], r["true"], html.escape(listing),
           marked, html.escape(r["cont"] or "(empty)")))
with open(os.path.join(OUT, "spot_check_sample.html"), "w") as f:
    f.write("<meta charset='utf-8'><title>Grader spot-check — 50 rows</title>"
            "<style>%s</style><h1>Grader spot-check (blind)</h1>"
            "<p>50 rows, stratified sample, seed %d. For each row decide A / C / F / U"
            " using the definitions under each row. Judge by reading the continuation"
            " against the program: which value do the downstream lines actually build on?"
            " If the model explicitly comments on the error or expresses doubt, label F"
            " regardless of the values. Record answers in spot_check_answers.txt.</p>%s"
            % (CSS, SEED, "".join(rows_html)))
with open(os.path.join(OUT, "spot_check_answers.txt"), "w") as f:
    f.write("# Grader spot-check answers — one letter per row (A/C/F/U)\n")
    for r in sample:
        f.write("%d: \n" % r["row"])
print("WROTE", OUT, "| sample:", len(sample), "| by label:",
      dict(Counter(r["label"] for r in sample)))
print("by source:", dict(Counter(r["src"] for r in sample)))
