"""E9 generation runner: two-arm goal-flip rollouts (GPU-gated).

Stages (mirror expd_matched_gradient's staging; two arms per world):
  prepare : generate + audit candidate worlds (CPU; delegates to gen_worlds_e9).
  gold    : greedy gold rollout for BOTH arms of every world (GPU).
  pair    : G1-G3 pairing audit from the gold traces (CPU; fail-closed, logged).
  rollout : R=8 T=0.7 continuations per (eligible world x arm x verification level),
            reusing the SHARED injected prefix (trunk 1..i + plant P), with plant-step
            surprisal captured via prompt-logprobs (registration 2.2). (GPU).

Reuses the canonical vLLM harness (tooling/vllm_gen.py) unchanged. GPU generation is
isolated in _gpu_* functions; every non-GPU function (pair audit, prefix build, manifest)
is CPU-testable and exercised by e9_run.py --self-test.

GPU sequencing (registration 6/12): this script REFUSES to run gold/rollout unless
CUDA_VISIBLE_DEVICES is pinned to a single GPU AND that GPU is <5GB used. The waiter
run_e9_queue.sh enforces the E1-terminal + GPU-free + E9_HUMAN_GO gates before invoking it.

Usage:
  python e9_run.py prepare  --run-dir <d> --seed 20260724 --n-families 500
  python e9_run.py gold     --run-dir <d>        # GPU
  python e9_run.py pair     --run-dir <d>        # CPU
  python e9_run.py rollout  --run-dir <d> --R 8 --temperature 0.7 [--verification low high]
  python e9_run.py --self-test
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (os.path.join(_EXP, "src"),
           os.path.join(_EXP, "improvement_plan", "expd"),
           os.path.join(_EXP, "improvement_plan", "tooling"),
           HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import gen_worlds_e9 as g9  # noqa: E402
from validator import validate_continuation, parse_fact, strip_marker  # noqa: E402

VERIF_INSTR = {
    "low": "",   # base INSTR from vllm_gen used as-is
    "high": ("Before answering, verify that each supplied fact is derivable from the "
             "rules. "),
}


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", (text or "").strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(p, row):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def paths(run_dir):
    return {
        "worlds": os.path.join(run_dir, "worlds"),
        "candidates": os.path.join(run_dir, "worlds", "candidate_worlds.jsonl"),
        "gold": os.path.join(run_dir, "gold_rollouts.jsonl"),
        "eligibility": os.path.join(run_dir, "eligibility_audit.jsonl"),
        "cohort": os.path.join(run_dir, "pair_cohort.jsonl"),
        "manifest": os.path.join(run_dir, "manifest.jsonl"),
        "raw": os.path.join(run_dir, "raw_generations.jsonl"),
        "status": os.path.join(run_dir, "queue_status.json"),
    }


# ------------------------------------------------------- CPU: trunk / pairing

def entity_fact_indices(steps, entity):
    return [i for i, s in enumerate(steps) if parse_fact(strip_marker(s), entity)]


def shared_trunk_prefix(steps_a, steps_b):
    """Longest byte-identical leading run of the two gold traces (the shared trunk)."""
    n = 0
    for x, y in zip(steps_a, steps_b):
        if norm(x) != norm(y):
            break
        n += 1
    return n


def pair_audit(world, steps_a, steps_b):
    """G1-G3 (registration 1.3), fail-closed. Returns (eligible, reason, meta)."""
    E, target_a = world["entity"], world["arms"]["usable"]["target"]
    target_b = world["arms"]["inert"]["target"]
    solved_a = bool(steps_a) and norm(steps_a[-1]) == norm(target_a)
    solved_b = bool(steps_b) and norm(steps_b[-1]) == norm(target_b)
    if not (solved_a and solved_b):
        return False, "G1_not_both_solved", {"solved_a": solved_a, "solved_b": solved_b}
    # both gold traces must validate as genuine re-derivations
    va = validate_continuation(world["question"], [], None, " ".join(steps_a), target_a, E)
    vb = validate_continuation(world["question"], [], None, " ".join(steps_b), target_b, E)
    if va.get("class") != "valid_rederivation" or vb.get("class") != "valid_rederivation":
        return False, "G1_gold_not_valid", {"class_a": va.get("class"), "class_b": vb.get("class")}
    tp = shared_trunk_prefix(steps_a, steps_b)
    if tp < 2:
        return False, "G2_no_shared_trunk", {"shared_prefix_len": tp}
    # plant slot = mid of the shared-trunk ENTITY facts (before the branch)
    ent_idx = [i for i in entity_fact_indices(steps_a, E) if i < tp]
    if len(ent_idx) < 2:
        return False, "G3_trunk_too_short", {"shared_prefix_len": tp, "entity_facts": len(ent_idx)}
    slot = ent_idx[len(ent_idx) // 2]
    # G3: slot must be inside the shared trunk (before either branch diverges)
    if slot >= tp:
        return False, "G3_slot_past_branch", {"slot": slot, "shared_prefix_len": tp}
    meta = {"shared_prefix_len": tp, "plant_slot": slot,
            "prefix_steps": steps_a[:slot], "solved_a": True, "solved_b": True,
            "gold_class_a": va.get("class"), "gold_class_b": vb.get("class")}
    return True, "eligible", meta


def build_pair_cohort(run_dir):
    p = paths(run_dir)
    worlds = {w["world_id"]: w for w in read_jsonl(p["candidates"])}
    gold = {}
    for r in read_jsonl(p["gold"]):
        gold.setdefault(r["world_id"], {})[r["arm"]] = r
    cohort = []
    for wid, w in worlds.items():
        g = gold.get(wid, {})
        if "usable" not in g or "inert" not in g:
            append_jsonl(p["eligibility"], {"world_id": wid, "eligible": False,
                         "reason": "gold_missing", "created_at": now_iso()})
            continue
        steps_a = split_sentences(g["usable"].get("continuation", ""))
        steps_b = split_sentences(g["inert"].get("continuation", ""))
        ok, reason, meta = pair_audit(w, steps_a, steps_b)
        append_jsonl(p["eligibility"], {"world_id": wid, "eligible": ok, "reason": reason,
                     "meta": {k: meta[k] for k in meta if k != "prefix_steps"},
                     "created_at": now_iso()})
        if ok:
            rec = {"world_id": wid, "entity": w["entity"], "plant": w["plant"],
                   "prefix_steps": meta["prefix_steps"], "plant_slot": meta["plant_slot"],
                   "z": w["z"], "bridge": w["bridge"], "A1": w["A1"],
                   "question": w["question"],
                   "arms": {a: {"target": w["arms"][a]["target"], "goal_adj": w["arms"][a]["goal_adj"]}
                            for a in ("usable", "inert")}}
            cohort.append(rec)
    with open(p["cohort"], "w") as f:
        for r in cohort:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    n_tot = len(worlds)
    n_elig = len(cohort)
    print(json.dumps({"worlds": n_tot, "eligible_pairs": n_elig,
                      "yield": round(n_elig / max(1, n_tot), 3)}, indent=2))
    return cohort


# --------------------------------------------------------------- GPU stages

def _require_pinned_gpu():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if dev == "" or "," in dev:
        raise SystemExit("REFUSED: CUDA_VISIBLE_DEVICES must pin exactly one GPU (got %r). "
                         "Launch via run_e9_queue.sh." % dev)


def _load_vllm(model, revision):
    import vllm_gen
    return vllm_gen


def _gpu_gold(run_dir, model, revision, max_new=192):
    _require_pinned_gpu()
    vg = _load_vllm(model, revision)
    p = paths(run_dir)
    tok = vg.get_tokenizer(model, revision)
    done = {(r["world_id"], r["arm"]) for r in read_jsonl(p["gold"])}
    worlds = read_jsonl(p["candidates"])
    rows, keys = [], []
    for w in worlds:
        for arm in ("usable", "inert"):
            if (w["world_id"], arm) in done:
                continue
            ids = vg.render_prompt_ids(tok, w["question"], w["arms"][arm]["target"], None)
            rows.append({"prompt_token_ids": ids})
            keys.append((w, arm))
    if not rows:
        print("gold: nothing to do"); return
    outs = vg.generate_continuations(model, rows, max_new_tokens=max_new, revision=revision)
    for (w, arm), o in zip(keys, outs):
        append_jsonl(p["gold"], {"world_id": w["world_id"], "arm": arm,
                     "target": w["arms"][arm]["target"], "continuation": o.get("text", ""),
                     "created_at": now_iso()})
    print("gold: wrote %d rollouts" % len(rows))


def _gpu_rollout(run_dir, model, revision, R=8, temperature=0.7, seed=20260724,
                 verifications=("low",), max_new=192):
    _require_pinned_gpu()
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    vg = _load_vllm(model, revision)
    p = paths(run_dir)
    tok = vg.get_tokenizer(model, revision)
    cohort = read_jsonl(p["cohort"])
    done = {r["run_id"] for r in read_jsonl(p["raw"])}
    llm = vg._get_llm(model, revision)
    stop = vg.eos_token_ids(model, tok, revision)
    for verif in verifications:
        instr = vg.INSTR if verif == "low" else (VERIF_INSTR["high"] + vg.INSTR)
        jobs = []
        for c in cohort:
            for arm in ("usable", "inert"):
                run_id = hashlib.sha256(json.dumps(
                    [c["world_id"], arm, verif, seed], sort_keys=True).encode()).hexdigest()[:24]
                if run_id in done:
                    continue
                prefix = " " + " ".join(c["prefix_steps"] + [c["plant"]])
                # prompt WITHOUT plant (to locate the plant token span for surprisal)
                base_ids = vg.render_prompt_ids(tok, c["question"], c["arms"][arm]["target"],
                                                " " + " ".join(c["prefix_steps"]),
                                                instruction=instr)
                full_ids = vg.render_prompt_ids(tok, c["question"], c["arms"][arm]["target"],
                                                prefix, instruction=instr)
                plant_span = (len(base_ids), len(full_ids))  # [start,end) plant tokens
                jobs.append((c, arm, verif, run_id, full_ids, plant_span))
        if not jobs:
            continue
        prompts = [TokensPrompt(prompt_token_ids=j[4]) for j in jobs]
        sp = SamplingParams(n=R, temperature=temperature, seed=seed, max_tokens=max_new,
                            stop_token_ids=stop, skip_special_tokens=True, prompt_logprobs=0)
        outs = llm.generate(prompts, sp)
        for (c, arm, verif, run_id, full_ids, span), o in zip(jobs, outs):
            surp = _mean_plant_surprisal(o, span)
            for k, comp in enumerate(o.outputs):
                append_jsonl(p["raw"], {
                    "run_id": "%s_r%d" % (run_id, k), "world_id": c["world_id"], "arm": arm,
                    "verification": verif, "rollout_idx": k,
                    "question": c["question"], "entity": read_entity(c), "plant": c["plant"],
                    "target": c["arms"][arm]["target"], "goal_adj": c["arms"][arm]["goal_adj"],
                    "prefix_steps": c["prefix_steps"], "continuation": comp.text,
                    "z": c["z"], "bridge": c["bridge"], "A1": c["A1"],
                    "plant_surprisal": surp, "temperature": temperature, "seed": seed,
                    "created_at": now_iso()})
        print("rollout[%s]: %d prompts x R=%d" % (verif, len(jobs), R))


def read_entity(cohort_row):
    # entity is the capitalized subject of the plant "E is a z."
    m = re.match(r"^([A-Z]\w*)\b", cohort_row["plant"])
    return m.group(1) if m else None


def _mean_plant_surprisal(vllm_out, span):
    """Mean negative-logprob of the injected plant tokens from prompt_logprobs."""
    try:
        pls = vllm_out.prompt_logprobs
        s, e = span
        vals = []
        for i in range(s, min(e, len(pls))):
            d = pls[i]
            if not d:
                continue
            # d maps token_id -> Logprob; take the realized token's logprob (rank 0 present)
            lp = min((v.logprob for v in d.values()), key=lambda x: -x)  # realized has highest
            vals.append(-lp)
        return round(sum(vals) / len(vals), 4) if vals else None
    except Exception:
        return None


# --------------------------------------------------------------- self-test

def self_test():
    fams, worlds = g9.generate(20260724, 3)
    w = worlds[0]; E = w["entity"]
    # Build plausible gold traces for both arms sharing a trunk, diverging at the branch.
    trunk = fams[0]["trunk"]
    trunk_steps = ["%s is a %s." % (E, trunk[0])]
    for i in range(len(trunk) - 1):
        trunk_steps += ["Every %s is a %s." % (trunk[i], trunk[i + 1]), "%s is a %s." % (E, trunk[i + 1])]
    steps_a = trunk_steps + ["%s is %s." % (E, w["goal_a_adj"])]
    steps_b = trunk_steps + ["%s is %s." % (E, w["goal_b_adj"])]
    ok, reason, meta = pair_audit(w, steps_a, steps_b)
    assert ok, "pair_audit should pass on shared-trunk gold: %s" % reason
    assert meta["plant_slot"] < meta["shared_prefix_len"], "slot must be inside the trunk"
    print("  ok: pair_audit passes on a shared-trunk pair (slot=%d, trunk=%d)"
          % (meta["plant_slot"], meta["shared_prefix_len"]))
    # tamper: arm B not solved
    ok2, reason2, _ = pair_audit(w, steps_a, trunk_steps + ["%s is %s." % (E, "wrongadj")])
    assert not ok2 and reason2 == "G1_not_both_solved", reason2
    print("  ok: unsolved arm B rejected (G1)")
    # tamper: no shared trunk (different openings)
    ok3, reason3, _ = pair_audit(w, steps_a, ["%s is a %s." % (E, "otherpus")] + steps_b[1:])
    assert not ok3, "divergent opening should fail G2/G1"
    print("  ok: divergent-opening pair rejected (%s)" % reason3)
    print("E9 runner self-test passed")


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=["prepare", "gold", "pair", "rollout"])
    ap.add_argument("--run-dir")
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--n-families", type=int, default=500)
    ap.add_argument("--surprise-control", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model-revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--verification", nargs="+", default=["low"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test(); return 0
    if not args.run_dir or not args.stage:
        ap.error("stage and --run-dir required unless --self-test")

    if args.stage == "prepare":
        os.makedirs(paths(args.run_dir)["worlds"], exist_ok=True)
        fams, worlds = g9.generate(args.seed, args.n_families, args.surprise_control)
        audits, failures = g9.run_audits(worlds)
        g9.write_outputs(paths(args.run_dir)["worlds"], args.seed, fams, worlds, audits,
                         failures, args.surprise_control)
        print(json.dumps({"worlds": len(worlds), "audit_failures": failures[:3]}, indent=2))
    elif args.stage == "gold":
        _gpu_gold(args.run_dir, args.model, args.model_revision)
    elif args.stage == "pair":
        build_pair_cohort(args.run_dir)
    elif args.stage == "rollout":
        _gpu_rollout(args.run_dir, args.model, args.model_revision, R=args.R,
                     temperature=args.temperature, verifications=tuple(args.verification))
    return 0


if __name__ == "__main__":
    sys.exit(main())
