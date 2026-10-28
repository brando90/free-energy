"""E11 DEPTH x OPACITY staged runner (single-trace in-stream continuation).

Reuses the EXPG program-trace substrate end to end:
  * worlds            e11_generator/gen_depth.py + audit_depth.py (validated)
  * trace format      expg/trace_format.py  (make_user_message, GOLD_PREFILL)
  * injection         expg/inject.py build_injection (k=0 anchor 'note' family);
                      opacity-aware value plant for k>=1 (this file), which
                      splices the generator's byte-audited site_body over the
                      site line of the model's OWN gold trace.
  * scoring           expg/validate.py classify_run (2-world re-execution DV)
  * generation        tooling/vllm_gen.py  (_get_llm honours EXPG_LLM_MAX_MODEL_LEN)

Stages (one per invocation, EXPG pattern):
  python e11_run.py prepare   --out-dir <D>                 (CPU: worlds+audit)
  python e11_run.py generate  --out-dir <D> --model .. ..   (GPU: gold+conts)
  python e11_run.py validate  --out-dir <D>                 (CPU: 3-way DV)
  python e11_run.py summarize --out-dir <D>                 (CPU: rates+contrasts)
  python e11_run.py report    --out-dir <D>                 (CPU: REPORT.md)

Cells: k in {0,1,2,3,5}. k=0 is the single readable anchor (opacity degenerate).
k>=1 x opacity{bare,full,partial}. Worlds are SHARED across opacity arms of a k;
gold is own-trace per world (EXPG regime).

DV (spec S3, exactly one label/rollout, maps onto EXPG columns):
  absorbed          final_output_absorbed (planted branch carried to output)
  silently_corrected repair_event OR final-output true-matches, with NO flag
  flagged           judge_reject if judged else doubt_lex (frozen lexicon)
  (unresolved)      residual: no output claim / output neither world -- printed,
                    never silently dropped.

Sampling: rollout 0 = greedy; rollouts 1..R = T=0.7 (spec: R=8 + greedy).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import traceback
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(*parts):
    return os.path.abspath(os.path.join(HERE, *parts))


EXPG_DIR = os.environ.get("E11_EXPG_DIR") or _resolve("..", "..", "..", "expg")
E11GEN_DIR = os.environ.get("E11_GEN_DIR") or _resolve("..", "e11_generator")
TOOLING_DIR = os.environ.get("E11_TOOLING") or _resolve("..", "..", "..", "tooling")
for _d in (E11GEN_DIR, EXPG_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import trace_format as tf                       # noqa: E402
import inject as inj_mod                        # noqa: E402
import validate as val_mod                      # noqa: E402
from interp import parse_program                # noqa: E402
import gen_depth as gd                          # noqa: E402
import audit_depth as ad                        # noqa: E402

KS = list(gd.KS)                                # (0,1,2,3,5)
OPACITIES = list(gd.OPACITIES)                  # bare, full, partial
DEFAULT_SEED = 0
PINNED = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "NousResearch/Meta-Llama-3.1-8B-Instruct": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
}


# --------------------------------------------------------------- plumbing

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha(parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def append_jsonl(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def paths(out_dir):
    return {
        "programs": os.path.join(out_dir, "programs.jsonl"),
        "manifest": os.path.join(out_dir, "manifest.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "cohort": os.path.join(out_dir, "cohort.json"),
        "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
        "summary": os.path.join(out_dir, "summary_tables.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "audit": os.path.join(out_dir, "generation_audit.jsonl"),
        "status": os.path.join(out_dir, "queue_status.json"),
        "report": os.path.join(out_dir, "REPORT.md"),
    }


def gold_run_id(pid, seed):
    return sha(["E11", pid, "gold", seed])


def cont_run_id(pid, family, rollout, seed):
    return sha(["E11", pid, family, rollout, seed])


def families_of(prog):
    """Family names carried by a program (k=0 -> anchor; k>=1 -> 3 opacity)."""
    return sorted(prog["families"].keys())


def cell_of(prog, family):
    """(k, opacity) for a family."""
    spec = prog["families"][family]
    return prog["k"], spec.get("opacity", "bare")


# ------------------------------------------------------- opacity injection

def build_injection_e11(prog, family, gold_full_text):
    """k=0 anchor -> reuse EXPG note injection unchanged. k>=1 -> value plant
    where the injected site line is the generator's opacity-rendered site_body
    (bare/full/partial), spliced over the model's own gold site line."""
    spec = prog["families"][family]
    if spec["transform"] != "value":
        return inj_mod.build_injection(prog, family, gold_full_text)
    j = prog["site_line"]
    raws = inj_mod.gold_trace_raw_lines(gold_full_text)
    idx = None
    for i, ln in enumerate(raws):
        p = tf.parse_trace_line(ln)
        if p is not None and p.get("line") == j and p["kind"] == "assign":
            idx = i
            break
    if idx is None:
        raise inj_mod.InjectError("site_line_not_found_in_gold")
    p = tf.parse_trace_line(raws[idx])
    if p["var"] != prog["site_var"]:
        raise inj_mod.InjectError("site_var_mismatch")
    if p["value"] != prog["site_true_value"]:
        raise inj_mod.InjectError("site_value_mismatch")
    injected = "line %d: %s" % (j, spec["site_body"])   # opacity-rendered, planted
    prefix_lines = raws[:idx] + [injected]
    return {
        "family": family, "site_line": j, "opacity": spec["opacity"],
        "injected_trace_line": injected, "original_trace_line": raws[idx],
        "prefix_text": "\n".join(prefix_lines) + "\n",
        "step_replaced": False, "expected_lines_from": j + 1,
        "cf": spec["planted_value"] is not None,
        "planted_var": spec["planted_var"], "planted_value": spec["planted_value"],
        "true_value": spec["true_value"], "force_after_line": spec["force_after_line"],
        "delta_policy": spec["delta_policy"], "out_cf": spec.get("out_cf"),
    }


# ---------------------------------------------------------------- backends

class VllmBackend:
    name = "vllm"

    def __init__(self, model, revision):
        if TOOLING_DIR not in sys.path:
            sys.path.insert(0, TOOLING_DIR)
        import vllm_gen
        self.vg = vllm_gen
        self.model = model
        self.revision = revision
        self.tok = vllm_gen.get_tokenizer(model, revision)

    def render_ids(self, user_msg, prefill):
        ids = self.vg._chat_ids(self.tok, [{"role": "user", "content": user_msg}])
        if prefill:
            ids = ids + list(self.tok(prefill, add_special_tokens=False)["input_ids"])
        return ids

    def greedy(self, ids_list, max_new):
        rows = [{"prompt_token_ids": ids} for ids in ids_list]
        outs = self.vg.generate_continuations(self.model, rows,
                                              max_new_tokens=max_new,
                                              revision=self.revision)
        return [[o["text"]] for o in outs]         # 1 text per prompt

    def sampled(self, ids_list, max_new, R, temp, seed):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(temperature=temp, top_p=1.0, n=R, seed=seed,
                            max_tokens=max_new,
                            stop_token_ids=self.vg.eos_token_ids(self.model, self.tok,
                                                                self.revision),
                            skip_special_tokens=True)
        llm = self.vg._get_llm(self.model, self.revision)
        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in ids_list]
        outs = llm.generate(prompts, sp)
        return [[c.text for c in o.outputs] for o in outs]   # R texts per prompt


class MockBackend:
    """CPU self-test backend: emits the interpreter-true trace as the gold /
    continuation, so gold_solve_eval passes and the plumbing is exercised
    without any model. NOT for real data."""
    name = "mock"

    def __init__(self, model, revision):
        self.model = model
        self.revision = revision
        self._prog_by_msg = {}

    def register(self, programs):
        for p in programs:
            um = tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))
            self._prog_by_msg[um] = p

    def render_ids(self, user_msg, prefill):
        return {"user_msg": user_msg, "prefill": prefill}

    def _true_cont(self, ids):
        p = self._prog_by_msg[ids["user_msg"]]
        stmts = parse_program(p["stmt_texts"])
        from interp import execute
        full = tf.render_true_trace(stmts, execute(stmts))
        pre = ids["prefill"]
        # continuation = the true trace with the prefill span removed
        if full.startswith("line 1:") and pre == tf.GOLD_PREFILL:
            return full[len(pre):]
        # for injected prefixes: emit the true-trace tail after the prefix's lines
        n_prefix = len([x for x in pre.split("\n") if x.strip()])
        tail = full.split("\n")[n_prefix:]
        return "\n".join(tail)

    def greedy(self, ids_list, max_new):
        return [[self._true_cont(ids)] for ids in ids_list]

    def sampled(self, ids_list, max_new, R, temp, seed):
        return [[self._true_cont(ids)] * R for ids in ids_list]


# ---------------------------------------------------------------- prepare

def _targets_for(k):
    t = ad.default_targets()
    t["nearest_anc_dist"] = None if k == 0 else 4
    return t


def _audit_fn_for(k):
    af = ad.make_audit_fn(_targets_for(k))

    def wrap(prog, _af=af):
        fails = _af(prog)
        prog.setdefault("audit_metrics", {})["nearest_anc_dist"] = \
            ad.nearest_ancestor_token_distance(prog)
        return fails
    return wrap


def stage_prepare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    P = paths(args.out_dir)
    if os.path.exists(P["programs"]) and not args.overwrite:
        raise SystemExit("programs.jsonl exists; use --overwrite")
    for key in ("programs", "manifest", "raw", "validated", "audit"):
        if os.path.exists(P[key]) and args.overwrite:
            os.remove(P[key])
    pool = args.world_pool                       # candidate worlds generated per k
    funnels = {}
    for k in KS:
        progs, funnel = gd.generate_cell(k, pool, 700000 + k * 911,
                                        audit_fn=_audit_fn_for(k))
        funnels[str(k)] = {"attempts": funnel["attempts"], "accepted": funnel["accepted"],
                           "rejects": funnel["rejects"], "starved": funnel["starved"]}
        for p in progs:
            append_jsonl(P["programs"], p)
    meta = {
        "experiment": "E11_DEPTH_OPACITY",
        "created_at": now_iso(), "seed": args.seed,
        "ks": KS, "opacities": OPACITIES,
        "world_pool_per_k": pool, "n_target_per_cell": args.n_target,
        "R_samples": args.R, "sampling_temperature": args.temp,
        "decoding": {"greedy_rollout": True, "R_at_T": args.temp,
                     "max_new_gold": args.max_new_gold,
                     "max_new_cont": args.max_new_cont},
        "prompt_sha256": tf.prompt_sha256(), "gold_prefill": tf.GOLD_PREFILL,
        "flag_channel": "judge_reject if judged else doubt_lex (frozen lexicon)",
        "doubt_lexicon": val_mod.DOUBT_LEX_RE.pattern,
        "generator_funnels": funnels,
        "dknobs": {k: gd.DKNOBS[k] for k in gd.DKNOBS},
    }
    write_json(P["metadata"], meta)
    print(json.dumps({"world_pool_per_k": pool, "funnels": funnels}, indent=2))
    print("[E11:prepare] wrote %d worlds -> %s" % (pool * len(KS), P["programs"]))


# ------------------------------------------------------ generate (GPU)

def _batched(fn_render, jobs, backend, raw_path, existing, max_new, batch, tag,
             sampled=False, R=1, temp=0.7, seed=0):
    """jobs: list of (base_row, user_msg, prefill, rollouts). Writes one raw row
    per (base_row, rollout). Resume-safe on run_id."""
    pending = [j for j in jobs if j[0]["run_id"] not in existing]
    done = 0
    for i in range(0, len(pending), batch):
        chunk = pending[i:i + batch]
        ids_list = [backend.render_ids(u, p) for _, u, p, _ in chunk]
        try:
            if sampled:
                outs = backend.sampled(ids_list, max_new, R, temp, seed)
            else:
                outs = backend.greedy(ids_list, max_new)
        except Exception:
            outs = None
        for (base, _u, _p, rollouts), texts in zip(chunk,
                                                    outs or [None] * len(chunk)):
            for ri, roll in enumerate(rollouts):
                rec = dict(base)
                rec["rollout"] = roll
                rec["run_id"] = base["run_id"] if len(rollouts) == 1 else \
                    base["run_id_fn"](roll)
                if texts is None or ri >= len(texts) or texts[ri] is None:
                    rec.update({"failed_generation": True, "created_at": now_iso()})
                else:
                    rec.update({"failed_generation": False, "continuation": texts[ri],
                                "created_at": now_iso()})
                rec.pop("run_id_fn", None)
                append_jsonl(raw_path, rec)
                existing.add(rec["run_id"])
        done += len(chunk)
        print("[E11:%s] %d/%d prompts" % (tag, done, len(pending)), flush=True)
    return len(pending)


def _set_status(P, **kw):
    st = {}
    if os.path.exists(P["status"]):
        try:
            st = json.load(open(P["status"]))
        except Exception:
            st = {}
    st.update(kw)
    st["updated_at"] = now_iso()
    write_json(P["status"], st)


def stage_generate(args):
    P = paths(args.out_dir)
    programs = read_jsonl(P["programs"])
    if not programs:
        raise SystemExit("run prepare first")
    prog_by_id = {p["program_id"]: p for p in programs}
    backend = MockBackend(args.model, args.revision) if args.backend == "mock" \
        else VllmBackend(args.model, args.revision)
    if args.backend == "mock":
        backend.register(programs)
    existing = {r["run_id"] for r in read_jsonl(P["raw"])}
    _set_status(P, stage="gold", model=args.model, backend=args.backend,
                n_worlds=len(programs))

    # ---- phase 1: own-trace gold for every candidate world
    gold_jobs = []
    for p in programs:
        base = {"run_id": gold_run_id(p["program_id"], args.seed),
                "program_id": p["program_id"], "k": p["k"], "condition": "gold"}
        gold_jobs.append((base, tf.make_user_message(tf.make_listing_text(p["stmt_texts"])),
                          tf.GOLD_PREFILL, [0]))
    _batched(None, gold_jobs, backend, P["raw"], existing, args.max_new_gold,
             args.batch, "gold")

    # ---- gold double-filter -> per-k eligible cohort (cap n_target)
    raw = read_jsonl(P["raw"])
    gold = {r["program_id"]: r for r in raw
            if r.get("condition") == "gold" and not r.get("failed_generation")}
    per_prog, by_k = [], defaultdict(list)
    for p in programs:
        g = gold.get(p["program_id"])
        if g is None:
            ev = {"solved": False, "reason": "gold_missing_or_failed"}
        else:
            e = inj_mod.gold_solve_eval(p, tf.GOLD_PREFILL + g["continuation"])
            ev = {"solved": e["solved"], "parse_ok": e["parse_ok"],
                  "complete": e["complete"], "correct": e["correct"],
                  "reason": "eligible" if e["solved"] else
                  (e["first_error"] or {}).get("type", "unknown")}
        per_prog.append(dict(ev, program_id=p["program_id"], k=p["k"]))
        if ev["solved"]:
            by_k[p["k"]].append(p)
    cohort_ids, k_solve = set(), {}
    for k in KS:
        elig = sorted(by_k[k], key=lambda p: p["program_id"])
        tested = sum(1 for pp in per_prog if pp["k"] == k)
        k_solve[str(k)] = {"tested": tested, "eligible": len(elig),
                           "solve_rate": round(len(elig) / tested, 4) if tested else None,
                           "kept": min(len(elig), args.n_target)}
        for p in elig[:args.n_target]:
            cohort_ids.add(p["program_id"])
    cohort = {"created_at": now_iso(), "by_k": k_solve,
              "n_cohort": len(cohort_ids),
              "gold_solve_overall": round(sum(pp["solved"] for pp in per_prog) /
                                          len(per_prog), 4) if per_prog else None,
              "failure_taxonomy": dict(Counter(pp["reason"] for pp in per_prog
                                               if not pp["solved"]))}
    write_json(P["cohort"], cohort)
    print("[E11] gold cohort: %s" % json.dumps(k_solve))
    _set_status(P, stage="manifest", gold_cohort=k_solve)

    # ---- manifest: one row per (cohort world x family), injection audited
    manifest_existing = {r["run_id"] for r in read_jsonl(P["manifest"])}
    n_rej = 0
    for p in programs:
        if p["program_id"] not in cohort_ids:
            continue
        g = gold[p["program_id"]]
        for fam in families_of(p):
            k, opac = cell_of(p, fam)
            rid = sha(["E11man", p["program_id"], fam, args.seed])
            if rid in manifest_existing:
                continue
            try:
                b = build_injection_e11(p, fam, tf.GOLD_PREFILL + g["continuation"])
            except inj_mod.InjectError as e:
                append_jsonl(P["audit"], {"stage": "injection", "program_id": p["program_id"],
                                          "family": fam, "rejected": True,
                                          "reason": e.reason, "created_at": now_iso()})
                n_rej += 1
                continue
            row = {"run_id": rid, "program_id": p["program_id"], "family": fam,
                   "k": k, "opacity": opac, "site_line": p["site_line"],
                   "site_var": p["site_var"], "r1": p["r1"], "r1_var": p["r1_var"],
                   "seed": args.seed, "created_at": now_iso()}
            row.update({kk: b[kk] for kk in ("injected_trace_line", "original_trace_line",
                        "prefix_text", "expected_lines_from", "cf", "planted_var",
                        "planted_value", "true_value", "force_after_line")})
            append_jsonl(P["manifest"], row)
            manifest_existing.add(rid)
    manifest = read_jsonl(P["manifest"])
    print("[E11] manifest: %d rows (%d injection rejects) by-cell=%s" %
          (len(manifest), n_rej,
           json.dumps(dict(Counter("k%s_%s" % (m["k"], m["opacity"]) for m in manifest)))))
    _set_status(P, stage="continuations", manifest_rows=len(manifest))

    # ---- phase 2: continuations, rollout 0 greedy + rollouts 1..R sampled
    #      greedy pass
    gjobs = []
    for m in manifest:
        p = prog_by_id[m["program_id"]]
        base = {"run_id": cont_run_id(m["program_id"], m["family"], 0, args.seed),
                "program_id": m["program_id"], "family": m["family"],
                "k": m["k"], "opacity": m["opacity"], "condition": "cont"}
        gjobs.append((base, tf.make_user_message(tf.make_listing_text(p["stmt_texts"])),
                      m["prefix_text"], [0]))
    _batched(None, gjobs, backend, P["raw"], existing, args.max_new_cont,
             args.batch, "cont-greedy")
    #      sampled pass (R rollouts per prompt)
    if args.R > 0:
        sjobs = []
        for m in manifest:
            p = prog_by_id[m["program_id"]]
            base = {"program_id": m["program_id"], "family": m["family"],
                    "k": m["k"], "opacity": m["opacity"], "condition": "cont",
                    "run_id": cont_run_id(m["program_id"], m["family"], 1, args.seed)}
            base["run_id_fn"] = (lambda roll, mm=m: cont_run_id(mm["program_id"],
                                                                mm["family"], roll, args.seed))
            sjobs.append((base, tf.make_user_message(tf.make_listing_text(p["stmt_texts"])),
                          m["prefix_text"], list(range(1, args.R + 1))))
        _batched(None, sjobs, backend, P["raw"], existing, args.max_new_cont,
                 args.batch, "cont-sampled", sampled=True, R=args.R, temp=args.temp,
                 seed=args.seed)
    print("[E11] generate complete")
    _set_status(P, stage="generated", done=True)


# ---------------------------------------------------------------- validate

def three_way_label(rec):
    """Exactly one of absorbed / silently_corrected / flagged / unresolved."""
    jr = rec.get("judge_reject")
    flagged = bool(jr) if jr is not None else bool(rec.get("doubt_lex"))
    if flagged:
        return "flagged"
    if rec.get("final_output_absorbed"):
        return "absorbed"
    if rec.get("repair_event") or rec.get("final_output_valid"):
        return "silently_corrected"
    return "unresolved"


def stage_validate(args):
    P = paths(args.out_dir)
    programs = {p["program_id"]: p for p in read_jsonl(P["programs"])}
    manifest = read_jsonl(P["manifest"])
    raw = {r["run_id"]: r for r in read_jsonl(P["raw"])}
    if os.path.exists(P["validated"]):
        os.remove(P["validated"])
    n = 0
    for m in manifest:
        p = programs[m["program_id"]]
        stmts = parse_program(p["stmt_texts"])
        inj = {kk: m[kk] for kk in ("cf", "planted_var", "planted_value",
                                    "force_after_line", "expected_lines_from")}
        for roll in range(0, args.R + 1):
            rid = cont_run_id(m["program_id"], m["family"], roll, m["seed"])
            r = raw.get(rid)
            rec = {"program_id": m["program_id"], "family": m["family"], "k": m["k"],
                   "opacity": m["opacity"], "rollout": roll, "run_id": rid,
                   "planted_value": m["planted_value"], "true_value": m["true_value"]}
            if r is None or r.get("failed_generation"):
                rec.update({"generation_failed": True, "label": "unresolved"})
                append_jsonl(P["validated"], rec)
                n += 1
                continue
            met = val_mod.classify_run(stmts, p, inj, r["continuation"])
            rec.update({kk: met[kk] for kk in ("final_output_absorbed", "final_output_valid",
                        "repair_event", "next_read_absorbed", "next_read_class",
                        "trace_valid", "doubt_lex", "doubt_broad", "unparsed",
                        "line_skip_rate")})
            rec["judge_reject"] = r.get("judge_reject")   # None unless a judge stage ran
            rec["generation_failed"] = False
            rec["label"] = three_way_label(rec)
            append_jsonl(P["validated"], rec)
            n += 1
    print("[E11] validated %d rollout-rows -> %s" % (n, P["validated"]))


# ---------------------------------------------------------------- summarize

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / float(n)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cluster_boot_rate(rows, pred, seed=0, n_boot=1000, key="program_id"):
    ids = sorted({r[key] for r in rows})
    if not ids:
        return [None, None]
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [rng.choice(ids) for _ in ids]
        num = den = 0
        for cid in samp:
            for r in by[cid]:
                den += 1
                num += 1 if pred(r) else 0
        vals.append(num / den if den else float("nan"))
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return [None, None]
    return [round(vals[int(0.025 * (len(vals) - 1))], 4),
            round(vals[int(0.975 * (len(vals) - 1))], 4)]


def _cell_summary(rows, seed):
    n = len(rows)
    dist = Counter(r["label"] for r in rows)
    out = {"n": n, "program_n": len({r["program_id"] for r in rows}),
           "label_counts": dict(dist)}
    for lab in ("absorbed", "silently_corrected", "flagged", "unresolved"):
        k = dist.get(lab, 0)
        out[lab] = {"count": k, "rate": round(k / n, 4) if n else None,
                    "wilson95": wilson(k, n),
                    "cluster_boot95": cluster_boot_rate(
                        rows, lambda r, L=lab: r["label"] == L, seed=seed)}
    return out


def _absorbed_rate(cell):
    return cell["absorbed"]["rate"] if cell and cell.get("absorbed") else None


def stage_summarize(args):
    P = paths(args.out_dir)
    rows = read_jsonl(P["validated"])
    if not rows:
        raise SystemExit("run validate first")
    meta = json.load(open(P["metadata"])) if os.path.exists(P["metadata"]) else {}
    cohort = json.load(open(P["cohort"])) if os.path.exists(P["cohort"]) else {}

    cells = {}
    for k in KS:
        opset = ["bare"] if k == 0 else OPACITIES
        for opac in opset:
            crows = [r for r in rows if r["k"] == k and r["opacity"] == opac]
            if crows:
                cells["k%d_%s" % (k, opac)] = _cell_summary(crows, args.seed)

    # ---- exclusion / power flags (spec S6.2)
    power = {}
    for name, c in cells.items():
        n = c["program_n"]
        power[name] = ("ok" if n >= 150 else "scored_flagged" if n >= 100 else
                       "ci_widened_flagged" if n >= 50 else "INVALID_underpowered")

    # ---- contrast 1: k0 vs k1 gap (bare)
    a0 = _absorbed_rate(cells.get("k0_bare"))
    a1 = _absorbed_rate(cells.get("k1_bare"))
    c1 = {"absorbed_k0": a0, "absorbed_k1_bare": a1,
          "gap_k1_minus_k0": round(a1 - a0, 4) if a0 is not None and a1 is not None else None}

    # ---- contrast 2: monotonic trend over k at bare
    bare_rates = [(k, _absorbed_rate(cells.get("k%d_bare" % k))) for k in KS]
    seq = [r for _, r in bare_rates if r is not None]
    isotonic_nondec = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1)) \
        if len(seq) >= 2 else None
    spearman = _spearman([k for k, r in bare_rates if r is not None], seq) \
        if len(seq) >= 3 else None
    c2 = {"bare_rate_by_k": {str(k): r for k, r in bare_rates},
          "isotonic_nondecreasing": isotonic_nondec, "spearman_rho": spearman,
          "k5_minus_k0": (round(seq[-1] - seq[0], 4)
                          if len(seq) >= 2 and bare_rates[0][1] is not None
                          and bare_rates[-1][1] is not None else None)}

    # ---- contrast 3: opacity restoration (bare - full), per k and pooled k>=1
    restor, pooled_b, pooled_f = {}, [], []
    for k in [x for x in KS if x >= 1]:
        b = _absorbed_rate(cells.get("k%d_bare" % k))
        f = _absorbed_rate(cells.get("k%d_full" % k))
        restor["k%d" % k] = {"bare": b, "full": f,
                             "restoration_bare_minus_full":
                             round(b - f, 4) if b is not None and f is not None else None}
        rb = [r for r in rows if r["k"] == k and r["opacity"] == "bare"]
        rf = [r for r in rows if r["k"] == k and r["opacity"] == "full"]
        pooled_b += rb
        pooled_f += rf
    pb = (sum(r["label"] == "absorbed" for r in pooled_b) / len(pooled_b)
          if pooled_b else None)
    pf = (sum(r["label"] == "absorbed" for r in pooled_f) / len(pooled_f)
          if pooled_f else None)
    c3 = {"per_k": restor,
          "pooled_kge1": {"bare": round(pb, 4) if pb is not None else None,
                          "full": round(pf, 4) if pf is not None else None,
                          "restoration": round(pb - pf, 4)
                          if pb is not None and pf is not None else None}}

    summary = {"experiment": "E11_DEPTH_OPACITY", "created_at": now_iso(),
               "model": meta.get("model") or args.model, "cells": cells,
               "power_flags": power, "gold_cohort": cohort.get("by_k"),
               "contrasts": {"k0_vs_k1_gap": c1, "monotonic_trend": c2,
                             "opacity_restoration": c3},
               "flag_channel": meta.get("flag_channel"),
               "stats_note": "Wilson95 + program-cluster bootstrap95 (n_boot=1000); "
                             "cluster=program family (program_id)"}
    write_json(P["summary"], summary)
    print(json.dumps({"contrasts": summary["contrasts"], "power_flags": power}, indent=2))
    print("[E11] summary -> %s" % P["summary"])


def _spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(s):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 4) if n > 1 else None


# ---------------------------------------------------------------- report

def stage_report(args):
    P = paths(args.out_dir)
    s = json.load(open(P["summary"]))
    L = []
    ap = L.append
    ap("# E11 DEPTH x OPACITY report -- %s" % s.get("model"))
    ap("")
    ap("Generated %s. Flag channel: %s." % (s["created_at"], s.get("flag_channel")))
    ap("")
    ap("## Cells (3-way DV distribution; absorbed rate is the headline)")
    ap("")
    cols = ["cell", "n", "prog_n", "power", "absorbed", "silently_corr", "flagged", "unresolved"]
    ap("| " + " | ".join(cols) + " |")
    ap("|" + "---|" * len(cols))
    for name, c in s["cells"].items():
        def r(x):
            m = c[x]
            lo, hi = m["wilson95"]
            ci = " [%.2f,%.2f]" % (lo, hi) if lo is not None else ""
            return "%.3f%s" % (m["rate"], ci) if m["rate"] is not None else "n.m."
        ap("| %s | %d | %d | %s | %s | %s | %s | %s |" %
           (name, c["n"], c["program_n"], s["power_flags"].get(name, "?"),
            r("absorbed"), r("silently_corrected"), r("flagged"), r("unresolved")))
    ap("")
    ap("## Contrasts")
    ap("")
    ap("```json")
    ap(json.dumps(s["contrasts"], indent=2, sort_keys=True))
    ap("```")
    ap("")
    ap("Gold cohort (solve rates / eligible per k): `%s`" % json.dumps(s.get("gold_cohort")))
    ap("")
    ap("Notes: %s" % s["stats_note"])
    with open(P["report"], "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[E11] report -> %s" % P["report"])


# ---------------------------------------------------------------- cli

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "generate", "validate",
                                     "summarize", "report"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--revision", default=None)
    p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--world-pool", type=int, default=350,
                   help="candidate worlds generated per k (before gold filter)")
    p.add_argument("--n-target", type=int, default=150,
                   help="eligible worlds kept per cell after gold filter")
    p.add_argument("--R", type=int, default=8, help="sampled rollouts (T>0) in addition to greedy")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-new-gold", type=int, default=512)
    p.add_argument("--max-new-cont", type=int, default=384)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args(argv)
    if a.revision is None:
        a.revision = PINNED.get(a.model)
    return a


def main(argv=None):
    a = parse_args(argv)
    {"prepare": stage_prepare, "generate": stage_generate, "validate": stage_validate,
     "summarize": stage_summarize, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
