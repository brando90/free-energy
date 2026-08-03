"""E9 scoring: licensed-split (D10) + reuse (D11) + unlicensed (D11.1) + engagement
gate (4.4) + realized effort (2.3) + active-path overlap (2.4), per continuation row.

Reuses the E5 classifier VERBATIM (iclr_exec/e5_licensed_split/e5_classify.classify_row)
and the canonical validator for the reuse/poisoning DV. Registration 4.3 gate:
the E5 six-anchor calibration does NOT transfer to E9 worlds; a stratified sample
for the arm-blind HARD-RULE-5 human audit is emitted here (audit_sample.jsonl) and
MUST be adjudicated by a non-author rater before any licensed count is trusted.

Input  : raw_generations.jsonl rows with fields
         {world_id, arm ('usable'|'inert'), verification ('low'|'high'),
          rollout_idx, question, entity, plant, target, goal_adj,
          prefix_steps, continuation, z, bridge, A1, [plant_surprisal]}
Output : scored_rows.jsonl (+ audit_sample.jsonl)

Usage:  python e9_score.py --run-dir <dir>
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (os.path.join(_EXP, "src"),
           os.path.join(_EXP, "improvement_plan", "expe"),
           os.path.join(_EXP, "improvement_plan", "iclr_exec", "e5_licensed_split"),
           HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from validator import parse_fact, strip_marker, validate_continuation  # noqa: E402
from expc_polarity_control import statement_predicate, opposite_pred  # noqa: E402
from e5_classify import classify_row  # noqa: E402


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", (text or "").strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def stated_complement(question, entity, plant, continuation):
    pred = statement_predicate(plant, entity)
    if pred is None:
        return False
    comp = (entity, opposite_pred(pred))
    return any(parse_fact(strip_marker(s), entity) == comp for s in split_sentences(continuation))


def engaged_goal(continuation, target):
    """Rollout-level engagement gate (4.4): did the rollout reach the queried goal?
    Symmetric across arms. Fail-closed to did_not_engage when the final asserted
    step is not the target."""
    steps = split_sentences(continuation)
    return bool(steps) and norm(steps[-1]) == norm(target)


def active_path_overlap(continuation, tokens):
    """2.4: fraction of plant-chain predicates (z, bridge, A1) surfaced in the
    realized continuation (contextual activation, not raw frequency)."""
    body = " " + norm(continuation) + " "
    hits = sum(1 for t in tokens if (" %s " % norm(t)) in body)
    return hits / max(1, len(tokens))


def score_row(r):
    q, E, plant = r["question"], r["entity"], r["plant"]
    cont, target = r.get("continuation", ""), r["target"]
    prefix = r.get("prefix_steps", [])
    out = dict(r)
    out.pop("continuation", None)  # keep scored rows compact; raw retains text

    engaged = engaged_goal(cont, target)
    stated = stated_complement(q, E, plant, cont)
    licensed_strict = 0
    unlicensed = 0
    klass = None
    if stated:
        c = classify_row(q, E, plant, prefix, cont)
        if c is not None:
            klass = c["klass"]
            licensed_strict = int(klass == "licensed")
            unlicensed = int(klass == "unlicensed")
    # D11 reuse / poisoning (derivational absorption of the plant)
    v = validate_continuation(q, prefix, plant, cont, target, E)
    reuse = int(v.get("class") == "poisoned")

    steps = split_sentences(cont)
    out.update({
        "engaged": int(engaged),
        "stated_complement": int(stated),
        "klass": klass,
        "licensed_strict": licensed_strict,
        "unlicensed_rejection": unlicensed,
        "reuse_poisoned": reuse,
        "realized_n_steps": len(steps),
        "realized_char_len": len(cont or ""),
        "active_path_overlap": active_path_overlap(cont, [r.get("z"), r.get("bridge"), r.get("A1")]),
        "validator_class": v.get("class"),
    })
    return out


def emit_audit_sample(scored, out_path, per_stratum=8, seed=20260724):
    """Registration 4.3: stratified-by-(arm x class) sample for the arm-blind
    HARD-RULE-5 human audit. Arm label is withheld in the emitted record; only a
    blind_id + the continuation text is exposed to the rater."""
    rng = random.Random(seed)
    buckets = {}
    for r in scored:
        klass = r.get("klass") or ("engaged_no_reject" if r.get("engaged") else "did_not_engage")
        buckets.setdefault((r["arm"], klass), []).append(r)
    sample = []
    for key, rows in sorted(buckets.items()):
        rng.shuffle(rows)
        sample.extend(rows[:per_stratum])
    rng.shuffle(sample)
    with open(out_path, "w") as f:
        for i, r in enumerate(sample):
            f.write(json.dumps({
                "blind_id": "E9AUD%04d" % i,
                "world_id": r["world_id"], "rollout_idx": r.get("rollout_idx"),
                "verification": r.get("verification"),
                # ARM WITHHELD from the rater; kept in a separate key file only.
                "continuation_ref": {"run_dir_row": r.get("run_id")},
                "classifier_klass": r.get("klass"),
            }, sort_keys=True) + "\n")
    # key file (arm reveal) written separately for post-audit unblinding
    with open(out_path.replace(".jsonl", "_KEY.jsonl"), "w") as f:
        for i, r in enumerate(sample):
            f.write(json.dumps({"blind_id": "E9AUD%04d" % i, "arm": r["arm"],
                                "world_id": r["world_id"]}, sort_keys=True) + "\n")
    return len(sample)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--raw", default=None, help="override raw_generations.jsonl path")
    args = ap.parse_args(argv)
    raw = args.raw or os.path.join(args.run_dir, "raw_generations.jsonl")
    rows = read_jsonl(raw)
    scored = [score_row(r) for r in rows if not r.get("failed_generation")]
    outp = os.path.join(args.run_dir, "scored_rows.jsonl")
    with open(outp, "w") as f:
        for r in scored:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    n_aud = emit_audit_sample(scored, os.path.join(args.run_dir, "audit_sample.jsonl"))
    print(json.dumps({"scored_rows": len(scored),
                      "engaged": sum(r["engaged"] for r in scored),
                      "stated_complement": sum(r["stated_complement"] for r in scored),
                      "licensed_strict": sum(r["licensed_strict"] for r in scored),
                      "reuse_poisoned": sum(r["reuse_poisoned"] for r in scored),
                      "audit_sample_emitted": n_aud}, indent=2))


if __name__ == "__main__":
    main()
