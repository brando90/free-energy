"""Formal step-validator for PrOntoQA continuations.

Parses the question's rules/facts into a tiny FOL world, then checks every
entity-fact sentence in a continuation for derivability. Distinguishes:

  valid_rederivation : every post-injection entity fact derivable from TRUE atoms
                       (premises + previously valid derivations), final = target.
                       The corrupted atom is NOT in the true state.
  poisoned           : >=1 post-injection fact derivable only via the corrupted atom
                       (the planted error propagated into the proof).
  parroted           : final sentence = target but >=1 intermediate fact underivable
                       (model jumped to the goal without a valid path).
  derailed           : final sentence != target.

`acknowledged` (doubt language) is reported as an independent flag.
Sentences that fail to parse are counted; instances with any unparsed sentence are
excluded from headline stats (reported separately).
"""
import os, json, re
from collections import defaultdict

DOUBT = re.compile(r"\b(wait|however|but|actually|mistake|error|contradict|incorrect|cannot|inconsistent)\b", re.I)
MARKER = re.compile(r"^(?:Therefore|Thus|Hence|So|Then|It follows that|We know that|This means(?: that)?),?\s+", re.I)

def strip_marker(s):
    """Remove leading discourse markers so style choices don't break parsing."""
    t = MARKER.sub("", s.strip())
    return t[0].upper() + t[1:] if t else t

def _sg(cat):
    return cat[:-2] if cat.endswith("es") else cat

def parse_rule(s):
    """Return (lhs_cat, rhs_pred) or None. rhs_pred = ('cat', c) | ('adj', a, pol)."""
    s = s.rstrip(".")
    m = re.match(r"^(?:Every|Each|All)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$", s)
    if m:
        return _sg(m.group(1).lower()), ("cat", _sg(m.group(2).lower()))
    m = re.match(r"^(?:Every|Each|All)\s+(\w+)\s+is\s+not\s+(?:a|an)\s+(\w+)$", s)
    if m:
        return _sg(m.group(1).lower()), ("not_cat", _sg(m.group(2).lower()))
    m = re.match(r"^(?:Every|Each|All)\s+(\w+)\s+is\s+(not\s+)?(\w+)$", s)
    if m:
        return _sg(m.group(1).lower()), ("adj", m.group(3).lower(), not m.group(2))
    m = re.match(r"^(\w+es)\s+are\s+(\w+es)$", s)
    if m:
        return _sg(m.group(1).lower()), ("cat", _sg(m.group(2).lower()))
    m = re.match(r"^(\w+es)\s+are\s+not\s+(\w+es)$", s)
    if m:
        return _sg(m.group(1).lower()), ("not_cat", _sg(m.group(2).lower()))
    m = re.match(r"^(\w+es)\s+are\s+(not\s+)?(\w+)$", s)
    if m:
        return _sg(m.group(1).lower()), ("adj", m.group(3).lower(), not m.group(2))
    return None

def parse_fact(s, entity=None):
    """Return (entity, pred) or None."""
    s = s.rstrip(".")
    m = re.match(r"^([A-Z]\w*)\s+is\s+(?:a|an)\s+(\w+)$", s)
    if m and (entity is None or m.group(1) == entity):
        return m.group(1), ("cat", _sg(m.group(2).lower()))
    m = re.match(r"^([A-Z]\w*)\s+is\s+not\s+(?:a|an)\s+(\w+)$", s)
    if m and (entity is None or m.group(1) == entity):
        return m.group(1), ("not_cat", _sg(m.group(2).lower()))
    m = re.match(r"^([A-Z]\w*)\s+is\s+(not\s+)?(\w+)$", s)
    if m and (entity is None or m.group(1) == entity):
        return m.group(1), ("adj", m.group(3).lower(), not m.group(2))
    return None

def parse_world(question):
    rules, facts, unparsed = [], [], []
    for s in re.split(r"(?<=\.)\s+", question.strip()):
        s = s.strip()
        if not s:
            continue
        r = parse_rule(s)
        if r:
            rules.append(r)
            continue
        f = parse_fact(s)
        if f:
            facts.append(f)
            continue
        unparsed.append(s)
    return rules, facts, unparsed

def closure(rules):
    """Transitive closure over cat->cat rules; cat->adj propagated through it.
    Returns dict: lhs_cat -> set of reachable preds (('cat',c) and ('adj',a,pol))."""
    direct = defaultdict(set)
    for lhs, rhs in rules:
        direct[lhs].add(rhs)
    reach = {c: set(d) for c, d in direct.items()}
    changed = True
    while changed:
        changed = False
        for c in list(reach):
            new = set()
            for p in reach[c]:
                if p[0] == "cat" and p[1] in direct:
                    new |= direct[p[1]]
            if not new <= reach[c]:
                reach[c] |= new
                changed = True
    return reach

def derivable(pred, state, reach):
    """pred entailed by any cat atom in state under the rule closure."""
    for a in state:
        if a == pred:
            return True
        if a[0] == "cat" and pred in reach.get(a[1], ()):
            return True
    return False

def validate_continuation(question, true_prefix_steps, corrupted_step, continuation_text, target, entity):
    rules, world_facts, unparsed_world = parse_world(question)
    reach = closure(rules)
    state = {p for e, p in world_facts if e == entity}
    # establish atoms from the (correct) prefix
    for s in true_prefix_steps:
        f = parse_fact(strip_marker(s), entity)
        if f:
            state.add(f[1])
    corrupted_atom = None
    if corrupted_step:
        f = parse_fact(strip_marker(corrupted_step), entity)
        corrupted_atom = f[1] if f else None

    sents = [x.strip() for x in re.split(r"(?<=\.)\s+", continuation_text.strip()) if x.strip()]
    sents = [strip_marker(s) for s in sents]
    n_unparsed = 0
    invalid, poisoned_steps = [], []
    for s in sents:
        f = parse_fact(s, entity)
        if f:
            pred = f[1]
            if derivable(pred, state, reach):
                state.add(pred)           # entailed by TRUE state under rule closure
            elif corrupted_atom and derivable(pred, state | {corrupted_atom}, reach):
                poisoned_steps.append(s)  # only reachable through the planted error
                state.add(pred)           # poisoned state propagates
            else:
                invalid.append(s)
            continue
        if parse_rule(s):
            continue                      # rule restatements: checked implicitly via derivability
        if re.match(r"^[A-Z]\w*\s+is\b", s):
            n_unparsed += 1               # entity-ish sentence we failed to parse

    def _norm(x):
        return re.sub(r"[^a-z0-9 ]", "", x.lower()).strip()
    final_ok = bool(sents) and _norm(sents[-1]) == _norm(target)

    if n_unparsed > 0:
        cls = "unparsed"
    elif not final_ok:
        cls = "derailed"
    elif poisoned_steps:
        cls = "poisoned"
    elif invalid:
        cls = "parroted"
    else:
        cls = "valid_rederivation"
    return {"class": cls, "final_ok": final_ok, "n_poisoned": len(poisoned_steps),
            "n_invalid": len(invalid), "n_unparsed": n_unparsed,
            "acknowledged": bool(DOUBT.search(continuation_text))}

def main(results_dirname="results", family_dir="perturbed", out_suffix=""):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    results = os.path.join(base, results_dirname)
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(base, "data", "pilot.jsonl"))}
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(results, "gold", "rollouts.jsonl"))}
    runs = [json.loads(l) for l in open(os.path.join(results, family_dir, "runs.jsonl"))]
    runs = [r for r in runs if "skip" not in r]
    # cohort: instances whose GOLD rollout formally validates (clean canonical reasoning)
    cohort = set()
    for gid, g in gold.items():
        if not g.get("solved"):
            continue
        inst = data[gid]
        steps = [x.strip() for x in re.split(r"(?<=\.)\s+", g["gen_text"].strip()) if x.strip()]
        entity = steps[0].split()[0]
        gv = validate_continuation(inst["question"], [], None, g["gen_text"], inst["target"], entity)
        if gv["class"] == "valid_rederivation":
            cohort.add(gid)

    out_path = os.path.join(results, f"validated{out_suffix}.jsonl")
    counts = defaultdict(lambda: defaultdict(int))
    n_excluded = 0
    with open(out_path, "w") as fh:
        for r in runs:
            if r["id"] not in cohort:
                n_excluded += 1
                continue
            inst = data[r["id"]]
            g = gold[r["id"]]
            steps = [x.strip() for x in re.split(r"(?<=\.)\s+", g["gen_text"].strip()) if x.strip()]
            entity = steps[0].split()[0]
            v = validate_continuation(inst["question"], steps[:r["sent_idx"]],
                                      r["corrupted_step"], r["continuation"],
                                      inst["target"], entity)
            counts[r["point"]][v["class"]] += 1
            counts[r["point"]]["acknowledged"] += int(v["acknowledged"])
            counts[r["point"]]["n"] += 1
            fh.write(json.dumps({**{k: r[k] for k in ["id", "point", "corrupted_step"]}, **v}) + "\n")
    summary = {p: dict(c) for p, c in counts.items()}
    summary["_cohort"] = {"gold_validated_instances": len(cohort),
                          "perturbed_runs_excluded_gold_not_valid": n_excluded}
    for p, c in summary.items():
        if p.startswith("_"):
            continue
        n = c["n"]
        c["valid_rederivation_rate"] = round(c.get("valid_rederivation", 0) / n, 4)
        c["poisoned_rate"] = round(c.get("poisoned", 0) / n, 4)
    with open(os.path.join(results, f"validated_summary{out_suffix}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

def sanity_on_gold():
    """Gold rollouts should validate ~perfectly: run before trusting perturbed numbers."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(base, "data", "pilot.jsonl"))}
    rows = [json.loads(l) for l in open(os.path.join(base, "results", "gold", "rollouts.jsonl"))]
    rows = [r for r in rows if r["solved"]]
    ok = bad = 0
    for r in rows:
        inst = data[r["id"]]
        steps = [x.strip() for x in re.split(r"(?<=\.)\s+", r["gen_text"].strip()) if x.strip()]
        entity = steps[0].split()[0]
        v = validate_continuation(inst["question"], [], None, r["gen_text"], inst["target"], entity)
        if v["class"] == "valid_rederivation":
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print("GOLD-FAIL", v["class"], "|", r["gen_text"][:140])
    print(f"gold sanity: valid={ok} not_valid={bad} ({ok/(ok+bad):.1%})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sanity":
        sanity_on_gold()
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        results_dirname = sys.argv[2] if len(sys.argv) > 2 else "results"
        for fam, d in [("wrong", "perturbed"), ("distractor", "perturbed_distractor"),
                       ("contradiction", "perturbed_contradiction")]:
            import os as _os
            if _os.path.exists(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                             "..", results_dirname, d, "runs.jsonl")):
                print(f"=== family: {fam} ===")
                main(results_dirname, d, "" if fam == "wrong" else f"_{fam}")
    else:
        main()
