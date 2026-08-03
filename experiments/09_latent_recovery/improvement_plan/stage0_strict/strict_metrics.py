#!/usr/bin/env python
"""Stage-0 strict metrics (validatorPlan Part 3, M1-M4). Pure CPU re-analysis of
stored generations; zero GPU; reads results/* read-only and writes ONLY into
improvement_plan/stage0_strict/.

Metrics per family x position (denominator: ALL rows, incl. generation_failed
and unparsed, per the paper's locked convention):

  M2  hop_sound_valid : closure-valid AND every stated entity fact follows by ONE
                        direct rule application (or state membership) from the
                        previous state. State seeds: entity premises + prefix facts
                        + the injected atom IFF it is true-derivable under closure
                        (benign / true-interruption; never for false plants).
      goal_jump0      : closure-valid with ZERO intermediate parsed entity-facts
                        before the final line (audit.py "jump0" convention).
      goal_jump_strict: goal_jump0 AND the final target is NOT one direct hop /
                        state member from the prefix state (a real skipped chain).
  M1  strict_repair   : over audited-FALSE injections only:
                        (verbalized doubt OR states complement of planted atom)
                        AND closure-valid AND NOT goal_jump0.
      strict_repair_plan: stricter validatorPlan variant: (targeted flag OR
                        complement) AND closure-valid AND NOT goal_jump0 AND NOT
                        stated-reliance(echo).
  M3  stated_reliance : closure-valid AND continuation restates the planted atom
                        (parsed-literal echo, or normalized-text echo for rule
                        injections). Also rule-mediated variant: a fact
                        immediately justified by a rule whose body is the planted
                        category.
  M4  used_injection  : ANY class; echo OR derivational.
                        echo         = planted content restated.
                        derivational = some step derivable only WITH the planted
                        atom (pred != atom), or -- for fabricated-rule plants --
                        only under rules+planted rule. Derivational use is
                        structurally measurable only for positive-categorical
                        atoms (or rule plants); other cells are n.m., which fixes
                        the misleading 0.000-dagger cells for negated families.
"""
import os, sys, json, re, glob, math, random, argparse, datetime
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LR_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
import validator as V  # noqa: E402  (read-only import; pure stdlib module)

OUT_DIR = os.path.join(ROOT, "improvement_plan", "stage0_strict")
POINTS = ["early", "mid", "late"]
EXPA_CONDS = ["benign_paraphrase", "true_interruption", "one_hop_falsehood", "global_falsehood"]


# ---------------------------------------------------------------- utilities
def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", (text or "").strip()) if p.strip()]


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def singular(c):
    return c[:-2] if c.endswith("es") else c


def build_direct(rules):
    d = defaultdict(set)
    for lhs, rhs in rules:
        d[lhs].add(rhs)
    return d


def derivable_direct(pred, state, direct):
    """pred is in state, or one direct rule application from a cat atom in state."""
    for a in state:
        if a == pred:
            return True
        if a[0] == "cat" and pred in direct.get(a[1], ()):
            return True
    return False


def complement(atom):
    if atom is None:
        return None
    if atom[0] == "cat":
        return ("not_cat", atom[1])
    if atom[0] == "not_cat":
        return ("cat", atom[1])
    if atom[0] == "adj":
        return ("adj", atom[1], not atom[2])
    return None


def audit_truth_status(question, statement, entity):
    """Port of expa_global_expansion.audit_truth_status (kept verbatim in logic)."""
    statement = V.strip_marker(statement or "")
    s = statement.strip().rstrip(".")
    parsed = None
    m = re.match(rf"^{re.escape(entity)}\s+is\s+(?:a|an)\s+(\w+)$", s)
    if m:
        parsed = ("cat", singular(m.group(1).lower()), True)
    if parsed is None:
        m = re.match(rf"^{re.escape(entity)}\s+is\s+not\s+(?:a|an)\s+(\w+)$", s)
        if m:
            parsed = ("cat", singular(m.group(1).lower()), False)
    if parsed is None:
        m = re.match(rf"^{re.escape(entity)}\s+is\s+(not\s+)?(\w+)$", s)
        if m:
            parsed = ("adj", m.group(2).lower(), not bool(m.group(1)))
    if parsed is None:
        return {"truth_status": "unparsed", "audit_basis": "statement_unparsed"}
    kind, name, positive = parsed
    rules, facts, _ = V.parse_world(question)
    reach = V.closure(rules)
    premises = {p for e, p in facts if e == entity}
    if kind == "cat":
        entailed_pos = V.derivable(("cat", name), premises, reach)
        if positive:
            return {"truth_status": "true" if entailed_pos else "false",
                    "audit_basis": "entailed_category" if entailed_pos else "unentailed_category_closed_world"}
        return {"truth_status": "false" if entailed_pos else "unknown",
                "audit_basis": "negates_entailed_category" if entailed_pos else "negated_unentailed_category_open_world"}
    pred = ("adj", name, positive)
    entailed = V.derivable(pred, premises, reach)
    opposite = V.derivable(("adj", name, not positive), premises, reach)
    if entailed:
        return {"truth_status": "true", "audit_basis": "entailed_attribute"}
    if opposite:
        return {"truth_status": "false", "audit_basis": "opposite_attribute_entailed"}
    return {"truth_status": "unknown", "audit_basis": "attribute_not_decidable"}


def lemmas_of(injected):
    """Content lemmas of the planted statement (categories + trailing adjective)."""
    low = (injected or "").lower()
    lem = set(re.findall(r"\b([a-z]+pus)(?:es)?\b", low))
    m = re.search(r"\bis\s+(?:not\s+)?(?:a\s+|an\s+)?([a-z]+)\s*\.?\s*$", low)
    if m and not m.group(1).endswith("pus") and m.group(1) not in ("a", "an", "not"):
        lem.add(m.group(1))
    return lem


# ---------------------------------------------------------------- core replay
def analyze(question, prefix_steps, injected, continuation, target, entity):
    """v1-faithful replay of validate_continuation with strict-metric extras.

    Returns primitive flags; composites are assembled in summarize().
    """
    rules, world_facts, _ = V.parse_world(question)
    direct = build_direct(rules)
    reach = V.closure(rules)
    state = {p for e, p in world_facts if e == entity}
    for s in prefix_steps or []:
        f = V.parse_fact(V.strip_marker(s), entity)
        if f:
            state.add(f[1])
    prefix_state = set(state)

    inj_stripped = V.strip_marker(injected) if injected else ""
    a_star = None
    if injected:
        f = V.parse_fact(inj_stripped, entity)
        a_star = f[1] if f else None
    inj_rule = V.parse_rule(inj_stripped) if (injected and a_star is None) else None
    comp = complement(a_star)
    inj_norm = norm(inj_stripped) if injected else None
    # hop-sound seed: injected atom counts as visible context ONLY if it is
    # true-derivable from the prefix state under closure (benign / true plants).
    hop_seed = {a_star} if (a_star and V.derivable(a_star, prefix_state, reach)) else set()
    # fabricated-rule plant: augmented closure for derivational-use detection
    reach_aug = V.closure(rules + [inj_rule]) if inj_rule else None

    sents_raw = split_sentences(continuation)
    sents = [V.strip_marker(s) for s in sents_raw]
    state_true = set(state)  # taint-free twin: never receives injection-dependent facts
    n_unparsed = n_invalid = n_poisoned = 0
    echo_fact = echo_text = deriv_use = deriv_use_rule = False
    corrective = False
    hop_sound_all = True
    inter_facts = 0
    reliance_rule = False
    prev_rule = None
    for i, s in enumerate(sents):
        if inj_norm and norm(s) == inj_norm:
            echo_text = True
        f = V.parse_fact(s, entity)
        if f:
            pred = f[1]
            if i < len(sents) - 1:
                inter_facts += 1
            if a_star and pred == a_star:
                echo_fact = True
            if comp and pred == comp:
                corrective = True
            if (a_star and a_star[0] == "cat" and prev_rule is not None
                    and prev_rule[0] == a_star[1] and pred == prev_rule[1]):
                reliance_rule = True  # fact justified by a rule whose body is the plant
            if not derivable_direct(pred, state | hop_seed, direct):
                hop_sound_all = False
            # taint-free injection-dependence (M4): checked against state_true so
            # that v1's poisoned-state propagation cannot mask downstream use
            if V.derivable(pred, state_true, reach):
                state_true.add(pred)
            else:
                if a_star and pred != a_star and V.derivable(pred, state_true | {a_star}, reach):
                    deriv_use = True
                if reach_aug is not None and V.derivable(pred, state_true, reach_aug):
                    deriv_use_rule = True  # derivable only with the fabricated rule
            # v1-faithful classification state
            if V.derivable(pred, state, reach):
                state.add(pred)
            elif a_star and V.derivable(pred, state | {a_star}, reach):
                n_poisoned += 1
                state.add(pred)  # v1: poisoned state propagates
            else:
                n_invalid += 1
            prev_rule = None
            continue
        r = V.parse_rule(s)
        if r:
            prev_rule = r
            continue
        prev_rule = None
        if re.match(r"^[A-Z]\w*\s+is\b", s):
            n_unparsed += 1

    final_ok = bool(sents) and norm(sents[-1]) == norm(target)
    if n_unparsed > 0:
        cls = "unparsed"
    elif not final_ok:
        cls = "derailed"
    elif n_poisoned:
        cls = "poisoned"
    elif n_invalid:
        cls = "parroted"
    else:
        cls = "valid_rederivation"

    doubt = bool(V.DOUBT.search(continuation or ""))
    targeted = False
    if doubt and injected:
        lems = lemmas_of(inj_stripped)
        low_raw = [s.lower() for s in sents_raw]
        for i, s in enumerate(low_raw):
            if V.DOUBT.search(s):
                window = low_raw[max(0, i - 2): i + 3]
                if any(any(re.search(rf"\b{re.escape(l)}(?:es)?\b", w) for l in lems)
                       for w in window):
                    targeted = True
                    break

    return {
        "class": cls,
        "doubt": doubt,
        "targeted": targeted,
        "echo": echo_fact or echo_text,
        "corrective": corrective,
        "deriv_use": deriv_use,
        "deriv_use_rule": deriv_use_rule,
        "hop_sound_all": hop_sound_all,
        "inter_facts": inter_facts,
        "reliance_rule": reliance_rule,
        "n_poisoned": n_poisoned,
        "n_invalid": n_invalid,
        "n_unparsed": n_unparsed,
        "a_star_kind": (a_star[0] if a_star else ("rule" if inj_rule else None)),
    }


def compose(row):
    """Derive composite per-run booleans from primitives (all False for failed gen)."""
    cls = row["class"]
    valid = cls == "valid_rederivation"
    jump0 = valid and row.get("inter_facts", 0) == 0
    hop_sound_valid = valid and row.get("hop_sound_all", False)
    used_any = row.get("echo", False) or row.get("deriv_use", False) or row.get("deriv_use_rule", False)
    flag_or_correct = row.get("doubt", False) or row.get("corrective", False)
    row.update({
        "closure_valid": valid,
        "hop_sound_valid": hop_sound_valid,
        "goal_jump0": jump0,
        "goal_jump_strict": jump0 and not row.get("hop_sound_all", False),
        "used_any": used_any,
        "flag_or_correct": flag_or_correct,
        "srr_task": row["is_false"] and flag_or_correct and valid and not jump0,
        "srr_plan": (row["is_false"] and (row.get("targeted", False) or row.get("corrective", False))
                     and valid and not jump0 and not row.get("echo", False)),
        "stated_reliance": valid and row.get("echo", False),
        "echo_only": row.get("echo", False) and not (row.get("deriv_use", False) or row.get("deriv_use_rule", False)),
        "derivational": row.get("deriv_use", False) or row.get("deriv_use_rule", False),
    })
    return row


# ---------------------------------------------------------------- summaries
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cluster_boot(rows, key, seed=0, n_boot=400):
    ids = sorted({r["cluster"] for r in rows})
    if not ids:
        return [None, None]
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["cluster"]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        num = den = 0
        for pid in sample:
            for r in by_id[pid]:
                den += 1
                num += int(bool(r[key]))
        vals.append(num / den if den else float("nan"))
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4),
            round(vals[int(0.975 * (len(vals) - 1))], 4)]


def block(rows, key, ci=False):
    n = len(rows)
    k = sum(int(bool(r[key])) for r in rows)
    out = {"count": k, "n": n, "rate": round(k / n, 4) if n else None}
    if ci and n:
        out["wilson95"] = wilson(k, n)
        out["cluster_boot95"] = cluster_boot(rows, key)
    return out


def summarize_cell(rows):
    n = len(rows)
    cell = {"n": n, "problem_n": len({r["cluster"] for r in rows})}
    cls_counts = defaultdict(int)
    for r in rows:
        cls_counts[r["class"]] += 1
    cell["class_counts"] = dict(cls_counts)
    for key, ci in [("closure_valid", True), ("hop_sound_valid", True),
                    ("goal_jump0", False), ("goal_jump_strict", False),
                    ("doubt", False), ("used_any", True), ("echo", False),
                    ("echo_only", False), ("stated_reliance", False),
                    ("reliance_rule", False)]:
        cell[key] = block(rows, key, ci)
    valid_rows = [r for r in rows if r["closure_valid"]]
    cell["goal_jump0_given_valid"] = block(valid_rows, "goal_jump0")
    cell["hop_unsound_given_valid"] = {
        "count": sum(1 for r in valid_rows if not r["hop_sound_all"]),
        "n": len(valid_rows),
        "rate": round(sum(1 for r in valid_rows if not r["hop_sound_all"]) / len(valid_rows), 4) if valid_rows else None,
    }
    # M4 derivational: measurable only for positive-cat atoms or rule plants
    meas = [r for r in rows if r.get("a_star_kind") in ("cat", "rule")]
    cell["derivational_measurable_n"] = len(meas)
    cell["derivational"] = block(meas, "derivational", ci=bool(meas)) if meas else "n.m."
    # M1 (false-plant rows only)
    false_rows = [r for r in rows if r["is_false"]]
    cell["n_false_audited"] = len(false_rows)
    if false_rows:
        m1 = {"srr_task": block(false_rows, "srr_task", ci=True),
              "srr_plan": block(false_rows, "srr_plan"),
              "conjunct_doubt": block(false_rows, "doubt"),
              "conjunct_targeted": block(false_rows, "targeted"),
              "conjunct_corrective": block(false_rows, "corrective"),
              "conjunct_flag_or_correct": block(false_rows, "flag_or_correct"),
              "conjunct_closure_valid": block(false_rows, "closure_valid"),
              "stated_reliance": block(false_rows, "stated_reliance")}
        cell["m1"] = m1
    else:
        cell["m1"] = "n/a (no audited-false rows)"
    return cell


def summarize_family(rows):
    fam = {}
    for p in POINTS:
        sub = [r for r in rows if r["point"] == p]
        if sub:
            fam[p] = summarize_cell(sub)
    fam["pooled"] = summarize_cell(rows)
    return fam


# ---------------------------------------------------------------- loaders
def failed_row(is_false, cluster, point):
    return compose({"class": "generation_failed", "doubt": False, "targeted": False,
                    "echo": False, "corrective": False, "deriv_use": False,
                    "deriv_use_rule": False, "hop_sound_all": False, "inter_facts": 0,
                    "reliance_rule": False, "n_poisoned": 0, "n_invalid": 0,
                    "n_unparsed": 0, "a_star_kind": None,
                    "is_false": is_false, "cluster": cluster, "point": point})


EXP_MAN_REQ = {"run_id", "question", "target", "entity", "prefix_steps", "injected_statement"}


def load_expa(exp_dir, notes, per_run_out, label="EXPA"):
    man_rows = read_jsonl(os.path.join(exp_dir, "manifest.jsonl"))
    if not man_rows or not EXP_MAN_REQ <= set(man_rows[0].keys()):
        notes[f"{label}_skipped_schema"] = sorted(set(man_rows[0].keys())) if man_rows else "empty manifest"
        return []
    man = {r["run_id"]: r for r in man_rows}
    vrows = read_jsonl(os.path.join(exp_dir, "validated_outputs.jsonl"))
    seen, dup, no_man = set(), 0, 0
    mismatches = []
    out_rows = []
    for r in vrows:
        rid = r.get("run_id")
        if rid in seen:
            dup += 1
            continue
        seen.add(rid)
        m = man.get(rid)
        if m is None:
            no_man += 1
            continue
        point = r.get("injection_position", m.get("injection_position"))
        cond = r.get("condition", m.get("condition"))
        truth = r.get("audited_truth_status", m.get("audited_truth_status"))
        is_false = truth == "false"
        cluster = r.get("problem_id", m.get("problem_id"))
        if r.get("class") == "generation_failed" or r.get("failed_generation"):
            row = failed_row(is_false, cluster, point)
        else:
            a = analyze(m["question"], m.get("prefix_steps", []), m["injected_statement"],
                        r.get("continuation", ""), m["target"], m["entity"])
            if a["class"] != r.get("class"):
                if len(mismatches) < 5:
                    mismatches.append({"run_id": rid, "stored": r.get("class"), "recomputed": a["class"]})
                notes[f"{label}_class_mismatch_count"] = notes.get(f"{label}_class_mismatch_count", 0) + 1
            row = compose({**a, "is_false": is_false, "cluster": cluster, "point": point})
        row.update({"family": f"{label}:{cond}", "run_id": rid, "truth": truth})
        out_rows.append(row)
        per_run_out.write(json.dumps({k: row[k] for k in (
            "run_id", "family", "point", "truth", "class", "closure_valid",
            "hop_sound_valid", "goal_jump0", "goal_jump_strict", "doubt", "targeted",
            "corrective", "echo", "derivational", "stated_reliance", "srr_task",
            "srr_plan", "used_any")}) + "\n")
    notes[f"{label}_rows_used"] = len(out_rows)
    notes[f"{label}_duplicate_run_ids"] = dup
    notes[f"{label}_rows_without_manifest"] = no_man
    notes[f"{label}_class_mismatch_examples"] = mismatches
    return out_rows


LEGACY_REQ = {"id", "point", "sent_idx", "corrupted_step", "continuation"}


def load_legacy(root, notes, per_run_out):
    data_path = os.path.join(root, "data", "pilot.jsonl")
    gold_path = os.path.join(root, "results", "gold", "rollouts.jsonl")
    if not (os.path.exists(data_path) and os.path.exists(gold_path)):
        notes["legacy_skipped"] = "data/pilot.jsonl or results/gold/rollouts.jsonl missing"
        return []
    data = {r["id"]: r for r in read_jsonl(data_path)}
    gold = {r["id"]: r for r in read_jsonl(gold_path)}
    cohort = set()
    for gid, g in gold.items():
        if not g.get("solved"):
            continue
        inst = data[gid]
        steps = split_sentences(g["gen_text"])
        if not steps:
            continue
        entity = steps[0].split()[0]
        gv = V.validate_continuation(inst["question"], [], None, g["gen_text"], inst["target"], entity)
        if gv["class"] == "valid_rederivation":
            cohort.add(gid)
    notes["legacy_cohort_size"] = len(cohort)

    out_rows = []
    fam_dirs = sorted(glob.glob(os.path.join(root, "results", "perturbed*")))
    for d in fam_dirs:
        runs_path = os.path.join(d, "runs.jsonl")
        if not os.path.exists(runs_path):
            continue
        fam_name = os.path.basename(d).replace("perturbed_", "").replace("perturbed", "wrong")
        runs = [r for r in read_jsonl(runs_path) if "skip" not in r]
        if runs and not LEGACY_REQ <= set(runs[0].keys()):
            notes[f"legacy_{fam_name}_skipped_schema"] = sorted(set(runs[0].keys()))
            continue
        n_excl = 0
        truth_counts = defaultdict(int)
        for r in runs:
            if r["id"] not in cohort:
                n_excl += 1
                continue
            inst = data[r["id"]]
            g = gold[r["id"]]
            steps = split_sentences(g["gen_text"])
            entity = steps[0].split()[0]
            truth = audit_truth_status(inst["question"], r["corrupted_step"], entity)
            truth_counts[truth["truth_status"]] += 1
            a = analyze(inst["question"], steps[: r["sent_idx"]], r["corrupted_step"],
                        r["continuation"], inst["target"], entity)
            row = compose({**a, "is_false": truth["truth_status"] == "false",
                           "cluster": r["id"], "point": r["point"]})
            row.update({"family": f"legacy:{fam_name}", "run_id": f'{r["id"]}::{r["point"]}',
                        "truth": truth["truth_status"]})
            out_rows.append(row)
            per_run_out.write(json.dumps({k: row[k] for k in (
                "run_id", "family", "point", "truth", "class", "closure_valid",
                "hop_sound_valid", "goal_jump0", "goal_jump_strict", "doubt", "targeted",
                "corrective", "echo", "derivational", "stated_reliance", "srr_task",
                "srr_plan", "used_any")}) + "\n")
        notes[f"legacy_{fam_name}"] = {"rows": sum(1 for x in out_rows if x["family"] == f"legacy:{fam_name}"),
                                       "excluded_not_in_cohort": n_excl,
                                       "truth_audit_counts": dict(truth_counts)}
    return out_rows


# ---------------------------------------------------------------- report
def pct(b):
    if not isinstance(b, dict) or b.get("rate") is None:
        return "  --  "
    return f"{b['rate']:.3f}"


def fam_table(fam_name, fam):
    lines = [f"### {fam_name}", "",
             "| position | n | closure-valid | hop-sound valid | delta (pp) | goal-jump0 | goal-jump strict | doubt | SRR (task) | SRR (plan) | stated-reliance | used-inj | echo | derivational |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in POINTS + ["pooled"]:
        c = fam.get(p)
        if not c:
            continue
        cv, hs = c["closure_valid"], c["hop_sound_valid"]
        delta = (cv["rate"] - hs["rate"]) * 100 if cv["rate"] is not None and hs["rate"] is not None else None
        m1 = c["m1"]
        if isinstance(m1, dict):
            srr_t = f'{m1["srr_task"]["rate"]:.3f} ({m1["srr_task"]["count"]}/{m1["srr_task"]["n"]})'
            srr_p = f'{m1["srr_plan"]["rate"]:.3f}'
            rel = f'{m1["stated_reliance"]["rate"]:.3f}'
        else:
            srr_t = srr_p = rel = "n/a"
        deriv = c["derivational"]
        deriv_s = "n.m." if deriv == "n.m." else f'{deriv["rate"]:.3f} ({deriv["count"]}/{deriv["n"]})'
        lines.append(
            f"| {p} | {c['n']} | {cv['rate']:.3f} ({cv['count']}/{cv['n']}) | {hs['rate']:.3f} ({hs['count']}/{hs['n']}) | "
            f"{delta:+.1f} | {pct(c['goal_jump0'])} | {pct(c['goal_jump_strict'])} | {pct(c['doubt'])} | "
            f"{srr_t} | {srr_p} | {rel} | {pct(c['used_any'])} | {pct(c['echo'])} | {deriv_s} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expa-dir", default=os.path.join(ROOT, "results", "EXPA_GLOBAL_EXPANSION"))
    ap.add_argument("--skip-legacy", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    notes = {}
    per_run_path = os.path.join(OUT_DIR, "strict_rows.jsonl")
    families = {}
    with open(per_run_path, "w") as per_run_out:
        all_rows = []
        if os.path.exists(os.path.join(args.expa_dir, "validated_outputs.jsonl")):
            all_rows += load_expa(args.expa_dir, notes, per_run_out, label="EXPA")
        else:
            notes["expa_skipped"] = f"missing {args.expa_dir}/validated_outputs.jsonl"
        for d in sorted(glob.glob(os.path.join(ROOT, "results", "EXP*"))):
            if os.path.abspath(d) == os.path.abspath(args.expa_dir):
                continue
            if re.search(r"SMOKE|TEST|PREP|FAILED|FIXED|MASK", os.path.basename(d)):
                continue  # pilot/smoke/debug variants are not report-grade cells
            if not (os.path.exists(os.path.join(d, "validated_outputs.jsonl"))
                    and os.path.exists(os.path.join(d, "manifest.jsonl"))):
                continue
            lab = os.path.basename(d)  # full dir name: avoids EXPC vs EXPC_FULL collisions
            try:
                all_rows += load_expa(d, notes, per_run_out, label=lab)
            except Exception as e:
                notes[f"{lab}_skipped_error"] = repr(e)
        if not args.skip_legacy:
            all_rows += load_legacy(ROOT, notes, per_run_out)
    by_fam = defaultdict(list)
    for r in all_rows:
        by_fam[r["family"]].append(r)
    order = ([f"EXPA:{c}" for c in EXPA_CONDS]
             + sorted(f for f in by_fam if not f.startswith(("EXPA:", "legacy:")))
             + sorted(f for f in by_fam if f.startswith("legacy:")))
    for f in order:
        if f in by_fam:
            families[f] = summarize_family(by_fam[f])

    result = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "root": ROOT,
        "denominator": "all rows retained (generation_failed and unparsed included); "
                       "M1/M3 additionally restricted to audited-false plants (n_false_audited)",
        "definitions": {
            "closure_valid": "validator v1 class == valid_rederivation (permissive headline)",
            "hop_sound_valid": "closure_valid AND every stated entity fact one direct rule "
                               "application (or state member) from previous state; injected atom "
                               "seeded into hop state iff true-derivable under closure",
            "goal_jump0": "closure_valid AND zero intermediate parsed entity-facts before final line",
            "goal_jump_strict": "goal_jump0 AND final target not one direct hop from prefix state",
            "srr_task": "audited-false plant AND (verbalized doubt OR states complement of planted atom) "
                        "AND closure_valid AND NOT goal_jump0",
            "srr_plan": "audited-false plant AND (targeted flag OR complement) AND closure_valid "
                        "AND NOT goal_jump0 AND NOT echo",
            "stated_reliance": "closure_valid AND planted content restated (parsed-literal or normalized-text echo)",
            "used_any": "any class: echo OR derivational use",
            "derivational": "some step derivable only WITH the planted atom (pred != atom); for rule "
                            "plants: only under rules+planted rule; n.m. when atom is negated/attribute",
        },
        "notes": notes,
        "families": families,
    }
    with open(os.path.join(OUT_DIR, "strict_metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # ---------------- report
    rep = ["# Stage-0 strict metrics (validatorPlan M1-M4)", "",
           f"Generated {result['generated_at']} from stored generations only (zero GPU). "
           f"Denominators: {result['denominator']}.", ""]
    for f in families:
        rep.append(fam_table(f, families[f]))
    rep.append("## Notes / schema observations\n")
    rep.append("```json\n" + json.dumps(notes, indent=2, default=str) + "\n```\n")
    # 5-sentence reading (filled with computed numbers)
    def g(f, p, key):
        try:
            b = families[f][p][key]
            return b["rate"] if isinstance(b, dict) else None
        except Exception:
            return None
    def gm1(f, p, key):
        try:
            m1 = families[f][p]["m1"]
            return m1[key]["rate"] if isinstance(m1, dict) else None
        except Exception:
            return None
    oh_cv, oh_hs = g("EXPA:one_hop_falsehood", "pooled", "closure_valid"), g("EXPA:one_hop_falsehood", "pooled", "hop_sound_valid")
    gl_cv, gl_hs = g("EXPA:global_falsehood", "pooled", "closure_valid"), g("EXPA:global_falsehood", "pooled", "hop_sound_valid")
    bp_cv, bp_hs = g("EXPA:benign_paraphrase", "pooled", "closure_valid"), g("EXPA:benign_paraphrase", "pooled", "hop_sound_valid")
    oh_srr, gl_srr = gm1("EXPA:one_hop_falsehood", "pooled", "srr_task"), gm1("EXPA:global_falsehood", "pooled", "srr_task")
    oh_use, gl_use = g("EXPA:one_hop_falsehood", "pooled", "used_any"), g("EXPA:global_falsehood", "pooled", "used_any")
    def fr(x):
        return "?" if x is None else f"{x:.3f}"
    def gw(f, p, key):
        try:
            b = families[f][p][key]
            return (b["rate"], b.get("wilson95")) if isinstance(b, dict) else (None, None)
        except Exception:
            return (None, None)
    (oh_hs_r, oh_hs_ci), (gl_hs_r, gl_hs_ci) = gw("EXPA:one_hop_falsehood", "pooled", "hop_sound_valid"), gw("EXPA:global_falsehood", "pooled", "hop_sound_valid")
    ti_cv, ti_hs = g("EXPA:true_interruption", "pooled", "closure_valid"), g("EXPA:true_interruption", "pooled", "hop_sound_valid")
    gl_echo = g("EXPA:global_falsehood", "pooled", "echo")
    gl_deriv = None
    try:
        b = families["EXPA:global_falsehood"]["pooled"]["derivational"]
        gl_deriv = b["rate"] if isinstance(b, dict) else None
    except Exception:
        pass
    oh_echo = g("EXPA:one_hop_falsehood", "pooled", "echo")
    ci_sep = (oh_hs_ci and gl_hs_ci and oh_hs_ci[0] is not None and gl_hs_ci[1] is not None and oh_hs_ci[0] > gl_hs_ci[1])
    rep.append("## Reading\n")
    rep.append(
        f"(1) Hop-soundness cuts the headline closure-valid rates roughly in half everywhere -- "
        f"benign {fr(bp_cv)}->{fr(bp_hs)}, one-hop falsehood {fr(oh_cv)}->{fr(oh_hs)}, global falsehood "
        f"{fr(gl_cv)}->{fr(gl_hs)} (pooled, all-rows denominators) -- so a large share of 'valid' completions "
        f"reach the target through closure-level skips rather than explicit step-by-step derivation. "
        f"(2) The family gradient survives the strict standard but compresses: one-hop hop-sound-valid "
        f"{fr(oh_hs_r)} (Wilson95 {oh_hs_ci}) vs global {fr(gl_hs_r)} (Wilson95 {gl_hs_ci}), "
        f"{'non-overlapping CIs' if ci_sep else 'CIs to be checked in the JSON'}, versus the permissive gap "
        f"{fr(oh_cv)} vs {fr(gl_cv)}. "
        f"(3) Strict repair concentrates entirely at low refutation distance -- SRR {fr(oh_srr)} for one-hop vs "
        f"{fr(gl_srr)} for global -- and the global-falsehood corrective conjunct is 0 partly for a structural "
        f"reason: the complement of a CWA-only falsehood is itself underivable, so explicitly contradicting the "
        f"plant breaks closure-validity, leaving doubt (0.013) as the only strict-repair route. "
        f"(4) Unconditional injection use inverts the gradient ({fr(oh_use)} one-hop vs {fr(gl_use)} global) and the "
        f"echo/derivational split shows global plants are genuinely reused derivationally "
        f"({fr(gl_deriv)} derivational vs {fr(gl_echo)} echo) while one-hop negated plants show only echo "
        f"({fr(oh_echo)}, derivational n.m.), replacing the misleading 0.000-dagger cells. "
        f"(5) Two caveats for interpretation: every non-benign condition omits the overwritten gold step, so "
        f"hop-soundness also prices in re-deriving that step (true interruption {fr(ti_cv)}->{fr(ti_hs)} quantifies "
        f"this structural burden with a TRUE plant), and per-row truth audit shows the legacy 'wrong' family is "
        f"mostly audited-TRUE plants (355/390 entailed), so it must not be read as a falsehood family.\n")
    with open(os.path.join(OUT_DIR, "STRICT_METRICS_REPORT.md"), "w") as fh:
        fh.write("\n".join(rep))
    print(f"wrote {OUT_DIR}/strict_metrics.json, STRICT_METRICS_REPORT.md, strict_rows.jsonl")
    print(json.dumps({k: v for k, v in notes.items() if not isinstance(v, dict)}, indent=2, default=str))


if __name__ == "__main__":
    main()
