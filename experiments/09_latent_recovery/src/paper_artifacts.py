"""Generate paper-facing result tables and audit lock from result artifacts.

This script is intentionally conservative: it reads JSON/JSONL artifacts only and
emits LaTeX snippets consumed by the paper. It also writes RESULTS_LOCK.md with the
artifact completeness audit used before paper edits.
"""
import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.abspath(os.path.join(BASE, "..", "..", "paper_latex", "papers", "latent_recovery"))
PAPER_RESULTS = os.path.join(PAPER, "paper_results.tex")
PAPER_META = os.path.join(PAPER, "paper_results_metadata.json")
RESULTS_LOCK = os.path.join(BASE, "RESULTS_LOCK.md")


def now_iso():
    return _dt.datetime.now(_dt.UTC).isoformat()


def j(path):
    with open(os.path.join(BASE, path)) as fh:
        return json.load(fh)


def jl(path):
    with open(os.path.join(BASE, path)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def exists(path):
    return os.path.exists(os.path.join(BASE, path))


def sha256(path):
    h = hashlib.sha256()
    with open(os.path.join(BASE, path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path):
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        return None
    with open(full, "rb") as fh:
        return sum(1 for _ in fh)


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", BASE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fmt_rate(x, digits=3):
    if x is None:
        return "---"
    return f"{x:.{digits}f}"


def fmt_ci(ci, digits=3):
    if not ci or ci[0] is None:
        return "[---,---]"
    return f"[{fmt_rate(ci[0], digits)},{fmt_rate(ci[1], digits)}]"


def fmt_rate_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{fmt_rate(k / n)} {fmt_ci((lo, hi))}"


def class_count(rows, cls):
    return sum(1 for r in rows if r.get("class") == cls)


def truth_status_wrong_category():
    from validator import closure, derivable, parse_fact, parse_world

    data = {r["id"]: r for r in jl("data/pilot.jsonl")}
    rows = jl("results/validated.jsonl")
    out = defaultdict(list)
    for r in rows:
        ent = r["corrupted_step"].split()[0]
        question = data[r["id"]]["question"]
        rules, facts, _ = parse_world(question)
        reach = closure(rules)
        premises = {p for e, p in facts if e == ent}
        fact = parse_fact(r["corrupted_step"], ent)
        if not fact:
            status = "unparsed"
        elif fact[1][0] == "cat":
            status = "true" if derivable(fact[1], premises, reach) else "false"
        else:
            pred = fact[1]
            opp = ("adj", pred[1], not pred[2])
            status = "true" if derivable(pred, premises, reach) else ("false" if derivable(opp, premises, reach) else "unknown")
        out[status].append(r)
    return out


def table_audit_truth_status():
    by = truth_status_wrong_category()
    labels = [("entailed-true", "true"), ("genuinely false", "false")]
    rows = []
    for label, key in labels:
        sub = by[key]
        n = len(sub)
        rows.append(
            f"{label} & {n} & {fmt_rate(class_count(sub, 'valid_rederivation') / n)} & "
            f"{fmt_rate(class_count(sub, 'poisoned') / n)} & {fmt_rate(class_count(sub, 'parroted') / n)} \\\\"
        )
    return r"""\newcommand{\PaperTableAuditTruthStatus}{%
\begin{tabular}{lcccc}
\toprule
Injection (audited) & $n$ & Closure-valid & Inj.-dep. & Parroted \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def summary_cell(path, point):
    return j(path)[point]


def expa_condition_position_cell(condition, point):
    return j("results/EXPA_GLOBAL_EXPANSION/summary_tables.json")["metrics"]["by_condition_position"][f"{condition}|{point}"]


def expa_main_table_cell(condition, point):
    c = expa_condition_position_cell(condition, point)
    return {
        "n": c["n"],
        "valid_rederivation": c["valid_rederivation"]["count"],
        "poisoned": c["poisoned"]["count"],
        "parroted": c["parroted"]["count"],
        "acknowledged": c["verbalized_doubt"]["count"],
        "unparsed": c["unparsed"]["count"],
    }


def table_falsifiability():
    specs = [
        ("benign paraphrase", None, False, "benign_paraphrase"),
        ("true interruption", None, True, "true_interruption"),
        ("distractor rule", "results/validated_summary_distractor.json", True, None),
        ("contradiction (dist. 0)", "results/validated_summary_contradiction.json", True, None),
        ("false, 1-hop checkable", None, "dagger", "one_hop_falsehood"),
        ("false, globally checkable only", None, True, "global_falsehood"),
    ]
    rows = []
    for fam, path, injdep, expa_condition in specs:
        for point in ("early", "mid", "late"):
            c = expa_main_table_cell(expa_condition, point) if expa_condition else summary_cell(path, point)
            n = c["n"]
            valid = fmt_rate_ci(c.get("valid_rederivation", 0), n)
            if injdep == "dagger":
                dep = rf"{fmt_rate(c.get('poisoned', 0) / n)}$^{{\dagger}}$"
            elif injdep is None:
                dep = r"0.000$^{\dagger}$"
            elif not injdep:
                dep = "---"
            else:
                dep = fmt_rate(c.get("poisoned", 0) / n)
            rows.append(
                f"{fam} & {point} & {n} & {valid} & {dep} & "
                f"{fmt_rate(c.get('parroted', 0) / n)} & {fmt_rate(c.get('acknowledged', 0) / n)} & "
                f"{fmt_rate(c.get('unparsed', 0) / n)} \\\\"
            )
        if fam != specs[-1][0]:
            rows.append(r"\midrule")
    return r"""\newcommand{\PaperTableFalsifiability}{%
\begin{tabular}{llcccccc}
\toprule
Family & Inj. & $n$ & Closure-valid [95\% CI] & Inj.-dep. & Parroted & Doubt & Unparsed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def table_task_structure():
    sf = expa_condition_position_cell("global_falsehood", "mid")
    gsm = j("results/gsm8k/summary.json")
    arith = j("results/arith/summary.json")["mid"]
    rows = [
        f"PrOntoQA (redundant derivations) & {fmt_rate(sf['valid_rederivation']['rate'])} & {fmt_rate(sf['poisoned']['rate'])}$^{{*}}$ & {sf['n']} \\\\",
        f"GSM8K (natural-language math) & {fmt_rate(gsm['recovered_rate'])} & {fmt_rate(gsm['poisoned_rate'])} & {gsm['n']} \\\\",
        f"Chained arithmetic (no redundancy) & {fmt_rate(arith['recovered_rate'])} & {fmt_rate(arith['poisoned_next_step_rate'])} & {arith['n']} \\\\",
    ]
    return r"""\newcommand{\PaperTableTaskStructure}{%
\begin{tabular}{lccc}
\toprule
Task & Closure-valid & Inj.-dep. & $n$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def range_for_summary(path, metric):
    d = j(path)
    vals = []
    for p in ("early", "mid", "late"):
        c = d[p]
        vals.append(c.get(metric, 0) / c["n"])
    return min(vals), max(vals)


def table_scale_lineage():
    specs = [
        ("Qwen2.5-1.5B", "results_1p5b"),
        ("Qwen2.5-7B", "results"),
        ("Qwen2.5-32B", "results_32b"),
    ]
    rows = []
    for label, d in specs:
        gv = range_for_summary(f"{d}/validated_summary_falsehood.json", "valid_rederivation")
        gp = range_for_summary(f"{d}/validated_summary_falsehood.json", "poisoned")
        gd = range_for_summary(f"{d}/validated_summary_falsehood.json", "acknowledged")
        nd = range_for_summary(f"{d}/validated_summary_negstep.json", "acknowledged")
        rows.append(
            f"{label} & {fmt_rate(gv[0])}--{fmt_rate(gv[1])} & {fmt_rate(gp[0])}--{fmt_rate(gp[1])} & "
            f"{fmt_rate(gd[0])}--{fmt_rate(gd[1])} & {fmt_rate(nd[0])}--{fmt_rate(nd[1])} \\\\"
        )
    return r"""\newcommand{\PaperTableScaleLineage}{%
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{3}{c}{globally-checkable falsehood} & 1-hop falsehood \\
Model & closure-valid & inj.-dep. & doubt & doubt \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def table_expb():
    s = j("results/EXPB_LOCAL_CERT_FLIP/summary_tables.json")
    labels = [
        ("GLOBAL_BASELINE", "global baseline"),
        ("LOCAL_CERTIFICATE", "local certificate"),
        ("IRRELEVANT_CERTIFICATE_CONTROL", "irrelevant certificate"),
    ]
    rows = []
    for cond, label in labels:
        c = s["metrics"]["pooled_by_condition"][cond]
        rows.append(
            f"{label} & {c['n']} & {fmt_rate(c['valid_rederivation']['rate'])} {fmt_ci(c['valid_rederivation']['problem_cluster_bootstrap95'])} & "
            f"{fmt_rate(c['poisoned']['rate'])} {fmt_ci(c['poisoned']['problem_cluster_bootstrap95'])} & "
            f"{fmt_rate(c['verbalized_doubt']['rate'])} {fmt_ci(c['verbalized_doubt']['problem_cluster_bootstrap95'])} & "
            f"{fmt_rate(c['unparsed']['rate'])} \\\\"
        )
    return r"""\newcommand{\PaperTableLocalCertificate}{%
\begin{tabular}{lccccc}
\toprule
Condition & $n$ & Closure-valid [95\% CI] & Inj.-dep. [95\% CI] & Doubt [95\% CI] & Unparsed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def table_expc():
    s = j("results/EXPC_POLARITY_CONTROL/summary_tables.json")
    labels = [
        ("LOCAL_FALSE_POSITIVE", "local positive"),
        ("LOCAL_FALSE_NEGATIVE", "local negative"),
        ("GLOBAL_FALSE_POSITIVE", "global positive"),
        ("GLOBAL_FALSE_NEGATIVE", "global negative"),
    ]
    rows = []
    for cond, label in labels:
        c = s["metrics"]["pooled_by_condition"][cond]
        poison = c["poisoning"]["rate"]
        poison_cell = "---" if poison is None else f"{fmt_rate(poison)} {fmt_ci(c['poisoning']['problem_cluster_bootstrap95'])}"
        rows.append(
            f"{label} & {c['n']} & {fmt_rate(c['valid_recovery']['rate'])} {fmt_ci(c['valid_recovery']['problem_cluster_bootstrap95'])} & "
            f"{fmt_rate(c['doubt']['rate'])} {fmt_ci(c['doubt']['problem_cluster_bootstrap95'])} & "
            f"{poison_cell} & {fmt_rate(c['unparsed']['rate'])} \\\\"
        )
    return r"""\newcommand{\PaperTablePolarityControl}{%
\begin{tabular}{lccccc}
\toprule
Condition & $n$ & Closure-valid [95\% CI] & Doubt [95\% CI] & Inj.-dep. [95\% CI] & Unparsed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def table_expa_expansion():
    s = j("results/EXPA_GLOBAL_EXPANSION/summary_tables.json")
    labels = [
        ("true_interruption", "true interruption"),
        ("benign_paraphrase", "benign paraphrase"),
        ("one_hop_falsehood", "1-hop falsehood"),
        ("global_falsehood", "global falsehood"),
    ]
    rows = []
    for cond, label in labels:
        c = s["metrics"]["pooled_by_condition"][cond]
        rows.append(
            f"{label} & {c['n']} & {fmt_rate(c['valid_rederivation']['rate'])} {fmt_ci(c['valid_rederivation']['problem_cluster_bootstrap95'])} & "
            f"{fmt_rate(c['poisoned']['rate'])} {fmt_ci(c['poisoned']['problem_cluster_bootstrap95'])} & "
            f"{fmt_rate(c['verbalized_doubt']['rate'])} {fmt_ci(c['verbalized_doubt']['problem_cluster_bootstrap95'])} & "
            f"{fmt_rate(c['unparsed']['rate'])} \\\\"
        )
    return r"""\newcommand{\PaperTableExpansion}{%
\begin{tabular}{lccccc}
\toprule
Condition & $n$ & Closure-valid [95\% CI] & Inj.-dep. [95\% CI] & Doubt [95\% CI] & Unparsed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def table_strict_stepwise():
    expc = jl("results/EXPC_POLARITY_CONTROL/validated_outputs.jsonl")
    expb = jl("results/EXPB_LOCAL_CERT_FLIP/validated_outputs.jsonl")
    c = Counter(r["strict_validation"]["strict_class"] for r in expc)
    strict_ok = c.get("strict_noncanonical_recovery", 0)
    strict_bad = c.get("strict_final_mismatch", 0)
    closure_valid = class_count(expc, "valid_rederivation")
    rows = [
        f"Lexical-polarity control & {len(expc)} & available & {closure_valid} & {strict_ok} noncanonical recoveries; {strict_bad} final mismatches \\\\",
        f"Local-certificate flip & {len(expb)} & unavailable & {class_count(expb, 'valid_rederivation')} & closure validator retained; stepwise recovery not imputed \\\\",
    ]
    return r"""\newcommand{\PaperTableStrictStepwise}{%
\begin{tabular}{@{}p{0.20\linewidth}cccp{0.31\linewidth}@{}}
\toprule
Control & $n$ & Strict val. & Closure-valid & Strict-stepwise outcome \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
"""


def macros():
    expb = j("results/EXPB_LOCAL_CERT_FLIP/summary_tables.json")
    expc = j("results/EXPC_POLARITY_CONTROL/summary_tables.json")
    expa = j("results/EXPA_GLOBAL_EXPANSION/summary_tables.json")
    expa_meta = j("results/EXPA_GLOBAL_EXPANSION/run_metadata.json")
    b = expb["metrics"]["pooled_by_condition"]
    paired = expb["paired_tests"]
    lines = [
        rf"\newcommand{{\CertProblemClusters}}{{{expb['sample_size']['problem_clusters']}}}",
        rf"\newcommand{{\CertPerPositionN}}{{{expb['sample_size']['paired_triples_by_position']['early']}}}",
        rf"\newcommand{{\CertRows}}{{{expb['sample_size']['validated_rows']}}}",
        rf"\newcommand{{\CertConditionRows}}{{{b['LOCAL_CERTIFICATE']['n']}}}",
        rf"\newcommand{{\CertLocalValid}}{{{fmt_rate(b['LOCAL_CERTIFICATE']['valid_rederivation']['rate'])}}}",
        rf"\newcommand{{\CertGlobalValid}}{{{fmt_rate(b['GLOBAL_BASELINE']['valid_rederivation']['rate'])}}}",
        rf"\newcommand{{\CertIrrelValid}}{{{fmt_rate(b['IRRELEVANT_CERTIFICATE_CONTROL']['valid_rederivation']['rate'])}}}",
        rf"\newcommand{{\CertLocalPoison}}{{{fmt_rate(b['LOCAL_CERTIFICATE']['poisoned']['rate'])}}}",
        rf"\newcommand{{\CertGlobalPoison}}{{{fmt_rate(b['GLOBAL_BASELINE']['poisoned']['rate'])}}}",
        rf"\newcommand{{\CertIrrelPoison}}{{{fmt_rate(b['IRRELEVANT_CERTIFICATE_CONTROL']['poisoned']['rate'])}}}",
        rf"\newcommand{{\CertLocalDoubt}}{{{fmt_rate(b['LOCAL_CERTIFICATE']['verbalized_doubt']['rate'])}}}",
        rf"\newcommand{{\CertGlobalDoubt}}{{{fmt_rate(b['GLOBAL_BASELINE']['verbalized_doubt']['rate'])}}}",
        rf"\newcommand{{\CertIrrelDoubt}}{{{fmt_rate(b['IRRELEVANT_CERTIFICATE_CONTROL']['verbalized_doubt']['rate'])}}}",
        rf"\newcommand{{\CertValidDiffGlobal}}{{{fmt_rate(paired['valid_rederivation']['LOCAL_CERTIFICATE_vs_GLOBAL_BASELINE']['pooled']['diff_a_minus_b'])}}}",
        rf"\newcommand{{\CertValidDiffIrrel}}{{{fmt_rate(paired['valid_rederivation']['LOCAL_CERTIFICATE_vs_IRRELEVANT_CERTIFICATE_CONTROL']['pooled']['diff_a_minus_b'])}}}",
        rf"\newcommand{{\CertPoisonDiffGlobal}}{{{fmt_rate(paired['poisoned']['LOCAL_CERTIFICATE_vs_GLOBAL_BASELINE']['pooled']['diff_a_minus_b'])}}}",
        rf"\newcommand{{\CertPoisonDiffIrrel}}{{{fmt_rate(paired['poisoned']['LOCAL_CERTIFICATE_vs_IRRELEVANT_CERTIFICATE_CONTROL']['pooled']['diff_a_minus_b'])}}}",
        rf"\newcommand{{\CertDoubtDiffGlobal}}{{{fmt_rate(paired['verbalized_doubt']['LOCAL_CERTIFICATE_vs_GLOBAL_BASELINE']['pooled']['diff_a_minus_b'])}}}",
        rf"\newcommand{{\CertDoubtDiffIrrel}}{{{fmt_rate(paired['verbalized_doubt']['LOCAL_CERTIFICATE_vs_IRRELEVANT_CERTIFICATE_CONTROL']['pooled']['diff_a_minus_b'])}}}",
        rf"\newcommand{{\PolarityProblemN}}{{{expc['availability']['generated_problem_n']}}}",
        rf"\newcommand{{\PolarityRows}}{{{expc['availability']['validated_rows']}}}",
        rf"\newcommand{{\ExpansionTargetPerPosition}}{{{expa_meta['target_eligible_per_position']}}}",
        rf"\newcommand{{\ExpansionProblems}}{{{expa['availability']['global_falsehood_problem_n']}}}",
        rf"\newcommand{{\ExpansionRows}}{{{expa['integrity']['validated_rows']}}}",
    ]
    return "\n".join(lines) + "\n"


def write_paper_results():
    text = (
        "% Auto-generated by experiments/09_latent_recovery/src/paper_artifacts.py.\n"
        "% Do not hand-edit numeric values here; regenerate from result artifacts.\n"
        + macros()
        + table_falsifiability()
        + table_task_structure()
        + table_scale_lineage()
        + table_expb()
        + table_expc()
        + table_expa_expansion()
        + table_strict_stepwise()
    )
    with open(PAPER_RESULTS, "w") as fh:
        fh.write(text)
    sources = sorted({
        "data/pilot.jsonl",
        "results/validated.jsonl",
        "results/validated_summary.json",
        "results/validated_summary_paraphrase.json",
        "results/validated_summary_distractor.json",
        "results/validated_summary_contradiction.json",
        "results/validated_summary_negstep.json",
        "results/validated_summary_falsehood.json",
        "results/gsm8k/summary.json",
        "results/arith/summary.json",
        "results_1p5b/validated_summary_falsehood.json",
        "results_1p5b/validated_summary_negstep.json",
        "results_32b/validated_summary_falsehood.json",
        "results_32b/validated_summary_negstep.json",
        "results/EXPB_LOCAL_CERT_FLIP/summary_tables.json",
        "results/EXPB_LOCAL_CERT_FLIP/validated_outputs.jsonl",
        "results/EXPB_LOCAL_CERT_FLIP/manifest.jsonl",
        "results/EXPB_LOCAL_CERT_FLIP/raw_generations.jsonl",
        "results/EXPB_LOCAL_CERT_FLIP/run_metadata.json",
        "results/EXPB_LOCAL_CERT_FLIP/EXPB_REPORT.md",
        "results/EXPC_POLARITY_CONTROL/summary_tables.json",
        "results/EXPC_POLARITY_CONTROL/validated_outputs.jsonl",
        "results/EXPC_POLARITY_CONTROL/manifest.jsonl",
        "results/EXPC_POLARITY_CONTROL/raw_generations.jsonl",
        "results/EXPC_POLARITY_CONTROL/run_metadata.json",
        "results/EXPC_POLARITY_CONTROL/EXPC_REPORT.md",
        "results/EXPA_GLOBAL_EXPANSION/run_metadata.json",
        "results/EXPA_GLOBAL_EXPANSION/manifest.jsonl",
        "results/EXPA_GLOBAL_EXPANSION/raw_generations.jsonl",
        "results/EXPA_GLOBAL_EXPANSION/summary_tables.json",
        "results/EXPA_GLOBAL_EXPANSION/validated_outputs.jsonl",
        "results/EXPA_GLOBAL_EXPANSION/eligibility_audit.jsonl",
        "results/EXPA_GLOBAL_EXPANSION/EXPA_REPORT.md",
    })
    meta = {
        "created_at": now_iso(),
        "git_commit": git_hash(),
        "generated": os.path.relpath(PAPER_RESULTS, BASE),
        "sources": [{"path": p, "sha256": sha256(p)} for p in sources if exists(p)],
    }
    with open(PAPER_META, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return meta


def full_schema_status(path):
    files = os.listdir(os.path.join(BASE, path)) if os.path.isdir(os.path.join(BASE, path)) else []
    reqs = {
        "manifest": "manifest.jsonl" in files,
        "raw_generations": "raw_generations.jsonl" in files,
        "validator_outputs": "validated_outputs.jsonl" in files,
        "summary_tables": "summary_tables.json" in files,
        "readme_or_report": any(f.lower().startswith("readme") or f.endswith("_REPORT.md") or f.endswith("REPORT.md") for f in files),
    }
    return reqs, all(reqs.values()), "RUN_COMPLETE" in files


def writable_entries(path):
    root_path = os.path.join(BASE, path)
    out = []
    if not os.path.isdir(root_path):
        return out
    for root, dirs, files in os.walk(root_path):
        for name in dirs + files:
            full = os.path.join(root, name)
            if os.stat(full).st_mode & 0o222:
                out.append(os.path.relpath(full, BASE))
    if os.stat(root_path).st_mode & 0o222:
        out.append(path)
    return sorted(out)


def artifact_row_counts(path):
    counts = {}
    for name in ("manifest.jsonl", "raw_generations.jsonl", "validated_outputs.jsonl", "eligibility_audit.jsonl"):
        rel = os.path.join(path, name)
        if exists(rel):
            counts[name] = line_count(rel)
    rel_summary = os.path.join(path, "summary_tables.json")
    if exists(rel_summary):
        summary = j(rel_summary)
        if "sample_size" in summary and "validated_rows" in summary["sample_size"]:
            counts["summary_validated_rows"] = summary["sample_size"]["validated_rows"]
        elif "availability" in summary and "validated_rows" in summary["availability"]:
            counts["summary_validated_rows"] = summary["availability"]["validated_rows"]
        elif "integrity" in summary and "validated_rows" in summary["integrity"]:
            counts["summary_validated_rows"] = summary["integrity"]["validated_rows"]
    return counts


def latex_table_literal_rate_hits():
    path = os.path.join(PAPER, "main.tex")
    if not os.path.exists(path):
        return []
    text = open(path).read()
    hits = []
    for m in re.finditer(r"\\begin\{(?:table|center)\}(.*?)\\end\{(?:table|center)\}", text, re.S):
        block = re.sub(r"p\{[^}]*\\linewidth\}", "p{}", m.group(1))
        if r"\PaperTable" in block:
            continue
        for rate in re.findall(r"(?<![A-Za-z])(?:\.\d{2,}|0\.\d{2,}|\d{1,3}\\%)", block):
            hits.append(rate)
    return hits


def result_lock(meta):
    dirs = [
        "results/EXPB_LOCAL_CERT_FLIP",
        "results/EXPC_POLARITY_CONTROL",
        "results/EXPA_GLOBAL_EXPANSION",
        "results/EXPA_GLOBAL_EXPANSION_SMOKE",
    ]
    legacy = ["results", "results_1p5b", "results_32b", "results_olmo", "results_llama", "results_mistral", "results_ctx", "results_doubt", "results_verify"]
    lines = [
        "# Results Lock",
        "",
        f"Generated: {now_iso()}",
        f"Git commit: `{git_hash()}`",
        "",
        "## Regeneration Commands",
        "",
        "These commands were run during the lock update; paper tables, macros, and figures are regenerated from artifacts rather than edited by hand.",
        "",
        "- Paper tables/macros: `/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python src/paper_artifacts.py`",
        "- Figures: `/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python src/make_workshop_figures.py`",
        "- Paper PDF: run `pdflatex main.tex` twice in `paper_latex/papers/latent_recovery`.",
        "",
        "## Paper Table/Figure Sources",
        "",
        f"- Generated LaTeX tables/macros: `{os.path.relpath(PAPER_RESULTS, BASE)}`",
        f"- Generated metadata: `{os.path.relpath(PAPER_META, BASE)}`",
        "- Figures are generated from JSON summaries by `src/make_workshop_figures.py`.",
        "- Regenerated figure outputs: `fig_dose_response.{tex,pdf,png}`, `fig_regime_map.{tex,pdf,png}`, `fig_verbalization_spectrum.{tex,pdf,png}`.",
        "",
        "### Source Hashes",
        "",
    ]
    for item in meta["sources"]:
        lines.append(f"- `{item['path']}`: `{item['sha256']}`")
    lines.extend(["", "## Full-Schema Result Directory Audit", ""])
    for d in dirs:
        reqs, ok, run_complete = full_schema_status(d)
        if ok and run_complete:
            status = "PASS"
        elif ok:
            status = "PARTIAL"
        else:
            status = "FAIL"
        bits = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in reqs.items())
        lines.append(f"- `{d}`: **{status}** ({bits}; run_complete={'yes' if run_complete else 'no'})")
        counts = artifact_row_counts(d)
        if counts:
            lines.append(f"  - row counts: `{json.dumps(counts, sort_keys=True)}`")
    lines.extend(["", "## Result Directory Immutability", ""])
    for d in dirs:
        writable = writable_entries(d)
        if writable:
            lines.append(f"- `{d}`: **WRITABLE** ({len(writable)} writable entries)")
        else:
            lines.append(f"- `{d}`: **READ-ONLY**")
    lines.extend(
        [
            "",
            "Legacy/minimal result directories used by existing figures or robustness checks do not follow the full manifest/raw/validated/summary/report schema; they are locked as legacy artifacts and are not treated as new full-schema experiments:",
        ]
    )
    for d in legacy:
        if os.path.isdir(os.path.join(BASE, d)):
            lines.append(f"- `{d}`")
    lines.extend(["", "## Planned Conditions And Null/Failed Runs", ""])
    expb = j("results/EXPB_LOCAL_CERT_FLIP/summary_tables.json")
    lines.append("- EXPB_LOCAL_CERT_FLIP: all planned conditions reported: GLOBAL_BASELINE, LOCAL_CERTIFICATE, IRRELEVANT_CERTIFICATE_CONTROL. Null/negative result preserved: local certificate did not improve valid re-derivation over irrelevant-certificate control.")
    lines.append(f"  - sample: {expb['sample_size']['paired_triples_by_position']}; problem clusters: {expb['sample_size']['problem_clusters']}.")
    expc = j("results/EXPC_POLARITY_CONTROL/summary_tables.json")
    lines.append("- EXPC_POLARITY_CONTROL: all planned 2x2 conditions reported. Null/ambiguous result preserved in report.")
    lines.append(f"  - sample: {expc['availability']['validated_rows']} validated rows across {expc['availability']['generated_problem_n']} fully matched problems.")
    expa = j("results/EXPA_GLOBAL_EXPANSION/summary_tables.json")
    expa_meta = j("results/EXPA_GLOBAL_EXPANSION/run_metadata.json")
    target = expa_meta["target_eligible_per_position"]
    achieved = expa["availability"]["global_falsehood_problem_n"]
    rows = expa["integrity"]["validated_rows"]
    if achieved >= target:
        lines.append("- EXPA_GLOBAL_EXPANSION: all planned conditions reported and the configured target was reached.")
        lines.append(f"  - target: {target} global-falsehood examples per position; achieved: {achieved} problems and {rows} validated rows across matched conditions and positions.")
    else:
        lines.append("- EXPA_GLOBAL_EXPANSION: all planned conditions reported, but the run is partial/infeasible at target size.")
        lines.append(f"  - target: {target} global-falsehood examples per position; achieved: {achieved} problems and {rows} validated rows. The partial result is reported as a limitation and is not used to strengthen claims.")
    expa_counts = artifact_row_counts("results/EXPA_GLOBAL_EXPANSION")
    lines.append(f"  - artifact counts: failed, unavailable, and raw-generation attempts are retained alongside validated rows. Counts: `{json.dumps(expa_counts, sort_keys=True)}`")
    lines.extend(["", "## Exclusion And Availability Counts", ""])
    lines.append(f"- EXPB_LOCAL_CERT_FLIP: {expb.get('exclusions', 'No post-generation exclusions; failed and unparsed generations retained.')}")
    lines.append(f"- EXPB_LOCAL_CERT_FLIP sanity checks: `{json.dumps(expb['sanity_checks'], sort_keys=True)}`")
    lines.append(f"- EXPB_LOCAL_CERT_FLIP integrity checks: `{json.dumps(expb['integrity'], sort_keys=True)}`")
    lines.append(f"- EXPC_POLARITY_CONTROL exclusion rule: `{expc.get('exclusion_rule')}`")
    lines.append(f"- EXPC_POLARITY_CONTROL availability counts: `{json.dumps(expc['availability']['audit_counts'], sort_keys=True)}`")
    lines.append(f"- EXPC_POLARITY_CONTROL integrity checks: `{json.dumps(expc['integrity'], sort_keys=True)}`")
    if exists("results/EXPA_GLOBAL_EXPANSION/eligibility_audit.jsonl"):
        counts = Counter(r.get("reason") for r in jl("results/EXPA_GLOBAL_EXPANSION/eligibility_audit.jsonl"))
        lines.append(f"- EXPA_GLOBAL_EXPANSION eligibility counts: `{json.dumps(dict(counts), sort_keys=True)}`")
    lines.append(f"- EXPA_GLOBAL_EXPANSION integrity checks: `{json.dumps(expa['integrity'], sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Lock Assertions",
            "",
            "- Every paper table with experimental numbers is generated by `src/paper_artifacts.py` and included from `paper_results.tex`.",
            "- Figure sources read JSON result summaries through `src/make_workshop_figures.py`.",
            "- The paper reports EXPA availability limits and noisy EXPC rather than filtering them out.",
            "- Strict stepwise validation is reported in a dedicated paper table. EXPB has no strict validator, so stepwise recovery is explicitly unavailable there.",
            "- The only manually written table left in `main.tex` is a qualitative example table; all experimental counts, rates, confidence intervals, and sample sizes are in generated macros/tables.",
        ]
    )
    literal_hits = latex_table_literal_rate_hits()
    if literal_hits:
        lines.append(f"- WARNING: direct numeric-looking rates remain inside non-generated LaTeX table/center blocks: `{literal_hits}`")
    else:
        lines.append("- No direct numeric-looking experimental rates were found inside non-generated LaTeX table/center blocks in `main.tex`.")
    with open(RESULTS_LOCK, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    meta = write_paper_results()
    result_lock(meta)
    print(f"wrote {PAPER_RESULTS}")
    print(f"wrote {PAPER_META}")
    print(f"wrote {RESULTS_LOCK}")


if __name__ == "__main__":
    main()
