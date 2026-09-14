#!/usr/bin/env python3
"""E17 prefix builder. For each ORIG-arm item of the renumbering replication, build four prefixes:
corrupted/clean x numeral-boundary/statement-boundary. Fail-closed filters per E17_STATEMENT_BOUNDARY.md.
Reads yee_memo/pairs.json + Yee baseline CoTs; writes scratch/e17_boundary/e17_items.json + build_log.json."""
import csv, json, os, re, collections
BASE = "/lfs/skampere2/0/eobbad/scratch"; OUT = BASE + "/e17_boundary"; os.makedirs(OUT, exist_ok=True)
MARKER = "A: Let's think step by step."; MODEL = "gpt-4-0314"; SUFFIX = " Therefore, the answer (arabic numerals) is"
BASELINE = BASE + "/yee_audit/repo/results/gsm8k/test.csv"
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CORR_RE = re.compile(r"\b(wait|actually|mistake|error|should be|correct|recheck|oops|sorry)\b", re.I)
csv.field_size_limit(10**9)
def collapse(s): return " ".join(s.split())
def val(s):
    try: return float(s.replace(",", "").rstrip("."))
    except ValueError: return None
def load_baseline():
    out = {}
    with open(BASELINE, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Model Name"] != MODEL or row.get("Prompt Style") != "sbs": continue
            fp = row["Full Prompt"].replace("\r\n", "\n"); i = fp.find(MARKER)
            if i < 0: continue
            cot = fp[i + len(MARKER):]
            if cot.endswith(SUFFIX): cot = cot[: -len(SUFFIX)]
            q = row["Question"].replace("\r\n", "\n")
            if q.startswith("Q:"): q = q[2:]
            mi2 = q.find(MARKER); prob = (q[:mi2] if mi2 > 0 else q).strip()
            out.setdefault(collapse(prob), cot)
    return out
base = load_baseline(); pairs = json.load(open(BASE + "/yee_memo/pairs.json"))
items, rejects = [], collections.Counter(); reject_log = []
for p in pairs:
    q = p["orig_question_full"]; mi = q.find(MARKER)
    problem = q[3:mi].strip() if q.startswith("Q:") else q[:mi].strip()
    pert = q[mi + len(MARKER):]
    ocot = base.get(collapse(problem))
    def rej(r, extra=""):
        rejects[r] += 1; reject_log.append({"phash": p["phash"], "reason": r, "extra": extra})
    if ocot is None: rej("no_baseline_cot"); continue
    ms = list(NUM_RE.finditer(pert))
    if not ms: rej("no_numeral_in_prefix"); continue
    last = ms[-1]
    if last.end() != len(pert.rstrip()) : rej("prefix_not_ending_on_numeral", pert[-30:]); continue
    if val(last.group()) != val(p["planted_orig"]): rej("last_numeral_ne_planted", f"{last.group()} vs {p['planted_orig']}"); continue
    k = last.start()
    if ocot[:k] != pert[:k]:
        if collapse(ocot[:k]) != collapse(pert[:k]): rej("prefix_mismatch_with_baseline"); continue
    m2 = NUM_RE.match(ocot, k)
    if not m2 or val(m2.group()) != val(p["site_true_orig"]): rej("baseline_site_mismatch", f"{m2.group() if m2 else None} vs {p['site_true_orig']}"); continue
    rest = ocot[m2.end():]
    if re.match(r"[ \t]*\n", rest) or rest == "": rej("numeral_ends_statement"); continue
    ms2 = re.match(r"([^\n]*?[.!?])(?=\s|$)", rest)
    if not ms2: rej("no_sentence_terminator_on_line"); continue
    suffix = ms2.group(1)
    if NUM_RE.search(suffix): rej("suffix_contains_number", suffix[:60]); continue
    if CORR_RE.search(suffix): rej("suffix_corrective_phrase", suffix[:60]); continue
    if len(suffix.split()) > 40: rej("suffix_too_long", str(len(suffix.split()))); continue
    true_prefix = pert[:k] + m2.group()  # clean numeral-boundary prefix (original value, same bytes)
    head = q[:mi + len(MARKER)]
    items.append({"phash": p["phash"], "numbering": "ORIG", "pos_class": p["pos_class"], "target": p["target_orig"], "propagated": p["propagated_orig"],
        "planted": p["planted_orig"], "site_true": p["site_true_orig"], "suffix": suffix, "suffix_words": len(suffix.split()),
        "punct_only": suffix.strip() in (".", "!", "?"), "prompts": {
            "corr_numeral": q, "corr_statement": q + suffix,
            "clean_numeral": head + true_prefix, "clean_statement": head + true_prefix + suffix}})
    # amendment 1: renumbered twin, same suffix (contains no number by filter); clean prefix = renum prefix with true value
    rq = p["renum_question_full"]; rmi = rq.find(MARKER); rpert = rq[rmi + len(MARKER):]; rms = list(NUM_RE.finditer(rpert))
    if rms and val(rms[-1].group()) == val(p["planted_new"]) and rms[-1].end() == len(rpert.rstrip()):
        rk = rms[-1].start(); rtrue = rpert[:rk] + str(p["site_true_new"]); rhead = rq[:rmi + len(MARKER)]
        items.append({"phash": p["phash"], "numbering": "RENUM", "pos_class": p["pos_class"], "target": p["target_new"], "propagated": p["propagated_new"],
            "planted": p["planted_new"], "site_true": p["site_true_new"], "suffix": suffix, "suffix_words": len(suffix.split()),
            "punct_only": suffix.strip() in (".", "!", "?"), "prompts": {
                "corr_numeral": rq, "corr_statement": rq + suffix,
                "clean_numeral": rhead + rtrue, "clean_statement": rhead + rtrue + suffix}})
    else: rej("renum_twin_unbuildable")
json.dump(items, open(OUT + "/e17_items.json", "w"), indent=1)
json.dump({"n_pairs": len(pairs), "n_items": len(items), "rejects": dict(rejects), "reject_log": reject_log}, open(OUT + "/build_log.json", "w"), indent=1)
print("pairs", len(pairs), "items", len(items), "rejects", dict(rejects))
print("suffix words: median", sorted(i["suffix_words"] for i in items)[len(items)//2], "max", max(i["suffix_words"] for i in items), "punct_only", sum(i["punct_only"] for i in items))
print("pos_class", collections.Counter(i["pos_class"] for i in items), "numbering", collections.Counter(i["numbering"] for i in items))
for i in items[:6]: print("  SUFFIX:", repr(i["suffix"]), "| corr_statement tail:", repr(i["prompts"]["corr_statement"][-70:]))
