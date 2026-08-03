"""Local no-GPU smoke: monkeypatch the backend with a deterministic fake model
(gold = interpreter-true trace; continuations = cf-world for falsehood cells,
true-world otherwise) and run prepare/generate/validate/summarize/report.
Verifies the full pipeline plumbing + expected metric signatures."""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import expg_progtrace as R
import trace_format as tf
from interp import execute, execute_cf, parse_program
from progtrace_tests import _continuation_from


class FakeBackend:
    name = "fake"

    def __init__(self, args):
        self.programs = {}
        for p in R.read_jsonl(R.out_paths(args.out_dir)["programs"]):
            self.programs[p["listing_text"]] = p

    def render(self, user_msg, prefill):
        listing = user_msg.split("Program:\n")[-1].split("\n\nTrace:")[0]
        return {"listing": listing, "prefill": prefill}

    def generate(self, rows, max_new_tokens):
        outs = []
        for r in rows:
            p = self.programs[r["listing"]]
            st = parse_program(p["stmt_texts"])
            tw = execute(st)
            if r["prefill"] == tf.GOLD_PREFILL:
                # perfect gold: full true trace, minus the prefill prefix
                txt = tf.render_true_trace(st, tw)
                assert txt.startswith(tf.GOLD_PREFILL)
                outs.append({"text": txt[len(tf.GOLD_PREFILL):],
                             "finish_reason": "stop"})
                continue
            # injected continuation: absorb falsehoods (cf world), else true
            fam = None
            last = [ln for ln in r["prefill"].split("\n") if ln.strip()][-1]
            for name, spec in p["families"].items():
                if spec["planted_value"] is not None and \
                        str(spec["planted_value"]) in last:
                    fam = spec
                    break
            j = p["site_line"]
            if fam:
                w = execute_cf(st, fam["planted_var"], fam["planted_value"],
                               fam["force_after_line"])
                from_line = j if last.startswith("note:") else j + 1
            else:
                w = tw
                from_line = j if last.startswith("note:") else j + 1
            outs.append({"text": _continuation_from(st, w, from_line),
                         "finish_reason": "stop"})
        return outs


def main():
    out = os.path.join(HERE, "results", "EXPG_PROGTRACE_SMOKE")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    R.make_backend = lambda args: FakeBackend(args)
    base = ["--out-dir", out, "--oversample", "8", "--cap", "5"]
    R.main(["prepare"] + base + ["--overwrite"])
    R.main(["generate"] + base)
    R.main(["validate"] + base)
    R.main(["summarize"] + base)
    R.main(["report"] + base)
    s = json.load(open(R.out_paths(out)["summary"]))
    ok = True

    def expect(name, cond):
        global _
        print("[%s] %s" % ("PASS" if cond else "FAIL", name))
        return cond

    c = s["cells"]
    ok &= expect("all_cells_present", set(c) == set(R.PILOT_CELLS))
    ok &= expect("n_capped", all(c[x]["n"] <= 5 for x in c))
    ok &= expect("fake_absorbs_kr1", c["opfree_kr1"]["next_read_absorbed"]["rate"] == 1.0)
    ok &= expect("fake_absorbs_out", c["deep_kc5"]["final_output_absorbed"]["rate"] == 1.0)
    ok &= expect("benign_valid", c["benign_paraphrase"]["trace_valid"]["rate"] == 1.0)
    ok &= expect("benign_nm_absorption", c["benign_paraphrase"]["next_read_absorbed"]["rate"] is None)
    ok &= expect("gateB_present", s["gate_B"]["full_grid_viable"] in (True, False))
    ok &= expect("gateA_pass", s["gate_A_pass"] is True)
    ok &= expect("report_written", os.path.exists(R.out_paths(out)["report"]))
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
