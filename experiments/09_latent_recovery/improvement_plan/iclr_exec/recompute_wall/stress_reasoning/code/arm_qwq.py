"""ARM C -- QwQ-32B (reasoning-RL trained on Qwen2.5-32B, the roster's own base)
on the recompute-wall cells, with a VISIBLE think channel.

Purpose: matched-base pre/post-reasoning-training pair. The roster row
EXPG_PROGTRACE_qwen32b (Qwen2.5-32B-Instruct) absorbs the 1-op computed cell at
1.00; this arm measures QwQ on the SAME worlds with the SAME two-world grader.

Protocol mirrors reasoning_arm/arm_b.py (R1-Distill) so the two think-models are
directly comparable; arm_b.py is imported for every shared helper so there is no
second implementation of gold/injection/grading/summary logic.

  C1  raw-completion continuation, no think tags (direct regime).
  C2  chat template WITH think channel; the FINAL answer (post-</think>) is what
      classify_run grades; the think text is scanned for re-derivation.

A prefill-mode run already exists (stress_reasoning/EXPG_PROGTRACE_qwq32b); its
in-think channel absorbed 0.98 while its post-</think> answer recovered ~0.73 --
that run is superseded for reporting by this arm, which grades the model's actual
answer by construction.

Usage: arm_qwq.py [--reps 4] [--c2-max-tokens 4000] [--cap 60]
"""
import argparse, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
IP = EXP + "/improvement_plan"
RW = IP + "/iclr_exec/recompute_wall"
ARMB_DIR = RW + "/reasoning_arm"
for p in (IP + "/expg", ARMB_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import arm_b as B                # noqa: E402  (shared harness)
import trace_format as tf        # noqa: E402
import inject as inj_mod         # noqa: E402

QWQ_PATH = ("/lfs/skampere2/0/eobbad/.cache/huggingface/hub/models--Qwen--QwQ-32B/"
            "snapshots")
OUT_DIR = RW + "/stress_reasoning/arm_qwq"
PROGRAMS = RW + "/e10_widen/results_skampere1/EXPG_PROGTRACE_qwen32b/programs.jsonl"


def resolve_snapshot(base):
    subs = [os.path.join(base, d) for d in os.listdir(base)]
    subs = [s for s in subs if os.path.isdir(s)]
    return sorted(subs, key=os.path.getmtime)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--c2-max-tokens", type=int, default=4000)
    ap.add_argument("--gold-oversample", type=int, default=4)
    ap.add_argument("--cap", type=int, default=60, help="max worlds per cell")
    args = ap.parse_args()

    # Point the shared harness at QwQ + our worlds/output, leaving arm_b.py untouched.
    B.MODEL_PATH = resolve_snapshot(QWQ_PATH)
    B.PROGRAMS = PROGRAMS
    B.OUT_DIR = OUT_DIR
    B.DS_EOS = "<|im_end|>"      # Qwen chat EOS (R1's DeepSeek token is wrong here)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "README_ARM_QWQ.md"), "w") as f:
        f.write("ARM C (QwQ-32B think channel), author-directed 2026-07-31.\n"
                "Harness: reasoning_arm/arm_b.py imported verbatim; only MODEL_PATH,\n"
                "PROGRAMS (qwen32b roster worlds), OUT_DIR, DS_EOS overridden.\n"
                "C1 = raw-completion continuation (no think). C2 = chat + think channel,\n"
                "graded post-</think>. Matched-base comparison: EXPG_PROGTRACE_qwen32b.\n"
                "Model snapshot: %s\n" % B.MODEL_PATH)

    progs, by_shape = B.programs_by_shape()
    if args.cap:
        by_shape = {s: v[:args.cap] for s, v in by_shape.items()}
    llm = B.make_llm()

    coh_path = os.path.join(OUT_DIR, "gold_cohort.json")
    if os.path.exists(coh_path):
        cj = json.load(open(coh_path))
        cohort = {r["program_id"]: r["gold_full_text"]
                  for r in B.read_jsonl(os.path.join(OUT_DIR, "gold_raw.jsonl")) if r["solved"]}
        print("[gold] resumed cohort n=%d" % len(cohort))
    else:
        cohort = B.stage_gold(llm, by_shape, args.gold_oversample)
        cj = json.load(open(coh_path))

    manifest = B.build_manifest(progs, by_shape, cohort)
    print("[manifest] cells=%s" % json.dumps(dict(Counter(m["cell"] for m in manifest))))
    c1 = B.stage_b1(llm, progs, manifest, args.reps)
    c2 = B.stage_b2(llm, progs, manifest, args.reps, args.c2_max_tokens)
    B.summarize(cj, c1, c2)
    os.replace(os.path.join(OUT_DIR, "summary_arm_b.json"),
               os.path.join(OUT_DIR, "summary_arm_qwq.json"))
    print("DONE -> %s/summary_arm_qwq.json" % OUT_DIR)


if __name__ == "__main__":
    main()
