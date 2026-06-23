"""Step 3: null ensemble — natural spread of valid trajectories.

For each instance/injection-point: prefix = the model's own steps up to and INCLUDING
the CORRECT step (no perturbation), then n=8 temperature-0.8 continuations.
Hidden states cached for every sample; correctness recorded so metrics can use
correct-only nulls per PLAN.md.
"""
import os, json, time
import torch
from common import (load_model, make_prompt_ids, sample_n, solved, split_sentences,
                    hidden_states_from, save_npz, DATA, RESULTS)
from perturb import injection_points

OUT = os.path.join(RESULTS, os.environ.get("LR_NULL_OUT", "null"))
N = int(os.environ.get("LR_NULL_N", "8"))
TEMP = float(os.environ.get("LR_NULL_T", "0.8"))
MAX_INST = int(os.environ.get("LR_NULL_MAX", "10000"))

def main():
    os.makedirs(os.path.join(OUT, "hs"), exist_ok=True)
    tok, model = load_model()
    runs = [json.loads(l) for l in open(os.path.join(RESULTS, "perturbed", "runs.jsonl"))]
    done_pairs = {(r["id"], r["point"]) for r in runs if "skip" not in r}
    rows = [json.loads(l) for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))]
    rows = [r for r in rows if r["solved"]][:MAX_INST]
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    t0 = time.time()
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for i, r in enumerate(rows):
            inst = data[r["id"]]
            steps = split_sentences(r["gen_text"])
            pts = injection_points(steps)
            if pts is None:
                continue
            safe = r["id"].replace("/", "_").replace("::", "__")
            for name, si in pts.items():
                if (r["id"], name) not in done_pairs:
                    continue
                prefix = " " + " ".join(steps[:si + 1])   # correct step included
                ids = make_prompt_ids(tok, inst["question"], inst["target"], answer_prefix=prefix)
                texts, outs = sample_n(tok, model, ids, n=N, temp=TEMP, max_new=192)
                corrects = []
                for j, (txt, full) in enumerate(zip(texts, outs)):
                    ok = solved((prefix + " " + txt).strip(), inst["target"])
                    corrects.append(ok)
                    hs = hidden_states_from(tok, model, full, ids.shape[1])
                    save_npz(os.path.join(OUT, "hs", f"{safe}__{name}__s{j}.npz"), hs,
                             id=r["id"], point=name, sample=j, correct=ok,
                             boundary=int(ids.shape[1]), text=txt)
                fh.write(json.dumps({"id": r["id"], "point": name,
                                     "n_correct": sum(corrects), "n": N,
                                     "boundary": int(ids.shape[1])}) + "\n")
                fh.flush()
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(rows)}] ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"DONE null ({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
