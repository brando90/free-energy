"""Step 1: greedy gold rollouts. Keep solved instances; cache hidden states.

Output:
  results/gold/rollouts.jsonl   one row per instance (text, solved, token boundaries)
  results/gold/hs/{id}.npz      hidden states (LAYERS x [answer_len, d]) for solved
"""
import os, json, time
import torch
from common import (load_model, make_prompt_ids, greedy, solved, split_sentences,
                    hidden_states_from, save_npz, DATA, RESULTS)

MAX_SOLVED = 200          # storage/time cap for the pilot
OUT_DIR = os.path.join(RESULTS, "gold")

def main():
    os.makedirs(os.path.join(OUT_DIR, "hs"), exist_ok=True)
    tok, model = load_model()
    rows = [json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))]
    n_solved = 0
    t0 = time.time()
    with open(os.path.join(OUT_DIR, "rollouts.jsonl"), "w") as fh:
        for i, r in enumerate(rows):
            if n_solved >= MAX_SOLVED:
                break
            ids = make_prompt_ids(tok, r["question"], r["target"])
            gen_text, full = greedy(tok, model, ids)
            ok = solved(gen_text, r["target"])
            rec = {"id": r["id"], "solved": ok, "gen_text": gen_text,
                   "prompt_len": int(ids.shape[1]), "total_len": int(full.shape[0]),
                   "target": r["target"], "n_hops": r["n_hops"]}
            if ok:
                n_solved += 1
                hs = hidden_states_from(tok, model, full, ids.shape[1])
                safe = r["id"].replace("/", "_").replace("::", "__")
                save_npz(os.path.join(OUT_DIR, "hs", safe + ".npz"), hs,
                         id=r["id"], prompt_len=int(ids.shape[1]))
                rec["hs_file"] = safe + ".npz"
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(rows)}] solved={n_solved} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"DONE gold: attempted={i+1} solved={n_solved} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
