#!/usr/bin/env python3
"""E17 API runner: clone of yee_memo/run_api.py (same build_messages, temperature 0, max_tokens 500, single call),
over e17_items.json x 4 prompt arms. gpt-4o only. Separate ledger, HARD STOP $5. Resumable."""
import argparse, json, os, sys, time, threading
OUT = "/lfs/skampere2/0/eobbad/scratch/e17_boundary"; ITEMS = OUT + "/e17_items.json"; RESP = OUT + "/responses.jsonl"; LEDGER = OUT + "/usage_ledger.json"
PRICE = {"gpt-4o": (2.5, 10.0)}; HARD_STOP = 5.0; TEMP = 0; MAX_TOKENS = 500; ARMS = ["corr_numeral", "corr_statement", "clean_numeral", "clean_statement"]
def build_messages(question):
    if "A:" in question:
        jq, reasoning = question.split("A:", 1); reasoning = "A: " + reasoning.replace("A:", "").strip()
        return [{"role": "user", "content": jq.strip()}, {"role": "assistant", "content": reasoning}]
    return [{"role": "user", "content": question}]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gpt-4o"); ap.add_argument("--threads", type=int, default=6); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    from openai import OpenAI; client = OpenAI()
    items = json.load(open(ITEMS)); led = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"models": {}, "cost_usd": 0.0}
    done = set()
    if os.path.exists(RESP):
        for line in open(RESP):
            try: d = json.loads(line); done.add((d["phash"], d["numbering"], d["arm"], d["model"]))
            except Exception: pass
    tasks = [(it["phash"], it["numbering"], arm, a.model, it["prompts"][arm]) for it in items for arm in ARMS if (it["phash"], it["numbering"], arm, a.model) not in done]
    if a.limit: tasks = tasks[:a.limit]
    print("tasks:", len(tasks), "done:", len(done), "ledger $%.3f" % led["cost_usd"]); it = iter(tasks); lock = threading.Lock(); out = open(RESP, "a"); stop = {"s": False}
    def worker():
        while True:
            with lock:
                if stop["s"] or led["cost_usd"] >= HARD_STOP: stop["s"] = True; return
                try: phash, numbering, arm, model, question = next(it)
                except StopIteration: return
            resp = None
            for attempt in range(5):
                try: resp = client.chat.completions.create(model=model, messages=build_messages(question), temperature=TEMP, max_tokens=MAX_TOKENS); break
                except Exception as e: sys.stderr.write(f"retry {phash} {arm}: {type(e).__name__}\n"); time.sleep(2 ** attempt * 2)
            rec = {"phash": phash, "numbering": numbering, "arm": arm, "model": model, "ts": time.time()}
            if resp is None: rec["error"] = "failed_5x"
            else:
                ch = resp.choices[0]; rec.update({"text": ch.message.content, "finish_reason": ch.finish_reason, "usage": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}, "resp_model": resp.model})
            with lock:
                out.write(json.dumps(rec) + "\n"); out.flush()
                if resp is not None:
                    m = led["models"].setdefault(model, {"in": 0, "out": 0, "calls": 0}); m["in"] += resp.usage.prompt_tokens; m["out"] += resp.usage.completion_tokens; m["calls"] += 1
                    pi, po = PRICE[model]; led["cost_usd"] += (resp.usage.prompt_tokens * pi + resp.usage.completion_tokens * po) / 1e6
                    json.dump(led, open(LEDGER, "w"), indent=1)
    th = [threading.Thread(target=worker) for _ in range(a.threads)]; [t.start() for t in th]; [t.join() for t in th]
    print("finished; ledger $%.3f; stopped_by_cap=%s" % (led["cost_usd"], stop["s"]))
if __name__ == "__main__": main()
