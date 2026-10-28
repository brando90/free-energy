# appended block: Fable refusal characterization (run separately)
import json, os, collections
TOP="/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e10_topup_fable"
gp=os.path.join(TOP,"raw_gold_claude-fable-5.jsonl")
rows=[json.loads(l) for l in open(gp)]
ref=sum(1 for r in rows if r.get("error")=="stop_reason_refusal")
failed=sum(1 for r in rows if r.get("failed_generation"))
ok=sum(1 for r in rows if not r.get("failed_generation"))
led=json.load(open(os.path.join(TOP,"cost_ledger.json")))
print("FABLE-5 gold attempts:", len(rows))
print("  failed_generation:", failed, " refusals(stop_reason_refusal):", ref, " succeeded:", ok)
print("  fable-5 spend (ledger):", led.get("usd_by_model",{}).get("claude-fable-5"))
print("  fable-5 tokens:", led.get("models",{}).get("claude-fable-5"))
