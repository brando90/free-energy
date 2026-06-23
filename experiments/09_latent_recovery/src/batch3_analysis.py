"""Validate batch-3 families and print the three decision tables."""
import json
from validator import main as vmain

for res, fam in [("results_verify", "falsehood"), ("results_verify", "negstep"),
                 ("results", "neghop4"), ("results", "neghop5"),
                 ("results_32b", "neghop2"), ("results_32b", "neghop3")]:
    try:
        vmain(res, f"perturbed_{fam}", f"_{fam}" if res != "results_verify" else f"_{fam}_verify")
    except Exception as e:
        print(f"VALIDATION FAILED {res}/{fam}: {e}")

def show(path, label, points=("early", "mid", "late")):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(f"{label}: MISSING")
        return
    for p in points:
        c = d.get(p, {})
        n = c.get("n", 0)
        if n == 0:
            continue
        print("%-28s %-6s n=%3d valid=%.3f poisoned=%.3f ack=%.3f" % (
            label, p, n, c.get("valid_rederivation", 0) / n,
            c.get("poisoned", 0) / n, c.get("acknowledged", 0) / n))

print()
print("### INTERVENTION (verification prompt) vs BASELINE ###")
show("../results/validated_summary_falsehood.json", "falsehood BASELINE")
show("../results_verify/validated_summary_falsehood_verify.json", "falsehood +VERIFY-PROMPT")
show("../results/validated_summary_negstep.json", "negstep BASELINE")
show("../results_verify/validated_summary_negstep_verify.json", "negstep +VERIFY-PROMPT")
print()
print("### DOSE-RESPONSE 7B EXTENDED (early) ###")
for fam, k in [("negstep", 1), ("neghop2", 2), ("neghop3", 3), ("neghop4", 4), ("neghop5", 5)]:
    show(f"../results/validated_summary_{fam}.json", f"7B k={k}", points=("early",))
print()
print("### DOSE-RESPONSE 32B (early) ###")
for fam, k in [("negstep", 1), ("neghop2", 2), ("neghop3", 3)]:
    show(f"../results_32b/validated_summary_{fam}.json", f"32B k={k}", points=("early",))
