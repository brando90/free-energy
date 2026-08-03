import json
SLUG = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/s7_rowcount"
NAT = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/natural_errors"
base = json.load(open(SLUG + "/out_baseline/summary.json"))
fix = json.load(open(SLUG + "/out_fixed/summary.json"))
pub = json.load(open(NAT + "/summary.json"))

def unentailed(s, seg):
    raw = s[seg]["n_rule_fabricated"]
    ent = s[seg]["fab_rule_true_entailed_implication"]
    return raw - ent, raw, ent

for seg in ["prontoqa", "expd_synthetic", "combined"]:
    pu = unentailed(pub, seg); bu = unentailed(base, seg); fu = unentailed(fix, seg)
    print(f"{seg}: unentailed pub={pu[0]} base={bu[0]} fixed={fu[0]}  (raw {bu[1]}->{fu[1]}, entailed {bu[2]}->{fu[2]})")
