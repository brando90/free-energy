import json
r = json.load(open("/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/review_audit_20260728/arm_overlap_audit.json"))
for tag in ["haiku", "llama8b", "qwen7b"]:
    for k in ["0", "1", "5"]:
        d = r["intersection_reanalysis"][tag][k]
        print("== %s k=%s  n_inter_programs=%d" % (tag, k, d["n_intersection_programs"]))
        for arm in ["A0", "A1", "A2", "A3"]:
            pa = d["per_arm"][arm]
            f = pa["full"]; i = pa["intersection"]; r0 = pa["intersection_rollout0"]
            print("  %s full n=%d abs=%s %s labels=%s | inter n=%d abs=%s %s | r0 n=%d abs=%s" % (
                arm, f["n"], f["rate"], f["wilson95"], f["labels"], i["n"], i["rate"], i["wilson95"], r0["n"], r0["rate"]))
        print("  delta_vs_A0:", json.dumps(d["delta_vs_A0"]))
