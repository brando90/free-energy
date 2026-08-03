import json, collections
p = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/stage0_strict/strict_rows.jsonl"
rows = [json.loads(l) for l in open(p)]
def cell(fam):
    rs = [r for r in rows if r["family"] == fam]
    n = len(rs)
    hs = sum(1 for r in rs if r.get("hop_sound_valid"))
    cv = sum(1 for r in rs if r.get("closure_valid"))
    srr = sum(1 for r in rs if r.get("srr_task"))
    tf = collections.Counter(r.get("truth") for r in rs)
    print(f"{fam}: n={n} hop_sound={hs}/{n}={hs/n:.3f} closure={cv}/{n}={cv/n:.3f} srr_task={srr}/{n}={srr/n:.3f} truth={dict(tf)}")
cell("EXPA:one_hop_falsehood")
cell("EXPA:global_falsehood")
# audited-false total again + parseable
N = len(rows)
false_ct = sum(1 for r in rows if r.get("truth") == "false")
parseable = sum(1 for r in rows if r.get("class") != "unparsed")
valid = sum(1 for r in rows if r.get("class") == "valid_rederivation")
print(f"TOTAL={N} audited_false={false_ct} parseable={parseable} valid_rederivation={valid}")
