"""Strict anchor-equivalence check for the E14 recon worlds (CPU, read-only).

Claim under test: recon_bare_pm10_20 / recon_full_pm10_20 are CONSTRUCTION-
IDENTICAL to E11's depth_k1_bare / depth_k1_full, so they act as anchors. If a
GPU run does not reproduce E11's k1_bare / k1_full numbers on those two cells,
the build is wrong.

Every field that reaches the model or the validator is compared:
  planted_var, planted_value, true_value, force_after_line, delta_policy,
  transform, opacity, out_cf, site_body, site_body_true
and, separately, the exact injected trace line that e11_run.build_injection_e11
would emit ("line %d: %s" % (site_line, site_body), e11_run.py:167).

Also re-asserts, per world: |delta| windows per regime, pm1 != pm10_20 plant,
the two opacity arms of a regime sharing one plant, and the full audit_depth
stack (make_audit_fn, nearest_anc_dist=4).
"""
import json
import os
import sys
from collections import Counter

IP = os.environ.get(
    "E14_IP",
    "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan")
RW = os.path.join(IP, "iclr_exec", "recompute_wall")
for _d in (os.path.join(RW, "e11_generator"), os.path.join(IP, "expg")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import audit_depth as ad                        # noqa: E402
import trace_format as tf                       # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recon_gen as rg                          # noqa: E402

WORLDS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    RW, "e14_recon", "worlds", "recon_worlds.jsonl")

FIELDS = ("planted_var", "planted_value", "true_value", "force_after_line",
          "delta_policy", "transform", "opacity", "out_cf",
          "site_body", "site_body_true")
PAIRS = {"recon_bare_pm10_20": "depth_k1_bare",
         "recon_full_pm10_20": "depth_k1_full"}

t = ad.default_targets()
t["nearest_anc_dist"] = 4
audit_fn = ad.make_audit_fn(t)

n = 0
bad = Counter()
field_mismatch = Counter()
inj_ok = 0
REGIMES = rg.REGIMES
mags = {c: Counter() for c in rg.RECON_FAMILIES}
seps = {c: Counter() for c in rg.RECON_FAMILIES}
with open(WORLDS) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        n += 1
        fams = p["families"]
        j, sv, vt = p["site_line"], p["site_var"], p["site_true_value"]

        for rc, ac in PAIRS.items():
            for f in FIELDS:
                if fams[rc][f] != fams[ac][f]:
                    field_mismatch["%s.%s" % (rc, f)] += 1
                    bad["anchor_field_mismatch"] += 1

        for c in mags:
            mags[c][abs(fams[c]["planted_value"] - vt)] += 1
        if abs(fams["recon_bare_pm1"]["planted_value"] - vt) != 1:
            bad["pm1_magnitude"] += 1
        if abs(fams["recon_bare_pm10_20"]["planted_value"] - vt) not in (10, 20):
            bad["pm10_20_magnitude"] += 1
        if not 100 <= abs(fams["recon_bare_large"]["planted_value"] - vt) <= 400:
            bad["large_magnitude"] += 1
        # v2 decoupling contract: pm10_20 preserves the unit digit, pm1 and
        # large both change it.
        for tag, want in (("pm10_20", True), ("pm1", False), ("large", False)):
            pv = fams["recon_bare_%s" % tag]["planted_value"]
            if (pv % 10 == vt % 10) != want:
                bad["last_digit_contract_%s" % tag] += 1
        if len({fams["recon_bare_%s" % t]["planted_value"]
                for t in REGIMES}) != len(REGIMES):
            bad["regimes_collide"] += 1
        # v2 separation bookkeeping: |out_cf - out_true| == 2*|delta|
        for c in mags:
            spec = fams[c]
            s_rec = spec.get("separation")
            seps[c][s_rec] += 1
            if s_rec != abs(spec["out_cf"] - p["out_true"]):
                bad["separation_mismatch_%s" % c] += 1
            if s_rec != 2 * abs(spec["planted_value"] - vt):
                bad["separation_not_2x_delta_%s" % c] += 1
        for tag in REGIMES:
            if fams["recon_bare_%s" % tag]["planted_value"] != \
                    fams["recon_full_%s" % tag]["planted_value"]:
                bad["opacity_arms_unpaired_%s" % tag] += 1

        # the exact line the runner injects must parse back to the plant
        for c in mags:
            inj = "line %d: %s" % (j, fams[c]["site_body"])
            q = tf.parse_trace_line(inj)
            if (q is None or q.get("kind") != "assign" or q["line"] != j
                    or q["var"] != sv
                    or q["value"] != fams[c]["planted_value"]):
                bad["injection_roundtrip_%s" % c] += 1
            else:
                inj_ok += 1

        # bare vs full differ ONLY by the inserted worked span
        for tag in REGIMES:
            b = fams["recon_bare_%s" % tag]["site_body"]
            f_ = fams["recon_full_%s" % tag]["site_body"]
            bh, _, bt = b.partition("; ")
            fh_, _, ft = f_.partition("; ")
            if bt != ft or not fh_.startswith(bh) or \
                    not fh_[len(bh):].startswith(" = "):
                bad["opacity_span_%s" % tag] += 1

        fails = audit_fn(p)
        if fails:
            bad["audit:%s" % fails[0]] += 1

print(json.dumps({
    "worlds_checked": n,
    "anchor_field_mismatches": dict(field_mismatch),
    "injection_roundtrips_ok": inj_ok,
    "injection_roundtrips_expected": n * len(mags),
    "separation_per_cell_min_max": {c: [min(seps[c]), max(seps[c])]
                                    for c in seps},
    "abs_delta_per_cell_min_max": {c: [min(mags[c]), max(mags[c])]
                                   for c in mags},
    "abs_delta_per_cell": {c: dict(sorted(v.items())) for c, v in mags.items()},
    "failures": dict(bad),
    "VERDICT": "ANCHOR_EQUIVALENCE_CONFIRMED" if not bad else "FAILED",
}, indent=2, sort_keys=True))
sys.exit(0 if not bad else 1)
