#!/usr/bin/env python3
"""E1 replication — per-model world preparation (CPU only, no GPU).

Reuses the LOCKED EXPD world generator (gen_worlds_expd.py) UNCHANGED. Oversampling
is obtained by generating the full confirmatory config under K distinct seeds and
merging the shards under an ``s{seed}_`` id prefix (+ a family_idx offset) so that
world_id / family_id / partner_world_id stay globally unique. Each shard is
fail-closed audited independently BEFORE the merge; any shard audit failure aborts.

Outputs (exactly the files the locked ``generate`` stage reads):
  <out>/worlds/worlds.json
  <out>/worlds/world_manifest.jsonl
  <out>/worlds/world_audit.json      (merged doc + per-shard provenance)
  <out>/candidate_pool.jsonl         (merged world dicts -- what generate() reads)
  <out>/e1_provenance.json           (E1 registration commit, model, seeds, cells)

Nothing under results/ or existing improvement_plan/ subdirs is modified.
Usage:
  python prepare_e1.py --out <model_results_dir> --seeds 0,1,2,3 \
      --model <repo> --revision <sha> --reg-commit <hash>
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

# Locate the LOCKED expd generator and import it unchanged (it also wires src/ in).
EXPD_DIR = os.environ.get(
    "EXPD_DIR",
    "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/expd",
)
if EXPD_DIR not in sys.path:
    sys.path.insert(0, EXPD_DIR)
import gen_worlds_expd as genw  # noqa: E402

FAMILY_IDX_STRIDE = 100000  # per-shard family_idx offset (>> max families/shard)


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _prefix_world(w, tag, idx_offset):
    """Return a copy of world dict w remapped into shard `tag`.

    Only the three id strings and family_idx are rewritten; every other field
    (question/target/entity/cells/canonical_proof_steps/anchors_supported/t0/...)
    is preserved byte-for-byte, so the locked plan_injections() sees an identical
    world modulo its unique id."""
    w = dict(w)
    w["world_id"] = tag + w["world_id"]
    w["family_id"] = tag + w["family_id"]
    if w.get("partner_world_id") is not None:
        w["partner_world_id"] = tag + w["partner_world_id"]
    w["family_idx"] = w["family_idx"] + idx_offset
    w["e1_shard_seed"] = None  # filled by caller
    return w


def build(out_dir, seeds, config_name="full"):
    worlds_dir = os.path.join(out_dir, "worlds")
    os.makedirs(worlds_dir, exist_ok=True)

    merged_worlds = []
    merged_audits = {}
    merged_fam_recs = []
    per_shard = []
    total_failures = []

    for j, seed in enumerate(seeds):
        config = genw.CONFIGS[config_name]
        families, worlds = genw.generate_worlds(seed, config)
        audits, fam_recs, failures = genw.run_audits(families, worlds)
        # Fail-closed per shard: an audit failure in ANY shard aborts the model.
        if failures:
            total_failures.extend(["seed=%d: %s" % (seed, f) for f in failures])
            continue
        tag = "s%d_" % seed
        idx_offset = j * FAMILY_IDX_STRIDE
        for w in worlds:
            old_id = w["world_id"]
            pw = _prefix_world(w, tag, idx_offset)
            pw["e1_shard_seed"] = seed
            merged_worlds.append(pw)
            a = audits.get(old_id)
            if a is not None:
                merged_audits[pw["world_id"]] = a
        for rec in fam_recs:
            rec = dict(rec)
            rec["family_id"] = tag + rec["family_id"]
            merged_fam_recs.append(rec)
        per_shard.append({"seed": seed, "world_count": len(worlds),
                          "family_count": len(families), "idx_offset": idx_offset,
                          "tag": tag})

    if total_failures:
        # Write the audit doc so the failures are inspectable, then abort.
        with open(os.path.join(worlds_dir, "world_audit.json"), "w") as fh:
            json.dump({"experiment": "EXPD-R (E1) world generation",
                       "created_at": now_iso(), "seeds": seeds,
                       "audit_failures": total_failures}, fh, indent=2, sort_keys=True)
        raise SystemExit("FAIL-CLOSED: %d shard audit failures (see world_audit.json):\n%s"
                         % (len(total_failures), "\n".join(total_failures[:20])))

    # ---- worlds.json (test_example dict, same shape as write_outputs) ----------
    ds = {}
    for w in merged_worlds:
        ds[w["world_id"]] = {"test_example": {
            "question": w["question"], "query": w["query"],
            "chain_of_thought": w["canonical_proof_steps"],
        }}
    with open(os.path.join(worlds_dir, "worlds.json"), "w") as fh:
        json.dump(ds, fh, indent=1, sort_keys=True)
    worlds_json_sha = hashlib.sha256(
        open(os.path.join(worlds_dir, "worlds.json"), "rb").read()).hexdigest()

    # ---- world_manifest.jsonl (row = world + its audit + created_at) -----------
    with open(os.path.join(worlds_dir, "world_manifest.jsonl"), "w") as fh:
        for w in merged_worlds:
            row = dict(w)
            row["audit"] = merged_audits.get(w["world_id"])
            row["created_at"] = now_iso()
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    # ---- world_audit.json (merged, with per-shard provenance) ------------------
    audit_doc = {
        "experiment": "EXPD-R (E1) world generation (multi-seed merge of the UNCHANGED "
                      "EXPD full config)",
        "created_at": now_iso(),
        "config": config_name,
        "seeds": seeds,
        "per_shard": per_shard,
        "merge_scheme": {
            "id_prefix": "s{seed}_ on world_id/family_id/partner_world_id",
            "family_idx_offset": "shard_index * %d" % FAMILY_IDX_STRIDE,
            "generator": "gen_worlds_expd.generate_worlds (reused unchanged)",
            "fail_closed": "any shard audit failure aborts before merge",
        },
        "world_count": len(merged_worlds),
        "family_count": len(merged_fam_recs) or None,
        "per_world_audits_passed": len(merged_audits),
        "family_audit_records": merged_fam_recs,
        "audit_failures": [],
        "worlds_json_sha256": worlds_json_sha,
    }
    with open(os.path.join(worlds_dir, "world_audit.json"), "w") as fh:
        json.dump(audit_doc, fh, indent=2, sort_keys=True)

    # ---- candidate_pool.jsonl (what the locked generate() reads) ---------------
    with open(os.path.join(out_dir, "candidate_pool.jsonl"), "w") as fh:
        for w in merged_worlds:
            fh.write(json.dumps(w, sort_keys=True) + "\n")

    return {"world_count": len(merged_worlds),
            "per_world_audits_passed": len(merged_audits),
            "worlds_json_sha256": worlds_json_sha,
            "seeds": seeds}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="model results dir")
    ap.add_argument("--seeds", default="0", help="comma-separated world seeds (K = oversample)")
    ap.add_argument("--config", default="full", choices=sorted(genw.CONFIGS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--reg-commit", default=None)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    os.makedirs(args.out, exist_ok=True)
    summary = build(args.out, seeds, args.config)
    prov = {
        "stage": "prepare_e1",
        "e1_registration_commit": args.reg_commit,
        "model": args.model, "revision": args.revision,
        "oversample_seeds": seeds, "oversample_factor": len(seeds),
        "world_config": args.config, "created_at": now_iso(),
        "generator": "gen_worlds_expd.py (LOCKED, reused unchanged)",
        **summary,
    }
    with open(os.path.join(args.out, "e1_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
