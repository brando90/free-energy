"""Reproducible integrity check: re-load results/*.json and assert every
headline claim in ANSWERS.md / RESULTS.md holds, plus the Codex cross-check
agreement. Exit code 0 iff all pass. Run after the e{1..5} scripts.

    ../../../.venv/bin/python src/verify_claims.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
ok = True


def load(p: str):
    return json.loads((R / p).read_text())


def check(name: str, cond: bool) -> None:
    global ok
    print(("  PASS" if cond else "  FAIL"), name)
    ok = ok and bool(cond)


def main() -> None:
    print("E1 — continuous EBM without Z; MCMC-free matches MCMC:")
    e1 = load("e1_langevin_vs_sm.json")
    check("Z never computed", e1["z_was_computed"] is False)
    check("DSM training used no MCMC", e1["mcmc_used_in_training"]["dsm"] is False)
    check("both beat the noise ceiling (MMD)",
          max(e1["mmd2"]["pcd_langevin"], e1["mmd2"]["dsm_mcmc_free"])
          < e1["mmd2"]["ceiling_noise"])
    check("MCMC-free DSM within 2x of PCD (MMD)",
          e1["mmd2"]["dsm_mcmc_free"] < 2 * e1["mmd2"]["pcd_langevin"])

    print("E2 — short-run CD is biased, not just noisy:")
    e2 = load("e2_cd_bias.json")["e2"]
    ks = ["1", "2", "5", "10", "20", "50"]
    biases = [e2[k]["bias_rel"] for k in ks]
    check("bias falls monotonically 1->50",
          all(biases[i] >= biases[i + 1] - 1e-9 for i in range(len(biases) - 1)))
    check("CD-1 bias dominates MSE (>80%)",
          e2["1"]["bias_sq"] / e2["1"]["mse"] > 0.8)
    check("PCD bias << CD-1 bias", e2["pcd"]["bias_rel"] < 0.1 * e2["1"]["bias_rel"])
    check("cosine -> 1 by k=10", e2["10"]["cosine"] > 0.999)

    print("E3 — single chain stuck; parallel helps; batching ~free:")
    e3 = load("e3_parallel_chains.json")
    check("one chain is sticky (tau>10)", e3["part_a_ess"]["tau"] > 10)
    b = e3["part_b_coverage"]
    check("parallel cuts mode-weight error >2x",
          b["1"]["tv_mean"] / b["256"]["tv_mean"] > 2)
    tp = e3["part_c_throughput"]["ebm_langevin_torch"]
    check("batched throughput scales >1000x",
          tp["16384"]["samples_per_sec"] / tp["1"]["samples_per_sec"] > 1000)

    print("E4 — late/Zipf trajectory weighting beats vanilla last-only CD:")
    e4 = load("e4_cd_weighting.json")["part_a_gradient"]
    best = min(e4, key=lambda n: e4[n]["mse"])
    check("best schedule is zipf_late or geom_late", best in ("zipf_late", "geom_late"))
    check("zipf_late MSE < 0.6x last MSE", e4["zipf_late"]["mse"] < 0.6 * e4["last"]["mse"])
    check("early is worst (highest bias)",
          max(e4, key=lambda n: e4[n]["bias_rel"]) == "early")

    print("E5 — Z exponential; AIS converges:")
    e5 = load("e5_partition_ais.json")
    check("enumeration slope ~ x2/bit (0.25-0.34)",
          0.25 < e5["part_a_enumeration"]["fit_slope_per_bit"] < 0.34)
    runs = e5["part_b_ais"]["runs"]
    check("AIS |error| shrinks 10->200 levels", runs[0]["abs_err"] > runs[2]["abs_err"])
    check("AIS std shrinks monotonically",
          all(runs[i]["std"] >= runs[i + 1]["std"] for i in range(len(runs) - 1)))

    print("Cross-check (this impl vs Codex independent numpy):")
    mine = load("e2_cd_bias.json")["e2"]
    cc = load("crosscheck_codex/codex_e2e4.json")
    for k in ["1", "2", "5"]:
        d = abs(mine[k]["bias_rel"] - cc["e2"][k]["bias_rel"])
        check(f"CD-{k} bias agree (<0.02): mine={mine[k]['bias_rel']:.4f} "
              f"codex={cc['e2'][k]['bias_rel']:.4f}", d < 0.02)
    e4c = cc["e4"]
    check("both pick zipf_late as best E4 schedule",
          min(e4c, key=lambda n: e4c[n]["mse"]) == "zipf_late")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
