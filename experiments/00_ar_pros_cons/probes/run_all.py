"""Run probes 01, 03, 05 in sequence and write a combined summary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # experiments/00_ar_pros_cons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--probes",
        nargs="+",
        default=[
            "probe_01",
            "probe_02",
            "probe_03",
            "probe_04",
            "probe_05",
            "probe_06",
            "probe_07",
            "probe_08",
            "probe_data_wall",
        ],
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else REPO_ROOT / "mnt" / "user-data" / "outputs"
    summary_dir = out_root / "summary" / args.tag
    summary_dir.mkdir(parents=True, exist_ok=True)

    script_for = {
        "probe_01": "probe_01_softmax_bottleneck.py",
        "probe_02": "probe_02_mode_covering.py",
        "probe_03": "probe_03_rank_collapse.py",
        "probe_04": "probe_04_partition_removable.py",
        "probe_05": "probe_05_fixed_compute.py",
        "probe_06": "probe_06_error_compounding.py",
        "probe_07": "probe_07_reversal_curse.py",
        "probe_08": "probe_08_lipschitz_margin.py",
        "probe_data_wall": "probe_data_wall.py",
    }

    summary = {"tag": args.tag, "device": args.device, "seed": args.seed, "started_at": time.time(), "probes": {}}
    overall_ok = True
    for name in args.probes:
        if name not in script_for:
            print(f"[run_all] unknown probe: {name}", file=sys.stderr)
            overall_ok = False
            continue
        cmd = [
            sys.executable,
            "-m",
            f"probes.{Path(script_for[name]).stem}",
            "--tag",
            args.tag,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if args.smoke:
            cmd.append("--smoke")
        print(f"[run_all] running: {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        probe_out = out_root / name / args.tag / "result.json"
        entry = {"return_code": rc, "result_path": str(probe_out), "control_passed": None}
        if probe_out.exists():
            with probe_out.open() as f:
                payload = json.load(f)
            entry["control_passed"] = payload.get("control_passed")
            entry["verdict"] = payload.get("verdict")
            entry["duration_s"] = payload.get("duration_s")
        summary["probes"][name] = entry
        if rc != 0 or not entry.get("control_passed"):
            overall_ok = False

    summary["finished_at"] = time.time()
    summary["duration_s"] = summary["finished_at"] - summary["started_at"]
    summary["overall_control_passed"] = overall_ok
    summary_path = summary_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    print(f"[run_all] wrote {summary_path}  overall_ok={overall_ok}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
