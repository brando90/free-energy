#!/usr/bin/env python3
"""Aggregate Q02 dynamic weighted CD report.json files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = [
    "coverage_1pct",
    "normalized_mode_entropy",
    "tv_to_uniform_modes",
    "mean_nearest_mode_distance",
    "mean_radial_error",
    "eval_data_energy",
    "eval_generated_energy",
    "eval_uniform_noise_energy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def main() -> None:
    args = parse_args()
    reports = sorted(args.input_dir.glob("seed_*/report.json"))
    if not reports:
        raise SystemExit(f"No report.json files under {args.input_dir}/seed_*")

    rows = []
    for path in reports:
        report = json.loads(path.read_text())
        seed = report["config"]["seed"]
        for method, payload in report["methods"].items():
            row = {
                "seed": seed,
                "method": method,
                "train_seconds": payload["train_seconds"],
                **payload["metrics"],
            }
            rows.append(row)

    methods = sorted({row["method"] for row in rows})
    aggregate: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "num_reports": len(reports),
        "seeds": sorted({row["seed"] for row in rows}),
        "methods": {},
        "rows": rows,
    }
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary = {"train_seconds": mean_std([r["train_seconds"] for r in method_rows])}
        for metric in METRICS:
            summary[metric] = mean_std([r[metric] for r in method_rows])
        aggregate["methods"][method] = summary

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(aggregate, indent=2))

    lines = [
        "# Aggregate Results",
        "",
        f"Input: `{args.input_dir}`",
        f"Reports: {len(reports)}",
        f"Seeds: {', '.join(str(s) for s in aggregate['seeds'])}",
        "",
        "| Method | Coverage | Entropy | TV to Uniform | Nearest Dist | Radial Error | Train s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        summary = aggregate["methods"][method]
        lines.append(
            "| {method} | {coverage:.2f}±{coverage_s:.2f} | "
            "{entropy:.3f}±{entropy_s:.3f} | {tv:.3f}±{tv_s:.3f} | "
            "{dist:.3f}±{dist_s:.3f} | {radial:.3f}±{radial_s:.3f} | "
            "{seconds:.2f}±{seconds_s:.2f} |".format(
                method=method,
                coverage=summary["coverage_1pct"]["mean"],
                coverage_s=summary["coverage_1pct"]["std"],
                entropy=summary["normalized_mode_entropy"]["mean"],
                entropy_s=summary["normalized_mode_entropy"]["std"],
                tv=summary["tv_to_uniform_modes"]["mean"],
                tv_s=summary["tv_to_uniform_modes"]["std"],
                dist=summary["mean_nearest_mode_distance"]["mean"],
                dist_s=summary["mean_nearest_mode_distance"]["std"],
                radial=summary["mean_radial_error"]["mean"],
                radial_s=summary["mean_radial_error"]["std"],
                seconds=summary["train_seconds"]["mean"],
                seconds_s=summary["train_seconds"]["std"],
            )
        )

    if "cd1" in aggregate["methods"] and "dynamic_weighted_cd" in aggregate["methods"]:
        cd = aggregate["methods"]["cd1"]
        dw = aggregate["methods"]["dynamic_weighted_cd"]
        dist_delta = cd["mean_nearest_mode_distance"]["mean"] - dw[
            "mean_nearest_mode_distance"
        ]["mean"]
        tv_delta = cd["tv_to_uniform_modes"]["mean"] - dw["tv_to_uniform_modes"]["mean"]
        lines += [
            "",
            "## Verdict",
            "",
            (
                "Dynamic weighted CD improved mean nearest-mode distance by "
                f"{dist_delta:.3f} when positive, and improved mode-balance TV by "
                f"{tv_delta:.3f} when positive. Interpret this as a heuristic "
                "negative-phase result, not an unbiased likelihood result."
            ),
        ]

    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate["methods"], indent=2))
    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
