#!/usr/bin/env python3
"""Read all sweep JSONs in results/ and produce a markdown summary table.

Usage:  python summarize_results.py [--results-dir DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _normalize_row(r: dict) -> dict | None:
    """Normalize a result row coming from either profiler format.

    Mine (profile_pytorch / profile_jax):
        {loss_name, in_dim, hidden, batch, backend, fwd_bwd_ms, peak_mem_mb, ...}
    Other instance (profile_sm_*):
        {estimator, D, hidden, batch, backend, mean_ms, peak_mem_mb, ...}
    """
    if "loss_name" in r and "in_dim" in r:
        return r
    if "estimator" in r and "D" in r:
        # rename other-instance keys to mine
        return {
            "loss_name": r["estimator"],
            "in_dim": r["D"],
            "hidden": r.get("hidden"),
            "batch": r.get("batch"),
            "backend": r.get("backend"),
            "fwd_bwd_ms": r.get("mean_ms", r.get("fwd_bwd_ms")),
            "peak_mem_mb": r.get("peak_mem_mb", float("nan")),
        }
    return None


def load_all(results_dir: Path):
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            raw_rows = payload.get("results", [])
        elif isinstance(payload, list):
            raw_rows = payload
        else:
            raw_rows = []
        for r in raw_rows:
            nr = _normalize_row(r)
            if nr is None:
                continue
            rows.append({**nr, "source": f.name})
    return rows


def make_speed_table(rows, fixed_hidden, fixed_batch):
    cols = ["mle_like", "dsm", "sm_hutch1", "sm_hutch4", "sm_exact"]
    by_backend = defaultdict(lambda: defaultdict(dict))
    dims_seen = set()
    for r in rows:
        if r["hidden"] != fixed_hidden or r["batch"] != fixed_batch:
            continue
        bk = r["backend"]
        by_backend[bk][r["in_dim"]][r["loss_name"]] = r["fwd_bwd_ms"]
        dims_seen.add(r["in_dim"])

    lines = []
    for bk in sorted(by_backend):
        lines.append(f"### {bk}  (hidden={fixed_hidden}, batch={fixed_batch}, ms / step)")
        lines.append("")
        header = "| d | " + " | ".join(cols) + " |"
        sep = "| ---: | " + " | ".join(["---:"] * len(cols)) + " |"
        lines.append(header)
        lines.append(sep)
        for d in sorted(dims_seen):
            row = by_backend[bk].get(d, {})
            cells = []
            for c in cols:
                v = row.get(c)
                if v is None:
                    cells.append("—")
                elif v != v:  # NaN
                    cells.append("nan")
                else:
                    cells.append(f"{v:.2f}")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def make_ratio_table(rows, fixed_hidden, fixed_batch):
    cols = ["dsm", "sm_hutch1", "sm_hutch4", "sm_exact"]
    by_backend = defaultdict(lambda: defaultdict(dict))
    dims_seen = set()
    for r in rows:
        if r["hidden"] != fixed_hidden or r["batch"] != fixed_batch:
            continue
        by_backend[r["backend"]][r["in_dim"]][r["loss_name"]] = r["fwd_bwd_ms"]
        dims_seen.add(r["in_dim"])

    lines = []
    for bk in sorted(by_backend):
        lines.append(f"### {bk}  (hidden={fixed_hidden}, batch={fixed_batch}, ratio vs DSM)")
        lines.append("")
        header = "| d | " + " | ".join(cols) + " |"
        sep = "| ---: | " + " | ".join(["---:"] * len(cols)) + " |"
        lines.append(header)
        lines.append(sep)
        for d in sorted(dims_seen):
            row = by_backend[bk].get(d, {})
            base = row.get("dsm")
            cells = []
            for c in cols:
                v = row.get(c)
                if v is None or base is None or base != base or v != v or base == 0:
                    cells.append("—")
                else:
                    cells.append(f"{v / base:.2f}×")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def make_memory_table(rows, fixed_hidden, fixed_batch):
    cols = ["mle_like", "dsm", "sm_hutch1", "sm_hutch4", "sm_exact"]
    by_backend = defaultdict(lambda: defaultdict(dict))
    dims_seen = set()
    for r in rows:
        if r["hidden"] != fixed_hidden or r["batch"] != fixed_batch:
            continue
        by_backend[r["backend"]][r["in_dim"]][r["loss_name"]] = r["peak_mem_mb"]
        dims_seen.add(r["in_dim"])

    lines = []
    for bk in sorted(by_backend):
        lines.append(f"### {bk}  (hidden={fixed_hidden}, batch={fixed_batch}, peak MB)")
        lines.append("")
        header = "| d | " + " | ".join(cols) + " |"
        sep = "| ---: | " + " | ".join(["---:"] * len(cols)) + " |"
        lines.append(header)
        lines.append(sep)
        for d in sorted(dims_seen):
            row = by_backend[bk].get(d, {})
            cells = []
            for c in cols:
                v = row.get(c)
                if v is None:
                    cells.append("—")
                elif v != v:
                    cells.append("nan")
                else:
                    cells.append(f"{v:.0f}")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "SUMMARY.md",
    )
    args = p.parse_args()

    rows = load_all(args.results_dir)
    if not rows:
        print(f"no results found in {args.results_dir}")
        return 1

    out = ["# Fisher-Divergence Profiling — Summary", ""]
    out.append(f"Aggregated {len(rows)} timing rows from `{args.results_dir.name}/`.")
    out.append("")

    out.append("## Wall-clock per training step")
    out.append("")
    hiddens = sorted({r["hidden"] for r in rows if r.get("hidden") is not None})
    batches = sorted({r["batch"] for r in rows if r.get("batch") is not None})
    for hidden in hiddens:
        for batch in batches:
            out.append(make_speed_table(rows, hidden, batch))

    out.append("## Ratio vs DSM (lower = closer to DSM cost)")
    out.append("")
    for hidden in hiddens:
        for batch in batches:
            out.append(make_ratio_table(rows, hidden, batch))

    out.append("## Peak memory")
    out.append("")
    for hidden in hiddens:
        for batch in batches:
            out.append(make_memory_table(rows, hidden, batch))

    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
