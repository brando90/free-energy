#!/usr/bin/env python3
"""Write the missing Lean Workbook hidden-state row indices to a JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "results" / "leanworkbook_plus_goedel_hidden_states_gpus0_3_contextonly"
DEFAULT_TOTAL_ROWS = 25214


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--total-rows", type=int, default=DEFAULT_TOTAL_ROWS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hs_dir = args.out_dir / "hidden_states_safetensors"
    output = args.output or args.out_dir / "missing_row_indices.json"
    have = {int(path.stem) for path in hs_dir.glob("*.safetensors") if path.stem.isdigit()}
    missing = [row_index for row_index in range(args.total_rows) if row_index not in have]
    payload = {"total_rows": args.total_rows, "present": len(have), "missing": len(missing), "indices": missing}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"present": len(have), "missing": len(missing), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
