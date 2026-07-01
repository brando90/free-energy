#!/usr/bin/env python3
"""Lean REPL environment helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
DEFAULT_REPL_DIR = HERE / "repl"
DEFAULT_LEAN_ENV_DIR = DEFAULT_REPL_DIR / "test" / "Mathlib"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repl_env(
    repl_dir: Path = DEFAULT_REPL_DIR,
    lean_env_dir: Path = DEFAULT_LEAN_ENV_DIR,
) -> tuple[Path, Path]:
    repl_dir = repl_dir.resolve()
    lean_env_dir = lean_env_dir.resolve()
    repl_bin = repl_dir / ".lake" / "build" / "bin" / "repl"
    setup_marker = lean_env_dir / ".leanworkbook_setup_complete"
    if not repl_dir.exists():
        _run(["git", "clone", "https://github.com/leanprover-community/repl.git", str(repl_dir)])
        _run(["git", "checkout", "adbbfcb9d4e61c12db96c45d227de92f21cc17dd"], cwd=repl_dir)
    if not repl_bin.exists():
        _run(["lake", "build"], cwd=repl_dir)
    if not setup_marker.exists():
        _run(["bash", "test.sh"], cwd=repl_dir / "test" / "Mathlib")
        setup_marker.write_text("ok\n", encoding="utf-8")
    return repl_dir, lean_env_dir
