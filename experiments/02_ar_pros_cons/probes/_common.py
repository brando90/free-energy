"""Shared helpers for probes."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import socket
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


@dataclass
class ProbeResult:
    probe: str
    tag: str
    seed: int
    device: str
    started_at: float
    finished_at: float = 0.0
    duration_s: float = 0.0
    host: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_available: bool = field(default_factory=torch.cuda.is_available)
    gpu_name: str = ""
    control_passed: bool = False
    verdict: str = "PENDING"
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tag", default="smoke", help="Run label, e.g. smoke or full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="auto",
        help="cuda | cpu | auto (cuda if available)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the JSON result (defaults to mnt/user-data/outputs/<probe>/<tag>)",
    )
    p.add_argument("--smoke", action="store_true", help="Use the smoke config")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return ""


def default_output_dir(probe: str, tag: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "mnt" / "user-data" / "outputs" / probe / tag


def write_result(result: ProbeResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.finished_at = time.time()
    result.duration_s = result.finished_at - result.started_at
    out_path = out_dir / "result.json"
    with out_path.open("w") as f:
        json.dump(asdict(result), f, indent=2, sort_keys=True, default=str)
    return out_path


def effective_rank(matrix: torch.Tensor, energy: float = 0.99) -> int:
    """Smallest k such that the top-k singular values explain >= energy of total energy."""
    with torch.no_grad():
        s = torch.linalg.svdvals(matrix.detach().float())
        if s.numel() == 0:
            return 0
        e = (s ** 2).cumsum(dim=0)
        total = e[-1].item()
        if total <= 0:
            return 0
        thresh = energy * total
        k = int(torch.searchsorted(e, torch.tensor(thresh, device=e.device)).item()) + 1
        return min(k, s.numel())
