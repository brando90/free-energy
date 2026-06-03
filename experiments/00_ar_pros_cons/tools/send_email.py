"""Send a results-summary email after the snap runs finish.

Reads the Gmail app password from $GMAIL_APP_PASSWORD (path or literal). Sends a
plain-text email summarising every probe result JSON it can find under
``mnt/user-data/outputs``.

Usage:
    python -m tools.send_email \
        --to brandojazz@gmail.com \
        --from brandojazz@gmail.com \
        --subject "[00_ar_pros_cons] full run done" \
        --outputs-dir mnt/user-data/outputs
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import socket
import ssl
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from typing import List


def load_password() -> str:
    raw = os.environ.get("GMAIL_APP_PASSWORD") or ""
    if raw and Path(raw).is_file():
        raw = Path(raw).read_text().strip()
    elif not raw:
        candidate = Path.home() / "keys" / "gmail_app_password.txt"
        if candidate.is_file():
            raw = candidate.read_text().strip()
    if not raw:
        raise SystemExit("ERROR: no Gmail app password found (set GMAIL_APP_PASSWORD or ~/keys/gmail_app_password.txt)")
    return raw.replace(" ", "")


def gather_results(outputs_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(outputs_dir.rglob("result.json")):
        try:
            with path.open() as f:
                payload = json.load(f)
        except Exception as e:  # noqa: BLE001
            rows.append({"path": str(path), "error": repr(e)})
            continue
        rows.append(
            {
                "path": str(path),
                "probe": payload.get("probe"),
                "tag": payload.get("tag"),
                "verdict": payload.get("verdict"),
                "control_passed": payload.get("control_passed"),
                "duration_s": payload.get("duration_s"),
                "device": payload.get("device"),
                "gpu_name": payload.get("gpu_name"),
                "host": payload.get("host"),
                "metrics_keys": sorted(list((payload.get("metrics") or {}).keys()))[:10],
            }
        )
    return rows


def format_body(rows: List[dict], extras: dict) -> str:
    lines: List[str] = []
    lines.append("ar_pros_cons probe runs summary")
    lines.append("=" * 40)
    lines.append("")
    for key, val in extras.items():
        lines.append(f"{key}: {val}")
    lines.append("")
    if not rows:
        lines.append("(no result.json files found)")
        return "\n".join(lines)
    for r in rows:
        lines.append(f"- {r.get('probe') or r.get('path')}  [{r.get('tag')}]")
        if "error" in r:
            lines.append(f"    ERROR reading: {r['error']}")
            continue
        lines.append(
            f"    verdict={r.get('verdict')}  control_passed={r.get('control_passed')}  "
            f"device={r.get('device')}  gpu={r.get('gpu_name')}  host={r.get('host')}"
        )
        lines.append(f"    duration_s={r.get('duration_s')}  path={r.get('path')}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--to", default="brandojazz@gmail.com")
    p.add_argument("--from", dest="from_addr", default="brandojazz@gmail.com")
    p.add_argument("--subject", default="[00_ar_pros_cons] probe run summary")
    p.add_argument("--outputs-dir", default="mnt/user-data/outputs")
    p.add_argument("--note", default="", help="Extra free-text note to include in the body")
    p.add_argument("--attach", action="append", default=[], help="Path of a file to attach; can repeat")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    rows = gather_results(outputs_dir)
    extras = {
        "host": socket.gethostname(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outputs_dir": str(outputs_dir),
        "note": args.note,
    }
    body = format_body(rows, extras)

    msg = EmailMessage()
    msg["From"] = args.from_addr
    msg["To"] = args.to
    msg["Subject"] = args.subject
    msg.set_content(body)
    for path_str in args.attach:
        path = Path(path_str)
        if not path.is_file():
            print(f"WARN: cannot attach missing file {path}", file=sys.stderr)
            continue
        with path.open("rb") as fh:
            data = fh.read()
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=path.name)

    password = load_password()
    try:
        import certifi  # type: ignore
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(args.from_addr, password)
        smtp.send_message(msg)
    print(f"[send_email] sent {len(rows)} probe rows to {args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
