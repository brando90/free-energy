#!/usr/bin/env python3
"""Atomically update queue_status.json after every E1 stage transition.

Maintains {"queue": [...models...], "current": {...}, "history": [...],
"updated_at": ...}. Each call records one event (start|done) for a (model, stage)
with an rc. Written via temp-file + os.replace so a reader never sees a partial file.
"""
import argparse
import datetime as _dt
import json
import os


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--event", required=True, choices=["start", "done"])
    ap.add_argument("--rc", type=int, default=None)
    ap.add_argument("--queue", default=None, help="comma-separated model order (init only)")
    ap.add_argument("--note", default=None)
    a = ap.parse_args()

    doc = {"queue": [], "current": {}, "history": [], "updated_at": None}
    if os.path.exists(a.file):
        try:
            doc = json.load(open(a.file))
        except Exception:
            pass
    if a.queue:
        doc["queue"] = a.queue.split(",")

    evt = {"model": a.model, "stage": a.stage, "event": a.event,
           "rc": a.rc, "at": now_iso()}
    if a.note:
        evt["note"] = a.note
    doc.setdefault("history", []).append(evt)
    # Flat current-state view (the spec's stage/model/started/done/rc).
    cur = doc.get("current") or {}
    cur["model"] = a.model
    cur["stage"] = a.stage
    if a.event == "start":
        cur["started"] = evt["at"]
        cur["done"] = None
        cur["rc"] = None
    else:
        cur["done"] = evt["at"]
        cur["rc"] = a.rc
    if a.note:
        cur["note"] = a.note
    doc["current"] = cur
    doc["updated_at"] = now_iso()

    tmp = a.file + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp, a.file)


if __name__ == "__main__":
    main()
