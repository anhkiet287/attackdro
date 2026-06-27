#!/usr/bin/env python3
"""Submit a job to the AttackDRO queue — run on the Mac (or the PC).

Everything after `--` is the command the PC will run from the repo root.

Examples:
    python jobs/submit.py --name exp1 -- python src/train.py --config configs/default.yaml
    python jobs/submit.py --name lr20  -- python src/train.py --epochs 20

After submitting from the Mac, push it to the PC:   bash jobs/sync.sh push
"""
import argparse
import json
import socket
from datetime import datetime
from pathlib import Path

JOBS = Path(__file__).resolve().parent
QUEUE = JOBS / "queue"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="job", help="short label for the run")
    p.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="command after --, e.g. -- python src/train.py"
    )
    a = p.parse_args()

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        p.error("provide a command after --, e.g. -- python src/train.py --config configs/default.yaml")

    jid = f"{datetime.now():%Y%m%d_%H%M%S}_{a.name}"
    spec = {
        "id": jid,
        "name": a.name,
        "cmd": cmd,
        "cwd": ".",
        "submitted_by": socket.gethostname(),
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "status": "queued",
    }
    QUEUE.mkdir(parents=True, exist_ok=True)
    out = QUEUE / f"{jid}.json"
    out.write_text(json.dumps(spec, indent=2))

    print(f"queued: {out.name}")
    print(f"  cmd : {' '.join(cmd)}")
    print("next: bash jobs/sync.sh push   (then the PC agent runs it)")


if __name__ == "__main__":
    main()
