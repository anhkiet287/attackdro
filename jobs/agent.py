#!/usr/bin/env python3
"""AttackDRO job-dispatch worker — runs on the PC, next to the GPU.

Watches jobs/queue/ for new job specs (JSON), runs them one at a time on the
GPU, and writes logs + status into jobs/logs, jobs/done, jobs/failed.

Run inside WSL2 with the venv active:
    source .venv/bin/activate
    python jobs/agent.py                                   # foreground
    nohup python jobs/agent.py > jobs/agent.out 2>&1 &     # background
"""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

JOBS = Path(__file__).resolve().parent
REPO = JOBS.parent
QUEUE, RUNNING, DONE, FAILED, LOGS = (
    JOBS / d for d in ("queue", "running", "done", "failed", "logs")
)
POLL_SECONDS = 3


def ensure_dirs() -> None:
    for d in (QUEUE, RUNNING, DONE, FAILED, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def claim_next() -> Path | None:
    """Atomically move the oldest queued job into running/ (acts as a lock)."""
    for f in sorted(QUEUE.glob("*.json")):
        target = RUNNING / f.name
        try:
            os.rename(f, target)
            return target
        except OSError:
            continue
    return None


def run_job(path: Path) -> None:
    spec = json.loads(path.read_text())
    jid = spec.get("id", path.stem)
    cmd = spec["cmd"]
    cwd = REPO / spec.get("cwd", ".")
    log_path = LOGS / f"{jid}.log"

    spec["status"] = "running"
    spec["started_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(spec, indent=2))
    print(f"[agent] running {jid}: {' '.join(cmd)}")

    with open(log_path, "w") as log:
        log.write(
            f"# job   : {jid}\n# cmd   : {' '.join(cmd)}\n"
            f"# start : {spec['started_at']}\n\n"
        )
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    code = proc.returncode

    spec["status"] = "done" if code == 0 else "failed"
    spec["exit_code"] = code
    spec["finished_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(spec, indent=2))
    dest = (DONE if code == 0 else FAILED) / path.name
    os.rename(path, dest)
    print(f"[agent] {jid} -> {spec['status']} (exit {code}); log: {log_path}")


def main() -> None:
    ensure_dirs()
    print(f"[agent] watching {QUEUE} every {POLL_SECONDS}s. Ctrl-C to stop.")
    try:
        while True:
            job = claim_next()
            if job is None:
                time.sleep(POLL_SECONDS)
                continue
            try:
                run_job(job)
            except Exception as e:  # never let one bad job kill the worker
                print(f"[agent] ERROR on {job.name}: {e}")
                try:
                    os.rename(job, FAILED / job.name)
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("\n[agent] stopped.")


if __name__ == "__main__":
    main()
