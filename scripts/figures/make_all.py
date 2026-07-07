#!/usr/bin/env python3
"""Regenerate every paper/presentation figure with one command.

Thin orchestrator: delegates to the existing plotters (which already know their
own inputs) and is tolerant of individual failures so a missing log for one
figure never blocks the rest. Reads only from results/ on disk -- no training,
CPU-only, safe to run any time.

    python scripts/figures/make_all.py            # regenerate all figures
    python scripts/figures/make_all.py --table    # also refresh EXPERIMENT_TABLE.md
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable

# (script, args) invocations that read from results/ with sensible defaults.
JOBS = [
    ("make_presentation_pack.py", []),          # 4 core figs + MANIFEST
    ("plot_union_vs_T.py", []),                 # default --out
    ("plot_state_of_play.py", []),              # leaderboard + per-norm finalists
]


def run(script, args):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        return script, "MISSING"
    try:
        subprocess.run([PY, path, *args], cwd=REPO, check=True,
                       capture_output=True, text=True, timeout=600)
        return script, "ok"
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or e.stdout or "").strip().splitlines()[-1:] or [""]
        return script, f"FAILED: {tail[0][:120]}"
    except Exception as e:  # noqa: BLE001
        return script, f"FAILED: {type(e).__name__}: {e}"


def main():
    if "--table" in sys.argv:
        run_ = run("../make_experiment_table.py", [])
        print(f"  table: {run_[1]}")
    print("Regenerating figures from results/ on disk ...")
    ok = 0
    for script, args in JOBS:
        name, status = run(script, args)
        print(f"  {name:32s} {status}")
        ok += status == "ok"
    print(f"{ok}/{len(JOBS)} figure jobs succeeded. "
          f"Outputs under results/figures/ (see MANIFEST).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
