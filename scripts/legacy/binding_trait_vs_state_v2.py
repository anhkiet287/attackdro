#!/usr/bin/env python3
"""Per-sample binding trait-vs-state diagnostic from fixed probe-binding dumps.

Reads dumps/probe_binding/<run>/probe_binding_epNNN.pt, where each file carries
the same CIFAR train sample indices and a [G, B] loss matrix. This is the v2
diagnostic that can actually measure per-sample binding persistence.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "dumps" / "probe_binding"
DEFAULT_OUT = REPO / "results" / "reports" / "trait_state_v2.md"


@dataclass
class RunStats:
    run: str
    epochs: list[int]
    norms: list[str]
    n_samples: int
    aligned: bool
    avg_persist: float
    avg_chance: float
    stable_frac: float
    switcher_frac: float
    stable_by_norm: dict[str, int]
    binding_start: dict[str, float]
    binding_end: dict[str, float]
    verdict: str
    decision_grade: bool


def load_run(run_dir: Path) -> tuple[list[dict], str | None]:
    files = sorted(run_dir.glob("probe_binding_ep*.pt"))
    if not files:
        return [], f"no probe_binding_ep*.pt files in {run_dir}"
    dumps = [torch.load(path, map_location="cpu") for path in files]
    ref_idx = dumps[0]["indices"]
    ref_y = dumps[0]["labels"]
    for d in dumps[1:]:
        if not torch.equal(ref_idx, d["indices"]):
            return dumps, "sample indices differ across dumps"
        if not torch.equal(ref_y, d["labels"]):
            return dumps, "labels differ across dumps"
    return dumps, None


def freq_dict(bind: torch.Tensor, norms: list[str]) -> dict[str, float]:
    freq = torch.bincount(bind, minlength=len(norms)).float() / bind.numel()
    return {norm: freq[i].item() for i, norm in enumerate(norms)}


def analyse_run(run_dir: Path, min_checkpoints: int) -> RunStats | str:
    dumps, err = load_run(run_dir)
    if not dumps:
        return err or f"{run_dir.name}: no dumps"

    norms = list(dumps[0]["norms"])
    epochs = [int(d["epoch"]) for d in dumps]
    Ls = [d["L"].float() for d in dumps]
    binds = [L.argmax(dim=0) for L in Ls]
    aligned = err is None
    if not aligned:
        return f"{run_dir.name}: {err}"

    persists = []
    chances = []
    for a, b in zip(binds, binds[1:]):
        persists.append((a == b).float().mean().item())
        fa = torch.bincount(a, minlength=len(norms)).float() / a.numel()
        fb = torch.bincount(b, minlength=len(norms)).float() / b.numel()
        chances.append(float((fa * fb).sum().item()))
    avg_persist = sum(persists) / len(persists) if persists else float("nan")
    avg_chance = sum(chances) / len(chances) if chances else float("nan")

    B = binds[0].numel()
    stack = torch.stack(binds, dim=0)  # [T, B]
    stable_by_norm = {norm: 0 for norm in norms}
    stable = 0
    for sample_col in stack.t():
        counts = torch.bincount(sample_col, minlength=len(norms))
        mode_count, mode_idx = counts.max(dim=0)
        mode_frac = mode_count.item() / len(epochs)
        if mode_frac > 0.8:
            stable += 1
            stable_by_norm[norms[int(mode_idx)]] += 1
    stable_frac = stable / B
    switcher_frac = 1.0 - stable_frac
    decision_grade = len(epochs) >= min_checkpoints

    if not decision_grade:
        verdict = "SMOKE ONLY - instrumentation/alignment validated; Gate alpha pending real run"
    elif avg_persist > 0.70 and stable_frac > 0.50 and avg_persist - avg_chance > 0.20:
        verdict = "TRAIT COMPONENT - Gate alpha PASS toward predictive weighting branch"
    elif abs(avg_persist - avg_chance) < 0.10 and stable_frac < 0.35:
        verdict = "PURE STATE - pivot to tracking/dynamics framing"
    else:
        verdict = "MIXED - both trait and state; two-tier framework"

    return RunStats(
        run=run_dir.name,
        epochs=epochs,
        norms=norms,
        n_samples=B,
        aligned=aligned,
        avg_persist=avg_persist,
        avg_chance=avg_chance,
        stable_frac=stable_frac,
        switcher_frac=switcher_frac,
        stable_by_norm=stable_by_norm,
        binding_start=freq_dict(binds[0], norms),
        binding_end=freq_dict(binds[-1], norms),
        verdict=verdict,
        decision_grade=decision_grade,
    )


def pct(x: float) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{100 * x:.1f}%"


def render(stats: list[RunStats], errors: list[str]) -> str:
    lines = [
        "# Trait-vs-State v2",
        "",
        "Source: fixed indexed probe-binding dumps (`dumps/probe_binding/<run>/probe_binding_epNNN.pt`).",
        "Each dump must carry the same `indices`, `labels`, `norms`, and `L=[G,B]` loss matrix.",
        "",
        "## Pre-registered Gate",
        "- Persistence >> chance and clear stable majority -> TRAIT component; Gate alpha PASS toward predictive weighting.",
        "- Persistence ~= chance -> pure STATE; pivot to tracking/dynamics framing.",
        "- Mixed -> both; two-tier framework.",
        "",
        "## Runs",
    ]
    if not stats:
        lines.append("- No valid fixed-probe runs found.")
    for s in stats:
        lines.extend([
            f"### {s.run}",
            f"- Checkpoints: {len(s.epochs)} epochs {s.epochs}",
            f"- Samples: {s.n_samples}; index alignment: {'OK' if s.aligned else 'FAILED'}",
            f"- Consecutive same-binding persistence: {pct(s.avg_persist)}",
            f"- Marginal-frequency chance baseline: {pct(s.avg_chance)}",
            f"- Stable-trait samples (>80% same binding): {pct(s.stable_frac)}",
            f"- Switchers: {pct(s.switcher_frac)}",
            "- Stable traits by norm: "
            + ", ".join(f"{k}={v}" for k, v in s.stable_by_norm.items()),
            "- Binding frequency first epoch: "
            + ", ".join(f"{k}={pct(v)}" for k, v in s.binding_start.items()),
            "- Binding frequency last epoch: "
            + ", ".join(f"{k}={pct(v)}" for k, v in s.binding_end.items()),
            f"- Verdict: **{s.verdict}**",
            "",
        ])
    if errors:
        lines.extend(["## Skipped/Errors", ""])
        lines.extend(f"- {e}" for e in errors)
        lines.append("")

    if any(s.decision_grade for s in stats):
        counts = Counter(s.verdict for s in stats if s.decision_grade)
        overall = counts.most_common(1)[0][0]
    elif stats:
        overall = "Gate alpha pending: only smoke/non-decision-grade fixed-probe dumps available."
    else:
        overall = "Gate alpha pending: no fixed-probe dumps available."
    lines.extend(["## Overall Verdict", f"- {overall}", ""])
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--min-checkpoints", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if args.run:
        run_dirs = [root / run for run in args.run]
    else:
        run_dirs = [Path(p) for p in sorted(glob.glob(str(root / "*")))]
    stats: list[RunStats] = []
    errors: list[str] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            errors.append(f"{run_dir}: not a directory")
            continue
        res = analyse_run(run_dir, args.min_checkpoints)
        if isinstance(res, RunStats):
            stats.append(res)
        else:
            errors.append(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(stats, errors), encoding="utf-8")
    print(f"Wrote {out.resolve().relative_to(REPO)}")


if __name__ == "__main__":
    main()
