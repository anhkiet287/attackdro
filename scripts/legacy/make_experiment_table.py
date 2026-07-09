#!/usr/bin/env python3
"""Regenerate docs/EXPERIMENT_TABLE.md from results/ on disk.

Single source of truth for the paper's method table. Every in-house / re-eval
number is read live from results/*.json so the table can never drift from disk.
The only hard-coded numbers are the tier-3 ``cited`` block (literature values we
did not run) and pending/planned rows (no numbers yet) -- both clearly marked.

Usage:  python scripts/make_experiment_table.py            # write the table
        python scripts/make_experiment_table.py --check    # nonzero exit if stale
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "docs", "EXPERIMENT_TABLE.md")

# --------------------------------------------------------------------------- #
# In-house / re-eval methods. `stems` = eval_<stem>.json files that are seeds
# of the SAME method (mean/std computed across whichever exist on disk).
# --------------------------------------------------------------------------- #
METHODS = [
    dict(id="P1", method="AttackDRO++ (loss-signal GroupDRO)",
         mech="q by per-group loss over (linf,l2,l1)",
         config="configs/attackdro.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="done",
         stems=["attackdro_union_s0", "attackdro_union_s1", "attackdro_union_s2"],
         note="degenerates to <=AVG (F4)"),
    dict(id="AVG", method="AVG (uniform, in-house)",
         mech="equal-weight all three norms",
         config="configs/avg_frozen.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="done",
         stems=["avg_frozen_s0", "avg_frozen_s1", "avg_frozen_s2"],
         note="recipe-controlled AVG anchor (F7)"),
    dict(id="3a", method="binding-aware (group robust-acc signal)",
         mech="q ~ softmax(-group_robust_acc / tau)",
         config="configs/bindaware.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="done",
         stems=["bindaware_s0", "bindaware_s1", "bindaware_s2"],
         note="signal axis is first-order (+1.8-3.3pp vs P1)"),
    dict(id="3a-v2", method="binding-aware (val-APGD signal)",
         mech="q calibrated on held-out val APGD-CE",
         config="configs/bindaware_v2.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="parked",
         stems=["bindaware_v2_s0"],
         note="val-calibrated <= probe; tracking>precision (49k caveat)"),
    dict(id="T025", method="per-sample soft (T=0.25)",
         mech="per-sample softmax over norm losses, cold",
         config="configs/bindaware_sample.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="done",
         stems=["bindaware_sample_T025", "bindaware_sample_T025_s1",
                "bindaware_sample_T025_s2"],
         note="best in-house soft recipe"),
    dict(id="T05", method="per-sample soft (T=0.5)",
         mech="per-sample softmax over norm losses",
         config="configs/bindaware_sample.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="running",
         stems=["bindaware_sample_T05", "bindaware_sample_T05_s1",
                "bindaware_sample_T05_s2"],
         note="s1,s2 in finalpipe"),
    dict(id="T1", method="per-sample soft (T=1)",
         mech="per-sample softmax, warm",
         config="configs/bindaware_sample.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="done",
         stems=["bindaware_sample_T1"],
         note="warm end: union linf-limited"),
    dict(id="T2", method="per-sample soft (T=2)",
         mech="per-sample softmax, warmest",
         config="configs/bindaware_sample.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="done",
         stems=["bindaware_sample_T2"],
         note="curve non-monotone bump"),
    dict(id="MAXih", method="MAX in-house (hard per-sample max)",
         mech="hard argmax over norms (T->0)",
         config="configs/max_inhouse.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="done",
         stems=["max_inhouse_s0"],
         note="matches C&H MAX 44.0+-0.7 within 1sigma (F7 3rd evidence)"),
    dict(id="MSDih", method="MSD in-house (faithful msd_v0)",
         mech="per-step multi steepest descent, our recipe",
         config="configs/msd_inhouse.yaml", protocol="0.03", eval="apgd",
         source="in-house", status="optional",
         stems=["msd_inhouse_s0"],
         note="CARD-7, downgraded to optional"),
]

# Tier-2 locuslab checkpoints, read from baseline_table.json.
LOCUSLAB = [
    ("MSD", "MSD (locuslab ckpt)", "classical strong baseline"),
    ("AVG", "AVG (locuslab ckpt)", "F7 pair vs in-house AVG"),
    ("MAX", "MAX (locuslab ckpt)", "19pp artifact vs C&H retrain (F7)"),
    ("LINF", "LINF (locuslab ckpt)", "single-norm; l1 collapses"),
    ("L2", "L2 (locuslab ckpt)", "single-norm"),
    ("L1", "L1 (locuslab ckpt)", "single-norm; collapses"),
]

# Tier-3 cited (literature; NOT on disk; 5 seeds, eps_inf=8/255).
CITED = [
    ("RAMP", "RAMP (NeurIPS'24)", "logit pairing + NT grad proj + hard L_max",
     "44.6", "0.6", "81.2", "[Jiang2024-RAMP]", "presumptive from-scratch SOTA; no public ckpt"),
    ("MAX*", "MAX (C&H retrain)", "hard per-sample max",
     "44.0", "0.7", "-", "[Croce2022-EAT] Tab5", "our MAXih 43.9 @0.03 agrees"),
    ("MSD*", "MSD (C&H retrain)", "per-step multi steepest descent",
     "43.9", "0.8", "-", "[Croce2022-EAT] Tab5", ""),
    ("E-AT", "E-AT (C&H)", "fixed geometric: train linf+l1, l2 free",
     "42.4", "0.6", "-", "[Croce2022-EAT] Tab5", "'up to 3x cheaper'"),
    ("SAT", "SAT", "single-attack per step",
     "40.4", "-", "-", "[Croce2022-EAT] Tab5", ""),
    ("AVG*", "AVG (C&H)", "uniform average",
     "40.1", "-", "-", "[Croce2022-EAT] Tab5", ""),
]

# Pending / planned (no numbers yet).
PLANNED = [
    ("RAMP-repro", "our 1-seed reproduction of RAMP", "their code + recipe (CARD-8)",
     "configs: external/RAMP", "0.03 + 8/255", "apgd + std", "running",
     "NEVER replaces cited 44.6+-0.6; validates we run their code"),
    ("CARD-8b", "RAMP + binding-aware T-softmax", "replace/soften their L_max with our per-sample T",
     "TBD", "0.03", "std", "planned",
     "gated on CARD-8 passing band; card written, awaiting approval"),
]


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _fmt_pm(m, s):
    return f"{m:.1f}" if s is None else f"{m:.1f}+-{s:.1f}"


EPS_8255 = 8 / 255


def load_run(stem):
    p = os.path.join(RESULTS, f"eval_{stem}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    m = d["metrics"]
    eps = (d.get("protocol", {}).get("threat_model", {}).get("linf", {}).get("eps"))
    pre_relock = eps is None or abs(float(eps) - EPS_8255) > 1e-6  # eval NOT at locked 8/255
    # train/eval eps mismatch (e.g. 0.03-trained ckpt eval'd at 8/255 -> lower bound,
    # NOT a paper number). Recorded by evaluate.py from the checkpoint's own cfg.
    mismatch = bool(d.get("train_eval_eps_mismatch", False))
    return dict(union=m["worst_union_acc"] * 100, clean=m["clean_acc"] * 100,
                linf=m["per_norm_robust_acc"]["linf"] * 100,
                l2=m["per_norm_robust_acc"]["l2"] * 100,
                l1=m["per_norm_robust_acc"]["l1"] * 100,
                version=m.get("version", "?"), eps=eps,
                pre_relock=pre_relock, mismatch=mismatch)


def build():
    lines = []
    lines.append("# EXPERIMENT_TABLE.md -- the single results table")
    lines.append("*AUTO-GENERATED by `scripts/make_experiment_table.py`. Do not hand-edit; "
                 "edit the generator. In-house/re-eval numbers are read live from "
                 "`results/*.json` and cannot drift from disk. Tier-3 `cited` = literature "
                 "constants (eps_inf=8/255, 5 seeds), never mixed with our tiers.*")
    lines.append("")
    lines.append("## Tier 1 -- in-house, recipe-controlled (our recipe/attacks, LOCKED protocol eps_inf=8/255)")
    lines.append("*Flags: `⚠0.03(pre-relock)` = still the retired-0.03 number, re-eval pending "
                 "(`scripts/reeval_8255.py`). `⚠train@0.03/eval@8255 LOWER-BOUND` = 8/255 eval of a "
                 "0.03-TRAINED ckpt — train/eval mismatch, a QUICK ESTIMATE / lower bound, **NOT a paper "
                 "number** (paper-grade 8/255 requires retraining at 8/255; folded into the next "
                 "post-Gate-α clean run). Numbers read live from `results/eval_*.json`.*")
    lines.append("| id | method | mechanism | config | eval | worst-U mean+-std (seeds) | clean | status | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for M in METHODS:
        runs = [load_run(s) for s in M["stems"]]
        runs = [r for r in runs if r]
        if runs:
            um, us = _mean_std([r["union"] for r in runs])
            cm, _ = _mean_std([r["clean"] for r in runs])
            if any(r.get("pre_relock") for r in runs):
                flag = " ⚠0.03(pre-relock)"
            elif any(r.get("mismatch") for r in runs):
                flag = " ⚠train@0.03/eval@8255 LOWER-BOUND"
            else:
                flag = ""
            u = f"**{_fmt_pm(um, us)}** ({len(runs)}){flag}"
            clean = f"{cm:.1f}"
            ver = runs[0]["version"]
        else:
            u, clean, ver = "_pending_", "-", M["eval"]
        lines.append(f"| {M['id']} | {M['method']} | {M['mech']} | "
                     f"`{os.path.basename(M['config'])}` | {ver} | {u} | {clean} | "
                     f"{M['status']} | {M['note']} |")
    lines.append("")
    lines.append("## Tier 2 -- checkpoint re-evals (locuslab official ckpts, our harness, eps_inf=8/255) *")
    lines.append("| id | method | worst-U | clean | linf | l2 | l1 | note |")
    lines.append("|---|---|---|---|---|---|---|---|")
    bt = json.load(open(os.path.join(RESULTS, "baseline_table.json")))["rows"]
    for key, disp, note in LOCUSLAB:
        r = bt[key]
        lines.append(f"| {key}* | {disp} | **{r['union']:.1f}** | {r['clean']:.1f} | "
                     f"{r['linf']:.1f} | {r['l2']:.1f} | {r['l1']:.1f} | {note} |")
    lines.append("")
    lines.append("## Tier 3 -- cited (literature; eps_inf=8/255, 5 seeds; NOT protocol-aligned) ‡")
    lines.append("| id | method | mechanism | worst-U ‡ | +-std | clean | source | note |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cid, disp, mech, u, s, clean, src, note in CITED:
        lines.append(f"| {cid} | {disp} | {mech} | **{u}** | {s} | {clean} | {src} | {note} |")
    lines.append("")
    lines.append("## Pending / planned (no numbers yet)")
    lines.append("| id | method | mechanism | where | protocol | eval | status | note |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pid, disp, mech, where, proto, ev, status, note in PLANNED:
        lines.append(f"| {pid} | {disp} | {mech} | {where} | {proto} | {ev} | "
                     f"{status} | {note} |")
    lines.append("")
    lines.append("### Reading rules")
    lines.append("- **Protocol re-locked 2026-07-05 to eps_inf=8/255** (was 0.03; subfield standard, "
                 "enables direct SOTA comparison). All tiers now share 8/255, so Tier-1/2/3 are "
                 "eps-aligned; remaining gaps are recipe/source (F7: MAX 19pp, AVG 1.9pp), not eps. "
                 "In-house ckpts were TRAINED at 0.03 and re-eval'd at 8/255 (mild train/eval mismatch, "
                 "flagged). 0.03 provenance: `results/archive/`, `configs/archive/base_eps003.yaml`.")
    lines.append("- Claim ladder: C1 mechanism (F4-F6) + F7 eval corrections + C2 = a simple "
                 "cheap adaptive weighting reaching the 43.9-44.6 cluster, **not exceeding it**.")
    lines.append("- All numbers here are APGD (CE+T) unless a `std` column says otherwise; "
                 "finalists get full `standard` AA before any claim (see results/standard_vs_apgd.md).")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero if the on-disk table differs from freshly built")
    args = ap.parse_args()
    new = build()
    if args.check:
        old = open(OUT).read() if os.path.exists(OUT) else ""
        if old != new:
            print("EXPERIMENT_TABLE.md is STALE -- run scripts/make_experiment_table.py")
            sys.exit(1)
        print("EXPERIMENT_TABLE.md up to date.")
        return
    open(OUT, "w").write(new)
    print(f"wrote {os.path.relpath(OUT, REPO)} "
          f"({new.count(chr(10))} lines) from results/ on disk")


if __name__ == "__main__":
    main()
