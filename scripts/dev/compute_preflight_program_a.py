#!/usr/bin/env python
"""Compute-budget preflight for Program-A (pre-reg v3-A A4; critic conditions 7,8).

FAIL-CLOSED. CODE ONLY: no model, no data, no attacks — pure schedule accounting.

For each config, the FINITE-RUN scheduled attack-step-unit total is computed
DIRECTLY from the resolved schedule (NOT the 12*B average shorthand), mirroring
the exact per-batch firing predicates in groupdro.py:

  s          = per-norm attack steps (full_attack_steps[g]); must be equal across norms
  G          = number of source norms (linf/l2/l1 -> 3)
  train_pool = train_dataset_size - dataset.val_holdout          (cifar10: 50000)
  B          = epochs * (train_pool // batch_size)               (drop_last=True)

  global_batch = epoch*len(train_loader) + batch_idx, contiguous over [0, B).

  static_cycle + static_extra  (B3-mis, B3-informed):
     extra fires  <=> global_batch % E == 0            (groupdro.py:773-774)
     N_extra   = |{b in [0,B): b % E == 0}|
     ordinary batch cost = s ; extra-firing batch cost = 2s
     C_static  = s*B + s*N_extra

  predictive_refresh           (B4-adaptive):
     refresh fires <=> global_batch % R == 0           (groupdro.py:917-921; b0 refreshes)
     N_refresh = |{b in [0,B): b % R == 0}|
     ordinary batch cost = s ; refresh batch cost = G*s (all norms)
     C_B4      = s*B + s*(G-1)*N_refresh

Assertions (any failure => exit 1):
  * tau_C = max(total) - min(total) == 0 across all given configs.
  * s equal across norms in every config; s and G identical across configs.
  * B identical across configs (same epochs/batch_size/val_holdout/dataset).
  * registered policy: static E == EXTRA_EVERY(5), predictive R == REFRESH_EVERY(10),
    static cycle length == CYCLE_LEN(4), G == EXPECT_G(3).
  * no attack spec carries restarts > 1 (would be uncounted extra work).
  * B4 issues NO uncounted attacks for its signal: predictive_use_cheap_probe is
    False and predictive_score_rule == 'loss_plus_bind' (the signal is derived from
    the refresh-batch all-source attacks that are already counted in C_B4).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robustdro.utils.io import load_config  # noqa: E402

# CIFAR-10 fixed sizes + registered Program-A policy constants.
TRAIN_TOTAL = 50000
EXTRA_EVERY = 5
REFRESH_EVERY = 10
CYCLE_LEN = 4
EXPECT_G = 3
EXPECT_STEPS = 10


class PreflightError(RuntimeError):
    pass


def _count_multiples(B: int, m: int) -> int:
    """|{b in [0, B): b % m == 0}| (includes b == 0)."""
    if m <= 0:
        raise PreflightError(f"modulus must be positive, got {m}")
    if B <= 0:
        return 0
    return (B + m - 1) // m


def _global_batches(cfg: dict) -> int:
    dcfg = cfg["dataset"]
    if dcfg.get("name") != "cifar10":
        raise PreflightError(f"preflight assumes cifar10, got {dcfg.get('name')!r}")
    tcfg = cfg["train"]
    holdout = int(dcfg.get("val_holdout", 0))
    bs = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])
    train_pool = TRAIN_TOTAL - holdout
    bpe = train_pool // bs  # drop_last=True
    return epochs * bpe, dict(train_pool=train_pool, batch_size=bs, epochs=epochs,
                              batches_per_epoch=bpe)


def _step_unit(cfg: dict) -> int:
    attacks = cfg["train"].get("attacks", []) or []
    if not attacks:
        raise PreflightError("no train.attacks in config")
    steps = []
    for a in attacks:
        s = int(a.get("steps"))
        if int(a.get("restarts", 1)) > 1:
            raise PreflightError(f"attack {a.get('norm')} has restarts>1 (uncounted work)")
        steps.append(s)
    if len(set(steps)) != 1:
        raise PreflightError(f"per-norm steps not equal: {steps}")
    s = steps[0]
    if s != EXPECT_STEPS:
        raise PreflightError(f"expected {EXPECT_STEPS} steps/norm, got {s}")
    return s, len(attacks)


def schedule_total(path: str) -> dict:
    cfg = load_config(path)
    tcfg = cfg["train"]
    mode = tcfg.get("allocation_mode")
    B, bmeta = _global_batches(cfg)
    s, G = _step_unit(cfg)
    if G != EXPECT_G:
        raise PreflightError(f"{path}: expected G={EXPECT_G} norms, got {G}")

    row = {"config": path, "allocation_mode": mode, "s": s, "G": G, "B": B, **bmeta}

    if mode == "static_cycle":
        order = tcfg.get("static_cycle_order") or []
        E = int(tcfg.get("static_extra_every_n_batches", 0))
        if len(order) != CYCLE_LEN:
            raise PreflightError(f"{path}: cycle length {len(order)} != {CYCLE_LEN}")
        if E != EXTRA_EVERY:
            raise PreflightError(f"{path}: static_extra_every {E} != {EXTRA_EVERY}")
        if str(tcfg.get("static_extra_mode")) != "aggregate_loss":
            raise PreflightError(f"{path}: static_extra_mode must be aggregate_loss")
        N_extra = _count_multiples(B, E)
        total = s * B + s * N_extra
        row.update(special_kind="extra", every=E, N_special=N_extra,
                   ordinary_cost=s, special_cost=2 * s, total=total)
    elif mode == "predictive_refresh":
        R = int(tcfg.get("predictive_refresh_every_n_batches", 0))
        if R != REFRESH_EVERY:
            raise PreflightError(f"{path}: refresh_every {R} != {REFRESH_EVERY}")
        if bool(tcfg.get("predictive_use_cheap_probe", False)):
            raise PreflightError(f"{path}: predictive_use_cheap_probe=True -> uncounted attacks")
        if str(tcfg.get("predictive_score_rule")) != "loss_plus_bind":
            raise PreflightError(f"{path}: score_rule must be loss_plus_bind")
        N_refresh = _count_multiples(B, R)
        total = s * B + s * (G - 1) * N_refresh
        row.update(special_kind="refresh", every=R, N_special=N_refresh,
                   ordinary_cost=s, special_cost=G * s, total=total)
    else:
        raise PreflightError(f"{path}: unsupported allocation_mode {mode!r}")
    return row


def run_preflight(paths: list[str]) -> list[dict]:
    if len(paths) < 2:
        raise PreflightError("need >= 2 configs to assert tau_C == 0")
    rows = [schedule_total(p) for p in paths]

    Bs = {r["B"] for r in rows}
    if len(Bs) != 1:
        raise PreflightError(f"B differs across configs: {Bs}")
    if len({r["s"] for r in rows}) != 1 or len({r["G"] for r in rows}) != 1:
        raise PreflightError("s or G differs across configs")

    totals = [r["total"] for r in rows]
    tau_C = max(totals) - min(totals)

    print("=== Program-A compute preflight (finite-run scheduled attack-step units) ===")
    hdr = f"{'config':<62} {'mode':<19} {'B':>7} {'special':>8} {'N_spec':>7} {'total':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{os.path.basename(r['config']):<62} {r['allocation_mode']:<19} "
              f"{r['B']:>7} {r['special_kind']:>8} {r['N_special']:>7} {r['total']:>10}")
    print("-" * len(hdr))
    print(f"tau_C = max-min = {tau_C}")

    if tau_C != 0:
        raise PreflightError(f"COMPUTE MISMATCH: tau_C = {tau_C} != 0 (totals={totals})")
    print("PREFLIGHT PASS: tau_C = 0 (exact finite-run equality).")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Program-A compute preflight (fail-closed).")
    ap.add_argument("configs", nargs="+", help="Resolved arm config paths (>=2).")
    args = ap.parse_args()
    try:
        run_preflight(args.configs)
    except PreflightError as e:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
