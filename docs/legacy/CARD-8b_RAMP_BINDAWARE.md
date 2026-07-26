# CARD-8b: RAMP + Binding-Aware Pilot Ladder

Status: DRAFT, show Kiet before any smoke or launch. Do not patch training until this card is approved. Phase-A RAMP baselines are locked from disk.

## CARD-8 / Phase-A Anchor

- CARD-8 validation result: local RAMP reproduction passed. RAMP's own `--final_eval` gave union `45.2`, clean `81.0`, Linf `46.7`, L2 `66.0`, L1 `48.9`.
- Validation protocol: RAMP internal final eval, `n=10000`, eps `(8/255, 0.5, 12)`, APGD CE+T via AutoAttack restricted to `['apgd-ce', 'apgd-t']`. This validates the code path but is not the controlled CARD-8b baseline.
- The cited RAMP `44.6 +/- 0.6` row stays literature-only and unchanged.

Controlled CARD-8b baselines from our independent harness:

| baseline row | n | attack | eps | clean | Linf | L2 | L1 | union | source |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| RAMP repro, locked protocol | 1000 | APGD CE+T | `(0.03, 0.5, 12)` | 81.2 | 48.4 | 65.8 | 49.7 | 47.2 | `results/eval_ramp_repro_eps003_apgd_n1000.json` |
| RAMP repro, RAMP-comparable | 1000 | APGD CE+T | `(8/255, 0.5, 12)` | 81.2 | 47.2 | 65.8 | 49.7 | 46.1 | `results/eval_ramp_repro_eps8255_apgd_n1000.json` |

Baseline rule: compare CARD-8b only to the matched local RAMP reproduction row. Never mix eps=0.03 and eps=8/255 rows in one delta.

## Strategic Correction

The 3.3pp gap between RAMP and our best in-house row is mostly Linf: RAMP has Linf `48.4` under our eps=0.03 harness row. That strength likely comes from RAMP's hard `L_max` term. A naive replacement with softer `T=0.25` could lower Linf and remove the exact mechanism that makes RAMP strong.

Therefore CARD-8b is now an augment-first ladder. The first pilot must preserve RAMP's hard `L_max`; replacement variants are only allowed after smoke evidence says Linf does not collapse.

## Exact Code Locus

File: `external/RAMP/RAMP.py`

- Hard selection is built at `RAMP.py:251-257`:
  `loss_best_t` and `loss_best_s` are stacked, `max_loss = loss_arr.max(dim=0)`, and `x_tr_best` is selected by argmax.
- RAMP forwards the selected adversarial image and forms the hard loss at `RAMP.py:304-308`:
  `outputs_best = model(x_tr_best)`, `loss_best = criterion(outputs_best, y_tr)`, then `loss = loss_best + loss_kl * args.lbd`.

Default-off implementation sketch:

```python
loss_hard = criterion(outputs_best, y_tr)
loss_pair = torch.stack([
    F.cross_entropy(outputs_t, y_tr, reduction="none"),
    F.cross_entropy(outputs_s, y_tr, reduction="none"),
], dim=0)
w = torch.softmax(loss_pair.detach() / args.bindaware_t, dim=0)
loss_soft = (w * loss_pair).sum(dim=0).mean()

if args.bindaware_variant == "augment":
    loss_best = loss_hard + args.bindaware_aux * loss_soft
elif args.bindaware_variant == "replace":
    loss_best = loss_soft
else:
    loss_best = loss_hard

loss = loss_best + loss_kl * args.lbd
```

This preserves the existing KL pairing and GP path. The soft weights are detached to match our swept objective in `src/robustdro/training/groupdro.py`.

## Three-Variant Ladder

Common invariants for all variants:

- RAMP upstream commit: `be4971f04cf8e70bd8255874a1ed2ab489cae682`.
- Model: `PreActResNet18(activation="softplus1")`, raw `[0,1]`, no normalization.
- Keep RAMP recipe: `--lr-max 0.05 --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 10 --eval_freq 10 --kl --max --gp --lbd 5`.
- Flags are default off. Baseline RAMP behavior must be identical when no binding-aware flag is passed.
- Seed 0 pilot first. No 3-seed commitment from card design alone.

### 8b-v1: AUGMENT, Pilot First

Purpose: preserve RAMP's hard `L_max` and Linf behavior, while adding our per-sample binding-aware auxiliary signal for stability / L1 support.

- Loss: `loss_best = loss_hard + bindaware_aux * loss_soft`.
- Start: `bindaware_aux=0.3`, `bindaware_t=0.25`.
- Proposed flags: `--bindaware_variant augment --bindaware_aux 0.3 --bindaware_t 0.25`.
- Run name: `RAMP_8b_v1_augment_aux0.3_T0.25_${SEED}`.
- This is the only first pilot. Do not run replacement variants before v1 smoke evidence.

### 8b-v2: REPLACE-Cold, Only If v1 Signals

Purpose: test whether near-hard softmax can replace hard argmax without giving away Linf.

- Loss: `loss_best = loss_soft`.
- Start: `bindaware_t=0.1`, no auxiliary because this is a replacement.
- Proposed flags: `--bindaware_variant replace --bindaware_t 0.1`.
- Run name: `RAMP_8b_v2_replace_T0.1_${SEED}`.
- Gate: only run if v1 smoke and pilot preserve Linf and show nonnegative union direction.

### 8b-v3: REPLACE-T0.25, Last / Highest Linf Risk

Purpose: test the original CARD-8b idea only after safer variants have evidence.

- Loss: `loss_best = loss_soft`.
- Start: `bindaware_t=0.25`.
- Proposed flags: `--bindaware_variant replace --bindaware_t 0.25`.
- Run name: `RAMP_8b_v3_replace_T0.25_${SEED}`.
- Gate: last resort. This is the highest Linf-loss risk because it is softer than v2 and removes hard `L_max`.

## Per-Variant Smoke Gate

Every variant must pass the same short gate before an 80-epoch commit:

1. One-epoch smoke first, no final eval. ETA `10-30min`. Goal: code path, checkpoint save, logging, no NaNs, no shape/device errors.
2. Short 2-3 epoch pilot next, with frequent cheap eval. Report Linf after epochs 2-3. If Linf collapses versus the RAMP trajectory, kill that variant before the 80-epoch run.
3. Only if Linf is stable and union direction is not clearly negative, approve one 80-epoch seed-0 pilot.
4. Only if seed 0 beats the matched Phase-A RAMP baselines by the preregistered rule, train seeds 1-2.

Do not commit roughly 40h of 3-seed runs from card design alone. Commit compute only on smoke evidence.

## Evaluation Plan

Primary compare-to-RAMP row:

- Our independent harness, `n=1000`, APGD CE+T, eps `(8/255, 0.5, 12)`.
- Compare against `results/eval_ramp_repro_eps8255_apgd_n1000.json` union `46.1`.

Locked-protocol table row:

- Our independent harness, `n=1000`, APGD CE+T, eps `(0.03, 0.5, 12)`.
- Compare against `results/eval_ramp_repro_eps003_apgd_n1000.json` union `47.2`.

If seed 0 signals:

- Run full AutoAttack `standard` under both matched eps rows before any final claim.
- Train seeds 1-2 only after the seed-0 APGD delta clears the preregistered threshold.

## Cost / ETA

- No CARD-8b smoke has been launched yet.
- Completed CARD-8 RAMP log gives the best local estimate: epoch times ranged from about `296s` to `634s`; 80 epochs are about `12-14h` depending on GPU sharing.
- RAMP final APGD CE+T `n=10000` eval took about `2.1h` from the three norm totals in `log_eval_final.txt`.
- Observed Phase-A `n=1000` APGD eval under our harness took minutes per eps row on the RTX 5070 Ti; L1 is the long tail.
- Per variant: 1-epoch smoke `10-30min`, 2-3 epoch Linf-gate pilot roughly `20-90min`, one full seed with final eval about `14-16h`.
- Seeds 1-2 are not pre-approved. They add roughly `28-32h` sequentially plus evaluation time only after seed-0 signal.

## Pre-Registered Read

Decision variables:

- `delta_8255 = union_8b_eps8255_apgd_n1000 - 46.1`.
- `delta_003 = union_8b_eps003_apgd_n1000 - 47.2`.

Clarification:

- `delta_8255` is compare-to-RAMP under RAMP-comparable eps.
- `delta_003` is our-table placement under the locked protocol.
- A meaningful beat requires both deltas to be positive, and the 3-seed mean must exceed `47.2` under eps=0.03 outside noise and survive full AutoAttack.

Seed-0 read:

- `delta_8255 > +0.5` and `delta_003 > +0.5`: signal. Consider seeds 1-2 for that variant.
- Both deltas positive but within `+0.5`: interesting but not enough for a 3-seed commitment unless smoke/trajectory evidence is especially clean; discuss before spending.
- Either delta within `+/-0.5` around zero: composable / no-harm. Report honestly; do not call it a beat.
- Either delta `< -0.5`, or early Linf collapse: hard `L_max` is load-bearing or the variant interacts badly with RAMP's KL + GP system. Kill or demote that variant.

Invariant: do not compare CARD-8b against cited RAMP `44.6` except as literature context. The controlled baselines are the two local RAMP reproduction JSONs above.
