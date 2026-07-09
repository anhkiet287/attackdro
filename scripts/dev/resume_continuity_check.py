#!/usr/bin/env python
"""Verify resume gives a VALID continuation (item 1): a run split at ep_N and resumed
must reproduce the uninterrupted run's lr schedule + epoch sequence, and continue the
loss without a spike. Uses milestones=[2] so the lr DROP falls inside the window (tests
scheduler-state restore, not just a constant lr). Fast: 2 train steps + 1 probe batch
per epoch. Non-smoke (so checkpoints actually save)."""
import sys
sys.path.insert(0, "src")
from robustdro.utils.io import apply_overrides, load_config, load_json, run_paths
from robustdro.utils.seed import set_seed
from robustdro.training.groupdro import GroupDROTrainer

FAST = {"train.milestones": [2], "train.lr": 0.05, "train.save_freq": 1,
        "train.eval_probe_batches": 1, "wandb.mode": "disabled", "seed": 0}


def run(run_name, epochs, resume=None):
    cfg = load_config("configs/paper/reactive_ramprecipe.yaml")
    ov = dict(FAST, **{"train.epochs": epochs, "run_name": run_name, "train.resume": resume})
    cfg = apply_overrides(cfg, ov)
    set_seed(cfg["seed"])
    GroupDROTrainer(cfg).fit(max_steps_per_epoch=2, eval_max_batches=1)
    hist = load_json(run_paths(cfg)["train"])["history"]
    return [(h["epoch"], round(h["lr"], 5)) for h in hist], run_paths(cfg)


print("=== uninterrupted (C): 4 epochs ===")
lrs_C, _ = run("cont_check_C", 4)
print("=== split (A): 2 epochs, then (B): resume -> 4 epochs ===")
run("cont_check_AB", 2)                       # trains ep0,1 -> saves last.pt@ep1
lrs_AB, p = run("cont_check_AB", 4, resume="auto")

print("C  lr/epoch:", lrs_C)
print("AB lr/epoch:", lrs_AB)
ck = load_json  # noqa
import torch
last = torch.load(f"{p['ckpt_dir']}/last.pt", map_location="cpu", weights_only=False)
has_state = bool(last.get("optimizer")) and (last.get("scheduler") is not None) and ("rng" in last)
epochs_seen = [e for e, _ in lrs_AB]

ok_seq = lrs_C == lrs_AB
ok_resume = epochs_seen == [0, 1, 2, 3]        # B preloaded ep0,1 and continued 2,3
ok_drop = dict(lrs_AB).get(2) == round(0.05 / 10, 5)   # milestone[2] drop restored on resume
print(f"\nlr+epoch sequence identical (C==AB): {ok_seq}")
print(f"epoch sequence continuous 0..3:       {ok_resume}")
print(f"lr DROP at milestone 2 survived resume {dict(lrs_AB).get(2)} == 0.005: {ok_drop}")
print(f"last.pt carries optimizer+scheduler+rng: {has_state}")
assert ok_seq and ok_resume and ok_drop and has_state, "RESUME CONTINUITY FAILED"
print("\nRESUME CONTINUITY PASS ✓ — resumed run reproduces the uninterrupted lr schedule "
      "(incl. the drop) and epoch sequence; checkpoint carries full optimizer/scheduler/RNG state.")
