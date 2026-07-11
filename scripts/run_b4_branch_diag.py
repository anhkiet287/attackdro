#!/usr/bin/env python
"""Authorized one-seed B4 small branching diagnostic.

This intentionally reconstructs a shared deterministic loader; it does not claim
historical replay fidelity. Outputs are restricted to results/branch_diag_b4_small_v1/.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robustdro.data import build_loaders, get_eval_split  # noqa: E402
from robustdro.eval.eval_union import evaluate_union  # noqa: E402
from robustdro.training.groupdro import GroupDROTrainer  # noqa: E402
from robustdro.utils.io import load_config, save_json  # noqa: E402


CONFIG = "configs/paper/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k.yaml"
SOURCE_RUN = Path("results/b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k/s0")
OUT = Path("results/branch_diag_b4_small_v1")
ANCHORS = {
    "epoch10": (SOURCE_RUN / "ckpt/ep010.pt", 10),
    "epoch40": (SOURCE_RUN / "ckpt/ep040.pt", 40),
    "epoch75": (SOURCE_RUN / "ckpt/val_best.pt", 75),
}
BRANCHES = [("linf", "linf"), ("linf_duplicate", "linf"), ("l2", "l2"), ("l1", "l1")]
NORMS = ("linf", "l2", "l1")


def restore_global_rng(rng: dict) -> None:
    torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([v.cpu().to(torch.uint8) for v in rng["cuda"]])


def loader_hash(loader) -> tuple[str, str]:
    """Hash first augmented batch and full label/order proxy for provenance.

    The preceding approved reconstruction checker established exact indexed order.
    Here the first tensor+label hash proves every launched branch uses that same build.
    """
    gen = loader.generator
    state = gen.get_state() if gen is not None else None
    it = iter(loader)
    x, y = next(it)
    h = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    if state is not None:
        gen.set_state(state)
    return h, hashlib.sha256(state.numpy().tobytes()).hexdigest() if state is not None else "none"


def make_cfg(base: dict, anchor: str, branch: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["device"] = "cuda"
    cfg["results_dir"] = str(OUT) + "/"
    cfg["run_name"] = f"{anchor}_{branch}_s0"
    cfg.setdefault("wandb", {})["mode"] = "disabled"
    # Continuation is the authorized balanced static cycle, not B4 adaptation.
    t = cfg["train"]
    t["allocation_mode"] = "static_cycle"
    t["static_cycle_order"] = list(NORMS)
    for key in list(t):
        if key.startswith("predictive_"):
            del t[key]
    return cfg


def load_anchor(trainer: GroupDROTrainer, ck: dict) -> None:
    trainer.model.load_state_dict(ck["model"])
    trainer.optimizer.load_state_dict(ck["optimizer"])
    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(ck["scheduler"])
    trainer.dro.load_state_dict(ck["dro"])
    restore_global_rng(ck["rng"])


def selected_update(trainer: GroupDROTrainer, batch, norm: str) -> None:
    x, y = (v.to(trainer.device) for v in batch)
    g = trainer.group_norms.index(norm)
    trainer.model.train()
    x_adv = trainer.attacks[g](trainer.model, x, y)
    trainer.model.train()
    loss = F.cross_entropy(trainer.model(x_adv), y)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainer.optimizer.step()


def run_branch(base: dict, anchor: str, checkpoint: Path, anchor_epoch: int,
               branch: str, first_norm: str, val_x, val_y) -> dict:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_stored = anchor_epoch - 1
    if int(ck.get("epoch", -999)) != expected_stored:
        raise RuntimeError(f"{checkpoint}: stored epoch {ck.get('epoch')} != {expected_stored}")
    required = {"model", "optimizer", "scheduler", "dro", "epoch", "rng"}
    if not required <= ck.keys():
        raise RuntimeError(f"{checkpoint}: missing {sorted(required - ck.keys())}")

    cfg = make_cfg(base, anchor, branch)
    restore_global_rng(ck["rng"])
    first_loaders = build_loaders(cfg, download=False)
    first_hash, generator_hash = loader_hash(first_loaders[0])
    # Rebuild after hashing so the actual first batch starts from the fixed seed.
    restore_global_rng(ck["rng"])
    first_loaders = build_loaders(cfg, download=False)
    trainer = GroupDROTrainer(cfg, download_data=False, loaders=first_loaders)
    load_anchor(trainer, ck)
    first_batch = next(iter(trainer.train_loader))
    selected_update(trainer, first_batch, first_norm)

    # A fresh identical build supplies the complete one-epoch continuation.
    restore_global_rng(ck["rng"])
    continuation, _ = build_loaders(cfg, download=False)
    trainer.train_loader = continuation
    continuation_first_hash, continuation_generator_hash = loader_hash(continuation)
    restore_global_rng(ck["rng"])
    continuation, _ = build_loaders(cfg, download=False)
    trainer.train_loader = continuation
    for i, batch in enumerate(trainer.train_loader):
        selected_update(trainer, batch, NORMS[i % len(NORMS)])
    continuation_batches = i + 1
    if trainer.scheduler is not None:
        trainer.scheduler.step()

    metrics = evaluate_union(
        trainer.model, val_x, val_y, cfg, norms=NORMS, version="apgd",
        device=trainer.device, bs=250, seed=0,
    )
    result = {
        "anchor": anchor,
        "anchor_epoch": anchor_epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_stored_epoch": int(ck["epoch"]),
        "branch": branch,
        "first_source": first_norm,
        "historical_replay_fidelity": "not_claimed",
        "loader": {
            "generator_seed": int(cfg.get("seed", 0)),
            "num_workers": int(cfg["dataset"].get("num_workers", 4)),
            "persistent_workers": False,
            "sampler": "RandomSampler",
            "first_batch_hash": first_hash,
            "generator_initial_state_hash": generator_hash,
            "continuation_first_batch_hash": continuation_first_hash,
            "continuation_generator_initial_state_hash": continuation_generator_hash,
        },
        "training": {
            "forced_first_updates": 1,
            "continuation_batches": continuation_batches,
            "optimizer_steps": 1 + continuation_batches,
            "scheduler_steps": 1,
            "cycle": list(NORMS),
            "attack_steps_per_update": 10,
            "attack_step_units": (1 + continuation_batches) * 10,
            "lr_after": trainer.optimizer.param_groups[0]["lr"],
        },
        "evaluation": {
            "split": "val_select",
            "n_examples": int(val_x.shape[0]),
            "version": "apgd",
            "steps": {"linf": 20, "l2": 20, "l1": 100},
            "restarts": 1,
            "clean": metrics["clean_acc"],
            "linf": metrics["per_norm_robust_acc"]["linf"],
            "l2": metrics["per_norm_robust_acc"]["l2"],
            "l1": metrics["per_norm_robust_acc"]["l1"],
            "worst_union": metrics["worst_union_acc"],
        },
    }
    save_json(result, str(OUT / anchor / f"{branch}.json"))
    del trainer
    torch.cuda.empty_cache()
    return result


def summarize(results: list[dict]) -> dict:
    summary = {}
    for anchor in ANCHORS:
        rr = {r["branch"]: r for r in results if r["anchor"] == anchor}
        u = {k: v["evaluation"]["worst_union"] for k, v in rr.items()}
        noise = abs(u["linf"] - u["linf_duplicate"])
        sources = {k: u[k] for k in ("linf", "l2", "l1")}
        spread = max(sources.values()) - min(sources.values())
        resolvable = spread > noise
        best = max(sources, key=sources.get) if resolvable else "NO_RESOLVABLE_DIFFERENCE"
        summary[anchor] = {
            "unions": u, "noise_floor": noise, "source_spread": spread,
            "resolvable": resolvable, "best_branch": best,
        }
    return summary


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"FAIL: output root already exists: {OUT}")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA unavailable")
    OUT.mkdir(parents=True)
    base = load_config(CONFIG)
    manifest = {
        "task_id": "B4-BRANCH-DIAGNOSTIC-EXECUTE-V1",
        "config": CONFIG,
        "anchors": {k: str(v[0]) for k, v in ANCHORS.items()},
        "branches": [b for b, _ in BRANCHES],
        "output_root": str(OUT),
        "historical_replay_fidelity": "not_claimed",
        "test_data_used": False,
    }
    save_json(manifest, str(OUT / "manifest.json"))
    val_x, val_y = get_eval_split(base, "val_select", n_examples=1000, download=False)
    results = []
    for anchor, (checkpoint, anchor_epoch) in ANCHORS.items():
        for branch, norm in BRANCHES:
            print(f"[branch] {anchor}/{branch} first={norm}", flush=True)
            results.append(run_branch(base, anchor, checkpoint, anchor_epoch,
                                      branch, norm, val_x, val_y))
            save_json({"results": results}, str(OUT / "progress.json"))
    save_json({"results": results, "summary": summarize(results)}, str(OUT / "results.json"))
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
