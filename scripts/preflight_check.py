#!/usr/bin/env python
"""Standalone deterministic pre-run gatekeeper.

This is a READ-ONLY preflight. It does not train, evaluate, write result JSONs,
kill processes, or edit any config. It only inspects the repo/config/GPU/data
state and reports whether it is safe to launch a run.

It is intentionally NOT wired into scripts/train.py or scripts/evaluate.py yet.
Wiring happens only after the active reactive baseline run completes.

Usage
-----
    .venv/bin/python scripts/preflight_check.py \
        --config configs/paper/reactive_softT_full10_ramp80_apgd_8255_t49k_v1k.yaml \
        --target-result-dir results/reactive_softT_full10/s0 \
        --gpu-index 0 \
        --min-free-vram-gb 13

Exit code is 0 only if all required checks pass, nonzero otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
PROJECT_STATE = os.path.join(REPO_ROOT, "docs", "STATE.md")

WANDB_PROJECT_EXPECTED = "attackdro-union"

# Expected canonical split sizes under the locked train49k/val1k protocol.
EXPECTED_SPLIT = {
    "train_core_n": 49000,
    "val_select_n": 1000,
    "overlap_n": 0,
    "test_monitor_n": 1000,
    "test_final_n": 10000,
}


# --------------------------------------------------------------------------- #
# Result accumulator
# --------------------------------------------------------------------------- #
class Report:
    """Collects per-check status lines and hard blockers."""

    def __init__(self):
        self.lines: dict[str, str] = {}
        self.blockers: list[str] = []
        self.notes: list[str] = []
        self.required_ok = True

    def set(self, key: str, value: str):
        self.lines[key] = value

    def require(self, ok: bool, key: str, value: str, blocker: str | None = None):
        """Record a required check. A failure flips overall status and records a blocker."""
        self.set(key, value)
        if not ok:
            self.required_ok = False
            if blocker:
                self.blockers.append(blocker)

    def note(self, msg: str):
        self.notes.append(msg)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def check_project_state(rep: Report):
    ok = os.path.isfile(PROJECT_STATE)
    rep.require(
        ok, "STATE", "PASS" if ok else "FAIL: missing docs/STATE.md",
        blocker=None if ok else "docs/STATE.md not found",
    )


def check_config(rep: Report, config_path: str):
    """Config exists (check 2) and YAML parses / resolves _base_ (check 3)."""
    exists = os.path.isfile(config_path)
    if not exists:
        rep.require(False, "CONFIG_EXISTS", f"FAIL: {config_path} not found",
                    blocker=f"config path does not exist: {config_path}")
        rep.require(False, "YAML_PARSE", "FAIL: config missing, cannot parse",
                    blocker="config could not be parsed (missing file)")
        return None
    rep.set("CONFIG_EXISTS", "PASS")

    cfg = None
    try:
        # Prefer the project's own loader so _base_ inheritance is resolved exactly
        # as the trainer would see it. Fall back to a plain yaml.safe_load.
        try:
            if SRC_DIR not in sys.path:
                sys.path.insert(0, SRC_DIR)
            from robustdro.utils.io import load_config  # noqa: E402
            cfg = load_config(config_path)
            rep.require(True, "YAML_PARSE", "PASS (resolved _base_ via robustdro loader)")
        except Exception as loader_exc:  # noqa: BLE001
            import yaml
            with open(config_path) as fh:
                cfg = yaml.safe_load(fh)
            rep.require(True, "YAML_PARSE",
                        f"PASS (plain yaml; robustdro loader unavailable: {loader_exc})")
    except Exception as exc:  # noqa: BLE001
        rep.require(False, "YAML_PARSE", f"FAIL: {exc}",
                    blocker=f"config YAML did not parse: {exc}")
        return None
    return cfg


def check_git_diff(rep: Report):
    """`git diff --check` must be clean (no conflict markers / whitespace errors)."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--check"], cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
    except Exception as exc:  # noqa: BLE001
        rep.require(False, "GIT_DIFF_CHECK", f"FAIL: could not run git ({exc})",
                    blocker=f"git diff --check could not run: {exc}")
        return
    if proc.returncode == 0:
        rep.require(True, "GIT_DIFF_CHECK", "PASS")
    else:
        detail = (proc.stdout or proc.stderr or "").strip().splitlines()
        first = detail[0] if detail else "unknown issue"
        rep.require(False, "GIT_DIFF_CHECK", f"FAIL: {first}",
                    blocker="git diff --check reported conflict/whitespace errors")


def check_target_dir(rep: Report, target_dir: str):
    """Target result dir must NOT already exist (protects active/completed results)."""
    exists = os.path.exists(target_dir)
    if exists:
        rep.require(False, "TARGET_EXISTS", f"FAIL: {target_dir} already exists",
                    blocker=f"target result dir already exists (would overwrite): {target_dir}")
    else:
        rep.require(True, "TARGET_EXISTS", "PASS (does not exist)")


def _gpu_via_pynvml(gpu_index: int):
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gb = mem.free / (1024 ** 3)
        return name, free_gb
    finally:
        pynvml.nvmlShutdown()


def _gpu_via_nvidia_smi(gpu_index: int):
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free",
         "--format=csv,noheader,nounits", "-i", str(gpu_index)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "nvidia-smi failed").strip())
    line = proc.stdout.strip().splitlines()[0]
    name, free_mib = [p.strip() for p in line.split(",")]
    free_gb = float(free_mib) / 1024.0  # MiB -> GiB
    return name, free_gb


def check_gpu(rep: Report, gpu_index: int, min_free_gb: float):
    name = None
    free_gb = None
    source = None
    err = None
    for fn, tag in ((_gpu_via_pynvml, "pynvml"), (_gpu_via_nvidia_smi, "nvidia-smi")):
        try:
            name, free_gb = fn(gpu_index)
            source = tag
            break
        except Exception as exc:  # noqa: BLE001
            err = f"{tag}: {exc}"
            continue

    if free_gb is None:
        rep.set("GPU_NAME", "UNKNOWN")
        rep.set("GPU_FREE_VRAM_GB", "UNKNOWN")
        rep.require(False, "GPU_CHECK", f"FAIL: could not query GPU ({err})",
                    blocker=f"GPU query failed on index {gpu_index}: {err}")
        return

    rep.set("GPU_NAME", f"{name} (via {source})")
    rep.set("GPU_FREE_VRAM_GB", f"{free_gb:.2f}")
    if free_gb + 1e-6 < min_free_gb:
        rep.require(
            False, "GPU_CHECK",
            f"FAIL: free {free_gb:.2f} GB < required {min_free_gb:.2f} GB",
            blocker=f"insufficient free VRAM on GPU {gpu_index}: "
                    f"{free_gb:.2f} GB < {min_free_gb:.2f} GB",
        )
    else:
        rep.require(True, "GPU_CHECK",
                    f"PASS (free {free_gb:.2f} GB >= {min_free_gb:.2f} GB)")


def check_active_processes(rep: Report):
    """Advisory: detect obvious active train/eval python processes. Never kills."""
    markers = ("scripts/train.py", "scripts/evaluate.py",
               "scripts/post_train_develop_eval.py")
    self_pid = os.getpid()
    try:
        proc = subprocess.run(["ps", "-eo", "pid,args"],
                              capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        rep.set("ACTIVE_PROCESS_CHECK", f"NOT_AVAILABLE: ps failed ({exc})")
        return

    hits = []
    for raw in proc.stdout.splitlines()[1:]:
        raw = raw.strip()
        if not raw:
            continue
        pid_str, _, cmd = raw.partition(" ")
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if "preflight_check.py" in cmd:
            continue
        if any(m in cmd for m in markers):
            hits.append((pid, cmd.strip()))

    if not hits:
        rep.set("ACTIVE_PROCESS_CHECK", "PASS (no active train/eval processes)")
    else:
        listing = "; ".join(f"pid={pid} {cmd[:80]}" for pid, cmd in hits)
        # Advisory only: does NOT fail the preflight, and does NOT kill anything.
        rep.set("ACTIVE_PROCESS_CHECK",
                f"WARN: {len(hits)} active train/eval process(es) detected: {listing}")
        rep.note(f"Active train/eval process(es) running (not killed): {listing}")


def _compute_split_summary(cfg: dict):
    """Deterministic split-size summary derived from the project's own data
    contract. Read-only: constructs CIFAR-10 index ranges (download=False), never
    trains. Returns (summary_dict, error_or_None)."""
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        import torchvision  # noqa: F401
        from torchvision.datasets import CIFAR10
    except Exception as exc:  # noqa: BLE001
        return None, f"torchvision unavailable: {exc}"

    dcfg = cfg.get("dataset", {}) or {}
    root = dcfg.get("root", "data/")
    if not os.path.isabs(root):
        root = os.path.join(REPO_ROOT, root)
    try:
        train_ds = CIFAR10(root=root, train=True, download=False)
        test_ds = CIFAR10(root=root, train=False, download=False)
    except Exception as exc:  # noqa: BLE001
        return None, f"CIFAR-10 not materialized at {root} (download=False): {exc}"

    n_train = len(train_ds)
    n_test = len(test_ds)
    holdout = int(dcfg.get("val_holdout", 0) or 0)

    # Mirror build_loaders()/get_train_holdout(): train pool is the leading
    # range, val_select is the trailing `holdout` tail -> disjoint by construction.
    train_core_idx = set(range(0, n_train - holdout))
    val_select_idx = set(range(n_train - holdout, n_train))
    overlap = train_core_idx & val_select_idx

    tcfg = cfg.get("train", {}) or {}
    test_monitor_n = int(tcfg.get("test_monitor_n_examples", 1000))

    summary = {
        "train_core_n": n_train - holdout,
        "val_select_n": holdout,
        "overlap_n": len(overlap),
        "test_monitor_n": test_monitor_n,
        "test_final_n": n_test,
    }
    return summary, None


def check_split(rep: Report, cfg: dict | None, allow_missing: bool):
    if cfg is None:
        rep.set("SPLIT_CHECK", "NOT_AVAILABLE: config unavailable")
        if not allow_missing:
            rep.required_ok = False
            rep.blockers.append("split check not available (config unavailable) and "
                                "--allow-missing-split-check not passed")
        return

    summary, err = _compute_split_summary(cfg)
    if summary is None:
        rep.set("SPLIT_CHECK", f"NOT_AVAILABLE: {err}")
        if allow_missing:
            rep.note(f"split check skipped by --allow-missing-split-check ({err})")
        else:
            rep.required_ok = False
            rep.blockers.append(f"split check not available: {err} "
                                "(pass --allow-missing-split-check to bypass)")
        return

    mismatches = [f"{k}={summary[k]} (expected {v})"
                  for k, v in EXPECTED_SPLIT.items() if summary[k] != v]
    got = ", ".join(f"{k}={summary[k]}" for k in EXPECTED_SPLIT)
    if mismatches:
        rep.require(False, "SPLIT_CHECK", f"FAIL: {'; '.join(mismatches)} | got {got}",
                    blocker=f"split sizes mismatch: {'; '.join(mismatches)}")
    else:
        rep.require(True, "SPLIT_CHECK", f"PASS ({got})")


def check_wandb(rep: Report, cfg: dict | None, target_dir: str):
    """W&B schema/config check. Non-fatal except project must be attackdro-union."""
    if cfg is None:
        rep.set("WANDB_CHECK", "NOT_AVAILABLE: config unavailable")
        return
    wcfg = cfg.get("wandb", {}) or {}
    parts = []

    project = wcfg.get("project")
    if project == WANDB_PROJECT_EXPECTED:
        parts.append(f"project={project} OK")
    else:
        parts.append(f"project={project!r} (expected {WANDB_PROJECT_EXPECTED!r})")
        rep.required_ok = False
        rep.blockers.append(f"wandb.project is {project!r}, expected {WANDB_PROJECT_EXPECTED!r}")

    # Group: present in config, or derivable (falls back to run grouping).
    group = wcfg.get("group")
    if group:
        parts.append(f"group={group}")
    else:
        parts.append("group=<null; derivable at runtime>")

    # Run name should match the target result folder (parent of s<seed>).
    run_name = cfg.get("run_name")
    folder_run = os.path.basename(os.path.dirname(os.path.normpath(target_dir))) if target_dir else None
    base_run = _strip_seed_suffix(run_name) if run_name else None
    if folder_run and base_run:
        if base_run == folder_run:
            parts.append(f"run_name~folder OK ({folder_run})")
        else:
            parts.append(f"run_name={base_run!r} != folder {folder_run!r}")
            rep.note(f"wandb run_name {base_run!r} does not match target folder {folder_run!r}")

    fatal = project != WANDB_PROJECT_EXPECTED
    prefix = "FAIL" if fatal else "PASS"
    rep.set("WANDB_CHECK", f"{prefix}: " + "; ".join(parts))


def _strip_seed_suffix(run_name: str) -> str:
    import re
    n = re.sub(r"_s\d+$", "", run_name or "run")
    return re.sub(r"_smoke$", "", n) or "run"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone deterministic pre-run gatekeeper (read-only).",
    )
    p.add_argument("--config", required=True, help="Path to the run config YAML.")
    p.add_argument("--target-result-dir", required=True,
                   help="Per-run result dir that must NOT yet exist, e.g. results/run/s0.")
    p.add_argument("--gpu-index", type=int, default=0, help="GPU index to inspect.")
    p.add_argument("--min-free-vram-gb", type=float, default=13.0,
                   help="Fail if free VRAM on --gpu-index is below this (GiB).")
    p.add_argument("--allow-missing-split-check", action="store_true",
                   help="Do not fail if the split dry-run helper / data is unavailable.")
    return p.parse_args()


def main():
    args = parse_args()
    rep = Report()

    rep.set("CONFIG", args.config)
    rep.set("TARGET_RESULT_DIR", args.target_result_dir)

    check_project_state(rep)
    cfg = check_config(rep, args.config)
    check_git_diff(rep)
    check_target_dir(rep, args.target_result_dir)
    check_gpu(rep, args.gpu_index, args.min_free_vram_gb)
    check_active_processes(rep)
    check_split(rep, cfg, args.allow_missing_split_check)
    check_wandb(rep, cfg, args.target_result_dir)

    status = "PASS" if rep.required_ok else "FAIL"
    order = [
        "CONFIG", "TARGET_RESULT_DIR", "STATE", "CONFIG_EXISTS", "YAML_PARSE",
        "GPU_NAME", "GPU_FREE_VRAM_GB", "GPU_CHECK", "GIT_DIFF_CHECK", "TARGET_EXISTS",
        "ACTIVE_PROCESS_CHECK", "SPLIT_CHECK", "WANDB_CHECK",
    ]

    print(f"PREFLIGHT_STATUS: {status}")
    for key in order:
        if key in rep.lines:
            print(f"{key}: {rep.lines[key]}")
    if rep.notes:
        print("NOTES:")
        for n in rep.notes:
            print(f"  - {n}")
    print("BLOCKERS:")
    if rep.blockers:
        for b in rep.blockers:
            print(f"  - {b}")
    else:
        print("  none")

    return 0 if rep.required_ok else 1


if __name__ == "__main__":
    sys.exit(main())
