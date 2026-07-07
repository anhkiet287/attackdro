"""Workspace health check for AttackDRO.

Run this every time you sit back down at the machine. It verifies — one item at
a time — that the environment, the GPU, and the training pipeline all work, so if
something is broken you know *exactly* which layer to fix.

    python scripts/healthcheck.py          # full check (incl. a tiny GPU train)
    python scripts/healthcheck.py --quick  # skip the training smoke test

Each line is PASS / FAIL / WARN. On any FAIL you get a hint pointing at the
likely cause (driver, cu128 build, config, etc.). Exit code is 0 only if every
critical check passes — handy for scripting / CI.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

# Repo root = parent of scripts/. Run from anywhere.
ROOT = Path(__file__).resolve().parents[2]

# --- pretty output -----------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


GREEN, RED, YELLOW, DIM, BOLD = "32", "31", "33", "2", "1"

results: list[tuple[str, bool]] = []  # (name, is_critical_failure)


def check(name: str, ok: bool, detail: str = "", hint: str = "", *, warn: bool = False) -> bool:
    """Record and print one check. `warn=True` => non-critical (won't fail the run)."""
    if ok:
        tag = _c(GREEN, "PASS")
    elif warn:
        tag = _c(YELLOW, "WARN")
    else:
        tag = _c(RED, "FAIL")
    line = f"  [{tag}] {name}"
    if detail:
        line += _c(DIM, f"  — {detail}")
    print(line)
    if not ok and hint:
        print(_c(DIM, f"         ↳ check: {hint}"))
    if not ok and not warn:
        results.append((name, True))
    else:
        results.append((name, False))
    return ok


def section(title: str) -> None:
    print()
    print(_c(BOLD, title))


# --- the checks --------------------------------------------------------------
def check_python() -> None:
    section("1. Python & virtualenv")
    v = sys.version_info
    check(
        "Python >= 3.10",
        v >= (3, 10),
        detail=f"{v.major}.{v.minor}.{v.micro}",
        hint="use the project .venv (see SETUP.md)",
    )
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    venv_path = Path(sys.prefix).resolve()
    is_project_venv = str(venv_path).startswith(str(ROOT))
    check(
        "Running inside project .venv",
        in_venv and is_project_venv,
        detail=str(venv_path),
        hint="run: source .venv/bin/activate  (or use .venv/bin/python)",
        warn=not in_venv or not is_project_venv,
    )


def check_packages() -> None:
    section("2. Core dependencies")
    for mod in ("numpy", "yaml", "tqdm", "matplotlib"):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            check(f"import {mod}", True, detail=ver)
        except Exception as e:  # noqa: BLE001
            check(
                f"import {mod}",
                False,
                detail=str(e),
                hint="pip install -r requirements.txt",
            )


def check_torch() -> object | None:
    section("3. PyTorch")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        check(
            "import torch",
            False,
            detail=str(e),
            hint="install the cu128 build — see SETUP.md (do NOT pip install plain torch)",
        )
        return None

    check("import torch", True, detail=torch.__version__)
    is_cu128 = "cu128" in torch.__version__ or (torch.version.cuda or "").startswith("12.8")
    check(
        "cu128 / CUDA 12.8 build",
        is_cu128,
        detail=f"torch.version.cuda={torch.version.cuda}",
        hint="RTX 5070 Ti (Blackwell) needs the cu128 wheel — SETUP.md",
        warn=not is_cu128,
    )
    return torch


def check_gpu(torch) -> None:
    section("4. GPU / CUDA")
    if torch is None:
        check("CUDA available", False, hint="torch failed to import (section 3)")
        return

    avail = torch.cuda.is_available()
    if not check(
        "CUDA available",
        avail,
        hint="NVIDIA driver on Windows host + cu128 torch in WSL — SETUP.md",
    ):
        return

    check("GPU device count >= 1", torch.cuda.device_count() >= 1,
          detail=f"{torch.cuda.device_count()} device(s)")
    name = torch.cuda.get_device_name(0)
    check("GPU detected", bool(name), detail=name)
    cap = torch.cuda.get_device_capability(0)
    check(
        "Compute capability (12, 0) — Blackwell/sm_120",
        cap == (12, 0),
        detail=str(cap),
        hint="wrong GPU or wrong torch build if not (12, 0)",
        warn=cap != (12, 0),
    )

    # Real kernel on the GPU — proves sm_120 kernels actually run, not just init.
    try:
        t0 = time.time()
        x = torch.randn(4096, 4096, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        dt = time.time() - t0
        check("Matmul kernel on GPU", True, detail=f"4096² @ ok, {dt*1000:.0f} ms")
    except Exception as e:  # noqa: BLE001
        check(
            "Matmul kernel on GPU",
            False,
            detail=str(e),
            hint="kernel launch failed — likely torch build vs GPU arch mismatch",
        )


def check_config() -> None:
    section("5. Config & repo layout")
    cfg_path = ROOT / "configs" / "default.yaml"
    if not check("configs/default.yaml exists", cfg_path.exists(),
                 hint=str(cfg_path)):
        return
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text())
        required = {"seed", "device", "input_dim", "hidden_dim", "output_dim",
                    "num_samples", "epochs", "batch_size", "lr"}
        missing = required - set(cfg)
        check(
            "config has all required keys",
            not missing,
            detail="ok" if not missing else f"missing: {sorted(missing)}",
            hint="restore keys in configs/default.yaml",
        )
    except Exception as e:  # noqa: BLE001
        check("config parses as YAML", False, detail=str(e),
              hint="fix YAML syntax in configs/default.yaml")

    check("src/train.py exists", (ROOT / "src" / "train.py").exists())

    # Job-queue folder structure (file-based dispatch).
    job_dirs = ["queue", "running", "done", "failed", "logs"]
    missing_dirs = [d for d in job_dirs if not (ROOT / "jobs" / d).is_dir()]
    check(
        "jobs/ queue folders present",
        not missing_dirs,
        detail="ok" if not missing_dirs else f"missing: {missing_dirs}",
        hint="recreate jobs/<name>/ folders (see jobs/README.md)",
        warn=bool(missing_dirs),
    )


def check_training(quick: bool) -> None:
    section("6. Training pipeline (end-to-end smoke test)")
    if quick:
        check("train smoke test", True, detail="skipped (--quick)", warn=True)
        return
    try:
        import torch
        import yaml

        sys.path.insert(0, str(ROOT))
        from src.train import MLP, make_data  # type: ignore
        from torch.utils.data import DataLoader

        cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
        cfg["num_samples"] = 4096  # keep it fast
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loader = DataLoader(make_data(cfg), batch_size=cfg["batch_size"],
                            shuffle=True, drop_last=True)
        model = MLP(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        loss_fn = torch.nn.CrossEntropyLoss()

        t0 = time.time()
        last = None
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            last = loss.item()
        dt = time.time() - t0
        ok = last is not None and last == last  # not NaN
        check(
            f"1-epoch train on {device}",
            ok,
            detail=f"loss={last:.4f}, {dt:.2f}s" if last is not None else "no batches",
            hint="model/data/optim path is broken — see src/train.py",
        )
    except Exception:  # noqa: BLE001
        check("train smoke test", False, detail="exception",
              hint="traceback below")
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description="AttackDRO workspace health check")
    parser.add_argument("--quick", action="store_true",
                        help="skip the end-to-end training smoke test")
    args = parser.parse_args()

    os.chdir(ROOT)
    print(_c(BOLD, f"AttackDRO health check  ·  {ROOT}"))

    check_python()
    check_packages()
    torch = check_torch()
    check_gpu(torch)
    check_config()
    check_training(args.quick)

    # --- summary -------------------------------------------------------------
    failures = [n for n, crit in results if crit]
    print()
    if not failures:
        print(_c(GREEN, _c(BOLD, "✓ ALL CHECKS PASSED — workspace is healthy.")))
        return 0
    print(_c(RED, _c(BOLD, f"✗ {len(failures)} critical check(s) FAILED:")))
    for n in failures:
        print(_c(RED, f"    - {n}"))
    print(_c(DIM, "  Fix the topmost failing section first; later ones often depend on it."))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
