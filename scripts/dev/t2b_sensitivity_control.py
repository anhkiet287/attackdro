"""T2b — POST-HOC control for T2 (labelled exploratory, not pre-registered).

T2 found d(clean, adv) LARGER for CLAMP than for the control, in every norm, on both bases —
the opposite of H-geometry. Within each model, survivors sit closer to clean than failures, so
"closer = more robust" holds INSIDE a model but inverts ACROSS models. That pattern is what a
difference in the encoder's intrinsic input sensitivity looks like.

This control measures that sensitivity directly: how far does f move under a RANDOM perturbation
of the same norm and size? The SAME random directions are reused for every model (rng(0)), so the
comparison is paired. The adversarial displacement is then reported RELATIVE to it:

    ratio_p = d(clean, x_adv_p) / d(clean, x_rand_p)

ratio isolates "displacement specific to the adversarial direction" from "this encoder moves more
under any input change". If the T2 sign flips once normalised, H-geometry was masked by sensitivity;
if it survives, CLAMP genuinely separates adversarial from clean representations.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
import importlib.util
_s = importlib.util.spec_from_file_location("t2", os.path.join(ROOT, "scripts/dev/t2_geometry.py"))
t2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(t2)

OUT = t2.OUT; NORMS = t2.NORMS; EPS = t2.EPS


def rand_pert(x, norm, eps, gen):
    """Random perturbation at the SAME budget as the attack, clamped to the valid image box."""
    d = torch.randn(x.shape, generator=gen, device=x.device)
    if norm == "linf":
        d = eps * d.sign()
    else:
        flat = d.view(len(d), -1)
        if norm == "l2":
            flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-12) * eps
        else:                                   # l1: put the budget on a random sparse-ish direction
            flat = flat / (flat.abs().sum(1, keepdim=True) + 1e-12) * eps
        d = flat.view_as(x)
    return (x + d).clamp(0., 1.)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xs, ys, _ = t2.subset("1k", device)
    G = json.load(open(os.path.join(OUT, "geometry.json")))
    rows = {}
    for arm, desc, ck in t2.MODELS:
        m = t2.load(os.path.join(ROOT, ck), device)
        gen = torch.Generator(device=device).manual_seed(0)      # SAME directions for every model
        Zc, Zr = [], {n: [] for n in NORMS}
        with torch.no_grad():
            for i in range(0, len(xs), 128):
                x = xs[i:i + 128].to(device)
                Zc.append(m.features(x).float().cpu())
                for n in NORMS:
                    Zr[n].append(m.features(rand_pert(x, n, EPS[n], gen)).float().cpu())
        Zc = torch.cat(Zc).numpy()
        rows[arm] = {n: float(t2.dist(Zc, torch.cat(Zr[n]).numpy(), "cos").mean()) for n in NORMS}
        del m; torch.cuda.empty_cache()

    print("=" * 104)
    print("T2b  SENSITIVITY CONTROL (post-hoc, exploratory) — cosine distance in f-space")
    print("=" * 104)
    print(f"{'model':<10} | {'d_rand ℓ∞':>10}{'d_rand ℓ2':>10}{'d_rand ℓ1':>10} | "
          f"{'d_adv ℓ∞':>9}{'d_adv ℓ2':>9}{'d_adv ℓ1':>9} | {'ratio ℓ∞':>9}{'ratio ℓ2':>9}{'ratio ℓ1':>9}")
    print("-" * 104)
    ratio = {}
    for arm, _, _ in t2.MODELS:
        adv = {n: G["models"][arm]["all"]["cos"][f"clean-{n}"] for n in NORMS}
        ratio[arm] = {n: adv[n] / rows[arm][n] for n in NORMS}
        print(f"{arm:<10} | " + "".join(f"{rows[arm][n]:>10.4f}" for n in NORMS) + " | "
              + "".join(f"{adv[n]:>9.4f}" for n in NORMS) + " | "
              + "".join(f"{ratio[arm][n]:>9.3f}" for n in NORMS))
    print("-" * 104)
    print("\nNORMALISED relative movement  Δratio = ratio(CLAMP) − ratio(control)   (negative = adversarial")
    print("displacement shrinks once the encoder's intrinsic sensitivity is divided out)\n")
    print(f"  {'pair':<26}{'Δratio ℓ∞':>12}{'Δratio ℓ2':>12}{'Δratio ℓ1':>12}   {'raw Δd sign':>14}")
    for A, B, lbl in t2.PAIRS:
        raw = t2.NORMS
        print(f"  {lbl:<26}" + "".join(f"{ratio[A][n] - ratio[B][n]:>+12.3f}" for n in NORMS)
              + f"{'all +  (T2)':>17}")
    print("\n  sensitivity ratio of the encoders (how much more f moves under the SAME random input change):")
    for A, B, lbl in t2.PAIRS:
        print(f"    {lbl:<26}" + "".join(f"  {n}: {rows[A][n]/rows[B][n]:.2f}x" for n in NORMS))
    json.dump({"d_random_cos": rows, "ratio_adv_over_random": ratio},
              open(os.path.join(OUT, "sensitivity_control.json"), "w"), indent=2)
    print(f"\n  saved -> results/analysis/T2_geometry/sensitivity_control.json")


if __name__ == "__main__":
    main()
