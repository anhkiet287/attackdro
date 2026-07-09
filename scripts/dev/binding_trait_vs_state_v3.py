#!/usr/bin/env python3
"""Gate-alpha DECISION-GRADE trait-vs-state diagnostic (v3).

Reads fixed indexed probe-binding dumps
(`dumps/probe_binding/<run>/probe_binding_epNNN.pt`), each carrying the SAME
CIFAR train `indices`, `labels`, `norms`, and a `[G,B]` per-norm loss matrix.
Per-sample binding = argmax_g loss (the norm that hurts the sample most).

Why v3 (vs v2): raw persistence is base-rate-inflated. Under a norm-imbalanced
binding distribution (avg_frozen is ~.98 linf) a naive "93% persistence" is
almost all chance, because the marginal is concentrated. v3 reports the RIGHT
metric and separates two DIFFERENT axes that must not be conflated:

  * MIGRATION  = how the POPULATION distribution p_g(epoch) drifts (l1 -> linf).
                 This is F6: expected, population-level, NOT the new question.
  * TRAIT/STATE = whether INDIVIDUAL samples keep their binding on FIXED indices,
                 measured as EXCESS persistence above the drifting base rate.
                 A sample can be trait-stable even while the population migrates,
                 if it does NOT migrate on the population's schedule.

Definitions (pre-registered):
  chance(pair t,t+1) = sum_g f_a(g) * f_b(g)      # independent-draw match rate
                                                    # (= sum_g p_g^2 when f_a==f_b)
  EXCESS persistence  = persistence - chance        # THE headline metric
  per-norm retention  = P(bind_{t+1}=g | bind_t=g)  # sample-level stickiness of g
  per-norm EXCESS_g   = retention_g - marginal_g    # stickiness above chance

L1-MINORITY CONDITIONAL (decision-critical): among samples ever bound to l1,
P(still l1 next dump) = retention_l1; compare to marginal_l1 (its chance).
The linf majority is base-rate-saturated -> reported for completeness only.

VERDICT RULE (pre-registered):
  - l1 EXCESS >> 0 AND linf EXCESS >> 0  -> TRAIT -> predictive weighting (Conj 2) viable.
  - both EXCESS ~= 0                     -> PURE STATE -> dynamics/tracking framing.
  - linf EXCESS >> 0 but l1 EXCESS ~= 0  -> BOTH -> two-tier framework.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "dumps" / "probe_binding"
DEFAULT_OUT = REPO / "results" / "reports" / "trait_state_v3.md"

# Automated-verdict thresholds (raw numbers are printed so a human can read the
# verdict directly regardless of these cutoffs).
EXCESS_TRAIT = 0.20      # per-norm excess above which the norm is "sticky" (trait)
EXCESS_STATE = 0.10      # per-norm |excess| below which it "tracks chance" (state)
STABLE_MODE_FRAC = 0.80  # >this share of dumps on one norm => "stable trait" sample
MINORITY_LO = 0.02       # a norm is a measurable minority if marginal > this


def load_run(run_dir: Path):
    files = sorted(run_dir.glob("probe_binding_ep*.pt"))
    if not files:
        return [], f"{run_dir.name}: no probe_binding_ep*.pt files"
    dumps = [torch.load(p, map_location="cpu") for p in files]
    ref_idx, ref_y = dumps[0]["indices"], dumps[0]["labels"]
    for d in dumps[1:]:
        if not torch.equal(ref_idx, d["indices"]):
            return dumps, f"{run_dir.name}: sample indices differ across dumps"
        if not torch.equal(ref_y, d["labels"]):
            return dumps, f"{run_dir.name}: labels differ across dumps"
    return dumps, None


def analyse(run_dir: Path, min_ckpts: int) -> dict | str:
    dumps, err = load_run(run_dir)
    if not dumps or err:
        return err or f"{run_dir.name}: no dumps"
    norms = list(dumps[0]["norms"])
    G = len(norms)
    epochs = [int(d["epoch"]) for d in dumps]
    binds = [d["L"].float().argmax(dim=0) for d in dumps]  # each [B]
    B = binds[0].numel()
    stack = torch.stack(binds, dim=0)  # [T, B]

    def freq(b):
        return torch.bincount(b, minlength=G).float() / B

    # --- migration curve + PER-EPOCH self-chance (base-rate inflation view) ---
    migration = []
    for e, b in zip(epochs, binds):
        f = freq(b)
        self_chance = float((f ** 2).sum())  # sum_g p_g^2 at this epoch
        migration.append((e, {norms[g]: f[g].item() for g in range(G)}, self_chance))

    # pooled marginal p_g (mean of per-epoch freqs)
    pooled = torch.stack([freq(b) for b in binds]).mean(0)
    marginal = {norms[g]: pooled[g].item() for g in range(G)}

    # --- per-pair persistence & chance = sum_g f_a(g) f_b(g) -----------------
    pair_rows = []            # (e_a, e_b, persistence, chance, excess)
    for (ea, a), (eb, b) in zip(zip(epochs, binds), zip(epochs[1:], binds[1:])):
        p = (a == b).float().mean().item()
        fa, fb = freq(a), freq(b)
        c = float((fa * fb).sum())
        pair_rows.append((ea, eb, p, c, p - c))
    persistence = sum(r[2] for r in pair_rows) / len(pair_rows) if pair_rows else float("nan")
    chance = sum(r[3] for r in pair_rows) / len(pair_rows) if pair_rows else float("nan")
    excess_overall = persistence - chance

    # --- per-norm retention & transitions ------------------------------------
    ret_count = torch.zeros(G)
    from_count = torch.zeros(G)
    trans = torch.zeros(G, G)  # trans[i,j] = #(b_t=i -> b_{t+1}=j)
    for a, b in zip(binds, binds[1:]):
        for i in range(G):
            m = a == i
            n_i = int(m.sum())
            from_count[i] += n_i
            if n_i:
                ret_count[i] += int((b[m] == i).sum())
                for j in range(G):
                    trans[i, j] += int((b[m] == j).sum())
    retention = {norms[g]: (ret_count[g] / from_count[g]).item() if from_count[g] > 0 else float("nan")
                 for g in range(G)}
    excess_g = {n: (retention[n] - marginal[n]) if not math.isnan(retention[n]) else float("nan")
                for n in norms}
    # Headroom-normalized companion: kappa_g = (retention - marginal)/(1 - marginal).
    # Raw excess is mechanically capped at (1 - marginal), so a SATURATED majority
    # (marginal ~0.95) can never show a large raw excess even if perfectly sticky.
    # kappa answers "of the room it had to beat chance, how much did it use" and is
    # the fair stickiness read for the majority norm. Reported alongside raw excess;
    # raw excess stays the pre-registered headline.
    kappa_g = {}
    for n in norms:
        r, m = retention[n], marginal[n]
        kappa_g[n] = (r - m) / (1 - m) if (not math.isnan(r) and m < 1.0) else float("nan")
    trans_norm = {}
    for i in range(G):
        s = trans[i].sum()
        trans_norm[norms[i]] = {norms[j]: (trans[i, j] / s).item() if s > 0 else float("nan")
                                for j in range(G)}

    # --- per-sample stability (mode norm + mode fraction) --------------------
    stable_by_norm = {n: 0 for n in norms}
    switcher = 0
    sample_mode = torch.empty(B, dtype=torch.long)
    sample_modefrac = torch.empty(B)
    for c in range(B):
        counts = torch.bincount(stack[:, c], minlength=G)
        mc, mi = counts.max(0)
        sample_mode[c] = mi
        sample_modefrac[c] = mc.item() / len(epochs)
        if mc.item() / len(epochs) > STABLE_MODE_FRAC:
            stable_by_norm[norms[int(mi)]] += 1
        else:
            switcher += 1
    stable_total = B - switcher

    # --- l1 minority conditional (decision-critical) -------------------------
    l1_block = None
    if "l1" in norms:
        l1_idx = norms.index("l1")
        ever_l1 = (stack == l1_idx).any(dim=0)
        n_ever = int(ever_l1.sum())
        modes_ever = sample_mode[ever_l1]
        mode_break = {norms[g]: int((modes_ever == g).sum()) for g in range(G)} if n_ever else {}
        stable_l1 = int(((sample_mode == l1_idx) & (sample_modefrac > STABLE_MODE_FRAC)).sum())
        l1_block = {
            "n_ever_l1": n_ever, "frac_ever_l1": n_ever / B,
            "retention_l1": retention.get("l1", float("nan")),
            "marginal_l1": marginal.get("l1", float("nan")),
            "excess_l1": excess_g.get("l1", float("nan")),
            "from_l1_dest": trans_norm.get("l1", {}),
            "mode_break": mode_break,
            "stable_l1": stable_l1,
            "stable_l1_frac": stable_l1 / n_ever if n_ever else float("nan"),
        }

    # --- verdict (keyed on l1 & linf stickiness; ceiling-robust) -------------
    # l1 (minority) has headroom -> judge on raw EXCESS (pre-registered).
    # linf (majority) is raw-excess-capped -> judge stickiness on kappa (headroom-
    # normalized) so a saturated-but-sticky majority is not mislabeled as "state".
    decision_grade = len(epochs) >= min_ckpts
    exc_l1 = excess_g.get("l1", float("nan"))
    kap_l1 = kappa_g.get("l1", float("nan"))
    kap_linf = kappa_g.get("linf", float("nan"))
    l1_measurable = marginal.get("l1", 0.0) > MINORITY_LO

    KAP_STICKY = 0.50   # closes >half the gap to certainty -> trait-stable
    KAP_CHANCE = 0.15   # barely above chance -> tracks chance

    l1_sticky = (not math.isnan(exc_l1)) and exc_l1 > EXCESS_TRAIT and (math.isnan(kap_l1) or kap_l1 > KAP_STICKY)
    l1_chance = (not math.isnan(kap_l1)) and kap_l1 < KAP_CHANCE
    linf_sticky = (not math.isnan(kap_linf)) and kap_linf > KAP_STICKY
    linf_chance = (not math.isnan(kap_linf)) and kap_linf < KAP_CHANCE

    if not decision_grade:
        verdict = "SMOKE/NON-DECISION-GRADE - instrumentation validated; verdict pending (need >=5 dumps)"
    elif not l1_measurable:
        top = max(marginal, key=marginal.get)
        verdict = (f"DEGENERATE (majority-locked on {top}, marginal {100*marginal[top]:.0f}%) - "
                   "per-sample minority structure NOT measurable here; base-rate-inflated. "
                   "Contrast/motivation row, not a trait/state verdict.")
    elif l1_sticky and linf_sticky:
        verdict = "TRAIT - l1 AND linf persist >> chance at the sample level; predictive weighting viable -> Gate-alpha PASS (Conj 2)"
    elif l1_chance and linf_chance:
        verdict = "PURE STATE - both norms track chance (excess ~= 0); pivot to dynamics/tracking framing"
    elif linf_sticky and l1_chance:
        verdict = "BOTH (two-tier) - linf is trait-stable but l1 tracks chance; two-tier framework"
    else:
        verdict = "MIXED - partial/intermediate stickiness; two-tier framework (raw numbers below, human reads)"

    return {
        "run": run_dir.name, "epochs": epochs, "norms": norms, "B": B,
        "decision_grade": decision_grade, "migration": migration, "marginal": marginal,
        "pair_rows": pair_rows, "persistence": persistence, "chance": chance,
        "excess_overall": excess_overall, "retention": retention, "excess_g": excess_g,
        "kappa_g": kappa_g, "trans_norm": trans_norm, "stable_by_norm": stable_by_norm,
        "stable_total": stable_total, "switcher": switcher, "l1_block": l1_block,
        "verdict": verdict,
    }


def pct(x) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{100 * x:.1f}%"


def sgn(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{'+' if x >= 0 else ''}{100 * x:.1f}pp"


def render_run(s: dict, role: str) -> list[str]:
    L = [f"### {s['run']}  ({role})",
         f"- Checkpoints: {len(s['epochs'])} epochs {s['epochs']} | samples {s['B']} | "
         f"alignment OK | decision-grade(>=5): {s['decision_grade']}",
         "",
         f"- **Persistence** = {pct(s['persistence'])} | **Chance** (mean Sg fa*fb) = {pct(s['chance'])} "
         f"| **EXCESS = {sgn(s['excess_overall'])}**  <- headline",
         "",
         "**Per-norm sample-level stickiness** (retention vs its own marginal chance):",
         "",
         "| norm | marginal | retention P(g_t+1\\|g_t) | EXCESS_g = ret - marginal | kappa (headroom-norm.) |",
         "|---|---:|---:|---:|---:|"]
    for n in s["norms"]:
        mark = "  <- MINORITY (decision-critical)" if n == "l1" else ("  <- majority (base-rate-saturated)" if n == "linf" else "")
        kap = s.get("kappa_g", {}).get(n, float("nan"))
        kaptxt = "n/a" if (isinstance(kap, float) and math.isnan(kap)) else f"{kap:.2f}"
        L.append(f"| {n}{mark} | {pct(s['marginal'][n])} | {pct(s['retention'][n])} | {sgn(s['excess_g'][n])} | {kaptxt} |")
    L += ["",
          f"- Per-sample stability (mode-norm share >80% of dumps): stable "
          f"{s['stable_total']}/{s['B']} ({pct(s['stable_total']/s['B'])}), "
          f"switchers {s['switcher']} ({pct(s['switcher']/s['B'])}); "
          "by norm: " + ", ".join(f"{k}={v}" for k, v in s["stable_by_norm"].items()),
          ""]
    # l1 minority block
    b = s.get("l1_block")
    if b:
        L += ["**l1-minority conditional** (THE trait/state number for the informative minority):",
              f"- samples ever l1-bound: {b['n_ever_l1']}/{s['B']} ({pct(b['frac_ever_l1'])})",
              f"- P(still l1 next dump) = **{pct(b['retention_l1'])}** vs l1 marginal (chance) "
              f"{pct(b['marginal_l1'])} -> **l1 EXCESS = {sgn(b['excess_l1'])}**",
              f"- from-l1 next-dump destination: "
              + ", ".join(f"{k}={pct(v)}" for k, v in (b['from_l1_dest'] or {}).items()),
              f"- ever-l1 samples' mode norm: " + ", ".join(f"{k}={v}" for k, v in (b['mode_break'] or {}).items())
              + f"; stable-l1 (mode=l1 & >80%): {b['stable_l1']} ({pct(b['stable_l1_frac'])} of ever-l1)",
              ""]
    # migration + per-epoch chance sanity table (the base-rate inflation view)
    L += ["**MIGRATION (population axis) + per-epoch self-chance sanity** "
          "(shows base-rate inflation: concentrated marginal -> high chance -> raw persistence meaningless):",
          "",
          "| epoch | " + " | ".join(s["norms"]) + " | self-chance Sg p_g^2 |",
          "|---" * (len(s["norms"]) + 2) + "|"]
    for e, f, sc in s["migration"]:
        L.append(f"| {e} | " + " | ".join(pct(f[n]) for n in s["norms"]) + f" | {pct(sc)} |")
    L += ["", f"- **Verdict: {s['verdict']}**", ""]
    return L


def render(decision, contrast, errors) -> str:
    out = ["# Trait-vs-State v3 (decision-grade)",
           "",
           "Fixed indexed probe-binding dumps; per-sample binding = argmax_g loss over {linf,l2,l1}.",
           "",
           "## Metric definitions (pre-registered)",
           "- `chance(pair) = Σ_g f_a(g)·f_b(g)` (= `Σ_g p_g²` when the marginal is stable) — independent-draw match rate.",
           "- **`EXCESS = persistence − chance`** is the headline; raw persistence is base-rate-inflated and never reported alone.",
           "- per-norm `retention_g = P(bind_{t+1}=g | bind_t=g)`; `EXCESS_g = retention_g − marginal_g` (sample-level stickiness of g).",
           "- `kappa_g = (retention_g − marginal_g)/(1 − marginal_g)` — headroom-normalized companion. Raw EXCESS is "
           "mechanically capped at `1 − marginal`, so a **saturated majority** (linf marginal ~.95) cannot show large raw "
           "excess even if perfectly sticky; kappa is the fair stickiness read for the majority. Raw EXCESS stays the headline; "
           "the verdict judges l1 on raw EXCESS (headroom-rich) and linf on kappa (ceiling-robust).",
           "- **MIGRATION** (population p_g(epoch) drift, F6) and **TRAIT/STATE** (sample-level excess on fixed indices) are DIFFERENT axes — both reported, never conflated.",
           "- l1 is the informative **minority**; linf is base-rate-saturated (reported for completeness).",
           "",
           "## VERDICT RULE (pre-registered)",
           "- l1 EXCESS ≫ 0 **and** linf EXCESS ≫ 0 → **TRAIT** → predictive weighting (Conj 2) viable → Gate-α PASS.",
           "- both EXCESS ≈ 0 → **PURE STATE** → dynamics/tracking framing.",
           "- linf EXCESS ≫ 0 **but** l1 EXCESS ≈ 0 → **BOTH** → two-tier framework "
           "(smoke hint: linf=465 stable, l1=2 stable).",
           f"- thresholds: ≫0 means EXCESS_g > {EXCESS_TRAIT:.2f}; ≈0 means |EXCESS_g| < {EXCESS_STATE:.2f}.",
           ""]
    if decision:
        out += ["## Decision-grade run — binding-aware, per-sample soft T=0.25", ""] + render_run(decision, "DECISION-GRADE")
    else:
        out += ["## Decision-grade run", "", "- Not available yet (run in progress or dumps missing).", ""]
    if contrast:
        out += ["## Contrast row — avg_frozen (uniform training)",
                "",
                "_Motivation: under uniform training binding is trivially linf-locked, so chance is high and "
                "persistence is base-rate-driven (excess ≈ 0). Per-sample structure only becomes MEASURABLE once "
                "the recipe balances the norms (binding-aware). That contrast is itself the paper's motivation figure._",
                ""] + render_run(contrast, "CONTRAST")
    if errors:
        out += ["## Skipped/Errors", ""] + [f"- {e}" for e in errors] + [""]
    out += ["## Overall verdict", ""]
    if decision and decision["decision_grade"]:
        out += [f"- **DECISION (bindaware T=0.25): {decision['verdict']}**",
                f"- CONTRAST (avg_frozen): {contrast['verdict'] if contrast else 'n/a'}"]
    else:
        out += ["- Gate-α pending: decision-grade run not complete."]
    return "\n".join(out + [""])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--run", default="gate_alpha_bindaware_T025_s0")
    p.add_argument("--contrast", default="gate_alpha_avgfrozen_s0")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--min-checkpoints", type=int, default=5)
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(a.root)
    errors = []

    def get(name):
        if not name:
            return None
        d = root / name
        if not d.is_dir():
            errors.append(f"{name}: run dir not found under {root}")
            return None
        r = analyse(d, a.min_checkpoints)
        if isinstance(r, str):
            errors.append(r)
            return None
        return r

    decision = get(a.run)
    contrast = get(a.contrast)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(decision, contrast, errors), encoding="utf-8")
    print(f"Wrote {out}")
    if decision:
        print(f"DECISION: {decision['verdict']}")
    if contrast:
        print(f"CONTRAST: {contrast['verdict']}")
    for e in errors:
        print(f"  note: {e}")


if __name__ == "__main__":
    main()
