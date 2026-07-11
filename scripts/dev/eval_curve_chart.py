#!/usr/bin/env python
"""Render the eval convergence curve as an SVG (3 panels: ℓ∞/ℓ2/ℓ1; 3 lines each:
reactive / pred-ks16 / pred-ks8). x = APGD steps (log), y = robust acc %. Plateau = converged.
Reads results/eval_curve/<run>__<norm>__s<steps>.json → writes results/reports/eval_curve_chart.svg."""
import glob
import json
import math
import os
import re

ROOT = "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"
DIR = f"{ROOT}/results/eval_curve"
OUT = f"{ROOT}/results/reports/eval_curve_chart.svg"
NORMS = [("linf", "ℓ∞ (8/255)"), ("l2", "ℓ2 (0.5)"), ("l1", "ℓ1 (12)")]
# categorical, fixed order by entity: baseline=slate, ks16=blue, ks8=amber
RUNS = [("reactive_apgd_8255", "reactive", "#64748B"),
        ("predictive_apgd_8255", "pred ks16", "#2563EB"),
        ("predictive_ks8_apgd_8255", "pred ks8", "#F59E0B")]


def series(run, norm):
    pts = []
    for f in glob.glob(f"{DIR}/{run}__{norm}__s*.json"):
        s = int(re.search(r"__s(\d+)\.json$", f).group(1))
        d = json.load(open(f)); m = d.get("metrics", d)
        pts.append((s, 100 * m["per_norm_robust_acc"][norm]))
    return sorted(pts)


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;")


def main():
    data = {(r[0], n[0]): series(r[0], n[0]) for r in RUNS for n in NORMS}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not any(data.values()):
        open(OUT, "w").write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="60">'
            '<text x="12" y="34" font-family="sans-serif" font-size="14" fill="#64748B">'
            'eval_curve_chart: no data yet (runs after the frontier sweep).</text></svg>')
        print("no data yet — placeholder svg"); return

    PW, PH, GAP = 300, 250, 34            # panel plot-area w/h, gap
    PADL, PADR, PADT, PADB = 46, 74, 84, 46
    W = 3 * (PADL + PW + PADR) + 2 * GAP
    H = PADT + PH + PADB
    S = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'font-family="-apple-system,Segoe UI,Roboto,sans-serif" viewBox="0 0 %d %d">' % (W, H, W, H)]
    S.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="#ffffff" stroke="#e2e8f0"/>')
    S.append(f'<text x="20" y="30" font-size="16" font-weight="700" fill="#0f172a">'
             f'Eval convergence — per-norm APGD robust acc vs steps (best.pt, n=1000)</text>')
    S.append(f'<text x="20" y="50" font-size="12" fill="#64748B">'
             f'flat tail = converged · reactive_ks8 doesn’t exist → reactive_apgd_8255 baseline</text>')
    # legend
    lx = 20
    for _, label, col in RUNS:
        S.append(f'<line x1="{lx}" y1="64" x2="{lx+22}" y2="64" stroke="{col}" stroke-width="3"/>')
        S.append(f'<circle cx="{lx+11}" cy="64" r="3.2" fill="{col}"/>')
        S.append(f'<text x="{lx+28}" y="68" font-size="12" fill="#334155">{esc(label)}</text>')
        lx += 34 + 8 * len(label)

    for pi, (nk, nlabel) in enumerate(NORMS):
        ox = pi * (PADL + PW + PADR + GAP) + PADL
        oy = PADT
        allpts = [p for r in RUNS for p in data[(r[0], nk)]]
        if not allpts:
            continue
        xs = sorted({p[0] for p in allpts})
        ys = [p[1] for p in allpts]
        ymin, ymax = min(ys), max(ys)
        pad = max(1.0, (ymax - ymin) * 0.18); ymin -= pad; ymax += pad
        lxmin, lxmax = math.log(min(xs)), math.log(max(xs))

        def X(s):
            return ox + (0 if lxmax == lxmin else (math.log(s) - lxmin) / (lxmax - lxmin) * PW)

        def Y(v):
            return oy + PH - (v - ymin) / (ymax - ymin) * PH

        # panel title + frame
        S.append(f'<text x="{ox+PW/2:.0f}" y="{oy-10}" font-size="13" font-weight="600" '
                 f'fill="#0f172a" text-anchor="middle">{esc(nlabel)}</text>')
        # y gridlines + labels (4)
        for t in range(5):
            v = ymin + (ymax - ymin) * t / 4
            yy = Y(v)
            S.append(f'<line x1="{ox}" y1="{yy:.1f}" x2="{ox+PW}" y2="{yy:.1f}" stroke="#eef2f7" stroke-width="1"/>')
            S.append(f'<text x="{ox-6}" y="{yy+4:.1f}" font-size="10" fill="#94a3b8" text-anchor="end">{v:.0f}</text>')
        # x ticks
        for s in xs:
            xx = X(s)
            S.append(f'<line x1="{xx:.1f}" y1="{oy+PH}" x2="{xx:.1f}" y2="{oy+PH+4}" stroke="#cbd5e1"/>')
            S.append(f'<text x="{xx:.1f}" y="{oy+PH+18}" font-size="10" fill="#64748b" text-anchor="middle">{s}</text>')
        S.append(f'<text x="{ox+PW/2:.0f}" y="{oy+PH+38}" font-size="10.5" fill="#94a3b8" '
                 f'text-anchor="middle">APGD steps</text>')
        # lines
        for run, label, col in RUNS:
            pts = data[(run, nk)]
            if not pts:
                continue
            d = " ".join(f"{'M' if i==0 else 'L'}{X(s):.1f} {Y(v):.1f}" for i, (s, v) in enumerate(pts))
            S.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
            for s, v in pts:
                S.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3" fill="{col}" stroke="#fff" stroke-width="1"/>')
            s_last, v_last = pts[-1]
            S.append(f'<text x="{X(s_last)+7:.1f}" y="{Y(v_last)+3:.1f}" font-size="10.5" '
                     f'font-weight="600" fill="{col}">{v_last:.1f}</text>')
    S.append("</svg>")
    open(OUT, "w").write("\n".join(S))
    print("wrote results/reports/eval_curve_chart.svg")


if __name__ == "__main__":
    main()
