"""Minimal shared style for paper figures (single-column, grayscale-safe).

Each norm keeps ONE identity everywhere: color (Okabe-Ito, colorblind-safe) +
linestyle + marker, so figures read correctly in grayscale print.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NORMS = ["linf", "l2", "l1"]
NORM_LABEL = {"linf": r"$\ell_\infty$", "l2": r"$\ell_2$", "l1": r"$\ell_1$"}
NORM_COLOR = {"linf": "#0072B2", "l2": "#009E73", "l1": "#E69F00"}
NORM_LS = {"linf": "-", "l2": "--", "l1": "-."}
NORM_MARKER = {"linf": "o", "l2": "s", "l1": "^"}

FIG_W = 3.5  # inches, single column


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (FIG_W, 2.4), "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "lines.linewidth": 1.3, "lines.markersize": 3.5,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.axisbelow": True,
    })
