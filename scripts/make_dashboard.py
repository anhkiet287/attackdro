#!/usr/bin/env python3
"""Build docs/dashboard.html from STATE plus result JSON files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from string import Template
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
STATE = DOCS / "STATE.md"
OUT = DOCS / "dashboard.html"


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path.read_text(encoding="utf-8")


def pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def as_percent(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if -1.0 <= number <= 1.0:
        number *= 100.0
    return number


def fmt_percent(value: Any) -> str:
    number = as_percent(value)
    if number is None:
        return ""
    return f"{number:.1f}"


def fmt_flops(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def baseline_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


@dataclass
class ResultRow:
    name: str
    clean: Any
    linf: Any
    l2: Any
    l1: Any
    union: Any
    flops: Any
    n: Any
    version: str
    source: str
    role: str = ""
    split: str = ""
    checkpoint: str = ""
    used_for_selection: Any = None
    tier: str = "in-house"
    eps: str = ""
    flag: str = ""


def infer_tier(name: str, source: str) -> str:
    s = (name + " " + source).lower()
    if source.startswith("baseline") or name.startswith("baseline:"):
        return "official"
    if "ramp" in s:
        return "repro"
    return "in-house"


def eps_label(payload: dict) -> str:
    """Human eps_inf label from a payload's recorded protocol (source of truth)."""
    tm = (payload.get("protocol") or {}).get("threat_model") or {}
    e = (tm.get("linf") or {}).get("eps")
    if e is None:
        return ""
    return "8/255" if abs(float(e) - 8 / 255) < 1e-6 else f"{float(e):g}"


def eps_flag(payload: dict) -> str:
    """Auto flag: retired-0.03 eval, or train@0.03/eval@8255 mismatch (lower bound)."""
    tm = (payload.get("protocol") or {}).get("threat_model") or {}
    e = (tm.get("linf") or {}).get("eps")
    if e is not None and abs(float(e) - 8 / 255) > 1e-6:
        return "0.03 pre-relock"
    if payload.get("train_eval_eps_mismatch"):
        return "train@0.03/eval@8255 — lower bound"
    return ""


def norm_value(metrics: dict, per_norm: dict, norm: str) -> Any:
    keys = [f"eval/robust_{norm}", f"robust_{norm}", norm]
    if norm == "linf":
        keys.append("l_inf")
    value = pick(metrics, *keys)
    if value is not None:
        return value
    return pick(per_norm, norm, "l_inf") if per_norm else None


def prefixed_norm_value(metrics: dict, prefix: str, norm: str) -> Any:
    value = pick(metrics, f"{prefix}/robust_{norm}")
    if value is not None:
        return value
    if norm == "linf":
        return pick(metrics, f"{prefix}/robust_l_inf")
    return None


def selection_label(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def train_flops_for_eval(path: Path) -> float | None:
    """Final train-time FLOPs ratio beside results/<run>/s<seed>/eval.json.

    Do not invent a reactive/RAMP value: if no train.json key exists, return None.
    """
    train_path = path.parent / "train.json"
    if not train_path.exists():
        return None
    try:
        payload = json.loads(train_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    history = payload.get("history") or []
    for row in reversed(history):
        value = pick(
            row,
            "efficiency/attack_flops_ratio",
            "pb/attack_flops_ratio",
            "exp/attack_flops_ratio",
        )
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def load_results() -> list[ResultRow]:
    rows: list[ResultRow] = []
    json_paths = sorted(RESULTS.glob("*.json")) + sorted((RESULTS / "ramp").glob("*.json"))
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if path.name == "baseline_table.json":
            table_rows = payload.get("rows", {})
            if isinstance(table_rows, dict):
                iterable = [dict(values, model=name) for name, values in table_rows.items()]
            else:
                iterable = table_rows
            protocol = payload.get("protocol", {})
            for item in iterable:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    ResultRow(
                        name=f"baseline:{pick(item, 'model', 'name')}",
                        clean=baseline_metric(pick(item, "clean")),
                        linf=baseline_metric(pick(item, "linf", "l_inf")),
                        l2=baseline_metric(pick(item, "l2")),
                        l1=baseline_metric(pick(item, "l1")),
                        union=baseline_metric(pick(item, "union", "worst_union", "worst_union_acc")),
                        flops=None,
                        n=protocol.get("n_examples", ""),
                        version=str(protocol.get("version", "")),
                        source=str(path.relative_to(RESULTS)),
                        role="reference",
                        tier="official",
                        eps=eps_label(payload),
                    )
                )
            continue

        if not path.name.startswith("eval_"):
            continue
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        union = pick(metrics, "eval/worst_union", "worst_union_acc", "worst_union")
        if union is None:
            continue
        per_norm = metrics.get("per_norm_robust_acc", {})
        if not isinstance(per_norm, dict):
            per_norm = {}
        name = path.stem.removeprefix("eval_")
        rows.append(
            ResultRow(
                name=name,
                clean=pick(metrics, "eval/clean_acc", "clean_acc", "clean"),
                linf=norm_value(metrics, per_norm, "linf"),
                l2=norm_value(metrics, per_norm, "l2"),
                l1=norm_value(metrics, per_norm, "l1"),
                union=union,
                flops=pick(metrics, "efficiency/attack_flops_ratio"),
                n=pick(metrics, "n", "n_examples") or payload.get("n_examples", ""),
                version=str(metrics.get("version", payload.get("version", ""))),
                source=str(path.relative_to(RESULTS)),
                role=("final" if "fullaa" in path.name.lower() else "eval"),
                split=str(metrics.get("eval/split", "")),
                checkpoint=str(metrics.get("eval/checkpoint_role", metrics.get("eval/checkpoint", ""))),
                used_for_selection=metrics.get("eval/used_for_selection"),
                tier=infer_tier(name, path.name),
                eps=eps_label(payload),
                flag=eps_flag(payload),
            )
        )

    # NEW per-run layout: results/<run>/s<seed>/eval.json (screening APGD) + eval_fullAA.json
    # (final full-AA). results/archive/* is excluded (its subdirs are not s<seed>).
    for path in sorted(RESULTS.glob("*/s*/eval*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = payload.get("metrics", payload)
        if not isinstance(metrics, dict):
            continue
        per_norm = metrics.get("per_norm_robust_acc", {}) or {}
        run = path.parent.parent.name              # results/<run>/s<seed>/eval.json -> <run>
        seed = path.parent.name                    # s<seed>
        nested_rows = 0
        for split in ("val_select", "test_monitor"):
            for checkpoint in ("val_best", "last"):
                prefix = f"eval/{split}/{checkpoint}"
                union = pick(metrics, f"{prefix}/worst_union")
                if union is None:
                    continue
                nested_rows += 1
                rows.append(
                    ResultRow(
                        name=f"{run}_{seed}_{split}_{checkpoint}",
                        clean=pick(metrics, f"{prefix}/clean_acc"),
                        linf=prefixed_norm_value(metrics, prefix, "linf"),
                        l2=prefixed_norm_value(metrics, prefix, "l2"),
                        l1=prefixed_norm_value(metrics, prefix, "l1"),
                        union=union,
                        flops=pick(metrics, "efficiency/attack_flops_ratio") or train_flops_for_eval(path),
                        n=pick(metrics, f"{prefix}/n_examples") or payload.get("n_examples", ""),
                        version=str(metrics.get("version", payload.get("version", ""))),
                        source=str(path.relative_to(RESULTS)),
                        role="develop",
                        split=split,
                        checkpoint=checkpoint,
                        used_for_selection=pick(metrics, f"{prefix}/used_for_selection"),
                        tier=infer_tier(run, path.name),
                        eps=eps_label(payload),
                        flag=eps_flag(payload),
                    )
                )
        final_prefix = "final/test_final"
        final_union = pick(metrics, f"{final_prefix}/worst_union")
        if final_union is not None:
            nested_rows += 1
            rows.append(
                ResultRow(
                    name=f"{run}_{seed}_test_final",
                    clean=pick(metrics, f"{final_prefix}/clean_acc"),
                    linf=prefixed_norm_value(metrics, final_prefix, "linf"),
                    l2=prefixed_norm_value(metrics, final_prefix, "l2"),
                    l1=prefixed_norm_value(metrics, final_prefix, "l1"),
                    union=final_union,
                    flops=pick(metrics, "efficiency/attack_flops_ratio") or train_flops_for_eval(path),
                    n=pick(metrics, f"{final_prefix}/n_examples") or payload.get("n_examples", ""),
                    version=str(metrics.get("version", payload.get("version", ""))),
                    source=str(path.relative_to(RESULTS)),
                    role="final",
                    split="test_final",
                    checkpoint=str(metrics.get("eval/checkpoint_role", metrics.get("eval/checkpoint", ""))),
                    used_for_selection=False,
                    tier=infer_tier(run, path.name),
                    eps=eps_label(payload),
                    flag=eps_flag(payload),
                )
            )
        if nested_rows:
            continue
        union = pick(metrics, "eval/worst_union", "worst_union_acc", "worst_union")
        if union is None:
            continue
        suffix = "_fullAA" if "fullAA" in path.stem else ""
        name = f"{run}_{seed}{suffix}"
        rows.append(
            ResultRow(
                name=name,
                clean=pick(metrics, "eval/clean_acc", "clean_acc", "clean"),
                linf=norm_value(metrics, per_norm, "linf"),
                l2=norm_value(metrics, per_norm, "l2"),
                l1=norm_value(metrics, per_norm, "l1"),
                union=union,
                flops=pick(metrics, "efficiency/attack_flops_ratio") or train_flops_for_eval(path),
                n=pick(metrics, "n", "n_examples") or payload.get("n_examples", ""),
                version=str(metrics.get("version", payload.get("version", ""))),
                source=str(path.relative_to(RESULTS)),
                role=("final" if suffix else "legacy-eval"),
                split=str(metrics.get("eval/split", payload.get("eval_split", ""))),
                checkpoint=str(metrics.get("eval/checkpoint_role", payload.get("checkpoint_role", ""))),
                used_for_selection=metrics.get("eval/used_for_selection", payload.get("used_for_selection")),
                tier=infer_tier(name, path.name),
                eps=eps_label(payload),
                flag=eps_flag(payload),
            )
        )

    rows.sort(key=lambda row: as_percent(row.union) if as_percent(row.union) is not None else -1, reverse=True)
    return rows


def inline_markdown(text: str) -> str:
    rendered = escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        rendered,
    )
    return rendered


def render_table(lines: list[str]) -> str:
    parsed: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        parsed.append(cells)
    if not parsed:
        return ""
    head = "".join(f"<th>{inline_markdown(cell)}</th>" for cell in parsed[0])
    body = []
    for row in parsed[1:]:
        body.append("<tr>" + "".join(f"<td>{inline_markdown(cell)}</td>" for cell in row) + "</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def render_fragment(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    html: list[str] = []
    list_open = False
    table_lines: list[str] = []

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            html.append("</ul>")
            list_open = False

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            close_list()
            html.append(render_table(table_lines))
            table_lines = []

    for raw in lines:
        line = raw.rstrip()
        if not line:
            flush_table()
            close_list()
            continue
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("- "):
            if not list_open:
                html.append("<ul>")
                list_open = True
            html.append(f"<li>{inline_markdown(line[2:])}</li>")
        elif line.startswith("#"):
            close_list()
            level = min(len(line) - len(line.lstrip("#")), 4)
            html.append(f"<h{level}>{inline_markdown(line.lstrip('#').strip())}</h{level}>")
        else:
            close_list()
            html.append(f"<p>{inline_markdown(line)}</p>")
    flush_table()
    close_list()
    return "\n".join(html)


def extract_section_like(markdown: str, needle: str) -> str:
    """Extract the body under the first `## ...<needle>...` header (substring match) —
    robust to STATE's numbered/decorated headers (e.g. '## 4 · FINDINGS F1-F9')."""
    m = re.search(rf"^##\s+.*{re.escape(needle)}.*$", markdown, flags=re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"^##\s+", markdown[start:], flags=re.MULTILINE)
    return (markdown[start:start + nxt.start()] if nxt else markdown[start:]).strip()


def extract_section(markdown: str, title: str) -> str:
    match = re.search(rf"^##\s+{re.escape(title)}\s*$", markdown, flags=re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_match = re.search(r"^##\s+", markdown[start:], flags=re.MULTILINE)
    end = start + next_match.start() if next_match else len(markdown)
    return markdown[start:end].strip()


def status_cards(memory_md: str) -> str:
    phase = extract_section_like(memory_md, "Current phase and allowed next actions")
    candidates = {
        "Phase": "",
        "Gate": "",
        "Next Deadline": "",
    }
    for line in phase.splitlines():
        clean = line.removeprefix("- ").strip()
        lower = clean.lower()
        if lower.startswith("current phase"):
            candidates["Phase"] = clean
        elif lower.startswith("g2"):
            candidates["Gate"] = clean
        elif lower.startswith("next deadline"):
            candidates["Next Deadline"] = clean

    cards = []
    for label, text in candidates.items():
        if text:
            cards.append(f"<article><span>{escape(label)}</span><p>{inline_markdown(text)}</p></article>")
    return "".join(cards) if cards else render_fragment(phase)


_TIER_COLORS = {"in-house": "#0f766e", "repro": "#b45309", "official": "#334155"}


def compare_view(rows: list[ResultRow]) -> str:
    """Union-acc bar across runs (sorted desc), per-norm breakdown, tier-filterable.
    Reads the same JSON-derived rows; JSON on disk remains the source of truth."""
    ranked = [r for r in rows if as_percent(r.union) is not None]
    if not ranked:
        return "<p>No union results on disk yet.</p>"
    scale = max((as_percent(r.union) for r in ranked), default=100.0) or 100.0
    bars = []
    for r in ranked:
        u = as_percent(r.union)
        width = max(2.0, 100.0 * u / scale)
        color = _TIER_COLORS.get(r.tier, "#64748b")
        pernorm = " · ".join(
            f"{lbl} {fmt_percent(val)}" for lbl, val in (("l∞", r.linf), ("l2", r.l2), ("l1", r.l1))
        )
        flag_html = f'<span class="cmp-flag">⚠ {escape(r.flag)}</span>' if r.flag else ""
        role_bits = " · ".join(v for v in (r.role, r.split, r.checkpoint) if v) or "eval"
        bars.append(
            f'<div class="cmp-row" data-tier="{escape(r.tier)}">'
            f'<div class="cmp-label">{escape(r.name)}'
            f'<span class="cmp-meta">{escape(r.tier)} · {escape(role_bits)} · eps {escape(r.eps or "?")} · {escape(r.version or "?")}{flag_html}</span></div>'
            f'<div class="cmp-track"><div class="cmp-bar" style="width:{width:.1f}%;background:{color}">'
            f'<span class="cmp-val">{fmt_percent(r.union)}</span></div></div>'
            f'<div class="cmp-pernorm">{escape(pernorm)}</div>'
            "</div>"
        )
    controls = (
        '<div class="cmp-filter">Tier: '
        '<select id="tierFilter" onchange="filterTier()">'
        '<option value="all">all</option>'
        '<option value="in-house">in-house</option>'
        '<option value="repro">repro</option>'
        '<option value="official">official</option>'
        "</select>"
        '<span class="cmp-legend">'
        '<span style="color:#0f766e">■ in-house</span> '
        '<span style="color:#b45309">■ repro</span> '
        '<span style="color:#334155">■ official</span></span></div>'
    )
    return controls + '<div class="cmp">' + "".join(bars) + "</div>"


def results_table(rows: list[ResultRow]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row.name)}</td>"
            f"<td>{fmt_percent(row.clean)}</td>"
            f"<td>{fmt_percent(row.linf)}</td>"
            f"<td>{fmt_percent(row.l2)}</td>"
            f"<td>{fmt_percent(row.l1)}</td>"
            f"<td><strong>{fmt_percent(row.union)}</strong></td>"
            f"<td>{fmt_flops(row.flops)}</td>"
            f"<td>{escape(row.role)}</td>"
            f"<td>{escape(row.split)}</td>"
            f"<td>{escape(row.checkpoint)}</td>"
            f"<td>{escape(selection_label(row.used_for_selection))}</td>"
            f"<td>{escape(row.tier)}</td>"
            f"<td>{escape(row.eps)}</td>"
            f"<td>{('⚠ ' + escape(row.flag)) if row.flag else ''}</td>"
            f"<td>{escape(str(row.n))}</td>"
            f"<td>{escape(row.version)}</td>"
            f"<td><code>{escape(row.source)}</code></td>"
            "</tr>"
        )
    return (
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Run</th><th>Clean</th><th>linf</th><th>l2</th><th>l1</th>"
        "<th>Union</th><th>FLOPs</th><th>Role</th><th>Split</th><th>Checkpoint</th><th>Select?</th>"
        "<th>Tier</th><th>eps</th><th>Flag</th><th>n</th><th>Version</th><th>Source</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def timeline_entries(log_md: str) -> str:
    pattern = re.compile(
        r"^(#{2,3})\s+(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\s+[—-]\s+(.+)$",
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(log_md))
    entries = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(log_md)
        body = log_md[start:end].strip()
        if len(body) > 2400:
            body = body[:2400].rstrip() + "\n\n..."
        entries.append((match.group(2), match.group(3).strip(), body))

    cards = []
    for date, title, body in reversed(entries):  # newest first
        cards.append(
            "<article class=\"timeline-item\">"
            f"<div class=\"timeline-date\">{escape(date)}</div>"
            f"<h3>{inline_markdown(title)}</h3>"
            f"{render_fragment(body)}"
            "</article>"
        )
    if not cards:
        return "<p>No dated log entries found.</p>"
    # Current-state-first: keep the newest 8 expanded, fold the rest behind a toggle.
    keep = 8
    recent = "\n".join(cards[:keep])
    if len(cards) <= keep:
        return recent
    older = "\n".join(cards[keep:])
    return (recent
            + f'<details class="tl-fold"><summary>Show older history ({len(cards) - keep} entries)'
              f'</summary>{older}</details>')


def render_ramp_chart() -> str:
    """Inline SVG line chart of RAMP worst-union vs epoch (single series → one hue, no
    legend; native <title> hover; ep70 lr-drop annotation = the story). Data-driven from
    the eval JSONs; empty string if any epoch is missing."""
    eps = [10, 20, 30, 40, 50, 60, 70, 80]
    pts = []
    for e in eps:
        p = RESULTS / "ramp" / f"eval_ramp_ep{e}_eps8255_apgd_n1000.json"
        if not p.exists():
            return ""
        try:
            m = json.loads(p.read_text(encoding="utf-8"))["metrics"]
            pts.append((e, 100 * m["worst_union_acc"]))
        except Exception:
            return ""
    W, H, Lm, Rm, Tm, Bm = 760, 320, 54, 92, 22, 46
    pw, ph = W - Lm - Rm, H - Tm - Bm
    ymin, ymax = 30, 48

    def X(e):
        return Lm + (e - eps[0]) / (eps[-1] - eps[0]) * pw

    def Y(u):
        return Tm + (ymax - u) / (ymax - ymin) * ph

    grid = "".join(
        f'<line x1="{Lm}" y1="{Y(g):.1f}" x2="{Lm + pw}" y2="{Y(g):.1f}" class="ch-grid"/>'
        f'<text x="{Lm - 8}" y="{Y(g) + 4:.1f}" class="ch-ylab" text-anchor="end">{g}</text>'
        for g in (30, 35, 40, 45))
    xlab = "".join(f'<text x="{X(e):.1f}" y="{Tm + ph + 18}" class="ch-xlab" text-anchor="middle">{e}</text>'
                   for e in eps)
    x70 = X(70)
    anno = (f'<line x1="{x70:.1f}" y1="{Tm}" x2="{x70:.1f}" y2="{Tm + ph}" class="ch-anno"/>'
            f'<text x="{x70 - 7:.1f}" y="{Tm + 12}" class="ch-annolab" text-anchor="end">lr-drop 0.05→0.005 @ep70</text>')
    poly = " ".join(f"{X(e):.1f},{Y(u):.1f}" for e, u in pts)
    dots = "".join(f'<circle cx="{X(e):.1f}" cy="{Y(u):.1f}" r="4.5" class="ch-dot">'
                   f'<title>ep{e}: union {u:.1f}%</title></circle>' for e, u in pts)
    trough = min(pts, key=lambda pu: pu[1])
    labels = (f'<text x="{X(80) + 8:.1f}" y="{Y(pts[-1][1]) + 4:.1f}" class="ch-endlab">{pts[-1][1]:.1f}</text>'
              f'<text x="{X(eps[0]):.1f}" y="{Y(pts[0][1]) - 9:.1f}" class="ch-lab" text-anchor="middle">{pts[0][1]:.1f}</text>'
              f'<text x="{X(trough[0]):.1f}" y="{Y(trough[1]) + 17:.1f}" class="ch-lab" text-anchor="middle">{trough[1]:.1f}</text>')
    ytitle = (f'<text transform="rotate(-90)" x="{-(Tm + ph / 2):.1f}" y="15" '
              f'class="ch-axtitle" text-anchor="middle">worst-union robust acc (%)</text>')
    xtitle = f'<text x="{Lm + pw / 2:.1f}" y="{H - 6}" class="ch-axtitle" text-anchor="middle">training epoch</text>'
    return (f'<figure class="ch-fig"><svg viewBox="0 0 {W} {H}" class="ch-svg" role="img" '
            f'aria-label="RAMP worst-union robust accuracy versus training epoch at eps 8/255, APGD n=1000">'
            f'{grid}{anno}<polyline points="{poly}" class="ch-line"/>{dots}{labels}{xlab}{ytitle}{xtitle}</svg>'
            f'<figcaption class="ch-cap">RAMP union vs epoch @8/255 (APGD n=1000) — plateaus ~43 across '
            f'ep30–70, jumps to 46 only after the ep70 lr-drop; ep50 (42.9) is a local trough.</figcaption></figure>')


_SRC_LABEL = {"in-house": "IN-HOUSE", "cited": "CITED", "cross-confirmed": "CROSS-CONFIRMED"}


def load_findings() -> list:
    p = RESULTS / "findings.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("findings", [])
    except Exception:
        return []


def render_findings(items: list) -> str:
    """F1-F9 as cards: conclusion + evidence + TYPED source + status badge (P-06). F9's
    reasoning chain is an expandable <details> so the R-chain stays visible without bloating."""
    if not items:
        return "<p>No <code>results/findings.json</code>.</p>"
    out = []
    for f in items:
        srct = f.get("source_type", "")
        st = f.get("status", "")
        card = [f'<article class="finding src-{escape(srct)}">',
                '<div class="finding-head">',
                f'<span class="fid">{escape(f.get("id", ""))}</span>',
                f'<span class="ftag ftag-{escape(srct)}">{escape(_SRC_LABEL.get(srct, srct))}</span>',
                f'<span class="fstatus fstatus-{escape(st)}">{escape(st)}</span>',
                '</div>',
                f'<p class="fconc">{inline_markdown(f.get("conclusion", ""))}</p>',
                f'<p class="fev"><strong>Evidence:</strong> {inline_markdown(f.get("evidence", ""))}</p>',
                f'<p class="fsrc"><strong>Source:</strong> {inline_markdown(f.get("source", ""))}</p>']
        chain = f.get("chain")
        if chain:
            lis = "".join(f"<li>{inline_markdown(c)}</li>" for c in chain)
            card.append('<details class="fchain"><summary>reasoning chain '
                        '(R1 inert-floor → R3 biased-φ → v2 confound → frontier)</summary>'
                        f'<ol>{lis}</ol></details>')
        card.append("</article>")
        out.append("".join(card))
    return "\n".join(out)


def main() -> None:
    memory_md = read_text(STATE)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    template = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AttackDRO Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-2: #334155;
      --soft: #eef6f5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.5;
    }
    header {
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff 0%, #f7f8fb 100%);
    }
    header h1 {
      margin: 0 0 6px;
      font-size: clamp(1.8rem, 3vw, 2.6rem);
      letter-spacing: 0;
    }
    header p { margin: 0; color: var(--muted); }
    main {
      width: min(1280px, calc(100vw - 32px));
      margin: 22px auto 40px;
      display: grid;
      gap: 18px;
    }
    section, .status article {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    section { padding: 18px; }
    h2 { margin: 0 0 12px; font-size: 1.1rem; letter-spacing: 0; }
    h3 { margin: 4px 0 8px; font-size: 1rem; }
    p { margin: 8px 0; }
    ul { margin: 8px 0 0 18px; padding: 0; }
    li { margin: 6px 0; }
    code {
      background: #eef2f7;
      border: 1px solid #dce3ec;
      padding: 0 4px;
      border-radius: 4px;
      font-size: 0.92em;
    }
    .status {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .status article { padding: 14px; }
    .status span {
      display: block;
      color: var(--accent);
      font-weight: 700;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .status p { margin-bottom: 0; }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
      gap: 18px;
      align-items: start;
    }
    .table-wrap {
      max-width: 100%;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      background: #fff;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: #f1f5f9;
      color: var(--accent-2);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    tbody tr:hover { background: var(--soft); }
    .timeline {
      max-height: 560px;
      overflow: auto;
      padding-right: 8px;
    }
    .timeline-item {
      border-left: 3px solid var(--accent);
      padding: 8px 0 14px 14px;
      margin: 0 0 12px;
    }
    .timeline-date {
      color: var(--muted);
      font-size: 0.86rem;
      font-weight: 700;
    }
    .tl-fold { margin-top: 8px; }
    .tl-fold summary { cursor: pointer; color: var(--accent); font-weight: 700; font-size: 0.88rem;
      padding: 6px 0; list-style: revert; }
    textarea {
      width: 100%;
      min-height: 170px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    .meta {
      color: var(--muted);
      font-size: 0.88rem;
      margin-top: 12px;
    }
    .status-header { display: grid; gap: 6px; margin: 0 0 14px; padding: 12px 14px;
      background: var(--soft); border: 1px solid var(--line); border-radius: 8px; }
    .sh-row { display: grid; grid-template-columns: 170px 1fr; gap: 10px; align-items: baseline; }
    .sh-k { color: var(--accent); font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; }
    .sh-v { font-size: 0.95rem; }
    .sh-meta { color: var(--muted); font-size: 0.78rem; margin-top: 4px; }
    .st-group { margin: 0 0 14px; }
    .st-group h3 { margin: 0 0 8px; font-size: 0.98rem; }
    .st-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 6px 8px;
      border-bottom: 1px solid var(--line); font-size: 0.88rem; }
    .st-badge, .st-tier { color: #fff; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.03em; padding: 2px 7px; border-radius: 999px; white-space: nowrap; }
    .st-name { font-weight: 600; }
    .st-note { color: var(--muted); font-weight: 400; font-size: 0.78rem; }
    .st-num { font-variant-numeric: tabular-nums; color: var(--accent-2); font-weight: 700; }
    .st-flag { color: #b45309; font-weight: 600; font-size: 0.78rem; }
    .st-detail { color: var(--muted); font-size: 0.82rem; flex: 1 1 200px; }
    .st-order { font-weight: 800; color: var(--accent-2); font-size: 0.82rem; min-width: 26px; }
    .st-upd { color: var(--muted); font-size: 0.72rem; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .st-hint { color: var(--muted); font-weight: 400; font-size: 0.78rem; }
    .pwrap { display: inline-flex; align-items: center; gap: 7px; }
    .pbar { width: 130px; height: 9px; background: #e5e9f0; border-radius: 999px; overflow: hidden; }
    .pfill { height: 100%; background: var(--accent); border-radius: 999px; transition: width .3s; }
    .pfill.queued { background: repeating-linear-gradient(45deg, #cbd5e1, #cbd5e1 5px, #eef2f7 5px, #eef2f7 10px); }
    .plabel { font-size: 0.74rem; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
    .cmp-filter { margin: 0 0 12px; color: var(--muted); font-size: 0.9rem; }
    .cmp-filter select { font: inherit; padding: 3px 6px; border: 1px solid var(--line); border-radius: 6px; }
    .cmp-legend { margin-left: 14px; font-size: 0.85rem; }
    .cmp { display: grid; gap: 8px; }
    .cmp-row { display: grid; grid-template-columns: minmax(150px, 220px) minmax(0, 1fr) minmax(150px, 210px); gap: 10px; align-items: center; }
    .cmp-label { font-size: 0.86rem; font-weight: 600; }
    .cmp-meta { display: block; color: var(--muted); font-weight: 400; font-size: 0.74rem; }
    .cmp-track { background: #eef2f7; border-radius: 6px; overflow: hidden; height: 22px; }
    .cmp-bar { height: 100%; display: flex; align-items: center; justify-content: flex-end; border-radius: 6px; min-width: 34px; transition: width .2s; }
    .cmp-val { color: #fff; font-size: 0.78rem; font-weight: 700; padding-right: 7px; }
    .cmp-pernorm { color: var(--muted); font-size: 0.78rem; white-space: nowrap; }
    .cmp-flag { color: #b45309; font-weight: 600; margin-left: 6px; }
    @media (max-width: 860px) {
      header { padding: 22px 16px 16px; }
      .cmp-row { grid-template-columns: 1fr; }
      main { width: min(100vw - 20px, 1280px); }
      .status, .grid { grid-template-columns: 1fr; }
      th, td { white-space: normal; }
    }
    /* --- Findings F1-F9 cards (P-06): typed source + status --- */
    .finding { border: 1px solid var(--line); border-left-width: 5px; border-radius: 8px;
               padding: 10px 13px; margin: 9px 0; background: var(--panel); }
    .finding.src-in-house { border-left-color: #0d9488; }
    .finding.src-cited { border-left-color: #d97706; }
    .finding.src-cross-confirmed { border-left-color: #7c3aed; }
    .finding-head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 5px; }
    .fid { font-weight: 800; font-size: 0.98rem; color: var(--accent-2); }
    .ftag, .fstatus { font-size: 11px; padding: 1.5px 8px; border-radius: 11px; font-weight: 700;
                      letter-spacing: 0.02em; }
    .ftag-in-house { background: #ccfbf1; color: #0f766e; }
    .ftag-cited { background: #fef3c7; color: #92400e; }
    .ftag-cross-confirmed { background: #ede9fe; color: #6d28d9; }
    .fstatus-confirmed { background: #dcfce7; color: #166534; }
    .fstatus-decision-grade { background: #dbeafe; color: #1e40af; }
    .fstatus-open-frontier { background: #ffedd5; color: #9a3412; }
    .fstatus-fixes-committed { background: #f3e8ff; color: #7e22ce; }
    .fconc { margin: 3px 0; font-weight: 600; color: var(--ink); }
    .fev, .fsrc { margin: 3px 0; font-size: 0.86rem; color: var(--muted); }
    .fsrc code, .fev code { font-size: 0.82rem; }
    .fchain { margin-top: 7px; font-size: 0.86rem; }
    .fchain summary { cursor: pointer; color: var(--accent); font-weight: 700; }
    .fchain ol { margin: 6px 0 2px 0; padding-left: 20px; }
    .fchain li { margin: 5px 0; color: var(--ink); }
    /* --- RAMP union-vs-epoch chart (P-07): single-series line --- */
    .ch-fig { margin: 0 0 10px; }
    .ch-svg { width: 100%; height: auto; max-width: 760px; display: block; }
    .ch-grid { stroke: var(--line); stroke-width: 1; }
    .ch-ylab, .ch-xlab { fill: var(--muted); font-size: 12px; }
    .ch-axtitle { fill: var(--muted); font-size: 12px; font-weight: 600; }
    .ch-line { fill: none; stroke: var(--accent); stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
    .ch-dot { fill: var(--accent); stroke: var(--panel); stroke-width: 1.5; }
    .ch-dot:hover { r: 6; }
    .ch-anno { stroke: var(--accent-2); stroke-width: 1.5; stroke-dasharray: 4 3; opacity: .65; }
    .ch-annolab { fill: var(--accent-2); font-size: 11px; font-weight: 600; }
    .ch-endlab { fill: var(--accent); font-size: 13px; font-weight: 800; }
    .ch-lab { fill: var(--ink); font-size: 11px; font-weight: 600; }
    .ch-cap { color: var(--muted); font-size: .8rem; margin-top: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>AttackDRO Dashboard</h1>
    <p>Regenerated from <code>docs/STATE.md</code> and result JSON files at $generated_at. &nbsp;·&nbsp; <a href="archive/landscape.html" style="font-weight:600">→ Archived landscape overview</a></p>
  </header>
  <main>
    <div class="status">$status_cards</div>
    <section>
      <h2>Current Phase And Allowed Next Actions</h2>
      $current_next_action
    </section>
    <section>
      <h2>Eval Split Roles</h2>
      $eval_split_roles
    </section>
    <section>
      <h2>Union robustness by run (compare view)</h2>
      $compare_view
    </section>
    <section>
      <h2>Results</h2>
      $results_table
    </section>
    <section>
      <h2>RAMP union-vs-epoch curve @8/255 (n=1000, APGD, restarts=1)</h2>
      $ramp_chart
    </section>
    <section>
      <h2>Findings — mechanism spine (F1–F9)</h2>
      <p style="margin:0 0 8px;color:var(--muted);font-size:.85rem;">Source type:
        <span class="ftag ftag-in-house">IN-HOUSE</span>
        <span class="ftag ftag-cited">CITED</span>
        <span class="ftag ftag-cross-confirmed">CROSS-CONFIRMED</span> — tiers never mixed.</p>
      $findings
    </section>
    <section>
      <h2>Known Risks And Blockers</h2>
      $open_items
    </section>
    <section>
      <h2>Run-Update Checklist</h2>
      $run_update_checklist
    </section>
    <section>
      <h2>Notes</h2>
      <textarea id="notes" placeholder="Private annotations stay in this browser via localStorage."></textarea>
      <p class="meta">Notes are local to your browser and are not written back to the repo.</p>
    </section>
  </main>
  <script>
    const key = "attackdro.dashboard.notes.v1";
    const notes = document.getElementById("notes");
    notes.value = localStorage.getItem(key) || "";
    notes.addEventListener("input", () => localStorage.setItem(key, notes.value));
    function filterTier() {
      const v = document.getElementById("tierFilter").value;
      document.querySelectorAll(".cmp-row").forEach(row => {
        row.style.display = (v === "all" || row.dataset.tier === v) ? "" : "none";
      });
    }
  </script>
</body>
</html>
"""
    )
    rows = load_results()
    html_page = template.substitute(
        generated_at=escape(generated_at),
        status_cards=status_cards(memory_md),
        current_next_action=render_fragment(extract_section_like(memory_md, "Current phase and allowed next actions")),
        eval_split_roles=render_fragment(extract_section_like(memory_md, "Split and checkpoint-selection protocol")),
        compare_view=compare_view(rows),
        results_table=results_table(rows),
        ramp_chart=render_ramp_chart(),
        findings=render_findings(load_findings()),
        open_items=render_fragment(extract_section_like(memory_md, "Known risks and blockers")),
        run_update_checklist=render_fragment(extract_section_like(memory_md, "Run-update checklist")),
    )

    OUT.write_text(html_page, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
