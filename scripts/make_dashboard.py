#!/usr/bin/env python3
"""Build the self-contained docs/dashboard.html from project docs and results."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from string import Template
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
STATE = DOCS / "PROJECT_STATE.md"          # consolidated sync surface (was MEMORY.md + LOG.md)
LOG = DOCS / "legacy" / "LOG.md"           # retired; still read for the timeline history
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
    n: Any
    version: str
    source: str
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


def load_results() -> list[ResultRow]:
    rows: list[ResultRow] = []
    for path in sorted(RESULTS.glob("*.json")):
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
                        n=protocol.get("n_examples", ""),
                        version=str(protocol.get("version", "")),
                        source=path.name,
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
        union = pick(metrics, "worst_union_acc", "worst_union")
        if union is None:
            continue
        per_norm = metrics.get("per_norm_robust_acc", {})
        if not isinstance(per_norm, dict):
            per_norm = {}
        name = path.stem.removeprefix("eval_")
        rows.append(
            ResultRow(
                name=name,
                clean=pick(metrics, "clean_acc", "clean"),
                linf=pick(per_norm, "linf", "l_inf") if per_norm else pick(metrics, "linf", "l_inf"),
                l2=pick(per_norm, "l2") if per_norm else pick(metrics, "l2"),
                l1=pick(per_norm, "l1") if per_norm else pick(metrics, "l1"),
                union=union,
                n=pick(metrics, "n", "n_examples") or payload.get("n_examples", ""),
                version=str(metrics.get("version", payload.get("version", ""))),
                source=path.name,
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
    robust to PROJECT_STATE's numbered/decorated headers (e.g. '## 4 · FINDINGS F1-F9')."""
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
    phase = extract_section_like(memory_md, "CURRENT PHASE")
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
        bars.append(
            f'<div class="cmp-row" data-tier="{escape(r.tier)}">'
            f'<div class="cmp-label">{escape(r.name)}'
            f'<span class="cmp-meta">{escape(r.tier)} · eps {escape(r.eps or "?")} · {escape(r.version or "?")}{flag_html}</span></div>'
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
        "<th>Union</th><th>Tier</th><th>eps</th><th>Flag</th><th>n</th><th>Version</th><th>Source</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


# ---- Experiment status view (parsed from results/run_status.json + live tmux) ----
# run_status.json holds the STATUS STRUCTURE (what is running/done/planned); this
# generator holds no facts of its own. Robustness NUMBERS for done rows are pulled
# live from results/*.json via each item's optional `result` key, so they can never
# drift. RUNNING rows are cross-checked against live `tmux ls`.
_BADGE = {"running": "#16a34a", "done": "#0f766e", "queued": "#2563eb",
          "gated": "#b45309", "killed": "#dc2626", "conditional": "#64748b",
          "stale": "#d97706"}
_TIER = {"paper": "#15803d", "estimate": "#b45309", "reference": "#475569", "diagnostic": "#7c3aed"}
_TIER_LABEL = {"paper": "PAPER-GRADE", "estimate": "ESTIMATE",
               "reference": "REFERENCE", "diagnostic": "DIAGNOSTIC"}
_GROUPS = [("running", "🟢 Running now"), ("done", "✅ Done — number on disk"),
           ("planned", "⏳ Planned / queued"), ("conditional", "🔵 Conditional / later")]


def live_tmux_sessions():
    """Set of live tmux session names, or None if tmux can't be read (so we can
    distinguish 'no sessions' from 'unverified')."""
    try:
        r = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return set()
        return {ln.split(":", 1)[0] for ln in r.stdout.splitlines() if ":" in ln}
    except Exception:
        return None


def load_run_status() -> dict:
    path = RESULTS / "run_status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _disk_number(result_key: str, rows_by_name: dict) -> str:
    row = rows_by_name.get(result_key)
    if row is None:
        return ""
    u = fmt_percent(row.union)
    if not u:
        return ""
    flag = f' <span class="st-flag">⚠{escape(row.flag)}</span>' if row.flag else ""
    return f'<span class="st-num">{u}% union · eps {escape(row.eps or "?")}</span>{flag}'


def status_header(status: dict, live) -> str:
    h = status.get("header", {}) or {}
    src = ("live tmux" if live else "status file (tmux unreadable)") if live is not None else \
        "status file (tmux unreadable)"
    upd = status.get("updated", "")

    def row(label, val):
        return (f'<div class="sh-row"><span class="sh-k">{escape(label)}</span>'
                f'<span class="sh-v">{inline_markdown(val)}</span></div>') if val else ""

    return ('<div class="status-header">'
            + row("Phase", h.get("phase", ""))
            + row("Critical path", h.get("critical_path", ""))
            + row("Decision owed (Kiet)", h.get("decision_owed", ""))
            + f'<div class="sh-meta">running-status source: {escape(src)}; status file as of '
              f'{escape(str(upd))}</div></div>')


def _fmt_ts(epoch_secs) -> str:
    if not epoch_secs:
        return ""
    return datetime.fromtimestamp(epoch_secs, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


_CONFIG_EPOCHS = {}


def _config_epochs(cfg_path):
    """Total epochs from the REAL run config (resolves `_base_` inheritance via the
    trainer's own loader) — never hardcoded. None if unreadable."""
    if not cfg_path:
        return None
    if cfg_path in _CONFIG_EPOCHS:
        return _CONFIG_EPOCHS[cfg_path]
    val = None
    try:
        import sys as _sys
        srcp = str(ROOT / "src")
        if srcp not in _sys.path:
            _sys.path.insert(0, srcp)
        from robustdro.utils.io import load_config
        val = int(load_config(str(ROOT / cfg_path))["train"]["epochs"])
    except Exception:
        val = None
    _CONFIG_EPOCHS[cfg_path] = val
    return val


def _live_progress(item: dict):
    """Live (current, total, unit, updated_ts, extra). Total epochs read from the run
    CONFIG (item['config']); current = epoch files of the ACTIVE run among
    item['progress_runs'] (the loss_mats dir with the newest mtime); updated_ts = that
    dir's newest file mtime. No hand-entered progress; denominator is the real config."""
    unit = "epoch"
    total = _config_epochs(item.get("config"))
    runs = item.get("progress_runs")
    if isinstance(runs, str):
        runs = [runs]
    if runs:
        best = None  # (mtime, count, index)
        for i, r in enumerate(runs):
            d = ROOT / "results" / "loss_mats" / r
            files = list(d.glob("ep*.pt")) if d.is_dir() else []
            if not files:
                continue
            newest = max(p.stat().st_mtime for p in files)
            if best is None or newest > best[0]:
                best = (newest, len(files), i)
        if best:
            extra = f"seed {best[2] + 1}/{len(runs)}" if len(runs) > 1 else ""
            return best[1], total, unit, best[0], extra
        return 0, total, unit, None, ""
    prog = item.get("progress") or {}
    return prog.get("current", 0), total or prog.get("total"), unit, None, ""


def _progress_bar(item: dict, queued: bool):
    current, total, unit, ts, extra = _live_progress(item)
    if total:
        pct = max(0.0, min(100.0, 100.0 * (current or 0) / total))
        fill = f'<div class="pfill{" queued" if queued else ""}" style="width:{pct:.0f}%"></div>'
        label = f'{current or 0}/{total} {unit} · {pct:.0f}%' + (f' · {extra}' if extra else "")
    elif queued:
        fill = '<div class="pfill queued" style="width:100%"></div>'
        label = 'queued'
    else:
        return "", ts
    return (f'<div class="pwrap"><div class="pbar">{fill}</div>'
            f'<span class="plabel">{escape(label)}</span></div>'), ts


_STALE_SECS = 900  # >15 min with no new epoch file => not a live "running"


def _status_item_row(it, key, live, status_updated, rows_by_name, position=None):
    tier = it.get("tier")
    tier_html = (f'<span class="st-tier" style="background:{_TIER[tier]}">{_TIER_LABEL[tier]}</span>'
                 if tier in _TIER else "")
    num = _disk_number(it.get("result", ""), rows_by_name) if it.get("result") else ""
    order_html = f'<span class="st-order">#{position}</span>' if position is not None else ""
    bar_html, bar_ts = ("", None)
    if key in ("running", "planned"):
        bar_html, bar_ts = _progress_bar(it, queued=(key == "planned"))
    badge = it.get("badge", key)
    note = ""
    if key == "running":
        # 1) dead process (tmux session gone) -> killed
        if it.get("session") is not None:
            if live is None:
                note = ' <span class="st-note">(tmux unverified)</span>'
            elif it["session"] not in live:
                badge = "killed"
                note = ' <span class="st-note">(session gone)</span>'
        # 2) hung process (session alive but no epoch progress for >15min) -> stale
        if badge != "killed" and bar_ts is not None:
            age = datetime.now(timezone.utc).timestamp() - bar_ts
            if age > _STALE_SECS:
                badge = "stale"
                note += (f' <span class="st-note">(stalled: no progress {int(age // 60)}min, '
                         f'as of {escape(_fmt_ts(bar_ts))})</span>')
    updated = _fmt_ts(bar_ts) or it.get("updated") or status_updated
    upd_html = f'<span class="st-upd">upd {escape(updated)}</span>' if updated else ""
    return (
        '<div class="st-row">'
        f'{order_html}'
        f'<span class="st-badge" style="background:{_BADGE.get(badge, "#64748b")}">{escape(badge)}</span>'
        f'<span class="st-name">{escape(it.get("name", ""))}{note}</span>'
        f'{tier_html}{num}{bar_html}'
        f'<span class="st-detail">{inline_markdown(it.get("detail", ""))}</span>'
        f'{upd_html}'
        '</div>')


def experiment_status(status: dict, rows_by_name: dict, live) -> str:
    if not status:
        return ('<p>No <code>results/run_status.json</code> found — status view is data-driven; '
                'runners/planner write that file.</p>')
    groups = status.get("groups", {}) or {}
    status_updated = str(status.get("updated", ""))
    out = []
    for key, title in _GROUPS:
        items = groups.get(key) or []
        if not items:
            continue
        # Planned group renders in explicit queue ORDER (by `order`, then given order).
        if key == "planned":
            items = sorted(enumerate(items), key=lambda t: (t[1].get("order", 10_000), t[0]))
            rows = [_status_item_row(it, key, live, status_updated, rows_by_name, position=i + 1)
                    for i, (_, it) in enumerate(items)]
            hint = ' <span class="st-hint">(execution order top→bottom)</span>'
        else:
            rows = [_status_item_row(it, key, live, status_updated, rows_by_name) for it in items]
            hint = ""
        out.append(f'<div class="st-group"><h3>{escape(title)}{hint}</h3>{"".join(rows)}</div>')
    return "".join(out) or "<p>run_status.json has no groups.</p>"


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


def main() -> None:
    memory_md = read_text(STATE)                       # PROJECT_STATE.md (consolidated)
    log_md = read_text(LOG) if LOG.exists() else ""     # legacy timeline (optional)
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
  </style>
</head>
<body>
  <header>
    <h1>AttackDRO Dashboard</h1>
    <p>Regenerated from <code>docs/PROJECT_STATE.md</code> and <code>results/*.json</code> (+ <code>run_status.json</code>) at $generated_at.</p>
  </header>
  <main>
    <div class="status">$status_cards</div>
    <section>
      <h2>Experiment status</h2>
      $status_header
      $experiment_status
    </section>
    <section>
      <h2>Union robustness by run (compare view)</h2>
      $compare_view
    </section>
    <section>
      <h2>Results</h2>
      $results_table
    </section>
    <div class="grid">
      <section>
        <h2>Findings</h2>
        $findings
      </section>
      <section>
        <h2>Open Decisions</h2>
        $open_items
      </section>
    </div>
    <section>
      <h2>Decision History</h2>
      <div class="timeline">$timeline</div>
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
    rows_by_name = {r.name: r for r in rows}
    run_status = load_run_status()
    live = live_tmux_sessions()
    html_page = template.substitute(
        generated_at=escape(generated_at),
        status_cards=status_cards(memory_md),
        status_header=status_header(run_status, live),
        experiment_status=experiment_status(run_status, rows_by_name, live),
        compare_view=compare_view(rows),
        results_table=results_table(rows),
        findings=render_fragment(extract_section_like(memory_md, "FINDINGS")),
        open_items=render_fragment(extract_section_like(memory_md, "CURRENT PHASE")),
        timeline=timeline_entries(log_md),
    )

    OUT.write_text(html_page, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
