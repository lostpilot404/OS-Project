"""Assemble the self-contained teacher dashboard HTML.

The document is generated once, from validated data, and opened directly from disk: all
styles, scripts and charts are inline and nothing is fetched at view time. Content is
rendered as ordinary HTML first, so the page still tells the whole story with JavaScript
disabled; a small script then adds the tab switching, the policy selector and the
"inspect this dispatch" read-out.

Every dynamic string is escaped with :func:`dashboard.charts.esc` before insertion, and
the embedded JSON payload is escaped so that it cannot close its own ``<script>`` element.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dashboard.charts import (
    ACCENT,
    BAD,
    LEARNED,
    MUTED,
    OK,
    POLICY_COLORS,
    WARN,
    BarItem,
    GanttRow,
    LifecycleStage,
    Marker,
    esc,
    gantt_chart,
    horizontal_bars,
    interval_rows,
    lifecycle_flow,
    research_bars,
    stage_colours,
    verdict_tone_key,
    verification_band,
)
from dashboard.model import DemoDocument, Fixture
from dashboard.research import ResearchSummary
from dashboard.teaching import TeachingPanel, TeachingPolicy

__all__ = ["render_dashboard", "format_number", "format_signed"]

_MISSING = "n/a"

_TEACHING_RULES = {
    "FCFS": "smallest (arrival, pid) among ready jobs: strict arrival order",
    "SJF": "smallest (burst, arrival, pid): shortest job first, non-preemptive",
    "Priority": "smallest (priority, arrival, pid): lower number runs first",
    "Round Robin": "FIFO ready queue, one quantum per turn, arrivals precede a requeue",
}

_VERDICT_TONE = {"accepted": OK, "rejected": BAD, "inconclusive": WARN, "held": MUTED}


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------


def format_number(value: Optional[float], digits: int = 2) -> str:
    """Format a number for display; ``None`` and non-finite values print as ``n/a``."""
    if value is None:
        return _MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _MISSING
    if not math.isfinite(number):
        return _MISSING
    if abs(number) < 0.5 * (10 ** -digits):
        number = 0.0
    return f"{number:.{digits}f}"


def format_signed(value: Optional[float], digits: int = 2) -> str:
    """Format a signed difference with an explicit sign; an exact zero keeps none."""
    if value is None:
        return _MISSING
    number = float(value)
    if abs(number) < 0.5 * (10 ** -digits):
        return f"{0.0:.{digits}f}"
    return f"{number:+.{digits}f}"


def _int_text(value: Optional[int]) -> str:
    return _MISSING if value is None else str(int(value))


def _percent(value: Optional[float], digits: int = 2) -> str:
    return _MISSING if value is None else f"{format_number(value, digits)}%"


def _pill(label: str, tone: str = ACCENT) -> str:
    return (
        f'<span class="pill" style="--pill-tone:{tone}">{esc(label)}</span>'
    )


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric_from: int = 1,
    numeric_to: Optional[int] = None,
    caption: Optional[str] = None,
    footer: Optional[Sequence[str]] = None,
) -> str:
    """Render an HTML table with the columns in ``[numeric_from, numeric_to)`` right aligned.

    Cell contents are expected to be escaped already.
    """

    def css(index: int) -> str:
        end = len(headers) if numeric_to is None else numeric_to
        return ' class="num"' if numeric_from <= index < end else ""

    parts: List[str] = ['<div class="table-wrap"><table class="data-table">']
    if caption:
        parts.append(f'<caption class="table-caption">{esc(caption)}</caption>')
    parts.append("<thead><tr>")
    for index, header in enumerate(headers):
        parts.append(f"<th{css(index)} scope=\"col\">{esc(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for index, cell in enumerate(row):
            parts.append(f"<td{css(index)}>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody>")
    if footer:
        parts.append("<tfoot><tr>")
        for index, cell in enumerate(footer):
            parts.append(f"<td{css(index)}>{cell}</td>")
        parts.append("</tr></tfoot>")
    parts.append("</table></div>")
    return "".join(parts)


def _legend(fixture: Fixture) -> str:
    used = []
    for interval in fixture.intervals:
        if interval.policy not in used:
            used.append(interval.policy)
    items = [
        f'<span class="legend-item"><span class="swatch" style="background:{POLICY_COLORS.get(policy, MUTED)}"></span>{esc(policy)} service</span>'
        for policy in used
    ]
    items.append(
        '<span class="legend-item"><span class="swatch swatch-hatch"></span>context switch (configured cost)</span>'
    )
    items.append(
        '<span class="legend-item"><span class="swatch swatch-arrival"></span>arrival</span>'
        '<span class="legend-item"><span class="swatch swatch-completion"></span>completion</span>'
    )
    return f'<p class="legend">{"".join(items)}</p>'


# --------------------------------------------------------------------------------------
# Page pieces
# --------------------------------------------------------------------------------------


_STYLE = """
:root{
  --ink:#16212b; --ink-soft:#455666; --muted:#5c6b7a;
  --paper:#ffffff; --page:#edf1f5; --panel:#f6f8fb; --line:#d5dde6;
  --accent:#1f5f8b; --accent-soft:#e9f1f7;
  --ok:#2f6f4f; --ok-soft:#eaf3ee;
  --warn:#a8691b; --warn-soft:#fbf1e2;
  --bad:#a03d3d; --bad-soft:#f9ebeb;
  --fcfs:#3d5a80; --sjf:#2f6f4f; --rr:#9a6b1f; --prio:#6b4a86;
}
*{box-sizing:border-box}
html{font-size:16px;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);
  font-family:"Segoe UI","Inter","Helvetica Neue",Arial,system-ui,sans-serif;
  line-height:1.55;font-variant-numeric:tabular-nums}
.page{max-width:1280px;margin:0 auto;padding:26px 26px 60px}
h1,h2,h3,h4{line-height:1.25;margin:0}
h1{font-size:2rem;letter-spacing:-0.01em}
h2{font-size:1.32rem;margin-top:0}
h3{font-size:1.05rem;margin:22px 0 6px}
h4{font-size:0.95rem;margin:18px 0 6px}
p{margin:0 0 10px}
a{color:var(--accent);text-underline-offset:2px}
code,.mono{font-family:"Cascadia Mono","Consolas","Menlo",monospace;font-size:0.9em}
.skip-link{position:absolute;left:-9999px;top:0;background:var(--ink);color:#fff;padding:8px 14px;z-index:10}
.skip-link:focus{left:12px;top:12px}
.masthead{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:26px 30px;margin-bottom:18px}
.eyebrow{margin:0 0 6px;font-size:0.78rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent);font-weight:700}
.lede{font-size:1.05rem;max-width:88ch;color:var(--ink-soft);margin-top:10px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px 22px;margin:20px 0 4px;padding:16px 0 0;border-top:1px solid var(--line)}
.facts div{margin:0}
.facts dt{font-size:0.74rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);font-weight:700}
.facts dd{margin:2px 0 0;font-size:0.92rem}
.scope{margin:16px 0 0;padding:12px 14px;background:var(--panel);border-left:3px solid var(--accent);border-radius:4px;font-size:0.92rem;color:var(--ink-soft)}
.section-nav{display:flex;flex-wrap:wrap;gap:6px 4px;margin:0 0 14px;padding:0;list-style:none}
.section-nav a{display:inline-block;padding:5px 10px;border-radius:999px;background:var(--paper);border:1px solid var(--line);color:var(--ink-soft);text-decoration:none;font-size:0.85rem}
.section-nav a:hover,.section-nav a:focus-visible{border-color:var(--accent);color:var(--accent)}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 16px}
.tab{font:inherit;font-size:0.9rem;padding:7px 14px;border-radius:999px;border:1px solid var(--line);background:var(--paper);color:var(--ink-soft);cursor:pointer}
.tab:hover{border-color:var(--accent);color:var(--accent)}
.tab[aria-selected="true"]{background:var(--ink);border-color:var(--ink);color:#fff;font-weight:600}
.section{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:26px 30px;margin-bottom:18px;box-shadow:0 1px 2px rgba(22,33,43,0.04)}
.section h2{display:flex;align-items:baseline;gap:10px}
.section .index{font-size:0.8rem;font-weight:700;letter-spacing:0.1em;color:var(--muted)}
.intro{color:var(--ink-soft);max-width:92ch;margin-bottom:16px}
.scenario-panel{border-top:1px solid var(--line);padding-top:16px;margin-top:16px}
.scenario-panel:first-of-type{border-top:0;padding-top:0;margin-top:0}
.scenario-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:4px}
.scenario-head h3{margin:0}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:0.76rem;font-weight:700;letter-spacing:0.02em;
  color:var(--pill-tone);background:color-mix(in srgb,var(--pill-tone) 12%,#fff);border:1px solid color-mix(in srgb,var(--pill-tone) 40%,#fff)}
.pill.neutral{color:var(--ink-soft);background:var(--panel);border-color:var(--line)}
.chart{width:100%;height:auto;display:block;margin:10px 0 6px}
.chart text{font-family:inherit}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:0.82rem;color:var(--ink-soft);margin:6px 0 14px}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.swatch{display:inline-block;width:14px;height:10px;border-radius:2px}
.swatch-hatch{background:repeating-linear-gradient(45deg,#eef1f4 0 3px,#b9c4d1 3px 5px);border:1px solid var(--line)}
.swatch-arrival{width:10px;height:10px;border-radius:50%;background:#fff;border:1.6px solid var(--muted)}
.swatch-completion{width:2px;height:12px;border-radius:0;background:var(--ink)}
.caption{font-size:0.86rem;color:var(--muted);margin:4px 0 12px}
.callout{padding:12px 16px;border-radius:6px;background:var(--panel);border-left:4px solid var(--accent);margin:14px 0;font-size:0.92rem;color:var(--ink-soft)}
.callout.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.callout.ok{border-left-color:var(--ok);background:var(--ok-soft)}
.callout.bad{border-left-color:var(--bad);background:var(--bad-soft)}
.callout strong{color:var(--ink)}
.table-wrap{overflow-x:auto;margin:8px 0 14px}
.data-table{width:100%;border-collapse:collapse;font-size:0.84rem}
.data-table caption{caption-side:top;text-align:left;font-weight:700;color:var(--ink-soft);padding:0 0 6px}
.data-table th,.data-table td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
.data-table th{background:var(--panel);font-size:0.76rem;letter-spacing:0.04em;text-transform:uppercase;color:var(--muted)}
.data-table tbody tr:hover{background:#fafcfe}
.data-table .num{text-align:right;font-variant-numeric:tabular-nums}
.data-table tfoot td{font-weight:700;border-top:2px solid var(--line);border-bottom:0}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px 26px;margin:10px 0 4px}
.detail-grid dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:0.86rem}
.detail-grid dt{color:var(--muted)}
.detail-grid dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
details{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0;background:#fcfdfe}
details summary{cursor:pointer;font-weight:600;color:var(--accent);font-size:0.9rem}
details summary::marker{color:var(--muted)}
details[open] summary{margin-bottom:8px}
.readout{min-height:2.6em;padding:9px 12px;background:var(--panel);border:1px dashed var(--line);border-radius:6px;font-size:0.88rem;color:var(--ink-soft)}
.readout strong{color:var(--ink)}
.slice{cursor:pointer;outline-offset:2px}
.slice:focus-visible rect,.slice.is-selected rect{stroke:#16212b;stroke-width:2}
.metrics-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:6px 0 16px}
.metric{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--panel)}
.metric dt{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);font-weight:700}
.metric dd{margin:3px 0 0;font-size:1.25rem;font-weight:700;line-height:1.2}
.metric .sub{display:block;font-size:0.78rem;font-weight:500;color:var(--muted)}
.finding{margin:0 0 10px;padding-left:20px;position:relative}
.finding::before{content:"";position:absolute;left:2px;top:0.55em;width:7px;height:7px;border-radius:50%;background:var(--accent)}
.grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px 24px}
.note{font-size:0.82rem;color:var(--muted)}
.noscript{margin:0 0 16px;padding:10px 14px;border:1px solid var(--warn);background:var(--warn-soft);border-radius:6px;font-size:0.9rem}
footer{margin-top:26px;color:var(--muted);font-size:0.8rem;line-height:1.7}
footer strong{color:var(--ink-soft)}
[data-panel-mode],[data-panel-scenario],[data-panel-teaching]{display:block}
.js [data-panel-mode]:not(.is-active),
.js [data-panel-scenario]:not(.is-active),
.js [data-panel-teaching]:not(.is-active){display:none}
@media (max-width:720px){.page{padding:16px 12px 40px}.section{padding:18px 16px}}
@media print{
  body{background:#fff}
  .page{max-width:none;padding:0}
  .section,.masthead{box-shadow:none;break-inside:avoid;page-break-inside:avoid;border-color:#bbb}
  .tabs,.section-nav,.readout-hint{display:none!important}
  .js [data-panel-mode]:not(.is-active),
  .js [data-panel-scenario]:not(.is-active),
  .js [data-panel-teaching]:not(.is-active){display:block!important}
  a{color:inherit;text-decoration:none}
}
"""

_SCRIPT = """
(function () {
  "use strict";
  function all(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }
  function setSelected(group, active) {
    all('[data-' + group + '-target]').forEach(function (button) {
      var on = button.getAttribute('data-' + group + '-target') === active;
      button.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    all('[data-panel-' + group + ']').forEach(function (panel) {
      var name = panel.getAttribute('data-panel-' + group);
      panel.classList.toggle('is-active', name === active || name === 'all' || active === 'all');
    });
  }
  function wireGroup(group) {
    var buttons = all('[data-' + group + '-target]');
    if (!buttons.length) { return; }
    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        setSelected(group, button.getAttribute('data-' + group + '-target'));
      });
    });
    var initial = buttons.filter(function (button) {
      return button.getAttribute('aria-selected') === 'true';
    })[0] || buttons[0];
    setSelected(group, initial.getAttribute('data-' + group + '-target'));
  }
  function attribute(block, name) {
    var value = block.getAttribute(name);
    return value === null || value === '' ? null : value;
  }
  function describe(block) {
    var pid = block.getAttribute('data-pid');
    var duration = block.getAttribute('data-duration');
    var text = 'P' + pid + ' · ' + block.getAttribute('data-policy') +
      ' · t=' + block.getAttribute('data-start') + ' to t=' + block.getAttribute('data-end') +
      ' · ' + duration + ' time unit' + (duration === '1' ? '' : 's');
    var burst = attribute(block, 'data-burst');
    if (burst !== null) {
      text += ' · burst ' + burst + ', priority ' + attribute(block, 'data-priority') +
        ', arrival t=' + attribute(block, 'data-arrival');
    }
    var switching = attribute(block, 'data-switch');
    if (switching !== null) {
      text += ' · context switch before: ' + switching + ' unit' + (switching === '1' ? '' : 's');
    }
    var waiting = attribute(block, 'data-waiting');
    if (waiting !== null) {
      text += ' · queue waiting accrued during the interval: ' + waiting;
    }
    var processWaiting = attribute(block, 'data-process-waiting');
    if (processWaiting !== null) {
      text += ' · process total waiting: ' + processWaiting + ' (completion − arrival − burst)';
    }
    return text;
  }
  function init() {
    document.documentElement.classList.add('js');
    try {
    wireGroup('mode');
    wireGroup('scenario');
    wireGroup('teaching');
    all('[data-jump-mode]').forEach(function (link) {
      link.addEventListener('click', function () {
        var target = link.getAttribute('data-jump-mode');
        if (target) { setSelected('mode', target); }
      });
    });
    document.addEventListener('click', function (event) {
      var block = event.target.closest ? event.target.closest('.slice') : null;
      if (!block) { return; }
      var seed = block.getAttribute('data-seed');
      var readout = document.querySelector('[data-readout="' + seed + '"]');
      all('.slice.is-selected').forEach(function (other) { other.classList.remove('is-selected'); });
      block.classList.add('is-selected');
      if (readout) {
        readout.textContent = describe(block);
      }
    });
    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter' && event.key !== ' ') { return; }
      var block = event.target.closest ? event.target.closest('.slice') : null;
      if (!block) { return; }
      event.preventDefault();
      block.click();
    });
    document.addEventListener('focusin', function (event) {
      var block = event.target.closest ? event.target.closest('.slice') : null;
      if (!block) { return; }
      var seed = block.getAttribute('data-seed');
      var readout = document.querySelector('[data-readout="' + seed + '"]');
      if (readout && !readout.getAttribute('data-locked')) {
        readout.textContent = describe(block);
      }
    });
    } catch (error) {
      // Never leave a half-initialised page hiding content: without the class every
      // section is rendered in full.
      document.documentElement.classList.remove('js');
      if (window.console && window.console.warn) {
        window.console.warn('Dashboard interaction disabled:', error);
      }
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
"""


def _masthead(doc: DemoDocument, research: Optional[ResearchSummary], provenance: Mapping[str, str]) -> str:
    seeds = ", ".join(str(fixture.seed) for fixture in doc.fixtures)
    verdicts = "; ".join(
        f"seed {fixture.seed}: {fixture.primary_incident.verdict if fixture.primary_incident else 'no incident'} "
        f"-> {fixture.final_policy}"
        for fixture in doc.fixtures
    )
    scheduler = doc.scheduler
    research_fact = (
        f"held-out {research.split.replace('_', '-')} split · {research.workloads} workloads · seed {research.final_test_seed}"
        if research and research.workloads is not None
        else ("not loaded" if research is None else research.split)
    )
    return f"""
<header class="masthead">
  <p class="eyebrow">Self-healing CPU scheduling · classroom dashboard</p>
  <h1>One CPU, recorded decisions, and the evidence behind them</h1>
  <p class="lede">
    Every chart on this page is drawn from output that the project's own simulator has already
    produced: the two <code>main.py demo</code> scenarios and the committed held-out study tables.
    Nothing is animated or simulated in the browser, and no value is typed into the page by hand.
    The sections below <strong>replay recorded runs</strong>; they do not re-run the scheduler.
  </p>
  <dl class="facts">
    <div><dt>Demo input</dt><dd><code>{esc(provenance.get("input_name", "demo.json"))}</code><br>sha256 <code>{esc(provenance.get("input_sha256_short", "n/a"))}</code></dd></div>
    <div><dt>Scenarios</dt><dd>Seeds {esc(seeds)}<br>{esc(verdicts)}</dd></div>
    <div><dt>Scheduler</dt><dd>single CPU · quantum {esc(_int_text(int(scheduler.get("round_robin_quantum", 0))))} · switch cost {esc(_int_text(int(scheduler.get("switching_cost", 0))))}</dd></div>
    <div><dt>Research evidence</dt><dd>{esc(research_fact)}</dd></div>
  </dl>
  <p class="scope">
    <strong>What this is.</strong> A single-CPU discrete-event simulation with synthetic processes.
    It is not a Linux kernel scheduler, not a hardware benchmark and not a production service.
    Time is measured in simulated time units, never seconds.
  </p>
</header>
"""


def _nav() -> str:
    items = (
        ("overview", "Overview", "demo"),
        ("scheduling", "Scheduling timeline", "demo"),
        ("teaching", "Worked example", "demo"),
        ("lifecycle", "Self-healing lifecycle", "demo"),
        ("performance", "Performance", "demo"),
        ("research", "Research results", "research"),
        # The limits section is visible in every mode, so its link only scrolls.
        ("limits", "Limits", ""),
    )
    links = "".join(
        f'<li><a href="#{anchor}"'
        + (f' data-jump-mode="{mode}"' if mode else "")
        + f">{esc(label)}</a></li>"
        for anchor, label, mode in items
    )
    return f'<nav aria-label="Page sections"><ul class="section-nav">{links}</ul></nav>'


def _mode_tabs() -> str:
    return """
<div class="tabs" role="tablist" aria-label="Evidence source">
  <button class="tab" type="button" role="tab" aria-selected="true" data-mode-target="demo">Short demo · two fixtures</button>
  <button class="tab" type="button" role="tab" aria-selected="false" data-mode-target="research">Research results</button>
  <button class="tab" type="button" role="tab" aria-selected="false" data-mode-target="all">Everything</button>
</div>
"""


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------


def _section_overview(doc: DemoDocument, research: Optional[ResearchSummary], provenance: Mapping[str, str]) -> str:
    healing = doc.healing
    rows = [
        [
            esc(fixture.family),
            str(fixture.seed),
            f"<code>{esc(fixture.fingerprint)}</code>",
            str(fixture.metrics.num_processes),
            format_number(fixture.metrics.avg_waiting_time),
            format_number(fixture.fcfs_metrics.avg_waiting_time),
            format_signed(fixture.mean_waiting_difference),
            esc(
                fixture.primary_incident.verdict if fixture.primary_incident else "no incident"
            ),
            esc(fixture.final_policy),
        ]
        for fixture in doc.fixtures
    ]
    threshold_rows = [
        ["Detection threshold", f"reference + {format_number(healing.get('threshold_band_multiplier'))} x dispersion band"],
        ["Hysteresis", f"{_int_text(healing.get('hysteresis_evaluations'))} consecutive suspect evaluations"],
        ["Verification window", f"{_int_text(healing.get('verify_decisions'))} decisions / {_int_text(healing.get('verify_completions'))} completions (max {_int_text(healing.get('verify_max_decisions'))})"],
        ["Accept / reject residual", f">= {format_signed(healing.get('accept_residual_factor'))} accept; <= {format_signed(healing.get('reject_residual_factor'))} reject; otherwise inconclusive"],
        ["Inconclusive posture", "roll back" if healing.get("inconclusive_rollback") else "keep the change"],
        ["Safety fallback", f"after {_int_text(healing.get('max_consecutive_failures'))} consecutive confirmed failures"],
    ]
    return f"""
<section class="section" id="overview" data-panel-mode="demo">
  <h2><span class="index">00</span> What you are looking at</h2>
  <p class="intro">
    The page has two kinds of evidence. The <strong>demo</strong> sections replay the two fixtures written by
    <code>main.py demo</code> — real inputs, dispatches, incidents and metrics from the simulator, chosen to show
    one inconclusive and one accepted recovery. The <strong>research</strong> section reads the committed
    held-out study tables. They answer different questions and are never mixed.
  </p>
  {_table(
      ("Scenario", "Seed", "Fingerprint", "Procs", "Mean wait", "FCFS mean wait", "Difference", "Verdict", "Final policy"),
      rows,
      numeric_from=3,
      numeric_to=7,
      caption="The two recorded fixtures (simulated time units; lower waiting is better)",
  )}
  <div class="grid-2">
    <div>
      <h3>Declared thresholds used by both runs</h3>
      <p class="note">Read from <code>demo.json → healing</code>, i.e. the configuration the engine actually used.</p>
      {_table(("Setting", "Recorded value"), [[esc(a), esc(b)] for a, b in threshold_rows], numeric_from=2)}
    </div>
    <div>
      <h3>Provenance</h3>
      <div class="detail-grid">
        <dl>
          <dt>Demo file</dt><dd><code>{esc(provenance.get("input_name", "demo.json"))}</code></dd>
          <dt>Demo sha256 (first 16 hex)</dt><dd><code>{esc(provenance.get("input_sha256_short", "n/a"))}</code></dd>
          <dt>Scope line</dt><dd>{esc(doc.scope)}</dd>
          <dt>Research file</dt><dd><code>{esc(research.source_name if research else "not loaded")}</code></dd>
          <dt>Comparisons file</dt><dd><code>{esc(research.comparisons_source if research and research.comparisons_source else "not loaded")}</code></dd>
          <dt>Interactive offline</dt><dd>yes — no server, CDN or network</dd>
        </dl>
      </div>
      <p class="note">
        Units are simulated time units. Waiting = completion − arrival − burst; turnaround = completion − arrival;
        response = first service − arrival. CPU utilisation is busy time / elapsed time.
      </p>
    </div>
  </div>
</section>
"""


# --------------------------------------------------------------------------------------
# Scheduling timeline
# --------------------------------------------------------------------------------------


def _scenario_tabs(doc: DemoDocument, *, note: bool = True) -> str:
    """The scenario selector; repeated per section and kept in sync by the page script."""
    buttons = []
    for index, fixture in enumerate(doc.fixtures):
        incident = fixture.primary_incident
        verdict = incident.verdict if incident else "no incident"
        selected = "true" if index == 0 else "false"
        buttons.append(
            f'<button class="tab" type="button" role="tab" aria-selected="{selected}" '
            f'data-scenario-target="{fixture.seed}">Seed {fixture.seed} · {esc(verdict)}</button>'
        )
    tabs = '<div class="tabs" role="tablist" aria-label="Demo scenario">' + "".join(buttons) + "</div>"
    if note:
        tabs += (
            '<p class="note">Both scenarios are in the page; the selector changes which one is shown, '
            "and it stays in step across the timeline, lifecycle and performance sections.</p>"
        )
    return tabs


def _scenario_summary(fixture: Fixture) -> str:
    incident = fixture.primary_incident
    pills = []
    if incident:
        pills.append(_pill(f"verdict: {incident.verdict}", _VERDICT_TONE.get(incident.verdict, MUTED)))
        if incident.rollback_target_policy:
            pills.append(_pill(f"rolled back to {incident.rollback_target_policy}", WARN))
        elif incident.verdict == "accepted":
            pills.append(_pill(f"{incident.settings_after_policy} kept in force", OK))
    pills.append(_pill(f"final policy: {fixture.final_policy}", ACCENT))
    pills.append(
        f'<span class="pill neutral">mean wait {format_number(fixture.metrics.avg_waiting_time)} vs FCFS '
        f'{format_number(fixture.fcfs_metrics.avg_waiting_time)}</span>'
    )
    return f'<div class="scenario-head"><h3>Seed {fixture.seed} · {esc(fixture.family)} workload · {len(fixture.outcomes)} processes</h3>{"".join(pills)}</div>'


def _timeline_caption(fixture: Fixture) -> str:
    incident = fixture.primary_incident
    if incident is None:
        return (
            f"The trace ends under {esc(fixture.final_policy)}. No degradation was declared in this fixture, "
            "so there is no intervention to show."
        )
    changed = [interval for interval in fixture.intervals if interval.policy == incident.settings_after_policy]
    applied = ""
    if changed and incident.action_changed:
        first = changed[0]
        applied = (
            f" The change takes effect at t={first.start}, after the {first.switch_before}-unit context "
            "switch that separates the two policies."
        )
    if incident.rollback_target_policy:
        ending = (
            f"At t={incident.window_end_time} the window closes {incident.verdict} and "
            f"{esc(incident.rollback_target_policy)} is restored; every later dispatch runs under "
            f"{esc(fixture.final_policy)}."
        )
    elif incident.verdict == "accepted":
        ending = (
            f"At t={incident.window_end_time} the window closes accepted, so "
            f"{esc(incident.settings_after_policy)} stays in force to the end of the trace."
        )
    else:
        ending = f" The trace ends under {esc(fixture.final_policy)}."
    return (
        f"Recorded dispatch intervals, coloured by the policy in force. Degradation is detected at "
        f"t={incident.simulated_time} (decision {incident.decision_index}) while "
        f"{esc(incident.settings_before_policy)} is dispatching.{applied} {ending} "
        f"Idle time in this trace: {fixture.total_idle} unit(s); recorded switch time: "
        f"{fixture.total_switch_time} unit(s)."
    )


def _gantt_section(fixture: Fixture) -> str:
    incident = fixture.primary_incident
    markers = []
    if incident is not None:
        markers.append(Marker(incident.simulated_time, f"degradation detected at t={incident.simulated_time}", WARN))
        if incident.window_end_time is not None:
            label = (
                f"rollback at t={incident.window_end_time}"
                if incident.rollback_target_policy
                else f"verification closes at t={incident.window_end_time}"
            )
            markers.append(Marker(incident.window_end_time, label, MUTED))
    chart = gantt_chart(
        name=f"gantt-{fixture.seed}",
        title=f"Recorded CPU execution timeline, seed {fixture.seed}",
        description=(
            "Gantt chart of the dispatch intervals recorded by the simulator: the CPU lane shows the "
            "exact service sequence, the lower lanes show each process's own slices, and the colour "
            "identifies the policy in force."
        ),
        rows=interval_rows(fixture),
        intervals=fixture.intervals,
        segments=fixture.policy_segments,
        domain_end=fixture.domain_end,
        markers=tuple(markers),
        seed=fixture.seed,
    )
    return f"""
{_scenario_summary(fixture)}
{chart}
{_legend(fixture)}
<p class="caption">{esc(_timeline_caption(fixture))}</p>
<div class="readout" data-readout="{fixture.seed}" aria-live="polite">
  <span class="readout-hint">Select a coloured block (or tab to one and press Enter) to inspect that dispatch interval.</span>
</div>
"""


def _workload_table(fixture: Fixture) -> str:
    rows = []
    for process in sorted(fixture.processes, key=lambda item: item.pid):
        outcome = fixture.process_row(process)
        rows.append(
            [
                f"P{process.pid}",
                str(process.arrival_time),
                str(process.burst_time),
                str(process.priority),
                str(outcome.start_time),
                str(outcome.completion_time),
                str(outcome.waiting_time),
                str(outcome.turnaround_time),
                str(outcome.response_time),
            ]
        )
    footer = [
        "mean",
        "—",
        "—",
        "—",
        "—",
        "—",
        format_number(fixture.mean_waiting_from_outcomes),
        format_number(sum(row.turnaround_time for row in fixture.outcomes) / len(fixture.outcomes)),
        format_number(sum(row.response_time for row in fixture.outcomes) / len(fixture.outcomes)),
    ]
    return _table(
        ("Process", "Arrival", "Burst", "Priority", "First start", "Completion", "Waiting", "Turnaround", "Response"),
        rows,
        numeric_from=1,
        caption=f"Seed {fixture.seed}: per-process outcome recorded by the engine (simulated time units)",
        footer=footer,
    )


def _interval_table(fixture: Fixture) -> str:
    rows = []
    for interval in fixture.intervals:
        rows.append(
            [
                str(interval.index),
                str(interval.start - interval.switch_before),
                str(interval.start),
                f"P{interval.pid}",
                esc(interval.policy),
                f"[{interval.start}, {interval.end})",
                str(interval.duration),
                str(interval.switch_before),
                str(interval.idle_before),
                _int_text(interval.waiting_increment),
                ", ".join(f"P{pid}" for pid in interval.completed_pids) or "—",
            ]
        )
    return _table(
        ("#", "Decision t", "Start", "Ran", "Policy", "Interval", "Units", "Switch", "Idle", "Wait accrued", "Completed"),
        rows,
        numeric_from=1,
        caption="Every recorded dispatch interval of this scenario, exactly as stored in demo.json",
    )


def _section_scheduling(doc: DemoDocument) -> str:
    panels = []
    for fixture in doc.fixtures:
        panels.append(
            f'<div class="scenario-panel" data-panel-scenario="{fixture.seed}">'
            f"{_gantt_section(fixture)}"
            f"<h4>Workload and per-process outcome</h4>{_workload_table(fixture)}"
            f"<details><summary>Every recorded dispatch interval ({len(fixture.intervals)})</summary>"
            f"{_interval_table(fixture)}</details>"
            "</div>"
        )
    return f"""
<section class="section" id="scheduling" data-panel-mode="demo">
  <h2><span class="index">01</span> Scheduling timeline <span class="pill neutral">recorded replay</span></h2>
  <p class="intro">
    The CPU lane is the exact sequence of dispatches the engine recorded; each lower lane shows the same
    process's own service slices. Colour marks the policy in force for that dispatch, so the intervention is
    visible as a colour change. Gaps between blocks are the configured 1-unit context switches; no idle gap is
    drawn unless the trace recorded one.
  </p>
  {_scenario_tabs(doc)}
  {"".join(panels)}
</section>
"""


# --------------------------------------------------------------------------------------
# Worked example
# --------------------------------------------------------------------------------------


def _teaching_panel(panel: TeachingPanel, policy: TeachingPolicy, index: int) -> str:
    rows = [
        GanttRow(
            pid=outcome.pid,
            arrival_time=outcome.arrival_time,
            completion_time=outcome.completion_time,
            burst_time=outcome.burst_time,
            priority=outcome.priority,
            waiting_time=outcome.waiting_time,
        )
        for outcome in policy.outcomes
    ]
    readout_key = 100 + index
    chart = gantt_chart(
        name=f"teaching-{index}",
        title=f"{policy.name} on the three-process worked example",
        description=f"Service intervals of the {policy.name} policy on the documented three-process example.",
        rows=rows,
        intervals=policy.intervals,
        segments=(),
        domain_end=policy.domain_end,
        markers=(),
        show_policy_row=False,
        width=800,
        seed=readout_key,
    )
    rows_html = [
        [
            f"P{outcome.pid}",
            str(outcome.arrival_time),
            str(outcome.burst_time),
            str(outcome.priority),
            str(outcome.start_time),
            str(outcome.completion_time),
            str(outcome.turnaround_time),
            str(outcome.waiting_time),
            str(outcome.response_time),
        ]
        for outcome in sorted(policy.outcomes, key=lambda outcome: outcome.pid)
    ]
    return f"""
<div class="scenario-panel" data-panel-teaching="{index}">
  <h3>{esc(policy.name)} <span class="pill neutral">mean wait {format_number(policy.mean_waiting_time)}</span>
    <span class="pill neutral">{esc(policy.service_order)}</span></h3>
  <p class="caption">{esc(_TEACHING_RULES.get(policy.name, ""))}</p>
  {chart}
  <div class="readout" data-readout="{readout_key}" aria-live="polite">
    <span class="readout-hint">Select a block to inspect that service interval
      ({policy.name}, quantum {panel.quantum}, switching cost {panel.switching_cost}).</span>
  </div>
  {_table(
      ("Process", "Arrival", "Burst", "Priority", "First start", "Completion", "Turnaround", "Waiting", "Response"),
      rows_html,
      numeric_from=1,
      caption=f"{policy.name}: mean turnaround {format_number(policy.mean_turnaround_time)}, "
              f"mean response {format_number(policy.mean_response_time)}, {policy.context_switches} context switches",
  )}
  <p class="note">Recomputed at build time by <code>scheduler/</code> with quantum {panel.quantum} and switching cost
    {panel.switching_cost}, and checked against the hand calculation in the teacher guide.</p>
</div>
"""


def _section_teaching(panel: TeachingPanel, doc: DemoDocument) -> str:
    items = [
        BarItem(
            label=policy.name,
            value=policy.mean_waiting_time,
            colour=POLICY_COLORS.get(policy.name, MUTED),
            note=f"mean turnaround {format_number(policy.mean_turnaround_time)}",
            strong=policy.name == panel.best_by_waiting,
        )
        for policy in panel.policies
    ]
    chart = horizontal_bars(
        name="teaching-waiting",
        title="Mean waiting time of the four policies on the worked example",
        description="Mean waiting time per process for FCFS, SJF, Priority and Round Robin on the same three processes.",
        items=items,
        unit="Mean waiting time (simulated time units per process)",
        width=800,
    )
    buttons = "".join(
        f'<button class="tab" type="button" role="tab" aria-selected="{"true" if index == 0 else "false"}" '
        f'data-teaching-target="{index}">{esc(policy.name)}</button>'
        for index, policy in enumerate(panel.policies)
    )
    panels = "".join(
        _teaching_panel(panel, policy, index) for index, policy in enumerate(panel.policies)
    )
    best = next(policy for policy in panel.policies if policy.name == panel.best_by_waiting)
    return f"""
<section class="section" id="teaching" data-panel-mode="demo">
  <h2><span class="index">02</span> Worked example: one workload, four policies</h2>
  <p class="intro">
    Three processes arrive together: P1 burst 5 priority 2, P2 burst 3 priority 1, P3 burst 1 priority 3
    (lower priority number runs first). The four policies below are executed by the project's real
    <code>scheduler/</code> classes, not redrawn from a table. Select a policy to compare the service order;
    with JavaScript disabled all four are shown.
  </p>
  <div class="callout">
    <strong>Hand check.</strong> This example is the one used in the classroom script, with quantum
    {panel.quantum} and switching cost {panel.switching_cost} for hand arithmetic. The executed waiting times
    {("match" if panel.hand_verified else "do NOT match")} the documented calculation:
    FCFS 0/5/8, SJF 4/1/0, Priority 3/0/8, Round Robin 4/5/4 for P1/P2/P3.
    The live self-healing demos above use quantum {esc(_int_text(int(doc.scheduler.get("round_robin_quantum", 0))))} and
    switching cost {esc(_int_text(int(doc.scheduler.get("switching_cost", 0))))}, so their waiting times are not
    comparable with this example.
  </div>
  {chart}
  <p class="caption">
    {esc(best.name)} has the lowest mean waiting time here ({format_number(best.mean_waiting_time)} units), but that is a
    property of this three-job arrival pattern, not a general ranking. SJF needs burst times; it is a
    different trade-off, not a free improvement.
  </p>
  <div class="tabs" role="tablist" aria-label="Teaching policy">{buttons}</div>
  {panels}
</section>
"""


# --------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------


def lifecycle_stages(fixture: Fixture) -> Tuple[Tuple[LifecycleStage, ...], str]:
    """Build the eight recorded stages. Card lines stay short; the details carry the rest."""
    incident = fixture.primary_incident
    if incident is None:
        return (), "No incident was declared in this fixture."
    accept, reject = incident.accept_residual_factor, incident.reject_residual_factor
    final_text = (
        f"rolled back to {incident.rollback_target_policy}"
        if incident.rollback_target_policy
        else f"{incident.settings_after_policy} kept in force"
    )
    verify_tone, decision_tone = stage_colours(incident.verdict)
    verdict_key = verdict_tone_key(incident.verdict)
    pre_decisions = [
        interval for interval in fixture.intervals if interval.index < incident.decision_index
    ]
    post_decisions = [
        interval
        for interval in fixture.intervals
        if incident.window_end_time is not None and interval.start >= incident.window_end_time
    ]
    stages = (
        LifecycleStage(
            "Normal scheduling",
            (
                f"{incident.settings_before_policy} in force",
                f"{len(pre_decisions)} dispatches",
                f"to t={incident.simulated_time}",
            ),
            ACCENT,
            detail=(
                f"{incident.settings_before_policy} dispatches the trace until the incident at "
                f"t={incident.simulated_time}: {len(pre_decisions)} recorded intervals before it."
            ),
        ),
        LifecycleStage(
            "Monitoring",
            (
                f"window mean {format_number(incident.window_mean_at_trigger)}",
                f"reference {format_number(incident.reference_at_trigger)}",
                f"band {format_number(incident.band_at_trigger)}",
            ),
            "#2b6f6a",
            tone_key="muted",
            detail=(
                f"Rolling statistic over {incident.window_size_at_trigger} completed processes: mean "
                f"{format_number(incident.window_mean_at_trigger)}, median "
                f"{format_number(incident.window_median_at_trigger)}, maximum "
                f"{format_number(incident.window_max_at_trigger)}; stable reference "
                f"{format_number(incident.reference_at_trigger)} with dispersion band "
                f"{format_number(incident.band_at_trigger)}."
            ),
        ),
        LifecycleStage(
            "Degradation detected",
            (
                f"t={incident.simulated_time}, decision {incident.decision_index}",
                f"threshold {format_number(incident.trigger_threshold)}",
                f"{incident.suspicion_evaluations} evaluations",
            ),
            WARN,
            tone_key="warn",
            detail=(
                f"The window mean {format_number(incident.window_mean_at_trigger)} exceeded the threshold "
                f"{format_number(incident.trigger_threshold)} = reference "
                f"{format_number(incident.reference_at_trigger)} + {format_number(incident.band_at_trigger)}-unit "
                f"band x the declared multiplier, for {incident.suspicion_evaluations} consecutive "
                f"evaluations. Detector reason: {incident.detector_reason}."
            ),
        ),
        LifecycleStage(
            "Diagnosis",
            (
                incident.diagnosis.replace("_", " "),
                f"state {incident.context_state_index}/648",
                f"{incident.ready_count_at_trigger} ready",
            ),
            "#5b5b8a",
            tone_key="muted",
            detail=(
                f"{incident.diagnosis_statement} Matched categories: "
                f"{incident.diagnosis_matched or 'none'}. Visible remaining work at the trigger: "
                f"{incident.visible_remaining_work_at_trigger} units."
            ),
        ),
        LifecycleStage(
            "Selected intervention",
            (
                incident.action_name,
                f"{incident.settings_before_policy} -> {incident.settings_after_policy}",
                "changed" if incident.action_changed else "no change",
            ),
            LEARNED,
            tone_key="accent",
            detail=(
                f"Offered actions: {incident.offered_actions}. Chosen: {incident.action_name} "
                f"({incident.request_note}). Action note: {incident.action_note or 'none'}. "
                f"Configuration: {incident.settings_before_policy} q={incident.settings_before_quantum} -> "
                f"{incident.settings_after_policy} q={incident.settings_after_quantum}."
            ),
        ),
        LifecycleStage(
            "Observation window",
            (
                f"t={incident.simulated_time} -> {_int_text(incident.window_end_time)}",
                f"{incident.verification_decisions} decisions",
                f"{incident.verification_completions} completions",
            ),
            ACCENT,
            tone_key="accent",
            detail=(
                f"The window closed after {incident.verification_decisions} decisions and "
                f"{incident.verification_completions} completions"
                + (" (truncated at the cap)." if incident.window_truncated else ".")
            ),
        ),
        LifecycleStage(
            "Verification",
            (
                incident.verdict,
                f"residual {format_signed(incident.closure_residual, 3)}",
                f"bands {format_signed(reject)} / {format_signed(accept)}",
            ),
            verify_tone,
            tone_key=verdict_key,
            detail=(
                f"Verdict {incident.verdict}; reason {incident.verifier_reason}. Closure fraction "
                f"{format_number(incident.closure_fraction, 3)} against the expected no-action closure "
                f"{format_number(incident.expected_closure, 3)}; residual "
                f"{format_signed(incident.closure_residual, 4)}. Declared boundaries: accept at or above "
                f"{format_signed(accept)}, reject at or below {format_signed(reject)}, otherwise "
                "inconclusive."
            ),
        ),
        LifecycleStage(
            "Final decision",
            (
                final_text,
                f"final policy {fixture.final_policy}",
                "confirmed failure" if incident.confirmed_failure else "no confirmed failure",
            ),
            decision_tone,
            tone_key=verdict_key,
            detail=(
                f"{final_text}. Confirmed failure: {incident.confirmed_failure}. Consecutive confirmed "
                f"failures: {incident.consecutive_failures}. Safe fallback triggered: "
                f"{incident.fallback_triggered}. Post-window dispatches: {len(post_decisions)}; final "
                f"policy {fixture.final_policy}. Reward {format_signed(incident.reward, 4)}; charged cost "
                f"{format_number(incident.cost_band_units, 4)} band units."
            ),
        ),
    )
    return stages, final_text


def _verdict_explainer(fixture: Fixture) -> str:
    incident = fixture.primary_incident
    if incident is None:
        return ""
    accept, reject = incident.accept_residual_factor, incident.reject_residual_factor
    residual = incident.closure_residual
    if incident.verdict == "inconclusive":
        return f"""
<div class="callout warn">
  <strong>Inconclusive is not rejected.</strong> The verifier rejects an intervention only when the residual
  is at or below {format_signed(reject)} — a <em>confirmed failure</em> that counts toward the safety fallback.
  Here the residual is {format_signed(residual, 4)}, inside the null range ({format_signed(reject)} to {format_signed(accept)}),
  so the window could neither verify nor condemn the change. The declared posture still rolls it back
  (a change the controller cannot verify is not kept), but <code>confirmed_failure</code> stays false and the
  failure counter does not advance. The final mean waiting time is measured over the whole trace, which is a
  different question from what one short window could certify.
</div>"""
    if incident.verdict == "accepted":
        return f"""
<div class="callout ok">
  <strong>Accepted.</strong> The residual {format_signed(residual, 4)} is at or above
  {format_signed(accept)}, so the observation window closed more of the trigger gap than the phase-aware
  reference predicted with no intervention. The new configuration ({esc(incident.settings_after_policy)})
  becomes the known-good setting and stays in force to the end of the trace. Acceptance is a bounded
  online evidence decision, not proof that this policy is globally best.
</div>"""
    if incident.verdict == "rejected":
        return f"""
<div class="callout bad">
  <strong>Rejected — a confirmed failure.</strong> The residual {format_signed(residual, 4)} is at or below
  {format_signed(reject)}: the window closed no part of the trigger gap, so the change is reverted and counted
  as a failure. Two consecutive confirmed failures pin the safe fallback configuration and stop adaptation for
  the rest of the trace.
</div>"""
    return ""


def _incident_details(fixture: Fixture) -> str:
    incident = fixture.primary_incident
    if incident is None:
        return '<p class="note">No incident record exists for this fixture.</p>'
    rows = [
        ("Decision index", str(incident.decision_index)),
        ("Trigger time", f"t={incident.simulated_time}"),
        ("Window end", f"t={_int_text(incident.window_end_time)}"),
        ("Window mean / threshold", f"{format_number(incident.window_mean_at_trigger)} / {format_number(incident.trigger_threshold)}"),
        ("Reference / band", f"{format_number(incident.reference_at_trigger)} / {format_number(incident.band_at_trigger)}"),
        ("Excess, band units", format_number(incident.excess_band_units)),
        ("Ready / visible work", f"{incident.ready_count_at_trigger} / {incident.visible_remaining_work_at_trigger}"),
        ("Diagnosis", incident.diagnosis),
        ("Matched categories", incident.diagnosis_matched or "none"),
        ("Context state", f"{incident.context_state_index}/648 (bins {list(incident.feature_bins)})"),
        ("Offered actions", incident.offered_actions),
        ("Chosen action", f"{incident.action_name} (index {incident.action})"),
        ("Policy before → after", f"{incident.settings_before_policy} → {incident.settings_after_policy}"),
        ("Action changed", "yes" if incident.action_changed else "no"),
        ("Epsilon / explored", f"{format_number(incident.epsilon)} / {'yes' if incident.explored else 'no'}"),
        ("Verdict / reason", f"{incident.verdict} / {incident.verifier_reason}"),
        ("Closure fraction", format_number(incident.closure_fraction, 4)),
        ("Expected closure", format_number(incident.expected_closure, 4)),
        ("Residual", format_signed(incident.closure_residual, 4)),
        ("Accept / reject residual", f">= {format_signed(incident.accept_residual_factor)} / <= {format_signed(incident.reject_residual_factor)}"),
        ("Verification decisions", str(incident.verification_decisions)),
        ("Verification completions", str(incident.verification_completions)),
        ("Window truncated", "yes" if incident.window_truncated else "no"),
        ("Pre / trigger / post load", f"{format_number(incident.pre_load_level)} / {format_number(incident.trigger_load_level)} / {format_number(incident.post_load_level)}"),
        ("Load band at trigger", format_number(incident.load_band_at_trigger)),
        ("Load improvement", format_signed(incident.load_improvement)),
        ("Post-window mean", format_number(incident.post_window_mean)),
        ("Waiting improvement", format_signed(incident.improvement)),
        ("Reward", format_signed(incident.reward, 4)),
        ("Cost, band units", format_number(incident.cost_band_units, 4)),
        ("Confirmed failure", "yes" if incident.confirmed_failure else "no"),
        ("Consecutive failures", str(incident.consecutive_failures)),
        ("Safe fallback triggered", "yes" if incident.fallback_triggered else "no"),
        ("Rollback target", incident.rollback_target_policy or "none (configuration kept)"),
        ("Switch overhead", f"{incident.switch_overhead_units} units"),
        ("Final trace policy", fixture.final_policy),
    ]
    body = "".join(
        f"<dt>{esc(label)}</dt><dd>{esc(value)}</dd>" for label, value in rows
    )
    return (
        "<div class=\"detail-grid\"><dl>"
        + body
        + "</dl></div>"
        + f'<p class="note">Diagnosis statement as recorded: “{esc(incident.diagnosis_statement)}”</p>'
    )


def _lifecycle_panel(fixture: Fixture) -> str:
    incident = fixture.primary_incident
    stages, final_text = lifecycle_stages(fixture)
    flow = ""
    band = ""
    if stages:
        flow = lifecycle_flow(
            name=f"lifecycle-{fixture.seed}",
            title=f"Self-healing lifecycle, seed {fixture.seed}",
            description=(
                "The eight recorded stages of the control loop from normal scheduling through detection, "
                "diagnosis, intervention, observation and verification to the final decision."
            ),
            stages=stages,
        )
        band = verification_band(
            name=f"band-{fixture.seed}",
            title=f"Verification residual against the declared boundaries, seed {fixture.seed}",
            description=(
                "The verification bands are the declared residual boundaries; the marker is the residual "
                "recorded for this incident."
            ),
            incident=incident,
        )
    return f"""
<div class="scenario-panel" data-panel-scenario="{fixture.seed}">
  {_scenario_summary(fixture)}
  {flow}
  {band if band else ""}
  {_verdict_explainer(fixture)}
  <div class="detail-grid">
    <div>
      <h4>Recorded trigger evidence</h4>
      <dl>
        <dt>Trigger</dt><dd>t={incident.simulated_time if incident else _MISSING} (decision {incident.decision_index if incident else _MISSING})</dd>
        <dt>Window mean</dt><dd>{format_number(incident.window_mean_at_trigger if incident else None)}</dd>
        <dt>Threshold</dt><dd>{format_number(incident.trigger_threshold if incident else None)}</dd>
        <dt>Reference ± band</dt><dd>{format_number(incident.reference_at_trigger if incident else None)} ± {format_number(incident.band_at_trigger if incident else None)}</dd>
        <dt>Suspicion evaluations</dt><dd>{_int_text(incident.suspicion_evaluations if incident else None)}</dd>
      </dl>
    </div>
    <div>
      <h4>Recorded verification</h4>
      <dl>
        <dt>Verdict</dt><dd>{esc(incident.verdict if incident else _MISSING)}</dd>
        <dt>Reason</dt><dd>{esc(incident.verifier_reason if incident else _MISSING)}</dd>
        <dt>Residual</dt><dd>{format_signed(incident.closure_residual if incident else None, 4)}</dd>
        <dt>Window</dt><dd>{_int_text(incident.verification_decisions if incident else None)} decisions / {_int_text(incident.verification_completions if incident else None)} completions</dd>
        <dt>Decision</dt><dd>{esc(final_text)}</dd>
      </dl>
    </div>
  </div>
  <details><summary>Full incident record (every field saved to demo.json)</summary>{_incident_details(fixture)}</details>
</div>
"""


def _section_lifecycle(doc: DemoDocument) -> str:
    healing = doc.healing
    panels = "".join(_lifecycle_panel(fixture) for fixture in doc.fixtures)
    return f"""
<section class="section" id="lifecycle" data-panel-mode="demo">
  <h2><span class="index">03</span> Self-healing lifecycle</h2>
  <p class="intro">
    One control-loop incident, stage by stage, using the timestamps, measurements and thresholds stored in
    <code>demo.json</code>. The loop never sees the future: every trigger and verification quantity below was
    observable at the moment it was used.
  </p>
  <p class="note">
    Configured window: at least {_int_text(healing.get("verify_decisions"))} decisions and
    {_int_text(healing.get("verify_completions"))} completions, capped at
    {_int_text(healing.get("verify_max_decisions"))} decisions; hysteresis
    {_int_text(healing.get("hysteresis_evaluations"))} evaluations; accept residual ≥
    {format_signed(healing.get("accept_residual_factor"))}, reject ≤ {format_signed(healing.get("reject_residual_factor"))};
    inconclusive posture: {("roll back" if healing.get("inconclusive_rollback") else "keep")}.
  </p>
  {_scenario_tabs(doc)}
  {panels}
</section>
"""


# --------------------------------------------------------------------------------------
# Performance comparison
# --------------------------------------------------------------------------------------


def _metrics_strip(fixture: Fixture) -> str:
    metrics, baseline = fixture.metrics, fixture.fcfs_metrics
    reduction = fixture.mean_waiting_reduction_fraction
    return f"""
<dl class="metrics-strip">
  <div class="metric"><dt>Mean waiting</dt><dd>{format_number(metrics.avg_waiting_time)}
    <span class="sub">healing run · FCFS {format_number(baseline.avg_waiting_time)}</span></dd></div>
  <div class="metric"><dt>Difference</dt><dd>{format_signed(fixture.mean_waiting_difference)}
    <span class="sub">{("%.1f%% lower than FCFS" % (100 * reduction)) if (reduction is not None and reduction >= 0) else (("%.1f%% higher than FCFS" % (-100 * reduction)) if reduction is not None else "n/a")}</span></dd></div>
  <div class="metric"><dt>Mean response</dt><dd>{format_number(metrics.avg_response_time)}
    <span class="sub">FCFS {format_number(baseline.avg_response_time)}</span></dd></div>
  <div class="metric"><dt>CPU utilisation</dt><dd>{_percent(metrics.cpu_utilization)}
    <span class="sub">FCFS {_percent(baseline.cpu_utilization)}</span></dd></div>
  <div class="metric"><dt>Context switches</dt><dd>{str(metrics.context_switches)}
    <span class="sub">FCFS {str(baseline.context_switches)}</span></dd></div>
</dl>
"""


def _performance_panel(fixture: Fixture) -> str:
    metrics, baseline = fixture.metrics, fixture.fcfs_metrics
    waiting_chart = horizontal_bars(
        name=f"perf-wait-{fixture.seed}",
        title=f"Mean waiting time, seed {fixture.seed}",
        description="Mean waiting time of the self-healing run and of the FCFS run on the identical workload.",
        items=(
            BarItem(label="Self-healing run", value=metrics.avg_waiting_time, colour=LEARNED, strong=True),
            BarItem(label="FCFS (same workload)", value=baseline.avg_waiting_time, colour=POLICY_COLORS["FCFS"]),
        ),
        unit="Mean waiting time (simulated time units per process)",
        width=720,
    )
    response_chart = horizontal_bars(
        name=f"perf-response-{fixture.seed}",
        title=f"Mean response time, seed {fixture.seed}",
        description="Mean response time of the self-healing run and of the FCFS run on the identical workload.",
        items=(
            BarItem(label="Self-healing run", value=metrics.avg_response_time, colour=LEARNED, strong=True),
            BarItem(label="FCFS (same workload)", value=baseline.avg_response_time, colour=POLICY_COLORS["FCFS"]),
        ),
        unit="Mean response time (simulated time units per process)",
        width=720,
    )
    reduction = fixture.mean_waiting_reduction_fraction
    comparison_rows = [
        [
            "Mean waiting time",
            format_number(metrics.avg_waiting_time),
            format_number(baseline.avg_waiting_time),
            format_signed(fixture.mean_waiting_difference),
        ],
        [
            "Mean response time",
            format_number(metrics.avg_response_time),
            format_number(baseline.avg_response_time),
            format_signed(metrics.avg_response_time - baseline.avg_response_time),
        ],
        [
            "Mean turnaround time",
            format_number(metrics.avg_turnaround_time),
            format_number(baseline.avg_turnaround_time),
            format_signed(metrics.avg_turnaround_time - baseline.avg_turnaround_time),
        ],
        [
            "Context switches",
            str(metrics.context_switches),
            str(baseline.context_switches),
            format_signed(metrics.context_switches - baseline.context_switches, 0),
        ],
        [
            "CPU utilisation (%)",
            format_number(metrics.cpu_utilization),
            format_number(baseline.cpu_utilization),
            format_signed(metrics.cpu_utilization - baseline.cpu_utilization),
        ],
        [
            "Elapsed time (units)",
            str(metrics.total_elapsed_time),
            str(baseline.total_elapsed_time),
            format_signed(metrics.total_elapsed_time - baseline.total_elapsed_time, 0),
        ],
        [
            "CPU busy time (units)",
            _int_text(metrics.cpu_busy_time),
            _int_text(baseline.cpu_busy_time),
            (
                format_signed(metrics.cpu_busy_time - baseline.cpu_busy_time, 0)
                if metrics.cpu_busy_time is not None and baseline.cpu_busy_time is not None
                else _MISSING
            ),
        ],
        [
            "CPU idle time (units)",
            _int_text(metrics.cpu_idle_time),
            _int_text(baseline.cpu_idle_time),
            (
                format_signed(metrics.cpu_idle_time - baseline.cpu_idle_time, 0)
                if metrics.cpu_idle_time is not None and baseline.cpu_idle_time is not None
                else _MISSING
            ),
        ],
        [
            "Throughput (processes/unit)",
            format_number(metrics.throughput, 4),
            format_number(baseline.throughput, 4),
            format_signed(
                (metrics.throughput - baseline.throughput)
                if metrics.throughput is not None and baseline.throughput is not None
                else None,
                4,
            ),
        ],
    ]
    if reduction is None:
        percent_text = ""
    else:
        percent_text = (
            f" ({format_number(100 * reduction, 1)}% lower)"
            if reduction >= 0
            else f" ({format_number(-100 * reduction, 1)}% higher)"
        )
    return f"""
<div class="scenario-panel" data-panel-scenario="{fixture.seed}">
  {_scenario_summary(fixture)}
  {_metrics_strip(fixture)}
  <div class="grid-2">
    <div>{waiting_chart}</div>
    <div>{response_chart}</div>
  </div>
  {_table(
      ("Metric", "Self-healing run", "FCFS (same workload)", "Difference"),
      comparison_rows,
      numeric_from=1,
      caption=f"Seed {fixture.seed}: all recorded metrics side by side; both runs share fingerprint {fixture.fingerprint}",
  )}
  <p class="caption">
    In this single recorded scenario the self-healing run's mean waiting time is
    {format_number(abs(fixture.mean_waiting_difference))} simulated time units
    {"lower" if fixture.mean_waiting_difference <= 0 else "higher"} than the FCFS run on the identical workload
    {esc(percent_text)}. Context switches match because every dispatch in these traces is non-preemptive.
    <strong>This is one illustrative trace</strong>, not an estimate of general performance superiority; the
    research section below is the evidence for any broader claim.
  </p>
</div>
"""


def _section_performance(doc: DemoDocument) -> str:
    panels = "".join(_performance_panel(fixture) for fixture in doc.fixtures)
    return f"""
<section class="section" id="performance" data-panel-mode="demo">
  <h2><span class="index">04</span> Performance comparison <span class="pill neutral">illustrative</span></h2>
  <p class="intro">
    For each fixture the same workload was also executed with plain FCFS (validated at build time to carry the
    same workload fingerprint). Both sets of numbers are the metrics the demo recorded; the difference is
    computed here from them.
  </p>
  {_scenario_tabs(doc, note=False)}
  {panels}
</section>
"""


# --------------------------------------------------------------------------------------
# Research
# --------------------------------------------------------------------------------------


def _arm_table(research: ResearchSummary) -> str:
    rows = []
    for arm in research.chart_arms:
        rows.append(
            [
                esc(arm.label),
                format_number(arm.mean_waiting_time),
                format_number(arm.p95_waiting_time),
                format_number(arm.mean_response_time),
                format_number(arm.context_switches),
                format_number(arm.incidents),
                format_number(arm.interventions),
            ]
        )
    return _table(
        ("Arm", "Mean waiting", "P95 waiting", "Mean response", "Context switches", "Incidents / trace", "Interventions / trace"),
        rows,
        numeric_from=1,
        caption=f"All arms in {esc(research.source_name)}; means over {research.workloads or 'n/a'} final-test traces per arm",
    )


def _comparison_table(research: ResearchSummary) -> str:
    wanted = (
        ("rule_wait", "fixed_fcfs"),
        ("q_recovery", "fixed_fcfs"),
        ("q_recovery", "rule_wait"),
        ("rule_wait", "fixed_sjf"),
        ("q_recovery", "fixed_sjf"),
    )
    rows = []
    for target, reference in wanted:
        comparison = research.comparison(target, reference)
        if comparison is None:
            rows.append(
                [
                    f"{esc(research.arm(target).label)} vs {esc(research.arm(reference).label)}",
                    format_signed(research.delta(target, reference)),
                    "not in artifact",
                    "—",
                    "point difference only",
                ]
            )
            continue
        if comparison.excludes_zero:
            verdict = (
                "favours " + research.arm(target).label
                if comparison.difference < 0
                else "favours " + research.arm(reference).label
            )
        else:
            verdict = "interval includes zero"
        rows.append(
            [
                f"{esc(research.arm(target).label)} vs {esc(research.arm(reference).label)}",
                format_signed(comparison.difference),
                f"[{format_signed(comparison.ci_low)}, {format_signed(comparison.ci_high)}]",
                "yes" if comparison.excludes_zero else "no",
                esc(verdict),
            ]
        )
    return _table(
        ("Comparison", "Difference", "95% interval", "Excludes zero", "Reading"),
        rows,
        numeric_from=1,
        caption="Difference = target − reference, so a negative value means the target waits less; intervals are the study's family-stratified bootstrap",
    )


def _section_research(research: Optional[ResearchSummary]) -> str:
    if research is None:
        return """
<section class="section" id="research" data-panel-mode="research">
  <h2><span class="index">05</span> Research results</h2>
  <div class="callout warn">The research summary was not loaded for this build, so this panel is intentionally
  empty. Rebuild without <code>--no-research</code>, or pass <code>--research PATH</code>.</div>
</section>
"""
    findings = research.finding_lines()
    split_line = (
        f"{research.split.replace('_', ' ')} split · {research.workloads} held-out workloads per arm · "
        f"declared final-test seed {research.final_test_seed} · training seeds "
        f"{', '.join(str(seed) for seed in research.training_seeds) or 'n/a'}"
    )
    chart = research_bars(
        name="research-waiting",
        title="Mean waiting time by study arm",
        description="Mean waiting time for every arm of the frozen held-out self-healing study.",
        arms=research.chart_arms,
    )
    return f"""
<section class="section" id="research" data-panel-mode="research">
  <h2><span class="index">05</span> Research results <span class="pill neutral">frozen held-out study</span></h2>
  <p class="intro">
    These are the committed findings of the self-healing study, not the two demo fixtures. Every arm was
    evaluated on the same held-out traces; the values are read from <code>{esc(research.source_name)}</code>
    {f"and <code>{esc(research.comparisons_source)}</code>" if research.comparisons_source else ""}.
  </p>
  <p class="note">{esc(split_line)}</p>
  {chart}
  <p class="legend">
    <span class="legend-item"><span class="swatch" style="background:{POLICY_COLORS["FCFS"]}"></span>fixed scheduler</span>
    <span class="legend-item"><span class="swatch" style="background:{POLICY_COLORS["SJF"]}"></span>rule-based recovery</span>
    <span class="legend-item"><span class="swatch" style="background:{LEARNED}"></span>learned recovery</span>
  </p>
  <h3>What the recorded numbers support</h3>
  {"".join(f'<p class="finding">{esc(line)}</p>' for line in findings)}
  <h3>Predeclared paired comparisons</h3>
  {_comparison_table(research)}
  <details><summary>Every arm in the study summary</summary>{_arm_table(research)}</details>
  <div class="callout">
    <strong>Three different things, deliberately kept apart.</strong>
    (1) The <em>self-healing controller</em> is the monitor → detect → diagnose → act → verify → roll back loop
    behind the demo and this study. (2) The <em>waiting-time rule</em> is one recovery policy inside that loop:
    a predeclared mapping, not learning. (3) The <em>runtime Q-learning workflow</em>
    (<code>results/runtime/</code>) is a separate, non-healing experiment that switches among the four dispatch
    policies repeatedly during a trace; it has no verifier and no rollback, and its numbers are not combined
    with the table above.
  </div>
  <div class="callout warn">
    <strong>Scope and limitations.</strong> Synthetic workloads, one CPU, exact burst sizes for the fixed
    baselines (SJF is an oracle, not an equal-information comparison), and a verifier that is a safety filter
    rather than a benefit estimator — most helpful interventions are inconclusive. The evidence therefore does
    <em>not</em> establish Q-learning as better than the rule, nor any claim of kernel integration or
    production performance.
  </div>
</section>
"""


# --------------------------------------------------------------------------------------
# Limits and footer
# --------------------------------------------------------------------------------------


def _section_limits() -> str:
    bullets = (
        "Single-CPU discrete-event simulation with synthetic processes; no I/O, caches, multicore or kernel module.",
        "Time is simulated (arbitrary units). Wall-clock figures in the study are not part of these charts.",
        "The two demo fixtures are illustrative seeds chosen for contrasting verdicts; they are not study results.",
        "Detection is a declared operational condition (threshold + hysteresis + gates), not proof of a fault; stable traces still trigger incidents.",
        "The verifier closes a short observation window; a beneficial change can still be inconclusive and rolled back.",
        "The research chart is the frozen held-out study, not a new experiment: nothing on this page trained, tuned or overwrote it.",
    )
    items = "".join(f'<p class="finding">{esc(text)}</p>' for text in bullets)
    return f"""
<section class="section" id="limits" data-panel-mode="all">
  <h2><span class="index">06</span> What is and is not implemented</h2>
  {items}
  <p class="note">
    Full discussion: <code>docs/LIMITATIONS.md</code>, <code>docs/TEACHER_QA.md</code> and
    <code>docs/MANUAL_VERIFICATION_REPORT.md</code>.
  </p>
</section>
"""


def _payload(doc: DemoDocument, teaching: TeachingPanel, research: Optional[ResearchSummary], provenance: Mapping[str, str]) -> Dict[str, Any]:
    scenarios = []
    for fixture in doc.fixtures:
        incident = fixture.primary_incident
        scenarios.append(
            {
                "seed": fixture.seed,
                "family": fixture.family,
                "fingerprint": fixture.fingerprint,
                "verdict": incident.verdict if incident else None,
                "final_policy": fixture.final_policy,
                "mean_waiting_time": fixture.metrics.avg_waiting_time,
                "fcfs_mean_waiting_time": fixture.fcfs_metrics.avg_waiting_time,
                "difference": fixture.mean_waiting_difference,
                "processes": [
                    {
                        "pid": process.pid,
                        "arrival_time": process.arrival_time,
                        "burst_time": process.burst_time,
                        "priority": process.priority,
                        "start_time": fixture.process_row(process).start_time,
                        "completion_time": fixture.process_row(process).completion_time,
                        "waiting_time": fixture.process_row(process).waiting_time,
                        "turnaround_time": fixture.process_row(process).turnaround_time,
                    }
                    for process in sorted(fixture.processes, key=lambda item: item.pid)
                ],
                "intervals": [
                    {
                        "pid": interval.pid,
                        "policy": interval.policy,
                        "start": interval.start,
                        "end": interval.end,
                        "switch_overhead": interval.switch_before,
                        "idle": interval.idle_before,
                        "waiting_increment": interval.waiting_increment,
                        "completed_pids": list(interval.completed_pids),
                    }
                    for interval in fixture.intervals
                ],
            }
        )
    payload: Dict[str, Any] = {
        "provenance": dict(provenance),
        "scope": doc.scope,
        "scheduler": dict(doc.scheduler),
        "scenarios": scenarios,
        "teaching": {
            "quantum": teaching.quantum,
            "switching_cost": teaching.switching_cost,
            "policies": [
                {
                    "name": policy.name,
                    "mean_waiting_time": policy.mean_waiting_time,
                    "hand_verified": policy.hand_verified,
                    "intervals": [
                        {"pid": item.pid, "start": item.start, "end": item.end}
                        for item in policy.intervals
                    ],
                }
                for policy in teaching.policies
            ],
        },
    }
    if research is not None:
        payload["research"] = {
            "source": research.source_name,
            "split": research.split,
            "workloads": research.workloads,
            "final_test_seed": research.final_test_seed,
            "training_seeds": list(research.training_seeds),
            "arms": [
                {"arm": arm.arm, "mean_waiting_time": arm.mean_waiting_time}
                for arm in research.chart_arms
            ],
            "comparisons": [
                {
                    "target": row.target,
                    "reference": row.reference,
                    "metric": row.metric,
                    "difference": row.difference,
                    "ci95_low": row.ci_low,
                    "ci95_high": row.ci_high,
                }
                for row in research.comparisons
                if row.metric == "avg_waiting_time"
            ],
        }
    return payload


def _embed_json(payload: Mapping[str, Any]) -> str:
    """Serialize the payload so it cannot terminate its own ``<script>`` element."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_dashboard(
    doc: DemoDocument,
    teaching: TeachingPanel,
    research: Optional[ResearchSummary],
    provenance: Mapping[str, str],
) -> str:
    """Render the complete HTML document as a string."""
    payload = _embed_json(_payload(doc, teaching, research, provenance))
    title = "Self-healing CPU scheduling · teacher dashboard"
    research_attr = research.source_name if research else "not loaded"
    footer = f"""
<footer>
  <p><strong>Generated from recorded simulator output.</strong>
  Demo: <code>{esc(provenance.get("input_name", "demo.json"))}</code> (sha256 <code>{esc(provenance.get("input_sha256_short", "n/a"))}</code>).
  Research: <code>{esc(research_attr)}</code>.</p>
  <p>Rebuild with <code>python tools/build_teacher_dashboard.py --input demo.json --output index.html</code>.
  This page contains no external requests of any kind; open the file directly in a browser.</p>
</footer>
"""
    return f"""<!DOCTYPE html>
<html lang="en" class="no-js">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
<div class="page">
{_masthead(doc, research, provenance)}
{_nav()}
<noscript><p class="noscript">JavaScript is disabled, so every section of this page is shown in full.
All charts and tables are static HTML/SVG and remain complete.</p></noscript>
{_mode_tabs()}
<main id="main">
  <div data-panel-mode="demo">
    {_section_overview(doc, research, provenance)}
    {_section_scheduling(doc)}
    {_section_teaching(teaching, doc)}
    {_section_lifecycle(doc)}
    {_section_performance(doc)}
  </div>
  {_section_research(research)}
  {_section_limits()}
</main>
{footer}
</div>
<script id="dashboard-data" type="application/json">{payload}</script>
<script>{_SCRIPT}</script>
</body>
</html>
"""
