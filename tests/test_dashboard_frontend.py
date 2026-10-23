"""Regression tests for the three interface defects fixed after browser verification.

These are static assertions over the shipped frontend source, in the same style as
``test_frontend_script_uses_only_relative_api_calls``: they cannot prove the interface looks
right (only a browser can), but they pin the exact structures whose absence caused the
defects, so the same mistake cannot return unnoticed.

The defects, as recorded in ``docs/manual_evidence/web-v3/recapture-report.json``:

1. ``runComparison()`` reset ``comparisonBusy`` in ``finally`` *after* painting, and the paint
   read the stale flag, so the button stayed disabled on "Running four policies…".
2. The footer took its run count from the health payload while the overview took a different
   one from the run list, so the two disagreed after every run.
3. At 420 px the workload table cut its headers off mid-column, the Gantt showed a fraction
   of the trace, and the Studio summary stranded one value against the panel edge.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "dashboard" / "web"
APP = (WEB / "app.js").read_text(encoding="utf-8")
CSS = (WEB / "styles.css").read_text(encoding="utf-8")


def function_body(name: str) -> str:
    """Return the source of a top-level declaration, up to its closing brace at column 2.

    Accepts ``function name(...)``, ``async function name(...)`` and ``PAGES.name = (...) =>``.
    """
    match = re.search(
        rf"\n  (?:(?:async )?function {re.escape(name)}\([^)]*\)|PAGES\.{re.escape(name)} = \([^)]*\) =>) \{{",
        APP,
    )
    if not match:
        # PAGES.<route> contains a dot, which the escaped name cannot match literally.
        prefix = name.split(".", 1)[1] if "." in name else name
        match = re.search(rf"\n  PAGES\.{re.escape(prefix)} = ", APP)
    assert match, f"{name}() not found in app.js"
    start = match.end()
    end = APP.find("\n  }\n", start)
    assert end > start, f"could not isolate the body of {name}()"
    return APP[start:end]


def css_block(selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert match, f"{selector} not found in styles.css"
    return match.group(1)


def mobile_block() -> str:
    match = re.search(r"@media \(max-width: 560px\) \{(.*)\n\}\n", CSS, re.S)
    assert match, "the 560 px breakpoint is missing"
    return match.group(1)


# ------------------------------------------------------------------ 1. comparison button


def test_comparison_resets_busy_before_the_final_paint():
    body = function_body("runComparison")
    finally_block = body[body.index("finally {") :]
    reset = finally_block.index("state.comparisonBusy = false")
    paint = finally_block.index("render()")
    assert reset < paint, (
        "comparisonBusy must be cleared before render(), or the page paints the in-flight "
        "button after the comparison has finished"
    )


def test_comparison_paints_after_success_and_after_failure():
    body = function_body("runComparison")
    assert body.count("render()") == 1
    # The single paint lives in finally, so both paths reach it.
    assert "finally {" in body
    assert body.index("catch (error)") < body.index("finally {")


def test_comparison_clears_a_stale_result_before_running():
    body = function_body("runComparison")
    assert body.index("state.comparison = null") < body.index("/api/comparisons")


def test_comparison_ignores_results_for_a_replaced_run():
    body = function_body("runComparison")
    assert "isCurrentRun(runId)" in body
    guard = function_body("isCurrentRun")
    assert "state.run.run_id === runId" in guard


def test_comparison_runs_one_request_at_a_time():
    body = function_body("runComparison")
    assert "if (!state.run || state.comparisonBusy) return;" in body


def test_button_label_is_derived_from_the_shared_helper():
    page = function_body("PAGES.compare")
    assert "comparisonLabel()" in page
    helper = function_body("comparisonLabel")
    assert "Run comparison again" in helper and "'Run comparison'" in helper
    setter = function_body("setComparisonBusy")
    assert "comparisonLabel()" in setter, "the idle label must come from the same helper"
    assert "button.disabled = busy" in setter


# ------------------------------------------------------------------ 2. run counter


def test_footer_count_comes_from_the_run_registry_not_the_health_probe():
    health = "\n".join(
        line for line in function_body("refreshHealth").splitlines()
        if not line.strip().startswith("*")
    )
    assert "stored_runs" not in health, (
        "the footer must not read its run count from /api/health; that payload is refreshed on "
        "a slow poll and drifted from the overview's figure"
    )
    assert "renderCounters()" in health
    counters = function_body("renderCounters")
    assert "state.sessionRuns" in counters


def test_run_count_is_read_from_the_runs_endpoint():
    loader = function_body("loadRecentRuns")
    assert "payload.count" in loader
    assert "state.sessionRuns" in loader


def test_every_run_count_on_screen_uses_the_same_helper():
    assert "runCountText('kept in memory')" in APP
    assert APP.count("runCountText('kept in memory')") == 2, (
        "the footer and the overview facts must print the same figure with the same wording"
    )
    # No second counter anywhere (comments may explain the old one; code may not use it).
    code = "\n".join(line for line in APP.splitlines() if not line.strip().startswith("*"))
    assert "recent.length} run" not in code
    assert "stored_runs" not in code


def test_truncated_history_says_so():
    hint = function_body("historyHint")
    assert "most recent of" in hint
    assert "state.recentRuns.length" in hint and "state.sessionRuns" in hint


# ------------------------------------------------------------------ 3. mobile layout


def test_workload_and_comparison_tables_are_labelled_for_stacking():
    assert APP.count('<table class="data stacked"') == 2
    # Every value cell carries the column name the stacked layout prints.
    assert 'data-label="${esc(label)}"' in APP
    for label in ("Mean waiting", "Turnaround", "Response", "Switches", "CPU busy", "Elapsed"):
        assert f'data-label="{label}"' in APP


def test_stacked_tables_keep_the_header_for_assistive_technology():
    block = mobile_block()
    assert "table.data.stacked thead" in block
    # Visually hidden, not display:none, so screen readers still get the header.
    assert "clip: rect(0 0 0 0)" in block
    assert "display: none" not in block.split("table.data.stacked thead")[1].split("}")[0]


def test_stacked_table_cells_show_their_labels():
    assert "content: attr(data-label)" in mobile_block()


def test_gantt_fits_the_panel_on_small_screens_without_shrinking_its_labels():
    assert 'data-action="toggle-chart-fit"' in APP
    assert "toggle-chart-fit" in function_body("bindEvents")
    block = mobile_block()
    # Fit mode removes the 880 px floor so the whole trace is on screen...
    assert '.chart-frame[data-fit="true"] #gantt svg { min-width: 0; }' in block
    # ...and enlarges the SVG type, because the scaled-down coordinates would render it at
    # roughly 4 px otherwise.
    assert '#gantt text { font-size: 20px; }' in block
    assert "min-width: 880px" in CSS, "actual-size mode must keep the full-detail chart"


def test_bar_charts_scroll_at_a_legible_width_on_small_screens():
    block = mobile_block()
    assert ".chart-scroll svg.chart { min-width: 560px; }" in block, (
        "scaling a 760 px bar chart into a 320 px panel renders its labels at about 4 px"
    )


def test_scrollable_charts_say_so_on_small_screens():
    block = mobile_block()
    assert ".chart-scroll::-webkit-scrollbar { height: 8px; }" in block
    assert ".chart-caption::after" in block and "scroll" in block.lower()


def test_run_summary_stacks_instead_of_stranding_a_value():
    block = mobile_block()
    assert ".run-card .summary div { flex-direction: column" in block
    assert ".run-card .summary dd { text-align: left; }" in block


def test_desktop_layout_is_untouched_by_the_mobile_rules():
    # The mobile rules must live inside the 560 px block, not in the base sheet.
    base = CSS.split("@media (max-width: 560px)")[0]
    assert "table.data.stacked tr {" not in base
    assert "chart-tools { display: flex" not in base
    assert ".chart-tools { display: none; }" in base
    assert "#gantt text { font-size: 20px; }" not in base


def test_banner_dismiss_button_has_a_listener():
    """#banner is a sibling of #view, so the delegated handler in view() cannot reach it."""
    assert "el('banner').addEventListener('click'" in APP
    banner_listener = APP[APP.index("el('banner').addEventListener"):]
    banner_listener = banner_listener[: banner_listener.index("});") + 3]
    assert "dismiss-banner" in banner_listener and "clearBanner()" in banner_listener
