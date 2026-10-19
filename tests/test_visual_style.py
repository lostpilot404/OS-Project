"""Visual consistency: all figures must use the centralized style."""

import pathlib


def test_no_hardcoded_dpi_outside_style_module() -> None:
    for path in pathlib.Path("visualization").glob("*.py"):
        if path.name == "style.py":
            continue
        text = path.read_text()
        # No direct dpi=140 literals outside style — figures must use DPI constant
        assert "dpi=140" not in text, f"{path} still hard-codes dpi=140, use DPI from style"
        assert "dpi = 140" not in text


def test_figures_use_central_palette() -> None:
    # healing_plots and plots must import from visualization.style
    for name in ["visualization/healing_plots.py", "visualization/plots.py", "visualization/study_plots.py"]:
        text = pathlib.Path(name).read_text()
        assert "from visualization.style import" in text or "visualization.style" in text, f"{name} does not use centralized style"


def test_style_exports_expected_constants() -> None:
    from visualization.style import DPI, FIGSIZE_WIDE, PALETTE

    assert DPI == 140
    assert len(FIGSIZE_WIDE) == 2
    assert "q_recovery" in PALETTE
    assert "rule_wait" in PALETTE
