#!/usr/bin/env python3
"""Fix table column widths in the DOCX so content isn't cut off.

Sets proportional column widths based on max character length per column,
with a cap so no single column dominates."""

from docx import Document
from docx.shared import Cm, Pt
from docx.oxml.ns import qn
from lxml import etree
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(BASE, "Final_Research_Paper.docx")

# Page width minus margins = usable table width in EMU
# A4 = 210mm wide, margins = 2.54cm each side
# Usable = 210 - 50.8 = 159.2 mm = 15920000 EMU (1mm = 36000 EMU)
PAGE_WIDTH_EMU = 15920000


def set_table_column_widths(table):
    """Set proportional column widths based on content."""
    nrows = len(table.rows)
    ncols = len(table.columns)
    if ncols == 0:
        return

    # Measure max character length per column
    col_max_chars = [0] * ncols
    for row in table.rows:
        for c, cell in enumerate(row.cells):
            col_max_chars[c] = max(col_max_chars[c], len(cell.text.strip()))

    # Handle edge case: all columns empty
    total_chars = sum(col_max_chars)
    if total_chars == 0:
        col_max_chars = [1] * ncols
        total_chars = ncols

    # Proportional widths, capped at 45% per column
    max_single_frac = 0.45
    raw_widths = [PAGE_WIDTH_EMU * (cw / total_chars) for cw in col_max_chars]
    adjusted = [min(w, PAGE_WIDTH_EMU * max_single_frac) for w in raw_widths]

    # Redistribute shortfall from capped columns
    shortfall = sum(raw_widths) - sum(adjusted)
    if shortfall > 100:  # Only redistribute meaningful amounts
        uncapped = [i for i, w in enumerate(raw_widths) if w <= PAGE_WIDTH_EMU * max_single_frac]
        if uncapped:
            per_col = shortfall / len(uncapped)
            for i in uncapped:
                adjusted[i] += per_col

    # Set tblW to auto (pct 5000 = 50% of page? No, use dxa)
    # Use the full available width
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = etree.SubElement(tbl, qn('w:tblPr'))

    tblW = tblPr.find(qn('w:tblW'))
    if tblW is None:
        tblW = etree.SubElement(tblPr, qn('w:tblW'))
    tblW.set(qn('w:w'), str(PAGE_WIDTH_EMU))
    tblW.set(qn('w:type'), 'dxa')

    # Set tblGrid with proportional gridCol widths
    grid = tbl.find(qn('w:tblGrid'))
    if grid is None:
        grid = etree.SubElement(tbl, qn('w:tblGrid'))
    else:
        # Remove existing gridCols
        for gc in grid.findall(qn('w:gridCol')):
            grid.remove(gc)

    for w in adjusted:
        gc = etree.SubElement(grid, qn('w:gridCol'))
        gc.set(qn('w:w'), str(int(w)))

    # Also set individual cell widths via tcW
    for row in table.rows:
        for c, cell in enumerate(row.cells):
            tc = cell._tc
            tcPr = tc.find(qn('w:tcPr'))
            if tcPr is None:
                tcPr = etree.SubElement(tc, qn('w:tcPr'))
            tcW = tcPr.find(qn('w:tcW'))
            if tcW is None:
                tcW = etree.SubElement(tcPr, qn('w:tcW'))
            tcW.set(qn('w:w'), str(int(adjusted[c])))
            tcW.set(qn('w:type'), 'dxa')

            # Allow text wrapping
            tcW2 = tcPr.find(qn('w:noWrap'))
            if tcW2 is not None:
                tcPr.remove(tcW2)


doc = Document(DOCX_PATH)

for i, table in enumerate(doc.tables):
    set_table_column_widths(table)

doc.save(DOCX_PATH)
print(f"Fixed column widths for {len(doc.tables)} tables in {DOCX_PATH}")
