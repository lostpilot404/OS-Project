"""Shared ReportLab layout for the Review 2 report.

Extracted from the original report builder so that the narrative lives in one place
(:mod:`review2_content`) and the PDF builder only renders it. The styles, flowable
helpers and the document template are independent of which experiment the report
describes: the running header, page numbers and table of contents are generic.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)


REPORT_TITLE = "Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads"
REPORT_SHORT = "Self-Healing CPU Scheduling in a Discrete-Event Simulator"
REVIEW_LINE = "Review 2 \u2014 Methodology and Partial Implementation (Lab Assessment 9)"
REPO_URL = "github.com/lostpilot404/OS-Project"
HEADER_LINE = ("Review 2  |  Methodology and Partial Implementation  |  " + REPORT_SHORT)

NAVY = colors.HexColor("#1F3864")
STEEL = colors.HexColor("#2E75B6")
BAND = colors.HexColor("#F2F5FB")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#B7C4DD")

# Page geometry. The text column is A4 minus both margins; every table and figure in the
# report is sized from CONTENT_W, so a change here propagates everywhere.
PAGE_W, PAGE_H = A4
MARGIN_X = 1.7 * cm
MARGIN_TOP = 1.9 * cm
MARGIN_BOTTOM = 1.6 * cm
CONTENT_W = PAGE_W - 2 * MARGIN_X

# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------
def build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20.5,
            leading=24, textColor=NAVY, spaceAfter=3, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12,
            leading=15, textColor=STEEL, alignment=TA_CENTER, spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", parent=base["Normal"], fontSize=9, leading=13,
            textColor=colors.HexColor("#222222"), alignment=TA_CENTER,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12.8,
            leading=15.5, textColor=NAVY, spaceBefore=9, spaceAfter=4,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=13, textColor=STEEL, spaceBefore=7, spaceAfter=3,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=12.5, alignment=TA_JUSTIFY, spaceAfter=4,
            textColor=colors.HexColor("#1A1A1A"),
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.0,
            leading=12.1, leftIndent=11, bulletIndent=2, spaceAfter=2.5,
            alignment=TA_JUSTIFY,
        ),
        "Caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=7.7,
            leading=9.5, textColor=GREY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=7,
        ),
        "Cell": ParagraphStyle(
            "Cell", parent=base["Normal"], fontSize=7.4, leading=9.1,
            textColor=colors.HexColor("#1A1A1A"),
        ),
        "CellH": ParagraphStyle(
            "CellH", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.4,
            leading=9.1, textColor=colors.white,
        ),
        "Small": ParagraphStyle(
            "Small", parent=base["Normal"], fontSize=8.0, leading=10.5, textColor=GREY,
        ),
        "Equation": ParagraphStyle(
            "Equation", parent=base["Normal"], fontName="Courier-Bold", fontSize=8.5,
            leading=12, alignment=TA_CENTER, textColor=NAVY, spaceBefore=3, spaceAfter=3,
        ),
        "Abstract": ParagraphStyle(
            "Abstract", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1,
            leading=12.6, alignment=TA_JUSTIFY, borderPadding=6,
            backColor=colors.HexColor("#F5F7FC"),
            textColor=colors.HexColor("#1A1A1A"),
        ),
    }



# --------------------------------------------------------------------------
S = build_styles()


# --------------------------------------------------------------------------
# Flowable helpers
# --------------------------------------------------------------------------
def para(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


def h1(text: str) -> Paragraph:
    return Paragraph(text, S["H1"])


def h2(text: str) -> Paragraph:
    return Paragraph(text, S["H2"])


def bullets(items: Sequence[str]) -> List[Paragraph]:
    return [Paragraph(item, S["Bullet"], bulletText="•") for item in items]


def caption(text: str) -> Paragraph:
    return Paragraph(text, S["Caption"])


def figure(path: Path, text: str, width: float = CONTENT_W,
           max_height: float = 20.6 * cm) -> List:
    """Embed a figure scaled to the text column, capped so it always fits one page."""
    image = Image(str(path))
    scale = min(width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return [KeepTogether([image, caption(text)])]


def code_block(text: str, font_size: float = 6.9, width: float = CONTENT_W) -> Table:
    """Render preformatted text (terminal output / pseudocode) in a framed box."""
    lines = [
        html.escape(line).replace(" ", "&nbsp;") for line in text.strip("\n").split("\n")
    ]
    body = "<br/>".join(lines) if lines else "&nbsp;"
    style = ParagraphStyle(
        "code", fontName="Courier", fontSize=font_size, leading=font_size + 2.2,
        textColor=colors.HexColor("#152238"),
    )
    table = Table([[Paragraph(body, style)]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6FA")),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_table(
    rows: Sequence[Sequence],
    widths: Sequence[float],
    align_right: Sequence[int] = (),
    header: bool = True,
    band: bool = True,
) -> Table:
    """Styled table; cells may be strings (auto-wrapped) or Paragraphs."""
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for cell in row:
            if isinstance(cell, Paragraph):
                out.append(cell)
            elif r_i == 0 and header:
                out.append(Paragraph(str(cell), S["CellH"]))
            else:
                out.append(Paragraph(str(cell), S["Cell"]))
        data.append(out)

    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TOPPADDING", (0, 0), (-1, 0), 3.4),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3.4),
        ]
    if band:
        for i in range(1 if header else 0, len(data)):
            if (i - (1 if header else 0)) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), BAND))
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    table.setStyle(TableStyle(style))
    return table


def kpi_strip(items: Sequence[Tuple[str, str]]) -> Table:
    cells = []
    for value, label in items:
        cells.append(
            Table(
                [
                    [Paragraph(
                        f'<font size="12.5"><b>{value}</b></font>',
                        ParagraphStyle("kv", alignment=TA_CENTER, textColor=NAVY, leading=14),
                    )],
                    [Paragraph(
                        f'<font size="6.5">{label}</font>',
                        ParagraphStyle("kl", alignment=TA_CENTER, textColor=GREY, leading=8),
                    )],
                ],
                colWidths=[CONTENT_W / len(items) - 3],
            )
        )
    strip = Table([cells], colWidths=[CONTENT_W / len(items)] * len(items))
    strip.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return strip


# --------------------------------------------------------------------------
# Document template
# --------------------------------------------------------------------------
class ReviewDoc(BaseDocTemplate):
    def __init__(self, filename, generated: str, **kwargs):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title="Review 2 - Methodology and Partial Implementation - "
                  "Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads",
            author="OS-Project (github.com/lostpilot404/OS-Project)",
            subject="Review 2 (Lab Assessment 9)",
            **kwargs,
        )
        self.generated = generated
        cover_frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W,
                            PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="cover")
        body_frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W,
                           PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - 0.9 * cm, id="body")
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=[cover_frame], onPage=self._cover),
                PageTemplate(id="Body", frames=[body_frame], onPage=self._decorate),
            ]
        )

    def _cover(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 1.15 * cm, PAGE_W, 1.15 * cm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(MARGIN_X, PAGE_H - 0.78 * cm,
                          "OPERATING SYSTEMS PROJECT  •  REVIEW 2")
        canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 0.78 * cm,
                               "LAB ASSESSMENT 9")
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_W, 0.55 * cm, stroke=0, fill=1)
        canvas.restoreState()

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        y_top = PAGE_H - MARGIN_TOP + 0.45 * cm
        canvas.line(MARGIN_X, y_top, PAGE_W - MARGIN_X, y_top)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(MARGIN_X, y_top + 0.16 * cm,
                          "Review 2  |  Methodology and Partial Implementation  |  "
                          + REPORT_SHORT)
        canvas.drawRightString(PAGE_W - MARGIN_X, y_top + 0.16 * cm, REPO_URL)
        y_bot = MARGIN_BOTTOM - 0.25 * cm
        canvas.line(MARGIN_X, y_bot, PAGE_W - MARGIN_X, y_bot)
        canvas.drawString(MARGIN_X, y_bot - 0.42 * cm,
                          f"Generated from committed artifacts on {self.generated}")
        canvas.drawRightString(PAGE_W - MARGIN_X, y_bot - 0.42 * cm,
                               f"Page {doc.page - 1}")
        canvas.restoreState()

    #: headings that must not appear in the table of contents
    toc_excluded = ("Contents",)

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            text = flowable.getPlainText()
            if text in self.toc_excluded:
                return
            if name == "H1":
                self.notify("TOCEntry", (0, text, self.page - 1))
            elif name == "H2":
                self.notify("TOCEntry", (1, text, self.page - 1))


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
