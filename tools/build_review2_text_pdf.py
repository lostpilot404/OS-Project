"""Render the Review 2 DOCX as a text-only PDF.

The PDF carries the same words as ``Review_2_Report.docx`` and nothing else: no tables, no
colour, no images. Headings, bullets, numbered steps, pseudocode and Mermaid source are laid
out with ReportLab in black type. Mermaid diagrams appear as copy-ready source, in the same
numbered order as the DOCX.

Usage::

    pip install -r requirements-report.txt
    python tools/build_review2_text_pdf.py                 # writes Review_2_Report.pdf + docs copy
    python tools/build_review2_text_pdf.py --docx path.docx --output /tmp/r.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
MONO = "Consolas"  # the DOCX monospace font name, used to recognise code paragraphs


def _register_fonts() -> None:
    """Register DejaVu fonts so every character in the document renders."""
    pdfmetrics.registerFont(TTFont("Body", FONT_DIR / "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("Body-Bold", FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Code", FONT_DIR / "DejaVuSansMono.ttf"))
    pdfmetrics.registerFont(TTFont("Code-Bold", FONT_DIR / "DejaVuSansMono-Bold.ttf"))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily

    registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
    registerFontFamily("Code", normal="Code", bold="Code-Bold", italic="Code", boldItalic="Code-Bold")


def _styles() -> dict:
    black = "#000000"
    return {
        "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=18, leading=23,
                                alignment=TA_CENTER, spaceAfter=10, textColor=black),
        "cover": ParagraphStyle("cover", fontName="Body", fontSize=11, leading=15,
                                alignment=TA_CENTER, spaceAfter=6, textColor=black),
        "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=14, leading=18,
                             spaceBefore=16, spaceAfter=6, keepWithNext=1, textColor=black),
        "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=11.5, leading=15,
                             spaceBefore=10, spaceAfter=4, keepWithNext=1, textColor=black),
        "body": ParagraphStyle("body", fontName="Body", fontSize=9.8, leading=13.6,
                               alignment=TA_LEFT, spaceAfter=5, textColor=black),
        "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.8, leading=13.6,
                                 leftIndent=16, bulletIndent=4, spaceAfter=3, textColor=black),
        "step": ParagraphStyle("step", fontName="Body", fontSize=9.8, leading=13.6,
                               leftIndent=20, bulletIndent=0, spaceAfter=3, textColor=black),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=8.8, leading=12,
                                  spaceBefore=4, spaceAfter=3, keepWithNext=1, textColor=black),
        "code": ParagraphStyle("code", fontName="Code", fontSize=7.6, leading=9.6,
                               leftIndent=18, spaceAfter=0, textColor=black),
    }


_NUMBERED = re.compile(r"^(\d+)\.\t(.*)$", re.S)


def _runs_markup(paragraph) -> str:
    """Convert python-docx runs to ReportLab mini-markup, keeping bold and italic only."""
    parts: List[str] = []
    for run in paragraph.runs:
        text = escape(run.text)
        if not text:
            continue
        if run.bold:
            text = f"<b>{text}</b>"
        if run.italic:
            text = f"<i>{text}</i>"
        parts.append(text)
    markup = "".join(parts)
    return markup.replace("\t", " ")


def _is_code(paragraph) -> bool:
    runs = [r for r in paragraph.runs if r.text]
    return bool(runs) and all(r.font.name == MONO for r in runs)


def _is_caption(paragraph) -> bool:
    runs = [r for r in paragraph.runs if r.text]
    return bool(runs) and all(r.italic and r.font.size is not None and r.font.size.pt <= 9.5 for r in runs)


def build_flowables(docx_path: Path) -> List:
    styles = _styles()
    document = Document(str(docx_path))
    flow: List = []
    code_buffer: List[str] = []

    def flush_code() -> None:
        if code_buffer:
            flow.append(Preformatted("\n".join(code_buffer), styles["code"]))
            flow.append(Spacer(1, 6))
            code_buffer.clear()

    for paragraph in document.paragraphs:
        style_name = paragraph.style.name if paragraph.style is not None else "Normal"
        text = paragraph.text

        if _is_code(paragraph):
            code_buffer.append(text if text.strip() else " ")
            continue
        flush_code()

        if not text.strip():
            flow.append(Spacer(1, 4))
            continue

        if style_name == "Title":
            flow.append(Paragraph(_runs_markup(paragraph), styles["title"]))
            continue
        if style_name == "Heading 1":
            flow.append(Paragraph(_runs_markup(paragraph), styles["h1"]))
            continue
        if style_name == "Heading 2":
            flow.append(Paragraph(_runs_markup(paragraph), styles["h2"]))
            continue
        if style_name == "List Bullet":
            flow.append(Paragraph(_runs_markup(paragraph), styles["bullet"], bulletText="•"))
            continue
        if paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER:
            flow.append(Paragraph(_runs_markup(paragraph), styles["cover"]))
            continue
        if _is_caption(paragraph):
            flow.append(Paragraph(_runs_markup(paragraph), styles["caption"]))
            continue

        match = _NUMBERED.match(text)
        if match:
            number, rest = match.group(1), match.group(2)
            # The first run of a numbered step is its bold prefix when present; rebuild markup
            # from the raw text so the number sits in the hanging indent.
            body = _runs_markup(paragraph)
            body = re.sub(r"^\s*\d+\.\s*", "", body, count=1)
            flow.append(Paragraph(body, styles["step"], bulletText=f"{number}."))
            continue

        flow.append(Paragraph(_runs_markup(paragraph), styles["body"]))

    flush_code()
    return flow


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm,
                             f"Operating Systems Project — Review 2   |   page {doc.page}")
    canvas.restoreState()


def render(docx_path: Path, output: Path) -> None:
    _register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Operating Systems Project — Review 2",
        author="",
        creator="tools/build_review2_text_pdf.py",
        subject="Methodology and Partial Implementation (Lab Assessment 9)",
    )
    doc.build(build_flowables(docx_path), onFirstPage=_footer, onLaterPages=_footer)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, default=REPO_ROOT / "Review_2_Report.docx",
                        help="source DOCX (default: <repo>/Review_2_Report.docx)")
    parser.add_argument("--output", type=Path, default=None,
                        help="PDF path (default: <repo>/Review_2_Report.pdf)")
    parser.add_argument("--docs-copy", type=Path, default=None,
                        help="second copy (default: <repo>/docs/Review_2_Report.pdf)")
    args = parser.parse_args(argv)

    output = args.output or REPO_ROOT / "Review_2_Report.pdf"
    docs_copy = args.docs_copy or REPO_ROOT / "docs" / "Review_2_Report.pdf"
    render(args.docx, output)
    if docs_copy.resolve() != output.resolve():
        docs_copy.parent.mkdir(parents=True, exist_ok=True)
        docs_copy.write_bytes(output.read_bytes())
    print(f"wrote {output} ({output.stat().st_size / 1024:,.0f} KB)")
    print(f"wrote {docs_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
