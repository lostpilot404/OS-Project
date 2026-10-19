"""Build the black-and-white, text-only academic report PDF from its Markdown source.

Usage from the repository root:
    python tools/build_plain_report.py
    python tools/build_plain_report.py --source docs/OS_Project_Report_Plain.md \
        --output /tmp/report.pdf

Only paragraphs, headings, list items, equations as text, and monospaced code blocks are
emitted. Mermaid blocks are copied as source; no diagram, image, or table renderer is used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEFAULT = ROOT / "docs" / "OS_Project_Report_Plain.md"
OUTPUT_DEFAULT = ROOT / "docs" / "OS_Project_Report_Plain.pdf"


def _font_paths() -> tuple[Path, Path, Path]:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/liberation2"),
    ]
    for directory in candidates:
        regular = directory / "DejaVuSerif.ttf"
        bold = directory / "DejaVuSerif-Bold.ttf"
        mono = directory / "DejaVuSansMono.ttf"
        if regular.exists() and bold.exists() and mono.exists():
            return regular, bold, mono
        liberation = (
            directory / "LiberationSerif-Regular.ttf",
            directory / "LiberationSerif-Bold.ttf",
            directory / "LiberationMono-Regular.ttf",
        )
        if all(path.exists() for path in liberation):
            return liberation
    raise RuntimeError(
        "A local serif font and monospaced font are required. Install DejaVu or Liberation fonts."
    )


def _inline(text: str) -> str:
    """Escape plain Markdown text and preserve its small inline emphasis/code vocabulary."""
    pattern = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")
    output: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        output.append(escape(text[cursor : match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            output.append(f"<b>{escape(token[2:-2])}</b>")
        elif token.startswith("`"):
            output.append(f'<font name="ReportMono">{escape(token[1:-1])}</font>')
        else:
            output.append(f"<i>{escape(token[1:-1])}</i>")
        cursor = match.end()
    output.append(escape(text[cursor:]))
    return "".join(output)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="ReportSerif-Bold",
            fontSize=21, leading=26, alignment=TA_CENTER, textColor="#000000",
            spaceAfter=13 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Heading2"], fontName="ReportSerif-Bold",
            fontSize=13, leading=17, alignment=TA_CENTER, textColor="#000000",
            spaceAfter=5 * mm,
        ),
        "h1": ParagraphStyle(
            "ReportH1", parent=base["Heading1"], fontName="ReportSerif-Bold",
            fontSize=15, leading=19, textColor="#000000", spaceBefore=5 * mm,
            spaceAfter=3 * mm, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=base["Heading2"], fontName="ReportSerif-Bold",
            fontSize=12, leading=15, textColor="#000000", spaceBefore=3.5 * mm,
            spaceAfter=2 * mm, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "ReportH3", parent=base["Heading3"], fontName="ReportSerif-Bold",
            fontSize=10.5, leading=13, textColor="#000000", spaceBefore=2.5 * mm,
            spaceAfter=1.5 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["BodyText"], fontName="ReportSerif",
            fontSize=9.4, leading=13, alignment=TA_JUSTIFY, textColor="#000000",
            spaceAfter=2.2 * mm, allowWidows=0, allowOrphans=0,
        ),
        "metadata": ParagraphStyle(
            "ReportMetadata", parent=base["BodyText"], fontName="ReportSerif",
            fontSize=9.5, leading=14, alignment=TA_CENTER, textColor="#000000",
            spaceAfter=2 * mm,
        ),
        "list": ParagraphStyle(
            "ReportList", parent=base["BodyText"], fontName="ReportSerif",
            fontSize=9.3, leading=12.7, alignment=TA_JUSTIFY, textColor="#000000",
            leftIndent=7 * mm, firstLineIndent=-7 * mm, spaceAfter=1.4 * mm,
            allowWidows=0, allowOrphans=0,
        ),
        "code": ParagraphStyle(
            "ReportCode", parent=base["Code"], fontName="ReportMono",
            fontSize=8.0, leading=9.6, textColor="#000000", backColor=None,
            borderWidth=0, leftIndent=0, rightIndent=0, spaceBefore=1.5 * mm,
            spaceAfter=1.5 * mm, splitLongWords=0,
        ),
    }


def _invariant_canvas(*args, **kwargs):
    kwargs["invariant"] = 1
    return canvas.Canvas(*args, **kwargs)


def _page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColorRGB(0, 0, 0)
    canvas.setFont("ReportSerif", 8)
    canvas.drawString(22 * mm, 13 * mm, "Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads")
    canvas.drawRightString(A4[0] - 22 * mm, 13 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _make_story(source: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = source.splitlines()
    story: list = []
    i = 0
    in_cover = True
    cover_metadata = True
    first_title_seen = False
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.startswith("```"):
            language = line[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError("Unclosed fenced code block in report source")
            code_text = "\n".join(code_lines)
            # Mermaid code is deliberately rendered as literal source. Keep the subsection
            # heading, its introductory sentence, and the complete code block together when
            # they fit, so a page break cannot strand the diagram explanation.
            block: list = []
            if len(story) >= 2 and isinstance(story[-1], Paragraph) and isinstance(story[-2], Paragraph):
                previous_style = getattr(story[-2], "style", None)
                if previous_style and previous_style.name in {"ReportH1", "ReportH2", "ReportH3"}:
                    block = [story[-2], story[-1]]
                    del story[-2:]
            elif story and isinstance(story[-1], Paragraph):
                block = [story.pop()]
            block.append(Preformatted(code_text, styles["code"]))
            story.append(KeepTogether(block))
            i += 1
            continue

        if line.startswith("# "):
            if first_title_seen:
                story.append(Paragraph(_inline(line[2:].strip()), styles["h1"]))
            else:
                story.append(Spacer(1, 18 * mm))
                story.append(Paragraph(_inline(line[2:].strip()), styles["title"]))
                first_title_seen = True
            i += 1
            continue

        if line.startswith("## "):
            title = line[3:].strip()
            if title == "Operating Systems Project Report":
                story.append(Paragraph(_inline(title), styles["subtitle"]))
            elif title == "Abstract" and in_cover:
                cover_metadata = False
                story.append(Paragraph(_inline(title), styles["h2"]))
            elif title == "1. Detailed methodology of the proposed system" and in_cover:
                story.append(PageBreak())
                in_cover = False
                story.append(Paragraph(_inline(title), styles["h1"]))
            else:
                story.append(Paragraph(_inline(title), styles["h1"]))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(_inline(line[4:].strip()), styles["h2"]))
            i += 1
            continue

        if line.startswith("#### "):
            story.append(Paragraph(_inline(line[5:].strip()), styles["h3"]))
            i += 1
            continue

        if line.startswith("- ") or re.match(r"\d+\. ", line):
            while i < len(lines) and (lines[i].startswith("- ") or re.match(r"\d+\. ", lines[i])):
                current = lines[i]
                ordered_match = re.match(r"(\d+)\. (.*)", current)
                if ordered_match:
                    marker, text = ordered_match.groups()
                    prefix = f"{marker}."
                else:
                    text = current[2:]
                    prefix = "•"
                story.append(Paragraph(f"{prefix} {_inline(text)}", styles["list"]))
                i += 1
            story.append(Spacer(1, 1 * mm))
            continue

        # Accumulate a Markdown paragraph. Hard source line breaks are collapsed as in
        # ordinary prose, while code-block line breaks are preserved above.
        paragraph = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "```", "- ")) and not re.match(r"\d+\. ", lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        text = " ".join(paragraph)
        style = styles["metadata"] if in_cover and cover_metadata else styles["body"]
        story.append(Paragraph(_inline(text), style))
    return story


def _register_fonts() -> None:
    regular, bold, mono = _font_paths()
    pdfmetrics.registerFont(TTFont("ReportSerif", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportSerif-Bold", str(bold)))
    pdfmetrics.registerFont(TTFont("ReportMono", str(mono)))


def build(source_path: Path, output_path: Path) -> int:
    _register_fonts()
    source = source_path.read_text(encoding="utf-8")
    styles = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    doc = BaseDocTemplate(
        str(output_path), pagesize=A4, title="Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads",
        author="OS-Project", subject="Plain-text academic project report",
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=21 * mm,
        pageCompression=1,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin, page_width - doc.leftMargin - doc.rightMargin,
        page_height - doc.topMargin - doc.bottomMargin, id="body", leftPadding=0,
        rightPadding=0, topPadding=0, bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_page)])
    doc.build(
        _make_story(source, styles),
        canvasmaker=_invariant_canvas,
    )
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised in installation guidance
        raise RuntimeError("pypdf is required to validate the generated PDF") from exc
    reader = PdfReader(str(output_path), strict=True)
    if len(reader.pages) < 2:
        raise RuntimeError("Report PDF unexpectedly contains fewer than two pages")
    for page_no, page in enumerate(reader.pages, 1):
        if not page.extract_text():
            raise RuntimeError(f"PDF page {page_no} contains no extractable text")
    return len(reader.pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    pages = build(args.source, args.output)
    print(f"Built {args.output} ({pages} pages) from {args.source}")


if __name__ == "__main__":
    main()
