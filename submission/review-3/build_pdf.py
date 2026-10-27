#!/usr/bin/env python3
"""Generate Final_Research_Paper.pdf from the DOCX, extracting inline images properly."""

import os
import sys
import tempfile
import shutil
from docx import Document
from docx.shared import Pt, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib import colors
    from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, Frame
except ImportError:
    print("reportlab not installed. Install with: pip install reportlab")
    sys.exit(1)

BASE = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(BASE, "Final_Research_Paper.docx")
OUTPUT_PDF = os.path.join(BASE, "Final_Research_Paper.pdf")


def get_alignment(paragraph):
    align = paragraph.alignment
    if align == WD_ALIGN_PARAGRAPH.CENTER:
        return TA_CENTER
    elif align == WD_ALIGN_PARAGRAPH.RIGHT:
        return TA_RIGHT
    elif align == WD_ALIGN_PARAGRAPH.JUSTIFY:
        return TA_JUSTIFY
    return TA_LEFT


def run_to_html(run):
    text = run.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = []
    if run.bold:
        parts.append(f"<b>{text}</b>")
    elif run.italic:
        parts.append(f"<i>{text}</i>")
    else:
        parts.append(text)
    return "".join(parts)


def extract_images_from_docx(doc, tmpdir):
    """Extract images from DOCX relationships and save to tmpdir.
    Returns list of (paragraph_index, image_path) for inline shapes."""
    # Build mapping: rel_id -> image file path
    rel_images = {}
    for rel in doc.part.rels.values():
        if 'image' in rel.reltype:
            img_data = rel.target_part.blob
            # Determine extension from content type
            content_type = rel.target_part.content_type
            ext = 'png'
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = 'jpg'
            elif 'svg' in content_type:
                ext = 'svg'
            fname = f"docx_image_{len(rel_images)}.{ext}"
            fpath = os.path.join(tmpdir, fname)
            with open(fpath, 'wb') as f:
                f.write(img_data)
            rel_images[rel.rId] = fpath

    # Map inline shapes to their paragraph indices
    # An inline shape is inside a run, which is inside a paragraph
    shape_paragraph_map = []
    for para_idx, para in enumerate(doc.paragraphs):
        # Check each run for inline shapes (drawing elements)
        for run in para.runs:
            # Look for blip elements (image references) in the run's XML
            blips = run._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
            if not blips:
                blips = run._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}anchor//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
            for blip in blips:
                embed = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if embed and embed in rel_images:
                    shape_paragraph_map.append((para_idx, rel_images[embed]))
    return shape_paragraph_map


def build_flowables(doc, styles, shape_paragraph_map):
    flowables = []

    # Set of paragraph indices that have inline images
    para_with_image = {}
    for para_idx, img_path in shape_paragraph_map:
        para_with_image[para_idx] = img_path

    for para_idx, p in enumerate(doc.paragraphs):
        style_name = p.style.name if p.style else ""
        text = p.text.strip()

        # Handle headings
        if "Heading" in style_name:
            level = 1
            try:
                level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                pass
            heading_style = styles[f"Heading{level}"]
            flowables.append(Paragraph(text, heading_style))
            continue

        # Handle empty paragraphs (but check for images first)
        if para_idx in para_with_image:
            img_path = para_with_image[para_idx]
            if os.path.exists(img_path):
                img = Image(img_path)
                max_width = 5.5 * inch
                if img.imageWidth > max_width:
                    ratio = max_width / img.imageWidth
                    img.drawWidth = max_width
                    img.drawHeight = img.imageHeight * ratio
                img.hAlign = "CENTER"
                flowables.append(Spacer(1, 12))
                flowables.append(img)
                flowables.append(Spacer(1, 12))
            continue

        if not text:
            flowables.append(Spacer(1, 6))
            continue

        # Build paragraph with runs
        para_html = ""
        for run in p.runs:
            para_html += run_to_html(run)

        if not para_html.strip():
            flowables.append(Spacer(1, 6))
            continue

        align = get_alignment(p)
        para_style = ParagraphStyle(
            "CustomPara", parent=styles["Normal"],
            alignment=align, spaceAfter=6, fontSize=12, leading=18
        )
        flowables.append(Paragraph(para_html, para_style))

    # Handle tables
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                row_data.append(Paragraph(cell_text, styles["Normal"]))
            table_data.append(row_data)

        col_count = len(table_data[0]) if table_data else 1
        available_width = A4[0] - 2 * 2.54 * cm
        col_width = available_width / col_count
        col_widths = [col_width] * col_count

        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.85, 0.85)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        flowables.append(Spacer(1, 6))
        flowables.append(t)
        flowables.append(Spacer(1, 6))

    return flowables


def add_page_number(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Times-Roman", 10)
    page_num = canvas_obj.getPageNumber()
    canvas_obj.drawCentredString(A4[0] / 2, 1.5 * cm, str(page_num))
    canvas_obj.restoreState()


def build_pdf(docx_path, output_pdf):
    doc = Document(docx_path)

    # Extract images to temp directory
    tmpdir = tempfile.mkdtemp()
    try:
        shape_paragraph_map = extract_images_from_docx(doc, tmpdir)
        print(f"Extracted {len(shape_paragraph_map)} images from DOCX")

        doc_template = BaseDocTemplate(
            output_pdf,
            pagesize=A4,
            leftMargin=2.54 * cm,
            rightMargin=2.54 * cm,
            topMargin=2.54 * cm,
            bottomMargin=2.54 * cm,
            title="Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads",
            author="[Student Name(s)]",
        )

        frame = Frame(
            doc_template.leftMargin, doc_template.bottomMargin,
            doc_template.width, doc_template.height,
            id="normal"
        )
        template = PageTemplate(id="main", frames=[frame], onPage=add_page_number)
        doc_template.addPageTemplates([template])

        styles = getSampleStyleSheet()
        styles["Heading1"].fontName = "Times-Bold"
        styles["Heading1"].fontSize = 14
        styles["Heading2"].fontName = "Times-Bold"
        styles["Heading2"].fontSize = 13
        styles["Heading3"].fontName = "Times-Bold"
        styles["Heading3"].fontSize = 12
        styles["Normal"].fontName = "Times-Roman"
        styles["Normal"].fontSize = 12
        styles["Normal"].leading = 18
        styles["Normal"].alignment = TA_JUSTIFY

        flowables = build_flowables(doc, styles, shape_paragraph_map)
        doc_template.build(flowables)

        size = os.path.getsize(output_pdf)
        print(f"PDF saved to {output_pdf}")
        print(f"PDF size: {size / 1024:.0f} KB")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if not os.path.exists(DOCX_PATH):
        print(f"DOCX not found: {DOCX_PATH}")
        sys.exit(1)
    build_pdf(DOCX_PATH, OUTPUT_PDF)
