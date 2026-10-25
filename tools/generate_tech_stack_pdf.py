"""Generate the VideoRAG technical stack and module reference PDF.

Usage:
    python tools/generate_tech_stack_pdf.py
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "VideoRAG_Technical_Stack_and_Modules.md"
DEFAULT_OUTPUT = ROOT / "docs" / "VideoRAG_Technical_Stack_and_Modules.pdf"

NAVY = HexColor("#0B132B")
BLUE = HexColor("#1C5D99")
CYAN = HexColor("#22A6B3")
ORANGE = HexColor("#F59E0B")
INK = HexColor("#172033")
MUTED = HexColor("#5E6B7E")
PALE = HexColor("#EFF5FA")
PALE_BLUE = HexColor("#E8F1F8")
LINE = HexColor("#CBD8E5")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str, str, str]:
    """Register readable Unicode fonts, with built-in fallbacks."""
    candidates = [
        (
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("C:/Windows/Fonts/segoeuii.ttf"),
            Path("C:/Windows/Fonts/consola.ttf"),
            Path("C:/Windows/Fonts/consolab.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
        ),
    ]
    for regular, bold, italic, mono, mono_bold in candidates:
        if all(path.exists() for path in (regular, bold, italic, mono, mono_bold)):
            pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("ReportSans-Italic", str(italic)))
            pdfmetrics.registerFont(TTFont("ReportMono", str(mono)))
            pdfmetrics.registerFont(TTFont("ReportMono-Bold", str(mono_bold)))
            pdfmetrics.registerFontFamily(
                "ReportSans",
                normal="ReportSans",
                bold="ReportSans-Bold",
                italic="ReportSans-Italic",
                boldItalic="ReportSans-Bold",
            )
            return (
                "ReportSans",
                "ReportSans-Bold",
                "ReportSans-Italic",
                "ReportMono",
                "ReportMono-Bold",
            )
    return ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier", "Courier-Bold")


FONT, FONT_BOLD, FONT_ITALIC, FONT_MONO, FONT_MONO_BOLD = register_fonts()


def build_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName=FONT,
        fontSize=8.7,
        leading=12.7,
        textColor=INK,
        spaceAfter=5,
        allowWidows=0,
        allowOrphans=0,
    )
    styles["Small"] = ParagraphStyle(
        "Small",
        parent=styles["Body"],
        fontSize=7.5,
        leading=10.2,
        textColor=MUTED,
    )
    styles["H1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName=FONT_BOLD,
        fontSize=17,
        leading=21,
        textColor=NAVY,
        spaceBefore=13,
        spaceAfter=7,
        keepWithNext=True,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName=FONT_BOLD,
        fontSize=12.2,
        leading=15.5,
        textColor=BLUE,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )
    styles["H3"] = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName=FONT_BOLD,
        fontSize=9.8,
        leading=12.5,
        textColor=CYAN,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=styles["Body"],
        leftIndent=12,
        firstLineIndent=-7,
        bulletIndent=3,
        spaceAfter=3,
    )
    styles["Number"] = ParagraphStyle(
        "Number",
        parent=styles["Body"],
        leftIndent=16,
        firstLineIndent=-12,
        spaceAfter=3,
    )
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=styles["Small"],
        fontName=FONT_BOLD,
        fontSize=7.1,
        leading=9.2,
        textColor=WHITE,
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=styles["Small"],
        fontSize=6.9,
        leading=9.2,
        textColor=INK,
    )
    styles["Code"] = ParagraphStyle(
        "Code",
        fontName=FONT_MONO,
        fontSize=6.6,
        leading=9,
        textColor=HexColor("#DCE7F3"),
        leftIndent=0,
        rightIndent=0,
        spaceAfter=0,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption",
        parent=styles["Small"],
        fontName=FONT_ITALIC,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceBefore=4,
        spaceAfter=8,
    )
    styles["CoverKicker"] = ParagraphStyle(
        "CoverKicker",
        fontName=FONT_BOLD,
        fontSize=10,
        leading=13,
        textColor=CYAN,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        fontName=FONT_BOLD,
        fontSize=29,
        leading=34,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    styles["CoverSub"] = ParagraphStyle(
        "CoverSub",
        fontName=FONT,
        fontSize=12,
        leading=17,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceAfter=17,
    )
    return styles


STYLES = build_styles()


class ReportDocTemplate(SimpleDocTemplate):
    """Document template that collects headings into a table of contents."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        levels = {"H1": 0, "H2": 1, "H3": 2}
        if style_name not in levels:
            return
        text = flowable.getPlainText()
        key = f"heading-{self.seq.nextf('heading')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, levels[style_name], closed=False)
        self.notify("TOCEntry", (levels[style_name], text, self.page, key))


def inline_markup(value: str) -> str:
    """Convert the limited inline Markdown used by the report into ReportLab XML."""
    escaped = html.escape(value.strip())
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<link href="{html.escape(m.group(2), quote=True)}" color="#1C5D99">{m.group(1)}</link>',
        escaped,
    )
    escaped = re.sub(
        r"`([^`]+)`",
        lambda m: f'<font name="{FONT_MONO}" color="#8A4B08">{m.group(1)}</font>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def arrow(drawing: Drawing, x1: float, y1: float, x2: float, y2: float, color=BLUE):
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.4))
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        points = [x2, y2, x2 - 6 * direction, y2 + 3.5, x2 - 6 * direction, y2 - 3.5]
    else:
        direction = 1 if y2 > y1 else -1
        points = [x2, y2, x2 - 3.5, y2 - 6 * direction, x2 + 3.5, y2 - 6 * direction]
    drawing.add(Polygon(points, fillColor=color, strokeColor=color))


def architecture_diagram(width: float) -> Drawing:
    """Create a compact vector architecture diagram."""
    w = min(width, 510)
    h = 275
    d = Drawing(w, h)

    def box(x, y, bw, bh, title, subtitle, fill, border=BLUE):
        d.add(Rect(x, y, bw, bh, rx=6, ry=6, fillColor=fill, strokeColor=border, strokeWidth=1.1))
        d.add(String(x + bw / 2, y + bh - 15, title, textAnchor="middle", fontName=FONT_BOLD, fontSize=8, fillColor=NAVY))
        d.add(String(x + bw / 2, y + 9, subtitle, textAnchor="middle", fontName=FONT, fontSize=6.3, fillColor=MUTED))

    box(w / 2 - 70, 225, 140, 35, "React forensic dashboard", "upload • chat • cases • playback", PALE_BLUE)
    box(35, 162, 125, 38, "FastAPI boundary", "routers • schemas • static media", PALE)
    box(w / 2 - 62, 162, 124, 38, "Ingestion orchestrator", "background task pipeline", HexColor("#FFF4DB"), ORANGE)
    box(w - 160, 162, 125, 38, "Retrieval agent", "decouple • retrieve • merge", PALE_BLUE)

    box(15, 92, 108, 42, "Media services", "OpenCV • FFmpeg • Whisper", PALE)
    box(133, 92, 108, 42, "Vision-language", "OpenCLIP • BLIP • OCR", PALE)
    box(251, 92, 108, 42, "Structured vision", "YOLO • scene graph", PALE)
    box(369, 92, 126, 42, "Legal evidence", "speakers • testimony • citations", PALE)

    box(45, 22, 125, 40, "Filesystem", "videos • frames • audio • metadata", HexColor("#F4F7FA"), MUTED)
    box(w / 2 - 63, 22, 126, 40, "ChromaDB", "image • caption • speech • OCR", HexColor("#E6F6F7"), CYAN)
    box(w - 170, 22, 125, 40, "SQLite", "objects • cases • audit logs", HexColor("#FFF4DB"), ORANGE)

    arrow(d, w / 2, 225, w / 2, 202)
    arrow(d, w / 2 - 70, 225, 98, 202)
    arrow(d, w / 2 + 70, 225, w - 98, 202)
    arrow(d, w / 2, 162, 69, 136, ORANGE)
    arrow(d, w / 2, 162, 187, 136, ORANGE)
    arrow(d, w / 2, 162, 305, 136, ORANGE)
    arrow(d, w / 2, 162, 432, 136, ORANGE)
    arrow(d, 69, 92, 107, 64, MUTED)
    arrow(d, 187, 92, w / 2, 64, CYAN)
    arrow(d, 305, 92, w - 108, 64, ORANGE)
    arrow(d, 432, 92, w - 108, 64, ORANGE)
    arrow(d, w - 98, 162, w / 2 + 54, 64, BLUE)
    arrow(d, w - 98, 162, w - 108, 64, BLUE)

    d.add(Rect(w - 155, 225, 120, 35, rx=6, ry=6, fillColor=HexColor("#FCE8EF"), strokeColor=HexColor("#B33A5B"), strokeWidth=1))
    d.add(String(w - 95, 245, "Groq API", textAnchor="middle", fontName=FONT_BOLD, fontSize=8, fillColor=HexColor("#8D2442")))
    d.add(String(w - 95, 233, "external reasoning boundary", textAnchor="middle", fontName=FONT, fontSize=6.3, fillColor=MUTED))
    arrow(d, w - 98, 200, w - 95, 223, HexColor("#B33A5B"))
    return d


def make_table(rows: list[list[str]], available_width: float) -> LongTable:
    columns = len(rows[0])
    lengths = [1.0] * columns
    for col in range(columns):
        samples = [re.sub(r"[`*_]", "", row[col]) for row in rows[: min(len(rows), 30)]]
        lengths[col] = max(7, min(48, max((len(item) for item in samples), default=7)))
    total = sum(lengths)
    widths = [available_width * length / total for length in lengths]
    if columns == 2:
        widths = [available_width * 0.31, available_width * 0.69]
    elif columns == 3:
        widths = [available_width * 0.23, available_width * 0.30, available_width * 0.47]
    elif columns == 4:
        widths = [available_width * 0.14, available_width * 0.25, available_width * 0.25, available_width * 0.36]

    formatted = []
    for row_index, row in enumerate(rows):
        style = STYLES["TableHeader"] if row_index == 0 else STYLES["TableCell"]
        formatted.append([Paragraph(inline_markup(cell), style) for cell in row])

    table = LongTable(formatted, colWidths=widths, repeatRows=1, splitByRow=True, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
            ]
        )
    )
    return table


def markdown_flowables(source: str, available_width: float, source_dir: Path) -> list:
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("## 1.")), 0)
    lines = lines[start:]
    story = []
    paragraph_lines: list[str] = []
    bullet_items: list[str] = []
    number_items: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            story.append(Paragraph(inline_markup(" ".join(part.strip() for part in paragraph_lines)), STYLES["Body"]))
            paragraph_lines = []

    def flush_bullets():
        nonlocal bullet_items
        if bullet_items:
            items = [ListItem(Paragraph(inline_markup(item), STYLES["Body"]), leftIndent=10) for item in bullet_items]
            story.append(ListFlowable(items, bulletType="bullet", bulletFontName=FONT, bulletFontSize=7, leftIndent=15, bulletColor=BLUE, spaceAfter=5))
            bullet_items = []

    def flush_numbers():
        nonlocal number_items
        if number_items:
            items = [ListItem(Paragraph(inline_markup(item), STYLES["Body"]), leftIndent=12) for item in number_items]
            story.append(ListFlowable(items, bulletType="1", start="1", bulletFontName=FONT_BOLD, bulletFontSize=7.5, leftIndent=19, bulletColor=BLUE, spaceAfter=5))
            number_items = []

    def flush_all():
        flush_paragraph()
        flush_bullets()
        flush_numbers()

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if stripped.startswith("```"):
            flush_all()
            if in_code:
                pre = Preformatted("\n".join(code_lines), STYLES["Code"])
                wrapper = Table([[pre]], colWidths=[available_width], hAlign="LEFT")
                wrapper.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("BOX", (0, 0), (-1, -1), 0.5, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
                story.extend([wrapper, Spacer(1, 6)])
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if stripped == "<!-- ARCHITECTURE_DIAGRAM -->":
            flush_all()
            story.extend([Spacer(1, 4), architecture_diagram(available_width), Paragraph("Figure 1. Source-derived runtime architecture and data boundary.", STYLES["Caption"])])
            i += 1
            continue
        if stripped == "<!-- DASHBOARD_SCREENSHOT -->":
            flush_all()
            image_path = ROOT / "kubrik.png"
            if image_path.exists():
                img = Image(str(image_path))
                max_width = available_width * 0.72
                max_height = 175 * mm
                scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                img.hAlign = "CENTER"
                story.extend([Spacer(1, 5), img, Paragraph("Figure 2. Case panel for speaker confirmation and role management.", STYLES["Caption"])])
            i += 1
            continue

        image_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image_match:
            flush_all()
            caption, target = image_match.groups()
            image_path = (source_dir / target).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Markdown image not found: {image_path}")
            img = Image(str(image_path))
            max_width = available_width
            max_height = 165 * mm
            scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
            img.hAlign = "CENTER"
            story.extend(
                [
                    Spacer(1, 5),
                    img,
                    Paragraph(
                        f"Figure. {html.escape(caption)}.", STYLES["Caption"]
                    ),
                ]
            )
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            flush_all()
            table_lines = [stripped]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines]
            story.extend([make_table(rows, available_width), Spacer(1, 7)])
            continue

        heading = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading:
            flush_all()
            level = len(heading.group(1)) - 1
            style_name = {1: "H1", 2: "H2", 3: "H3"}.get(level, "H3")
            story.append(Paragraph(inline_markup(heading.group(2)), STYLES[style_name]))
            i += 1
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if bullet:
            flush_paragraph()
            flush_numbers()
            bullet_items.append(bullet.group(1))
            i += 1
            continue
        if numbered:
            flush_paragraph()
            flush_bullets()
            number_items.append(numbered.group(1))
            i += 1
            continue

        if stripped == "---":
            flush_all()
            story.append(HRFlowable(width="100%", thickness=0.7, color=LINE, spaceBefore=6, spaceAfter=7))
            i += 1
            continue
        if not stripped:
            flush_all()
            i += 1
            continue

        if (bullet_items or number_items) and raw.startswith("  "):
            target = bullet_items if bullet_items else number_items
            target[-1] += " " + stripped
        else:
            flush_bullets()
            flush_numbers()
            paragraph_lines.append(stripped)
        i += 1

    flush_all()
    return story


def cover_story(source_text: str, output_name: str) -> list:
    commit_match = re.search(r"\*\*Code snapshot:\*\*\s*`([^`]+)`", source_text)
    date_match = re.search(r"^\*\*Report date:\*\*\s*(.+?)\s*$", source_text, re.MULTILINE)
    commit = commit_match.group(1) if commit_match else "unknown"
    report_date = date_match.group(1).strip() if date_match else ""

    badges = Table(
        [
            [Paragraph("CODE-DERIVED", STYLES["TableHeader"]), Paragraph("FULL MODULE INVENTORY", STYLES["TableHeader"]), Paragraph("PDF HANDOVER", STYLES["TableHeader"])],
        ],
        colWidths=[42 * mm, 53 * mm, 42 * mm],
        hAlign="CENTER",
    )
    badges.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), BLUE), ("BACKGROUND", (1, 0), (1, 0), CYAN), ("BACKGROUND", (2, 0), (2, 0), ORANGE), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))

    metadata = Table(
        [
            [Paragraph("Repository", STYLES["Small"]), Paragraph("MogSquad-FinalYear/VideoRAG", STYLES["Body"])],
            [Paragraph("Snapshot", STYLES["Small"]), Paragraph(f'<font name="{FONT_MONO}">{commit}</font>', STYLES["Body"])],
            [Paragraph("Report date", STYLES["Small"]), Paragraph(report_date, STYLES["Body"])],
            [Paragraph("Deliverable", STYLES["Small"]), Paragraph(output_name, STYLES["Body"])],
        ],
        colWidths=[33 * mm, 112 * mm],
        hAlign="CENTER",
    )
    metadata.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE_BLUE), ("BACKGROUND", (1, 0), (1, -1), WHITE), ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))

    return [
        Spacer(1, 24 * mm),
        Paragraph("VIDEORAG • TECHNICAL HANDOVER", STYLES["CoverKicker"]),
        Paragraph("Technical Stack &amp;<br/>Module Reference", STYLES["CoverTitle"]),
        Paragraph("A source-derived guide to architecture, ingestion, retrieval, APIs, data stores, frontend modules, evaluation, fallbacks, and operational constraints.", STYLES["CoverSub"]),
        badges,
        Spacer(1, 18 * mm),
        metadata,
        Spacer(1, 16 * mm),
        Paragraph("VideoRAG for Courtroom Evidence", ParagraphStyle("CoverFooter", parent=STYLES["CoverSub"], fontName=FONT_BOLD, fontSize=11, textColor=NAVY, spaceAfter=5)),
        Paragraph("Cross-session speaker identification • testimony contradiction detection • citation-verified video question answering", ParagraphStyle("CoverTagline", parent=STYLES["Small"], alignment=TA_CENTER, fontSize=8.5, leading=12)),
        PageBreak(),
    ]


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setTitle("VideoRAG Technical Stack and Module Reference")
    canvas.setAuthor("MogSquad-FinalYear / VideoRAG")
    canvas.setSubject("Source-derived technical stack, architecture, and module inventory")
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 7 * mm, width, 7 * mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN)
    canvas.rect(0, height - 8 * mm, width * 0.62, 1 * mm, fill=1, stroke=0)
    canvas.setFillColor(ORANGE)
    canvas.rect(width * 0.62, height - 8 * mm, width * 0.38, 1 * mm, fill=1, stroke=0)

    if doc.page > 1:
        canvas.setFont(FONT, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(19 * mm, height - 13 * mm, "VideoRAG • Technical Stack and Module Reference")
        canvas.drawRightString(width - 19 * mm, 11 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(19 * mm, 15 * mm, width - 19 * mm, 15 * mm)
    canvas.restoreState()


def build_pdf(source_path: Path, output_path: Path):
    source_text = source_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ReportDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=19 * mm,
        title="VideoRAG Technical Stack and Module Reference",
        author="MogSquad-FinalYear / VideoRAG",
        subject="Source-derived technical stack, architecture, and module inventory",
    )

    story = cover_story(source_text, output_path.name)
    story.append(Paragraph("Contents", STYLES["H1"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName=FONT_BOLD, fontSize=9.2, leading=13, leftIndent=0, firstLineIndent=0, textColor=NAVY, spaceBefore=4),
        ParagraphStyle("TOC2", fontName=FONT, fontSize=8.2, leading=11, leftIndent=12, firstLineIndent=0, textColor=BLUE),
        ParagraphStyle("TOC3", fontName=FONT, fontSize=7.4, leading=10, leftIndent=24, firstLineIndent=0, textColor=MUTED),
    ]
    story.extend([toc, PageBreak()])
    story.extend(markdown_flowables(source_text, doc.width, source_path.parent))
    doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_pdf(args.source.resolve(), args.output.resolve())
    print(f"Generated {args.output.resolve()}")


if __name__ == "__main__":
    main()
