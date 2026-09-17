#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把回测报告的 Markdown 渲染为 PDF。

只支持本仓库报告生成器用到的 Markdown 子集：
标题（#/##/###）、管道表格、无序列表、引用（>）、加粗（**）、空行。
中文使用 reportlab 内置的 Adobe CID 字体 STSong-Light（宋体），ASCII
字母与数字使用内置标准字体 Times-Roman（Times New Roman 同源字稿），
均无需在各操作系统上寻找或嵌入字体文件，macOS / Windows 显示一致。
中文与字母/数字交界处插入窄间距，避免两类字形挤在一起。
"""

import html
import io
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_FONT = "STSong-Light"   # 中文：宋体
_LATIN = "Times-Roman"   # ASCII 字母/数字：Times New Roman 同源字稿
_LATIN_BOLD = "Times-Bold"
# 中英/中数交界处的窄间距：Times 的不换行空格（约 0.25em，随字号缩放）
_GAP = '<font name="Times-Roman">&#160;</font>'

pdfmetrics.registerFont(UnicodeCIDFont(_FONT))
# 中文没有独立粗体字稿：让 <b> 里的中文仍映射到 STSong-Light，避免找不到字体
pdfmetrics.registerFontFamily(
    _FONT, normal=_FONT, bold=_FONT, italic=_FONT, boldItalic=_FONT,
)

# 连续的 ASCII 可打印字符（字母、数字及报告里出现的 ASCII 标点/空格）
_ASCII_RE = re.compile(r"([\x20-\x7e]+)")
# 汉字（含部首/扩展 A/兼容表意文字）；全角标点不算汉字——它们字形自带留白
_HAN_RE = re.compile(r"[⺀-⿰㐀-䶿一-鿿豈-﫿]")


def _mix_fonts(text: str, bold: bool = False) -> str:
    """ASCII 段套 Times，其余（汉字与中文标点）保留宋体。

    仅在汉字与 ASCII 段交界处插入不换行窄间距；ASCII 段边缘原有的普通
    空格被该间距取代，避免双重空隙。全角标点（（）：，等）两侧不加间距。
    """
    latin = _LATIN_BOLD if bold else _LATIN
    tokens = [t for t in _ASCII_RE.split(text) if t != ""]
    # kind: "latin" | "han"（非 ASCII 段，记录两端是否紧贴汉字）
    segs: list[tuple[str, str, bool, bool]] = []
    for idx, tok in enumerate(tokens):
        if tok.isascii():
            prev_non_ascii = idx > 0 and not tokens[idx - 1].isascii()
            next_non_ascii = idx + 1 < len(tokens) and not tokens[idx + 1].isascii()
            if prev_non_ascii:
                tok = tok.lstrip(" \t")
            if next_non_ascii:
                tok = tok.rstrip(" \t")
            if not tok:
                # 夹在中文之间的纯空白（罕见）：保留一个窄间距
                segs.append(("gap", _GAP, False, False))
                continue
            segs.append(("latin", f'<font name="{latin}">{tok}</font>', False, False))
        else:
            segs.append(("han", tok,
                         bool(_HAN_RE.fullmatch(tok[0])),
                         bool(_HAN_RE.fullmatch(tok[-1]))))

    out = []
    for idx, (kind, rendered, left_han, right_han) in enumerate(segs):
        if kind == "gap":
            out.append(rendered)
            continue
        if idx > 0:
            prev = segs[idx - 1]
            han_latin_boundary = (
                (kind == "latin" and prev[0] == "han" and prev[3])
                or (kind == "han" and left_han and prev[0] == "latin")
            )
            if han_latin_boundary:
                out.append(_GAP)
        out.append(rendered)
    return "".join(out)

_styles = {
    "h1": ParagraphStyle("h1", fontName=_FONT, fontSize=18, leading=24,
                         spaceBefore=6, spaceAfter=12, alignment=TA_LEFT),
    "h2": ParagraphStyle("h2", fontName=_FONT, fontSize=14, leading=20,
                         spaceBefore=14, spaceAfter=8, alignment=TA_LEFT),
    "h3": ParagraphStyle("h3", fontName=_FONT, fontSize=12, leading=18,
                         spaceBefore=10, spaceAfter=6, alignment=TA_LEFT),
    "body": ParagraphStyle("body", fontName=_FONT, fontSize=10.5, leading=17,
                           spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName=_FONT, fontSize=10.5,
                             leading=17, leftIndent=16, bulletIndent=4, spaceAfter=2),
    "quote": ParagraphStyle("quote", fontName=_FONT, fontSize=9.5, leading=15,
                            leftIndent=10, textColor=colors.grey, spaceAfter=4),
    "table": ParagraphStyle("table", fontName=_FONT, fontSize=10, leading=14),
}


def _inline(text: str) -> str:
    """转义 HTML 特殊字符，还原 **加粗**，并按字形分配中英文字体。"""
    text = html.escape(text)
    # 捕获组 split：奇数段为 ** ** 内的粗体文本
    parts = re.split(r"\*\*(.+?)\*\*", text)
    out = []
    for idx, part in enumerate(parts):
        bold = idx % 2 == 1
        rendered = _mix_fonts(part, bold=bold)
        out.append(f"<b>{rendered}</b>" if bold else rendered)
    return "".join(out)


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells)


def _table_widths(rows: list[list[str]], usable: float) -> list[float]:
    """按各列内容宽度（中文按 2 个单位计）分配列宽。"""
    weights = [1.0] * max(len(r) for r in rows)
    for r in rows:
        for i, c in enumerate(r):
            n_gaps = c.count("&#160;")  # 交界处的窄间距
            plain = re.sub(r"&[#a-zA-Z0-9]+;", "", re.sub(r"<[^>]+>", "", c))
            width = sum(2 if ord(ch) > 127 else 1 for ch in plain)
            weights[i] = max(weights[i], width + 0.5 * n_gaps)
    total = sum(weights)
    return [usable * w / total for w in weights]


def markdown_to_pdf(markdown: str) -> bytes:
    """把报告 Markdown 字符串渲染为 PDF，返回 PDF 字节内容。"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48,
        title="回测报告",
    )
    story = []
    lines = markdown.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line:
            story.append(Spacer(1, 4))
            i += 1
            continue

        # 表格块：连续的 | ... | 行（含一行 --- 分隔）
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = _split_row(lines[i])
                if not _is_separator(cells):
                    rows.append([_inline(c) for c in cells])
                i += 1
            if rows:
                usable = A4[0] - 96
                table = Table(
                    [[Paragraph(c, _styles["table"]) for c in r] for r in rows],
                    colWidths=_table_widths(rows, usable),
                    repeatRows=1,
                )
                table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), _FONT),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f5")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(Spacer(1, 4))
                story.append(table)
                story.append(Spacer(1, 8))
            continue

        if line.startswith("### "):
            story.append(Paragraph(_inline(line[4:]), _styles["h3"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), _styles["h2"]))
        elif line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), _styles["h1"]))
        elif line.startswith("- "):
            story.append(Paragraph(_inline(line[2:]), _styles["bullet"], bulletText="•"))
        elif line.startswith("> "):
            story.append(Paragraph(_inline(line[2:]), _styles["quote"]))
        else:
            story.append(Paragraph(_inline(line), _styles["body"]))
        i += 1

    doc.build(story)
    return buf.getvalue()
