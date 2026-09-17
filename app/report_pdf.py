#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把回测报告的 Markdown 渲染为 PDF。

只支持本仓库报告生成器用到的 Markdown 子集：
标题（#/##/###）、管道表格、无序列表、引用（>）、加粗（**）、空行。
中文使用 reportlab 内置的 Adobe CID 字体 STSong-Light，无需在各操作系统
上寻找或嵌入字体文件，macOS / Windows 均可正常显示。
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

_FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(_FONT))

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
    """转义 HTML 特殊字符后还原 **加粗** 标记。"""
    text = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _split_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return [_inline(c) for c in cells]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells)


def _table_widths(rows: list[list[str]], usable: float) -> list[float]:
    """按各列内容宽度（中文按 2 个单位计）分配列宽。"""
    weights = [1.0] * max(len(r) for r in rows)
    for r in rows:
        for i, c in enumerate(r):
            plain = re.sub(r"<[^>]+>", "", c)
            width = sum(2 if ord(ch) > 127 else 1 for ch in plain)
            weights[i] = max(weights[i], width)
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
                    rows.append(cells)
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
