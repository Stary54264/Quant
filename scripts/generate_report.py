#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成固定配置（EMA55 / sh.600000 / 2006–2025）的回测报告 Markdown。

报告写入 strategy/ema55/report.md。计算与渲染逻辑在 report_lib 中，
与 UI 共用同一套口径。

运行（仓库根目录）：
    python3 scripts/generate_report.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from report_lib import run_analysis, build_report_markdown

CODE = "sh.600000"
START_DATE = "2006-01-01"
END_DATE = "2025-12-31"
STRATEGY = "ema55"
REPORT_PATH = ROOT / "strategy" / STRATEGY / "report.md"


def main() -> None:
    analysis = run_analysis(CODE, START_DATE, END_DATE, STRATEGY)
    markdown = build_report_markdown(analysis)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown, encoding="utf-8")
    print(f"报告已写入 {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
