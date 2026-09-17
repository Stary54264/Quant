#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回测报告小程序（Streamlit）。

启动（仓库根目录）：
    streamlit run app/app.py
或在 macOS Finder 双击仓库根目录下的「启动回测.command」。

操作：输入精确个股代码 → 选择起止日期（日历）→ 选择策略 → 生成报告，
报告会在弹窗中直接渲染，并提供 .md 下载。
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import streamlit as st

from report_lib import (
    BacktestInputError,
    build_report_markdown,
    list_strategies,
    report_filename,
    run_analysis,
)

# 数据集覆盖的交易日范围（data/backtest/a_share，2006–2025 固定快照）
MIN_DATE = date(2006, 1, 4)
MAX_DATE = date(2025, 12, 31)

st.set_page_config(page_title="A 股策略回测", page_icon="📈")
st.title("A 股策略回测报告")

# 1. 精确代码（不做模糊匹配/联想，必须与数据中的代码完全一致）
code = st.text_input(
    "个股代码",
    placeholder="如 sh.600000",
    help="必须输入完整代码，不支持名称或模糊搜索；代码对照见 data/backtest/a_share/README.md",
)

# 2. 起止日期（日历选择器，限定在数据区间内）
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "起始日期", value=date(2006, 1, 4), min_value=MIN_DATE, max_value=MAX_DATE
    )
with col2:
    end_date = st.date_input(
        "结束日期", value=date(2025, 12, 31), min_value=MIN_DATE, max_value=MAX_DATE
    )

# 3. 策略列表：自动扫描 strategy/ 下符合约定的子文件夹名
strategy_name = st.selectbox("策略", list_strategies())


@st.dialog("回测报告", width="large")
def show_report(markdown: str, filename: str) -> None:
    """弹窗中渲染报告，并提供 .md 文件下载。"""
    st.markdown(markdown)
    st.divider()
    st.download_button(
        "下载报告 (.md)",
        data=markdown,
        file_name=filename,
        mime="text/markdown",
    )


# 4. 生成按钮
if st.button("生成报告", type="primary"):
    if not code.strip():
        st.error("请输入个股代码，如 sh.600000")
    elif start_date > end_date:
        st.error("起始日期不能晚于结束日期")
    else:
        try:
            with st.spinner("回测运行中，请稍候…"):
                analysis = run_analysis(
                    code.strip().lower(),
                    start_date.isoformat(),
                    end_date.isoformat(),
                    strategy_name,
                )
                markdown = build_report_markdown(analysis)
                filename = report_filename(analysis)
            show_report(markdown, filename)
        except BacktestInputError as exc:
            st.error(str(exc))
