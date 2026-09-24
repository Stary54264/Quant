#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回测报告小程序（Streamlit）。

启动（仓库根目录）：
    streamlit run app/app.py
或在 Finder / 资源管理器双击 app/ 下对应系统的「启动程序」启动器。

操作：选择市场 → 输入精确个股代码 → 选择起止日期（日历）→ 选择策略 →
生成报告，报告会在弹窗中直接渲染，并提供 PDF 下载；关闭弹窗点右上角的 ×。
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

from markets import MARKETS, DEFAULT_MARKET, available_markets
from generate_report import (
    BacktestInputError,
    generate_report,
    list_strategies,
    report_filename,
)
from report_pdf import markdown_to_pdf

st.set_page_config(page_title="姜砚尊最聪明最帅", page_icon="📈")
st.title("姜砚尊最聪明最帅")

# 任何市场的数据都缺失时无法运行：列出全部应有路径后停下
_markets = available_markets()
if not _markets:
    st.error("未找到任何市场的行情数据，请检查以下文件是否存在：")
    for _m in MARKETS.values():
        st.code(f"{_m.stocks_path}\n{_m.indices_path}")
    st.stop()

# 初始默认值统一放 session_state，控件只绑定 key，
# 避免同时给 value= 和 session_state 赋值引发警告
_strategies = list_strategies()
_default_key = DEFAULT_MARKET if DEFAULT_MARKET in _markets else next(iter(_markets))
_default_market = _markets[_default_key]
st.session_state.setdefault("market", _default_key)
st.session_state.setdefault("code_input", "")
st.session_state.setdefault("start_date", _default_market.min_date)
st.session_state.setdefault("end_date", _default_market.max_date)
st.session_state.setdefault("strategy", None)  # 下拉框默认空白，必须主动选择


def _on_market_change() -> None:
    """切换市场：代码各市场不通用需清空；日期主动夹到新区间。

    Streamlit 的 keyed date_input 在值超出新 min/max 时会静默重置为今天，
    因此必须在这里先夹好，不能依赖控件自身行为。
    """
    m = MARKETS[st.session_state["market"]]
    st.session_state["code_input"] = ""
    for key in ("start_date", "end_date"):
        d = st.session_state[key]
        st.session_state[key] = min(max(d, m.min_date), m.max_date)


# 1. 市场（只列出数据齐全的市场）
market_key = st.selectbox(
    "市场",
    list(_markets),
    format_func=lambda k: MARKETS[k].display_name,
    key="market",
    on_change=_on_market_change,
)
market = MARKETS[market_key]

# 2. 精确代码（不做模糊匹配/联想，必须与数据中的代码完全一致）
code = st.text_input(
    "个股代码",
    placeholder=f"如 {market.code_example}",
    key="code_input",
    help="必须输入完整代码，不支持名称或模糊搜索；"
         f"代码对照见 {market.readme.relative_to(ROOT)}",
)

# 3. 起止日期（日历选择器，限定在所选市场的数据区间内）
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "起始日期", min_value=market.min_date, max_value=market.max_date,
        key="start_date",
    )
with col2:
    end_date = st.date_input(
        "结束日期", min_value=market.min_date, max_value=market.max_date,
        key="end_date",
    )

# 4. 策略列表：自动扫描 strategy/ 下符合约定的子文件夹名，默认空白待选
strategy_name = st.selectbox(
    "策略",
    _strategies,
    key="strategy",
    index=None,
    placeholder="请选择策略",
)


@st.dialog("回测报告", width="large")
def show_report(markdown: str, chart: bytes, pdf_bytes: bytes, pdf_filename: str) -> None:
    """弹窗中渲染报告并提供 PDF 下载；关闭弹窗用右上角的 ×。"""
    # 在占位符位置内联插入图表，其余部分正常渲染 Markdown
    if "[[CHART]]" in markdown:
        before, after = markdown.split("[[CHART]]", 1)
        st.markdown(before)
        st.image(chart, width="stretch")
        if after.strip():
            st.markdown(after)
    else:
        st.markdown(markdown)
    st.divider()
    st.download_button(
        "下载报告 (PDF)",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True,
    )


# 5. 生成按钮
if st.button("生成报告", type="primary"):
    if not code.strip():
        st.error(f"请输入个股代码，如 {market.code_example}")
    elif not strategy_name:
        st.error("请选择策略")
    elif start_date > end_date:
        st.error("起始日期不能晚于结束日期")
    else:
        try:
            with st.spinner("回测运行中，请稍候…"):
                # 报告只在弹窗展示，不落盘；需要保存请在弹窗下载 PDF
                markdown, analysis = generate_report(
                    code.strip().lower(),
                    start_date.isoformat(),
                    end_date.isoformat(),
                    strategy_name,
                    market=market_key,
                )
                chart = analysis["chart"]
                pdf_bytes = markdown_to_pdf(markdown, chart)
                pdf_filename = report_filename(analysis)
            show_report(markdown, chart, pdf_bytes, pdf_filename)
        except BacktestInputError as exc:
            st.error(str(exc))
