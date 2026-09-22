#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""绘制回测报告中的净值与仓位图。

横轴为时间，左轴为净值（策略与买入持有两条线），右轴为仓位
（半透明背景柱状图）。中文标签优先使用系统自带的宋体/黑体类字体
（macOS 的 Songti、Windows 的微软雅黑/黑体等），找不到中文字体时
回退为英文标签，避免出现方块缺字。
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境不需要图形界面
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# 报告中文字体候选（按平台常见自带字体，无需额外安装）
_CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",       # macOS 宋体
    "/System/Library/Fonts/Supplemental/SIMSUN.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",                           # Windows 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",                         # Windows 黑体
    "C:/Windows/Fonts/simsun.ttc",                         # Windows 宋体
]


def _setup_cjk_font() -> bool:
    """注册首个可用的中文字体；成功返回 True。"""
    for path in _CJK_FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


_CJK = _setup_cjk_font()

_LABELS = (
    {
        "nav": "净值",
        "position": "仓位",
        "strategy": "策略",
        "benchmark": "买入持有",
    }
    if _CJK
    else {
        "nav": "NAV",
        "position": "Position",
        "strategy": "Strategy",
        "benchmark": "Buy & Hold",
    }
)

# PNG 像素宽度（PDF 与 UI 均按容器宽度缩放）
_FIG_WIDTH = 7.2   # inch，约 A4 正文宽
_FIG_HEIGHT = 3.6
_DPI = 200


def build_chart(analysis: dict) -> bytes:
    """根据 run_analysis 的结果绘制净值/仓位图，返回 PNG 字节。"""
    result, benchmark_result = analysis["result"], analysis["bench_result"]

    fig, ax_nav = plt.subplots(
        figsize=(_FIG_WIDTH, _FIG_HEIGHT), dpi=_DPI,
    )
    ax_pos = ax_nav.twinx()

    # 背景仓位柱：放在净值轴下层，半透明
    ax_pos.set_zorder(ax_nav.get_zorder() - 1)
    ax_nav.patch.set_visible(False)
    ax_pos.bar(
        result["date"], result["position"],
        width=1.0, color="#4c78a8", alpha=0.18, align="center",
    )

    (line_strat,) = ax_nav.plot(
        result["date"], result["nav"],
        color="#c0392b", linewidth=1.4, label=_LABELS["strategy"],
    )
    (line_bench,) = ax_nav.plot(
        benchmark_result["date"], benchmark_result["nav"],
        color="#2c3e50", linewidth=1.2, linestyle="--",
        label=_LABELS["benchmark"],
    )

    ax_nav.set_ylabel(_LABELS["nav"])
    ax_pos.set_ylabel(_LABELS["position"])
    ax_pos.set_ylim(0, 1.15)
    ax_pos.set_yticks([0, 0.5, 1])
    if _CJK:
        ax_pos.set_yticklabels(["0%", "50%", "100%"])
    ax_nav.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    ax_nav.margins(x=0.01)

    # 合并两轴图例（仓位用半透明色块表示）
    handles = [
        line_strat,
        line_bench,
        Patch(facecolor="#4c78a8", alpha=0.35, label=_LABELS["position"]),
    ]
    ax_nav.legend(handles=handles, loc="upper left", framealpha=0.85, fontsize=9)

    # x 轴按年标注，避免 20 年数据标签重叠
    ax_nav.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI)
    plt.close(fig)
    return buf.getvalue()
