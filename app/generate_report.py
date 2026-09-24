#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用回测报告生成：给定标的、区间与策略，调用评价/检验函数生成 Markdown
报告，每份报告同时附带一张净值/仓位图（PNG）。

既可作为模块被 UI 直接调用（弹窗展示 md 与图，并提供 PDF 下载），也可在
命令行运行（报告 Markdown 直接打印到标准输出，不落盘，图表节在终端剔除）：

    python3 app/generate_report.py sh.600000 2006-01-01 2025-12-31 ema55
    python3 app/generate_report.py sh.000300 2006-01-04 2025-12-31 ema55
    python3 app/generate_report.py us.14593 2006-01-03 2024-12-31 ema55 -m us
    python3 app/generate_report.py us.SPX 2006-01-03 2025-12-31 ema55 -m us

策略发现约定：``strategy/<策略名>/`` 子文件夹的 ``__init__.py`` 暴露
``generate_position(data)`` 统一入口，并可提供 ``STRATEGY_RULES`` 元信息。
"""

import argparse
import importlib
import io
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 无头环境不需要图形界面
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

from markets import MARKETS, DEFAULT_MARKET, get_market
from backtest.query import query
from backtest.backtest import backtest, DEFAULT_COMMISSION
from backtest.evaluate_strategy import (
    annualized_sharpe,
    deflated_sharpe,
    max_drawdown,
    max_drawdown_duration,
)
from backtest.test_strategy import test_lookahead_bias

TRADING_DAYS = 252
N_TRIALS = 1          # DSR 的独立试验次数：单一参数配置，未做参数搜索
TRUNCATION_DAYS = 10  # 前视偏差截断测试截去的最近交易日数

STRATEGY_DIR = ROOT / "strategy"

# 默认市场的行情文件（保留作向后兼容别名；实际取数按 market 参数走注册表）
STOCK_DATA_PATH = MARKETS[DEFAULT_MARKET].stocks_path

# Markdown 报告中的图表占位符（UI/PDF 据此插入图片）
CHART_PLACEHOLDER = "[[CHART]]"

# 图表中文字体候选（各平台自带字体，无需额外安装）
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
    for font_path in _CJK_FONT_CANDIDATES:
        if not Path(font_path).exists():
            continue
        try:
            font_manager.fontManager.addfont(font_path)
            name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


_CJK = _setup_cjk_font()

_CHART_LABELS = (
    {"nav": "净值", "position": "仓位", "strategy": "策略",
     "benchmark": "买入持有"}
    if _CJK
    else {"nav": "NAV", "position": "Position", "strategy": "Strategy",
          "benchmark": "Buy & Hold"}
)

# PNG 尺寸（PDF 与 UI 均按容器宽度缩放）
_CHART_WIDTH = 7.2   # inch，约 A4 正文宽
_CHART_HEIGHT = 3.6
_CHART_DPI = 200


class BacktestInputError(ValueError):
    """用户输入有误（代码不存在、区间无行情、策略不符合约定等）。"""


def _validate_market(key: str):
    """校验市场已注册且数据齐全，返回 Market；否则抛 BacktestInputError。"""
    try:
        market = get_market(key)
    except KeyError:
        options = "、".join(MARKETS)
        raise BacktestInputError(
            f"未知市场 {key!r}，可选：{options}")
    missing = [str(p) for p in (market.stocks_path, market.indices_path)
               if not p.exists()]
    if missing:
        raise BacktestInputError(
            f"{market.display_name}数据不完整，缺少：" + "；".join(missing))
    return market


def list_strategies() -> list[str]:
    """扫描 strategy/ 下暴露了 generate_position 的子文件夹名，按名称排序。"""
    names = []
    for child in sorted(STRATEGY_DIR.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            mod = importlib.import_module(f"strategy.{child.name}")
            if callable(getattr(mod, "generate_position", None)):
                names.append(child.name)
    return names


def _load_strategy(strategy_name: str):
    """按子文件夹名加载策略模块，不存在或不符合约定时报明确错误。"""
    init_file = STRATEGY_DIR / strategy_name / "__init__.py"
    if not init_file.exists():
        raise BacktestInputError(
            f"策略 {strategy_name!r} 不存在（strategy/{strategy_name}/ 下无 __init__.py）"
        )
    module = importlib.import_module(f"strategy.{strategy_name}")
    if not callable(getattr(module, "generate_position", None)):
        raise BacktestInputError(
            f"策略 {strategy_name!r} 未暴露 generate_position(data) 入口"
        )
    return module


def _evaluate_returns(returns) -> dict:
    """调用 evaluate_strategy 的全部函数，外加净值/年化等基础统计。"""
    r = np.asarray(returns, dtype=float)
    nav = float((1 + r).cumprod()[-1])
    return {
        "期末净值": nav,
        "总收益": nav - 1,
        "年化收益": nav ** (TRADING_DAYS / len(r)) - 1,
        "年化夏普": annualized_sharpe(r),
        "Deflated Sharpe": deflated_sharpe(r, n_trials=N_TRIALS),
        "最大回撤": max_drawdown(r),
        "最长回撤天数": max_drawdown_duration(r),
    }


def build_chart(analysis: dict) -> bytes:
    """绘制净值/仓位图：时间横轴，左轴净值（策略/买入持有），右轴仓位（半透明柱状）。"""
    result, benchmark_result = analysis["result"], analysis["bench_result"]

    fig, ax_nav = plt.subplots(figsize=(_CHART_WIDTH, _CHART_HEIGHT), dpi=_CHART_DPI)
    ax_pos = ax_nav.twinx()

    # 背景仓位柱放在净值轴下层，半透明
    ax_pos.set_zorder(ax_nav.get_zorder() - 1)
    ax_nav.patch.set_visible(False)
    ax_pos.bar(
        result["date"], result["position"],
        width=1.0, color="#4c78a8", alpha=0.18, align="center",
    )

    (line_strat,) = ax_nav.plot(
        result["date"], result["nav"],
        color="#c0392b", linewidth=1.4, label=_CHART_LABELS["strategy"],
    )
    (line_bench,) = ax_nav.plot(
        benchmark_result["date"], benchmark_result["nav"],
        color="#2c3e50", linewidth=1.2, linestyle="--",
        label=_CHART_LABELS["benchmark"],
    )

    ax_nav.set_ylabel(_CHART_LABELS["nav"])
    ax_pos.set_ylabel(_CHART_LABELS["position"])
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
        Patch(facecolor="#4c78a8", alpha=0.35, label=_CHART_LABELS["position"]),
    ]
    ax_nav.legend(handles=handles, loc="upper left", framealpha=0.85, fontsize=9)

    # x 轴按年标注，避免长区间标签重叠
    ax_nav.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_CHART_DPI)
    plt.close(fig)
    return buf.getvalue()


def run_analysis(
    code: str,
    start_date: str,
    end_date: str,
    strategy_name: str,
    commission: float = DEFAULT_COMMISSION,
    market: str = DEFAULT_MARKET,
) -> dict:
    """跑通单标的单策略的完整分析，返回原始结果（逐日明细、指标、检验结果等）。"""
    m = _validate_market(market)
    module = _load_strategy(strategy_name)
    strategy_label = strategy_name

    code = m.normalize_code(code)
    is_index = m.is_index(code)
    data = query(m.path_for(code), [code], start_date, end_date)
    if data.empty:
        raise BacktestInputError(
            f"{code} 在 {start_date} ~ {end_date} 内无行情记录（请检查代码与区间）"
        )
    data = data.reset_index(drop=True)

    position = module.generate_position(data)
    result = backtest(data, position, commission=commission)
    benchmark = backtest(data, pd.Series(np.ones(len(data))), commission=commission)
    bias = test_lookahead_bias(module.generate_position, data, n_days=TRUNCATION_DAYS)

    return {
        "code": code,
        "market": m.key,
        "market_name": m.display_name,
        "kind": "指数" if is_index else "个股",
        "strategy_name": strategy_name,
        "strategy_label": strategy_label,
        "strategy_rules": getattr(module, "STRATEGY_RULES", []),
        "commission": commission,
        "data": data,
        "result": result,
        "bench_result": benchmark,
        "strat_stats": _evaluate_returns(result["net_ret"]),
        "bench_stats": _evaluate_returns(benchmark["net_ret"]),
        "bias": bias,
        "n_trades": int(result["turnover"].gt(0).sum()),
        "total_cost": float(result["cost"].sum()),
    }


def build_report_markdown(a: dict) -> str:
    """把 run_analysis 的结果渲染为 Markdown 报告字符串。"""
    result = a["result"]
    strat_stats, bench_stats = a["strat_stats"], a["bench_stats"]

    lines = [
        f"# {a['strategy_label']}策略回测报告",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 策略规则",
        "",
    ]
    lines += [f"- {rule}" for rule in a["strategy_rules"]]
    lines += [
        "- 成交时点：信号日 T 收盘确认，T+1 日开盘成交，持仓收益按开盘到开盘计",
        f"- 手续费：单边 {a['commission']:.2%}（{a['commission'] * 1e4:.0f} bp），按换手收取",
        ("- 价格口径：指数原始点位，指数无公司行为、不存在复权，"
         "收益不含成分股分红" if a["kind"] == "指数"
         else "- 价格口径：前复权，收益已含分红再投资"),
        "",
        "## 标的与区间",
        "",
        f"- 市场：{a['market_name']}",
        f"- 类型：{a['kind']}",
        f"- 标的：{a['code']}",
        f"- 区间：{result['date'].iloc[0]} ~ {result['date'].iloc[-1]}",
        f"- 交易日数：{len(result)}",
        f"- 调仓次数：{a['n_trades']}（累计费用 {a['total_cost']:.2%}）",
        "",
        "## 绩效指标",
        "",
        "| 指标 | 策略 | 买入持有基准 |",
        "| --- | --- | --- |",
    ]
    for key in strat_stats:
        if key == "最长回撤天数":
            lines.append(f"| {key} | {strat_stats[key]} | {bench_stats[key]} |")
        elif key in ("年化夏普", "Deflated Sharpe", "期末净值"):
            lines.append(f"| {key} | {strat_stats[key]:.3f} | {bench_stats[key]:.3f} |")
        else:
            lines.append(f"| {key} | {strat_stats[key]:.2%} | {bench_stats[key]:.2%} |")

    lines += [
        "",
        f"> Deflated Sharpe 取 n_trials={N_TRIALS}（未做参数搜索）；",
        f"> 夏普口径：年化无风险利率 3%，一年 {TRADING_DAYS} 个交易日。",
        "",
        "## 净值走势与仓位",
        "",
        CHART_PLACEHOLDER,
        "",
        "## 前视偏差检验（截断法）",
        "",
        f"- 方法：全量数据与截去最近 {TRUNCATION_DAYS} 个交易日的数据分别跑一次，逐日比较仓位",
        f"- 结果：**{'存在前视偏差' if a['bias']['has_bias'] else '未发现前视偏差'}**",
        f"- 不一致的交易日数：{a['bias']['n_mismatches']}",
        "",
    ]
    return "\n".join(lines)


def report_filename(a: dict, suffix: str = ".pdf") -> str:
    """生成报告下载文件名：代码_策略_首日_末日.<suffix>。"""
    result = a["result"]
    return (
        f"{a['code']}_{a['strategy_name']}_"
        f"{result['date'].iloc[0]}_{result['date'].iloc[-1]}{suffix}"
    )


def generate_report(
    code: str,
    start_date: str,
    end_date: str,
    strategy_name: str,
    commission: float = DEFAULT_COMMISSION,
    market: str = DEFAULT_MARKET,
) -> tuple[str, dict]:
    """通用入口：接收标的、区间、策略名，返回 (Markdown 文本, 分析结果)，不落盘。

    每份报告必带图表，PNG 字节放在 ``analysis["chart"]`` 中。
    """
    analysis = run_analysis(
        code, start_date, end_date, strategy_name, commission, market)
    analysis["chart"] = build_chart(analysis)
    return build_report_markdown(analysis), analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="生成指定标的、区间、策略的回测报告")
    parser.add_argument("code", help="个股或基准指数代码，精确匹配，如 sh.600000、sh.000300（美股如 us.14593、us.SPX）")
    parser.add_argument("start_date", help="起始日期 YYYY-MM-DD")
    parser.add_argument("end_date", help="结束日期 YYYY-MM-DD")
    parser.add_argument("strategy", help="策略名（strategy/ 下的子文件夹名）")
    parser.add_argument("-m", "--market", default=DEFAULT_MARKET,
                        help=f"市场（默认 {DEFAULT_MARKET}；美股填 us）")
    args = parser.parse_args()

    try:
        markdown, _ = generate_report(
            args.code, args.start_date, args.end_date, args.strategy,
            market=args.market,
        )
    except BacktestInputError as exc:
        sys.exit(f"错误：{exc}")
    # 终端无法内嵌图片：去掉图表节标题与占位符行
    markdown = "\n".join(
        line
        for line in markdown.splitlines()
        if line not in (CHART_PLACEHOLDER, "## 净值走势与仓位")
    )
    print(markdown)


if __name__ == "__main__":
    main()
