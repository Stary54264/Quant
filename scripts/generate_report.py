#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 EMA55 交叉策略的回测报告（Markdown）。

流程：取数 → 策略信号 → 回测（T+1 开盘成交、扣手续费）→
调用 evaluate_strategy 与 test_strategy 中的全部函数计算指标与检验，
把结果写入 strategy/ema55/report.md。

运行（仓库根目录）：
    python3 scripts/generate_report.py
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from query_a_share import query
from strategy.ema55.ema_crossover import ema_crossover
from backtest.backtest import backtest, DEFAULT_COMMISSION
from evaluate_strategy import (
    annualized_sharpe,
    deflated_sharpe,
    max_drawdown,
    max_drawdown_duration,
)
from test_strategy import test_lookahead_bias

# ---- 报告配置（目前只跑这一个 dummy 策略，后续再扩展）----
CODE = "sh.600000"
START_DATE = "2006-01-01"
END_DATE = "2025-12-31"
SPAN = 55
N_TRIALS = 1          # DSR 的独立试验次数：单一参数配置，未做参数搜索
TRUNCATION_DAYS = 10  # 前视偏差截断测试截掉的最近交易日数
REPORT_PATH = ROOT / "strategy" / "ema55" / "report.md"

TRADING_DAYS = 252


def evaluate_returns(returns) -> dict:
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


def fmt_pct(x: float) -> str:
    return f"{x:.2%}"


def main() -> None:
    # 1. 取数与回测
    data = query(CODE, START_DATE, END_DATE).reset_index(drop=True)
    position = ema_crossover(data, SPAN)
    result = backtest(data, position, commission=DEFAULT_COMMISSION)

    # 基准：同期一直满仓（首日开盘建仓一次）
    benchmark = backtest(data, pd.Series(np.ones(len(data))),
                         commission=DEFAULT_COMMISSION)

    strat_stats = evaluate_returns(result["net_ret"])
    bench_stats = evaluate_returns(benchmark["net_ret"])

    # 2. test_strategy 的全部函数（目前为截断法前视偏差检验）
    bias = test_lookahead_bias(ema_crossover, data, n_days=TRUNCATION_DAYS)

    n_trades = int(result["turnover"].gt(0).sum())
    total_cost = float(result["cost"].sum())

    # 3. 拼装 Markdown
    lines = [
        "# EMA55 均线交叉策略回测报告",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 策略规则",
        "",
        "- 因子：55 日指数移动平均线（EMA55）",
        "- 买入：收盘价上穿 EMA55，次日开盘按市价买入（满仓）",
        "- 卖出：收盘价下穿 EMA55，次日开盘按市价卖出（空仓）",
        "- 成交时点：信号日 T 收盘确认，T+1 日开盘成交，持仓收益按开盘到开盘计",
        f"- 手续费：单边 {DEFAULT_COMMISSION:.2%}（{DEFAULT_COMMISSION * 1e4:.0f} bp），按换手收取",
        "- 价格口径：前复权，收益已含分红再投资",
        "",
        "## 标的与区间",
        "",
        f"- 标的：{CODE}",
        f"- 区间：{result['date'].iloc[0]} ~ {result['date'].iloc[-1]}",
        f"- 交易日数：{len(result)}",
        f"- 调仓次数：{n_trades}（累计费用 {total_cost:.2%}）",
        "",
        "## 绩效指标",
        "",
        "| 指标 | 策略 | 买入持有基准 |",
        "| --- | --- | --- |",
    ]
    for key in strat_stats:
        if key in ("最长回撤天数",):
            lines.append(f"| {key} | {strat_stats[key]} | {bench_stats[key]} |")
        elif key in ("年化夏普", "Deflated Sharpe", "期末净值"):
            lines.append(f"| {key} | {strat_stats[key]:.3f} | {bench_stats[key]:.3f} |")
        else:
            lines.append(f"| {key} | {fmt_pct(strat_stats[key])} | {fmt_pct(bench_stats[key])} |")

    lines += [
        "",
        f"> Deflated Sharpe 取 n_trials={N_TRIALS}（未做参数搜索）；",
        f"> 夏普口径：年化无风险利率 3%，一年 {TRADING_DAYS} 个交易日。",
        "",
        "## 前视偏差检验（截断法）",
        "",
        f"- 方法：全量数据与截去最近 {TRUNCATION_DAYS} 个交易日的数据分别跑一次，逐日比较仓位",
        f"- 结果：**{'存在前视偏差' if bias['has_bias'] else '未发现前视偏差'}**",
        f"- 不一致的交易日数：{bias['n_mismatches']}",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
