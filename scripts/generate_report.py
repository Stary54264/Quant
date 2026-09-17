#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用回测报告生成：给定标的、区间与策略，调用评价/检验函数生成 Markdown 报告。

既可作为模块被 UI 直接调用（返回报告字符串），也可在命令行运行：

    python3 scripts/generate_report.py sh.600000 2006-01-01 2025-12-31 ema55
    python3 scripts/generate_report.py sh.600000 2006-01-01 2025-12-31 ema55 -o report.md

不带 -o 时报告打印到标准输出；带 -o 时写入指定文件。

策略发现约定：``strategy/<策略名>/`` 子文件夹的 ``__init__.py`` 暴露
``generate_position(data)`` 统一入口，并可提供 ``STRATEGY_RULES`` 元信息。
"""

import argparse
import importlib
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from query_a_share import query
from backtest.backtest import backtest, DEFAULT_COMMISSION
from evaluate_strategy import (
    annualized_sharpe,
    deflated_sharpe,
    max_drawdown,
    max_drawdown_duration,
)
from test_strategy import test_lookahead_bias

TRADING_DAYS = 252
N_TRIALS = 1          # DSR 的独立试验次数：单一参数配置，未做参数搜索
TRUNCATION_DAYS = 10  # 前视偏差截断测试截去的最近交易日数

STRATEGY_DIR = ROOT / "strategy"


class BacktestInputError(ValueError):
    """用户输入有误（代码不存在、区间无行情、策略不符合约定等）。"""


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


def run_analysis(
    code: str,
    start_date: str,
    end_date: str,
    strategy_name: str,
    commission: float = DEFAULT_COMMISSION,
) -> dict:
    """跑通单标的单策略的完整分析，返回原始结果（逐日明细、指标、检验结果等）。"""
    module = _load_strategy(strategy_name)
    strategy_label = strategy_name

    data = query(code, start_date, end_date, kind="stock")
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
        "strategy_name": strategy_name,
        "strategy_label": strategy_label,
        "strategy_rules": getattr(module, "STRATEGY_RULES", []),
        "commission": commission,
        "data": data,
        "result": result,
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
        "- 价格口径：前复权，收益已含分红再投资",
        "",
        "## 标的与区间",
        "",
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
        "## 前视偏差检验（截断法）",
        "",
        f"- 方法：全量数据与截去最近 {TRUNCATION_DAYS} 个交易日的数据分别跑一次，逐日比较仓位",
        f"- 结果：**{'存在前视偏差' if a['bias']['has_bias'] else '未发现前视偏差'}**",
        f"- 不一致的交易日数：{a['bias']['n_mismatches']}",
        "",
    ]
    return "\n".join(lines)


def generate_report(
    code: str,
    start_date: str,
    end_date: str,
    strategy_name: str,
    commission: float = DEFAULT_COMMISSION,
) -> str:
    """通用入口：接收标的、区间、策略名，返回 Markdown 报告字符串。"""
    analysis = run_analysis(code, start_date, end_date, strategy_name, commission)
    return build_report_markdown(analysis)


def report_filename(a: dict) -> str:
    """生成下载用文件名：代码_策略_首日_末日.md。"""
    result = a["result"]
    return (
        f"{a['code']}_{a['strategy_name']}_"
        f"{result['date'].iloc[0]}_{result['date'].iloc[-1]}.md"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成指定标的、区间、策略的回测报告")
    parser.add_argument("code", help="个股代码，精确匹配，如 sh.600000")
    parser.add_argument("start_date", help="起始日期 YYYY-MM-DD")
    parser.add_argument("end_date", help="结束日期 YYYY-MM-DD")
    parser.add_argument("strategy", help="策略名（strategy/ 下的子文件夹名）")
    parser.add_argument("-o", "--output", help="报告写入路径；缺省打印到标准输出")
    args = parser.parse_args()

    markdown = generate_report(args.code, args.start_date, args.end_date, args.strategy)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"报告已写入 {out}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
