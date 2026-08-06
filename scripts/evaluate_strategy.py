#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略评价脚本
用法: python scripts/evaluate_strategy.py <returns_csv>
功能: 输入每日策略收益率序列，计算年化夏普比率

输入 CSV 要求:
  - 包含一列日期（列名含 date，或第一列），格式 YYYYMMDD 或 YYYY-MM-DD
  - 包含一列收益率（列名含 return/ret/strategy，或除日期外的唯一数值列），为小数（0.01 表示 1%）
"""

import sys
import numpy as np
import pandas as pd


def load_returns(csv_path: str) -> pd.Series:
    """从 CSV 加载每日收益率序列，返回以日期为索引的 Series。"""
    df = pd.read_csv(csv_path)

    # 识别日期列
    date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
    if df[date_col].isna().all():
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col)

    # 识别收益率列：优先匹配 return/ret/strategy，否则取唯一数值列
    ret_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ('return', 'ret', 'strategy'))),
        None,
    )
    if ret_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) != 1:
            raise ValueError(
                f"无法自动识别收益率列，请将列名命名为 return/ret/strategy。候选列: {list(df.columns)}"
            )
        ret_col = numeric_cols[0]

    returns = pd.to_numeric(df[ret_col], errors='coerce').dropna()
    returns = returns.sort_index()
    return returns


def annualized_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """计算年化夏普比率（无风险利率默认 0）。

    Sharpe = mean(daily) / std(daily) * sqrt(periods_per_year)
    """
    if len(daily_returns) < 2:
        return float('nan')
    std = daily_returns.std()
    if std == 0 or np.isnan(std):
        return float('nan')
    return daily_returns.mean() / std * np.sqrt(periods_per_year)


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/evaluate_strategy.py <returns_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    returns = load_returns(csv_path)

    sharpe = annualized_sharpe(returns)

    print(f"样本区间: {returns.index[0].date()} 至 {returns.index[-1].date()}")
    print(f"交易日数: {len(returns)}")
    print(f"日均收益: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
    print(f"日波动率: {returns.std():.6f} ({returns.std()*100:.4f}%)")
    print(f"年化夏普: {sharpe:.4f}")


if __name__ == "__main__":
    main()
