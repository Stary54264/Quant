#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""策略评价脚本：输入日收益率的 numpy 1D array，计算年化夏普、最大回撤。"""

import numpy as np

# 年化无风险利率常数
RISK_FREE_RATE = 0.03


def annualized_sharpe(returns: np.ndarray, subtract_rf: bool = True) -> float:
    """计算年化夏普比率。

    Parameters
    ----------
    returns : np.ndarray
        每日策略收益率（小数，0.01 表示 1%），1D array。
    subtract_rf : bool, default True
        是否减去无风险利率。年化无风险利率为常数 0.03，
        折算为每日 0.03/252 后从日均收益中扣除。

    Returns
    -------
    float
        年化夏普 = (mean(daily) - rf_daily) / std(daily) * sqrt(252)
    """
    mean = returns.mean() - subtract_rf * RISK_FREE_RATE / 252
    std = returns.std()

    return mean / std * np.sqrt(252)


def max_drawdown(returns: np.ndarray) -> float:
    """计算最大回撤。

    先由日收益率累乘得到净值曲线，再对每个时点取此前的净值最大值
    （running max），当日回撤 = max(running_max - 当前净值, 0)，
    返回回撤序列的最大值（正数，表示最大跌幅幅度）。

    Parameters
    ----------
    returns : np.ndarray
        每日策略收益率（小数，0.01 表示 1%），1D array。

    Returns
    -------
    float
        最大回撤幅度（非负，0.20 表示从峰值最多回撤 20%）。
    """
    nav = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(nav)
    drawdowns = np.maximum(running_max - nav, 0)
    return drawdowns.max()
