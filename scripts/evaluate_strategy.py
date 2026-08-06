#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""策略评价脚本：输入日收益率的 numpy 1D array，计算年化夏普。"""

import numpy as np

# 年化无风险利率常数
RISK_FREE_RATE = 0.03


def annualized_sharpe(returns: np.ndarray, subtract_rf: bool = False) -> float:
    """计算年化夏普比率。

    Parameters
    ----------
    returns : np.ndarray
        每日策略收益率（小数，0.01 表示 1%），1D array。
    subtract_rf : bool
        是否减去无风险利率。若为 True，年化无风险利率取常数 0.03，
        折算为每日 0.03/252 后从日均收益中扣除。

    Returns
    -------
    float
        年化夏普 = (mean(daily) - rf_daily) / std(daily) * sqrt(252)
    """
    mean = returns.mean()
    std = returns.std()

    if subtract_rf:
        mean = mean - RISK_FREE_RATE / 252

    return mean / std * np.sqrt(252)
