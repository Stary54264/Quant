#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""技术因子：指数移动平均线（Exponential Moving Average, EMA）。"""

import pandas as pd


def ema(close: pd.Series, span: int) -> pd.Series:
    """计算收盘价的 N 日指数移动平均线。

    递推形式（``adjust=False``，与常见行情软件口径一致）：

        EMA_t = alpha * close_t + (1 - alpha) * EMA_{t-1}
        alpha = 2 / (span + 1)

    序列首个有效值直接用首日收盘价初始化，因此早期数值受起点影响较大，
    span 个交易日后权重衰减到可忽略。输入含 NaN 时 EMA 会按有效观测
    递推（pandas ewm 默认行为）。

    Parameters
    ----------
    close : pd.Series
        收盘价序列（按日期升序）。
    span : int
        EMA 窗口长度。

    Returns
    -------
    pd.Series
        与 ``close`` 等长、按索引对齐的 EMA 序列。
    """
    return close.ewm(span=span, adjust=False).mean()
