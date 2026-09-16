#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dummy 策略：收盘价上穿/下穿 N 日 EMA。

规则（信号在 t 日收盘确认，t+1 日开盘才调仓）：
  - 收盘价上穿 EMA：次日买入（满仓持有）
  - 收盘价下穿 EMA：次日卖出（空仓）

策略函数签名与 scripts/test_strategy.py 的前视偏差检查兼容：
输入按日期升序的行情 DataFrame，返回与输入行数逐日对齐的每日仓位。
"""

import pandas as pd


def ema_crossover(data: pd.DataFrame, span: int = 20) -> pd.Series:
    """收盘价上穿 EMA 则次日持仓，下穿则次日空仓。

    Parameters
    ----------
    data : pd.DataFrame
        按日期升序排列的日线行情，必须包含 ``close`` 列。
    span : int, default 20
        EMA 窗口长度，默认 20 日。

    Returns
    -------
    pd.Series
        每日目标仓位，与 ``data`` 等长、按索引对齐：1 表示当日持有，
        0 表示空仓。t 日收盘产生的交叉信号最早反映在 t+1 日的仓位上，
        因此仓位序列相对信号整体滞后一日，无前视偏差。
    """
    close = data["close"]
    ema = close.ewm(span=span, adjust=False).mean()

    # t 日收盘时收盘价是否位于 EMA 上方
    above = close > ema
    # t+1 日仓位 = t 日收盘的多空状态：信号滞后一日生效
    position = above.shift(1, fill_value=False).astype(int)
    position.name = "position"
    return position
