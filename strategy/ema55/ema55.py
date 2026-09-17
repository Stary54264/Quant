#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dummy 策略：收盘价上穿/下穿 55 日 EMA。

规则（信号在 t 日收盘确认，t+1 日开盘才调仓）：
  - 收盘价上穿 EMA55：次日开盘买入（满仓持有）
  - 收盘价下穿 EMA55：次日开盘卖出（空仓）

策略函数签名与 backtest/test_strategy.py 的前视偏差检查兼容：
输入按日期升序的行情 DataFrame，返回与输入行数逐日对齐的每日仓位。
"""

import pandas as pd

from factor.ema import ema


def ema55(data: pd.DataFrame, span: int = 55) -> pd.Series:
    """收盘价上穿 EMA55 则次日持仓，下穿则次日空仓。

    Parameters
    ----------
    data : pd.DataFrame
        按日期升序排列的日线行情，必须包含 ``close`` 列。
    span : int, default 55
        EMA 窗口长度，默认 55 日。

    Returns
    -------
    pd.Series
        每日目标仓位，与 ``data`` 等长、按索引对齐：1 表示当日持有，
        0 表示空仓。t 日收盘产生的交叉信号最早反映在 t+1 日的仓位上，
        因此仓位序列相对信号整体滞后一日，无前视偏差。
    """
    close = data["close"]
    ema_line = ema(close, span)

    # t 日收盘时收盘价是否位于 EMA 上方
    above = close > ema_line
    # t+1 日仓位 = t 日收盘的多空状态：信号滞后一日生效
    position = above.shift(1, fill_value=False).astype(int)
    position.name = "position"
    return position
