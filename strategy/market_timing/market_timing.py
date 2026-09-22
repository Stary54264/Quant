#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""市场择时策略：四条 EMA 组成短期、长期两个通道。

短期通道为 EMA21/EMA34，长期通道为 EMA55/EMA89。只有收盘价整体
穿透一个通道（越过两条线）才动作，避免单条均线附近的假信号：

  - 收盘价高于通道两条线（越上轨）：通道状态转为被上穿
  - 收盘价低于通道两条线（破下轨）：通道状态转为被下穿
  - 收盘价夹在两条线之间：通道状态维持不变

每个通道对应 50% 仓位，两个通道的状态相加得到 0% / 50% / 100%
目标仓位；t 日收盘确认，t+1 日仓位才生效，无前视偏差。
"""

import numpy as np
import pandas as pd

from factor.ema import ema

# 短期 / 长期通道的 EMA 窗口
SHORT_SPANS = (21, 34)
LONG_SPANS = (55, 89)
# 单个通道贡献的仓位
_CHANNEL_WEIGHT = 0.5


def _channel_state(close: pd.Series, spans: tuple[int, ...]) -> pd.Series:
    """收盘价整体上穿/下穿一个 EMA 通道的状态（0/1），中间区域保持不变。"""
    lines = pd.concat([ema(close, span) for span in spans], axis=1)
    upper, lower = lines.max(axis=1), lines.min(axis=1)

    # 越上轨记 1、破下轨记 0、夹在两线之间留空向前沿用
    raw = pd.Series(
        np.where(close > upper, 1, np.where(close < lower, 0, np.nan)),
        index=close.index,
        dtype=float,
    )
    return raw.ffill().fillna(0).astype(int)


def market_timing(
    data: pd.DataFrame,
    short_spans: tuple[int, ...] = SHORT_SPANS,
    long_spans: tuple[int, ...] = LONG_SPANS,
) -> pd.Series:
    """按两个通道的穿透状态返回每日目标仓位（0/0.5/1），信号次日生效。"""
    close = data["close"]
    short_state = _channel_state(close, short_spans)
    long_state = _channel_state(close, long_spans)

    position = _CHANNEL_WEIGHT * (short_state + long_state)
    # t+1 日仓位 = t 日收盘的通道状态：信号滞后一日生效
    position = position.shift(1, fill_value=0.0)
    position.name = "position"
    return position
