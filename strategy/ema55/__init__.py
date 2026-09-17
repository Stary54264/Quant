#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ema55 策略。

策略的唯一名就是子文件夹名（ema55），UI 下拉框、报告标题均直接使用
文件夹名，不再单独维护显示名。每个策略子文件夹通过在 ``__init__.py``
中暴露统一入口 ``generate_position`` 被发现：输入按日期升序的行情
DataFrame，返回逐日对齐的仓位。
"""

from .ema55 import ema55

# 策略专属固定参数
SPAN = 55
STRATEGY_RULES = [
    "因子：55 日指数移动平均线（EMA55）",
    "买入：收盘价上穿 EMA55，次日开盘按市价买入（满仓）",
    "卖出：收盘价下穿 EMA55，次日开盘按市价卖出（空仓）",
]


def generate_position(data):
    """返回该策略在给定行情上的每日目标仓位。"""
    return ema55(data, SPAN)
