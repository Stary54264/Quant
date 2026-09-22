#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""market_timing 策略。

策略的唯一名就是子文件夹名（market_timing），UI 下拉框、报告标题均
直接使用文件夹名。每个策略子文件夹通过在 ``__init__.py`` 中暴露统一
入口 ``generate_position`` 被发现：输入按日期升序的行情 DataFrame，
返回逐日对齐的仓位。
"""

from .market_timing import market_timing

# 策略专属固定参数：短期 / 长期通道的 EMA 窗口
SHORT_SPANS = (21, 34)
LONG_SPANS = (55, 89)

STRATEGY_RULES = [
    "因子：四条指数移动平均线——EMA21/EMA34 组成短期通道，EMA55/EMA89 组成长期通道",
    "穿透确认：收盘价越过通道上轨（同时高于两条线）才算上穿，跌破下轨（同时低于两条线）才算下穿；夹在两线之间时维持原仓位，防止假信号频繁交易",
    "仓位：每个通道对应 50%——短期通道上穿加 50%、下穿减 50%，长期通道同理，总仓位 0% / 50% / 100%",
]


def generate_position(data):
    """返回该策略在给定行情上的每日目标仓位。"""
    return market_timing(data, SHORT_SPANS, LONG_SPANS)
