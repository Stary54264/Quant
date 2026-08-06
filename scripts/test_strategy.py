#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""策略回测检查：按 Ernest Chan 的截断法检测前视偏差（look-ahead bias）。"""

import numpy as np
import pandas as pd


def test_lookahead_bias(strategy, data: pd.DataFrame, n_days: int = 10) -> dict:
    """用截断测试检测回测中的前视偏差。

    方法（Chan, *Quantitative Trading*）：
      1. 用全部历史数据运行策略，得到每日仓位 A；
      2. 截掉最近 n_days 天再运行一次，得到每日仓位 B；
      3. 把 A 也截掉同样的最近 n_days 天，使两者末日与长度一致；
      4. 逐日比较 A、B：若完全相同则无前视偏差，任何差异都说明程序在
         生成早期信号时用到了被截断部分的未来数据。

    Parameters
    ----------
    strategy : callable
        策略函数，输入行情 DataFrame（按日期升序），返回与输入行数对齐的
        每日仓位（pd.Series / pd.DataFrame / np.ndarray 均可）。
    data : pd.DataFrame
        完整历史行情，按日期升序排列。
    n_days : int, default 10
        截断的最近交易日数 N（Chan 建议 10~100）。

    Returns
    -------
    dict
        - ``has_bias`` (bool)：是否存在前视偏差。
        - ``n_mismatches`` (int)：不一致的交易日数。
        - ``mismatch_dates`` (DatetimeIndex)：不一致的日期。
        - ``positions_full``：全量回测得到的仓位（已截去末尾 n_days 天）。
        - ``positions_truncated``：截断数据回测得到的仓位。
    """
    if n_days < 1:
        raise ValueError("n_days 必须为正整数")
    if len(data) <= n_days:
        raise ValueError(f"数据长度({len(data)})不足以截掉 {n_days} 天")

    # 1. 全量数据跑一次 -> A
    pos_a = strategy(data)
    # 2. 截掉最近 n_days 天再跑一次 -> B
    pos_b = strategy(data.iloc[:-n_days])

    # 统一为 pandas 对象以便按索引对齐、比较
    if not isinstance(pos_a, (pd.Series, pd.DataFrame)):
        pos_a = pd.Series(np.asarray(pos_a), index=data.index)
    if not isinstance(pos_b, (pd.Series, pd.DataFrame)):
        pos_b = pd.Series(np.asarray(pos_b), index=data.iloc[:-n_days].index)

    # 3. A 也截掉末尾 n_days 天，使两者末日一致
    pos_a = pos_a.iloc[:-n_days]

    # 4. 逐日比较（按索引对齐，避免长度/日期错位）
    if isinstance(pos_a, pd.DataFrame) or isinstance(pos_b, pd.DataFrame):
        diff_mask = ~(pos_a == pos_b).all(axis=1)
    else:
        diff_mask = (pos_a != pos_b)
        # NaN != NaN 在 pandas 中为 True，但两边同为 NaN 应视为一致
        diff_mask = diff_mask & ~(pos_a.isna() & pos_b.isna())

    mismatch_dates = pos_a.index[diff_mask]
    return {
        'has_bias': bool(mismatch_dates.size > 0),
        'n_mismatches': int(mismatch_dates.size),
        'mismatch_dates': mismatch_dates,
        'positions_full': pos_a,
        'positions_truncated': pos_b,
    }
