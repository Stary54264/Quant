#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""日频回测引擎：T 日收盘出信号，T+1 日开盘按市价成交，单边收取手续费。

价格口径
--------
输入行情使用**前复权** OHLC：前复权对除权除息日之前的价格统一乘调整因子，
其收益率已隐含现金分红、送股、转增的再投资（总回报口径），因此引擎不再
单独处理除权事件，也不能再叠加分红收益率（否则双重计算）。

时序约定（无前视）
------------------
- ``position.iloc[i]`` 表示：第 i 个交易日开盘调仓完成后持有的目标仓位。
  它由第 i-1 日收盘及之前的信息决定（strategy 层输出已滞后一日），
  对应"T-1 日收盘后挂市价单，T 日开盘成交"。
- 持仓收益按**开盘到开盘**计：``open[i+1] / open[i] - 1``。T 日开盘买入后
  当日日内波动与 T 日收盘到 T+1 开盘的隔夜跳空都由持仓人承担，与真实持有
  隔夜的情形一致。
- 换仓率 ``turnover[i] = |position[i] - position[i-1]|``，首日为
  ``|position[0]|``（建仓同样收费）；当日手续费 = turnover × commission，
  在成交当日的收益中扣除。
- 最后一个交易日没有次日开盘可计价，其收益记 0（净值止于最后一次可得开盘价）。
"""

import numpy as np
import pandas as pd

# 单边手续费率：10 个基点（成交金额的 0.10%），同时覆盖佣金与买卖价差的近似
DEFAULT_COMMISSION = 0.001


def backtest(
    data: pd.DataFrame,
    position: pd.Series,
    commission: float = DEFAULT_COMMISSION,
) -> pd.DataFrame:
    """运行逐日回测，返回成交、收益、成本与净值明细。

    Parameters
    ----------
    data : pd.DataFrame
        按日期升序排列的日线行情，必须包含 ``date``、``open`` 列，
        价格为前复权口径。停牌日应已缺席（无行情行）。
    position : pd.Series
        每日开盘调仓后的目标仓位，与 ``data`` 等长、按行对齐；
        0/1 分别表示空仓/满仓，也支持任意分数仓位（如 0.5 半仓）。
    commission : float, default 0.001
        单边手续费率（10bp），按成交金额（换仓率）收取，买卖均收。

    Returns
    -------
    pd.DataFrame
        逐日明细，列包括：

        - ``date``：交易日
        - ``position``：当日开盘后的持仓
        - ``turnover``：当日开盘的换仓率（0~1）
        - ``asset_ret``：标的开盘到开盘收益率
        - ``gross_ret``：扣费前策略收益（position × asset_ret）
        - ``cost``：当日手续费
        - ``net_ret``：扣费后策略收益（gross_ret − cost）
        - ``nav``：净值曲线，期初为 1
    """
    if len(data) != len(position):
        raise ValueError(f"data 行数({len(data)}) 与 position 长度({len(position)}) 不一致")
    if not (0 <= commission < 0.01):
        raise ValueError(f"commission 应为 [0, 0.01) 内的小数，收到：{commission}")

    out = pd.DataFrame({"date": data["date"].to_numpy()})
    out["position"] = np.asarray(position, dtype=float)

    open_ = pd.to_numeric(data["open"], errors="coerce")
    if open_.isna().any():
        bad = out["date"][open_.isna()].tolist()
        raise ValueError(f"存在开盘价缺失的交易日，无法按开盘价成交：{bad[:5]}")

    # 开盘到开盘的标的收益；最后一日无次日开盘，记 0
    asset_ret = (open_.shift(-1) / open_ - 1).fillna(0.0).to_numpy()
    out["asset_ret"] = asset_ret
    out["gross_ret"] = out["position"] * asset_ret

    # 换仓率：首日视为从空仓建仓
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["turnover"] * commission
    out["net_ret"] = out["gross_ret"] - out["cost"]
    out["nav"] = (1 + out["net_ret"]).cumprod()
    return out
