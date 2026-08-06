#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""策略评价脚本：输入日收益率的 numpy 1D array，计算年化夏普、最大回撤。"""

import numpy as np
from scipy.stats import norm, skew, kurtosis

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
    std = np.std(returns, ddof=1)

    return mean / std * np.sqrt(252)


def deflated_sharpe(
    returns: np.ndarray, n_trials: int = 1, subtract_rf: bool = True
) -> float:
    """计算 Deflated Sharpe Ratio (Bailey & López de Prado, 2014)。

    在 Probabilistic Sharpe Ratio (PSR) 基础上，把基准从 0 替换为多重试验
    下的期望最大 Sharpe（SR0），从而校正"从 N 个配置里挑出最优回测"所引入
    的选择偏差。返回值是扣除运气成分后，策略真实 Sharpe 仍高于 SR0 的概率。

    公式中 Sharpe 一律使用与观测同频的**非年化**日频 Sharpe；偏度/峰度修正
    直接采用日频矩（Lo, 2002）：

        PSR(SR0) = Phi((SR_hat - SR0) * sqrt(T-1) /
                       sqrt(1 - g3*SR_hat + (g4-1)/4*SR_hat^2))
        SR0 = sigma_SR * [(1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N e))]

    其中 gamma 为 Euler-Mascheroni 常数，sigma_SR 在原假设（无技能）下取
    IID 正态近似 1/sqrt(T)。

    Parameters
    ----------
    returns : np.ndarray
        每日策略收益率（小数，0.01 表示 1%），1D array。
    n_trials : int, default 1
        独立尝试过的策略/参数配置数量 N。N=1 时无选择偏差，DSR 退化为
        PSR(SR0=0)，即真实 Sharpe 大于 0 的概率。
    subtract_rf : bool, default True
        是否扣除年化无风险利率 0.03（与 annualized_sharpe 口径一致）。

    Returns
    -------
    float
        Deflated Sharpe Ratio，取值 [0, 1]，越接近 1 表示策略越可信。
    """
    r = np.asarray(returns, dtype=float)
    t = len(r)

    # 日频（非年化）Sharpe，与 annualized_sharpe 同口径（样本标准差 ddof=1，
    # 与下方无偏偏度/峰度 bias=False 的设定一致）
    excess_mean = r.mean() - subtract_rf * RISK_FREE_RATE / 252
    sr_hat = excess_mean / np.std(r, ddof=1)

    # 日频偏度与（非超额）峰度
    g3 = skew(r, bias=False)
    g4 = kurtosis(r, bias=False, fisher=False)

    # SR 估计量的非正态标准误修正项。
    # 理论上根号内为正，但有限样本下偏度/峰度的无偏估计可能出现极端值
    # （如 sr_hat < 0 且 g3 > 0 时 -g3*sr_hat 为负），将其截断到 1e-8 以上，
    # 避免 sqrt(负数) 产生 NaN / RuntimeWarning。
    se_squared = 1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat ** 2
    se_factor = np.sqrt(np.maximum(se_squared, 1e-8))

    if n_trials <= 1:
        sr0 = 0.0
    else:
        # 原假设下各试验 SR 的截面标准差，IID 正态近似为 1/sqrt(T)
        sigma_sr = 1.0 / np.sqrt(t)
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = sigma_sr * ((1 - np.euler_gamma) * z1 + np.euler_gamma * z2)

    z = (sr_hat - sr0) * np.sqrt(t - 1) / se_factor
    return float(norm.cdf(z))


def max_drawdown(returns: np.ndarray) -> float:
    """计算最大回撤。

    先由日收益率累乘得到净值曲线，再对每个时点取此前的净值最大值
    （running max），当日回撤 = max((running_max - 当前净值) / running_max, 0)，
    返回回撤序列的最大值（正数，表示最大跌幅比例）。

    Parameters
    ----------
    returns : np.ndarray
        每日策略收益率（小数，0.01 表示 1%），1D array。

    Returns
    -------
    float
        最大回撤比例（非负，0.20 表示从峰值最多回撤 20%）。
    """
    nav = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(nav)
    drawdowns = np.maximum((running_max - nav) / running_max, 0)
    return drawdowns.max()


def max_drawdown_duration(returns: np.ndarray) -> int:
    """计算最长回撤持续时间（交易日数）。

    对每个时点，统计其此前（含当日）最近一次净值创历史新高距今天的天数，
    取该序列的最大值。

    Parameters
    ----------
    returns : np.ndarray
        每日策略收益率（小数，0.01 表示 1%），1D array。

    Returns
    -------
    int
        最长回撤持续的交易日数。
    """
    nav = (1 + returns).cumprod()
    n = len(nav)
    running_max = np.maximum.accumulate(nav)
    # 标记每个创新高的位置（首日视为新高，回到前高即重置），向前填充最近一次新高的索引
    new_peak = np.concatenate([[True], nav[1:] >= running_max[:-1]])
    peak_idx = np.where(new_peak, np.arange(n), -1)
    last_peak = np.maximum.accumulate(peak_idx)
    durations = np.arange(n) - last_peak
    return int(durations.max())
