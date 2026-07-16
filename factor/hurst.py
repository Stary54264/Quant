"""
Hurst 指数（H 值）因子

三种估计器 + 一个滚动窗口封装：

- `hurst_rs`       R/S 分析（经典 Mandelbrot-Hurst）
- `hurst_dfa`      DFA 去趋势波动分析（对非平稳序列更稳健）
- `hurst_wavelet`  基于小波方差的多尺度估计（对短序列友好）
- `hurst_ensemble` 三估计器的中位数集成（抗单点异常）
- `rolling_hurst`  在时间序列上做滚动 H 值，返回 pandas.Series

判读：
    H ≈ 0.5   → 随机游走 / 无记忆
    H > 0.5   → 长记忆、趋势持续
    H < 0.5   → 反持续、均值回归

参考文献：
    Peters (1994) Fractal Market Analysis
    Peng et al. (1994) DFA
    Simonsen et al. (1998) Wavelet estimator of Hurst exponent
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

Method = Literal["rs", "dfa", "wavelet", "ensemble"]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _to_array(x: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """转成一维 float ndarray，剥掉 NaN。"""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    return x


def _default_scales(n: int, min_scale: int = 8, max_frac: float = 0.5) -> np.ndarray:
    """在 [min_scale, n*max_frac] 内取对数等距的尺度序列。"""
    upper = max(min_scale + 1, int(n * max_frac))
    if upper <= min_scale:
        return np.array([min_scale], dtype=int)
    scales = np.unique(
        np.logspace(np.log10(min_scale), np.log10(upper), num=12).astype(int)
    )
    return scales


def _loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """对 (log x, log y) 做最小二乘，返回斜率。"""
    if len(x) < 2:
        return np.nan
    lx, ly = np.log(x), np.log(y)
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)


# --------------------------------------------------------------------------- #
# 1. R/S 估计
# --------------------------------------------------------------------------- #
def hurst_rs(series: Sequence[float] | pd.Series, scales: Iterable[int] | None = None) -> float:
    """
    R/S 分析估计 Hurst 指数。

    步骤：
      1. 将序列切成长度 s 的不重叠子段
      2. 每段计算累计偏差范围 R 与标准差 S，取 (R/S) 均值
      3. 对 log s vs log <R/S> 做线性回归，斜率即 H
    """
    x = _to_array(series)
    n = len(x)
    if n < 32:
        return np.nan

    scales = np.asarray(list(scales) if scales is not None else _default_scales(n), dtype=int)
    rs_means: list[float] = []
    kept: list[int] = []

    for s in scales:
        if s < 8 or s > n // 2:
            continue
        m = n // s
        segs = x[: m * s].reshape(m, s)
        # 去均值 → 累积和 → 极差
        mean = segs.mean(axis=1, keepdims=True)
        z = np.cumsum(segs - mean, axis=1)
        R = z.max(axis=1) - z.min(axis=1)
        S = segs.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(S > 0, R / S, np.nan)
        rs = rs[np.isfinite(rs)]
        if len(rs) == 0:
            continue
        rs_means.append(float(rs.mean()))
        kept.append(int(s))

    if len(kept) < 3:
        return np.nan
    return _loglog_slope(np.asarray(kept, dtype=float), np.asarray(rs_means))


# --------------------------------------------------------------------------- #
# 2. DFA 估计
# --------------------------------------------------------------------------- #
def hurst_dfa(
    series: Sequence[float] | pd.Series,
    scales: Iterable[int] | None = None,
    order: int = 1,
) -> float:
    """
    Detrended Fluctuation Analysis 估计 Hurst 指数。

    步骤：
      1. profile = cumsum(x - mean(x))
      2. 分段长度 s 上做 order 阶多项式去趋势，得到波动 F(s)
      3. 对 log s vs log F(s) 线性回归，斜率即 H

    order=1 → 去除线性趋势（标准 DFA1，最常用）
    """
    x = _to_array(series)
    n = len(x)
    if n < 32:
        return np.nan

    profile = np.cumsum(x - x.mean())
    scales = np.asarray(list(scales) if scales is not None else _default_scales(n), dtype=int)
    F: list[float] = []
    kept: list[int] = []

    for s in scales:
        if s < order + 2 or s > n // 2:
            continue
        m = n // s
        segs = profile[: m * s].reshape(m, s)
        t = np.arange(s)
        # 每段拟合 order 阶多项式并求残差
        # 用向量化 polyfit：np.polyfit 支持 y 为 (s, m) 时对每列独立拟合
        coeffs = np.polyfit(t, segs.T, order)  # shape: (order+1, m)
        trend = np.polyval(coeffs, t[:, None]).T  # shape: (m, s)
        resid = segs - trend
        F.append(float(np.sqrt(np.mean(resid**2))))
        kept.append(int(s))

    if len(kept) < 3:
        return np.nan
    return _loglog_slope(np.asarray(kept, dtype=float), np.asarray(F))


# --------------------------------------------------------------------------- #
# 3. 小波估计
# --------------------------------------------------------------------------- #
def hurst_wavelet(series: Sequence[float] | pd.Series, max_level: int | None = None) -> float:
    """
    基于 Haar 小波方差的 Hurst 估计（Simonsen et al. 1998）。

    先将输入累积成 fBm 型序列（与 RS/DFA 内部累积保持一致），再做 Haar 分解。
    此时小波系数方差满足 Var(d_j) ∝ 2^(j·(2H+1))，
    log2 Var(d_j) vs j 的斜率 β 满足  H = (β - 1) / 2 。
    """
    x = _to_array(series)
    n = len(x)
    if n < 32:
        return np.nan

    if max_level is None:
        max_level = int(np.floor(np.log2(n))) - 2
    max_level = max(1, max_level)

    variances: list[float] = []
    levels: list[int] = []
    # 与 RS/DFA 一样，先累积；这样白噪声输入 → H≈0.5，趋势序列 → H→1
    y = np.cumsum(x - x.mean())

    for j in range(1, max_level + 1):
        # Haar 小波：detail = (y[0::2] - y[1::2]) / sqrt(2)
        k = (len(y) // 2) * 2
        if k < 4:
            break
        pair = y[:k].reshape(-1, 2)
        detail = (pair[:, 0] - pair[:, 1]) / np.sqrt(2.0)
        approx = (pair[:, 0] + pair[:, 1]) / np.sqrt(2.0)
        variances.append(float(np.var(detail, ddof=1)))
        levels.append(j)
        y = approx  # 下一层继续分解

    if len(levels) < 3 or any(v <= 0 for v in variances):
        return np.nan

    log2_var = np.log2(np.asarray(variances))
    slope, _ = np.polyfit(np.asarray(levels, dtype=float), log2_var, 1)
    H = (slope - 1.0) / 2.0
    return float(H)


# --------------------------------------------------------------------------- #
# 4. 集成
# --------------------------------------------------------------------------- #
def hurst_ensemble(series: Sequence[float] | pd.Series) -> float:
    """三种估计器的中位数集成，任一 NaN 会被忽略。"""
    vals = [hurst_rs(series), hurst_dfa(series), hurst_wavelet(series)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.median(vals))


# --------------------------------------------------------------------------- #
# 5. 滚动窗口
# --------------------------------------------------------------------------- #
def rolling_hurst(
    series: pd.Series,
    window: int = 128,
    step: int = 1,
    method: Method = "dfa",
    min_periods: int | None = None,
) -> pd.Series:
    """
    在时间序列上滚动计算 Hurst 指数。

    参数
    ----
    series : pd.Series
        原始价格或对数收益（**推荐用对数收益，避免非平稳污染**）。
    window : int
        滚动窗口长度，默认 128。金融序列常用 {64, 128, 256, 512}。
    step : int
        每 step 步计算一次；中间值前向填充。step=1 每根 K 线都算。
    method : {"rs", "dfa", "wavelet", "ensemble"}
        估计器选择，默认 DFA。
    min_periods : int
        起始 warm-up；默认 = window。

    返回
    ----
    pd.Series
        与输入 index 对齐；前 window-1 行为 NaN。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if min_periods is None:
        min_periods = window

    estimator = {
        "rs": hurst_rs,
        "dfa": hurst_dfa,
        "wavelet": hurst_wavelet,
        "ensemble": hurst_ensemble,
    }[method]

    values = np.full(len(series), np.nan, dtype=float)
    arr = series.to_numpy(dtype=float)

    for end in range(min_periods, len(arr) + 1, step):
        start = end - window
        if start < 0:
            continue
        window_slice = arr[start:end]
        values[end - 1] = estimator(window_slice)

    out = pd.Series(values, index=series.index, name=f"H_{method}_{window}")
    if step > 1:
        out = out.ffill()
    return out


# --------------------------------------------------------------------------- #
# 命令行 smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 4096

    # 1) 白噪声 → H ≈ 0.5
    wn = rng.standard_normal(n)
    # 2) 布朗运动（随机游走的累积） → H ≈ 0.5 若用增量估计
    bm = np.cumsum(wn)
    # 3) 强趋势序列 → H → 1
    trend = np.linspace(0, 5, n) + 0.1 * wn

    print(f"{'series':<12}{'RS':>8}{'DFA':>8}{'Wave':>8}{'Ens':>8}")
    for name, s in [("white", wn), ("brownian", bm), ("trend", trend)]:
        row = (hurst_rs(s), hurst_dfa(s), hurst_wavelet(s), hurst_ensemble(s))
        print(f"{name:<12}" + "".join(f"{v:>8.3f}" for v in row))
