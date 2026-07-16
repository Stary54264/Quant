"""factor.hurst 的最小自检 —— 直接 `python tests/test_hurst.py` 或 `pytest tests/`。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor.hurst import (  # noqa: E402
    hurst_dfa,
    hurst_ensemble,
    hurst_rs,
    hurst_wavelet,
    rolling_hurst,
)

import pandas as pd  # noqa: E402


RNG = np.random.default_rng(42)
N = 4096


def _within(x: float, lo: float, hi: float, name: str) -> None:
    assert lo <= x <= hi, f"{name} = {x:.3f} 不在 [{lo}, {hi}]"
    print(f"  ok  {name} = {x:.3f}  ∈ [{lo}, {hi}]")


def test_white_noise_h_near_half() -> None:
    """白噪声 H ≈ 0.5，允许 ±0.1 波动。"""
    x = RNG.standard_normal(N)
    print("[white noise]")
    _within(hurst_rs(x), 0.35, 0.65, "RS")
    _within(hurst_dfa(x), 0.35, 0.65, "DFA")
    _within(hurst_wavelet(x), 0.30, 0.70, "Wavelet")
    _within(hurst_ensemble(x), 0.35, 0.65, "Ensemble")


def test_trend_h_large() -> None:
    """强线性趋势 H 应显著 > 0.5。"""
    x = np.linspace(0, 5, N) + 0.1 * RNG.standard_normal(N)
    print("[trend]")
    assert hurst_dfa(x) > 0.7, "DFA 应识别出明显趋势"
    assert hurst_rs(x) > 0.7, "RS 应识别出明显趋势"
    print(f"  ok  DFA = {hurst_dfa(x):.3f} > 0.7")
    print(f"  ok  RS  = {hurst_rs(x):.3f} > 0.7")


def test_mean_reverting_h_small() -> None:
    """AR(1) 强负相关 → 反持续 H < 0.5。"""
    phi = -0.8
    e = RNG.standard_normal(N)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = phi * x[i - 1] + e[i]
    print("[mean-reverting AR(1) phi=-0.8]")
    h = hurst_dfa(x)
    assert h < 0.45, f"DFA H = {h:.3f} 应 < 0.45"
    print(f"  ok  DFA = {h:.3f} < 0.45")


def test_rolling_shape_and_nan() -> None:
    """rolling_hurst 输出 index 对齐 & warm-up 段全 NaN。"""
    s = pd.Series(RNG.standard_normal(1024), name="ret")
    out = rolling_hurst(s, window=128, method="dfa")
    print("[rolling_hurst]")
    assert len(out) == len(s), "长度必须一致"
    assert out.iloc[:127].isna().all(), "前 window-1 应为 NaN"
    assert out.iloc[127:].notna().mean() > 0.95, "warm-up 之后 NaN 应极少"
    print(f"  ok  len = {len(out)}  NaN warm-up = 127  tail non-NaN ratio = "
          f"{out.iloc[127:].notna().mean():.3f}")


if __name__ == "__main__":
    for fn in [
        test_white_noise_h_near_half,
        test_trend_h_large,
        test_mean_reverting_h_small,
        test_rolling_shape_and_nan,
    ]:
        fn()
    print("\nall hurst tests passed.")
