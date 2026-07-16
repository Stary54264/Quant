"""
factor package — 非线性动力学因子库

约定：
- 每个因子模块导出至少一个 `compute(series, **kwargs) -> pd.Series` 函数
- 输入以 pandas.Series（收盘价或对数收益）为主，index 建议是 DatetimeIndex
- 输出对齐输入 index，前 warm-up 段为 NaN
"""

from .hurst import (
    hurst_rs,
    hurst_dfa,
    hurst_wavelet,
    hurst_ensemble,
    rolling_hurst,
)

__all__ = [
    "hurst_rs",
    "hurst_dfa",
    "hurst_wavelet",
    "hurst_ensemble",
    "rolling_hurst",
]
