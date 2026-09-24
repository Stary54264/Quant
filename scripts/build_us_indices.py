#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""美股基准指数数据集构建：把指数原始 CSV 清理为与个股同构的 parquet。

输入：data/backtest/ 下的 us_index_*.csv（date,open,high,low,close）
输出：data/backtest/us/daily_indices.parquet

口径：
    - 指数为点位序列，无公司行为，OHLC 即官方点位（价格指数口径）；
    - 10 列 schema 与个股 parquet 逐列一致：指数没有成交量/成交额/换手率，
      volume 记 0、amount/turn 记 NULL；
    - pctChg 由相邻交易日收盘点位比计算（百分数），首日留空；
    - 区间 2006-01-01 ~ 2025-12-31。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "backtest"
OUT = ROOT / "data" / "backtest" / "us" / "daily_indices.parquet"

# 原始文件名后缀 → 成品代码
INDICES = {
    "dji": "us.DJI",
    "spx": "us.SPX",
    "ixic": "us.IXIC",
    "rut": "us.RUT",
}
START_DATE, END_DATE = "2006-01-01", "2025-12-31"


def main() -> None:
    frames = []
    for suffix, code in INDICES.items():
        path = RAW_DIR / f"us_index_{suffix}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
        df["code"] = code
        df["pctChg"] = df["close"].pct_change() * 100.0
        frames.append(df)
        print(f"{code}: {len(df):,} 行，{df['date'].min().date()} ~ {df['date'].max().date()}")

    out = pd.concat(frames, ignore_index=True)
    out["volume"] = np.int64(0)
    out["amount"] = np.nan
    out["turn"] = np.nan
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    out = out[["date", "code", "open", "high", "low", "close",
               "volume", "amount", "turn", "pctChg"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, compression="zstd")
    print(f"daily_indices.parquet：{len(out):,} 行，{out['code'].nunique()} 只指数")


if __name__ == "__main__":
    sys.exit(main())
