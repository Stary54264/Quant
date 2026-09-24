#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Yahoo 2025 raw CSV → 2025 美股日线 parquet（与权威数据集同口径）。

输入：data/backtest/yahoo2025/<TICKER>.csv
      （date,open,high,low,close,volume；由 fetch_yahoo_2025.py 抓取，注意
      该脚本存的 quote 是原始价，复权信息需要重新从 Yahoo 取 adjclose——
      本脚本假设 raw CSV 同目录下另有 <TICKER>.adj.csv：date,adjclose）
输出：data/backtest/us/daily_stocks_2025_yahoo.parquet

口径：
    - 总回报前复权（与主数据集一致）：用 adjclose 的逐日比作为总收益，
      以最后交易日为锚重建 OHLC——open/high/low 按当日 (adjclose/close)
      同比例缩放；pctChg 为 adjclose 逐日比 ×100，首日留空；
    - 2025 年内的拆股/分红由此自动调齐；
    - 代码暂用 us.<TICKER>（2025 权威 CRSP 抽取到位后改用 PERMNO 重建）；
      PERMNO 代码为纯数字，不会与 ticker 代码冲突；
    - volume 为原始量；amount = close × volume；turn 留空（无流通股数据）；
    - 仅取 2025-01-01 ~ 2025-12-31。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "backtest" / "yahoo2025"
OUT = ROOT / "data" / "backtest" / "us" / "daily_stocks_2025_yahoo.parquet"
START, END = "2025-01-01", "2025-12-31"


def build_one(ticker: str, path: Path, adj_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    adj = pd.read_csv(adj_path, parse_dates=["date"])
    df = df.merge(adj, on="date", how="left").sort_values("date").reset_index(drop=True)

    # 总回报前复权：以最后有效日为锚
    a = df["adjclose"].fillna(df["close"])
    scale0 = df["close"].iloc[-1] / a.iloc[-1]
    factor = a * scale0 / df["close"]          # 当日 OHLC 缩放因子

    out = pd.DataFrame()
    out["date"] = df["date"]
    out["code"] = f"us.{ticker}"
    out["open"] = df["open"] * factor
    out["high"] = df["high"] * factor
    out["low"] = df["low"] * factor
    out["close"] = df["close"] * factor
    out["volume"] = df["volume"].fillna(0).astype("int64")
    out["amount"] = df["close"] * df["volume"].fillna(0)
    out["turn"] = np.nan
    out["pctChg"] = a.pct_change() * 100.0
    return out


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(RAW_DIR)
    frames, n_missing_adj = [], 0
    for path in sorted(RAW_DIR.glob("*.csv")):
        if path.name.startswith("_") or path.stem.endswith(".adj"):
            continue                                 # 跳过 adj/状态文件
        ticker = path.stem
        adj_path = RAW_DIR / f"{ticker}.adj.csv"
        if not adj_path.exists():
            n_missing_adj += 1
            continue
        frames.append(build_one(ticker, path, adj_path))
    if not frames:
        raise RuntimeError(f"{RAW_DIR} 下没有可构建的 ticker 数据")
    if n_missing_adj:
        print(f"跳过缺 adjclose 的 ticker：{n_missing_adj}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    out = out[["date", "code", "open", "high", "low", "close",
               "volume", "amount", "turn", "pctChg"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, compression="zstd")
    print(f"daily_stocks_2025_yahoo.parquet：{len(out):,} 行，"
          f"{out['code'].nunique():,} 只，{out['date'].min().date()} ~ {out['date'].max().date()}")


if __name__ == "__main__":
    sys.exit(main())
