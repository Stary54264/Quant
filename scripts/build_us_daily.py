#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""美股日线数据集构建：把 CRSP 抽取 CSV 清理为与 A 股同构的 parquet。

输入：data/backtest/ 下的 CRSP 抽取 CSV（支持多个文件，追加 2025 等年份时
把新文件放同目录、文件名以 crsp_ 开头即可，脚本会全部读入后去重）。
输出（data/backtest/us/）：
    daily_stocks.parquet   10 列，与 a_share 的 parquet 逐列一致
    securities.csv         全部 PERMNO 的对照/生命周期表

口径：
    - 仅保留普通普通股（CRSP share code 首位为 1；ETF/ADR/基金/外国公司等排除，
      但仍记入 securities.csv，included=False）；
    - 非交易日占位行（OHLC 全 0、无成交量、成交量 -99、收益 -77/-99）整行剔除，
      与 A 股"停牌日无记录"口径一致；收益缺失码 -66（价格有效）用相邻有效
      价格重算（个别日子漏分红，微小近似），重算超 ±400% 的不可信行整行剔除；
    - close 为总回报口径的前复权价：用逐日 RET 链重建、以最后交易日原始价格为
      锚（adj_close_t = prc_last × ∏(1+ret)/链末值）；不依赖 facpr/facshr
      （抽取中只在事件日有值）；open/high/low 按当日 adj_close/|prc| 缩放；
    - 退市末日：pctChg 并入 DLRET（仅取 (-1,1) 内的有效小数，如雷曼 -0.6；
      -55/-66 等整数为缺失码，不并入）；
    - prc 为负表示当日无收盘价（数值为买卖报价均值），取绝对值；
    - volume 为原始成交量；amount = |prc| × volume；turn = vol/(shrout×1000)×100。

运行需要 pyarrow（pandas 写 parquet），属数据构建专用依赖。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "backtest"
OUT_DIR = ROOT / "data" / "backtest" / "us"

# 抽取文件的 18 列，按原始顺序
RAW_COLS = [
    "date", "permno", "htick", "hcomnam", "shrcd", "exchcd", "tsymbol",
    "open", "prc", "high", "low", "vol", "ret", "shrout",
    "dlret", "dlstcd", "facpr", "facshr",
]
# 读入时保留的列（facpr/facshr 不可用，不读）
USE_COLS = [
    "date", "permno", "hcomnam", "shrcd", "exchcd", "tsymbol",
    "open", "prc", "high", "low", "vol", "ret", "shrout",
    "dlret", "dlstcd",
]
DTYPES = {
    "permno": "int32",
    "hcomnam": "string",
    "shrcd": "string",
    "exchcd": "string",
    "tsymbol": "string",
    "open": "float64",
    "prc": "float64",
    "high": "float64",
    "low": "float64",
    "vol": "float64",
    "ret": "float64",
    "shrout": "float64",
    "dlret": "float64",
    "dlstcd": "string",
}

# 非交易日/数据缺失的数值哨兵（CHASS 数字显示版）
BAD_RETS = (-77.0, -99.0)   # 与零价格行一同出现，整行剔除
DERIVE_RETS = (-66.0,)      # 价格有效但收益缺失，用相邻价格重算
DERIVE_BOUND = 4.0          # 重算收益超 ±400% 视为不可信（长期停牌后恢复等），整行剔除
# 退市收益为缺失码（整数显示，如 -55/-66/-1/1/3），仅 (-1,1) 内的小数值可用
DLRET_MIN, DLRET_MAX = -1.0, 1.0

CHUNKSIZE = 1_000_000


def load_raw() -> pd.DataFrame:
    """分块读入全部 CRSP 抽取 CSV，返回保留列的合并 DataFrame（未清洗）。"""
    files = sorted(RAW_DIR.glob("crsp_*.csv"))
    if not files:
        raise FileNotFoundError(f"{RAW_DIR} 下没有 crsp_*.csv 抽取文件")

    frames = []
    for path in files:
        n_file = 0
        for chunk in pd.read_csv(
            path,
            skiprows=2,          # 前两行是标题与列名，列名用 RAW_COLS 代替
            header=None,
            names=RAW_COLS,
            usecols=USE_COLS,
            dtype=DTYPES,
            chunksize=CHUNKSIZE,
            na_values=["", "."],
        ):
            n_file += len(chunk)
            frames.append(chunk)
        print(f"读入 {path.name}：{n_file:,} 行")

    return pd.concat(frames, ignore_index=True)


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """剔除非交易日占位行；-66 收益用相邻有效价格重算。按 permno/date 排序。"""
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    # 非交易日：OHLC 全 0、无成交量（停牌，仅有报价），或成交量为负哨兵
    nontrade = (
        ((df["open"] == 0) & (df["prc"] == 0) & (df["high"] == 0) & (df["low"] == 0))
        | (df["vol"] <= 0)
        | df["ret"].isin(BAD_RETS)
    )
    print(f"剔除非交易日占位行：{int(nontrade.sum()):,}")
    df = df[~nontrade].copy()

    # 价格有效但收益缺失：用上一有效行价格重算；无前值或重算结果超界则整行剔除
    bad = df["ret"].isin(DERIVE_RETS)
    abs_prc = df["prc"].abs()
    prev_prc = abs_prc.groupby(df["permno"]).shift(1)
    derived = abs_prc / prev_prc - 1.0
    use_derived = bad & prev_prc.notna() & (derived.abs() <= DERIVE_BOUND)
    drop_bad = bad & ~use_derived
    df.loc[use_derived, "ret"] = derived[use_derived]
    print(f"收益缺失码重算：{int(use_derived.sum()):,} 行；"
          f"不可信剔除：{int(drop_bad.sum()):,} 行")
    return df[~drop_bad].copy()


def build_securities(df: pd.DataFrame) -> pd.DataFrame:
    """全部 PERMNO 的对照/生命周期表（含被排除的非普通股）。"""
    grouped = df.groupby("permno", sort=True)
    sec = grouped.agg(
        ticker=("tsymbol", "last"),
        name=("hcomnam", "last"),
        exchcd=("exchcd", "last"),
        shrcd=("shrcd", "last"),
        first_date=("date", "first"),
        last_date=("date", "last"),
        dlstcd=("dlstcd", "last"),
    ).reset_index()
    sec["code"] = "us." + sec["permno"].astype(str)
    sec["active"] = sec["dlstcd"] == "100"
    sec["included"] = sec["shrcd"].str.startswith("1")
    return sec[
        ["code", "permno", "ticker", "name", "exchcd", "shrcd",
         "first_date", "last_date", "active", "included", "dlstcd"]
    ]


def build_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """过滤普通股并重建总回报复权价，返回 10 列成品。"""
    df = df[df["shrcd"].str.startswith("1")].copy()
    df = df.drop_duplicates(["permno", "date"], keep="last").reset_index(drop=True)
    print(f"普通股行：{len(df):,}")

    permno = df["permno"]

    # 退市末日（每组最后一行）与有效退市收益（-88 为"无退市收益"占位）
    grp_index = permno.groupby(permno).cumcount()
    grp_size = permno.groupby(permno).transform("size")
    is_last = grp_index + 1 == grp_size
    valid_dl = df["dlret"].notna() & df["dlret"].between(
        DLRET_MIN, DLRET_MAX, inclusive="neither")
    del grp_index, grp_size

    # 总回报链：adj_close_t = |prc_last| × chain_t/chain_last
    chain = (1.0 + df["ret"]).groupby(permno).cumprod()
    chain_last = chain.groupby(permno).transform("last")
    abs_prc = df["prc"].abs()
    anchor = abs_prc.groupby(permno).transform("last")
    adj_close = anchor * chain / chain_last
    day_scale = adj_close / abs_prc
    del chain, chain_last

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"])
    out["code"] = "us." + permno.astype(str)
    out["open"] = df["open"] * day_scale
    out["high"] = df["high"] * day_scale
    out["low"] = df["low"] * day_scale
    out["close"] = adj_close
    out["volume"] = df["vol"].round().astype("int64")
    out["amount"] = abs_prc * df["vol"]
    # 流通股数为千股；非正时无法算换手率。turn 为百分数口径（与 A 股表一致）
    shares = df["shrout"] * 1000.0
    out["turn"] = np.where(shares > 0, df["vol"] / shares * 100.0, np.nan)
    out["pctChg"] = df["ret"] * 100.0
    # 退市末日把退市收益并入当日 pctChg（与复权价的差异只体现在最后一天）
    merged = ((1.0 + df["ret"]) * (1.0 + df["dlret"]) - 1.0) * 100.0
    out.loc[is_last & valid_dl, "pctChg"] = merged[is_last & valid_dl]

    return out[["date", "code", "open", "high", "low", "close",
                "volume", "amount", "turn", "pctChg"]]


def main() -> None:
    raw = load_raw()
    print(f"原始行合计：{len(raw):,}")

    clean = clean_rows(raw)
    del raw

    securities = build_securities(clean)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    securities.to_csv(OUT_DIR / "securities.csv", index=False)
    print(f"securities.csv：{len(securities):,} 只标的，"
          f"其中普通股 {int(securities['included'].sum()):,} 只、"
          f"在市 {int(securities['active'].sum()):,} 只")

    stocks = build_stocks(clean)
    del clean
    stocks.to_parquet(OUT_DIR / "daily_stocks.parquet", compression="zstd")
    print(f"daily_stocks.parquet：{len(stocks):,} 行，"
          f"{stocks['code'].nunique():,} 只标的，"
          f"{stocks['date'].min().date()} ~ {stocks['date'].max().date()}")


if __name__ == "__main__":
    sys.exit(main())
