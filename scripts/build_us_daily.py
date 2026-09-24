#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""美股日线数据集构建：把 CRSP 抽取 CSV 清理为与 A 股同构的 parquet。

输入：data/backtest/ 下的 CRSP 抽取 CSV（支持多个文件，追加 2025 等年份时
把新文件放同目录、文件名以 crsp_ 开头即可，脚本会全部读入后去重）。
输出（data/backtest/us/）：
    daily_stocks.parquet   10 列，与 a_share 的 parquet 逐列一致
    securities.csv         全部 PERMNO 的对照/生命周期表

口径：
    - 仅保留普通股（CRSP share code 首位为 1；ETF/ADR/基金/外国公司等排除，
      但仍记入 securities.csv，included=False）；
    - 无信息行（OHLC 全 0、prc 为 0/缺失、收益 -77/-99）整行剔除；零成交但
      有有效买卖报价（prc 为负，数值＝报价均值）的日子保留——它们是源数据
      计算次日收益的前收基准，复权事件也常落在这类日子（不保留会漏调乘数、
      产生假的价格缺口）；
    - 价格用源数据自带的事件乘数调整（乘子法，不用收益反推价格）：事件日
      facpr 非空且非 0 时，当日复权比例为 1/(1+facpr)（拆股、送股、合股等；
      facpr=0 为现金分红除息日标记，跳过、价格不变），逐日累乘后以最后交易
      日为锚归一化（前复权）。因此收盘价是"价格口径"，不含现金分红；
    - pctChg 为总回报口径（含现金分红再投资），故除息日 pctChg 与
      close/前收−1 天然不同，这是两套口径而非错误；源收益为缺失码
      （-77/-99/-66）时用相邻调整后收盘价补算，超 ±400% 不可信则留空；
    - open 缺失（源记 0，多为盘中恢复交易）用前一交易日调整后收盘价填充，
      首日无前收则用当日收盘价；
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
# 读入时保留的列（facshr 不用于价格调整，不读）
USE_COLS = [
    "date", "permno", "hcomnam", "shrcd", "exchcd", "tsymbol",
    "open", "prc", "high", "low", "vol", "ret", "shrout",
    "dlret", "dlstcd", "facpr",
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
    "facpr": "float64",
}

# 非交易日的数值哨兵（CHASS 数字显示版）
BAD_RETS = (-77.0, -99.0)   # 与零价格行一同出现，整行剔除
# 收益缺失/不可信码：pctChg 缺这些值时改用相邻调整后收盘价补算
MISSING_RETS = (-77.0, -99.0, -66.0)
DERIVE_BOUND = 4.0          # 补算收益超 ±400% 视为不可信（长期停牌后恢复等），留空
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
    """剔除无信息行：OHLC 全 0、无有效价格（prc 为 0/缺失）或收益坏码。

    零成交但有有效报价（|prc|>0）的行保留；收益缺失码不在此处理——价格
    不再由收益反推，缺失收益只影响 pctChg。
    """
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    no_info = (
        ((df["open"] == 0) & (df["prc"] == 0) & (df["high"] == 0) & (df["low"] == 0))
        | (df["prc"].abs() == 0)
        | df["prc"].isna()
        | df["ret"].isin(BAD_RETS)
    )
    print(f"剔除无信息行：{int(no_info.sum()):,}")
    n_quote = int(((df["vol"] <= 0) & ~no_info).sum())
    print(f"其中保留零成交报价行：{n_quote:,}")
    return df[~no_info].copy()


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
    """过滤普通股，用事件乘数调整价格，返回 10 列成品。"""
    df = df[df["shrcd"].str.startswith("1")].copy()
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    print(f"普通股行（去重前）：{len(df):,}")

    # 事件乘数组件：facpr 非空且非 0（现金分红日 facpr=0，跳过）；
    # 同日多个事件行（抽取把同一日的多个公司行为拆成多行）按比例相乘
    has_event = df["facpr"].notna() & (df["facpr"] != 0.0)
    df["event_ratio"] = np.where(
        has_event, 1.0 / (1.0 + df["facpr"].to_numpy()), 1.0)
    agg = {c: "last" for c in df.columns if c not in ("permno", "date", "event_ratio")}
    agg["event_ratio"] = "prod"
    df = df.groupby(["permno", "date"], as_index=False, sort=True).agg(agg)
    print(f"普通股行：{len(df):,}；事件乘数调整日：{int((df['event_ratio'] != 1.0).sum()):,}")

    permno = df["permno"]

    # 退市末日（每组最后一行）与有效退市收益（-88 为"无退市收益"占位）
    grp_index = permno.groupby(permno).cumcount()
    grp_size = permno.groupby(permno).transform("size")
    is_last = grp_index + 1 == grp_size
    valid_dl = df["dlret"].notna() & df["dlret"].between(
        DLRET_MIN, DLRET_MAX, inclusive="neither")
    del grp_index, grp_size

    cumfac = df["event_ratio"].groupby(permno).cumprod()
    # 前复权：以最后交易日为锚（末日乘数 1，历史价格按事件缩小）
    scale = cumfac.groupby(permno).transform("last") / cumfac

    abs_prc = df["prc"].abs()
    adj_close = abs_prc * scale
    prev_close = adj_close.groupby(permno).shift(1)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"])
    out["code"] = "us." + permno.astype(str)

    # open 源记 0（无开盘价）→ 前一交易日调整后收盘价；首日 → 当日收盘价
    missing_open = df["open"] == 0.0
    open_fill = prev_close.where(prev_close.notna(), adj_close)
    out["open"] = np.where(missing_open, open_fill, df["open"] * scale)
    n_open_fill = int(missing_open.sum())
    n_firstday_fill = int((missing_open & prev_close.isna()).sum())
    print(f"open 缺失填充：{n_open_fill:,} 行（其中首日用当日收盘价：{n_firstday_fill} 行）")

    # 报价日 high/low 偶发缺失（0）→ 以当日报价（close 基准）代替
    raw_high = df["high"].where(df["high"] != 0.0, abs_prc)
    raw_low = df["low"].where(df["low"] != 0.0, abs_prc)
    out["high"] = raw_high * scale
    out["low"] = raw_low * scale
    out["close"] = adj_close
    out["volume"] = df["vol"].round().astype("int64")
    out["amount"] = abs_prc * df["vol"]
    # 流通股数为千股；非正时无法算换手率。turn 为百分数口径（与 A 股表一致）
    shares = df["shrout"] * 1000.0
    out["turn"] = np.where(shares > 0, df["vol"] / shares * 100.0, np.nan)

    # pctChg 为总回报口径；源缺失码用相邻调整后收盘价补算，不可信则留空
    pct = df["ret"].copy()
    missing = pct.isna() | pct.isin(MISSING_RETS)
    derived = adj_close / prev_close - 1.0
    use_derived = missing & prev_close.notna() & (derived.abs() <= DERIVE_BOUND)
    pct.loc[use_derived] = derived.loc[use_derived]
    pct.loc[missing & ~use_derived] = np.nan
    n_blank = int((missing & ~use_derived).sum())
    print(f"pctChg 缺失补算：{int(use_derived.sum()):,} 行；不可信留空：{n_blank:,} 行")

    out["pctChg"] = pct * 100.0
    # 退市末日把退市收益并入当日 pctChg（ret 当日已无效时无法并入，跳过）
    can_merge = is_last & valid_dl & df["ret"].notna() & ~df["ret"].isin(MISSING_RETS)
    merged = ((1.0 + df["ret"]) * (1.0 + df["dlret"]) - 1.0) * 100.0
    out.loc[can_merge, "pctChg"] = merged.loc[can_merge]
    print(f"退市末日并入退市收益：{int(can_merge.sum()):,} 行")

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
