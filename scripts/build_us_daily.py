#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""美股日线数据集构建：把 CRSP 抽取 CSV 清理为与 A 股同构的 parquet。

输入：data/backtest/ 下的 CRSP 抽取 CSV（支持多个文件，追加 2025 等年份时
把新文件放同目录、文件名以 crsp_ 开头即可，脚本会全部读入后按日聚合）。
输出（data/backtest/us/）：
    daily_stocks.parquet   10 列，与 a_share 的 parquet 逐列一致
    securities.csv         全部 PERMNO 的对照/生命周期表

口径：
    - 价格用源数据自带的事件乘数调整（乘子法，不用收益反推价格）：
      * 同一交易日多个分布行时，facpr 相加（复合比例 = 1/(1+Σfacpr)，
        不是各因子相乘）；
      * facpr=0 为现金分红除息日标记，价格不调，分红只体现在 pctChg；
      * 事件可能落在无行情占位行（prc=0、vol=-99）。乘数链在全部日历日
        上 cumprod，占位日的事件自动并入下一有效交易日（该日不输出数据）；
    - 因而 close 是"价格口径"（拆股/合股/送股已调，不含现金分红）；
      pctChg 是总回报口径（含现金分红），除息日与 close/前收−1 天然不同；
    - 仅保留普通股（CRSP share code 首位为 1；其他标的仍记入 securities.csv，
      included=False）；无任何有效行情的占位日不输出；
    - open 缺失（源记 0，多为盘中恢复交易）用前一有效交易日调整后收盘价
      填充，首日无前收则用当日收盘价；
    - 退市末日：pctChg 并入 DLRET（仅取 (-1,1) 内的有效小数，如雷曼 -0.6；
      -55/-66 等整数为缺失码，不并入）；
    - prc 为负表示数值是买卖报价均值（可能零成交），取绝对值；
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
# facshr 不用于价格调整，不读
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

# 收益缺失/不可信码：pctChg 缺这些值时改用相邻调整后收盘价补算
MISSING_RETS = (-77.0, -99.0, -66.0)
DERIVE_BOUND = 4.0          # 补算收益超 ±400% 视为不可信，留空
# 退市收益仅接受严格位于 (-1,1) 内的小数（整数是缺失码）
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


def blank_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """把哨兵值统一置为 NaN，方便按日聚合时跳过（负 prc 取绝对值）。"""
    df = df.copy()
    df["prc"] = df["prc"].abs().where(df["prc"].abs() > 0)
    for col in ("open", "high", "low", "vol"):
        df[col] = df[col].where(df[col] > 0)
    df["ret"] = df["ret"].where(~df["ret"].isin(MISSING_RETS))
    return df


def aggregate_days(df: pd.DataFrame) -> pd.DataFrame:
    """按 (permno, date) 聚合：同日多分布行 facpr 求和，其他字段取最后有效值。

    无行情占位行（prc NaN）也保留在结果中——其 sum_facpr 上的事件要进入
    乘数链并 carry 到下一有效日，有效与否用 prc 是否缺失判断。
    """
    df = df.sort_values(["permno", "date"])
    # last 自动跳过 NaN：同日"占位行 + 有效行"取到有效行情
    agg = {col: "last" for col in df.columns
           if col not in ("permno", "date", "facpr")}
    agg["facpr"] = "sum"       # 无事件（全 NaN）求和为 0；多分布相加
    out = df.groupby(["permno", "date"], sort=True).agg(agg).reset_index()
    return out.rename(columns={"facpr": "sum_facpr"})


def build_securities(days: pd.DataFrame) -> pd.DataFrame:
    """全部 PERMNO 的对照/生命周期表（含被排除的非普通股）。"""
    info = days.groupby("permno").agg(
        ticker=("tsymbol", "last"),
        name=("hcomnam", "last"),
        exchcd=("exchcd", "last"),
        shrcd=("shrcd", "last"),
        dlstcd=("dlstcd", "last"),
    )
    valid = days[days["prc"].notna()]
    spans = valid.groupby("permno").agg(
        first_date=("date", "first"),
        last_date=("date", "last"),
    )
    sec = info.join(spans).reset_index()
    sec["code"] = "us." + sec["permno"].astype(str)
    sec["active"] = sec["dlstcd"] == "100"
    sec["included"] = sec["shrcd"].str.startswith("1")
    return sec[
        ["code", "permno", "ticker", "name", "exchcd", "shrcd",
         "first_date", "last_date", "active", "included", "dlstcd"]
    ]


def build_stocks(days: pd.DataFrame) -> pd.DataFrame:
    """过滤普通股，按事件乘链重建价格，返回 10 列成品。"""
    days = days[days["shrcd"].str.startswith("1")].copy()
    permno_all = days["permno"]

    # 乘数链在全部日历日（含无行情占位日）上 cumprod，占位日事件得以 carry
    denom = 1.0 + days["sum_facpr"]
    n_bad_denom = int((denom <= 0).sum())
    if n_bad_denom:
        print(f"警告：Σfacpr 使 1+Σ<=0 的日 {n_bad_denom} 个，事件跳过")
    day_factor = pd.Series(
        np.where(denom > 0, 1.0 / denom, 1.0), index=days.index)
    cumfac_all = day_factor.groupby(permno_all).cumprod()

    n_carry = int(((days["prc"].isna()) & (days["sum_facpr"] != 0.0)).sum())
    print(f"占位日上的事件并入下一有效日：{n_carry} 个")

    # 只输出有有效行情的日子
    valid = days["prc"].notna()
    days = days[valid].reset_index(drop=True)
    cumfac = cumfac_all[valid].reset_index(drop=True)
    permno = days["permno"]

    # 退市末日（每组最后一有效行）与有效退市收益
    grp_index = permno.groupby(permno).cumcount()
    grp_size = permno.groupby(permno).transform("size")
    is_last = grp_index + 1 == grp_size
    valid_dl = days["dlret"].notna() & days["dlret"].between(
        DLRET_MIN, DLRET_MAX, inclusive="neither")
    del grp_index, grp_size

    # 前复权：以最后有效日为锚（末日乘数 1，历史价格按事件缩小）
    scale = cumfac.groupby(permno).transform("last") / cumfac
    adj_close = days["prc"] * scale
    prev_close = adj_close.groupby(permno).shift(1)
    print(f"事件乘数调整日：{int((days['sum_facpr'] != 0.0).sum()):,}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(days["date"])
    out["code"] = "us." + permno.astype(str)

    # open 源缺失 → 前一有效交易日调整后收盘价；首日 → 当日收盘价
    missing_open = days["open"].isna()
    open_fill = prev_close.where(prev_close.notna(), adj_close)
    out["open"] = np.where(missing_open, open_fill, days["open"] * scale)
    n_open_fill = int(missing_open.sum())
    n_firstday_fill = int((missing_open & prev_close.isna()).sum())
    print(f"open 缺失填充：{n_open_fill:,} 行（其中首日用当日收盘价：{n_firstday_fill} 行）")

    # high/low 偶发缺失 → 以当日有效价补（报价日通常只有报价）
    high = days["high"].where(days["high"].notna(), days["prc"])
    low = days["low"].where(days["low"].notna(), days["prc"])
    out["high"] = high * scale
    out["low"] = low * scale
    out["close"] = adj_close
    # 零成交报价日（行情有效、vol 缺失）成交量/额/换手记 0
    vol = days["vol"].fillna(0.0)
    out["volume"] = vol.round().astype("int64")
    out["amount"] = days["prc"] * vol
    # 流通股数为千股；turn 为百分数口径（与 A 股表一致）
    shares = days["shrout"] * 1000.0
    out["turn"] = np.where(shares > 0, vol / shares * 100.0, np.nan)

    # pctChg 总回报口径；源缺失用相邻调整后收盘价补算，不可信则留空
    pct = days["ret"].copy()
    missing = pct.isna()
    derived = adj_close / prev_close - 1.0
    use_derived = missing & prev_close.notna() & (derived.abs() <= DERIVE_BOUND)
    pct.loc[use_derived] = derived.loc[use_derived]
    pct.loc[missing & ~use_derived] = np.nan
    n_blank = int((missing & ~use_derived).sum())
    print(f"pctChg 缺失补算：{int(use_derived.sum()):,} 行；不可信留空：{n_blank:,} 行")

    out["pctChg"] = pct * 100.0
    # 退市末日并入退市收益
    can_merge = is_last & valid_dl & days["ret"].notna()
    merged = ((1.0 + days["ret"]) * (1.0 + days["dlret"]) - 1.0) * 100.0
    out.loc[can_merge, "pctChg"] = merged.loc[can_merge]
    print(f"退市末日并入退市收益：{int(can_merge.sum()):,} 行")

    return out[["date", "code", "open", "high", "low", "close",
                "volume", "amount", "turn", "pctChg"]]


def main() -> None:
    raw = load_raw()
    print(f"原始行合计：{len(raw):,}")

    days = aggregate_days(blank_sentinels(raw))
    del raw
    print(f"按日聚合后：{len(days):,} 个 (PERMNO, 日期)，"
          f"其中有效行情日 {int(days['prc'].notna().sum()):,}")

    securities = build_securities(days)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    securities.to_csv(OUT_DIR / "securities.csv", index=False)
    print(f"securities.csv：{len(securities):,} 只标的，"
          f"其中普通股 {int(securities['included'].sum()):,} 只、"
          f"在市 {int(securities['active'].sum()):,} 只")

    stocks = build_stocks(days)
    del days
    stocks.to_parquet(OUT_DIR / "daily_stocks.parquet", compression="zstd")
    print(f"daily_stocks.parquet：{len(stocks):,} 行，"
          f"{stocks['code'].nunique():,} 只标的，"
          f"{stocks['date'].min().date()} ~ {stocks['date'].max().date()}")


if __name__ == "__main__":
    sys.exit(main())
