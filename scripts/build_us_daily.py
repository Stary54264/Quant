#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""美股日线数据集构建：把 CRSP 抽取 CSV 清理为与 A 股同构的 parquet。

输入：data/backtest/ 下的 CRSP 抽取 CSV（支持多个文件，追加 2025 等年份时
把新文件放同目录、文件名以 crsp_ 开头即可，脚本会全部读入后按日聚合）。
输出（data/backtest/us/）：
    daily_stocks.parquet   10 列，与 a_share 的 parquet 逐列一致
    securities.csv         全部 PERMNO 的对照/生命周期表

口径（与 A 股数据集对齐：OHLC 与 pctChg 同为总回报口径，数值一致）：
    - open/high/low/close 为前复权价。逐日链因子 total_day = event_day/div_day
      （财富因子的作用方向与前复权链比相反），cumprod 后以末日为锚：
      * 价格事件（拆股、合股、送股）坚持用源数据事件乘数，不用收益反推：
        同一交易日多个分布行时 facpr 相加，事件比例 = 1/(1+Σfacpr)（不是
        各因子相乘）；事件可能落在无行情占位行（prc=0、vol=-99），乘数链
        在全部日历日上 cumprod，占位日的事件自动并入下一有效交易日；
      * facpr=0 为现金分红除息日标记，事件乘数为 1。现金分红金额在抽取中
        没有独立列，分红乘数仅在 facpr 显式非空的"分布事件日"上由官方日
        收益的恒等式残差提取（RET 仅 6 位小数，非事件日残差噪声太大，不
        能逐日提取）：
            分红乘数 = (1+RET) / [(原始价比) × (1+Σfacpr)]
        非分布事件日强制为 1；
    - pctChg 为日涨跌幅（%，总回报口径）：取自源数据官方日收益 RET；官方
      值缺失（-66 等）时用相邻保留日的前复权收盘比补算（超 ±400% 视为
      不可信，留空），首日有官方 RET 照常输出；
    - 退市末日：pctChg 在有官方 RET 时并入退市期间收益 DLRET
      ((1+RET)(1+DLRET)-1)，保证持有到退市的总回报完整；DLRET 仅取严格
      位于 (-1,1) 的小数（-88/-55/-66 等整数为缺失码），并另行导出
      delisting_returns.csv；
    - open 缺失（源记 0，多为盘中才出现第一笔交易）时，以**当日高低均值**
      填充——盘中价格不含隔夜跳空，优于用前收盘；
    - 仅保留普通股（CRSP share code 首位为 1；其他标的仍记入 securities.csv，
      included=False）；无任何有效行情的占位日不输出；
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

# 日收益缺失码：置 NaN 后改用相邻保留日的前复权收盘比补算
MISSING_RETS = (-77.0, -99.0, -66.0)
DERIVE_BOUND = 4.0          # 补算收益超 ±400% 视为不可信，留空
# 分红乘数的可信区间（价格事件已由 facpr 处理，单日分红率不会极端）
DIV_FACTOR_MIN, DIV_FACTOR_MAX = 0.1, 10.0
DIV_EPS = 1e-9              # 偏离 1 小于此值视为浮点噪声，归一为 1
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

    另输出 n_facpr（当日 facpr 显式非空行数）：全 NaN 求和同样为 0，必须靠
    它区分"现金分红除息日（facpr 显式 =0）"与"无事件日"。
    """
    df = df.sort_values(["permno", "date"])
    # last 自动跳过 NaN：同日"占位行 + 有效行"取到有效行情
    kwargs = {col: (col, "last") for col in df.columns
              if col not in ("permno", "date", "facpr")}
    kwargs["sum_facpr"] = ("facpr", "sum")
    kwargs["n_facpr"] = ("facpr", "count")
    return df.groupby(["permno", "date"], sort=True).agg(**kwargs).reset_index()


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


def build_delisting(days: pd.DataFrame) -> pd.DataFrame:
    """退市期间收益表：每只有退市记录的证券一行。

    源数据把 DLRET 作为证券级常量填在该证券的每一条行情行上，故按 PERMNO
    取值（任意行）并以其最后有效交易日为日期。仅接受严格位于 (-1,1) 的
    小数（-88/-55/-66 等整数为缺失码）。成品 pctChg 在退市末日已并入该值，
    本表另行完整保留。
    """
    valid_days = days[days["prc"].notna()]
    g = valid_days.groupby("permno").agg(
        dlret=("dlret", "last"),
        date=("date", "last"),
    )
    g = g[g["dlret"].between(DLRET_MIN, DLRET_MAX, inclusive="neither")]
    g = g.reset_index()
    g["code"] = "us." + g["permno"].astype(str)
    g["dlret_pct"] = g["dlret"] * 100.0
    out = g[["code", "date", "dlret_pct"]]
    return out.sort_values("code").reset_index(drop=True)


def build_stocks(days: pd.DataFrame) -> pd.DataFrame:
    """过滤普通股，按"事件乘数 × 分红乘数"链重建总回报前复权价，返回成品。"""
    days = days[days["shrcd"].str.startswith("1")].copy()
    permno_all = days["permno"]

    # --- 事件日乘数（全部日历日，含占位日，使占位日事件得以 carry）---
    denom = 1.0 + days["sum_facpr"]
    n_bad_denom = int((denom <= 0).sum())
    if n_bad_denom:
        print(f"警告：Σfacpr 使 1+Σ<=0 的日 {n_bad_denom} 个，事件跳过")
    event_day = pd.Series(
        np.where(denom > 0, 1.0 / denom, 1.0), index=days.index)

    n_carry = int(((days["prc"].isna()) & (days["sum_facpr"] != 0.0)).sum())
    print(f"占位日上的事件并入下一有效日：{n_carry} 个")

    # --- 分红日乘数（仅有效行情日、且当日有显式分布事件 facpr 时才提取）---
    # RET 只有 6 位小数，非事件日"RET/价格商"的残差噪声达 1e-7~1e-4，故
    # 不能逐日提取；n_facpr>0（facpr 显式非空）是"当日有分布事件"的权威
    # 开关：现金分红日 facpr 显式 =0，拆股日非 0，残差给出事件净效应
    valid_all = days["prc"].notna()
    v = days[valid_all]
    raw_prev = v["prc"].groupby(v["permno"]).shift(1)
    raw_ratio = v["prc"] / raw_prev
    implied = (1.0 + v["ret"]) / (raw_ratio * (1.0 + v["sum_facpr"]))
    has_dist = v["n_facpr"] > 0
    ok = (has_dist & v["ret"].notna() & raw_prev.notna()
          & implied.between(DIV_FACTOR_MIN, DIV_FACTOR_MAX))
    div_day = pd.Series(1.0, index=v.index)
    div_day[ok] = implied[ok]
    div_day = div_day.where(div_day.sub(1.0).abs() >= DIV_EPS, 1.0)
    days["div_day"] = 1.0
    days.loc[valid_all, "div_day"] = div_day
    n_div = int((div_day != 1.0).sum())
    n_dist = int(has_dist.sum())
    print(f"显式分布事件日 {n_dist:,}，其中现金分红等残差乘数调整：{n_div:,}")

    # --- 总乘数链在全部日历日上 cumprod ---
    # 链比的作用方向与财富因子相反：拆股日 event_day=1/(1+facpr) 使 scale 在
    # 事件日反向乘 (1+facpr)；分红日 d>1 使 scale 反向乘 d。故：
    #     total_day = event_day / div_day
    # 保证 raw_ratio / total_day = RET
    total_day = event_day / days["div_day"]
    cumfac_all = total_day.groupby(permno_all).cumprod()

    # 只输出有有效行情的日子
    days = days[valid_all].reset_index(drop=True)
    cumfac = cumfac_all[valid_all].reset_index(drop=True)
    permno = days["permno"]

    # 前复权：以最后有效日为锚（末日乘数 1，历史价格按累计乘数缩小）
    scale = cumfac.groupby(permno).transform("last") / cumfac
    adj_close = days["prc"] * scale
    prev_close = adj_close.groupby(permno).shift(1)
    print(f"事件乘数调整日：{int((days['sum_facpr'] != 0.0).sum()):,}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(days["date"])
    out["code"] = "us." + permno.astype(str)

    # high/low 偶发缺失 → 以当日有效价补（报价日通常只有报价）
    high = days["high"].where(days["high"].notna(), days["prc"])
    low = days["low"].where(days["low"].notna(), days["prc"])
    out["high"] = high * scale
    out["low"] = low * scale
    out["close"] = adj_close

    # open 源缺失 → 以当日高低均值填（盘中价、不含隔夜跳空；不用前收）
    out["open"] = (days["open"] * scale).fillna(
        (out["high"] + out["low"]) / 2.0)
    n_open_fill = int(days["open"].isna().sum())
    print(f"open 缺失用当日高低均值填充：{n_open_fill:,} 行")
    # 零成交报价日（行情有效、vol 缺失）成交量/额/换手记 0
    vol = days["vol"].fillna(0.0)
    out["volume"] = vol.round().astype("int64")
    out["amount"] = days["prc"] * vol
    # 流通股数为千股；turn 为百分数口径（与 A 股表一致）
    shares = days["shrout"] * 1000.0
    out["turn"] = np.where(shares > 0, vol / shares * 100.0, np.nan)

    # pctChg 总回报口径：优先官方 RET；官方缺失用前复权收盘比补算，
    # 超 ±DERIVE_BOUND 不可信留空（如停牌价 0.02 → 复牌 75.65）
    derived = adj_close / prev_close - 1.0
    fill_rows = (days["ret"].isna() & prev_close.notna()
                 & derived.abs().le(DERIVE_BOUND))
    pct = days["ret"].copy()
    pct.loc[fill_rows] = derived.loc[fill_rows]
    n_filled = int(fill_rows.sum())
    n_blank = int(pct.isna().sum())
    print(f"pctChg 官方缺失用前复权收盘比补算：{n_filled:,} 行；留空：{n_blank:,} 行")

    out["pctChg"] = pct * 100.0

    # 退市末日并入退市期间收益（持有到退市的总回报才完整）
    grp_index = permno.groupby(permno).cumcount()
    grp_size = permno.groupby(permno).transform("size")
    is_last = grp_index + 1 == grp_size
    valid_dl = days["dlret"].notna() & days["dlret"].between(
        DLRET_MIN, DLRET_MAX, inclusive="neither")
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
    delisting = build_delisting(days)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    securities.to_csv(OUT_DIR / "securities.csv", index=False)
    print(f"securities.csv：{len(securities):,} 只标的，"
          f"其中普通股 {int(securities['included'].sum()):,} 只、"
          f"在市 {int(securities['active'].sum()):,} 只")
    delisting.to_csv(OUT_DIR / "delisting_returns.csv", index=False)
    print(f"delisting_returns.csv：{len(delisting):,} 行退市期间收益")

    stocks = build_stocks(days)
    del days
    stocks.to_parquet(OUT_DIR / "daily_stocks.parquet", compression="zstd")
    print(f"daily_stocks.parquet：{len(stocks):,} 行，"
          f"{stocks['code'].nunique():,} 只标的，"
          f"{stocks['date'].min().date()} ~ {stocks['date'].max().date()}")


if __name__ == "__main__":
    sys.exit(main())
