#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取指定标的在给定时间区间内的日线数据。

用法：
    from query_a_share import query

    df = query("sh.600000", "2020-01-01", "2020-12-31")            # 个股
    df = query("sh.000300", "2015-01-01", "2015-06-30", "index")   # 指数
"""

from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest" / "a_share"
_PATHS = {
    "stock": DATA_DIR / "daily_stocks.parquet",
    "index": DATA_DIR / "daily_indices.parquet",
}


def query(
    code: str,
    start_date: str,
    end_date: str,
    kind: str = "stock",
) -> pd.DataFrame:
    """读取单只标的在 [start_date, end_date] 内的日线数据。

    Parameters
    ----------
    code : str
        证券代码，如 ``sh.600000``（个股）或 ``sh.000300``（指数）。
    start_date, end_date : str
        起止日期，``YYYY-MM-DD`` 格式，区间两端均包含。
    kind : {"stock", "index"}, default "stock"
        标的类型：``"stock"`` 查个股（含退市股），``"index"`` 查基准指数。

    Returns
    -------
    pd.DataFrame
        按日期升序排列；查无此标的或区间内无行情时返回空 DataFrame。
    """
    if kind not in _PATHS:
        raise ValueError(f"kind 只能是 'stock' 或 'index'，收到：{kind!r}")

    con = duckdb.connect()
    try:
        df = con.execute(
            """
            SELECT * FROM read_parquet($path)
            WHERE code = $code
              AND CAST(date AS DATE) BETWEEN $start_date AND $end_date
            ORDER BY date
            """,
            {
                "path": str(_PATHS[kind]),
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
            },
        ).fetchdf()
    finally:
        con.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
