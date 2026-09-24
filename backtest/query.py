#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一行情读取：所有市场（A 股、美股及后续新增数据源）共用同一读取方法。

约定各数据源产出同构文件（列名、顺序、类型一致），区别只在文件路径——
路径由调用方给定，本模块不绑定任何市场或目录：

    date, code, open, high, low, close, volume, amount, turn, pctChg

用法：
    from backtest.query import query

    df = query("data/backtest/a_share/daily_stocks.parquet",
               ["sh.600000"], "2020-01-01", "2020-12-31")
    df = query("data/backtest/us/daily_stocks.parquet",
               ["us.14593", "us.93436"], "2020-01-01", "2020-12-31")

个股与指数、历史快照与后续实时数据只是不同的数据文件，调用方传对应路径即可。
"""

import duckdb
import pandas as pd


def query(
    path: str,
    codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """从指定数据文件中读取若干标的在 [start_date, end_date] 内的日线数据。

    Parameters
    ----------
    path : str
        数据文件路径（parquet），如某市场的 ``daily_stocks.parquet`` 或
        ``daily_indices.parquet``。
    codes : list[str]
        证券代码列表，如 ``["sh.600000"]``、``["us.14593", "us.93436"]``
        或指数代码；一次可查一只或多只。
    start_date, end_date : str
        起止日期，``YYYY-MM-DD`` 格式，区间两端均包含。

    Returns
    -------
    pd.DataFrame
        按代码、日期升序排列；查无此标的或区间内无行情时返回空 DataFrame。
    """
    con = duckdb.connect()
    try:
        df = con.execute(
            """
            SELECT * FROM read_parquet($path)
            WHERE code = ANY($codes)
              AND CAST(date AS DATE) BETWEEN $start_date AND $end_date
            ORDER BY code, date
            """,
            {
                "path": str(path),
                "codes": codes,
                "start_date": start_date,
                "end_date": end_date,
            },
        ).fetchdf()
    finally:
        con.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
