#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取指定标的在给定时间区间内的日线数据。

用法：
    python scripts/query_a_share.py sh.600000 2020-01-01 2020-12-31
    python scripts/query_a_share.py sh.000300 2015-01-01 2015-06-30 --csv

也可作为模块导入：
    from query_a_share import query
    df = query("sh.600000", "2020-01-01", "2020-12-31")
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backtest" / "a_share"
STOCKS_PATH = DATA_DIR / "daily_stocks.parquet"
INDICES_PATH = DATA_DIR / "daily_indices.parquet"


def query(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """读取单只标的（个股或指数）在 [start_date, end_date] 内的日线数据。

    Parameters
    ----------
    code : str
        证券代码，如 ``sh.600000``（个股）或 ``sh.000300``（指数）。
    start_date, end_date : str
        起止日期，``YYYY-MM-DD`` 格式，区间两端均包含。

    Returns
    -------
    pd.DataFrame
        按日期升序排列；查无此标的或区间内无行情时返回空 DataFrame。
    """
    con = duckdb.connect()
    try:
        # 个股和指数分属两个文件，先在个股中找，找不到再查指数
        df = con.execute(
            """
            SELECT * FROM read_parquet($stocks)
            WHERE code = $code
              AND CAST(date AS DATE) BETWEEN $start_date AND $end_date
            ORDER BY date
            """,
            {
                "stocks": str(STOCKS_PATH),
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
            },
        ).fetchdf()

        if df.empty:
            df = con.execute(
                """
                SELECT * FROM read_parquet($indices)
                WHERE code = $code
                  AND CAST(date AS DATE) BETWEEN $start_date AND $end_date
                ORDER BY date
                """,
                {
                    "indices": str(INDICES_PATH),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="读取指定标的、指定时间区间的 A 股日线数据")
    parser.add_argument("code", help="证券代码，如 sh.600000 或 sh.000300")
    parser.add_argument("start_date", help="起始日期 YYYY-MM-DD（含）")
    parser.add_argument("end_date", help="终止日期 YYYY-MM-DD（含）")
    parser.add_argument("--csv", action="store_true", help="以 CSV 格式输出（默认打印表格）")
    args = parser.parse_args()

    df = query(args.code, args.start_date, args.end_date)
    if df.empty:
        print(f"未找到 {args.code} 在 {args.start_date} ~ {args.end_date} 的数据")
        return

    if args.csv:
        print(df.to_csv(index=False))
    else:
        print(df.to_string(index=False))
        print(f"\n共 {len(df)} 个交易日")


if __name__ == "__main__":
    main()
