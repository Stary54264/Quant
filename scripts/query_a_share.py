#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A 股日线数据集查询封装（DuckDB 直读 Parquet，无需导入）。

库内使用：
    from query_a_share import AShareDB          # 或 from scripts.query_a_share import ...
    db = AShareDB()
    db.stocks(start="2020-01-01", codes=["sh.600000"])
    db.index(name="沪深300", start="2020-01-01")
命令行：
    python scripts/query_a_share.py index --name 沪深300 --start 2026-01-01 --tail 5
    python scripts/query_a_share.py stock sh.600000 --start 2026-01-01 --tail 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "backtest" / "a_share"

# 与抓取脚本保持一致的指数代码表
INDEX_NAMES: dict[str, str] = {
    "sh.000001": "上证综指",
    "sz.399001": "深证成指",
    "sz.399006": "创业板指",
    "sh.000016": "上证50",
    "sh.000300": "沪深300",
    "sh.000905": "中证500",
    "sh.000852": "中证1000",
}
NAME_TO_CODE = {v: k for k, v in INDEX_NAMES.items()}


class AShareDB:
    """对 daily_stocks / daily_indices 两个 Parquet 的只读 DuckDB 查询封装。"""

    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.stocks_path = self.data_dir / "daily_stocks.parquet"
        self.indices_path = self.data_dir / "daily_indices.parquet"
        self.con = duckdb.connect()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "AShareDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _where(codes: Optional[Iterable[str]], start: Optional[str],
               end: Optional[str], code_col: str = "code") -> tuple[str, dict]:
        where, params = [], {}
        if codes is not None:
            codes = list(codes)
            if codes:
                where.append(f"{code_col} = ANY($codes)")
                params["codes"] = codes
        if start:
            where.append("date >= $start")
            params["start"] = pd.Timestamp(start)
        if end:
            where.append("date <= $end_date")
            params["end_date"] = pd.Timestamp(end)
        sql = (" WHERE " + " AND ".join(where)) if where else ""
        return sql, params

    def _read(self, path: Path, codes, start, end) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}（请先运行 fetch_a_share_daily.py）")
        where, params = self._where(codes, start, end)
        return self.con.execute(
            f"SELECT * FROM read_parquet($path){where} ORDER BY code, date",
            {"path": str(path), **params},
        ).fetchdf()

    # ------------------------------------------------------------------
    def stocks(self, codes: Optional[Iterable[str]] = None,
               start: Optional[str] = None,
               end: Optional[str] = None) -> pd.DataFrame:
        """个股日线（前复权），含已退市股票；codes 为 baostock 代码列表。"""
        return self._read(self.stocks_path, codes, start, end)

    def indices(self, codes: Optional[Iterable[str]] = None,
                start: Optional[str] = None,
                end: Optional[str] = None) -> pd.DataFrame:
        """全部基准指数日线。"""
        return self._read(self.indices_path, codes, start, end)

    def index(self, name_or_code: str, start: Optional[str] = None,
              end: Optional[str] = None) -> pd.DataFrame:
        """按名称（如“沪深300”）或代码（如 sh.000300）取单只指数。"""
        code = NAME_TO_CODE.get(name_or_code, name_or_code)
        return self._read(self.indices_path, [code], start, end)

    # ------------------------------------------------------------------
    def trading_days(self, start: Optional[str] = None,
                     end: Optional[str] = None) -> pd.DatetimeIndex:
        """以沪深300的交易日历为准。"""
        df = self.index("沪深300", start, end)[["date"]].drop_duplicates()
        return pd.DatetimeIndex(sorted(df["date"]))

    def tradeable_codes(self, day: str) -> list[str]:
        """某历史时点真实可交易的股票代码（当日有行情记录；含此后退市的股票，
        不含当日停牌股票）。"""
        rows = self.con.execute(
            "SELECT DISTINCT code FROM read_parquet($path) WHERE date = $d ORDER BY code",
            {"path": str(self.stocks_path), "d": pd.Timestamp(day)},
        ).fetchall()
        return [r[0] for r in rows]


def _cli() -> None:
    ap = argparse.ArgumentParser(description="查询 A 股日线数据集")
    sub = ap.add_subparsers(dest="kind", required=True)
    s1 = sub.add_parser("stock", help="查询个股，可传多个代码")
    s1.add_argument("codes", nargs="+")
    s1.add_argument("--start"); s1.add_argument("--end"); s1.add_argument("--tail", type=int)
    s2 = sub.add_parser("index", help="查询指数（名称或代码，默认沪深300）")
    s2.add_argument("--name", default="沪深300")
    s2.add_argument("--start"); s2.add_argument("--end"); s2.add_argument("--tail", type=int)
    args = ap.parse_args()

    with AShareDB() as db:
        if args.kind == "stock":
            df = db.stocks(args.codes, args.start, args.end)
        else:
            df = db.index(args.name, args.start, args.end)
    if args.tail:
        df = df.groupby("code", group_keys=False).tail(args.tail)
    with pd.option_context("display.max_rows", 100, "display.width", 160):
        print(df.to_string(index=False))


if __name__ == "__main__":
    _cli()
