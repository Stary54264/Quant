#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""市场注册表：各市场的显示名、数据区间与 parquet 路径。

纯数据、零三方依赖，供 UI（app.py）、报告入口（generate_report.py）与启动器
自检共用。新增市场只需在 MARKETS 中注册一项。
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Market:
    key: str
    display_name: str
    min_date: date
    max_date: date
    code_example: str
    readme: Path
    stocks_path: Path
    indices_path: Path

    @property
    def available(self) -> bool:
        """个股与指数两个 parquet 齐全才算该市场可用。"""
        return self.stocks_path.exists() and self.indices_path.exists()


def _market(key: str, display_name: str, min_date: date, max_date: date,
            code_example: str) -> Market:
    base = ROOT / "data" / "backtest" / key
    return Market(
        key=key,
        display_name=display_name,
        min_date=min_date,
        max_date=max_date,
        code_example=code_example,
        readme=base / "README.md",
        stocks_path=base / "daily_stocks.parquet",
        indices_path=base / "daily_indices.parquet",
    )


# 注册顺序即 UI 展示顺序
MARKETS: dict[str, Market] = {
    "a_share": _market("a_share", "A股", date(2006, 1, 4),
                       date(2025, 12, 31), "sh.600000"),
    "us": _market("us", "美股", date(2006, 1, 3),
                  date(2024, 12, 31), "us.14593"),
}

DEFAULT_MARKET = "a_share"


def get_market(key: str) -> Market:
    return MARKETS[key]


def available_markets() -> dict[str, Market]:
    """返回数据齐全的市场子集，保持注册顺序。"""
    return {k: v for k, v in MARKETS.items() if v.available}
