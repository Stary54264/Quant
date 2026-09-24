#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""市场注册表：各市场的显示名、数据区间、指数代码与 parquet 路径。

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
    indices: tuple[str, ...]
    readme: Path
    stocks_path: Path
    indices_path: Path

    @property
    def available(self) -> bool:
        """个股与指数两个 parquet 齐全才算该市场可用。"""
        return self.stocks_path.exists() and self.indices_path.exists()

    def path_for(self, code: str) -> Path:
        """指数代码走指数文件，其余一律走个股文件（不存在由查询判空）。"""
        return self.indices_path if code in self.indices else self.stocks_path

    def is_index(self, code: str) -> bool:
        return code in self.indices

    def normalize_code(self, code: str) -> str:
        """规范化用户输入：A 股整体小写；美股前缀小写、后缀大写。

        美股个股后缀是数字（大小写无关），指数后缀是字母（us.SPX，必须大写）。
        """
        if self.key == "us":
            head, sep, tail = code.partition(".")
            return f"{head.lower()}{sep}{tail.upper()}" if sep else code.lower()
        return code.lower()


def _market(key: str, display_name: str, min_date: date, max_date: date,
            code_example: str, indices: tuple[str, ...]) -> Market:
    base = ROOT / "data" / "backtest" / key
    return Market(
        key=key,
        display_name=display_name,
        min_date=min_date,
        max_date=max_date,
        code_example=code_example,
        indices=indices,
        readme=base / "README.md",
        stocks_path=base / "daily_stocks.parquet",
        indices_path=base / "daily_indices.parquet",
    )


# 注册顺序即 UI 展示顺序
MARKETS: dict[str, Market] = {
    "a_share": _market(
        "a_share", "A股", date(2006, 1, 4), date(2025, 12, 31),
        "sh.600000",
        ("sh.000001", "sh.000016", "sh.000300", "sh.000852",
         "sh.000905", "sz.399001", "sz.399006"),
    ),
    # max 取个股与指数覆盖的较大者：个股止于 2024，指数已含 2025；
    # 个股选到 2025 时报告按实际最后交易日显示（与新股/退市股行为一致）
    "us": _market(
        "us", "美股", date(2006, 1, 3), date(2025, 12, 31),
        "us.14593",
        ("us.DJI", "us.SPX", "us.IXIC", "us.RUT"),
    ),
}

DEFAULT_MARKET = "a_share"


def get_market(key: str) -> Market:
    return MARKETS[key]


def available_markets() -> dict[str, Market]:
    """返回数据齐全的市场子集，保持注册顺序。"""
    return {k: v for k, v in MARKETS.items() if v.available}
