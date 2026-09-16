# A 股日线回测数据集

面向策略回测的 A 股全市场日线数据，**包含已退市股票**，可直接用于无幸存者偏差
（survivorship-bias-free）的历史回测。

- 区间：**2006-01-01 ~ 2025-12-31（20 个自然年）**，实际首日交易日为 2006-01-04
- 复权：**前复权**（固定快照，历史价不会再变动）
- 频率：日线
- 规模：个股 5464 只（含区间内退市的股票）、约 1502 万行；指数 7 只、32,698 行

## 如何消除幸存者偏差

股票范围为 20 年间**全部上市过的股票**，包括在 2006–2025 年内退市、如今已从行情
软件消失的标的（如邯郸钢铁 sh.600001，2009 年退市；退市昌鱼 sh.600275，2022 年
退市），它们退市前的完整行情都在数据集中，没有做任何"当前成分股"过滤。

构造任意历史时点的可交易股票池时，以**当日是否存在行情记录**为准——有记录即当时
已上市且未退市；退市股在退市次日后自然不再出现。停牌日同样没有记录，而停牌股票
本就不可交易，因此该口径可直接用于回测。

## 文件说明

| 文件 | 内容 |
|---|---|
| `daily_stocks.parquet` | 全部个股日线（含退市），ZSTD 压缩，按 `code, date` 排序 |
| `daily_indices.parquet` | 7 只基准指数日线，ZSTD 压缩，按 `code, date` 排序 |

## 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `date` | TIMESTAMP | 交易日 |
| `code` | VARCHAR | 证券代码，如 `sh.600000`、`sz.000002` |
| `open` / `high` / `low` / `close` | DOUBLE | 前复权开高低收（元） |
| `volume` | BIGINT | 成交量（股） |
| `amount` | DOUBLE | 成交额（元） |
| `turn` | DOUBLE | 换手率（%） |
| `pctChg` | DOUBLE | 日涨跌幅（%，不复权口径） |

停牌日无记录；个别历史字段缺失为 NULL。

## 基准指数

| 代码 | 名称 | 备注 |
|---|---|---|
| sh.000001 | 上证综指 | |
| sz.399001 | 深证成指 | |
| sz.399006 | 创业板指 | 2010-06-01 起 |
| sh.000016 | 上证 50 | |
| sh.000300 | 沪深 300 | |
| sh.000905 | 中证 500 | |
| sh.000852 | 中证 1000 | 发布前点位由指数公司回算填充 |

## 查询示例

DuckDB 直接读 Parquet（无需导入）：

```python
import duckdb
con = duckdb.connect()

# 取浦发银行 2020 年以来行情
con.execute("""
    SELECT * FROM read_parquet('data/backtest/a_share/daily_stocks.parquet')
    WHERE code = 'sh.600000' AND date >= '2020-01-01'
""").fetchdf()

# 某历史时点真实可交易的股票池（含此后退市的股票）：当日有行情记录即可
con.execute("""
    SELECT DISTINCT code
    FROM read_parquet('data/backtest/a_share/daily_stocks.parquet')
    WHERE date = DATE '2015-06-15'
""").fetchdf()

# 沪深300 基准
con.execute("""
    SELECT date, close FROM read_parquet('data/backtest/a_share/daily_indices.parquet')
    WHERE code = 'sh.000300' ORDER BY date
""").fetchdf()
```

或使用查询封装 `scripts/query_a_share.py`：

```python
from scripts.query_a_share import AShareDB

db = AShareDB()
db.stocks(codes=["sh.600000"], start="2020-01-01")
db.index("沪深300")                       # 也可传代码 sh.000300
db.trading_days("2024-01-01", "2024-12-31")
```

命令行：

```bash
python scripts/query_a_share.py index --name 沪深300 --tail 5
python scripts/query_a_share.py stock sh.600000 sz.000002 --start 2025-01-01
```

## 注意事项

1. 个股范围为沪深 A 股股票（不含 B 股、ETF、可转债）。
2. 本数据集为截止 2025-12-31 的固定快照，不做每日更新；前复权基准已固定。
3. `pctChg` 为真实日涨跌幅，回测收益率建议优先用它，而非前复权收盘价之比。
