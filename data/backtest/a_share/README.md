# A 股日线回测数据集（baostock）

面向策略回测的 A 股全市场日线数据，**包含已退市股票**，可直接用于无幸存者偏差
（survivorship-bias-free）的历史回测。

- 数据源：[baostock](http://baostock.com) `query_history_k_data_plus`
- 区间：**2006-01-01 ~ 最新交易日**（实际起止以各标的上市/退市日为准）
- 复权：**前复权**（`adjustflag="2"`，复权基准为最近一个交易日，每次增量更新后历史价会整体重锚）
- 频率：日线
- 更新方式：重跑 `scripts/fetch_a_share_daily.py`，已下载标的自动跳过，失败标的自动重试

## 如何消除幸存者偏差

证券清单取自 `bs.query_stock_basic()`，筛选 `type='1'`（股票）且
`status ∈ {'1','0'}`（在市 + 退市，baostock 实际用 `0` 表示退市），不做任何
"当前成分股"过滤。因此像邯郸钢铁（sh.600001，2009 年退市）、齐鲁石化
（sh.600002，2006 年退市）这类已消失的标的及其退市前的完整行情都在数据集中。

> 回测构造任意历史时点的可交易股票池时，应按当日日期过滤：
> `ipoDate <= d <= outDate`（`outDate` 为空表示仍在市），元数据见 `stock_basic.csv`。

## 文件说明

| 文件 | 内容 |
|---|---|
| `daily_stocks.parquet` | 全部个股日线（含退市），ZSTD 压缩，按 `code, date` 排序 |
| `daily_indices.parquet` | 7 只基准指数日线，ZSTD 压缩，按 `code, date` 排序 |
| `stock_basic.csv` | baostock 证券清单快照（代码、名称、类型、状态、上市/退市日期） |
| `manifest.json` | 本次构建的行数、覆盖区间、指数明细等元数据 |

单标的临时文件与断点目录为 `data/raw/a_share/{code}.parquet`（已被 .gitignore
忽略），每只标的下载成功后独立落盘；网络中断后直接重跑即可续传。

## 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `date` | TIMESTAMP | 交易日 |
| `code` | VARCHAR | baostock 代码，如 `sh.600000`、`sz.000002` |
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
| sh.000852 | 中证 1000 | 发布前点位由中证指数公司回算填充 |
| ~~sh.000688~~ | ~~科创 50~~ | **baostock 不提供**（指数名录与 K 线均为空），需另接数据源 |
| ~~sh.932000~~ | ~~中证 2000~~ | **baostock 不提供**，同上 |

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

# 某历史时点“真实可交易”的股票池（含此后退市的股票）
con.execute("""
    WITH px AS (
        SELECT DISTINCT code
        FROM read_parquet('data/backtest/a_share/daily_stocks.parquet')
        WHERE date = DATE '2015-06-15'
    )
    SELECT b.code, b.code_name
    FROM read_csv_auto('data/backtest/a_share/stock_basic.csv') b
    JOIN px ON px.code = b.code
    WHERE b.type = '1'
      AND b.ipoDate <= '2015-06-15'
      AND (b.outDate = '' OR b.outDate > '2015-06-15')
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
db.universe()                             # 含退市状态的证券清单
```

命令行：

```bash
python scripts/query_a_share.py index --name 沪深300 --tail 5
python scripts/query_a_share.py stock sh.600000 sz.000002 --start 2026-01-01
```

## 重新抓取 / 增量更新

```bash
# 全量（断点续传；实测 8 进程并发约 7s/只，建议 --workers 8，16 进程以上服务端反而限速）
python scripts/fetch_a_share_daily.py --workers 8

# 仅重新合并（改了合并逻辑时用）
python scripts/fetch_a_share_daily.py --skip-download
```

- 每只标的最多重试 3 轮（指数退避 + 重新登录）；最终仍失败的代码写入
  `data/raw/a_share/_failed.tsv`，重跑脚本会只重试这些，已完成的不会重复下载。
- 增量更新时，已存在的单标的文件默认跳过；如需刷新最近行情，删除对应
  `{code}.parquet` 后重跑（前复权数据刷新后历史价格会重新锚定，属正常现象）。

## 注意事项

1. 个股范围为 baostock `type='1'` 的沪深股票（不含 B 股、ETF、可转债）。
2. 前复权价以最新交易日为基准，跨更新批次比较绝对价格前请重新拉取对齐。
3. `pctChg` 为真实日涨跌幅，回测收益率建议优先用它，而非前复权收盘价之比
   （后者在复权基准变动后会有整体漂移）。
