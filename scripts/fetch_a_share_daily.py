#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A 股历史回测日线数据抓取流水线（baostock）

特性：
  * 无幸存者偏差：证券清单来自 bs.query_stock_basic()，同时包含在市(status=1)
    与已退市(status=0)的全部股票（type=1）。
  * 断点续传：每只标的单独落盘 data/raw/a_share/{code}.parquet，已存在则跳过；
    查询失败不写文件，重跑时自动只重试失败标的。
  * 网络重试：每次查询最多尝试 3 轮（指数退避 + 重新登录），失败打印告警并跳过。
  * 最终用 DuckDB 合并、ZSTD 压缩、按 code/date 排序输出。

用法：
  python scripts/fetch_a_share_daily.py                 # 下载 + 合并
  python scripts/fetch_a_share_daily.py --workers 4     # 4 进程并发下载
  python scripts/fetch_a_share_daily.py --skip-download # 只重新合并
  python scripts/fetch_a_share_daily.py --limit 20      # 抽样试跑
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

# baostock 0.9.3 内部仍在调用 pandas 2.0 已移除的 DataFrame.append
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = pd.DataFrame._append  # type: ignore[attr-defined]

import baostock as bs  # noqa: E402
import duckdb  # noqa: E402
from tqdm import tqdm  # noqa: E402

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "a_share"            # 临时单标的文件 / 断点
OUT_DIR = ROOT / "data" / "backtest" / "a_share"       # 最终数据集
UNIVERSE_RAW = RAW_DIR / "stock_basic.csv"             # 接口原始清单快照
UNIVERSE_PUB = OUT_DIR / "stock_basic.csv"             # 随数据集留存的清单
STOCKS_OUT = OUT_DIR / "daily_stocks.parquet"
INDICES_OUT = OUT_DIR / "daily_indices.parquet"
MANIFEST_OUT = OUT_DIR / "manifest.json"
FAILED_LOG = RAW_DIR / "_failed.tsv"

START_DATE = "2006-01-01"
END_DATE = date.today().isoformat()
ADJUSTFLAG = "2"  # 前复权
MAX_TRIES = 3     # 每次查询的最大尝试轮数（含首次）
QUERY_TIMEOUT = 180  # 单次查询看门狗超时（秒）；baostock 无内置超时，recv 可能永久挂死
FIELDS = ["date", "code", "open", "high", "low", "close",
          "volume", "amount", "turn", "pctChg"]
FLOAT_COLS = ["open", "high", "low", "close", "amount", "turn", "pctChg"]

# 基准指数（baostock 实际覆盖 7 只）
INDICES: dict[str, str] = {
    "sh.000001": "上证综指",
    "sz.399001": "深证成指",
    "sz.399006": "创业板指",
    "sh.000016": "上证50",
    "sh.000300": "沪深300",
    "sh.000905": "中证500",
    "sh.000852": "中证1000",
}
# 用户要求的这两只 baostock 指数名录(type=2)中不存在、K 线接口恒返回空，
# 在此显式标注，避免静默缺失。
INDICES_UNAVAILABLE: dict[str, str] = {
    "sh.000688": "科创50",
    "sh.932000": "中证2000",
}

logger = logging.getLogger("fetch_a_share")


# ----------------------------------------------------------------------------
# baostock 会话与重试
# ----------------------------------------------------------------------------
def _login() -> bool:
    """登录 baostock（吞掉其自带的 print 噪音）。"""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            lg = bs.login()
        return lg.error_code == "0"
    except Exception:
        return False


def _logout() -> None:
    with contextlib.suppress(Exception):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bs.logout()


def _reconnect() -> None:
    _logout()
    time.sleep(1)
    _login()


def _ensure_session() -> None:
    if not _login():
        raise RuntimeError("baostock login failed")


def _watchdog(fn, timeout: int):
    """在守护线程中执行 baostock 调用（含 get_data 收数），超时则关掉底层连接解卡。"""
    box: dict = {}

    def _run() -> None:
        try:
            box["result"] = fn()
        except Exception as exc:  # 连接重置 / broken pipe 等
            box["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        # baostock 连接是模块级全局 socket，logout 会 close 它，令卡死的 recv 抛错
        _logout()
        raise TimeoutError(f"查询超过 {timeout}s 未响应")
    if "error" in box:
        raise box["error"]
    return box["result"]


def query_with_retry(query_fn, what: str) -> pd.DataFrame:
    """执行一次 baostock 查询（query + get_data 全程看门狗），
    异常 / error_code!=0 / 挂死均触发重试，最多 MAX_TRIES 轮。"""
    def _one_attempt():
        rs = query_fn()
        return rs, (rs.get_data() if rs is not None and rs.error_code == "0" else None)

    last_err = ""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            rs, df = _watchdog(_one_attempt, QUERY_TIMEOUT)
            if rs is not None and rs.error_code == "0":
                return df
            last_err = f"{getattr(rs, 'error_code', '?')} {getattr(rs, 'error_msg', '')}"
        except Exception as exc:  # 连接重置/超时等
            last_err = repr(exc)
        if attempt < MAX_TRIES:
            backoff = min(5 * attempt, 25)  # 5s, 10s...
            logger.warning("[%s] 第 %d 次尝试失败：%s；%ds 后重试",
                           what, attempt, last_err, backoff)
            time.sleep(backoff)
            _reconnect()
        else:
            _reconnect()  # 保证下一只标的拿到干净连接
    raise RuntimeError(f"{what} 连续 {MAX_TRIES} 次失败：{last_err}")


# ----------------------------------------------------------------------------
# 类型规范化（保证所有单标的文件 schema 一致，空文件也不例外）
# ----------------------------------------------------------------------------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0 or list(df.columns) != FIELDS:
        df = pd.DataFrame(columns=FIELDS)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce")
    out["code"] = df["code"].astype("string")
    for col in FLOAT_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    # 成交量单位为“股”，整数；历史缺失用 NULL
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    return out[FIELDS]


# ----------------------------------------------------------------------------
# 步骤 1：证券清单（含退市）
# ----------------------------------------------------------------------------
def fetch_universe() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if UNIVERSE_RAW.exists():
        logger.info("复用已有证券清单：%s", UNIVERSE_RAW)
        return pd.read_csv(UNIVERSE_RAW, dtype=str)

    _ensure_session()
    df = query_with_retry(bs.query_stock_basic, "query_stock_basic")
    df.to_csv(UNIVERSE_RAW, index=False, encoding="utf-8-sig")
    logger.info("证券清单已保存：%s（%d 条）", UNIVERSE_RAW, len(df))
    return df


def stock_codes(universe: pd.DataFrame) -> list[str]:
    """全量股票：type=1（股票），status 0=退市 / 1=在市 全部保留。"""
    m = (universe["type"] == "1") & (universe["status"].isin(["0", "1"]))
    return sorted(universe.loc[m, "code"].tolist())


# ----------------------------------------------------------------------------
# 步骤 2：逐标的下载（进程 worker）
# ----------------------------------------------------------------------------
_WORKER_READY = False


def _worker_init() -> None:
    global _WORKER_READY
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    _WORKER_READY = _login()


def _query_kline(code: str) -> pd.DataFrame:
    fields = ",".join(FIELDS)
    return query_with_retry(
        lambda: bs.query_history_k_data_plus(
            code, fields,
            start_date=START_DATE, end_date=END_DATE,
            frequency="d", adjustflag=ADJUSTFLAG),
        what=code,
    )


def download_one(code: str) -> tuple[str, str, int]:
    """返回 (code, status, rows)；status ∈ {skip, ok, empty, failed}。"""
    path = RAW_DIR / f"{code}.parquet"
    if path.exists():
        return code, "skip", 0

    global _WORKER_READY
    if not _WORKER_READY:
        _WORKER_READY = _login()

    try:
        raw = _query_kline(code)
        df = normalize(raw)
        # 成功但 0 行也写“空 schema”文件作为完成标记（例如区间内无交易），
        # 与网络失败区分：失败不写文件，下次重跑继续重试。
        # 先写临时文件再原子替换，避免进程被中途杀掉留下损坏的“已完成”文件。
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="snappy")
        os.replace(tmp, path)
        return code, ("ok" if len(df) else "empty"), len(df)
    except Exception as exc:
        logger.error("下载失败 %s：%s", code, exc)
        path.with_suffix(".parquet.tmp").unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        return code, "failed", 0


def run_download(codes: list[str], workers: int) -> dict[str, list[str]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in codes if not (RAW_DIR / f"{c}.parquet").exists()]
    done = len(codes) - len(todo)
    logger.info("待下载 %d / 共 %d（已完成 %d 断点跳过），workers=%d",
                len(todo), len(codes), done, workers)

    buckets: dict[str, list[str]] = {"ok": [], "empty": [], "failed": [], "skip": []}
    if not todo:
        buckets["skip"] = codes
        return buckets

    if workers <= 1:
        _worker_init()
        it = (download_one(c) for c in todo)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
        futures = {pool.submit(download_one, c): c for c in todo}

        def it():  # noqa: ANN202
            try:
                for fut in as_completed(futures):
                    yield fut.result()
            finally:
                pool.shutdown(wait=True)
        it = it()

    with tqdm(total=len(todo), desc="download", ncols=88, unit="code") as bar:
        for code, status, _n in it:
            buckets[status].append(code)
            bar.update(1)
            bar.set_postfix(ok=len(buckets["ok"]),
                            empty=len(buckets["empty"]),
                            fail=len(buckets["failed"]))

    if buckets["failed"]:
        with open(FAILED_LOG, "w", encoding="utf-8") as fh:
            fh.write("code\n" + "\n".join(buckets["failed"]) + "\n")
        logger.warning("%d 只标的失败（已记录 %s），重跑本脚本会自动重试；"
                       "合并步骤仍继续执行。", len(buckets["failed"]), FAILED_LOG)
    else:
        FAILED_LOG.unlink(missing_ok=True)
    return buckets


# ----------------------------------------------------------------------------
# 步骤 3：DuckDB 合并压缩
# ----------------------------------------------------------------------------
def run_merge(universe: pd.DataFrame, index_codes: list[str]) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(str(p) for p in RAW_DIR.glob("*.parquet")
                       if not p.name.startswith("_"))
    if not raw_files:
        raise RuntimeError(f"{RAW_DIR} 下没有 parquet 文件，无法合并")

    con = duckdb.connect()
    con.execute("SET threads TO ?", [os.cpu_count() or 4])

    con.execute(
        """
        COPY (
            SELECT * FROM read_parquet($files)
            WHERE code NOT IN (SELECT unnest($idx))
            ORDER BY code, date
        ) TO $out (FORMAT PARQUET, COMPRESSION 'ZSTD')
        """,
        {"files": raw_files, "idx": index_codes, "out": str(STOCKS_OUT)})

    con.execute(
        """
        COPY (
            SELECT * FROM read_parquet($files)
            WHERE code IN (SELECT unnest($idx))
            ORDER BY code, date
        ) TO $out (FORMAT PARQUET, COMPRESSION 'ZSTD')
        """,
        {"files": raw_files, "idx": index_codes, "out": str(INDICES_OUT)})

    # 清单复制一份随数据集留存（CSV 入库，供回测核对在市/退市、上市/退市日期）
    universe.to_csv(UNIVERSE_PUB, index=False, encoding="utf-8-sig")

    manifest = _build_manifest(con, universe, index_codes)
    MANIFEST_OUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    con.close()
    return manifest


def _build_manifest(con: duckdb.DuckDBPyConnection,
                    universe: pd.DataFrame, index_codes: list[str]) -> dict:
    s = con.execute(
        f"""SELECT COUNT(*) AS rows, COUNT(DISTINCT code) AS codes,
                   MIN(date) AS start, MAX(date) AS end
            FROM read_parquet('{STOCKS_OUT}')""").fetchone()
    i = con.execute(
        f"""SELECT COUNT(*) AS rows, COUNT(DISTINCT code) AS codes,
                   MIN(date) AS start, MAX(date) AS end
            FROM read_parquet('{INDICES_OUT}')""").fetchone()

    per_index = {}
    for code, name in INDICES.items():
        r = con.execute(
            f"""SELECT COUNT(*), MIN(date), MAX(date)
                FROM read_parquet('{INDICES_OUT}') WHERE code = ?""",
            [code]).fetchone()
        per_index[code] = {"name": name, "rows": r[0],
                           "start": r[1], "end": r[2]}

    n_listed = int(((universe["type"] == "1") & (universe["status"] == "1")).sum())
    n_delisted = int(((universe["type"] == "1") & (universe["status"] == "0")).sum())

    return {
        "generated_at": date.today().isoformat(),
        "source": "baostock query_history_k_data_plus (frequency=d, adjustflag=2 前复权)",
        "requested_range": {"start": START_DATE, "end": END_DATE},
        "files": {
            "daily_stocks.parquet": {"rows": s[0], "codes": s[1],
                                     "start": s[2], "end": s[3],
                                     "size_bytes": STOCKS_OUT.stat().st_size},
            "daily_indices.parquet": {"rows": i[0], "codes": i[1],
                                      "start": i[2], "end": i[3],
                                      "size_bytes": INDICES_OUT.stat().st_size},
        },
        "universe": {"stocks_listed": n_listed, "stocks_delisted": n_delisted},
        "indices": per_index,
        "indices_unavailable_in_baostock": INDICES_UNAVAILABLE,
        "fields": {
            "date": "交易日(TIMESTAMP)", "code": "baostock 证券代码",
            "open/high/low/close": "前复权价(float)",
            "volume": "成交量(股, BIGINT)", "amount": "成交额(元, float)",
            "turn": "换手率(%, float)", "pctChg": "涨跌幅(%, float)",
        },
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="A 股日线（含退市股 + 基准指数）抓取")
    ap.add_argument("--workers", type=int, default=1,
                    help="下载并发进程数（每个进程独立登录 baostock，默认 1）")
    ap.add_argument("--limit", type=int, default=0, help="只下载前 N 只股票（试跑用）")
    ap.add_argument("--skip-download", action="store_true", help="跳过下载，仅合并")
    ap.add_argument("--skip-merge", action="store_true", help="只下载，不合并")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = fetch_universe()
    codes = stock_codes(universe)
    if args.limit:
        codes = codes[:args.limit]
    targets = codes + list(INDICES)
    logger.info("区间 %s ~ %s，前复权；股票 %d 只 + 指数 %d 只",
                START_DATE, END_DATE, len(codes), len(INDICES))
    for c, n in INDICES_UNAVAILABLE.items():
        logger.warning("指数 %s(%s) 在 baostock 无数据，已跳过", c, n)

    rc = 0
    if not args.skip_download:
        buckets = run_download(targets, args.workers)
        if buckets["failed"]:
            rc = 2

    if not args.skip_merge:
        manifest = run_merge(universe, list(INDICES))
        logger.info("合并完成：%s（%s 行 / %s 只）、%s（%s 行 / %s 只）",
                    STOCKS_OUT.relative_to(ROOT),
                    f"{manifest['files']['daily_stocks.parquet']['rows']:,}",
                    manifest["files"]["daily_stocks.parquet"]["codes"],
                    INDICES_OUT.relative_to(ROOT),
                    f"{manifest['files']['daily_indices.parquet']['rows']:,}",
                    manifest["files"]["daily_indices.parquet"]["codes"])
        logger.info("清单与元数据：%s、%s",
                    UNIVERSE_PUB.relative_to(ROOT), MANIFEST_OUT.relative_to(ROOT))
    return rc


if __name__ == "__main__":
    sys.exit(main())
