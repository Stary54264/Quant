#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2025 美股日线 raw 数据抓取（Yahoo Finance chart API，按 ticker）。

目的：在取得权威 2025 CRSP 抽取前，提供一份覆盖"现有在市股 + 2025 IPO"的
2025 日线 raw 数据；退市 ticker 的退市前历史 Yahoo 通常保留。

输入：ticker 清单文件（默认 /tmp/ticker_master.txt，每行一个 ticker）
输出：data/backtest/yahoo2025/<TICKER>.csv（date,open,high,low,close,volume）
      data/backtest/yahoo2025/_status.csv（每个 ticker 的抓取结果）

特性：多线程抓取（每线程独立 requests.Session），429/5xx 指数退避重试，
query1/query2 主机轮换；已存在的 ticker 文件跳过（断点续抓）。
"""

import argparse
import csv
import datetime
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "backtest" / "yahoo2025"

P1 = int(datetime.datetime(2024, 12, 20).timestamp())
P2 = int(datetime.datetime(2026, 1, 6).timestamp())
HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128.0"
MAX_TRIES = 25
NY = ZoneInfo("America/New_York")

# 自适应限速：任意两次请求至少间隔 SPACING 秒；收到 429 后全体暂停 PAUSE 秒
SPACING = 2.2
PAUSE = 180.0
_lock = threading.Lock()
_next_slot = 0.0
_pause_until = 0.0
_tls = threading.local()


def wait_slot() -> None:
    global _next_slot, _pause_until
    while True:
        with _lock:
            now = time.time()
            wait = max(_pause_until, _next_slot) - now
            if wait <= 0:
                _next_slot = now + SPACING
                return
        time.sleep(min(wait, 30) + 0.05)


def session() -> requests.Session:
    if not getattr(_tls, "s", None):
        _tls.s = requests.Session()
        _tls.s.headers.update({"User-Agent": UA, "Accept": "application/json"})
    return _tls.s


def fetch(ticker: str) -> tuple[str, str, int, str]:
    """返回 (ticker, status, rows, note)；status ∈ ok/skip/empty/error。"""
    path = OUT_DIR / f"{ticker}.csv"
    if path.exists():
        return ticker, "skip", 0, ""

    global _pause_until
    last_note = ""
    n_429 = 0
    attempt = 0
    while attempt < MAX_TRIES:
        host = HOSTS[attempt % 2]
        url = (f"https://{host}/v8/finance/chart/{ticker}"
               f"?period1={P1}&period2={P2}&interval=1d&events=history")
        try:
            wait_slot()
            r = session().get(url, timeout=30)
            if r.status_code in (429, 502, 503, 504):
                last_note = f"HTTP {r.status_code}"
                if r.status_code == 429:
                    n_429 += 1
                    with _lock:
                        _pause_until = time.time() + PAUSE
                    # 429 不消耗尝试轮数
                    continue
                time.sleep(min(2.0 ** attempt, 20))
                attempt += 1
                continue
            if r.status_code == 404:
                return ticker, "empty", 0, "404"
            if r.status_code != 200:
                last_note = f"HTTP {r.status_code}"
                time.sleep(2 + attempt)
                attempt += 1
                continue
            j = r.json()
            ch = j.get("chart") or {}
            if ch.get("error"):
                return ticker, "empty", 0, json.dumps(ch["error"])[:120]
            res = ch["result"][0]
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            seen, rows, adj_rows = set(), [], []
            for t, o, h, l, c, v, ac in zip(
                    res["timestamp"], q["open"], q["high"], q["low"],
                    q["close"], q["volume"],
                    adj if adj is not None else [None] * len(res["timestamp"])):
                d = datetime.datetime.fromtimestamp(t, NY).date()
                if d in seen:
                    continue
                seen.add(d)
                rows.append((d.isoformat(),
                             "" if o is None else round(o, 6),
                             "" if h is None else round(h, 6),
                             "" if l is None else round(l, 6),
                             "" if c is None else round(c, 6),
                             "" if v is None else int(v)))
                adj_rows.append((d.isoformat(),
                                 "" if ac is None else round(ac, 6)))
            rows.sort()
            adj_rows.sort()
            with path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                w.writerows(rows)
            with (RAW_DIR / f"{ticker}.adj.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["date", "adjclose"])
                w.writerows(adj_rows)
            if not rows:
                return ticker, "empty", 0, ""
            return ticker, "ok", len(rows), f"{rows[0][0]}..{rows[-1][0]}"
        except (requests.RequestException, ValueError, KeyError) as e:
            last_note = type(e).__name__ + ":" + str(e)[:80]
            time.sleep(1 + attempt * 2)
            attempt += 1
    note = f"{last_note}; 429x{n_429}" if n_429 else last_note
    return ticker, "error", 0, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="/tmp/ticker_master.txt")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tickers = Path(args.tickers).read_text().split()
    if args.limit:
        tickers = tickers[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    todo = [t for t in tickers if not (OUT_DIR / f"{t}.csv").exists()]
    print(f"待抓 {len(todo):,} / 总计 {len(tickers):,}，{args.workers} 线程", flush=True)
    stats = {"ok": 0, "skip": 0, "empty": 0, "error": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, t): t for t in todo}
        results = []
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            stats[res[1]] += 1
            if i % 200 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1)
                print(f"  {i:,}/{len(todo):,} | {rate:.1f} ticker/s | {stats}", flush=True)

    # 合并已有的 skip 状态，写 _status.csv
    rows_status = results
    if not todo:
        rows_status = [(t, "skip", 0, "") for t in tickers]
    with (OUT_DIR / "_status.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "status", "rows", "note"])
        w.writerows(rows_status)
    print("done:", stats, f"{(time.time()-t0)/60:.1f} min")
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
