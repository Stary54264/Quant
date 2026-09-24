#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2025 美股日线 raw 数据抓取（Yahoo Finance chart API，按 ticker）。

目的：在取得权威 2025 CRSP 抽取前，提供一份覆盖"现有在市股 + 2025 IPO"的
2025 日线 raw 数据；退市 ticker 的退市前历史 Yahoo 通常保留。

输入：ticker 清单文件（默认 /tmp/ticker_master.txt，每行一个 ticker）；
      可选 --priority 文件中的 ticker 排到最前（如 2025 IPO 清单）。
输出：data/backtest/yahoo2025/<TICKER>.csv（date,open,high,low,close,volume）
      data/backtest/yahoo2025/<TICKER>.adj.csv（date,adjclose）
      data/backtest/yahoo2025/_results.jsonl（增量结果，每行一个 ticker）

限速：自适应 15 分钟滑动窗口；窗口内请求数达到 BURST 即排队等最旧请求出窗。
Yahoo 按 HTTP/TLS 指纹区分通道：HTTP/1.1（requests/普通 curl）几乎一律 429，
**HTTP/2 + Chrome 指纹**（curl_cffi impersonate）按浏览器配额放行，故必须经
curl_cffi 访问。收到 429（或慢速 drip 硬超时）后**完全静默**（继续发请求只会
刷新封禁计时），暂停时长 300s 起逐次递增；连续封禁超 MAX_429 轮则熔断退出。
已存在的 ticker 文件跳过（断点续抓，重跑不重复计数）。
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

from curl_cffi import requests
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "backtest" / "yahoo2025"
RAW_DIR = OUT_DIR          # adjclose 明细与主文件同目录

P1 = int(datetime.datetime(2024, 12, 20).timestamp())
P2 = int(datetime.datetime(2026, 1, 6).timestamp())
HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128.0"
MAX_TRIES = 25
NY = ZoneInfo("America/New_York")


class HardTimeout(Exception):
    """http_get 超过硬墙钟截止（慢速 drip / 连接挂死）。"""

# 自适应限速：15 分钟滑动窗口，最多 BURST 次请求；相邻请求至少 SPACING 秒；
# 429 后全体静默
WINDOW = 900.0
BURST = 180
SPACING = 1.0
PAUSE0 = 300.0
PAUSE_MAX = 900.0
MAX_429 = 4               # 封禁轮数上限（每轮静默 5~15 分钟），超过即熔断
_lock = threading.Lock()
_stamps: list[float] = []  # 近期请求时刻（滑动窗口）
_pause_until = 0.0
_tls = threading.local()


def wait_slot() -> None:
    """全局节奏：静默期全员等待；窗口满则等最旧请求出窗；相邻请求拉开间隔。"""
    while True:
        with _lock:
            now = time.time()
            if _pause_until > now:
                wait = _pause_until - now
            else:
                old = [t for t in _stamps if now - t < WINDOW]
                _stamps[:] = old
                if len(old) < BURST and (not old or now - old[-1] >= SPACING):
                    _stamps.append(now)
                    return
                if len(old) >= BURST:
                    wait = old[0] + WINDOW - now
                else:
                    wait = SPACING - (now - old[-1])
        time.sleep(min(max(wait, 0.1), 60) + 0.05)


def note_pause(seconds: float) -> None:
    global _pause_until
    with _lock:
        _pause_until = max(_pause_until, time.time() + seconds)


def http_get(url: str, deadline: float = 45.0) -> requests.Response:
    """带**硬墙钟截止**的 GET。封禁期 Yahoo 会接受连接后慢速 drip 甚至完全
    不发字节，requests 的 timeout 只管相邻字节间隔，可能无限挂死；用独立
    线程执行、join 超时即抛 Timeout（悬挂的守护线程随后由 GC/进程退出回收）。"""
    box: dict = {}

    def work() -> None:
        try:
            box["r"] = session().get(url, timeout=(10, 15))
        except Exception as e:  # 交回主线程按原逻辑分类
            box["e"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(deadline)
    if "r" in box:
        return box["r"]
    if "e" in box:
        raise box["e"]
    raise HardTimeout(f"hard deadline {deadline:.0f}s")


def session() -> requests.Session:
    if not getattr(_tls, "s", None):
        # impersonate 同时给出 h2 与 Chrome 的 TLS/JA3 指纹——Yahoo 的 h1 通道
        # 对未认证请求近乎全量 429，h2+Chrome 指纹才按浏览器配额放行
        _tls.s = requests.Session(impersonate="chrome")
        _tls.s.headers.update({"Accept": "application/json"})
    return _tls.s


def fetch(ticker: str) -> tuple[str, str, int, str]:
    """返回 (ticker, status, rows, note)；status ∈ ok/skip/empty/error。"""
    path = OUT_DIR / f"{ticker}.csv"
    if path.exists():
        return ticker, "skip", 0, ""

    last_note = ""
    n_429 = 0
    pause = PAUSE0
    attempt = 0
    while attempt < MAX_TRIES:
        host = HOSTS[attempt % 2]
        url = (f"https://{host}/v8/finance/chart/{ticker}"
               f"?period1={P1}&period2={P2}&interval=1d&events=history")
        try:
            wait_slot()
            try:
                r = http_get(url)
            except HardTimeout:
                # 慢速 drip 与 429 同属封禁信号：完全静默
                n_429 += 1
                last_note = f"drip-timeout x{n_429}"
                if n_429 > MAX_429:
                    return ticker, "error", 0, last_note
                note_pause(pause)
                pause = min(pause * 1.5, PAUSE_MAX)
                continue
            if r.status_code == 429:
                n_429 += 1
                last_note = f"HTTP 429 x{n_429}"
                if n_429 > MAX_429:
                    return ticker, "error", 0, last_note
                note_pause(pause)        # 完全静默，不刷新封禁计时
                pause = min(pause * 1.5, PAUSE_MAX)
                continue                 # 静默不消耗硬错误轮数
            if r.status_code in (502, 503, 504):
                last_note = f"HTTP {r.status_code}"
                time.sleep(min(2.0 ** attempt, 20))
                attempt += 1
                continue
            if r.status_code in (400, 404):
                # 证券不存在 / 区间内无数据（chart.error 会写明），不可重试
                note = f"HTTP {r.status_code}"
                try:
                    ce = (r.json().get("chart") or {}).get("error")
                    if ce:
                        note = ce.get("description", note)[:100]
                except ValueError:
                    pass
                return ticker, "empty", 0, note
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
        except (requests.RequestsError, ValueError, KeyError) as e:
            last_note = type(e).__name__ + ":" + str(e)[:80]
            time.sleep(1 + attempt * 2)
            attempt += 1
    note = f"{last_note}; 429x{n_429}" if n_429 else last_note
    return ticker, "error", 0, note


def load_tickers(path: str) -> list[str]:
    p = Path(path)
    return p.read_text().split() if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="/tmp/ticker_master.txt")
    ap.add_argument("--priority", default="/tmp/ticker_ipo2025.txt",
                    help="优先抓取的 ticker 清单（默认 2025 IPO 清单）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tickers = load_tickers(args.tickers)
    pri = [t for t in load_tickers(args.priority) if t in tickers]
    pri_set = set(pri)
    ordered = pri + [t for t in tickers if t not in pri_set]
    if args.limit:
        ordered = ordered[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    todo = [t for t in ordered if not (OUT_DIR / f"{t}.csv").exists()]
    print(f"待抓 {len(todo):,} / 总计 {len(ordered):,}（优先 {len(pri):,}），"
          f"{args.workers} 线程，窗口 {int(WINDOW)}s/{BURST}", flush=True)
    stats = {"ok": 0, "skip": 0, "empty": 0, "error": 0}
    results_path = OUT_DIR / "_results.jsonl"
    t0 = time.time()
    rc = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            with results_path.open("a") as fh:
                fh.write(json.dumps({"ticker": res[0], "status": res[1],
                                     "rows": res[2], "note": res[3]}) + "\n")
            stats[res[1]] += 1
            if res[1] == "error":
                rc = 1
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1)
                print(f"  {i:,}/{len(todo):,} | {rate*60:.0f}/h | {stats}",
                      flush=True)

    print("done:", stats, f"{(time.time()-t0)/60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
