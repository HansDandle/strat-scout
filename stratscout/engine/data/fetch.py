"""Daily bar downloader.

Tries data sources in order:
  1. Alpaca (if API key is set) — free for stocks/ETFs, paper-account keys work
  2. yfinance (no key, rate-limited but always available) — fallback

Writes results into the same feather format the rest of the engine reads from
(data/daily/{SYMBOL}.feather with columns ['date','open','high','low','close','volume']).

Used by the /data/download endpoint and the legacy download_data.py CLI.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from stratscout.engine import credentials
from stratscout.engine.settings import daily_dir

log = logging.getLogger(__name__)

DEFAULT_START = "2018-01-01"


@dataclass
class DownloadProgress:
    total: int
    done: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (symbol, reason)
    source_used: dict[str, str] = field(default_factory=dict)    # symbol → 'alpaca' | 'yfinance' | 'skip'
    log_lines: list[str] = field(default_factory=list)


# ── Source: Alpaca ────────────────────────────────────────────────────────────

def _fetch_alpaca(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Daily bars via Alpaca data API. Free tier works for stocks + ETFs."""
    api_key = credentials.get("alpaca", "api_key")
    api_secret = credentials.get("alpaca", "api_secret")
    if not api_key or not api_secret:
        return None
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": symbol,
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "split",
        "feed": "iex",   # free-tier feed
    }
    try:
        r = requests.get(
            url,
            params=params,
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        log.warning("alpaca network error for %s: %s", symbol, e)
        return None
    if r.status_code != 200:
        log.warning("alpaca %s returned %d: %s", symbol, r.status_code, r.text[:200])
        return None
    j = r.json()
    bars = (j.get("bars") or {}).get(symbol, [])
    if not bars:
        return None
    df = pd.DataFrame(bars)
    df = df.rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
    return df


# ── Source: yfinance (no key needed) ──────────────────────────────────────────

def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Free, rate-limited fallback. Slower than Alpaca and occasionally flaky."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        log.warning("yfinance error for %s: %s", symbol, e)
        return None
    if df is None or df.empty:
        return None
    # yfinance returns MultiIndex columns when multiple tickers; collapse for single
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # Normalize column names — yfinance gives 'Date', 'Open', etc.
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    if "date" not in df.columns:
        for cand in ("Date", "Datetime", "index"):
            if cand.lower() in [c.lower() for c in df.columns]:
                df = df.rename(columns={c: "date" for c in df.columns if str(c).lower() == cand.lower()})
                break
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True)
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    if len(keep) < 5:
        return None
    return df[keep].sort_values("date").reset_index(drop=True)


# ── Driver ────────────────────────────────────────────────────────────────────

def download_symbols(
    symbols: list[str],
    start: str = DEFAULT_START,
    end: str | None = None,
    overwrite: bool = False,
) -> DownloadProgress:
    """Download daily bars for a list of symbols, writing to data/daily/{SYM}.feather.

    Returns a progress report — what succeeded, what failed, which source was used.
    The "smart" behavior: tries Alpaca (if keys set) → falls back to yfinance.
    Symbols that already have data are skipped unless overwrite=True.
    """
    end = end or date.today().isoformat()
    out_dir = daily_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = DownloadProgress(total=len(symbols))
    have_alpaca = (
        credentials.get("alpaca", "api_key") is not None
        and credentials.get("alpaca", "api_secret") is not None
    )

    for sym in symbols:
        sym_u = sym.upper().strip()
        progress.log_lines.append(f"--- {sym_u} ---")
        out_path = out_dir / f"{sym_u}.feather"

        if out_path.exists() and not overwrite:
            try:
                existing = pd.read_feather(out_path, columns=["date"])
                existing["date"] = pd.to_datetime(existing["date"], utc=True)
                if existing["date"].max().date() >= date.today() - timedelta(days=2):
                    progress.source_used[sym_u] = "skip"
                    progress.log_lines.append(f"  already up to date ({len(existing)} rows)")
                    progress.done += 1
                    continue
            except (OSError, ValueError, KeyError):
                pass  # treat as needing redownload

        df: pd.DataFrame | None = None

        if have_alpaca:
            progress.log_lines.append("  trying Alpaca…")
            df = _fetch_alpaca(sym_u, start, end)
            if df is not None and len(df) > 0:
                progress.source_used[sym_u] = "alpaca"
                progress.log_lines.append(f"  alpaca: {len(df)} bars")

        if df is None:
            progress.log_lines.append("  trying yfinance…")
            df = _fetch_yfinance(sym_u, start, end)
            if df is not None and len(df) > 0:
                progress.source_used[sym_u] = "yfinance"
                progress.log_lines.append(f"  yfinance: {len(df)} bars")

        if df is None or df.empty:
            progress.failed.append((sym_u, "no data from any source"))
            progress.log_lines.append("  FAILED")
            progress.done += 1
            continue

        df.to_feather(out_path)
        progress.done += 1
        time.sleep(0.05)  # be polite to rate limits

    return progress
