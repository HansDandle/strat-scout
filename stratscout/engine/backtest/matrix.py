"""
Pre-computed indicator matrices for the ETF backtest.

precompute_backtest_matrices() runs once per backtest call (after histories are
sliced). It turns every per-day .loc / rolling computation into a numpy array so
the simulation loop replaces O(n) pandas work with O(1) array indexing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestMatrices:
    dates: list               # list[Timestamp] — common trading dates
    n: int                    # len(dates)

    # Regime detection — indexed by position in dates (i = yesterday in loop)
    agg_risk_on: np.ndarray   # bool (n,): AGG cumret > BIL cumret over agg_bil_lookback
    tlt_rising: np.ndarray    # bool (n,): TLT cumret < BIL cumret over tlt_bil_lookback
    tlt_surge: np.ndarray     # bool (n,): TLT fast momentum > 0 (flight-to-quality signal)

    # RSI — {(sym, window): float64 ndarray shape (n,)}
    rsi: dict

    # Per-symbol position sizing (combo weights)
    momentum: dict            # {sym: float64 (n,)} raw return over combo_momentum_lookback
    inv_vol: dict             # {sym: float64 (n,)} 1/rolling_std over combo_vol_lookback
    ema_cross: dict           # {sym: float64 (n,)} sigmoid EMA-cross score [0,1]
    vol_surge: dict           # {sym: float64 (n,)} current/avg_vol capped

    # SPY realized vol for adaptive vol targeting
    spy_realized_vol: np.ndarray  # float64 (n,) annualized % — 21-day window
    spy_median_vol: np.ndarray    # float64 (n,) rolling 126-period median of 21-day vol

    # Price matrix for fast portfolio valuation: shape (n, n_syms)
    price_matrix: np.ndarray
    symbols: list             # column order for price_matrix


def _rolling_rsi_sma(close_arr: np.ndarray, window: int) -> np.ndarray:
    """RSI using simple rolling mean — matches stratscout v1 exactly."""
    s = pd.Series(close_arr)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta).clip(lower=0).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(50.0).values.astype(np.float64)


def precompute_backtest_matrices(
    histories: dict[str, pd.DataFrame],
    params: dict[str, Any],
    common_dates: list,
) -> BacktestMatrices:
    """
    Pre-compute all indicator arrays over common_dates.

    common_dates: the full list of common trading dates used by the simulation loop
    (including warmup). Matrix index i corresponds to common_dates[i].
    """
    n = len(common_dates)
    date_index = pd.DatetimeIndex(common_dates)

    def _align_close(sym: str) -> np.ndarray:
        if sym not in histories:
            return np.full(n, np.nan)
        return (
            histories[sym]["close"]
            .reindex(date_index, method="ffill")
            .values.astype(np.float64)
        )

    def _align_volume(sym: str) -> np.ndarray | None:
        if sym not in histories or "volume" not in histories[sym].columns:
            return None
        return (
            histories[sym]["volume"]
            .reindex(date_index, method="ffill")
            .values.astype(np.float64)
        )

    # ── Regime detection ──────────────────────────────────────────────────────
    abl = params["agg_bil_lookback"]
    tbl = params["tlt_bil_lookback"]

    agg_arr = _align_close("AGG")
    bil_arr = _align_close("BIL")
    tlt_arr = _align_close("TLT")

    with np.errstate(invalid="ignore", divide="ignore"):
        # AGG vs BIL over agg_bil_lookback
        agg_cumret = np.zeros(n)
        bil_cumret_agg = np.zeros(n)
        if n > abl:
            denom_agg = agg_arr[:n - abl]
            denom_bil_a = bil_arr[:n - abl]
            agg_cumret[abl:] = np.where(denom_agg > 0, agg_arr[abl:] / denom_agg - 1, 0.0)
            bil_cumret_agg[abl:] = np.where(denom_bil_a > 0, bil_arr[abl:] / denom_bil_a - 1, 0.0)

        # TLT vs BIL over tlt_bil_lookback
        tlt_cumret = np.zeros(n)
        bil_cumret_tlt = np.zeros(n)
        if n > tbl:
            denom_tlt = tlt_arr[:n - tbl]
            denom_bil_t = bil_arr[:n - tbl]
            tlt_cumret[tbl:] = np.where(denom_tlt > 0, tlt_arr[tbl:] / denom_tlt - 1, 0.0)
            bil_cumret_tlt[tbl:] = np.where(denom_bil_t > 0, bil_arr[tbl:] / denom_bil_t - 1, 0.0)

    agg_risk_on = agg_cumret > bil_cumret_agg
    tlt_rising = tlt_cumret < bil_cumret_tlt

    # Fast TLT momentum (flight-to-quality surge detector)
    tfl = params.get("tlt_fast_lookback", 3)
    tlt_fast_cumret = np.zeros(n)
    if n > tfl:
        denom_tlt_f = tlt_arr[:n - tfl]
        tlt_fast_cumret[tfl:] = np.where(denom_tlt_f > 0, tlt_arr[tfl:] / denom_tlt_f - 1, 0.0)
    tlt_surge = tlt_fast_cumret > 0

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_matrices: dict = {}
    for window, syms in [
        (params["risk_on_rsi_window"], params["risk_on_pool"]),
        (params["risk_off_rsi_window"], params["risk_off_rising_pool"]),
    ]:
        for sym in syms:
            if sym in histories and (sym, window) not in rsi_matrices:
                close_arr = _align_close(sym)
                rsi_matrices[(sym, window)] = _rolling_rsi_sma(close_arr, window)

    # ── Position sizing indicators ────────────────────────────────────────────
    mom_lb = params.get("combo_momentum_lookback", 21)
    vol_lb = params.get("combo_vol_lookback", 21)
    need_ema   = params.get("ema_weight", 0.0) > 0
    need_surge = params.get("vol_score_weight", 0.0) > 0
    ema_fast_p = params.get("ema_fast", 10)
    ema_slow_p = params.get("ema_slow", 40)
    vol_surge_window = params.get("vol_score_window", 20)
    vol_surge_cap    = params.get("vol_surge_cap", 3.0)

    # Only pool symbols that actually exist in histories
    all_pool_syms: set[str] = set()
    for key in ("risk_on_pool", "risk_off_rising_pool", "risk_off_falling_pool", "low_vol_pool"):
        pool = params.get(key) or []
        all_pool_syms.update(pool)
    all_pool_syms = all_pool_syms & set(histories.keys())

    # Pre-align close arrays once per symbol (avoids re-alignment inside the loop)
    close_cache: dict[str, np.ndarray] = {}
    for sym in all_pool_syms:
        close_cache[sym] = _align_close(sym)

    momentum_mat: dict = {}
    inv_vol_mat: dict = {}
    ema_cross_mat: dict = {}
    vol_surge_mat: dict = {}

    for sym, close_arr in close_cache.items():
        close_s = pd.Series(close_arr)

        # Momentum: total return over mom_lb (vectorized shift)
        mom = np.zeros(n)
        if n > mom_lb:
            denom = close_arr[:n - mom_lb]
            mom[mom_lb:] = np.where(denom > 0, close_arr[mom_lb:] / denom - 1, 0.0)
        momentum_mat[sym] = mom

        # Inverse volatility
        rets = close_s.pct_change(fill_method=None)
        roll_std = rets.rolling(vol_lb, min_periods=vol_lb).std().values
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_vol_mat[sym] = np.where(roll_std > 1e-8, 1.0 / roll_std, 1.0)

        # EMA cross — only when ema_weight > 0 (most runs skip this)
        if need_ema:
            fast_ema = close_s.ewm(span=ema_fast_p, adjust=False).mean().values
            slow_ema = close_s.ewm(span=ema_slow_p, adjust=False).mean().values
            with np.errstate(invalid="ignore", divide="ignore"):
                raw_cross = np.where(slow_ema > 0, (fast_ema - slow_ema) / slow_ema, 0.0)
            ema_cross_mat[sym] = 1.0 / (1.0 + np.exp(-50.0 * np.nan_to_num(raw_cross)))

        # Volume surge — only when vol_score_weight > 0 (most runs skip this)
        if need_surge:
            vol_arr = _align_volume(sym)
            if vol_arr is not None:
                vol_s = pd.Series(vol_arr)
                rolling_avg = vol_s.shift(1).rolling(vol_surge_window, min_periods=vol_surge_window).mean().values
                with np.errstate(invalid="ignore", divide="ignore"):
                    surge = np.where(rolling_avg > 1, vol_arr / rolling_avg, 1.0)
                vol_surge_mat[sym] = np.clip(np.nan_to_num(surge, nan=1.0), 0, vol_surge_cap)
            else:
                vol_surge_mat[sym] = np.ones(n)

    # ── SPY vol — only computed when SPY is actually in histories ─────────────
    # SPY is present only when it's in params pools. Most runs skip this block.
    spy_realized_vol = np.zeros(n)
    spy_median_vol   = np.zeros(n)
    if "SPY" in histories:
        spy_s = pd.Series(_align_close("SPY"))
        spy_rets = spy_s.pct_change()
        spy_21_std = spy_rets.rolling(21, min_periods=21).std()
        spy_realized_vol = (spy_21_std * math.sqrt(252) * 100).fillna(0.0).values.astype(np.float64)
        spy_median_vol = (
            spy_21_std.rolling(126, min_periods=30).median() * math.sqrt(252) * 100
        ).fillna(0.0).values.astype(np.float64)

    return BacktestMatrices(
        dates=common_dates,
        n=n,
        agg_risk_on=agg_risk_on,
        tlt_rising=tlt_rising,
        tlt_surge=tlt_surge,
        rsi=rsi_matrices,
        momentum=momentum_mat,
        inv_vol=inv_vol_mat,
        ema_cross=ema_cross_mat,
        vol_surge=vol_surge_mat,
        spy_realized_vol=spy_realized_vol,
        spy_median_vol=spy_median_vol,
        price_matrix=np.zeros((0, 0)),  # unused — skipped to save allocation
        symbols=[],
    )
