"""
v2 ETF backtest engine — matrix-accelerated variant.

Identical to stratscout.engine.backtest.etf except:
  - precompute_backtest_matrices() is called once per run after histories are sliced
  - choose_targets / compute_combo_weights accept optional matrices + date_idx
  - When matrices provided: O(1) array indexing replaces per-day pandas work
  - matrices=None falls back to original behaviour exactly (backward compat)

stratscout/ is never touched — this file is the only copy that changes.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal
from datetime import datetime
from typing import Any

import pandas as pd
from pathlib import Path

# Re-export stable pieces directly from v1 (no duplication)
from stratscout.engine.backtest.core import (
    value_of_portfolio,
    rebalance_positions,
    compute_performance,
    BacktestError,
)
from stratscout.engine.settings import daily_dir as _daily_dir
from stratscout.engine.data.universes import (
    ANCHORS,
    RISK_ON_POOL,
    RISK_OFF_RISING_POOL,
    RISK_OFF_FALLING_POOL,
    ALL_SYMBOLS,
    MIN_RISK_ON,
    MIN_RISK_OFF_RISING,
    MIN_RISK_OFF_FALLING,
    SECTOR_BUCKETS,
)

from stratscout_v2.engine.backtest.matrix import BacktestMatrices, precompute_backtest_matrices

DAILY_DIR = _daily_dir()

# ETFs with 3x daily leverage — used for the leverage cap param
LEVERAGED_3X: frozenset[str] = frozenset([
    "SOXL", "TQQQ", "UPRO", "TECL", "SPXL", "FAS", "CURE", "LABU",
    "ERX", "DRN", "FNGU", "UTSL", "MIDU", "TNA", "URTY",
    "JNUG", "GDXU", "TMF", "NUGT", "UGL",
    "FAZ", "TZA", "SPXS", "SQQQ",
])

# ── Re-export helpers unchanged from v1 ──────────────────────────────────────
from stratscout.engine.backtest.etf import (
    load_local_histories,
    DEFAULT_PARAMS,
    random_params,
    refine_params,
    compute_vol_weights,
    _vol_surge_score,
    _rolling_rsi,
    _ema_cross_score,
    _cumret,
    precompute_rsi_cache,
    _sharpe_filter,
    _buy_and_hold_nav,
)


# ── Matrix-aware RSI helper ───────────────────────────────────────────────────

def _rsi_at_matrix(sym: str, window: int, date_idx: int,
                   bm: BacktestMatrices, histories: dict, as_of) -> float:
    """Return RSI for sym/window at date_idx, using matrix when available."""
    if bm is not None and (sym, window) in bm.rsi:
        return float(bm.rsi[(sym, window)][date_idx])
    # Fallback: original pandas path
    s = _rolling_rsi(histories[sym]["close"], window)
    return float(s.loc[:as_of].iloc[-1])


# ── Matrix-aware combo weights ────────────────────────────────────────────────

def compute_combo_weights(
    histories: dict,
    symbols: list[str],
    as_of,
    momentum_lookback: int = 21,
    vol_lookback: int = 21,
    alpha: float = 0.5,
    max_weight: float = 1.0,
    score_normalize_window: int = 0,
    lev_3x_cap: float = 1.0,
    matrices: BacktestMatrices | None = None,
    date_idx: int | None = None,
) -> dict[str, float]:
    """
    Weight each symbol by alpha*momentum + (1-alpha)*inv_vol, normalized to 1.

    When matrices and date_idx are provided, reads precomputed arrays instead of
    recomputing rolling windows from scratch on every call.
    """
    raw_mom: dict = {}
    raw_invvol: dict = {}

    if matrices is not None and date_idx is not None:
        for sym in symbols:
            if sym in matrices.momentum:
                raw_mom[sym] = float(matrices.momentum[sym][date_idx])
            else:
                raw_mom[sym] = 0.0
            if sym in matrices.inv_vol:
                raw_invvol[sym] = float(matrices.inv_vol[sym][date_idx])
            else:
                raw_invvol[sym] = 1.0
    else:
        # Original pandas path (fallback / when score_normalize_window > 0)
        for sym in symbols:
            if sym not in histories:
                raw_mom[sym] = 0.0
                raw_invvol[sym] = 1.0
                continue
            close = histories[sym]["close"].loc[:as_of]
            if len(close) >= momentum_lookback + 1:
                raw_mom[sym] = float(close.iloc[-1] / close.iloc[-momentum_lookback - 1] - 1)
            else:
                raw_mom[sym] = 0.0
            if len(close) >= vol_lookback + 2:
                vol = close.iloc[-(vol_lookback + 1):].pct_change().dropna().std()
                raw_invvol[sym] = 1.0 / vol if vol > 1e-8 else 1.0
            else:
                raw_invvol[sym] = 1.0

    # Z-score normalization (unchanged from v1 — uses pandas path)
    if score_normalize_window > 0:
        import statistics
        normalized_mom = {}
        for sym in symbols:
            if sym not in histories:
                normalized_mom[sym] = raw_mom[sym]
                continue
            close = histories[sym]["close"].loc[:as_of]
            step = momentum_lookback
            past_readings = []
            for i in range(score_normalize_window):
                end_idx = len(close) - 1 - i * step
                start_idx = end_idx - momentum_lookback
                if start_idx < 0:
                    break
                past_readings.append(float(close.iloc[end_idx] / close.iloc[start_idx] - 1))
            if len(past_readings) >= 3:
                mu = statistics.mean(past_readings)
                sigma = statistics.stdev(past_readings)
                normalized_mom[sym] = (raw_mom[sym] - mu) / sigma if sigma > 1e-8 else 0.0
            else:
                normalized_mom[sym] = raw_mom[sym]
        raw_mom = normalized_mom

    def _minmax_norm(d: dict) -> dict:
        lo, hi = min(d.values()), max(d.values())
        rng = hi - lo
        if rng < 1e-10:
            return {k: 0.5 for k in d}
        return {k: (v - lo) / rng for k, v in d.items()}

    mom_n    = _minmax_norm(raw_mom)
    invvol_n = _minmax_norm(raw_invvol)
    combo = {sym: alpha * mom_n[sym] + (1 - alpha) * invvol_n[sym] for sym in symbols}
    combo = {sym: max(v, 1e-6) for sym, v in combo.items()}
    total = sum(combo.values())
    weights = {sym: v / total for sym, v in combo.items()}

    if max_weight < 1.0:
        for _ in range(20):
            excess = sum(max(0, w - max_weight) for w in weights.values())
            if excess < 1e-8:
                break
            under = {sym: w for sym, w in weights.items() if w < max_weight}
            under_total = sum(under.values())
            capped = {sym: min(w, max_weight) for sym, w in weights.items()}
            if under_total > 0:
                for sym in under:
                    capped[sym] += excess * (under[sym] / under_total)
            total = sum(capped.values())
            weights = {sym: v / total for sym, v in capped.items()}

    # Cap total allocation to 3x leveraged ETFs
    if lev_3x_cap < 1.0:
        lev_syms   = [s for s in symbols if s in LEVERAGED_3X]
        unlev_syms = [s for s in symbols if s not in LEVERAGED_3X]
        lev_total  = sum(weights.get(s, 0.0) for s in lev_syms)
        if lev_total > lev_3x_cap and lev_total > 0:
            scale = lev_3x_cap / lev_total
            overflow = lev_total - lev_3x_cap
            for s in lev_syms:
                weights[s] = weights.get(s, 0.0) * scale
            # redistribute overflow to unleveraged symbols proportionally
            if unlev_syms:
                unlev_total = sum(weights.get(s, 0.0) for s in unlev_syms)
                if unlev_total > 1e-8:
                    for s in unlev_syms:
                        weights[s] = weights.get(s, 0.0) + overflow * (weights[s] / unlev_total)
            # renormalize
            total = sum(weights.values())
            if total > 1e-8:
                weights = {s: v / total for s, v in weights.items()}

    return weights


# ── Matrix-aware choose_targets ───────────────────────────────────────────────

def choose_targets(
    histories: dict[str, pd.DataFrame],
    as_of,
    params: dict,
    rsi_cache: dict | None = None,
    matrices: BacktestMatrices | None = None,
    date_idx: int | None = None,
) -> list[str]:
    use_matrix = matrices is not None and date_idx is not None

    # ── Regime detection ──────────────────────────────────────────────────────
    if use_matrix:
        risk_on = bool(matrices.agg_risk_on[date_idx])
        rising_rates = bool(matrices.tlt_rising[date_idx])
        tlt_surging = bool(matrices.tlt_surge[date_idx])
    else:
        agg = histories["AGG"]["close"]
        bil = histories["BIL"]["close"]
        tlt = histories["TLT"]["close"]
        risk_on = _cumret(agg, params["agg_bil_lookback"], as_of) > _cumret(bil, params["agg_bil_lookback"], as_of)
        rising_rates = _cumret(tlt, params["tlt_bil_lookback"], as_of) < _cumret(bil, params["tlt_bil_lookback"], as_of)
        tfl = params.get("tlt_fast_lookback", 3)
        tlt_surging = _cumret(tlt, tfl, as_of) > 0

    vsw   = params.get("vol_score_weight", 0.0)
    vswin = params.get("vol_score_window", 20)
    vscap = params.get("vol_surge_cap", 3.0)
    emaw  = params.get("ema_weight", 0.0)
    ema_fast_w = params.get("ema_fast", 10)
    ema_slow_w = params.get("ema_slow", 40)

    def _rsi_at(sym: str, window: int) -> float:
        if use_matrix and (sym, window) in matrices.rsi:
            return float(matrices.rsi[(sym, window)][date_idx])
        if rsi_cache is not None and (sym, window) in rsi_cache:
            s = rsi_cache[(sym, window)]
        else:
            s = _rolling_rsi(histories[sym]["close"], window)
        return float(s.loc[:as_of].iloc[-1])

    def _vsurge(sym: str) -> float:
        if use_matrix and sym in matrices.vol_surge:
            return float(matrices.vol_surge[sym][date_idx])
        return _vol_surge_score(histories, sym, as_of, vswin, vscap)

    def _emacross(sym: str) -> float:
        if use_matrix and sym in matrices.ema_cross:
            return float(matrices.ema_cross[sym][date_idx])
        if sym in histories:
            return _ema_cross_score(histories[sym]["close"], as_of, ema_fast_w, ema_slow_w)
        return 0.5

    def _blend_score(sym: str, rsi_val: float, direction: str) -> float:
        rsi_norm = rsi_val / 100.0
        if direction == "lowest":
            rsi_norm = 1.0 - rsi_norm
        surge = _vsurge(sym)
        vol_norm = 1.0 / surge
        ema_norm = _emacross(sym) if emaw > 0 else 0.5
        rsi_w = max(0.0, 1.0 - vsw - emaw)
        signal = rsi_w * rsi_norm + vsw * (1.0 - vol_norm) + emaw * ema_norm
        return 1.0 - signal

    def _sector_filter(ranked_pairs: list, n: int) -> list[str]:
        if not params.get("sector_diverse", False):
            return [sym for sym, _ in ranked_pairs[:n]]
        seen: set[str] = set()
        picks: list[str] = []
        for sym, _ in ranked_pairs:
            bucket = SECTOR_BUCKETS.get(sym, sym)
            if bucket not in seen:
                picks.append(sym)
                seen.add(bucket)
            if len(picks) == n:
                break
        return picks

    use_sharpe_filter = params.get("pool_sharpe_filter", False)

    if risk_on:
        # Low-vol regime check
        low_vol_threshold = params.get("low_vol_threshold", 0.0)
        if low_vol_threshold > 0 and "SPY" in histories:
            if use_matrix:
                realized_vol = float(matrices.spy_realized_vol[date_idx])
            else:
                spy_close = histories["SPY"]["close"].loc[:as_of]
                if len(spy_close) >= 22:
                    daily_rets = spy_close.iloc[-22:].pct_change().dropna()
                    realized_vol = daily_rets.std() * math.sqrt(252) * 100
                else:
                    realized_vol = 999.0
            if realized_vol < low_vol_threshold:
                pool = params.get("low_vol_pool", ["QQQ", "MTUM", "VGT", "XLK"])
                pool = [s for s in pool if s in histories] or params["risk_on_pool"]
            else:
                pool = params["risk_on_pool"]
        else:
            pool = params["risk_on_pool"]
        if use_sharpe_filter:
            pool = _sharpe_filter(pool, histories, as_of)
        n = min(params["n_risk_on"], len(pool))
        window = params["risk_on_rsi_window"]
        rsi_vals = {sym: _rsi_at(sym, window) for sym in pool if sym in histories}
        direction = params.get("risk_on_rsi_direction", "lowest")
        if vsw > 0 or emaw > 0:
            scored = {sym: _blend_score(sym, rv, direction) for sym, rv in rsi_vals.items()}
            ranked = sorted(scored.items(), key=lambda kv: kv[1])
        else:
            ranked = sorted(rsi_vals.items(), key=lambda kv: kv[1], reverse=(direction == "highest"))
        return _sector_filter(ranked, n)

    if rising_rates:
        pool = params["risk_off_rising_pool"]
        if use_sharpe_filter:
            pool = _sharpe_filter(pool, histories, as_of)
        n_from_pool = max(1, params["n_risk_off_rising"] - (1 if params.get("rising_rate_include_uup") else 0))
        n_from_pool = min(n_from_pool, len(pool))
        window = params["risk_off_rsi_window"]
        rsi_vals = {sym: _rsi_at(sym, window) for sym in pool if sym in histories}
        direction = params.get("risk_off_rsi_direction", "lowest")
        if vsw > 0 or emaw > 0:
            scored = {sym: _blend_score(sym, rv, direction) for sym, rv in rsi_vals.items()}
            ranked = sorted(scored.items(), key=lambda kv: kv[1])
        else:
            ranked = sorted(rsi_vals.items(), key=lambda kv: kv[1], reverse=(direction == "highest"))
        picks = _sector_filter(ranked, n_from_pool)
        if params.get("rising_rate_include_uup") and "UUP" in histories:
            picks = ["UUP"] + picks
        return picks

    # Flight-to-quality override: if TLT is surging on a fast lookback (bonds
    # rallying during equity stress), go to cash (BIL) rather than the falling
    # pool — we're late to the rotation and chasing will hurt more than sitting out.
    if tlt_surging and params.get("tlt_surge_cash", False):
        return ["BIL"] if "BIL" in histories else []

    pool = params["risk_off_falling_pool"]
    if use_sharpe_filter:
        pool = _sharpe_filter(pool, histories, as_of)
    n = min(params["n_risk_off_falling"], len(pool))
    return pool[:n]


# ── Main backtest runner (matrix-accelerated) ────────────────────────────────

def run_etf_backtest(
    params: dict[str, Any],
    start: str,
    end: str,
    cash: float = 100_000.0,
    massive_key: str | None = None,
    polygon_key: str | None = None,
    preloaded_histories: "dict | None" = None,
) -> dict[str, Any]:
    needed = set(ANCHORS)
    needed.update(params["risk_on_pool"])
    needed.update(params["risk_off_rising_pool"])
    needed.update(params["risk_off_falling_pool"])
    if params.get("rising_rate_include_uup"):
        needed.add("UUP")
    symbols = sorted(needed)

    warmup = max(params["agg_bil_lookback"], params["tlt_bil_lookback"],
                 params["risk_on_rsi_window"], params["risk_off_rsi_window"]) + 5

    if preloaded_histories is not None:
        pre_start_ts = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=warmup * 2)
        end_ts = pd.Timestamp(end, tz="UTC")
        histories = {
            sym: preloaded_histories[sym].loc[pre_start_ts:end_ts]
            for sym in symbols
            if sym in preloaded_histories
        }
    else:
        pre_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup * 2)).strftime("%Y-%m-%d")
        histories = load_local_histories(symbols, pre_start, end)

    anchor_indexes = [set(histories[a].index) for a in ANCHORS if a in histories]
    all_common = sorted(set.intersection(*anchor_indexes))
    start_ts = pd.Timestamp(start, tz="UTC")
    live_from = next((i for i, d in enumerate(all_common) if d >= start_ts), None)
    if live_from is None or live_from < warmup:
        live_from = warmup
    common = all_common

    # ── Pre-compute indicator matrices once ───────────────────────────────────
    bm = precompute_backtest_matrices(histories, params, common)

    positions: dict[str, int] = {sym: 0 for sym in symbols}
    portfolio_cash = Decimal(str(cash))
    nav: list[dict] = []
    trade_history: list[dict] = []
    first_live_day = None
    bnh_symbols: list[str] = []
    min_hold = params.get("min_hold_days", 1)
    last_rebalance_idx = -min_hold
    rsi_cache = precompute_rsi_cache(histories, params)
    stop_loss_pct = params.get("stop_loss_pct", 0.0)
    stop_loss_lockout = int(params.get("stop_loss_lockout_days", 22))
    entry_value: float | None = None
    vol_target_pct = params.get("vol_target_pct", 0.0)
    vol_target_lookback = int(params.get("vol_target_lookback", 21))
    nav_for_vol: list[float] = []

    for idx in range(1, len(common)):
        today = common[idx]
        yesterday = common[idx - 1]
        date_idx = idx - 1  # matrix index for yesterday (= as_of in choose_targets)
        port_value = float(value_of_portfolio(histories, positions, portfolio_cash, today))

        if idx < live_from:
            continue

        if (stop_loss_pct > 0 and entry_value is not None
                and port_value < entry_value * (1.0 - stop_loss_pct / 100.0)):
            positions, portfolio_cash, sl_trades = rebalance_positions(
                histories, positions, portfolio_cash, [], today, weights=None
            )
            trade_history.extend(sl_trades)
            entry_value = None
            last_rebalance_idx = idx + stop_loss_lockout

        nav_for_vol.append(port_value)

        try:
            targets = choose_targets(
                histories, yesterday, params, rsi_cache,
                matrices=bm, date_idx=date_idx,
            )
            if first_live_day is None:
                first_live_day = today
                bnh_symbols = targets[:]
            current_held = {sym for sym, qty in positions.items() if qty > 0}
            held_long_enough = (idx - last_rebalance_idx) >= min_hold
            rebalance_now = (set(targets) != current_held) and held_long_enough
            if rebalance_now:
                combo_alpha = params.get("combo_alpha")
                if combo_alpha is not None:
                    weights = compute_combo_weights(
                        histories, targets, yesterday,
                        momentum_lookback=params.get("combo_momentum_lookback", 21),
                        vol_lookback=params.get("combo_vol_lookback", 21),
                        alpha=combo_alpha,
                        max_weight=params.get("combo_max_weight", 1.0),
                        score_normalize_window=params.get("score_normalize_window", 0),
                        lev_3x_cap=params.get("lev_3x_cap", 1.0),
                        matrices=bm,
                        date_idx=date_idx,
                    )
                else:
                    vww = params.get("vol_weight_window", 0)
                    weights = compute_vol_weights(histories, targets, yesterday, vww) if vww > 0 else None

                # Vol targeting
                in_low_vol_regime = False
                low_vol_threshold = params.get("low_vol_threshold", 0.0)
                if low_vol_threshold > 0 and "SPY" in histories:
                    spy_vol = float(bm.spy_realized_vol[date_idx]) if bm else 999.0
                    if not bm:
                        spy_close = histories["SPY"]["close"].loc[:yesterday]
                        if len(spy_close) >= 22:
                            spy_rets = spy_close.iloc[-22:].pct_change().dropna()
                            spy_vol = spy_rets.std() * math.sqrt(252) * 100
                        else:
                            spy_vol = 999.0
                    in_low_vol_regime = spy_vol < low_vol_threshold

                if vol_target_pct > 0 and not in_low_vol_regime and len(nav_for_vol) >= vol_target_lookback + 1:
                    nav_window = nav_for_vol[-(vol_target_lookback + 1):]
                    daily_rets = [nav_window[i] / nav_window[i-1] - 1 for i in range(1, len(nav_window))]
                    mean_r = sum(daily_rets) / len(daily_rets)
                    variance = sum((r - mean_r) ** 2 for r in daily_rets) / len(daily_rets)
                    realized_vol = math.sqrt(variance * 252) * 100

                    effective_vol_target = vol_target_pct
                    if params.get("vol_target_adaptive") and bm:
                        spy_cur = float(bm.spy_realized_vol[date_idx])
                        spy_med = float(bm.spy_median_vol[date_idx])
                        if spy_cur > spy_med > 0:
                            effective_vol_target = vol_target_pct * (spy_med / spy_cur)
                    elif params.get("vol_target_adaptive") and "SPY" in histories:
                        spy_close = histories["SPY"]["close"].loc[:yesterday]
                        if len(spy_close) >= 147:
                            spy_rets_s = spy_close.pct_change().dropna()
                            spy_cur = float(spy_rets_s.iloc[-21:].std() * math.sqrt(252) * 100)
                            spy_rolling = spy_rets_s.rolling(21).std().iloc[-126:] * math.sqrt(252) * 100
                            spy_med = float(spy_rolling.dropna().median())
                            if spy_cur > spy_med > 0:
                                effective_vol_target = vol_target_pct * (spy_med / spy_cur)

                    if realized_vol > 0.1:
                        vol_scale = min(1.0, effective_vol_target / realized_vol)
                        if weights:
                            weights = {k: v * vol_scale for k, v in weights.items()}
                        else:
                            eq = 1.0 / len(targets) if targets else 1.0
                            weights = {s: eq * vol_scale for s in targets}

                positions, portfolio_cash, trades = rebalance_positions(
                    histories, positions, portfolio_cash, targets, today, weights=weights
                )
                trade_history.extend(trades)
                last_rebalance_idx = idx
                entry_value = float(value_of_portfolio(histories, positions, portfolio_cash, today))
        except (BacktestError, KeyError, IndexError):
            pass

        port_value = float(value_of_portfolio(histories, positions, portfolio_cash, today))
        nav.append({"date": today, "value": port_value})

    if not nav:
        raise BacktestError(
            "Backtest produced no data — check that anchor symbols (AGG, BIL, TLT) "
            "have data in the requested date range."
        )
    nav_df = pd.DataFrame(nav).set_index("date")
    perf = compute_performance(nav_df["value"])
    trade_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()

    live_index = nav_df.index[nav_df.index >= first_live_day] if first_live_day else nav_df.index
    bnh_series = _buy_and_hold_nav(histories, bnh_symbols, cash, live_index)

    return {
        "params": params,
        "nav_df": nav_df,
        "first_live_day": first_live_day,
        "bnh_df": bnh_series,
        "bnh_symbols": bnh_symbols,
        "trade_df": trade_df,
        "perf": perf,
        "score": perf.get("total_return_pct", -999.0),
        "symbols_used": symbols,
    }
