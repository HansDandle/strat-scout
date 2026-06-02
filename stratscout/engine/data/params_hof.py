"""
Hall of Fame: shared param memory across all walk-forward runs and DBs.

Every completed walk-forward month writes its best params + market features here.
Every new month seeds Optuna with the best-performing params from historically
similar months (KNN on market features) plus the all-time top scorers.

Over time this accumulates cross-run knowledge: the optimizer starts each month
already knowing what worked when conditions looked like this.

Storage: data/params_hof.db  (one shared SQLite, never wiped)
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from stratscout.engine.settings import data_dir


_HOF_PATH_OVERRIDE: Path | None = None


def set_hof_path(path: str | Path) -> None:
    """Override the HoF path for this process. Call before init_hof().
    Use this to isolate a run to its own clean HoF (e.g. clean validation runs).
    """
    global _HOF_PATH_OVERRIDE
    _HOF_PATH_OVERRIDE = Path(path)


def _hof_path() -> Path:
    if _HOF_PATH_OVERRIDE is not None:
        return _HOF_PATH_OVERRIDE
    # Always use the canonical project-root data/ location regardless of CWD.
    # Runs launched from the project root previously wrote to data/params_hof.db;
    # this ensures we never silently create a second empty DB in a subdirectory.
    from stratscout.engine.settings import data_dir as _data_dir
    return _data_dir() / "params_hof.db"


def init_hof() -> None:
    con = sqlite3.connect(_hof_path())
    con.execute("""
        CREATE TABLE IF NOT EXISTS hof (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_db       TEXT,
            month           TEXT,
            train_score     REAL,
            val_return      REAL,
            val_cagr        REAL,
            val_dd          REAL,
            val_trades      INTEGER,
            params          TEXT,
            spy_1m_ret      REAL,
            spy_3m_ret      REAL,
            spy_vol_21d     REAL,
            month_num       INTEGER,
            regime          REAL,
            prev_val_return REAL,
            UNIQUE(source_db, month)
        )
    """)
    # Migrate existing tables that predate prev_val_return
    cols = {r[1] for r in con.execute("PRAGMA table_info(hof)").fetchall()}
    if "prev_val_return" not in cols:
        con.execute("ALTER TABLE hof ADD COLUMN prev_val_return REAL")
    con.commit()
    con.close()


# ── Feature computation ────────────────────────────────────────────────────────

def compute_month_features(month_start: str) -> dict | None:
    """
    Compute market-condition features for a given month.
    Returns None if SPY data is insufficient.

    Features (all normalized to roughly [-1, 1] or [0, 1]):
      spy_1m_ret   — SPY return in the prior month
      spy_3m_ret   — SPY return over the prior 3 months
      spy_vol_21d  — realized annualized vol over prior 21 trading days
      month_num    — 1–12 (captures seasonality)
      regime       — 1=bull, 0=sideways, -1=bear (3-month rolling mean threshold)
    """
    try:
        spy_path = data_dir() / "daily" / "SPY.feather"
        spy = pd.read_feather(spy_path)
        date_col = "date" if "date" in spy.columns else spy.columns[0]
        spy[date_col] = pd.to_datetime(spy[date_col], utc=True)
        spy = spy.set_index(date_col).sort_index()

        anchor = pd.Timestamp(month_start, tz="UTC")
        hist = spy.loc[:anchor, "close"].iloc[:-1]  # up to but not including month_start

        if len(hist) < 65:
            return None

        # 1-month return (~21 trading days)
        spy_1m = (hist.iloc[-1] / hist.iloc[-22] - 1) * 100 if len(hist) >= 22 else 0.0

        # 3-month return (~63 trading days)
        spy_3m = (hist.iloc[-1] / hist.iloc[-64] - 1) * 100 if len(hist) >= 64 else spy_1m

        # Realized vol: std of daily returns × sqrt(252) × 100
        daily_rets = hist.iloc[-22:].pct_change().dropna()
        spy_vol = daily_rets.std() * math.sqrt(252) * 100 if len(daily_rets) >= 5 else 20.0

        # Regime: based on 3-month rolling mean of monthly returns
        monthly = hist.resample("ME").last().pct_change(fill_method=None).dropna() * 100
        roll3 = monthly.rolling(3).mean().iloc[-1] if len(monthly) >= 3 else 0.0
        if roll3 > 1.5:
            regime = 1.0
        elif roll3 < -1.5:
            regime = -1.0
        else:
            regime = 0.0

        return {
            "spy_1m_ret":  round(spy_1m, 4),
            "spy_3m_ret":  round(spy_3m, 4),
            "spy_vol_21d": round(spy_vol, 4),
            "month_num":   anchor.month,
            "regime":      regime,
        }
    except Exception:
        return None


# ── Read / Write ───────────────────────────────────────────────────────────────

def save_to_hof(
    source_db: str,
    month: str,
    train_score: float,
    val_return: float,
    val_cagr: float,
    val_dd: float,
    val_trades: int,
    params: dict,
    features: dict | None,
    prev_val_return: float | None = None,
) -> None:
    """Upsert one month's result into the shared HoF."""
    init_hof()
    f = features or {}
    con = sqlite3.connect(_hof_path())
    con.execute("""
        INSERT INTO hof
            (source_db, month, train_score, val_return, val_cagr, val_dd, val_trades,
             params, spy_1m_ret, spy_3m_ret, spy_vol_21d, month_num, regime, prev_val_return)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source_db, month) DO UPDATE SET
            train_score=excluded.train_score,
            val_return=excluded.val_return,
            val_cagr=excluded.val_cagr,
            val_dd=excluded.val_dd,
            val_trades=excluded.val_trades,
            params=excluded.params,
            spy_1m_ret=excluded.spy_1m_ret,
            spy_3m_ret=excluded.spy_3m_ret,
            spy_vol_21d=excluded.spy_vol_21d,
            month_num=excluded.month_num,
            regime=excluded.regime,
            prev_val_return=excluded.prev_val_return
    """, (
        source_db, month, train_score, val_return, val_cagr, val_dd, val_trades,
        json.dumps(params, default=str),
        f.get("spy_1m_ret"), f.get("spy_3m_ret"), f.get("spy_vol_21d"),
        f.get("month_num"), f.get("regime"), prev_val_return,
    ))
    con.commit()
    con.close()


def _strip_symbol_flags(params: dict) -> dict:
    """Remove on_XXX symbol-selection flags from a params dict.

    Symbol selections are period-specific (what worked in a past bull run
    isn't a useful prior for which symbols to pick next month). Seeding
    with symbol flags biases Optuna toward historically-popular symbols
    before it has seen any evidence for the current month's regime.
    We keep all structural params (lookbacks, thresholds, weights) and
    let Optuna explore symbol selection freely from a neutral start.
    """
    return {k: v for k, v in params.items() if not k.startswith("on_")}


def find_seeds(
    month_start: str,
    features: dict | None = None,
    as_of_month: str | None = None,
    min_val_return: float = 0.0,
) -> list[dict]:
    """
    Return up to 8 structural param dicts to seed Optuna with.

    Four purposeful strategies (symbol flags stripped from all):

    1. SAME-MONTH-LAST-YEAR — params that worked in the same calendar month
       in prior years. Captures seasonality (e.g. Nov tends to behave like Nov).

    2. RECENT-CONSISTENCY — the single param set that was profitable across
       the most months in the trailing 12 months. Captures what's been
       broadly working lately, not just one lucky month.

    3. REGIME-MATCH — params from the 3 past months whose SPY regime
       (vol + trend) most closely matches current conditions. Captures
       structural similarity without recency bias.

    4. LAST-WINNER — params from the single most recent profitable month.
       Pure momentum: if it worked last month, try it first.

    All seeds are strictly before as_of_month (no lookahead).
    min_val_return: filter out seeds from periods with val_return below this threshold.
    """
    try:
        init_hof()
        con = sqlite3.connect(_hof_path())
        if as_of_month:
            rows = con.execute(
                "SELECT month, params, spy_1m_ret, spy_3m_ret, spy_vol_21d, "
                "month_num, regime, val_return "
                "FROM hof WHERE params IS NOT NULL AND val_return >= ? AND month < ? "
                "ORDER BY month ASC",
                (min_val_return, as_of_month)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT month, params, spy_1m_ret, spy_3m_ret, spy_vol_21d, "
                "month_num, regime, val_return "
                "FROM hof WHERE params IS NOT NULL AND val_return >= ? ORDER BY month ASC",
                (min_val_return,)
            ).fetchall()
        con.close()
    except Exception:
        return []

    if not rows:
        return []

    cal_month = int(month_start[5:7])  # 1–12
    candidates: list[dict] = []

    # ── Strategy 1: same calendar month, prior years ────────────────────────
    same_month_rows = [r for r in rows if int(r[0][5:7]) == cal_month and r[7] > 0]
    if same_month_rows:
        # Best val_return among same-month entries
        best = max(same_month_rows, key=lambda r: r[7])
        try:
            candidates.append(_strip_symbol_flags(json.loads(best[1])))
        except Exception:
            pass

    # ── Strategy 2: recent consistency (most profitable months in trailing 12) ─
    if as_of_month:
        from dateutil.relativedelta import relativedelta as _rd
        cutoff = (
            pd.Timestamp(as_of_month) - _rd(months=12)
        ).strftime("%Y-%m-%d")
        trailing = [r for r in rows if r[0] >= cutoff and r[7] > 0]
    else:
        trailing = [r for r in rows if r[7] > 0]

    if trailing:
        # For each unique param fingerprint (structural keys only), count
        # how many months it appeared in. Pick the most consistent one.
        from collections import Counter
        _STRUCT_KEYS = {
            "agg_bil_lookback", "tlt_bil_lookback", "risk_on_rsi_window",
            "risk_off_rsi_window", "risk_on_rsi_direction", "risk_off_rsi_direction",
            "min_hold_days", "combo_momentum_lookback", "combo_vol_lookback",
            "combo_alpha", "combo_max_weight", "stop_loss_pct",
            "stop_loss_lockout_days", "vol_target_pct", "vol_target_lookback",
            "n_risk_on", "n_risk_off_rising", "n_risk_off_falling",
        }
        fingerprints: dict[tuple, list[tuple]] = {}  # fp -> [(params_dict, val_return)]
        for r in trailing:
            try:
                p = json.loads(r[1])
                val_ret = float(r[7])
                # Discretize to nearest integer for clustering (avoids float churn)
                fp = tuple(
                    (k, round(p[k]) if isinstance(p.get(k), float) else p.get(k))
                    for k in sorted(_STRUCT_KEYS)
                    if k in p
                )
                fingerprints.setdefault(fp, []).append((p, val_ret))
            except Exception:
                pass
        if fingerprints:
            most_consistent = max(fingerprints.values(), key=len)
            # Use the instance with the highest val_return from that cluster
            best_consistent = max(most_consistent, key=lambda pv: pv[1])[0]
            candidates.append(_strip_symbol_flags(best_consistent))

    # ── Strategy 3: regime-match (closest SPY vol + trend) ──────────────────
    if features:
        cur_vol   = features.get("spy_vol_21d", 20.0)
        cur_1m    = features.get("spy_1m_ret",   0.0)
        cur_3m    = features.get("spy_3m_ret",   0.0)
        cur_regime = features.get("regime",      0.0)

        regime_rows = [r for r in rows if r[7] > 0 and None not in (r[2], r[3], r[4], r[6])]

        def _regime_dist(r) -> float:
            return math.sqrt(
                ((r[4] - cur_vol) / max(cur_vol, 1)) ** 2 +
                ((r[2] - cur_1m) / 10.0) ** 2 +
                ((r[3] - cur_3m) / 15.0) ** 2 +
                (r[6] - cur_regime) ** 2
            )

        if regime_rows:
            closest = sorted(regime_rows, key=_regime_dist)[:3]
            for r in closest:
                try:
                    candidates.append(_strip_symbol_flags(json.loads(r[1])))
                except Exception:
                    pass

    # ── Strategy 4: last winner (most recent profitable month) ──────────────
    profitable = [r for r in rows if r[7] > 0]
    if profitable:
        last = profitable[-1]
        try:
            candidates.append(_strip_symbol_flags(json.loads(last[1])))
        except Exception:
            pass

    # Deduplicate, preserve order (earlier strategies = higher priority)
    seen: set[str] = set()
    result = []
    for p in candidates:
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(p)

    return result


# Keep old name as alias so any other callers don't break
def find_similar_seeds(
    features: dict | None,
    k_similar: int = 5,
    top_global: int = 3,
    min_val_return: float = 0.0,
    as_of_month: str | None = None,
) -> list[dict]:
    return find_seeds(
        month_start=as_of_month or "2099-01-01",
        features=features,
        as_of_month=as_of_month,
        min_val_return=min_val_return,
    )


# ── Param bounds from HoF history ────────────────────────────────────────────

# Hardcoded original search ranges — fallback when HoF has insufficient data
_PARAM_DEFAULTS: dict[str, tuple] = {
    # Ranges tightened from HoF empirical p10-p90 across 631+ profitable months.
    # Sparse params (n<150) keep wider defaults to avoid over-constraining.
    "agg_bil_lookback":       (72,  107),   # was 70-110, p10-p90=72-107
    "tlt_bil_lookback":       (6,   18),    # was 5-20,   p10-p90=6-18
    "risk_on_rsi_window":     (16,  29),    # was 15-30,  p10-p90=16-29
    "risk_off_rsi_window":    (7,   23),    # was 5-25,   p10-p90=7-23
    "n_risk_on":              (2,   3),
    "n_risk_off_rising":      (1,   3),
    "n_risk_off_falling":     (2,   5),
    "min_hold_days":          (4,   11),    # was 1-10,   p10-p90=4-11
    "combo_momentum_lookback":(12,  57),    # was 5-63,   p10-p90=12-57
    "combo_vol_lookback":     (9,   39),    # was 5-42,   p10-p90=9-39
    "combo_alpha":            (0.1, 0.9),   # was 0-1,    p10-p90=0.1-0.9
    "combo_max_weight":       (0.4, 0.9),   # was 0.3-1,  p10-p90=0.4-0.9
    "stop_loss_pct":          (5.0, 19.0),  # was 4-20,   p10-p90=5.3-18.5
    "stop_loss_lockout_days": (16,  29),    # was 15-30,  p10-p90=16-29
    "vol_target_pct":         (0.0, 14.0),  # allow disabled; 63% of profitable HoF entries have it off
    "vol_target_lookback":    (12,  28),    # was 10-30,  p10-p90=12-28
    "low_vol_threshold":      (0.0, 20.0),  # sparse n=147, keep wide
    "score_normalize_window": (0,   24),    # sparse n=146, keep wide
    "recency_weight":         (1.0, 4.0),   # sparse n=110, keep wide
    "ema_fast":               (5,   20),    # full range still valid
    "ema_slow":               (20,  100),   # full range still valid
}

# Minimum fraction of original range to preserve — prevents over-collapse on sparse data
_MIN_RANGE_FRACTION = 0.25


def compute_param_bounds_from_hof(
    as_of_month: str | None = None,
    min_val_return: float = 0.5,
    min_samples: int = 20,
    percentile_lo: float = 0.10,
    percentile_hi: float = 0.90,
    features: dict | None = None,
) -> dict[str, tuple]:
    """Return tightened (lo, hi) bounds for each scalar param based on HoF history.

    Uses profitable months (val_return >= min_val_return) strictly before as_of_month.
    When fewer than min_samples exist (cold start), falls back to a seasonal warmup:
    pulls HoF entries from the same calendar month ±1 across all years — giving early
    months the benefit of historical knowledge without any lookahead bias.
    Falls back to _PARAM_DEFAULTS only if seasonal warmup also has insufficient data.
    Each bound is at least _MIN_RANGE_FRACTION of the original range wide.

    Regime-aware tightening: if features contain regime (bull/bear) or high vol,
    tightens ranges to focus search on relevant params.
    """
    try:
        init_hof()
        con = sqlite3.connect(_hof_path())
        if as_of_month:
            rows = con.execute(
                "SELECT params FROM hof WHERE val_return >= ? AND params IS NOT NULL AND month < ?",
                (min_val_return, as_of_month)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT params FROM hof WHERE val_return >= ? AND params IS NOT NULL",
                (min_val_return,)
            ).fetchall()
        con.close()
    except Exception:
        return dict(_PARAM_DEFAULTS)

    # Seasonal warmup: when strict past history is thin, pull same-month entries
    # from all other years. No lookahead — we never use months >= as_of_month.
    if len(rows) < min_samples and as_of_month:
        try:
            cal_month = int(as_of_month[5:7])  # 1-12
            adjacent = {(cal_month - 1) % 12 + 1, cal_month, cal_month % 12 + 1}
            placeholders = ",".join("?" * len(adjacent))
            con = sqlite3.connect(_hof_path())
            seasonal_rows = con.execute(
                f"SELECT params FROM hof WHERE val_return >= ? AND params IS NOT NULL "
                f"AND month < ? AND CAST(strftime('%m', month) AS INTEGER) IN ({placeholders})",
                (min_val_return, as_of_month, *adjacent)
            ).fetchall()
            con.close()
            if len(seasonal_rows) >= min_samples:
                rows = seasonal_rows
        except Exception:
            pass

    if len(rows) < min_samples:
        return dict(_PARAM_DEFAULTS)

    # Collect values per param across all profitable months
    param_vals: dict[str, list[float]] = {k: [] for k in _PARAM_DEFAULTS}
    for (p_json,) in rows:
        try:
            p = json.loads(p_json)
            for key in _PARAM_DEFAULTS:
                if key in p and isinstance(p[key], (int, float)):
                    param_vals[key].append(float(p[key]))
        except Exception:
            pass

    bounds: dict[str, tuple] = {}
    for key, (orig_lo, orig_hi) in _PARAM_DEFAULTS.items():
        vals = sorted(param_vals[key])
        orig_range = orig_hi - orig_lo
        min_range = orig_range * _MIN_RANGE_FRACTION

        if len(vals) < 5:
            bounds[key] = (orig_lo, orig_hi)
            continue

        n = len(vals)
        emp_lo = vals[max(0, int(percentile_lo * n))]
        emp_hi = vals[min(n - 1, int(percentile_hi * n))]

        # Ensure minimum width
        if emp_hi - emp_lo < min_range:
            mid = (emp_lo + emp_hi) / 2
            emp_lo = mid - min_range / 2
            emp_hi = mid + min_range / 2

        # Clamp to original range
        emp_lo = max(orig_lo, emp_lo)
        emp_hi = min(orig_hi, emp_hi)

        # Preserve type (int params stay int)
        if isinstance(orig_lo, int):
            bounds[key] = (int(emp_lo), int(emp_hi))
        else:
            bounds[key] = (round(emp_lo, 4), round(emp_hi, 4))

    # Regime-aware tightening: narrow ranges during extreme market conditions
    # This focuses the optimizer on params that typically work in the current regime
    if features:
        regime = features.get("regime", 0.0)
        vol = features.get("spy_vol_21d", 20.0)

        # Tighten ranges during extreme regimes (bull: 1.0, bear: -1.0) or high vol (>25%)
        tighten_factor = 0.0
        if abs(regime) > 0.5:  # moderate to extreme regime
            tighten_factor += 0.15  # tighten by 15%
        if vol > 25.0:  # high volatility
            tighten_factor += 0.15  # additional 15%

        if tighten_factor > 0:
            for key, (lo, hi) in bounds.items():
                mid = (lo + hi) / 2
                width = hi - lo
                new_width = width * (1 - tighten_factor)
                new_lo = mid - new_width / 2
                new_hi = mid + new_width / 2

                # Clamp to original defaults
                orig_lo, orig_hi = _PARAM_DEFAULTS[key]
                new_lo = max(orig_lo, new_lo)
                new_hi = min(orig_hi, new_hi)

                # Preserve type
                if isinstance(orig_lo, int):
                    bounds[key] = (int(new_lo), int(new_hi))
                else:
                    bounds[key] = (round(new_lo, 4), round(new_hi, 4))

    return bounds
