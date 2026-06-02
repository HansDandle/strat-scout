"""
Walk-forward validator for the ETF rotation strategy.

For each month M from --start to --end:
  Train:    fuzz N trials on [M-12mo, M)  split into 3 sub-windows → find best param set
  Validate: run that param set on [M, M+1mo) → record out-of-sample result

With 7 years of ETF data we use a 12-month training window (vs 3 months for options)
for much more stable signal. Walk-forward over 2020-2026 gives ~72 out-of-sample months.

Results saved to walk_forward_etf.db (never wiped — reruns skip completed months).

Usage:
    python walk_forward_etf.py
    python walk_forward_etf.py --start 2020-01-01 --trials 300 --workers 4
    python walk_forward_etf.py --summary   # just print results, no new runs
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sqlite3
import time
from datetime import date
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

DB_PATH = Path("walk_forward_etf.db")

# Small universe for fast smoke-tests (--mini flag). Covers key regimes in ~6x fewer symbols.
MINI_UNIVERSE = ["AGG", "BIL", "TLT", "TQQQ", "UPRO", "QID", "GLD", "FPX", "SPY"]

# Minimal universe for --fast-test framework: 8 symbols covering all 3 regimes.
FAST_TEST_UNIVERSE = ["AGG", "BIL", "TLT", "TQQQ", "UPRO", "QID", "GLD", "FPX"]

# Fixed chaining orders for the --fast-test drawdown distribution. The 24 stratified
# months are numbered 1-24 by regime group: 1-8 = risk-on (bull), 9-16 = risk-off
# rising, 17-24 = risk-off falling. Each order below chains those same 24 months "as
# if consecutive" in a different sequence. CAGR is identical across orders (commutative
# product) — only the drawdown path changes, so these probe path/sequence risk. The
# suite always uses these exact orders. Append more lists to widen the DD distribution.
FAST_TEST_ORDERS = [
    [13, 2, 16, 11, 7, 9, 5, 6, 15, 17, 3, 12, 21, 1, 4, 14, 19, 23, 8, 24, 10, 18, 22, 20],
    [4, 5, 3, 9, 18, 17, 11, 14, 20, 6, 15, 1, 2, 24, 7, 19, 12, 22, 21, 13, 16, 10, 8, 23],
    [6, 13, 8, 12, 3, 2, 7, 11, 1, 24, 21, 18, 20, 4, 17, 15, 23, 16, 10, 22, 5, 14, 9, 19],
    [18, 12, 8, 22, 23, 19, 10, 5, 15, 7, 14, 3, 11, 17, 20, 4, 21, 2, 1, 16, 6, 13, 24, 9],
    [20, 16, 6, 10, 17, 3, 5, 11, 22, 24, 2, 19, 12, 18, 9, 1, 15, 4, 13, 7, 14, 21, 23, 8],
]
# Fail loudly if an order is edited into something other than a permutation of 1..24.
for _i, _o in enumerate(FAST_TEST_ORDERS):
    assert sorted(_o) == list(range(1, 25)), f"FAST_TEST_ORDERS[{_i}] is not a permutation of 1..24"

# ETF inception dates — used to gate pool membership for pre-2015 runs.
# Only symbols with inception <= validation period start are eligible.
# Source: ETF issuer prospectus / Yahoo Finance historical data.
INCEPTION_DATES: dict[str, str] = {
    # Regime anchors (always required — BIL is the hard floor at 2007-05-30)
    "AGG":  "2003-09-29", "BIL":  "2007-05-30", "TLT":  "2002-07-30",
    # Pre-2009 risk-on
    "QQQ":  "1999-03-10", "IWM":  "2000-05-26", "EFA":  "2001-08-27",
    "EEM":  "2003-04-14", "VNQ":  "2004-09-29", "GLD":  "2004-11-18",
    "XLK":  "1998-12-22", "XLF":  "1998-12-22", "XLE":  "1998-12-22",
    "XLV":  "1998-12-22", "VGT":  "2004-01-30", "FPX":  "2006-04-12",
    "MSTR": "1998-06-11", "UUP":  "2007-03-01",
    # 2008-2009 cohort
    "SPXL": "2008-11-05", "FAS":  "2008-11-19", "ERX":  "2008-11-19",
    "TNA":  "2008-11-19", "TECL": "2008-12-30", "UGL":  "2008-12-03",
    "TBT":  "2008-05-22", "MIDU": "2009-01-08", "UPRO": "2009-06-25",
    "DRN":  "2009-07-16", "TBF":  "2009-08-20", "TMF":  "2009-04-16",
    # 2010-2012 cohort
    "SOXL": "2010-03-11", "TQQQ": "2010-02-11", "URTY": "2010-02-11",
    "SIL":  "2010-04-20", "SQQQ": "2010-02-11", "NUGT": "2010-12-08",
    "PSQ":  "2006-06-21", "QID":  "2006-07-13",
    "SILJ": "2012-11-29", "BTAL": "2011-09-13", "CURE": "2011-06-15",
    "XLP":  "1998-12-22",
    # 2013+ cohort
    "JNUG": "2013-10-03", "MTUM": "2013-04-18",
    # 2015+ (no gate needed, all data starts 2015)
    "LABU": "2015-05-28", "GBTC": "2015-05-11",
    # 2017+
    "UTSL": "2017-05-03",
    # 2020+
    "QQQM": "2020-10-13", "GDXU": "2020-12-03",
    # 2021+
    "FNGU": "2021-01-11",
    # Crypto 2022+ (CONL/BITX/IBIT removed 2026-06-01 — sub-50% win rate)
    # Pre-2015 extended pools (added for 2007-present runs)
    "SLV":  "2006-04-28",
    # Uncurated additions for randomized-universe experiment
    "XLU":  "1998-12-22", "XLI":  "1998-12-22", "XLB":  "1998-12-22",
    "XLC":  "2018-06-18", "XLY":  "1998-12-22",
    "VEA":  "2007-07-26", "VWO":  "2005-03-10", "MDY":  "1995-05-04",
    "GDX":  "2006-05-22", "GDXJ": "2009-11-10", "IBB":  "2001-02-08",
    "XBI":  "2006-01-31",
    "SH":   "2006-06-21", "FAZ":  "2008-11-19", "TZA":  "2008-11-19",
    "SPXS": "2009-06-25",
    "LQD":  "2002-07-30", "TIP":  "2003-12-05",
}

# ── Randomized-universe experiment ───────────────────────────────────────────
# Master pools — superset of curated + uncurated ETFs. When --random-universe
# SEED is passed, a random subset is drawn from each master pool at startup and
# frozen for the whole run. This tests whether the regime gate is the alpha
# source vs the specific ETF picks.
MASTER_RISK_ON_POOL: list[str] = [
    # Current curated risk-on pool
    "SOXL", "TQQQ", "UPRO", "TECL", "SPXL", "FAS", "CURE", "LABU",
    "ERX", "DRN", "FNGU", "UTSL", "MIDU", "TNA", "URTY",
    "MSTR", "GBTC",
    "JNUG", "GDXU", "SILJ", "SIL",
    "MTUM", "VGT", "XLK", "QQQM", "FPX",
    # Uncurated broad-market / sector (not hand-picked for performance)
    "QQQ", "IWM", "VNQ", "XLF", "XLE", "XLV", "EEM", "EFA",
    "XLU", "XLI", "XLB", "XLC", "XLY", "VEA", "VWO", "MDY",
    "GDX", "GDXJ", "IBB", "XBI",
]

MASTER_RISK_OFF_RISING_POOL: list[str] = [
    # Curated inverse-bond / short-equity
    "QID", "TBF", "SQQQ", "TBT", "PSQ",
    # Uncurated additions
    "SH", "FAZ", "TZA", "SPXS",
]

MASTER_RISK_OFF_FALLING_POOL: list[str] = [
    # Curated defensive
    "UGL", "TMF", "BTAL", "XLP", "NUGT", "UUP", "GLD", "SLV",
    # Uncurated additions
    "XLU", "LQD", "TIP", "GDX",
]

# Set at startup by --random-universe; None = use curated pools
_random_on_pool:      list[str] | None = None
_random_rising_pool:  list[str] | None = None
_random_falling_pool: list[str] | None = None

# Experiment config — set at startup via CLI args
_lev3x_cap_choices:   list[float] = [0.25, 0.35, 0.50, 0.65, 0.80, 1.0]
_vol_target_range:    tuple[float, float] = (0.0, 0.0)   # (min, max); 0,0 = disabled
_lockout_min:         int = 15
# NOTE: "zany" signals (moon phase / Nikkei trigger / SPY-TLT corr) were removed 2026-05-31
# — proven to have no edge (CAGR -21pp vs baseline). Do not reintroduce.

# Extended pools that include pre-2015 symbols — used when --start predates 2015.
# The inception gate in _build_optuna_params filters these to what was actually trading.
_RISK_ON_POOL_EXTENDED: list[str] = [
    # Pre-2009: unleveraged broad market
    "QQQ", "IWM", "EFA", "EEM", "VNQ", "XLF", "XLE", "XLV", "VGT", "XLK", "FPX", "MSTR",
    # 2008-2009: first leveraged wave
    "SPXL", "FAS", "ERX", "TNA", "TECL", "MIDU", "UPRO", "DRN",
    # 2010+: second leveraged wave
    "SOXL", "TQQQ", "URTY", "SIL",
    # 2011+
    "CURE",
    # 2013+
    "JNUG", "MTUM", "SILJ",
    # 2015+
    "LABU", "GBTC",
    # 2017+
    "UTSL",
    # 2020+
    "QQQM", "GDXU", "FNGU",
    # Crypto (CONL/BITX/IBIT removed 2026-06-01 — sub-50% win rate at 1000+ periods)
]

_RISK_OFF_RISING_POOL_EXTENDED: list[str] = [
    "PSQ", "QID",           # 2006
    "TBT",                  # 2008
    "TBF", "SQQQ",          # 2009-2010
]

_RISK_OFF_FALLING_POOL_EXTENDED: list[str] = [
    "GLD", "SLV", "XLP",    # pre-2007
    "UGL", "TMF",            # 2008-2009
    "UUP",                   # 2007
    "NUGT",                  # 2010
    "BTAL",                  # 2011
    "CURE",                  # 2011 (dual-listed in falling for defensive health)
]

# Worker globals
_histories: dict = {}
_factors: dict = {}  # {name: pd.Series} loaded in _worker_init if factors exist
_no_factors: bool = False  # set True via --no-factors to run pure baseline
_use_calmar: bool = True   # set False via --no-calmar to use raw geo-mean scoring
_mini_mode: bool = False   # set True via --mini to use MINI_UNIVERSE
_recency_weight: float = 1.5  # w3/w1 ratio; w2 = geometric mean. Tunable via Optuna.


# ── Scoring: Calmar-based (CAGR / |MaxDD| per sub-window) ────────────────────

def _score_calmar(w1_cagr, w2_cagr, w3_cagr,
                  w1_dd=0.0, w2_dd=0.0, w3_dd=0.0,
                  w1_tr=1, w2_tr=1, w3_tr=1,
                  recency_weight: float = 1.5) -> float:
    """Score = weighted average of per-window Calmar ratios (CAGR / |MaxDD|).
    Cash sub-windows (trades=0) are neutral — not penalised — because a
    stop-loss correctly parking in cash should not hurt the score.
    recency_weight = w3/w1 ratio; w2 gets the geometric midpoint.
    """
    worst_dd = min(w1_dd, w2_dd, w3_dd)
    if worst_dd < -60.0:
        return -999.0
    if max(w1_cagr, w2_cagr, w3_cagr) > 500.0:
        return -999.0

    def _calmar(cagr: float, dd: float, trades: int) -> float:
        if trades == 0:
            return 0.0  # neutral: cash is fine, just not scored
        if cagr <= 0:
            return cagr / 10.0
        return min(cagr / max(abs(dd), 0.5), 200.0)

    c1 = _calmar(w1_cagr, w1_dd, w1_tr)
    c2 = _calmar(w2_cagr, w2_dd, w2_tr)
    c3 = _calmar(w3_cagr, w3_dd, w3_tr)

    rw = recency_weight
    weights = [1.0, math.sqrt(rw), rw]
    score = (weights[0]*c1 + weights[1]*c2 + weights[2]*c3) / sum(weights)

    # Consistency bonus for all-positive windows
    n_pos = sum(1 for c in (w1_cagr, w2_cagr, w3_cagr) if c > 0)
    score *= (1.0 + 0.1 * n_pos)
    # Penalty for negative windows
    n_neg = sum(1 for c in (w1_cagr, w2_cagr, w3_cagr) if c < 0)
    if n_neg > 0:
        score *= (0.6 ** n_neg)
    # No idle penalty: cash after a stop-out is the correct outcome
    return score


def _score_raw(w1_cagr, w2_cagr, w3_cagr,
               w1_dd=0.0, w2_dd=0.0, w3_dd=0.0,
               w1_tr=1, w2_tr=1, w3_tr=1,
               recency_weight: float = 1.5) -> float:
    """Original geo-mean score with Calmar-style DD penalty. Used with --no-calmar."""
    worst_dd = min(w1_dd, w2_dd, w3_dd)
    if worst_dd < -60.0:
        return -999.0
    if max(w1_cagr, w2_cagr, w3_cagr) > 500.0:
        return -999.0

    CAP = 500.0
    w1c = max(-100.0, min(CAP, w1_cagr))
    w2c = max(-100.0, min(CAP, w2_cagr))
    w3c = max(-100.0, min(CAP, w3_cagr))

    rw = recency_weight
    weights = [1.0, math.sqrt(rw), rw]
    factors = [1 + c / 100 for c in (w1c, w2c, w3c)]
    if any(f <= 0 for f in factors):
        return min(w1c, w2c, w3c)
    total_w = sum(weights)
    log_avg = sum(w * math.log(f) for w, f in zip(weights, factors)) / total_w
    raw = (math.exp(log_avg) - 1) * 100

    dd_penalty = 1.0 / (1.0 + abs(worst_dd) / 20.0)
    score = raw * dd_penalty

    n_pos = sum(1 for c in (w1c, w2c, w3c) if c > 0)
    score *= (1.0 + 0.15 * n_pos)
    n_neg = sum(1 for c in (w1c, w2c, w3c) if c < 0)
    if n_neg > 0:
        score *= (0.5 ** n_neg)
    n_idle = sum(1 for t in (w1_tr, w2_tr, w3_tr) if t == 0)
    if n_idle > 0:
        score *= (0.7 ** n_idle)
    return score


def _combined_score(w1_cagr, w2_cagr, w3_cagr,
                    w1_dd=0.0, w2_dd=0.0, w3_dd=0.0,
                    w1_tr=1, w2_tr=1, w3_tr=1,
                    recency_weight: float = 1.5) -> float:
    if _use_calmar:
        return _score_calmar(w1_cagr, w2_cagr, w3_cagr, w1_dd, w2_dd, w3_dd, w1_tr, w2_tr, w3_tr, recency_weight)
    return _score_raw(w1_cagr, w2_cagr, w3_cagr, w1_dd, w2_dd, w3_dd, w1_tr, w2_tr, w3_tr, recency_weight)


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker_init(no_factors: bool = False, exp_config: dict | None = None):
    global _histories, _factors
    # Propagate experiment config to spawned workers (Windows uses spawn, so module
    # globals set in main() are NOT inherited — they must be passed explicitly here).
    if exp_config:
        global _random_on_pool, _random_rising_pool, _random_falling_pool
        global _lev3x_cap_choices, _vol_target_range, _lockout_min
        global _use_calmar, _mini_mode
        _random_on_pool      = exp_config.get("on_pool",      _random_on_pool)
        _random_rising_pool  = exp_config.get("rising_pool",  _random_rising_pool)
        _random_falling_pool = exp_config.get("falling_pool", _random_falling_pool)
        _lev3x_cap_choices   = exp_config.get("lev3x",        _lev3x_cap_choices)
        _vol_target_range    = exp_config.get("vol_range",    _vol_target_range)
        _lockout_min         = exp_config.get("lockout_min",  _lockout_min)
        _use_calmar          = exp_config.get("use_calmar",   _use_calmar)
        _mini_mode           = exp_config.get("mini",         _mini_mode)
    from stratscout_v2.engine.backtest.etf import load_local_histories
    from stratscout.engine.data.universes import ALL_SYMBOLS
    if _mini_mode:
        syms = MINI_UNIVERSE
    else:
        # SPY needed for low-vol regime detection (realized vol threshold)
        syms = list(dict.fromkeys(ALL_SYMBOLS + ["SPY", "QQQ", "MTUM", "VGT", "XLK", "QQQM", "FPX"]))
    _histories = load_local_histories(syms, "2005-01-01", date.today().isoformat())
    if no_factors:
        _factors = {}
    else:
        try:
            from stratscout.engine.data.factors import load_local_factors
            _factors = load_local_factors()
        except Exception:
            _factors = {}
    label = "no-factors (baseline)" if no_factors else f"{len(_factors)} factors"
    print(f"  [worker {os.getpid()}] ready - {len(_histories)} symbols, {label}", flush=True)


def _run_backtest(params: dict, start: str, end: str,
                  _window_histories: dict | None = None) -> tuple:
    """Returns (total_return_pct, cagr_pct, max_dd_pct, n_trades).

    _window_histories: pre-sliced histories for this exact window. When
    provided, run_etf_backtest skips both disk I/O and the .loc slice.
    """
    from stratscout_v2.engine.backtest.etf import run_etf_backtest
    try:
        r = run_etf_backtest(params, start, end, cash=10_000.0,
                             preloaded_histories=_window_histories if _window_histories
                             else (_histories if _histories else None))
        p = r["perf"]
        trade_df = r.get("trade_df")
        return (
            p.get("total_return_pct", 0),
            p.get("cagr_pct", 0),
            p.get("max_drawdown_pct", 0),
            len(trade_df) if trade_df is not None else 0,
        )
    except Exception:
        return (0.0, 0.0, 0.0, 0)


def _apply_experiment_params(p: dict, rng) -> None:
    """Apply experiment-config controlled params to a param dict in-place."""
    p["lev_3x_cap"] = rng.choice(_lev3x_cap_choices)
    vmin, vmax = _vol_target_range
    if vmax > 0:
        p["vol_target_pct"]      = round(rng.uniform(vmin, vmax), 1)
        p["vol_target_lookback"] = rng.choice([10, 14, 21, 28])
        p["vol_target_adaptive"] = rng.random() > 0.5
    else:
        p["vol_target_pct"]      = 0.0
        p["vol_target_lookback"] = 21
        p["vol_target_adaptive"] = False
    p["stop_loss_lockout_days"] = rng.choice(
        [d for d in [15, 18, 20, 22, 25, 30] if d >= _lockout_min]
    )


def _random_params_gated(exclude: list[str], as_of_date: str | None = None) -> dict:
    """random_params() with inception-date gating for pre-2015 runs.
    When _random_on_pool is set (--random-universe), uses that pool instead of curated pools.
    """
    import random as _rand
    from stratscout_v2.engine.backtest.etf import random_params
    from stratscout.engine.data.universes import MIN_RISK_ON, MIN_RISK_OFF_RISING, MIN_RISK_OFF_FALLING

    # Randomized universe mode — skip curated pool logic entirely
    if _random_on_pool is not None:
        excl = set(exclude or [])
        on_universe      = [s for s in _random_on_pool      if s not in excl]
        rising_universe  = [s for s in _random_rising_pool  if s not in excl]
        falling_universe = [s for s in _random_falling_pool if s not in excl]
        # fall through to pool sampling below
    elif as_of_date is None or as_of_date >= "2015-01-01":
        import random as _rand2
        p = random_params(exclude=exclude)
        p["lev_3x_cap"] = _rand2.choice(_lev3x_cap_choices)
        _apply_experiment_params(p, _rand2)
        return p
    else:
        excl = set(exclude or [])

        def _available(sym: str) -> bool:
            inc = INCEPTION_DATES.get(sym)
            return inc is not None and inc <= as_of_date and sym not in excl

        on_universe      = [s for s in _RISK_ON_POOL_EXTENDED      if _available(s)]
        rising_universe  = [s for s in _RISK_OFF_RISING_POOL_EXTENDED  if _available(s)]
        falling_universe = [s for s in _RISK_OFF_FALLING_POOL_EXTENDED if _available(s)]

    # Ensure minimums
    if len(on_universe)      < MIN_RISK_ON:          on_universe      = ["QQQ", "IWM"]
    if len(rising_universe)  < MIN_RISK_OFF_RISING:  rising_universe  = ["QID"]
    if len(falling_universe) < MIN_RISK_OFF_FALLING: falling_universe = ["GLD", "TLT"]

    n_risk_on = _rand.randint(MIN_RISK_ON, min(3, len(on_universe)))
    on_pool = _rand.sample(on_universe, min(n_risk_on + _rand.randint(0, 2), len(on_universe)))
    if len(on_pool) < MIN_RISK_ON:
        on_pool = on_universe[:MIN_RISK_ON]

    n_rising  = _rand.randint(MIN_RISK_OFF_RISING,  min(3, len(rising_universe)))
    rising_pool = _rand.sample(rising_universe, min(n_rising + _rand.randint(0, 1), len(rising_universe)))

    n_falling = _rand.randint(MIN_RISK_OFF_FALLING, min(5, len(falling_universe)))
    falling_pool = _rand.sample(falling_universe, min(n_falling + _rand.randint(0, 2), len(falling_universe)))

    include_uup = "UUP" in falling_universe and _rand.random() > 0.3

    # Reuse scalar params from standard random_params to stay consistent
    base = random_params(exclude=list(excl))
    base.update({
        "n_risk_on":             n_risk_on,
        "n_risk_off_rising":     n_rising + (1 if include_uup else 0),
        "n_risk_off_falling":    n_falling,
        "risk_on_pool":          on_pool,
        "risk_off_rising_pool":  rising_pool,
        "risk_off_falling_pool": falling_pool,
        "rising_rate_include_uup": include_uup,
    })
    _apply_experiment_params(base, _rand)
    return base


def _make_train_params(t_start: str, t_mid1: str, t_mid2: str, t_end: str,
                       n_trials: int, exclude: list[str],
                       as_of_date: str | None = None) -> tuple[dict | None, float]:
    """Fuzz n_trials on 3 training sub-windows, return (best_params, best_score)."""
    import random
    from stratscout_v2.engine.backtest.etf import random_params, refine_params

    best_score = -999.0
    best_params = None
    top: list[dict] = []

    for _ in range(n_trials):
        if not top or random.random() < 0.5:
            p = _random_params_gated(exclude, as_of_date)
        else:
            base = random.choice(top)
            p = refine_params(base, strength=random.uniform(0.1, 0.4), exclude=exclude)

        r1 = _run_backtest(p, t_start, t_mid1)
        r2 = _run_backtest(p, t_mid1,  t_mid2)
        r3 = _run_backtest(p, t_mid2,  t_end)

        score = _combined_score(
            r1[1], r2[1], r3[1],
            r1[2], r2[2], r3[2],
            r1[3], r2[3], r3[3],
            recency_weight=p.get("recency_weight", 1.5),
        )

        if score > best_score:
            best_score = score
            best_params = p.copy()

        if score > 0:
            p["__score__"] = score
            top.append(p)
            if len(top) > 20:
                top.sort(key=lambda x: x.get("__score__", 0), reverse=True)
                top = top[:20]

    return best_params, best_score


def _make_train_params_fast(t_start: str, t_end: str,
                            n_trials: int, exclude: list[str],
                            as_of_date: str | None = None) -> tuple[dict | None, float]:
    """Single-window version: 3× faster, slightly less overfit-resistant."""
    import random
    from stratscout_v2.engine.backtest.etf import random_params, refine_params

    best_score = -999.0
    best_params = None
    top: list[dict] = []

    for _ in range(n_trials):
        if not top or random.random() < 0.5:
            p = _random_params_gated(exclude, as_of_date)
        else:
            base = random.choice(top)
            p = refine_params(base, strength=random.uniform(0.1, 0.4), exclude=exclude)

        r = _run_backtest(p, t_start, t_end)
        # Mirror the 3-window scoring using the single window for all three slots
        score = _combined_score(r[1], r[1], r[1], r[2], r[2], r[2], r[3], r[3], r[3])

        if score > best_score:
            best_score = score
            best_params = p.copy()

        if score > 0:
            p["__score__"] = score
            top.append(p)
            if len(top) > 20:
                top.sort(key=lambda x: x.get("__score__", 0), reverse=True)
                top = top[:20]

    return best_params, best_score


def _build_optuna_params(
    trial, exclude: list[str],
    hof_bounds: dict | None = None,
    as_of_date: str | None = None,
) -> dict:
    """Define ETF param space for an optuna trial.
    hof_bounds: tightened (lo, hi) per scalar param from HoF history.
    as_of_date: ISO date string — gates pool to symbols with inception <= this date.
                When None, uses the standard 2015+ universe (no gating).
    """
    def _b(key: str, default_lo, default_hi):
        if hof_bounds and key in hof_bounds:
            return hof_bounds[key]
        return default_lo, default_hi

    from stratscout.engine.data.universes import (
        RISK_ON_POOL, RISK_OFF_RISING_POOL, RISK_OFF_FALLING_POOL,
        MIN_RISK_ON, MIN_RISK_OFF_RISING, MIN_RISK_OFF_FALLING,
    )

    # Select base pools — randomized, extended (inception-gated), or curated
    if _random_on_pool is not None:
        base_on      = _random_on_pool
        base_rising  = _random_rising_pool
        base_falling = _random_falling_pool
    else:
        use_extended = as_of_date is not None and as_of_date < "2015-01-01"
        if use_extended:
            base_on      = _RISK_ON_POOL_EXTENDED
            base_rising  = _RISK_OFF_RISING_POOL_EXTENDED
            base_falling = _RISK_OFF_FALLING_POOL_EXTENDED
            def _available(sym: str) -> bool:
                inc = INCEPTION_DATES.get(sym)
                return inc is not None and inc <= as_of_date
            base_on      = [s for s in base_on      if _available(s)]
            base_rising  = [s for s in base_rising  if _available(s)]
            base_falling = [s for s in base_falling if _available(s)]
        else:
            base_on      = RISK_ON_POOL
            base_rising  = RISK_OFF_RISING_POOL
            base_falling = RISK_OFF_FALLING_POOL

    excl = set(exclude or [])
    if _mini_mode:
        mini_set = set(MINI_UNIVERSE)
        on_universe      = [s for s in base_on      if s not in excl and s in mini_set]
        rising_universe  = [s for s in base_rising  if s not in excl and s in mini_set]
        falling_universe = [s for s in base_falling if s not in excl and s in mini_set]
        if len(on_universe)      < MIN_RISK_ON:          on_universe      = ["QQQ", "SPY"]
        if len(rising_universe)  < MIN_RISK_OFF_RISING:  rising_universe  = ["QID"]
        if len(falling_universe) < MIN_RISK_OFF_FALLING: falling_universe = ["GLD", "TLT"]
    else:
        on_universe      = [s for s in base_on      if s not in excl]
        rising_universe  = [s for s in base_rising  if s not in excl]
        falling_universe = [s for s in base_falling if s not in excl]

    # Pool membership — binary inclusion per symbol, enforce minimums
    on_pool = [s for s in on_universe if trial.suggest_categorical(f"on_{s}", [True, False])]
    if len(on_pool) < MIN_RISK_ON:
        on_pool = on_universe[:MIN_RISK_ON]

    rising_pool = [s for s in rising_universe if trial.suggest_categorical(f"rise_{s}", [True, False])]
    if len(rising_pool) < MIN_RISK_OFF_RISING:
        rising_pool = rising_universe[:MIN_RISK_OFF_RISING]

    falling_pool = [s for s in falling_universe if trial.suggest_categorical(f"fall_{s}", [True, False])]
    if len(falling_pool) < MIN_RISK_OFF_FALLING:
        falling_pool = falling_universe[:MIN_RISK_OFF_FALLING]

    include_uup = trial.suggest_categorical("rising_rate_include_uup", [True, False])
    n_risk_on = trial.suggest_int("n_risk_on", MIN_RISK_ON, min(3, len(on_pool)))
    n_rising  = trial.suggest_int("n_risk_off_rising", MIN_RISK_OFF_RISING, min(3, len(rising_pool)))
    n_falling = trial.suggest_int("n_risk_off_falling", MIN_RISK_OFF_FALLING, min(5, len(falling_pool)))

    return {
        "agg_bil_lookback":      trial.suggest_int("agg_bil_lookback",    *_b("agg_bil_lookback",    63,  126)),
        "tlt_bil_lookback":      trial.suggest_int("tlt_bil_lookback",    *_b("tlt_bil_lookback",    6,   18)),
        "risk_on_rsi_window":    trial.suggest_int("risk_on_rsi_window",  *_b("risk_on_rsi_window",  16,  29)),
        "risk_off_rsi_window":   trial.suggest_int("risk_off_rsi_window", *_b("risk_off_rsi_window", 7,   23)),
        "risk_on_rsi_direction": trial.suggest_categorical("risk_on_rsi_direction", ["lowest", "highest"]),
        "risk_off_rsi_direction":trial.suggest_categorical("risk_off_rsi_direction", ["lowest", "highest"]),
        "n_risk_on":             n_risk_on,
        "n_risk_off_rising":     n_rising + (1 if include_uup else 0),
        "n_risk_off_falling":    n_falling,
        "risk_on_pool":          on_pool,
        "risk_off_rising_pool":  rising_pool,
        "risk_off_falling_pool": falling_pool,
        "rising_rate_include_uup": include_uup,
        "min_hold_days":         trial.suggest_int("min_hold_days",       *_b("min_hold_days",       4,   11)),
        "vol_weight_window":     trial.suggest_categorical("vol_weight_window", [0, 0, 0, 0, 5, 10]),
        "vol_score_weight":      trial.suggest_categorical("vol_score_weight", [0.0, 0.0, 0.0, 0.1, 0.2]),
        "vol_score_window":      trial.suggest_categorical("vol_score_window", [10, 15, 20, 30]),
        "vol_surge_cap":         trial.suggest_categorical("vol_surge_cap", [2.0, 3.0, 4.0, 5.0]),
        "ema_weight":            trial.suggest_categorical("ema_weight", [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5]),
        "ema_fast":              trial.suggest_categorical("ema_fast", [5, 8, 10, 12, 15, 20]),
        "ema_slow":              trial.suggest_categorical("ema_slow", [20, 30, 40, 50, 60, 80, 100]),
        "sector_diverse":        False,
        "combo_momentum_lookback": trial.suggest_int("combo_momentum_lookback", *_b("combo_momentum_lookback", 14,  42)),
        "combo_vol_lookback":      trial.suggest_int("combo_vol_lookback",      *_b("combo_vol_lookback",      10,  30)),
        "combo_alpha":             trial.suggest_float("combo_alpha",            *_b("combo_alpha",            0.1, 0.9)),
        "combo_max_weight":        trial.suggest_float("combo_max_weight",       *_b("combo_max_weight",       0.4, 0.9)),
        "stop_loss_pct":           trial.suggest_float("stop_loss_pct",          *_b("stop_loss_pct",          8.0, 19.0)),
        "stop_loss_lockout_days":  trial.suggest_int("stop_loss_lockout_days",   *_b("stop_loss_lockout_days", 16,  29)),
        "vol_target_pct":          trial.suggest_float("vol_target_pct",         *_b("vol_target_pct",         0.0, 10.0)),
        "vol_target_lookback":     trial.suggest_int("vol_target_lookback",      *_b("vol_target_lookback",    12,  28)),
        "low_vol_threshold":       trial.suggest_float("low_vol_threshold",      *_b("low_vol_threshold",      0.0, 15.0)),
        "score_normalize_window":  trial.suggest_int("score_normalize_window",   *_b("score_normalize_window", 0,   20)),
        "recency_weight":          trial.suggest_float("recency_weight",          *_b("recency_weight",         1.0, 4.0)),
        "vol_target_adaptive":     trial.suggest_categorical("vol_target_adaptive", [True, False]),
        "pool_sharpe_filter":      trial.suggest_categorical("pool_sharpe_filter",  [True, False]),
        "lev_3x_cap":              trial.suggest_categorical("lev_3x_cap", _lev3x_cap_choices),
    }


def _apply_factor_overrides(params: dict, month_start: str) -> dict:
    """Factor overrides disabled — pure price signal outperforms all tested overrides."""
    return params


# Scalar param keys that map directly to Optuna suggest_* names — used for HoF seeding.
_OPTUNA_SCALAR_KEYS = {
    "agg_bil_lookback", "tlt_bil_lookback", "risk_on_rsi_window", "risk_off_rsi_window",
    "risk_on_rsi_direction", "risk_off_rsi_direction", "n_risk_on", "n_risk_off_rising",
    "n_risk_off_falling", "rising_rate_include_uup", "min_hold_days",
    "vol_weight_window", "vol_score_weight", "vol_score_window", "vol_surge_cap",
    "ema_weight", "ema_fast", "ema_slow",
    "fg_fear_threshold", "fg_greed_threshold", "fomc_caution_days",
    "opex_caution_days", "layoffs_caution_zscore",
    "combo_momentum_lookback", "combo_vol_lookback", "combo_alpha", "combo_max_weight",
    "stop_loss_pct", "stop_loss_lockout_days", "vol_target_pct", "vol_target_lookback",
    "low_vol_threshold", "score_normalize_window", "recency_weight",
    "vol_target_adaptive", "pool_sharpe_filter",
}


def _run_month_optuna(args_tuple) -> dict:
    """Bayesian (TPE) variant of _run_month. Uses optuna for sample-efficient search.

    args_tuple positions:
      0  month_start, 1  month_end, 2  t_start, 3  t_mid1, 4  t_mid2,
      5  n_trials,    6  exclude,   7  fast_mode, 8  workers,
      9  seed_list (optional list[dict] from HoF)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    month_start = args_tuple[0]
    month_end   = args_tuple[1]
    t_start     = args_tuple[2]
    t_mid1      = args_tuple[3]
    t_mid2      = args_tuple[4]
    n_trials    = args_tuple[5]
    exclude     = args_tuple[6]
    fast_mode   = args_tuple[7] if len(args_tuple) > 7 else False
    workers     = args_tuple[8] if len(args_tuple) > 8 else 1
    seed_list        = args_tuple[9]  if len(args_tuple) > 9  else []
    hof_bounds       = args_tuple[10] if len(args_tuple) > 10 else {}
    min_train_score  = args_tuple[11] if len(args_tuple) > 11 else 0.0

    # Pre-slice histories once per training window so each trial reuses the
    # same sliced dict instead of re-slicing _histories on every call.
    # Uses a large warmup buffer so any lookback param fits within the slice.
    _MAX_WARMUP_DAYS = 230  # 107 agg_bil_lookback * 2 + buffer
    def _preslice(start: str, end: str) -> dict | None:
        if not _histories:
            return None
        import pandas as _pd
        s = _pd.Timestamp(start, tz="UTC") - _pd.Timedelta(days=_MAX_WARMUP_DAYS)
        e = _pd.Timestamp(end, tz="UTC")
        return {sym: df.loc[s:e] for sym, df in _histories.items()}

    _h_w1   = _preslice(t_start,     t_mid1)
    _h_w2   = _preslice(t_mid1,      t_mid2)
    _h_w3   = _preslice(t_mid2,      month_start)
    _h_full = _preslice(t_start,     month_start)
    _h_val  = _preslice(month_start, month_end)

    def objective(trial):
        p = _build_optuna_params(trial, exclude, hof_bounds, as_of_date=month_start)
        rw = p.get("recency_weight", 1.5)
        if fast_mode:
            r = _run_backtest(p, t_start, month_start, _h_full)
            return _combined_score(r[1], r[1], r[1], r[2], r[2], r[2], r[3], r[3], r[3], recency_weight=rw)
        else:
            r1 = _run_backtest(p, t_start, t_mid1, _h_w1)
            # Intermediate report after w1 — Optuna prunes if clearly below median
            partial = _combined_score(r1[1], r1[1], r1[1], r1[2], r1[2], r1[2], r1[3], r1[3], r1[3], recency_weight=rw)
            trial.report(partial, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            r2 = _run_backtest(p, t_mid1, t_mid2, _h_w2)
            r3 = _run_backtest(p, t_mid2, month_start, _h_w3)
            return _combined_score(r1[1], r2[1], r3[1], r1[2], r2[2], r3[2], r1[3], r2[3], r3[3], recency_weight=rw)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=min(20, n_trials // 3)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    # Enqueue HoF seeds — evaluated first, then Optuna explores from there.
    n_seeded = 0
    for seed_params in (seed_list or []):
        seed_scalar = {k: v for k, v in seed_params.items() if k in _OPTUNA_SCALAR_KEYS}
        if seed_scalar:
            try:
                study.enqueue_trial(seed_scalar)
                n_seeded += 1
            except Exception:
                pass

    import os as _os, time as _time
    seed_note = f" ({n_seeded} seeds)" if n_seeded else ""
    print(f"  [Bayesian pid={_os.getpid()}] {month_start} — {n_trials} trials, {workers} jobs{seed_note}", flush=True)
    _t0 = _time.monotonic()
    study.optimize(objective, n_trials=n_trials, n_jobs=workers, show_progress_bar=False, catch=(ValueError,))
    print(f"  [Bayesian pid={_os.getpid()}] {month_start} done in {_time.monotonic()-_t0:.0f}s best={study.best_value:.3f}", flush=True)

    best = study.best_trial
    best_params = _build_optuna_params(best, exclude) if best.params else None
    # Reconstruct params from best trial's suggest values
    class _FakeTrial:
        def __init__(self, params): self._p = params
        def suggest_int(self, name, *a, **kw): return self._p.get(name, a[0])
        def suggest_float(self, name, *a, **kw): return self._p.get(name, a[0])
        def suggest_categorical(self, name, choices): return self._p.get(name, choices[0])
    best_params = _build_optuna_params(_FakeTrial(best.params), exclude, as_of_date=month_start)
    train_score = best.value if best.value is not None else -999.0

    if best_params is None:
        return {
            "month": month_start, "train_score": -999.0,
            "val_return": 0.0, "val_cagr": 0.0, "val_dd": 0.0,
            "val_trades": 0, "positive": False, "params": None,
        }

    best_params = _apply_factor_overrides(best_params, month_start)

    if min_train_score > 0 and train_score < min_train_score:
        return {
            "month": month_start, "train_score": train_score,
            "val_return": 0.0, "val_cagr": 0.0, "val_dd": 0.0,
            "val_trades": 0, "positive": False, "params": json.dumps(best_params, sort_keys=True, default=str),
        }

    val = _run_backtest(best_params, month_start, month_end, _h_val)
    return {
        "month":       month_start,
        "train_score": train_score,
        "val_return":  val[0],
        "val_cagr":    val[1],
        "val_dd":      val[2],
        "val_trades":  val[3],
        "positive":    val[0] > 0,
        "params":      json.dumps(best_params, sort_keys=True, default=str),
    }


def _run_gp_backtest(strategy: dict, start: str, end: str) -> tuple:
    """Returns (total_return_pct, cagr_pct, max_dd_pct, n_trades) for a GP strategy."""
    from stratscout.engine.fuzzers.gp_backtest import run_gp_backtest
    try:
        r = run_gp_backtest(strategy, _histories, start, end, cash=10_000.0)
        p = r["perf"]
        return (
            p.get("total_return_pct", 0),
            p.get("cagr_pct", 0),
            p.get("max_drawdown_pct", 0),
            r["n_trades"],
        )
    except Exception:
        return (0.0, 0.0, 0.0, 0)


def _run_month_gp(args_tuple) -> dict:
    """GP-evolution variant of _run_month. args: (month_start, month_end, t_start,
    t_mid1, t_mid2, population_size, n_generations, exclude)."""
    month_start, month_end, t_start, t_mid1, t_mid2 = args_tuple[:5]
    population_size = int(args_tuple[5]) if len(args_tuple) > 5 else 100
    n_generations   = int(args_tuple[6]) if len(args_tuple) > 6 else 30

    from stratscout.engine.fuzzers.gp_evolution import evolve_in_worker
    best_strategy, train_score = evolve_in_worker(
        t_start, t_mid1, t_mid2, month_start,
        population_size=population_size,
        n_generations=n_generations,
    )

    if best_strategy is None:
        return {
            "month": month_start, "train_score": -999.0,
            "val_return": 0.0, "val_cagr": 0.0, "val_dd": 0.0,
            "val_trades": 0, "positive": False, "params": None,
        }

    val = _run_gp_backtest(best_strategy, month_start, month_end)
    return {
        "month":       month_start,
        "train_score": train_score,
        "val_return":  val[0],
        "val_cagr":    val[1],
        "val_dd":      val[2],
        "val_trades":  val[3],
        "positive":    val[0] > 0,
        "params":      json.dumps(best_strategy, sort_keys=True, default=str),
    }


def _run_month(args_tuple) -> dict:
    import os as _os, time as _time
    fast_mode       = args_tuple[7] if len(args_tuple) > 7 else False
    min_train_score = args_tuple[8] if len(args_tuple) > 8 else 0.0
    month_start, month_end, t_start, t_mid1, t_mid2, n_trials, exclude = args_tuple[:7]
    print(f"  [Random pid={_os.getpid()}] {month_start} — {n_trials} trials", flush=True)
    _t0 = _time.monotonic()

    if fast_mode:
        best_params, train_score = _make_train_params_fast(
            t_start, month_start, n_trials, exclude, as_of_date=month_start
        )
    else:
        best_params, train_score = _make_train_params(
            t_start, t_mid1, t_mid2, month_start, n_trials, exclude, as_of_date=month_start
        )

    if best_params is None:
        return {
            "month": month_start, "train_score": -999.0,
            "val_return": 0.0, "val_cagr": 0.0, "val_dd": 0.0,
            "val_trades": 0, "positive": False, "params": None,
        }

    if min_train_score > 0 and train_score < min_train_score:
        return {
            "month": month_start, "train_score": train_score,
            "val_return": 0.0, "val_cagr": 0.0, "val_dd": 0.0,
            "val_trades": 0, "positive": False, "params": json.dumps(best_params, sort_keys=True, default=str),
        }

    val = _run_backtest(best_params, month_start, month_end)
    return {
        "month":       month_start,
        "train_score": train_score,
        "val_return":  val[0],
        "val_cagr":    val[1],
        "val_dd":      val[2],
        "val_trades":  val[3],
        "positive":    val[0] > 0,
        "params":      json.dumps(best_params, sort_keys=True, default=str),
    }


# ── DB ────────────────────────────────────────────────────────────────────────

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            month       TEXT UNIQUE,
            train_score REAL,
            val_return  REAL,
            val_cagr    REAL,
            val_dd      REAL,
            val_trades  INTEGER,
            positive    INTEGER,
            params      TEXT
        )
    """)
    con.commit()
    con.close()


def _save(row: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR IGNORE INTO results "
        "(month, train_score, val_return, val_cagr, val_dd, val_trades, positive, params) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (row["month"], row["train_score"], row["val_return"], row["val_cagr"],
         row["val_dd"], row["val_trades"], int(row["positive"]), row["params"])
    )
    con.commit()
    con.close()


def _spy_return(month_start: str, month_end: str) -> float:
    try:
        spy = pd.read_feather("data/daily/SPY.feather")
        spy["date"] = pd.to_datetime(spy["date"], utc=True)
        spy = spy.set_index("date").sort_index()
        s = pd.Timestamp(month_start, tz="UTC")
        e = pd.Timestamp(month_end,   tz="UTC")
        sub = spy.loc[s:e, "close"]
        if len(sub) < 2:
            return 0.0
        return (sub.iloc[-1] / sub.iloc[0] - 1) * 100
    except Exception:
        return 0.0


def _print_summary():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT month, train_score, val_return, val_cagr, val_dd, val_trades, positive "
        "FROM results ORDER BY month"
    ).fetchall()
    con.close()
    if not rows:
        print("No results yet.")
        return

    print(f"\n{'Month':<12} {'SPY%':>6} {'TrainScore':>10} {'ValReturn':>10} {'ValDD':>8} {'Trades':>7} {'Verdict':>12}")
    print("-" * 80)

    hits = active = missed_up = correct_out = losses = 0
    monthly_returns = []

    # Derive period end from the next row's start (works for any period length)
    period_ends = [rows[i+1][0] for i in range(len(rows)-1)] + [
        (date.fromisoformat(rows[-1][0]) + relativedelta(months=1)).isoformat()
    ]

    for r, month_end in zip(rows, period_ends):
        spy_ret   = _spy_return(r[0], month_end)
        market_up = spy_ret > 1.0

        if r[5] == 0:
            verdict = "MISSED-UP" if market_up else "cash-ok"
            if market_up:
                missed_up += 1
            else:
                correct_out += 1
            monthly_returns.append(0.0)
        elif r[6]:
            verdict = "HIT"
            hits += 1; active += 1
            monthly_returns.append(r[2])
        else:
            verdict = "LOSS"
            losses += 1; active += 1
            monthly_returns.append(r[2])

        print(f"{r[0]:<12} {spy_ret:>+6.1f}% {r[1]:>+10.1f} {r[2]:>+10.1f}% {r[4]:>+8.1f}% {r[5]:>7} {verdict:>12}")

    total = len(rows)
    avg_ret = sum(monthly_returns) / len(monthly_returns) if monthly_returns else 0

    print(f"\nSummary ({total} months):")
    print(f"  HIT (traded, positive):            {hits}")
    print(f"  LOSS (traded, negative):           {losses}")
    print(f"  MISSED-UP (no trade, mkt up):      {missed_up}")
    print(f"  Cash-OK (no trade, mkt flat/down): {correct_out}")
    print(f"  Avg monthly return (all months):   {avg_ret:+.2f}%")

    if active > 0:
        print(f"\n  Active win rate: {hits}/{active} = {hits/active*100:.0f}%")
    up_months = hits + losses + missed_up
    if up_months > 0:
        print(f"  Up-month capture: {hits}/{up_months} = {hits/up_months*100:.0f}%")

    # Compound the monthly returns to show equity curve
    equity = 10_000.0
    for ret in monthly_returns:
        equity *= (1 + ret / 100)
    print(f"\n  $10,000 -> ${equity:,.0f} out-of-sample ({total} months)")
    spy_equity = 10_000.0
    for r, month_end in zip(rows, period_ends):
        spy_equity *= (1 + _spy_return(r[0], month_end) / 100)
    print(f"  SPY buy-hold ->  ${spy_equity:,.0f} same period")


def _compute_stratified_months_deterministic(n_per_regime: int = 8, replicate: int = 0,
                                              grouped: bool = False) -> list[str]:
    """Compute deterministic stratified months (n_per_regime each from bull /
    rising_rates / falling_rates) for fast testing.

    Groups all historical months by regime, then draws n_per_regime months from
    each using a seeded RNG keyed on (regime, replicate). Deterministic: same months
    every time for a given replicate (seed depends only on regime + replicate).

    grouped=False -> flat list sorted by date.
    grouped=True  -> regime-grouped: bull (sorted) + rising_rates (sorted) +
                     falling_rates (sorted). This defines the 1..24 indexing that
                     FAST_TEST_ORDERS permute (1-8 bull, 9-16 rising, 17-24 falling).
    Defaults to 8 per regime -> 24 months total.
    """
    import random
    from stratscout.engine.backtest.etf import load_local_histories
    from stratscout.engine.data.universes import ANCHORS

    # Load regime signals for all historical data
    try:
        histories = load_local_histories(["AGG", "BIL", "TLT"], "2018-01-01", "2026-06-01")
    except Exception as e:
        print(f"Warning: could not load regime histories: {e}. Using full date range instead.")
        return []

    if not histories or len(histories) < 3:
        return []

    # Helper to compute cumulative return over last N days
    def _cumret(series, lookback_days, as_of_date_str):
        """Cumulative return over lookback_days ending on as_of_date."""
        try:
            as_of = pd.Timestamp(as_of_date_str, tz='UTC')
            # pandas 2.x: get_loc dropped method=; use get_indexer for nearest match.
            idx = int(series.index.get_indexer([as_of], method='nearest')[0])
            start_idx = max(0, idx - lookback_days)
            if idx <= start_idx:
                return 0.0
            return float(series.iloc[idx] / series.iloc[start_idx] - 1)
        except Exception:
            return 0.0

    # Classify regime for a given month-end date
    def _classify_regime(month_end_str: str) -> str:
        """Bull / rising_rates / falling_rates based on AGG/BIL and TLT/BIL ratios."""
        agg = histories["AGG"]["close"]
        bil = histories["BIL"]["close"]
        tlt = histories["TLT"]["close"]

        agg_bil_lookback, tlt_bil_lookback = 60, 20

        if _cumret(agg, agg_bil_lookback, month_end_str) > _cumret(bil, agg_bil_lookback, month_end_str):
            return "bull"
        elif _cumret(tlt, tlt_bil_lookback, month_end_str) < _cumret(bil, tlt_bil_lookback, month_end_str):
            return "rising_rates"
        else:
            return "falling_rates"

    # Build month list from 2018-present, classify each
    month_regimes = {}
    m = date(2018, 1, 1)
    end = date(2026, 6, 1)

    while m < end:
        m_end = m + relativedelta(months=1)
        # Use last day of month as the regime decision point
        m_last_day = m_end - relativedelta(days=1)
        regime = _classify_regime(m_last_day.isoformat())
        month_regimes[m.isoformat()] = regime
        m = m_end

    # Group by regime, sort chronologically
    by_regime = {"bull": [], "rising_rates": [], "falling_rates": []}
    for month_str in sorted(month_regimes.keys()):
        regime = month_regimes[month_str]
        by_regime[regime].append(month_str)

    # Sample n_per_regime months from each regime with a seeded RNG keyed on
    # (regime, replicate). Deterministic per replicate.
    grouped_months: list[str] = []   # regime order: bull, rising, falling (sorted within)
    for regime in ["bull", "rising_rates", "falling_rates"]:
        months = by_regime[regime]
        if not months:
            print(f"  [Fast-Test] WARNING: regime '{regime}' has 0 months — "
                  f"stratified sample will be unbalanced.")
            continue

        rng = random.Random(hash((regime, replicate)) & 0xFFFFFFFF)
        k = min(n_per_regime, len(months))
        grouped_months.extend(sorted(rng.sample(months, k)))

    return grouped_months if grouped else sorted(grouped_months)


# ── Fast-test benchmark ─────────────────────────────────────────────────────────

def _chain_maxdd(returns: list[float]) -> float:
    """Max drawdown (%) of the equity curve formed by chaining `returns` in order."""
    nav = 10_000.0
    peak = nav
    maxdd = 0.0
    for r in returns:
        nav *= (1 + r / 100)
        peak = max(peak, nav)
        maxdd = max(maxdd, (peak - nav) / peak * 100)
    return maxdd


def _compound_nav(returns: list[float]) -> float:
    nav = 10_000.0
    for r in returns:
        nav *= (1 + r / 100)
    return nav


def _proxy_cagr(nav: float, n: int) -> float:
    # Proxy: treat each stratified period as ~1 month (12/yr). Months aren't
    # contiguous, so CAGR is a *relative* figure for ranking configs, not a real curve.
    return ((nav / 10_000.0) ** (12.0 / n) - 1) * 100 if n else 0.0


def _print_fast_test_aggregate(suite: dict) -> None:
    rows = suite["rows"]
    if not rows:
        return
    import statistics as _stats
    elastic = suite["elastic"]
    R, K = suite["n_runs"], suite["K"]

    def stats(vals):
        std = _stats.pstdev(vals) if len(vals) > 1 else 0.0
        return _stats.mean(vals), std, min(vals), max(vals)

    unit = "pass(es)" if elastic else "trial-run(s)"
    print("\n" + "=" * 80)
    print(f"Fast-Test summary — {R} trial-run(s) × {K} ordering(s) on 24 fixed stratified months"
          + ("  [ELASTIC: each ordering is its own consecutive pass]" if elastic else ""))
    print("=" * 80)
    print(f"{'Pass' if elastic else 'Run':<8}{'NAV':>12}{'CAGR%':>8}{'AvgRet%':>9}{'Hit%':>6}"
          f"{'DDmed%':>8}{'DDworst%':>9}{'min':>7}")
    print("-" * 80)
    for r in rows:
        print(f"{r['label']:<8}{r['nav']:>12,.0f}{r['cagr']:>+8.1f}{r['avg_ret']:>+9.2f}"
              f"{r['hit_rate']:>6.0f}{r['dd_med']:>8.1f}{r['dd_worst']:>9.1f}"
              f"{r.get('elapsed_min', 0):>7.1f}")
    print("-" * 80)

    cm, cs, clo, chi = stats(suite["opt_cagrs"])
    nm, ns, _, _     = stats(suite["opt_navs"])
    am, *_           = stats(suite["opt_avg_rets"])
    all_dds          = suite["all_maxdds"]
    dmean, dstd, dlo, dhi = stats(all_dds)
    n_opt = len(suite["opt_cagrs"])

    cagr_src = (f"order+trial variance, {n_opt} passes" if elastic
                else f"trial-noise, {R} run{'s' if R > 1 else ''}; order-invariant")
    print(f"  CAGR  ({cagr_src}):")
    print(f"          mean {cm:+.1f}%   std {cs:.1f}pp   [{clo:+.1f}, {chi:+.1f}]")
    print(f"  NAV   : mean ${nm:,.0f}   std ${ns:,.0f}")
    print(f"  MaxDD (path risk, {len(all_dds)} passes):  "
          f"median {_stats.median(all_dds):.1f}%   worst {dhi:.1f}%   best {dlo:.1f}%   "
          f"mean {dmean:.1f}% (std {dstd:.1f}pp)")
    print(f"  AvgRet: mean {am:+.2f}%")
    if K > 1:
        per_order = suite["per_order_maxdds"]
        cells = "  ".join(f"o{j + 1} {_stats.mean(per_order[j]):.1f}%" for j in range(K))
        print(f"  Per-ordering mean MaxDD:  {cells}")
    print("\n  Goal: CAGR high & low-std; MaxDD low — especially worst-case path.")
    print("=" * 80)


def _run_elastic_pass(spec: dict) -> dict:
    """Run ONE elastic-martingale pass: the 24 months executed sequentially in the
    given ordering, with trials scaled on consecutive losses (budget walks the order).

    Top-level + picklable so independent passes run concurrently in a process pool —
    the sequentiality is *within* a pass; passes themselves are independent. Writes its
    own isolated DB and returns the pass's returns/metrics.
    """
    global DB_PATH
    import time as _t
    from dateutil.relativedelta import relativedelta as _rd
    from datetime import timedelta as _td

    order   = spec["order"]
    canon   = spec["canonical"]
    base    = spec["trials"]
    m1, m2  = spec["mult1"], spec["mult2"]
    tm      = spec["train_months"]
    pdays   = spec["period_days"]
    exclude, fast, mts = spec["exclude"], spec["fast"], spec["min_train_score"]

    DB_PATH = Path(spec["db"])
    try:
        if DB_PATH.exists():
            DB_PATH.unlink()
    except OSError:
        pass
    _init_db()

    def _mk(ms: str, trials: int) -> tuple:
        m     = date.fromisoformat(ms)
        m_end = (m + _td(days=pdays)) if pdays else (m + _rd(months=1))
        return (
            m.isoformat(), m_end.isoformat(),
            (m - _rd(months=tm)).isoformat(),
            (m - _rd(months=(tm * 2) // 3)).isoformat(),
            (m - _rd(months=tm // 3)).isoformat(),
            trials, exclude, fast, mts,
        )

    t0 = _t.perf_counter()
    recent: list[float] = []          # last 2 OOS returns, in pass order
    seq_returns: list[float] = []      # returns in pass (ordering) order
    ret_by_month: dict[str, float] = {}
    hits = 0
    budgets: list[int] = []
    for idx in order:
        ms = canon[idx - 1]
        consec = 0
        for r in reversed(recent):
            if r < 0:
                consec += 1
            else:
                break
        eff = base * (m2 if consec >= 2 else m1 if consec == 1 else 1)
        result = _run_month(_mk(ms, eff))
        _save(result)
        seq_returns.append(result["val_return"])
        ret_by_month[ms] = result["val_return"]
        hits += 1 if result["positive"] else 0
        recent.append(result["val_return"])
        recent = recent[-2:]
        budgets.append(eff // base)
    return {
        "label": spec["label"], "order_index": spec["order_index"], "repeat": spec["repeat"],
        "seq_returns": seq_returns, "ret_by_month": ret_by_month, "hits": hits,
        "budgets": budgets, "elapsed_min": (_t.perf_counter() - t0) / 60,
    }


def _run_fast_test(args) -> None:
    """Self-contained fast benchmark: minimal 8-symbol universe over a FIXED set of 24
    stratified months (8 each from bull / rising / falling), pooled across months.

    --repeat R re-runs the optimization R times (fresh random trials, same months) to
    measure CAGR variance from trial noise. Within each run the 24 monthly returns are
    chained under every FAST_TEST_ORDERS sequence to measure MaxDD path risk (CAGR is
    order-invariant, so only drawdown changes). Both knobs are deterministic across
    invocations, so any two ideas are screened on identical months + identical orderings.
    """
    global DB_PATH, _random_on_pool, _random_rising_pool, _random_falling_pool

    import time as _time_mod
    from datetime import timedelta as _timedelta

    # Minimal universe (FPX = IPO momentum for extra regime variation).
    _random_on_pool      = ["TQQQ", "UPRO", "FPX"]
    _random_rising_pool  = ["QID"]
    _random_falling_pool = ["GLD"]

    tm           = args.train_months
    _period_days = args.period_days
    n_workers    = max(1, args.workers)
    n_runs       = max(1, args.repeat)

    # Fixed, regime-grouped 24-month set: index 1-8 bull, 9-16 rising, 17-24 falling.
    # This is the canonical order that FAST_TEST_ORDERS permute.
    canonical = _compute_stratified_months_deterministic(8, replicate=0, grouped=True)
    if not canonical:
        print("  [Fast-Test] ERROR: no stratified months computed; aborting.")
        return
    if len(canonical) != 24:
        print(f"  [Fast-Test] WARNING: expected 24 months, got {len(canonical)} — "
              f"fixed orderings disabled, using single chained MaxDD.")

    # Config passed to every spawned worker — without this, Windows spawn would reset
    # the universe + experiment flags (lockout/lev3x/vol) to module defaults.
    exp_config = {
        "on_pool": _random_on_pool, "rising_pool": _random_rising_pool,
        "falling_pool": _random_falling_pool,
        "lev3x": _lev3x_cap_choices, "vol_range": _vol_target_range,
        "lockout_min": _lockout_min,
        "use_calmar": _use_calmar, "mini": False,
    }

    # Helper: build a validation-period tuple for one month at a given trial count.
    def _mk_period(ms: str, trials: int) -> tuple:
        m       = date.fromisoformat(ms)
        m_end   = (m + _timedelta(days=_period_days)) if _period_days else (m + relativedelta(months=1))
        t_mid2  = m - relativedelta(months=tm // 3)
        t_mid1  = m - relativedelta(months=(tm * 2) // 3)
        t_start = m - relativedelta(months=tm)
        return (
            m.isoformat(), m_end.isoformat(),
            t_start.isoformat(), t_mid1.isoformat(), t_mid2.isoformat(),
            trials, args.exclude, args.fast, args.min_train_score,
        )

    # Elastic martingale scales trials on consecutive losses. Each ordering is a
    # CONSECUTIVE PASS: the 24 months are run sequentially IN THAT ORDER, and the
    # consecutive-loss counter walks the ordering (so the same month gets different
    # trial budgets — and different returns — in different orderings). Requires
    # sequential execution; the parallel pool is used only for the fixed-trial baseline.
    # NOTE: months are non-contiguous stratified samples, so "consecutive losses" is a
    # weaker proxy here than in a full 218-period run — treat as a first-pass screen.
    elastic = args.elastic_martingale
    mult1, mult2 = 2, 4
    if elastic and args.elastic_multipliers:
        try:
            mult1, mult2 = (int(x) for x in args.elastic_multipliers.split(","))
        except (ValueError, TypeError):
            print(f"  WARNING: bad --elastic-multipliers '{args.elastic_multipliers}', using 2,4")

    K = len(FAST_TEST_ORDERS)
    ts   = _time_mod.strftime("%Y%m%d_%H%M%S")
    base = Path(args.db).with_suffix("") if args.db else Path(f"fast_test_{ts}")

    print(f"\n[Fast-Test] universe={FAST_TEST_UNIVERSE}")
    print(f"  Risk-on {_random_on_pool} | Rising {_random_rising_pool} | Falling {_random_falling_pool}")
    print(f"  24 fixed months | trials/month={args.trials} | workers={n_workers} | "
          f"trial-runs={n_runs} | orderings={K} | "
          f"period={'%dd' % _period_days if _period_days else 'calendar-month'}")
    print(f"  [Experiment] lev3x={_lev3x_cap_choices} vol={_vol_target_range} "
          f"lockout_min={_lockout_min}"
          + (f" | ELASTIC x{mult1}/x{mult2}: each ordering = consecutive pass "
             f"({n_runs}×{K}={n_runs * K} sequential passes)" if elastic else ""))

    suite = {
        "elastic": elastic, "n_runs": n_runs, "K": K,
        "rows": [], "opt_cagrs": [], "opt_navs": [], "opt_avg_rets": [],
        "all_maxdds": [], "per_order_maxdds": [[] for _ in range(K)],
    }

    def _record_opt(returns_canonical: list[float], hits: int):
        nav = _compound_nav(returns_canonical)
        suite["opt_cagrs"].append(_proxy_cagr(nav, len(returns_canonical)))
        suite["opt_navs"].append(nav)
        suite["opt_avg_rets"].append(sum(returns_canonical) / len(returns_canonical))
        return nav

    if elastic:
        # Each (repeat, ordering) is an independent consecutive pass. Passes are
        # mutually independent, so run them CONCURRENTLY across the pool — only the
        # 24 months *within* a pass are sequential. Big speedup, zero correctness change.
        specs = [
            {
                "label": f"r{k + 1}.o{j + 1}", "order_index": j, "repeat": k,
                "order": order, "canonical": canonical,
                "trials": args.trials, "mult1": mult1, "mult2": mult2,
                "train_months": tm, "period_days": _period_days,
                "exclude": args.exclude, "fast": args.fast, "min_train_score": args.min_train_score,
                "db": f"{base}_r{k + 1}_o{j + 1}.db",
            }
            for k in range(n_runs)
            for j, order in enumerate(FAST_TEST_ORDERS)
        ]
        print(f"\n[Fast-Test] launching {len(specs)} elastic passes on {n_workers} workers "
              f"(each pass = 24 months sequential in its ordering)...", flush=True)
        with mp.Pool(processes=n_workers, initializer=_worker_init,
                     initargs=(args.no_factors, exp_config)) as pool:
            for pr in pool.imap_unordered(_run_elastic_pass, specs):
                returns_canon = [pr["ret_by_month"][ms] for ms in canonical]
                nav = _record_opt(returns_canon, pr["hits"])
                dd = _chain_maxdd(pr["seq_returns"])           # MaxDD in pass order
                suite["all_maxdds"].append(dd)
                suite["per_order_maxdds"][pr["order_index"]].append(dd)
                bc = {b: pr["budgets"].count(b) for b in sorted(set(pr["budgets"]))}
                budget_str = " ".join(f"{b}x×{c}" for b, c in bc.items())
                suite["rows"].append({
                    "label": pr["label"], "nav": nav, "_k": (pr["repeat"], pr["order_index"]),
                    "cagr": _proxy_cagr(nav, 24), "avg_ret": sum(pr["seq_returns"]) / 24,
                    "hit_rate": pr["hits"] / 24 * 100, "dd_med": dd, "dd_worst": dd,
                    "elapsed_min": pr["elapsed_min"],
                })
                print(f"  [pass {pr['label']}] CAGR {_proxy_cagr(nav, 24):+.1f}%  MaxDD {dd:.1f}%  "
                      f"hit {pr['hits'] / 24 * 100:.0f}%  budgets[{budget_str}]  "
                      f"({pr['elapsed_min']:.1f}min)", flush=True)
        suite["rows"].sort(key=lambda r: r.get("_k", (0, 0)))
    else:
        # Fixed trials: per-month returns are order-independent → optimize once per
        # repeat (pool), then chain under each ordering for the MaxDD distribution.
        for k in range(n_runs):
            DB_PATH = Path(f"{base}_r{k + 1}.db") if n_runs > 1 else Path(f"{base}.db")
            if DB_PATH.exists():
                DB_PATH.unlink()
            _init_db()
            print(f"\n[Fast-Test r{k + 1}/{n_runs}] 24 months -> {DB_PATH}", flush=True)
            t0 = time.perf_counter()
            ret_by_month = {}
            hits = 0
            period_args = [_mk_period(ms, args.trials) for ms in canonical]
            with mp.Pool(processes=n_workers, initializer=_worker_init,
                         initargs=(args.no_factors, exp_config)) as pool:
                for i, result in enumerate(pool.imap_unordered(_run_month, period_args)):
                    _save(result)
                    ret_by_month[result["month"]] = result["val_return"]
                    hits += 1 if result["positive"] else 0
                    hit = "HIT " if result["positive"] else "miss"
                    print(f"  [{i + 1}/24] {result['month']}  "
                          f"train={result['train_score']:+.1f}  "
                          f"val={result['val_return']:+.2f}%  {hit}", flush=True)
            elapsed = (time.perf_counter() - t0) / 60
            returns_canon = [ret_by_month.get(ms, 0.0) for ms in canonical]
            nav = _record_opt(returns_canon, hits)
            dds = []
            for j, order in enumerate(FAST_TEST_ORDERS):
                dd = _chain_maxdd([returns_canon[i - 1] for i in order])
                suite["all_maxdds"].append(dd)
                suite["per_order_maxdds"][j].append(dd)
                dds.append(dd)
            suite["rows"].append({
                "label": f"r{k + 1}", "nav": nav,
                "cagr": _proxy_cagr(nav, 24), "avg_ret": sum(returns_canon) / 24,
                "hit_rate": hits / 24 * 100,
                "dd_med": sorted(dds)[len(dds) // 2], "dd_worst": max(dds),
                "elapsed_min": elapsed,
            })

    _print_fast_test_aggregate(suite)
    return suite


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default="2021-01-01",
                        help="First validation month (needs 12mo train data before this)")
    parser.add_argument("--end",     default=None,
                        help="Last validation month start (default: current month)")
    parser.add_argument("--trials",  type=int, default=300,
                        help="Fuzzing trials per month (default 300)")
    parser.add_argument("--train-months", type=int, default=12,
                        help="Training window in months (default 12)")
    parser.add_argument("--period-days", type=int, default=None,
                        help="Validation period length in days (default: calendar month). "
                             "Use 14 for bi-weekly reopt. Training window stays --train-months.")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--exclude", nargs="+", default=[])
    parser.add_argument("--db",         default=None)
    parser.add_argument("--summary",    action="store_true",
                        help="Just print summary of existing results, no new runs")
    parser.add_argument("--no-factors", action="store_true",
                        help="Disable factor loading (baseline comparison)")
    parser.add_argument("--no-calmar", action="store_true",
                        help="Use raw geo-mean score instead of Calmar (stop-loss still active)")
    parser.add_argument("--optuna",     action="store_true",
                        help="Use Bayesian (TPE) search instead of random search")
    parser.add_argument("--min-train-score", type=float, default=0.0,
                        help="Skip to cash for any period where optimizer best train score "
                             "is below this threshold (default: 0 = no filter)")
    parser.add_argument("--fast",       action="store_true",
                        help="Single training window (3x faster, less overfit-resistant)")
    parser.add_argument("--mini",       action="store_true",
                        help="Use mini universe (9 symbols) for fast smoke-tests (~6x faster)")
    parser.add_argument("--fast-test",  action="store_true",
                        help="Fast testing framework: hardcoded 8-symbol universe + 24 stratified months "
                             "(8 per regime: bull/rising_rates/falling_rates). No config needed. Takes ~10min.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="(--fast-test only) Re-run the optimization N times on the SAME fixed "
                             "24-month set (fresh random trials each time) to measure CAGR variance "
                             "from trial noise. MaxDD path-risk comes from the fixed FAST_TEST_ORDERS "
                             "orderings within each run. Use 5 for a variance estimate. Default: 1.")
    parser.add_argument("--parallel-months", action="store_true",
                        help="Run all months in parallel (across months, not within). "
                             "Seeds/bounds are pre-fetched from HoF before any month starts "
                             "so there is zero intra-run lookahead. Fastest for re-runs where "
                             "HoF already has data from a prior run on the same date range. "
                             "Disables double-trials and intra-run HoF updates.")
    parser.add_argument("--hof-db", default=None,
                        help="Path to an isolated HoF SQLite file for this run. "
                             "Use a fresh path to run with zero lookahead from prior runs. "
                             "Default: shared data/params_hof.db")
    parser.add_argument("--random-universe", type=int, default=None,
                        help="Seed for randomizing ETF pools from master pool. "
                             "Tests whether regime gate is the alpha vs specific ETF picks.")
    parser.add_argument("--sample-months", type=int, default=None,
                        help="Evenly subsample N validation months for fast param tuning. "
                             "e.g. --sample-months 40 runs ~40 spread across the full date range.")
    parser.add_argument("--resume", action="store_true",
                        help="Allow appending to an existing DB that already has results. "
                             "Without this flag, running against a non-empty DB aborts.")
    parser.add_argument("--label", default=None,
                        help="Short label embedded in the auto-generated DB filename, "
                             "e.g. --label rand_universe (ignored when --db is explicit)")
    # ── Experiment config ────────────────────────────────────────────────────
    parser.add_argument("--elastic-martingale", action="store_true",
                        help="Enable martingale trial scaling. "
                             "Default: disabled (run true baseline with fixed trials)")
    parser.add_argument("--elastic-multipliers", default="2,4",
                        help="Trial multipliers for elastic martingale: loss1_mult,loss2_mult. "
                             "E.g. --elastic-multipliers 2,4 means 2× on 1 loss, 4× on 2+ losses. "
                             "Default: 2,4")
    parser.add_argument("--lev3x-cap", default=None,
                        help="Comma-separated lev_3x_cap choices, e.g. 0.50,0.60,0.70 "
                             "(default: 0.25,0.35,0.50,0.65,0.80,1.0)")
    parser.add_argument("--vol-target-range", default="0,0",
                        help="vol_target_pct range as min,max e.g. 5,15; 0,0 = disabled (default: 0,0)")
    parser.add_argument("--lockout-min", type=int, default=15,
                        help="Minimum stop_loss_lockout_days (default: 15)")
    args = parser.parse_args()

    global DB_PATH, _factors, _use_calmar, _mini_mode, \
           _random_on_pool, _random_rising_pool, _random_falling_pool, \
           _lev3x_cap_choices, _vol_target_range, _lockout_min
    if args.db:
        DB_PATH = Path(args.db)
    _use_calmar = not args.no_calmar
    _mini_mode  = args.mini

    if args.lev3x_cap:
        _lev3x_cap_choices = [float(x) for x in args.lev3x_cap.split(",")]
    vmin_s, vmax_s = args.vol_target_range.split(",")
    _vol_target_range = (float(vmin_s), float(vmax_s))
    _lockout_min    = args.lockout_min
    print(f"  [Experiment] lev3x_cap={_lev3x_cap_choices} vol_target={_vol_target_range} "
          f"lockout_min={_lockout_min}")

    if args.random_universe is not None:
        import random as _rnd
        _rnd.seed(args.random_universe)
        _random_on_pool      = _rnd.sample(MASTER_RISK_ON_POOL,
                                           k=min(15, len(MASTER_RISK_ON_POOL)))
        _random_rising_pool  = _rnd.sample(MASTER_RISK_OFF_RISING_POOL,
                                           k=min(4, len(MASTER_RISK_OFF_RISING_POOL)))
        _random_falling_pool = _rnd.sample(MASTER_RISK_OFF_FALLING_POOL,
                                           k=min(6, len(MASTER_RISK_OFF_FALLING_POOL)))
        print(f"  [Random Universe seed={args.random_universe}]")
        print(f"    Risk-on:  {_random_on_pool}")
        print(f"    Rising:   {_random_rising_pool}")
        print(f"    Falling:  {_random_falling_pool}")

    # ── Fast-test mode: self-contained 8-symbol + stratified-month benchmark ───────
    if args.fast_test:
        _run_fast_test(args)
        return

    _init_db()

    if args.summary:
        _print_summary()
        return

    # Guard against accidental overwrites: if the DB already has results, require --resume
    if not args.resume:
        con = sqlite3.connect(DB_PATH)
        existing = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        con.close()
        if existing > 0:
            print(f"ERROR: {DB_PATH} already contains {existing} results.")
            print("  Use --resume to append, or choose a different --db path.")
            raise SystemExit(1)

    use_optuna  = args.optuna
    no_factors  = args.no_factors
    fast_mode   = args.fast

    val_start = date.fromisoformat(args.start)
    val_end   = date.fromisoformat(args.end) if args.end else date.today().replace(day=1)
    tm        = args.train_months

    # HoF setup — initialise the shared DB so it exists before workers start
    try:
        from stratscout.engine.data.params_hof import (
            set_hof_path, init_hof, compute_month_features, find_similar_seeds, find_seeds,
            save_to_hof, compute_param_bounds_from_hof,
        )
        if args.hof_db:
            set_hof_path(args.hof_db)
            print(f"  [HoF] using isolated DB: {args.hof_db}")
        init_hof()
        hof_available = True
    except Exception:
        hof_available = False

    # Build period list — monthly by default, or fixed-day periods via --period-days
    from datetime import timedelta as _timedelta
    _period_days = args.period_days  # None = calendar month

    months = []
    m = val_start
    while m < val_end:
        m_end   = (m + _timedelta(days=_period_days)) if _period_days else (m + relativedelta(months=1))
        t_mid2  = m - relativedelta(months=tm // 3)
        t_mid1  = m - relativedelta(months=(tm * 2) // 3)
        t_start = m - relativedelta(months=tm)
        if use_optuna:
            seeds = []
            bounds = {}
            if hof_available:
                feats = compute_month_features(m.isoformat())
                seeds = find_similar_seeds(feats, k_similar=5, top_global=3, as_of_month=m.isoformat())
                bounds = compute_param_bounds_from_hof(as_of_month=m.isoformat())
            months.append((
                m.isoformat(), m_end.isoformat(),
                t_start.isoformat(), t_mid1.isoformat(), t_mid2.isoformat(),
                args.trials, args.exclude, fast_mode, args.workers, seeds, bounds,
            ))
        else:
            months.append((
                m.isoformat(), m_end.isoformat(),
                t_start.isoformat(), t_mid1.isoformat(), t_mid2.isoformat(),
                args.trials, args.exclude, fast_mode, args.min_train_score,
            ))
        m = m_end

    # Skip already-completed months
    con = sqlite3.connect(DB_PATH)
    done = {r[0] for r in con.execute("SELECT month FROM results").fetchall()}
    # Also load features for already-done months so we can save them to HoF
    done_rows = {
        r[0]: r for r in con.execute(
            "SELECT month, train_score, val_return, val_cagr, val_dd, val_trades, params "
            "FROM results"
        ).fetchall()
    }
    con.close()

    # Backfill HoF with any already-completed months not yet in HoF
    if hof_available:
        for month_str, row in done_rows.items():
            try:
                feats = compute_month_features(month_str)
                params = json.loads(row[6]) if row[6] else {}
                save_to_hof(
                    str(DB_PATH), month_str,
                    row[1], row[2], row[3], row[4], row[5],
                    params, feats,
                )
            except Exception:
                pass

    remaining = [m for m in months if m[0] not in done]

    # Parse elastic multipliers
    elastic_mult_1, elastic_mult_2 = 2, 4  # defaults
    if args.elastic_martingale and args.elastic_multipliers:
        try:
            parts = args.elastic_multipliers.split(',')
            elastic_mult_1 = int(parts[0])
            elastic_mult_2 = int(parts[1])
        except (ValueError, IndexError):
            print(f"  WARNING: Invalid --elastic-multipliers '{args.elastic_multipliers}', using defaults (2,4)")

    if args.sample_months and args.sample_months < len(remaining):
        step = len(remaining) / args.sample_months
        remaining = [remaining[int(i * step)] for i in range(args.sample_months)]
        print(f"  [--sample-months] Subsampled to {len(remaining)} evenly-spaced months")

    mode_label = "Bayesian/TPE" if use_optuna else "Random"
    factor_label = "DISABLED (baseline)" if no_factors else "enabled"
    hof_label = "enabled" if hof_available else "unavailable"
    score_label = "Calmar (CAGR/DD)" if _use_calmar else "Raw geo-mean + DD penalty"
    print(f"\nWalk-Forward ETF Validator")
    print(f"  Validation months: {len(months)} total, {len(done)} done, {len(remaining)} remaining")
    print(f"  Training window:   {tm} months split into 3 sub-windows")
    print(f"  Trials per month:  {args.trials}  ({mode_label})")
    print(f"  Score objective:   {score_label}")
    print(f"  Workers:           {args.workers}")
    print(f"  Factors:           {factor_label}")
    print(f"  Hall of Fame:      {hof_label}")
    print(f"  DB:                {DB_PATH}")

    if not remaining:
        print("\nAll months already computed.")
        _print_summary()
        return

    t0 = time.perf_counter()
    run_fn = _run_month_optuna if use_optuna else _run_month

    # --parallel-months: pre-bake seeds/bounds from existing HoF, then run all
    # months simultaneously across a process pool (1 worker per month).
    # Lookahead safety: each month's seeds are fetched with as_of_month=month_str
    # BEFORE any month starts, so no month can observe another month's results.
    # Intra-run HoF updates are disabled — seeds come from prior runs only.
    if args.parallel_months and use_optuna:
        print("  [parallel-months] pre-fetching seeds/bounds for all months...", flush=True)
        prepped = []
        for month_args in remaining:
            month_str = month_args[0]
            if hof_available:
                feats = compute_month_features(month_str) or {}
                # prev_val_return unknown at pre-fetch time — use None (neutral)
                feats["prev_val_return"] = None
                seeds = find_seeds(month_str, features=feats, as_of_month=month_str)
                bounds = compute_param_bounds_from_hof(as_of_month=month_str)
                # Force 1 worker per month — parallelism is across months now
                ma = month_args[:8] + (1,) + (seeds, bounds)
            else:
                ma = month_args[:8] + (1,) + ([], {})
            prepped.append(ma)

        print(f"  [parallel-months] launching {len(prepped)} months on {args.workers} workers", flush=True)
        with mp.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(no_factors,),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(run_fn, prepped)):
                _save(result)
                elapsed = time.perf_counter() - t0
                hit = "HIT " if result["positive"] else "miss"
                print(
                    f"  [{i+1}/{len(remaining)}] {result['month']}  "
                    f"train={result['train_score']:+.1f}  "
                    f"val={result['val_return']:+.2f}%  "
                    f"{hit}  ({elapsed/60:.1f}min)"
                )
        _print_summary()
        return

    # Optuna path: run months sequentially so each can use updated HoF seeds.
    # Parallelism is within each month (n_jobs=workers passed in args tuple).
    # Random path: parallel pool across months as before.
    if use_optuna:
        _worker_init(no_factors)
        prev_val_return: float | None = None  # tracks prior month's OOS result
        recent_returns: list[float] = []  # last 2 val returns for consecutive-miss detection
        for i, month_args in enumerate(remaining):
            # ELASTIC TRIALS: martingale scaling based on recent losses (only if --elastic-martingale flag set)
            effective_trials = args.trials
            elasticity_reason = ""

            if args.elastic_martingale:
                # Count consecutive losses
                consecutive_losses = 0
                for ret in reversed(recent_returns):
                    if ret < 0:
                        consecutive_losses += 1
                    else:
                        break

                if consecutive_losses == 0:
                    # Win or first period: baseline (no reason to print for baseline)
                    effective_trials = args.trials
                elif consecutive_losses == 1:
                    # 1 loss: scale by elastic_mult_1
                    effective_trials = args.trials * elastic_mult_1
                    elasticity_reason = f"1 loss → {elastic_mult_1}× trials"
                elif consecutive_losses >= 2:
                    # 2+ losses: scale by elastic_mult_2
                    effective_trials = args.trials * elastic_mult_2
                    elasticity_reason = f"{consecutive_losses} losses → {elastic_mult_2}× trials"

                if elasticity_reason:
                    print(f"  [elastic] {month_args[0]:10s} | {elasticity_reason}", flush=True)
                if effective_trials != args.trials:
                    month_args = month_args[:5] + (effective_trials,) + month_args[6:]

            # Refresh seeds AND bounds from HoF just before each month runs.
            # prev_val_return helps KNN find months that recovered from similar outcomes.
            # Bounds refresh lets later months in a run benefit from newly learned params.
            if hof_available:
                month_str = month_args[0]
                feats = compute_month_features(month_str) or {}
                feats["prev_val_return"] = prev_val_return
                seeds = find_similar_seeds(feats, k_similar=5, top_global=3, as_of_month=month_str)
                bounds = compute_param_bounds_from_hof(as_of_month=month_str)
                month_args = month_args[:9] + (seeds, bounds)
            result = run_fn(month_args)
            _save(result)
            # Write result to shared HoF immediately
            if hof_available and result.get("params"):
                try:
                    feats = compute_month_features(result["month"]) or {}
                    feats["prev_val_return"] = prev_val_return
                    params = json.loads(result["params"])
                    save_to_hof(
                        str(DB_PATH), result["month"],
                        result["train_score"], result["val_return"],
                        result.get("val_cagr", 0), result.get("val_dd", 0),
                        result.get("val_trades", 0), params, feats,
                        prev_val_return=prev_val_return,
                    )
                except Exception:
                    pass
            prev_val_return = result["val_return"]
            recent_returns.append(result["val_return"])
            if len(recent_returns) > 2:
                recent_returns = recent_returns[-2:]
            elapsed = time.perf_counter() - t0
            hit = "HIT " if result["positive"] else "miss"
            print(
                f"  [{i+1}/{len(remaining)}] {result['month']}  "
                f"train={result['train_score']:+.1f}  "
                f"val={result['val_return']:+.2f}%  "
                f"{hit}  ({elapsed/60:.1f}min)"
            )
    else:
        # Random search: run sequentially to enable elastic scaling based on prior month results
        _worker_init(no_factors)
        prev_val_return: float | None = None
        recent_returns: list[float] = []
        for i, month_args in enumerate(remaining):
            # ELASTIC TRIALS: martingale scaling based on recent losses (only if --elastic-martingale flag set)
            effective_trials = args.trials
            elasticity_reason = ""

            if args.elastic_martingale:
                # Count consecutive losses
                consecutive_losses = 0
                for ret in reversed(recent_returns):
                    if ret < 0:
                        consecutive_losses += 1
                    else:
                        break

                if consecutive_losses == 0:
                    # Win or first period: baseline (no reason to print for baseline)
                    effective_trials = args.trials
                elif consecutive_losses == 1:
                    # 1 loss: scale by elastic_mult_1
                    effective_trials = args.trials * elastic_mult_1
                    elasticity_reason = f"1 loss → {elastic_mult_1}× trials"
                elif consecutive_losses >= 2:
                    # 2+ losses: scale by elastic_mult_2
                    effective_trials = args.trials * elastic_mult_2
                    elasticity_reason = f"{consecutive_losses} losses → {elastic_mult_2}× trials"

                if elasticity_reason:
                    print(f"  [elastic] {month_args[0]:10s} | {elasticity_reason}", flush=True)
                if effective_trials != args.trials:
                    month_args = month_args[:5] + (effective_trials,) + month_args[6:]

            result = run_fn(month_args)
            _save(result)
            prev_val_return = result["val_return"]
            recent_returns.append(result["val_return"])
            if len(recent_returns) > 2:
                recent_returns = recent_returns[-2:]
            elapsed = time.perf_counter() - t0
            hit = "HIT " if result["positive"] else "miss"
            print(
                f"  [{i+1}/{len(remaining)}] {result['month']}  "
                f"train={result['train_score']:+.1f}  "
                f"val={result['val_return']:+.2f}%  "
                f"{hit}  ({elapsed/60:.1f}min)"
            )

    _print_summary()


if __name__ == "__main__":
    mp.freeze_support()
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        raise SystemExit("pip install python-dateutil")
    main()
