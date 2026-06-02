"""Symbol universe definitions for both ETF rotation and small-cap strategies.

Mirrors the legacy etf_universe.py and smallcap_universe.py exports so engine
code can import from a single location:

    from stratscout.engine.data.universes import ANCHORS, RISK_ON_POOL, ALL_ETFS, SMALLCAP_UNIVERSE

The ETF section keeps the original constants intact so backtest reproducibility
is bit-for-bit with the legacy code path.
"""
from __future__ import annotations

# ── ETF universe (legacy etf_universe.py — DO NOT reorder without re-running golden tests)

ANCHORS: list[str] = ["AGG", "BIL", "TLT"]

RISK_ON_POOL: list[str] = [
    "SOXL", "TQQQ", "UPRO", "TECL", "SPXL", "FAS", "CURE", "LABU",
    "ERX", "DRN", "FNGU", "UTSL", "MIDU", "TNA", "URTY",
    "MSTR", "GBTC",
    "JNUG", "GDXU", "SILJ", "SIL",
    # Unleveraged momentum — performs in low-vol grinding bull markets
    # where leveraged ETFs get vol-targeted to cash
    "MTUM", "VGT", "XLK", "QQQM",
    # IPO/spinoff momentum — captures new-issue alpha (FPX has 20yr track record)
    "FPX",
    # EFO/EET removed 2026-05-29 — both flagged <$50M AUM, closure risk
    # BITX/CONL/IBIT removed 2026-06-01 — sub-50% win rate across 1000+ periods
]

RISK_OFF_RISING_POOL: list[str] = ["QID", "TBF", "SQQQ", "TBT", "PSQ", "BIL"]

RISK_OFF_FALLING_POOL: list[str] = [
    "UGL", "TMF", "BTAL", "XLP", "NUGT", "UUP", "GLD", "SLV",
    "KMLM", "DBMF", "TAIL",
]

ALL_ETFS: list[str] = ANCHORS + RISK_ON_POOL + RISK_OFF_RISING_POOL + RISK_OFF_FALLING_POOL
# Backwards-compat alias used by legacy modules
ALL_SYMBOLS = ALL_ETFS

SECTOR_BUCKETS: dict[str, str] = {
    "SOXL": "tech", "TECL": "tech", "TQQQ": "tech",
    "UPRO": "broad", "SPXL": "broad",
    "FAS": "financials",
    "CURE": "healthcare", "LABU": "healthcare",
    "ERX": "energy",
    "DRN": "realestate",
    "UTSL": "utilities",
    "MIDU": "midcap",
    "TNA": "smallcap", "URTY": "smallcap",
    "QID": "short_tech", "SQQQ": "short_tech", "PSQ": "short_tech",
    "TBF": "short_bonds", "TBT": "short_bonds",
    "UGL": "gold", "GLD": "gold",
    "TMF": "long_bonds",
    "BTAL": "hedge",
    "XLP": "staples",
    "UUP": "dollar",
    "SLV": "silver",
    "MTUM": "momentum", "QQQM": "tech",
    "VGT": "tech", "XLK": "tech",
    "FPX": "ipo",
    "MSTR": "crypto", "GBTC": "crypto", "BITX": "crypto",
    "CONL": "crypto", "IBIT": "crypto",
    "JNUG": "gold_miners", "GDXU": "gold_miners", "NUGT": "gold_miners",
    "SILJ": "silver_miners", "SIL": "silver_miners",
    "KMLM": "managed_futures", "DBMF": "managed_futures",
    "TAIL": "tail_risk",
}

MIN_RISK_ON: int = 2
MIN_RISK_OFF_RISING: int = 1
MIN_RISK_OFF_FALLING: int = 2


# ── Small-cap universe (loaded lazily from legacy file to avoid duplicating the long list)

def smallcap_universe() -> list[str]:
    """Return the small-cap ticker universe."""
    try:
        from smallcap_universe import SMALLCAP_UNIVERSE  # type: ignore
        return list(SMALLCAP_UNIVERSE)
    except ImportError:
        return []
