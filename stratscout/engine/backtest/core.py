# Thin re-export — all logic lives in stratscout.engine.backtest.core
from stratscout.engine.backtest.core import (
    BacktestError,
    value_of_portfolio,
    rebalance_positions,
    compute_performance,
)

__all__ = ["BacktestError", "value_of_portfolio", "rebalance_positions", "compute_performance"]
