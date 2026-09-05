"""Backtesting: the only thing that decides whether a strategy is real."""

from .engine import BacktestConfig, BacktestResult, Trade, run_backtest, split_backtest
from .metrics import compute_metrics, robust_score

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "compute_metrics",
    "robust_score",
    "run_backtest",
    "split_backtest",
]
