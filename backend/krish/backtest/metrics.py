"""Performance metrics.

Two things matter here beyond the usual numbers:

* every metric is computed the same way for in-sample and out-of-sample, so the
  degradation comparison is meaningful;
* :func:`robust_score` is the single number the judge and tuner optimise, and it
  is built to punish the things that make backtests lie — too few trades, deep
  drawdowns, dependence on one lucky trade, and in-sample/out-of-sample gaps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _safe(value: float, default: float = 0.0) -> float:
    return float(value) if value is not None and math.isfinite(value) else default


def max_drawdown(equity: Sequence[float]) -> tuple[float, int]:
    """Return (max drawdown fraction, longest underwater length in bars)."""
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return 0.0, 0
    peaks = np.maximum.accumulate(arr)
    dd = np.where(peaks > 0, (peaks - arr) / peaks, 0.0)
    underwater = arr < peaks
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return float(dd.max()), int(longest)


def compute_metrics(
    equity: Sequence[float],
    trades: Sequence[dict[str, Any]],
    *,
    bars_per_year: float,
    initial_balance: float,
    bars: int,
    exposure_bars: int = 0,
) -> dict[str, Any]:
    eq = np.asarray(equity, dtype=float)
    n_trades = len(trades)
    final = float(eq[-1]) if eq.size else float(initial_balance)

    returns = np.diff(eq) / np.where(eq[:-1] == 0, np.nan, eq[:-1]) if eq.size > 1 else np.array([])
    returns = returns[np.isfinite(returns)]

    total_return = (final / initial_balance) - 1.0 if initial_balance else 0.0
    years = max(bars / bars_per_year, 1e-9)
    cagr = (final / initial_balance) ** (1 / years) - 1.0 if final > 0 and initial_balance else -1.0

    if returns.size > 1 and returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(bars_per_year))
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    if downside.size > 1 and downside.std(ddof=1) > 0:
        sortino = float(returns.mean() / downside.std(ddof=1) * math.sqrt(bars_per_year))
    else:
        sortino = 0.0

    dd, dd_len = max_drawdown(eq)

    pnls = np.array([float(t.get("pnl", 0.0)) for t in trades], dtype=float)
    r_multiples = np.array([float(t.get("r", 0.0)) for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    # How much of the profit came from the single best trade? >40% is fragile.
    top_trade_share = float(wins.max() / gross_win) if wins.size and gross_win > 0 else 0.0

    return {
        "final_balance": round(final, 2),
        "total_return_pct": round(total_return * 100, 3),
        "cagr_pct": round(_safe(cagr) * 100, 3),
        "sharpe": round(_safe(sharpe), 3),
        "sortino": round(_safe(sortino), 3),
        "max_drawdown_pct": round(dd * 100, 3),
        "max_drawdown_bars": dd_len,
        "calmar": round(_safe(cagr / dd) if dd > 0 else 0.0, 3),
        "profit_factor": round(profit_factor, 3),
        "trades": n_trades,
        "win_rate_pct": round(float((pnls > 0).mean() * 100), 2) if n_trades else 0.0,
        "avg_win": round(float(wins.mean()), 2) if wins.size else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if losses.size else 0.0,
        "expectancy": round(float(pnls.mean()), 3) if n_trades else 0.0,
        "expectancy_r": round(float(r_multiples.mean()), 4) if n_trades else 0.0,
        "best_trade": round(float(pnls.max()), 2) if n_trades else 0.0,
        "worst_trade": round(float(pnls.min()), 2) if n_trades else 0.0,
        "top_trade_profit_share": round(top_trade_share, 3),
        "gross_profit": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "exposure_pct": round(exposure_bars / bars * 100, 2) if bars else 0.0,
        "bars": bars,
        "years": round(years, 2),
        "trades_per_year": round(n_trades / years, 1) if years else 0.0,
        "max_consecutive_losses": _max_streak(pnls, positive=False),
        "max_consecutive_wins": _max_streak(pnls, positive=True),
    }


def _max_streak(pnls: np.ndarray, *, positive: bool) -> int:
    longest = current = 0
    for pnl in pnls:
        hit = pnl > 0 if positive else pnl < 0
        current = current + 1 if hit else 0
        longest = max(longest, current)
    return int(longest)


def monthly_returns(equity: pd.Series) -> dict[str, float]:
    """Month-by-month percentage returns, for the UI heatmap."""
    if not isinstance(equity.index, pd.DatetimeIndex) or equity.empty:
        return {}
    monthly = equity.resample("ME").last().pct_change().dropna()
    return {ts.strftime("%Y-%m"): round(float(val) * 100, 2) for ts, val in monthly.items()}


def robust_score(
    metrics: dict[str, Any],
    *,
    min_trades: int = 40,
    is_metrics: dict[str, Any] | None = None,
    n_params: int = 0,
    complexity_penalty: float = 0.0,
) -> float:
    """The factory's fitness function. Higher is better; <= 0 means "no edge".

    Built from out-of-sample numbers only, then penalised for fragility.
    """
    sharpe = float(metrics.get("sharpe", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    dd = float(metrics.get("max_drawdown_pct", 0.0))
    trades = int(metrics.get("trades", 0))
    expectancy_r = float(metrics.get("expectancy_r", 0.0))

    if trades == 0 or sharpe <= 0 or pf <= 1.0:
        return 0.0

    base = sharpe * min(pf, 3.0) / 3.0

    # sample-size confidence: results from 5 trades are not results
    confidence = min(1.0, trades / max(min_trades, 1)) ** 0.5
    base *= confidence

    # drawdown penalty, steep past 25%
    if dd > 0:
        base *= 1.0 / (1.0 + (dd / 25.0) ** 2)

    # single-trade dependence
    share = float(metrics.get("top_trade_profit_share", 0.0))
    if share > 0.4:
        base *= max(0.2, 1.0 - (share - 0.4) * 2.0)

    # in-sample -> out-of-sample degradation
    if is_metrics:
        is_sharpe = float(is_metrics.get("sharpe", 0.0))
        if is_sharpe > 0:
            retention = max(0.0, min(1.5, sharpe / is_sharpe))
            base *= min(1.0, retention)

    # every tuned knob is a chance to have curve-fitted
    if n_params and complexity_penalty:
        base *= max(0.3, 1.0 - complexity_penalty * n_params)

    if expectancy_r <= 0:
        base *= 0.3

    return round(max(0.0, base), 4)
