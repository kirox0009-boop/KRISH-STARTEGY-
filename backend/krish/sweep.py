"""Find where a strategy actually works: every timeframe, every asset.

The same rules can be dead on GOLD H1 and excellent on BITCOIN H4. Judging an
idea on the one market it happened to be born in throws away most of its value.

Two rules make this honest rather than a machine for producing false positives:

1. **The sweep only ever sees in-sample data.** It picks a winner using the first
   70% of history; the out-of-sample tail stays untouched, so the winning cell can
   still be judged on data nothing has looked at. Sweeping on full history and
   then "validating" on the same bars would be self-congratulation.

2. **The winner is deflated for the number of cells tried.** Best-of-20 flatters
   itself; some of that edge is the luckiest draw. The haircut grows with the
   number of combinations, so a winner from 20 cells must be genuinely better
   than a winner from 2.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .assets import universe
from .backtest.engine import BacktestConfig, run_backtest
from .backtest.metrics import deflated_by_selection, robust_score
from .ir.schema import StrategyIR

log = logging.getLogger("krish.sweep")

#: A cell needs at least this many in-sample trades to be taken seriously. Matches
#: the judge's out-of-sample bar: promoting a 20-trade cell would just be promoting
#: noise, and the judge would reject it anyway.
MIN_IS_TRADES = 30


def cross_market_score(metrics: dict[str, Any]) -> float:
    """Rank cells on a measure that does NOT depend on the timeframe.

    The obvious choice, ``robust_score``, is built on an annualised Sharpe - and
    annualising from M15 assumes ~24,800 bars a year against 252 for D1. The same
    real edge therefore scores several times higher on a fast timeframe, so
    ranking a cross-timeframe sweep by Sharpe would hand the win to M15 almost
    every time regardless of merit. That is a measurement artefact, not an edge.

    So ranking uses only quantities with no time unit in them:

        profit factor  - how much is won per unit lost
        trade count    - sample size, square-rooted so it cannot dominate
        drawdown       - what it cost to get there

    ``robust_score`` is still recorded per cell for reference; it just does not
    decide the winner.
    """
    pf = min(float(metrics.get("profit_factor", 0.0) or 0.0), 3.0)
    trades = int(metrics.get("trades", 0) or 0)
    dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    expectancy = float(metrics.get("expectancy_r", 0.0) or 0.0)
    if pf <= 1.0 or trades <= 0 or expectancy <= 0:
        return 0.0
    sample = math.sqrt(min(trades, 250) / MIN_IS_TRADES)
    return round((pf - 1.0) * sample / (1.0 + (dd / 25.0) ** 2), 4)


@dataclass(slots=True)
class SweepCell:
    asset: str
    timeframe: str
    score: float = 0.0  # timeframe-neutral, decides the winner
    robust: float = 0.0  # annualised robust score, for reference only
    trades: int = 0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_bars_held: float = 0.0
    bars: int = 0
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and self.trades >= MIN_IS_TRADES and self.score > 0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usable"] = self.usable
        return d


@dataclass(slots=True)
class SweepResult:
    cells: list[SweepCell] = field(default_factory=list)
    best: SweepCell | None = None
    tried: int = 0
    raw_best_score: float = 0.0
    deflated_best_score: float = 0.0

    def top(self, k: int) -> list[SweepCell]:
        return [c for c in sorted(self.cells, key=lambda c: -c.score) if c.usable][:k]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tried": self.tried,
            "usable": sum(1 for c in self.cells if c.usable),
            "raw_best_score": self.raw_best_score,
            "deflated_best_score": self.deflated_best_score,
            "best": self.best.as_dict() if self.best else None,
            "cells": [c.as_dict() for c in sorted(self.cells, key=lambda c: -c.score)],
        }


FrameLoader = Callable[[str, str], pd.DataFrame]


def in_sample_slice(frame: pd.DataFrame, oos_fraction: float) -> pd.DataFrame:
    """The only part of history the sweep is allowed to look at."""
    split = int(len(frame) * (1.0 - oos_fraction))
    return frame.iloc[:split]


def sweep(
    ir: StrategyIR,
    load_frame: FrameLoader,
    *,
    assets: list[str] | None = None,
    timeframes: list[str] | None = None,
    config: BacktestConfig | None = None,
    max_cells: int = 24,
) -> SweepResult:
    """Score ``ir`` on every asset/timeframe combination, in-sample only."""
    config = config or BacktestConfig.from_factory_config()
    uni = universe()
    assets = assets or uni.keys()
    result = SweepResult()

    for asset_key in assets:
        try:
            asset = uni.get(asset_key)
        except KeyError:
            continue
        wanted = timeframes or list(asset.timeframes)
        for tf in wanted:
            if tf not in asset.timeframes:
                continue
            if result.tried >= max_cells:
                break
            result.tried += 1
            cell = SweepCell(asset=asset.key, timeframe=tf)
            try:
                frame = load_frame(asset.key, tf)
                is_frame = in_sample_slice(frame, config.oos_fraction)
                cell.bars = len(is_frame)
                if len(is_frame) < 400:
                    cell.error = f"only {len(is_frame)} in-sample bars"
                else:
                    variant = ir.model_copy(deep=True)
                    variant.asset = asset.key
                    variant.timeframe = tf
                    run = run_backtest(variant, is_frame, config=config, kind="sweep")
                    if not run.ok:
                        cell.error = run.error or "no trades"
                    else:
                        m = run.metrics
                        cell.trades = int(m.get("trades", 0))
                        cell.sharpe = float(m.get("sharpe", 0.0))
                        cell.profit_factor = float(m.get("profit_factor", 0.0))
                        cell.max_drawdown_pct = float(m.get("max_drawdown_pct", 0.0))
                        cell.avg_bars_held = float(m.get("avg_bars_held", 0.0))
                        cell.robust = robust_score(m, min_trades=MIN_IS_TRADES)
                        cell.score = cross_market_score(m)
            except Exception as exc:  # a dead market must not kill the sweep
                cell.error = str(exc)
            result.cells.append(cell)

    usable = [c for c in result.cells if c.usable]
    if usable:
        result.best = max(usable, key=lambda c: c.score)
        result.raw_best_score = result.best.score
        # Deflate by cells TRIED, not cells that happened to survive. The search
        # space is what creates the multiple-comparison problem: finding one
        # winner after looking at 24 markets is a weaker claim than finding it
        # after looking at 2, whether or not the other 23 were usable.
        result.deflated_best_score = deflated_by_selection(result.best.score, result.tried)
    log.info(
        "sweep of '%s': %d cells, %d usable, best %s",
        ir.name,
        result.tried,
        len(usable),
        f"{result.best.asset} {result.best.timeframe} @ {result.raw_best_score:.3f} "
        f"(deflated {result.deflated_best_score:.3f})"
        if result.best
        else "none",
    )
    return result


def relocate(ir: StrategyIR, cell: SweepCell) -> StrategyIR:
    """A copy of ``ir`` moved to the market and timeframe it works best on."""
    moved = ir.model_copy(deep=True)
    moved.id = StrategyIR.model_fields["id"].default_factory()  # type: ignore[misc]
    moved.asset = cell.asset
    moved.timeframe = cell.timeframe
    moved.parents = [ir.id]
    moved.generation = ir.generation + 1
    moved.origin = "promoted"
    moved.name = f"{ir.name.split(' [')[0]} [{cell.asset} {cell.timeframe}]"
    moved.notes = (
        f"{ir.notes} | promoted from {ir.asset} {ir.timeframe} after an in-sample "
        f"sweep: best of the field at {cell.asset} {cell.timeframe}"
    )
    return moved


__all__ = ["MIN_IS_TRADES", "SweepCell", "SweepResult", "in_sample_slice", "relocate", "sweep"]
