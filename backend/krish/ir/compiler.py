"""IR -> Python signal compiler.

Turns a :class:`StrategyIR` into boolean entry/exit series aligned to the bar
index. Two rules are enforced here and never relaxed:

1. **No peeking.** Operand shifts are non-negative and every indicator is
   causal, so a signal at bar *i* only knows bars <= i.
2. **Act later.** This compiler produces the *decision* series; the backtester
   executes it on the **next** bar's open. Signal generation and execution are
   deliberately separated so no strategy can accidentally trade its own bar's
   close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import indicators as ind
from .schema import Condition, ConditionOp, Operand, OperandKind, RuleGroup, StrategyIR


class IRCompileError(ValueError):
    """The IR could not be turned into signals."""


@dataclass(slots=True)
class CompiledStrategy:
    ir: StrategyIR
    frame: pd.DataFrame  # OHLCV + every named indicator column
    entry_long: pd.Series
    entry_short: pd.Series
    exit_long: pd.Series
    exit_short: pd.Series
    atr: pd.Series
    warmup: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def index(self) -> pd.Index:
        return self.frame.index

    def signal_counts(self) -> dict[str, int]:
        return {
            "entry_long": int(self.entry_long.sum()),
            "entry_short": int(self.entry_short.sum()),
            "exit_long": int(self.exit_long.sum()),
            "exit_short": int(self.exit_short.sum()),
        }


# --------------------------------------------------------------------------- #
# operand / condition evaluation
# --------------------------------------------------------------------------- #


def _operand_series(op: Operand, frame: pd.DataFrame) -> pd.Series:
    if op.kind is OperandKind.CONST:
        return pd.Series(float(op.value or 0.0), index=frame.index, dtype=float)
    if op.kind is OperandKind.PRICE:
        series = ind._src(frame, str(op.ref))
    else:
        alias = str(op.ref)
        if alias not in frame.columns:
            raise IRCompileError(f"condition references unknown indicator alias '{alias}'")
        series = frame[alias].astype(float)
    return series.shift(op.shift) if op.shift else series


def _condition_series(cond: Condition, frame: pd.DataFrame) -> pd.Series:
    left = _operand_series(cond.left, frame)

    if cond.op in {ConditionOp.RISING, ConditionOp.FALLING}:
        delta = left.diff()
        want_up = cond.op is ConditionOp.RISING
        step = (delta > 0) if want_up else (delta < 0)
        return step.rolling(cond.lookback, min_periods=cond.lookback).sum() == cond.lookback

    right = _operand_series(cond.right, frame)  # type: ignore[arg-type]

    match cond.op:
        case ConditionOp.GT:
            out = left > right
        case ConditionOp.LT:
            out = left < right
        case ConditionOp.GTE:
            out = left >= right
        case ConditionOp.LTE:
            out = left <= right
        case ConditionOp.CROSS_ABOVE:
            out = (left > right) & (left.shift(1) <= right.shift(1))
        case ConditionOp.CROSS_BELOW:
            out = (left < right) & (left.shift(1) >= right.shift(1))
        case ConditionOp.BETWEEN:
            upper = _operand_series(cond.right2, frame)  # type: ignore[arg-type]
            low, high = np.minimum(right, upper), np.maximum(right, upper)
            out = (left >= low) & (left <= high)
        case _:  # pragma: no cover - schema keeps this unreachable
            raise IRCompileError(f"unsupported operator '{cond.op}'")

    # NaN (warm-up) must never read as True
    valid = left.notna() & right.notna()
    return (out & valid).fillna(False)


def _rule_group(group: RuleGroup, frame: pd.DataFrame, *, default: bool) -> pd.Series:
    if group.is_empty():
        return pd.Series(default, index=frame.index, dtype=bool)
    series = [_condition_series(c, frame) for c in group.conditions]
    combined = series[0]
    for nxt in series[1:]:
        combined = (combined & nxt) if group.logic == "and" else (combined | nxt)
    return combined.fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #


def _time_mask(ir: StrategyIR, frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if not isinstance(frame.index, pd.DatetimeIndex):
        return mask
    filters = ir.filters
    if filters.allowed_hours:
        mask &= frame.index.hour.isin(filters.allowed_hours)
    if filters.allowed_weekdays:
        mask &= frame.index.dayofweek.isin(filters.allowed_weekdays)
    return mask


def _volatility_mask(ir: StrategyIR, frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    filters = ir.filters
    if filters.min_atr_pct is None and filters.max_atr_pct is None:
        return mask
    atr_pct = ind.atr_pct(frame, ir.risk.atr_period)
    if filters.min_atr_pct is not None:
        mask &= atr_pct >= filters.min_atr_pct
    if filters.max_atr_pct is not None:
        mask &= atr_pct <= filters.max_atr_pct
    return mask.fillna(False)


def _news_mask(frame: pd.DataFrame, ir: StrategyIR) -> pd.Series:
    """Blackout windows come in as a ``news_blackout`` column from the data layer.

    Absent that column (Phase 0), the filter is a no-op rather than a lie.
    """
    if not ir.filters.avoid_high_impact_news or "news_blackout" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    return ~frame["news_blackout"].fillna(False).astype(bool)


def _trend_mask(ir: StrategyIR, frame: pd.DataFrame, *, long_side: bool) -> pd.Series:
    filters = ir.filters
    if filters.trend_filter_mode == "off" or not filters.trend_filter_alias:
        return pd.Series(True, index=frame.index, dtype=bool)
    alias = filters.trend_filter_alias
    if alias not in frame.columns:
        raise IRCompileError(f"trend filter references unknown alias '{alias}'")
    line = frame[alias].astype(float)
    close = frame["close"].astype(float)
    if filters.trend_filter_mode == "with_slope":
        slope_up = line.diff() > 0
        mask = slope_up if long_side else ~slope_up
    elif filters.trend_filter_mode == "above":
        mask = (close > line) if long_side else (close < line)
    else:  # "below" inverts the reading (mean-reversion style filters)
        mask = (close < line) if long_side else (close > line)
    return (mask & line.notna()).fillna(False)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


REQUIRED_COLUMNS = ("open", "high", "low", "close")


def compile_ir(ir: StrategyIR, df: pd.DataFrame) -> CompiledStrategy:
    """Compile ``ir`` against OHLCV ``df`` (DatetimeIndex, lowercase columns)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IRCompileError(f"price frame missing columns: {missing}")
    if len(df) < 50:
        raise IRCompileError(f"not enough bars to compile: {len(df)}")

    frame = df.copy()
    warmup = 30

    for alias, spec in ir.indicators.items():
        if alias in REQUIRED_COLUMNS or alias == "volume":
            raise IRCompileError(f"indicator alias '{alias}' shadows a price column")
        try:
            frame[alias] = ind.compute(spec.type, frame, spec.params, spec.source)
        except Exception as exc:
            raise IRCompileError(f"indicator '{alias}' ({spec.type}) failed: {exc}") from exc
        warmup = max(warmup, ind.min_bars_needed(spec.type, spec.params))

    atr_series = ind.atr(frame, ir.risk.atr_period)
    frame["_atr"] = atr_series
    warmup = max(warmup, ir.risk.atr_period * 3)

    long_ok = ir.trades_long()
    short_ok = ir.trades_short()

    entry_long = _rule_group(ir.entry_long, frame, default=False) if long_ok else _false(frame)
    entry_short = _rule_group(ir.entry_short, frame, default=False) if short_ok else _false(frame)
    exit_long = _rule_group(ir.exit_long, frame, default=False)
    exit_short = _rule_group(ir.exit_short, frame, default=False)

    gate = _time_mask(ir, frame) & _volatility_mask(ir, frame) & _news_mask(frame, ir)
    entry_long &= gate & _trend_mask(ir, frame, long_side=True)
    entry_short &= gate & _trend_mask(ir, frame, long_side=False)

    # Nothing may fire during warm-up, and the last bar has no "next open".
    blackout = pd.Series(True, index=frame.index, dtype=bool)
    blackout.iloc[: min(warmup, len(frame))] = False
    entry_long &= blackout
    entry_short &= blackout

    diagnostics = {
        "warmup_bars": warmup,
        "bars": len(frame),
        "gate_pass_rate": round(float(gate.mean()), 4),
        "indicator_nan_rate": {
            alias: round(float(frame[alias].isna().mean()), 4) for alias in ir.indicators
        },
    }

    return CompiledStrategy(
        ir=ir,
        frame=frame,
        entry_long=entry_long.astype(bool),
        entry_short=entry_short.astype(bool),
        exit_long=exit_long.astype(bool),
        exit_short=exit_short.astype(bool),
        atr=atr_series,
        warmup=warmup,
        diagnostics=diagnostics,
    )


def _false(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=frame.index, dtype=bool)
