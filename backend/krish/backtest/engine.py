"""Event-driven backtest engine.

Design choices that exist specifically to stop the backtest from lying:

* **Signals decide on bar close, execution happens on the next bar's open.**
  Nothing can trade the candle it was computed from.
* **Costs are always charged**: half spread + slippage as an adverse price on
  both entry and exit, plus per-lot commission, from the asset's own cost model.
* **Stop wins ties.** If a bar's range touches both stop and target, the stop is
  assumed hit. Real fills are never generous.
* **Position size follows risk, not fantasy.** Lots come from the configured
  risk-per-trade and the actual stop distance, capped by a leverage limit.
* Equity is marked to market every bar, so drawdown is real drawdown, not a
  closed-trade illusion.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..assets import Asset, bars_per_year, universe
from ..config import factory_section
from ..ir.compiler import CompiledStrategy, compile_ir
from ..ir.schema import StrategyIR
from .metrics import compute_metrics, monthly_returns, robust_score

Side = Literal["long", "short"]
MAX_LEVERAGE = 50.0
MIN_LOTS = 0.01


@dataclass(slots=True)
class BacktestConfig:
    initial_balance: float = 10_000.0
    risk_per_trade_pct: float | None = None  # None -> take it from the IR
    oos_fraction: float = 0.3
    min_trades: int = 40
    equity_curve_points: int = 400  # downsample for the UI
    max_trades_stored: int = 500
    cost_multiplier: float = 1.0  # stress-test knob (robustness agent)

    @classmethod
    def from_factory_config(cls) -> BacktestConfig:
        cfg = factory_section("backtest")
        return cls(
            initial_balance=float(cfg.get("initial_balance", 10_000.0)),
            risk_per_trade_pct=cfg.get("risk_per_trade_pct"),
            oos_fraction=float(cfg.get("oos_fraction", 0.3)),
            min_trades=int(cfg.get("min_trades", 40)),
        )


@dataclass(slots=True)
class Trade:
    side: Side
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    lots: float
    pnl: float
    r: float
    bars_held: int
    exit_reason: str
    mae_r: float = 0.0  # worst excursion against the trade, in R
    mfe_r: float = 0.0  # best excursion in favour, in R

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BacktestResult:
    strategy_id: str
    asset: str
    timeframe: str
    kind: str
    metrics: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    monthly: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    period: dict[str, str] = field(default_factory=dict)
    duration_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.metrics.get("trades", 0) > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "kind": self.kind,
            "metrics": self.metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "monthly": self.monthly,
            "diagnostics": self.diagnostics,
            "period": self.period,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# core simulation
# --------------------------------------------------------------------------- #


def _stop_distance(ir: StrategyIR, price: float, atr_value: float) -> float:
    """Price distance from entry to stop."""
    risk = ir.risk
    if risk.stop_kind == "atr":
        return max(atr_value * risk.stop_value, 0.0)
    if risk.stop_kind == "points":
        return risk.stop_value
    if risk.stop_kind == "percent":
        return price * risk.stop_value / 100.0
    # no stop: still need a risk yardstick for sizing and R multiples
    return max(atr_value * 3.0, price * 0.01)


def _target_distance(ir: StrategyIR, price: float, atr_value: float, stop_dist: float) -> float:
    risk = ir.risk
    if risk.target_kind == "none":
        return 0.0
    if risk.target_kind == "atr":
        return atr_value * risk.target_value
    if risk.target_kind == "points":
        return risk.target_value
    if risk.target_kind == "percent":
        return price * risk.target_value / 100.0
    return stop_dist * risk.target_value  # "rr"


def _simulate(
    compiled: CompiledStrategy,
    asset: Asset,
    config: BacktestConfig,
    *,
    start: int,
    end: int,
) -> tuple[list[Trade], np.ndarray, int]:
    """Run the bar loop over ``frame[start:end]``. Returns trades, equity, exposure."""
    ir = compiled.ir
    frame = compiled.frame
    risk_pct = config.risk_per_trade_pct or ir.risk.risk_per_trade_pct

    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    atrs = compiled.atr.to_numpy(dtype=float)
    index = frame.index

    e_long = compiled.entry_long.to_numpy(dtype=bool)
    e_short = compiled.entry_short.to_numpy(dtype=bool)
    x_long = compiled.exit_long.to_numpy(dtype=bool)
    x_short = compiled.exit_short.to_numpy(dtype=bool)

    tick = asset.tick_size or 0.01
    cost = asset.cost
    adverse = (cost.spread_points / 2.0 + cost.slippage_points) * tick * config.cost_multiplier
    commission_per_lot = cost.commission_per_lot * config.cost_multiplier
    point_value = asset.point_value or 1.0

    equity = config.initial_balance
    equity_curve = np.empty(end - start, dtype=float)
    trades: list[Trade] = []
    exposure_bars = 0

    # open position state
    side: Side | None = None
    entry_price = stop_price = target_price = 0.0
    lots = 0.0
    risk_amount = 0.0
    entry_i = 0
    be_done = False
    mae = mfe = 0.0

    pending: Literal["long", "short", "exit"] | None = None
    cooldown_until = -1

    for i in range(start, end):
        o, hi, lo, c = opens[i], highs[i], lows[i], closes[i]
        atr_value = atrs[i]

        # ---------- 1. execute what last bar decided, at this bar's open ----
        if pending == "exit" and side is not None:
            fill = o - adverse if side == "long" else o + adverse
            trades.append(
                _close_trade(
                    side,
                    entry_price,
                    fill,
                    lots,
                    point_value,
                    commission_per_lot,
                    risk_amount,
                    index[entry_i],
                    index[i],
                    i - entry_i,
                    "rule_exit",
                    mae,
                    mfe,
                )
            )
            equity += trades[-1].pnl
            side, pending = None, None
            cooldown_until = i + ir.filters.cooldown_bars
        elif pending in {"long", "short"} and side is None and i >= cooldown_until:
            new_side: Side = "long" if pending == "long" else "short"
            fill = o + adverse if new_side == "long" else o - adverse
            if np.isfinite(atr_value) and atr_value > 0:
                stop_dist = _stop_distance(ir, fill, atr_value)
                target_dist = _target_distance(ir, fill, atr_value, stop_dist)
                sized = _size_position(equity, risk_pct, stop_dist, point_value, fill)
                if sized > 0 and stop_dist > 0:
                    side = new_side
                    entry_price, lots, entry_i = fill, sized, i
                    risk_amount = stop_dist * lots * point_value
                    if ir.risk.stop_kind == "none":
                        stop_price = 0.0
                    else:
                        stop_price = fill - stop_dist if side == "long" else fill + stop_dist
                    target_price = (
                        0.0
                        if target_dist <= 0
                        else (fill + target_dist if side == "long" else fill - target_dist)
                    )
                    be_done = False
                    mae = mfe = 0.0
            pending = None
        else:
            pending = None

        # ---------- 2. manage the open position inside this bar -------------
        if side is not None:
            exposure_bars += 1
            stop_dist_now = (
                abs(entry_price - stop_price)
                if stop_price
                else risk_amount / max(lots * point_value, 1e-9)
            )
            if stop_dist_now > 0:
                if side == "long":
                    mfe = max(mfe, (hi - entry_price) / stop_dist_now)
                    mae = min(mae, (lo - entry_price) / stop_dist_now)
                else:
                    mfe = max(mfe, (entry_price - lo) / stop_dist_now)
                    mae = min(mae, (entry_price - hi) / stop_dist_now)

            hit_stop = bool(stop_price) and (
                lo <= stop_price if side == "long" else hi >= stop_price
            )
            hit_target = bool(target_price) and (
                hi >= target_price if side == "long" else lo <= target_price
            )

            if hit_stop:  # pessimistic tie-break: stop before target
                fill = stop_price - adverse if side == "long" else stop_price + adverse
                reason = "stop"
            elif hit_target:
                fill = target_price - adverse if side == "long" else target_price + adverse
                reason = "target"
            else:
                fill, reason = 0.0, ""

            if reason:
                trades.append(
                    _close_trade(
                        side,
                        entry_price,
                        fill,
                        lots,
                        point_value,
                        commission_per_lot,
                        risk_amount,
                        index[entry_i],
                        index[i],
                        i - entry_i,
                        reason,
                        mae,
                        mfe,
                    )
                )
                equity += trades[-1].pnl
                side = None
                cooldown_until = i + ir.filters.cooldown_bars
            else:
                # time stop
                if ir.risk.max_bars_in_trade and (i - entry_i) >= ir.risk.max_bars_in_trade:
                    fill = c - adverse if side == "long" else c + adverse
                    trades.append(
                        _close_trade(
                            side,
                            entry_price,
                            fill,
                            lots,
                            point_value,
                            commission_per_lot,
                            risk_amount,
                            index[entry_i],
                            index[i],
                            i - entry_i,
                            "time_stop",
                            mae,
                            mfe,
                        )
                    )
                    equity += trades[-1].pnl
                    side = None
                    cooldown_until = i + ir.filters.cooldown_bars
                else:
                    stop_price, be_done = _manage_stop(
                        ir, side, entry_price, stop_price, c, atr_value, stop_dist_now, be_done
                    )

        # ---------- 3. mark to market --------------------------------------
        floating = 0.0
        if side is not None:
            move = (c - entry_price) if side == "long" else (entry_price - c)
            floating = move * lots * point_value
        equity_curve[i - start] = equity + floating

        # ---------- 4. decide for the next bar -----------------------------
        if side is None:
            if e_long[i]:
                pending = "long"
            elif e_short[i]:
                pending = "short"
        elif (side == "long" and x_long[i]) or (side == "short" and x_short[i]):
            pending = "exit"

    # close anything still open at the last close, so equity is honest
    if side is not None:
        last = end - 1
        fill = closes[last] - adverse if side == "long" else closes[last] + adverse
        trades.append(
            _close_trade(
                side,
                entry_price,
                fill,
                lots,
                point_value,
                commission_per_lot,
                risk_amount,
                index[entry_i],
                index[last],
                last - entry_i,
                "end_of_data",
                mae,
                mfe,
            )
        )
        equity += trades[-1].pnl
        equity_curve[-1] = equity

    return trades, equity_curve, exposure_bars


def _manage_stop(
    ir: StrategyIR,
    side: Side,
    entry_price: float,
    stop_price: float,
    close: float,
    atr_value: float,
    stop_dist: float,
    be_done: bool,
) -> tuple[float, bool]:
    """Breakeven move, then trailing. Stops only ever move in our favour."""
    risk = ir.risk
    if stop_price and risk.breakeven_at_r and not be_done and stop_dist > 0:
        gained_r = (
            (close - entry_price) / stop_dist
            if side == "long"
            else (entry_price - close) / stop_dist
        )
        if gained_r >= risk.breakeven_at_r:
            stop_price = (
                max(stop_price, entry_price) if side == "long" else min(stop_price, entry_price)
            )
            be_done = True
    if stop_price and risk.trailing and np.isfinite(atr_value) and atr_value > 0:
        trail = atr_value * risk.trail_atr_mult
        candidate = close - trail if side == "long" else close + trail
        stop_price = max(stop_price, candidate) if side == "long" else min(stop_price, candidate)
    return stop_price, be_done


def _size_position(
    equity: float, risk_pct: float, stop_dist: float, point_value: float, price: float
) -> float:
    if equity <= 0 or stop_dist <= 0:
        return 0.0
    risk_amount = equity * risk_pct / 100.0
    lots = risk_amount / (stop_dist * point_value)
    notional_cap = (equity * MAX_LEVERAGE) / max(price * point_value, 1e-9)
    lots = min(lots, notional_cap)
    lots = math_floor(lots, 2)
    return lots if lots >= MIN_LOTS else 0.0


def math_floor(value: float, digits: int) -> float:
    factor = 10**digits
    return int(value * factor) / factor


def _close_trade(
    side: Side,
    entry_price: float,
    exit_price: float,
    lots: float,
    point_value: float,
    commission_per_lot: float,
    risk_amount: float,
    entry_time: Any,
    exit_time: Any,
    bars_held: int,
    reason: str,
    mae: float,
    mfe: float,
) -> Trade:
    move = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
    pnl = move * lots * point_value - commission_per_lot * lots
    return Trade(
        side=side,
        entry_time=str(entry_time),
        entry_price=round(entry_price, 5),
        exit_time=str(exit_time),
        exit_price=round(exit_price, 5),
        lots=round(lots, 2),
        pnl=round(pnl, 2),
        r=round(pnl / risk_amount, 4) if risk_amount > 0 else 0.0,
        bars_held=bars_held,
        exit_reason=reason,
        mae_r=round(mae, 3),
        mfe_r=round(mfe, 3),
    )


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #


def run_backtest(
    ir: StrategyIR,
    df: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    kind: str = "full",
    compiled: CompiledStrategy | None = None,
    slice_range: tuple[int, int] | None = None,
) -> BacktestResult:
    """Compile (if needed) and simulate. Never raises for strategy-quality reasons."""
    config = config or BacktestConfig.from_factory_config()
    started = time.perf_counter()
    asset = universe().get(ir.asset)

    try:
        compiled = compiled or compile_ir(ir, df)
    except Exception as exc:
        return BacktestResult(
            strategy_id=ir.id,
            asset=ir.asset,
            timeframe=ir.timeframe,
            kind=kind,
            metrics={},
            trades=[],
            equity_curve=[],
            error=f"compile failed: {exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    n = len(compiled.frame)
    start, end = slice_range or (compiled.warmup, n)
    start = max(start, compiled.warmup)
    if end - start < 50:
        return BacktestResult(
            strategy_id=ir.id,
            asset=ir.asset,
            timeframe=ir.timeframe,
            kind=kind,
            metrics={},
            trades=[],
            equity_curve=[],
            error=f"not enough bars after warm-up ({end - start})",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    trades, equity, exposure = _simulate(compiled, asset, config, start=start, end=end)
    bars = end - start
    metrics = compute_metrics(
        equity,
        [t.as_dict() for t in trades],
        bars_per_year=bars_per_year(ir.timeframe, session=asset.session),
        initial_balance=config.initial_balance,
        bars=bars,
        exposure_bars=exposure,
    )
    metrics["robust_score"] = robust_score(metrics, min_trades=config.min_trades)

    eq_series = pd.Series(equity, index=compiled.frame.index[start:end])
    return BacktestResult(
        strategy_id=ir.id,
        asset=ir.asset,
        timeframe=ir.timeframe,
        kind=kind,
        metrics=metrics,
        trades=[t.as_dict() for t in trades[-config.max_trades_stored :]],
        equity_curve=_downsample(eq_series, config.equity_curve_points),
        monthly=monthly_returns(eq_series),
        diagnostics={**compiled.diagnostics, "signals": compiled.signal_counts()},
        period={"start": str(eq_series.index[0]), "end": str(eq_series.index[-1])},
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


def split_backtest(
    ir: StrategyIR, df: pd.DataFrame, *, config: BacktestConfig | None = None
) -> dict[str, Any]:
    """The standard verdict input: full, in-sample, out-of-sample + degradation.

    The OOS tail is never shown to the tuner, which is the whole point.
    """
    config = config or BacktestConfig.from_factory_config()
    compiled = None
    try:
        compiled = compile_ir(ir, df)
    except Exception as exc:
        return {"error": f"compile failed: {exc}"}

    n = len(compiled.frame)
    split_at = compiled.warmup + int((n - compiled.warmup) * (1 - config.oos_fraction))

    full = run_backtest(ir, df, config=config, kind="full", compiled=compiled)
    is_run = run_backtest(
        ir,
        df,
        config=config,
        kind="is",
        compiled=compiled,
        slice_range=(compiled.warmup, split_at),
    )
    oos_run = run_backtest(
        ir, df, config=config, kind="oos", compiled=compiled, slice_range=(split_at, n)
    )

    oos_metrics = oos_run.metrics or {}
    is_metrics = is_run.metrics or {}
    score = robust_score(oos_metrics, min_trades=config.min_trades, is_metrics=is_metrics)

    is_sharpe = float(is_metrics.get("sharpe", 0.0))
    oos_sharpe = float(oos_metrics.get("sharpe", 0.0))
    retention = round(oos_sharpe / is_sharpe, 3) if is_sharpe > 0 else 0.0

    return {
        "full": full.as_dict(),
        "is": is_run.as_dict(),
        "oos": oos_run.as_dict(),
        "split_index": split_at,
        "robust_score": score,
        "sharpe_retention": retention,
        "compiled_diagnostics": compiled.diagnostics,
    }


def walk_forward(
    ir: StrategyIR,
    df: pd.DataFrame,
    *,
    folds: int = 4,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """Anchored walk-forward: does the edge survive across separate eras?"""
    config = config or BacktestConfig.from_factory_config()
    try:
        compiled = compile_ir(ir, df)
    except Exception as exc:
        return {"error": f"compile failed: {exc}", "folds": []}

    n = len(compiled.frame)
    usable = n - compiled.warmup
    if usable < folds * 200:
        return {"error": f"not enough history for {folds} folds", "folds": []}

    size = usable // folds
    results: list[dict[str, Any]] = []
    for k in range(folds):
        start = compiled.warmup + k * size
        end = n if k == folds - 1 else start + size
        run = run_backtest(
            ir, df, config=config, kind=f"wf{k + 1}", compiled=compiled, slice_range=(start, end)
        )
        results.append(
            {
                "fold": k + 1,
                "period": run.period,
                "trades": run.metrics.get("trades", 0),
                "sharpe": run.metrics.get("sharpe", 0.0),
                "profit_factor": run.metrics.get("profit_factor", 0.0),
                "return_pct": run.metrics.get("total_return_pct", 0.0),
                "max_drawdown_pct": run.metrics.get("max_drawdown_pct", 0.0),
            }
        )

    sharpes = [float(r["sharpe"]) for r in results]
    profitable = sum(1 for r in results if float(r["return_pct"]) > 0)
    return {
        "folds": results,
        "profitable_folds": profitable,
        "fold_count": folds,
        "consistency": round(profitable / folds, 3),
        "mean_sharpe": round(float(np.mean(sharpes)), 3) if sharpes else 0.0,
        "worst_sharpe": round(float(np.min(sharpes)), 3) if sharpes else 0.0,
        "sharpe_std": round(float(np.std(sharpes)), 3) if sharpes else 0.0,
    }


def _downsample(series: pd.Series, points: int) -> list[dict[str, Any]]:
    if series.empty:
        return []
    step = max(1, len(series) // max(points, 1))
    sampled = series.iloc[::step]
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
    return [{"t": str(ts), "equity": round(float(val), 2)} for ts, val in sampled.items()]
