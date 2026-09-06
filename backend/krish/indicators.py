"""Indicator library.

Every indicator here is **causal**: the value at bar *i* uses only bars <= i.
That is enforced by construction (no ``center=True``, no negative shifts) and
re-checked by the developer agent's look-ahead audit.

Adding an indicator = one function + one entry in ``REGISTRY``. The architect
agent can then use it immediately, because it discovers indicators from this
registry rather than from hardcoded lists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

Frame = pd.DataFrame
Series = pd.Series


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def _src(df: Frame, source: str) -> Series:
    source = source.lower()
    if source in df.columns:
        return df[source].astype(float)
    if source == "hl2":
        return (df["high"] + df["low"]) / 2.0
    if source == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if source == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    raise KeyError(f"unknown price source '{source}'")


def sma(df: Frame, period: int = 20, source: str = "close") -> Series:
    return _src(df, source).rolling(int(period), min_periods=int(period)).mean()


def ema(df: Frame, period: int = 20, source: str = "close") -> Series:
    return _src(df, source).ewm(span=int(period), adjust=False, min_periods=int(period)).mean()


def wma(df: Frame, period: int = 20, source: str = "close") -> Series:
    period = int(period)
    weights = np.arange(1, period + 1, dtype=float)
    return (
        _src(df, source)
        .rolling(period, min_periods=period)
        .apply(lambda w: float(np.dot(w, weights) / weights.sum()), raw=True)
    )


def rsi(df: Frame, period: int = 14, source: str = "close") -> Series:
    delta = _src(df, source).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    period = int(period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0).where(avg_gain.notna())


def true_range(df: Frame) -> Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: Frame, period: int = 14) -> Series:
    period = int(period)
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_pct(df: Frame, period: int = 14) -> Series:
    return atr(df, period) / df["close"] * 100.0


def stddev(df: Frame, period: int = 20, source: str = "close") -> Series:
    return _src(df, source).rolling(int(period), min_periods=int(period)).std(ddof=0)


def bb_upper(df: Frame, period: int = 20, mult: float = 2.0, source: str = "close") -> Series:
    return sma(df, period, source) + float(mult) * stddev(df, period, source)


def bb_lower(df: Frame, period: int = 20, mult: float = 2.0, source: str = "close") -> Series:
    return sma(df, period, source) - float(mult) * stddev(df, period, source)


def bb_percent(df: Frame, period: int = 20, mult: float = 2.0, source: str = "close") -> Series:
    upper, lower = bb_upper(df, period, mult, source), bb_lower(df, period, mult, source)
    width = (upper - lower).replace(0.0, np.nan)
    return (_src(df, source) - lower) / width * 100.0


def macd(df: Frame, fast: int = 12, slow: int = 26, source: str = "close") -> Series:
    return ema(df, fast, source) - ema(df, slow, source)


def macd_signal(
    df: Frame, fast: int = 12, slow: int = 26, signal: int = 9, source: str = "close"
) -> Series:
    line = macd(df, fast, slow, source)
    return line.ewm(span=int(signal), adjust=False, min_periods=int(signal)).mean()


def macd_hist(
    df: Frame, fast: int = 12, slow: int = 26, signal: int = 9, source: str = "close"
) -> Series:
    return macd(df, fast, slow, source) - macd_signal(df, fast, slow, signal, source)


def donchian_high(df: Frame, period: int = 20) -> Series:
    return df["high"].rolling(int(period), min_periods=int(period)).max()


def donchian_low(df: Frame, period: int = 20) -> Series:
    return df["low"].rolling(int(period), min_periods=int(period)).min()


def donchian_mid(df: Frame, period: int = 20) -> Series:
    return (donchian_high(df, period) + donchian_low(df, period)) / 2.0


def highest(df: Frame, period: int = 20, source: str = "high") -> Series:
    return _src(df, source).rolling(int(period), min_periods=int(period)).max()


def lowest(df: Frame, period: int = 20, source: str = "low") -> Series:
    return _src(df, source).rolling(int(period), min_periods=int(period)).min()


def roc(df: Frame, period: int = 10, source: str = "close") -> Series:
    s = _src(df, source)
    return (s / s.shift(int(period)) - 1.0) * 100.0


def momentum(df: Frame, period: int = 10, source: str = "close") -> Series:
    s = _src(df, source)
    return s - s.shift(int(period))


def stoch_k(df: Frame, period: int = 14, smooth: int = 3) -> Series:
    hh = df["high"].rolling(int(period), min_periods=int(period)).max()
    ll = df["low"].rolling(int(period), min_periods=int(period)).min()
    raw = (df["close"] - ll) / (hh - ll).replace(0.0, np.nan) * 100.0
    return raw.rolling(int(smooth), min_periods=int(smooth)).mean()


def cci(df: Frame, period: int = 20) -> Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(int(period), min_periods=int(period)).mean()
    md = (tp - ma).abs().rolling(int(period), min_periods=int(period)).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, np.nan))


def adx(df: Frame, period: int = 14) -> Series:
    period = int(period)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0.0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0.0)
    tr = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap_session(df: Frame) -> Series:
    """Rolling session VWAP; falls back to typical price when volume is absent."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    if "volume" not in df.columns or df["volume"].fillna(0).sum() <= 0:
        return tp.expanding().mean()
    day = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else None
    if day is None:
        return (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    grouped = pd.DataFrame({"pv": tp * df["volume"], "v": df["volume"]}).groupby(day).cumsum()
    return grouped["pv"] / grouped["v"].replace(0.0, np.nan)


def zscore(df: Frame, period: int = 50, source: str = "close") -> Series:
    s = _src(df, source)
    mean = s.rolling(int(period), min_periods=int(period)).mean()
    sd = s.rolling(int(period), min_periods=int(period)).std(ddof=0).replace(0.0, np.nan)
    return (s - mean) / sd


def keltner_upper(df: Frame, period: int = 20, mult: float = 2.0) -> Series:
    return ema(df, period) + float(mult) * atr(df, period)


def keltner_lower(df: Frame, period: int = 20, mult: float = 2.0) -> Series:
    return ema(df, period) - float(mult) * atr(df, period)


# --------------------------------------------------------------------------- #
# Market structure / "smart money" concepts
#
# These are the patterns discretionary traders actually talk about: swing
# structure, fair value gaps, order blocks, liquidity sweeps, premium vs discount.
# They are heuristics, not established factors - popular does not mean profitable -
# so they are implemented faithfully and then judged by the same out-of-sample
# machinery as everything else. The point is to let the evidence decide.
#
# CAUSALITY IS THE WHOLE GAME HERE. A swing high is only knowable `right` bars
# after it forms, and every function below shifts its result by exactly that much.
# Getting this wrong would produce spectacular backtests that cannot be traded,
# which is the single easiest way for this project to start lying.
# --------------------------------------------------------------------------- #


def _pivot_mask(series: Series, left: int, right: int, *, high: bool) -> Series:
    """True where ``series`` is a local extreme with ``left`` older and ``right``
    newer bars on either side. Uses future bars by design; callers must shift."""
    cond = pd.Series(True, index=series.index)
    for k in range(1, int(left) + 1):
        cond &= series >= series.shift(k) if high else series <= series.shift(k)
    for k in range(1, int(right) + 1):
        cond &= series >= series.shift(-k) if high else series <= series.shift(-k)
    return cond.fillna(False)


def swing_high_level(df: Frame, left: int = 2, right: int = 2) -> Series:
    """Price of the most recent *confirmed* swing high, held until the next one.

    Shifted by ``right`` so the level only becomes visible once the market has
    actually printed the bars that confirm it.
    """
    right = int(right)
    mask = _pivot_mask(df["high"], left, right, high=True)
    return df["high"].where(mask).shift(right).ffill()


def swing_low_level(df: Frame, left: int = 2, right: int = 2) -> Series:
    right = int(right)
    mask = _pivot_mask(df["low"], left, right, high=False)
    return df["low"].where(mask).shift(right).ffill()


def fvg_bull_level(df: Frame) -> Series:
    """Midpoint of the most recent bullish fair value gap.

    A three-bar imbalance: this bar's low sits above the high two bars back, so
    the market skipped a price band on the way up. Uses only bars <= i.
    """
    gap = df["low"] > df["high"].shift(2)
    level = (df["high"].shift(2) + df["low"]) / 2.0
    return level.where(gap).ffill()


def fvg_bear_level(df: Frame) -> Series:
    gap = df["high"] < df["low"].shift(2)
    level = (df["low"].shift(2) + df["high"]) / 2.0
    return level.where(gap).ffill()


def ob_bull_level(df: Frame) -> Series:
    """Low of the last down-close candle before an up-impulse (bullish order block).

    Detected when a bar closes above the previous bar's high and that previous bar
    closed down - the classic "last bearish candle before the move" reading.
    """
    impulse = (df["close"] > df["high"].shift(1)) & (df["close"].shift(1) < df["open"].shift(1))
    return df["low"].shift(1).where(impulse).ffill()


def ob_bear_level(df: Frame) -> Series:
    impulse = (df["close"] < df["low"].shift(1)) & (df["close"].shift(1) > df["open"].shift(1))
    return df["high"].shift(1).where(impulse).ffill()


def liquidity_sweep_high(df: Frame, left: int = 2, right: int = 2) -> Series:
    """100 when this bar ran the stops above a swing high and closed back below it.

    The wick takes out the level, the body does not hold it - a failed breakout,
    which is what "liquidity sweep" describes.
    """
    level = swing_high_level(df, left, right)
    hit = (df["high"] > level) & (df["close"] < level) & level.notna()
    return hit.astype(float) * 100.0


def liquidity_sweep_low(df: Frame, left: int = 2, right: int = 2) -> Series:
    level = swing_low_level(df, left, right)
    hit = (df["low"] < level) & (df["close"] > level) & level.notna()
    return hit.astype(float) * 100.0


def equilibrium(df: Frame, period: int = 50) -> Series:
    """Midpoint of the recent range. Above it is premium, below it is discount."""
    period = int(period)
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    return (hh + ll) / 2.0


def displacement(df: Frame, period: int = 14) -> Series:
    """Candle body measured in ATRs — how impulsive this bar was."""
    body = (df["close"] - df["open"]).abs()
    return body / atr(df, period).replace(0.0, np.nan)


def wick_up_pct(df: Frame) -> Series:
    """Upper wick as a percentage of the bar's range: rejection from above."""
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return (df["high"] - df[["open", "close"]].max(axis=1)) / rng * 100.0


def wick_down_pct(df: Frame) -> Series:
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return (df[["open", "close"]].min(axis=1) - df["low"]) / rng * 100.0


# --------------------------------------------------------------------------- #
# registry — this is what the architect agent browses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IndicatorMeta:
    fn: Callable[..., Series]
    family: str  # trend | momentum | volatility | oscillator | level
    #: tunable params -> (low, high, is_int)
    params: dict[str, tuple[float, float, bool]]
    #: comparable scale: "price" can be compared to price, "0-100" to constants
    scale: str = "price"
    takes_source: bool = True
    doc: str = ""


REGISTRY: dict[str, IndicatorMeta] = {
    "sma": IndicatorMeta(sma, "trend", {"period": (5, 300, True)}, "price", True, "Simple MA"),
    "ema": IndicatorMeta(ema, "trend", {"period": (5, 300, True)}, "price", True, "Exponential MA"),
    "wma": IndicatorMeta(wma, "trend", {"period": (5, 200, True)}, "price", True, "Weighted MA"),
    "donchian_high": IndicatorMeta(
        donchian_high, "level", {"period": (10, 200, True)}, "price", False, "N-bar high"
    ),
    "donchian_low": IndicatorMeta(
        donchian_low, "level", {"period": (10, 200, True)}, "price", False, "N-bar low"
    ),
    "donchian_mid": IndicatorMeta(
        donchian_mid, "level", {"period": (10, 200, True)}, "price", False, "Donchian midline"
    ),
    "bb_upper": IndicatorMeta(
        bb_upper, "volatility", {"period": (10, 100, True), "mult": (1.0, 3.5, False)}, "price"
    ),
    "bb_lower": IndicatorMeta(
        bb_lower, "volatility", {"period": (10, 100, True), "mult": (1.0, 3.5, False)}, "price"
    ),
    "keltner_upper": IndicatorMeta(
        keltner_upper,
        "volatility",
        {"period": (10, 100, True), "mult": (1.0, 3.5, False)},
        "price",
        False,
    ),
    "keltner_lower": IndicatorMeta(
        keltner_lower,
        "volatility",
        {"period": (10, 100, True), "mult": (1.0, 3.5, False)},
        "price",
        False,
    ),
    "vwap": IndicatorMeta(vwap_session, "level", {}, "price", False, "Session VWAP"),
    "rsi": IndicatorMeta(rsi, "oscillator", {"period": (5, 50, True)}, "0-100", True),
    "stoch_k": IndicatorMeta(
        stoch_k, "oscillator", {"period": (5, 50, True), "smooth": (1, 10, True)}, "0-100", False
    ),
    "bb_percent": IndicatorMeta(
        bb_percent, "oscillator", {"period": (10, 100, True), "mult": (1.0, 3.5, False)}, "0-100"
    ),
    "adx": IndicatorMeta(adx, "momentum", {"period": (7, 40, True)}, "0-100", False),
    "cci": IndicatorMeta(cci, "oscillator", {"period": (10, 60, True)}, "unbounded", False),
    "macd": IndicatorMeta(
        macd, "momentum", {"fast": (5, 30, True), "slow": (20, 80, True)}, "zero-centered"
    ),
    "macd_signal": IndicatorMeta(
        macd_signal,
        "momentum",
        {"fast": (5, 30, True), "slow": (20, 80, True), "signal": (3, 20, True)},
        "zero-centered",
    ),
    "macd_hist": IndicatorMeta(
        macd_hist,
        "momentum",
        {"fast": (5, 30, True), "slow": (20, 80, True), "signal": (3, 20, True)},
        "zero-centered",
    ),
    "roc": IndicatorMeta(roc, "momentum", {"period": (2, 60, True)}, "zero-centered"),
    "momentum": IndicatorMeta(momentum, "momentum", {"period": (2, 60, True)}, "zero-centered"),
    "zscore": IndicatorMeta(zscore, "oscillator", {"period": (20, 200, True)}, "zero-centered"),
    "atr": IndicatorMeta(atr, "volatility", {"period": (7, 50, True)}, "unbounded", False),
    "atr_pct": IndicatorMeta(atr_pct, "volatility", {"period": (7, 50, True)}, "unbounded", False),
    "stddev": IndicatorMeta(stddev, "volatility", {"period": (10, 100, True)}, "unbounded"),
    "highest": IndicatorMeta(highest, "level", {"period": (5, 200, True)}, "price"),
    "lowest": IndicatorMeta(lowest, "level", {"period": (5, 200, True)}, "price"),
    # --- market structure / SMC -------------------------------------------
    "swing_high_level": IndicatorMeta(
        swing_high_level,
        "structure",
        {"left": (1, 8, True), "right": (1, 8, True)},
        "price",
        False,
        "Last confirmed swing high",
    ),
    "swing_low_level": IndicatorMeta(
        swing_low_level,
        "structure",
        {"left": (1, 8, True), "right": (1, 8, True)},
        "price",
        False,
        "Last confirmed swing low",
    ),
    "fvg_bull_level": IndicatorMeta(
        fvg_bull_level,
        "structure",
        {},
        "price",
        False,
        "Bullish fair value gap midpoint",
    ),
    "fvg_bear_level": IndicatorMeta(
        fvg_bear_level,
        "structure",
        {},
        "price",
        False,
        "Bearish fair value gap midpoint",
    ),
    "ob_bull_level": IndicatorMeta(
        ob_bull_level,
        "structure",
        {},
        "price",
        False,
        "Bullish order block low",
    ),
    "ob_bear_level": IndicatorMeta(
        ob_bear_level,
        "structure",
        {},
        "price",
        False,
        "Bearish order block high",
    ),
    "liquidity_sweep_high": IndicatorMeta(
        liquidity_sweep_high,
        "structure",
        {"left": (1, 8, True), "right": (1, 8, True)},
        "binary",
        False,
        "Stops run above a swing high, closed back below",
    ),
    "liquidity_sweep_low": IndicatorMeta(
        liquidity_sweep_low,
        "structure",
        {"left": (1, 8, True), "right": (1, 8, True)},
        "binary",
        False,
        "Stops run below a swing low, closed back above",
    ),
    "equilibrium": IndicatorMeta(
        equilibrium,
        "structure",
        {"period": (20, 200, True)},
        "price",
        False,
        "Range midpoint: premium above, discount below",
    ),
    "displacement": IndicatorMeta(
        displacement,
        "volatility",
        {"period": (7, 50, True)},
        "unbounded",
        False,
        "Candle body in ATRs",
    ),
    "wick_up_pct": IndicatorMeta(
        wick_up_pct,
        "oscillator",
        {},
        "0-100",
        False,
        "Upper wick as % of range",
    ),
    "wick_down_pct": IndicatorMeta(
        wick_down_pct,
        "oscillator",
        {},
        "0-100",
        False,
        "Lower wick as % of range",
    ),
}

PRICE_SOURCES = ("close", "open", "high", "low", "hl2", "hlc3", "ohlc4")


def compute(name: str, df: Frame, params: dict[str, Any], source: str = "close") -> Series:
    """Compute one indicator by registry name."""
    try:
        meta = REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown indicator '{name}'. known: {sorted(REGISTRY)}") from exc
    kwargs: dict[str, Any] = {}
    for key, (_low, _high, is_int) in meta.params.items():
        if key in params:
            kwargs[key] = int(params[key]) if is_int else float(params[key])
    if meta.takes_source:
        kwargs["source"] = source
    return meta.fn(df, **kwargs)


def min_bars_needed(name: str, params: dict[str, Any]) -> int:
    """Warm-up bars an indicator needs before it produces a usable value."""
    meta = REGISTRY.get(name)
    if meta is None:
        return 50
    longest = 0
    for key in meta.params:
        if key in params:
            with_val = params[key]
            if isinstance(with_val, (int, float)):
                longest = max(longest, int(with_val))
    return max(longest * 3, 20)


def indicators_by_family() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, meta in REGISTRY.items():
        out.setdefault(meta.family, []).append(name)
    return {k: sorted(v) for k, v in sorted(out.items())}
