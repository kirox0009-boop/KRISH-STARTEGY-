"""Price providers + Parquet cache.

Phase 0 uses free Yahoo data, which is good enough to prove the whole pipeline
but has real limits (intraday history is capped, no true tick/spread data). Phase
1 swaps in ccxt for crypto and MT5/Dukascopy exports for FX and CFDs — callers
only ever touch :func:`fetch_ohlcv`, so nothing above this file changes.

Every frame returned is: DatetimeIndex (UTC, sorted, unique), lowercase columns
``open, high, low, close, volume``, no NaNs in OHLC, no zero/negative prices.
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..assets import TIMEFRAME_MINUTES, universe
from ..config import CACHE_DIR
from ..storage import cache_key, offload_price_cache, price_cache_mode, store

log = logging.getLogger("krish.data")

# Yahoo's hard limits on how far back intraday data goes.
YF_INTRADAY_LIMIT_DAYS = {"M1": 7, "M5": 59, "M15": 59, "M30": 59, "H1": 729, "H4": 729}
YF_INTERVAL = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "1h",
    "D1": "1d",
    "W1": "1wk",
}
CACHE_TTL_SECONDS = int(os.getenv("KRISH_CACHE_TTL", "3600"))


class DataError(RuntimeError):
    """No usable price data could be produced."""


def _cache_path(asset_key: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{asset_key.upper()}_{timeframe.upper()}.parquet"


#: In-memory price cache, used when local disk is being avoided. A frame is
#: roughly 0.5 MB, so the whole universe costs single-digit megabytes of RAM -
#: cheaper than the disk it replaces, and it disappears on restart rather than
#: accumulating.
_MEM: dict[tuple[str, str], tuple[pd.DataFrame, float]] = {}


def _mem_get(asset: str, timeframe: str, max_age: float) -> pd.DataFrame | None:
    hit = _MEM.get((asset, timeframe))
    if hit is None:
        return None
    frame, ts = hit
    return frame if (time.time() - ts) < max_age else None


def cache_status() -> list[dict[str, Any]]:
    """What the dashboard shows on its data-health panel."""
    out: list[dict[str, Any]] = []
    memory_mode = price_cache_mode() == "memory"
    for asset in universe().all():
        for tf in asset.timeframes:
            if memory_mode:
                hit = _MEM.get((asset.key, tf))
                out.append(
                    {
                        "asset": asset.key,
                        "timeframe": tf,
                        "cached": hit is not None,
                        "where": "memory",
                        "bars": len(hit[0]) if hit else None,
                        "start": str(hit[0].index[0]) if hit else None,
                        "end": str(hit[0].index[-1]) if hit else None,
                        "age_seconds": int(time.time() - hit[1]) if hit else None,
                    }
                )
                continue
            path = _cache_path(asset.key, tf)
            if not path.exists():
                out.append({"asset": asset.key, "timeframe": tf, "cached": False})
                continue
            try:
                df = pd.read_parquet(path)
                out.append(
                    {
                        "asset": asset.key,
                        "timeframe": tf,
                        "cached": True,
                        "bars": len(df),
                        "start": str(df.index[0]) if len(df) else None,
                        "end": str(df.index[-1]) if len(df) else None,
                        "age_seconds": int(time.time() - path.stat().st_mtime),
                    }
                )
            except Exception as exc:  # pragma: no cover
                out.append({"asset": asset.key, "timeframe": tf, "cached": True, "error": str(exc)})
    return out


def load_cached(asset_key: str, timeframe: str) -> pd.DataFrame | None:
    path = _cache_path(asset_key, timeframe)
    if not path.exists():
        return None
    try:
        return _clean(pd.read_parquet(path))
    except Exception:  # pragma: no cover
        log.exception("corrupt cache file %s; deleting", path)
        path.unlink(missing_ok=True)
        return None


def fetch_ohlcv(
    asset_key: str,
    timeframe: str = "H1",
    *,
    years: int | None = None,
    refresh: bool = False,
    allow_synthetic: bool | None = None,
) -> pd.DataFrame:
    """Return clean OHLCV for an asset/timeframe, using the cache when fresh."""
    asset = universe().get(asset_key)
    timeframe = timeframe.upper()
    years = years or universe().history_years

    if price_cache_mode() == "memory":
        return _fetch_memory_only(asset, timeframe, years, refresh, allow_synthetic)

    path = _cache_path(asset.key, timeframe)

    if not refresh and path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            cached = load_cached(asset.key, timeframe)
            if cached is not None and len(cached) > 200:
                return cached

    # Nothing usable locally: the librarian may have pruned it after uploading.
    # Pulling it back from object storage beats re-downloading years of history.
    pruned_locally = not refresh and not path.exists() and offload_price_cache()
    if pruned_locally and store().get(cache_key(asset.key, timeframe), path):
        restored = load_cached(asset.key, timeframe)
        if restored is not None and len(restored) > 200:
            return restored

    frame: pd.DataFrame | None = None
    symbol = asset.symbol_for("yfinance")
    if symbol:
        try:
            frame = _fetch_yfinance(symbol, timeframe, years)
        except Exception as exc:
            log.warning("yfinance fetch failed for %s (%s): %s", asset.key, symbol, exc)

    if frame is None or len(frame) < 200:
        stale = load_cached(asset.key, timeframe)
        if stale is not None and len(stale) > 200:
            log.warning("using stale cache for %s %s", asset.key, timeframe)
            return stale
        if allow_synthetic is None:
            allow_synthetic = os.getenv("KRISH_ALLOW_SYNTHETIC", "0") == "1"
        if allow_synthetic:
            log.warning(
                "SYNTHETIC data for %s %s - results are for plumbing tests only",
                asset.key,
                timeframe,
            )
            frame = _synthetic(asset.key, timeframe, years)
        else:
            raise DataError(
                f"no data for {asset.key} {timeframe}: live fetch failed and no usable cache. "
                "Set KRISH_ALLOW_SYNTHETIC=1 only for plumbing tests."
            )

    frame = _clean(frame)
    if len(frame) < 200:
        raise DataError(f"only {len(frame)} clean bars for {asset.key} {timeframe}")

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    log.info("cached %s bars for %s %s", len(frame), asset.key, timeframe)
    if offload_price_cache():
        store().put(path, cache_key(asset.key, timeframe))
    return frame


def _fetch_memory_only(
    asset: Any, timeframe: str, years: int, refresh: bool, allow_synthetic: bool | None
) -> pd.DataFrame:
    """Zero-local-disk path: RAM, then object storage, then the provider.

    Nothing is ever written to the VPS filesystem. The object store keeps a copy
    so a restart does not have to re-download years of history, but if there is no
    object store configured this simply falls back to re-fetching, which costs
    seconds and no disk.
    """
    key = (asset.key, timeframe)

    if not refresh:
        cached = _mem_get(asset.key, timeframe, CACHE_TTL_SECONDS)
        if cached is not None and len(cached) > 200:
            return cached

        if store().enabled:
            raw = store().get_bytes(cache_key(asset.key, timeframe))
            if raw:
                try:
                    frame = _clean(pd.read_parquet(io.BytesIO(raw)))
                    if len(frame) > 200:
                        _MEM[key] = (frame, time.time())
                        log.info(
                            "%s %s restored from object storage (%s bars, no local file)",
                            asset.key,
                            timeframe,
                            len(frame),
                        )
                        return frame
                except Exception:
                    log.warning("could not read %s %s from object storage", asset.key, timeframe)

    frame: pd.DataFrame | None = None
    symbol = asset.symbol_for("yfinance")
    if symbol:
        try:
            frame = _fetch_yfinance(symbol, timeframe, years)
        except Exception as exc:
            log.warning("yfinance fetch failed for %s (%s): %s", asset.key, symbol, exc)

    if frame is None or len(frame) < 200:
        stale = _MEM.get(key)
        if stale is not None and len(stale[0]) > 200:
            log.warning("using stale in-memory data for %s %s", asset.key, timeframe)
            return stale[0]
        if allow_synthetic is None:
            allow_synthetic = os.getenv("KRISH_ALLOW_SYNTHETIC", "0") == "1"
        if allow_synthetic:
            log.warning("SYNTHETIC data for %s %s - plumbing tests only", asset.key, timeframe)
            frame = _synthetic(asset.key, timeframe, years)
        else:
            raise DataError(
                f"no data for {asset.key} {timeframe}: live fetch failed and nothing "
                "cached in memory or object storage."
            )

    frame = _clean(frame)
    if len(frame) < 200:
        raise DataError(f"only {len(frame)} clean bars for {asset.key} {timeframe}")

    _MEM[key] = (frame, time.time())
    if store().enabled:
        buf = io.BytesIO()
        frame.to_parquet(buf)
        store().put_bytes(buf.getvalue(), cache_key(asset.key, timeframe))
    log.info("%s %s held in memory (%s bars, 0 bytes on disk)", asset.key, timeframe, len(frame))
    return frame


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #


def _fetch_yfinance(symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    import yfinance as yf  # heavy import, keep it lazy

    interval = YF_INTERVAL.get(timeframe)
    if interval is None:
        raise DataError(f"timeframe {timeframe} not supported by the yfinance provider")

    end = datetime.now(UTC)
    days = years * 365
    limit = YF_INTRADAY_LIMIT_DAYS.get(timeframe)
    if limit:
        days = min(days, limit)
    start = end - timedelta(days=days)

    raw = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise DataError(f"yfinance returned nothing for {symbol} {interval}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    frame = raw.rename(columns={"adj close": "adj_close"})
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
    frame = frame[keep]

    if timeframe == "H4":  # Yahoo has no 4h bar; build it from 1h
        frame = _resample(frame, 240)
    return frame


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}min"
    agg: dict[str, Any] = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open", "close"])


def _synthetic(asset_key: str, timeframe: str, years: int) -> pd.DataFrame:
    """Deterministic pseudo-market for plumbing tests only.

    Regime-switching GBM with volatility clustering — enough structure that
    strategies produce trades, never used for real verdicts.
    """
    minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
    bars = min(int(years * 365 * 24 * 60 / minutes), 40_000)
    rng = np.random.default_rng(abs(hash((asset_key, timeframe))) % (2**32))

    base = {"GOLD": 1900.0, "US30": 34_000.0, "US100": 15_000.0, "BITCOIN": 45_000.0, "OIL": 78.0}
    price = base.get(asset_key.upper(), 100.0)
    per_bar_vol = 0.012 * np.sqrt(minutes / 1440)

    prices = np.empty(bars)
    vol = per_bar_vol
    drift = 0.0
    for i in range(bars):
        if rng.random() < 0.002:  # regime switch
            drift = rng.normal(0, per_bar_vol * 0.4)
            vol = per_bar_vol * rng.uniform(0.5, 2.2)
        vol = 0.97 * vol + 0.03 * per_bar_vol * rng.uniform(0.6, 1.8)
        price *= float(np.exp(drift + vol * rng.standard_normal()))
        prices[i] = price

    close = pd.Series(prices)
    open_ = close.shift(1).fillna(close.iloc[0])
    spread = close * per_bar_vol
    high = np.maximum(open_, close) + spread * rng.uniform(0.1, 0.9, bars)
    low = np.minimum(open_, close) - spread * rng.uniform(0.1, 0.9, bars)

    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=bars, freq=f"{minutes}min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": open_.to_numpy(),
            "high": high,
            "low": low,
            "close": close.to_numpy(),
            "volume": rng.integers(500, 5000, bars).astype(float),
        },
        index=index,
    )
    frame.attrs["synthetic"] = True
    return frame


# --------------------------------------------------------------------------- #
# cleaning — the data quality gate
# --------------------------------------------------------------------------- #


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(f"price frame missing {missing}")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    frame = frame[[*required, "volume"]].astype(float)
    frame = frame.dropna(subset=required)
    frame = frame[(frame[required] > 0).all(axis=1)]

    # geometry must hold: high >= max(open, close), low <= min(open, close)
    frame["high"] = frame[["high", "open", "close"]].max(axis=1)
    frame["low"] = frame[["low", "open", "close"]].min(axis=1)

    # kill absurd single-bar jumps (bad ticks), keep real gaps
    ret = frame["close"].pct_change().abs()
    frame = frame[(ret < 0.35) | ret.isna()]

    return frame
