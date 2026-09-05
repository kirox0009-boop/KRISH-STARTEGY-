"""Tradable universe, loaded from config/assets.yaml.

Adding an asset is a config edit, never a code edit. Every asset carries its own
cost model so the backtester charges GOLD differently from OIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import load_yaml, save_yaml


@dataclass(slots=True, frozen=True)
class CostModel:
    """Pessimistic-by-design execution costs, in price points."""

    spread_points: float = 10.0
    commission_per_lot: float = 0.0
    slippage_points: float = 5.0

    def round_trip_points(self) -> float:
        """Points lost per completed trade (both sides)."""
        return self.spread_points + 2 * self.slippage_points


@dataclass(slots=True, frozen=True)
class Asset:
    key: str
    name: str
    asset_class: str
    tick_size: float = 0.01
    point_value: float = 1.0
    session: str = "24h"
    timeframes: tuple[str, ...] = ("H1", "D1")
    cost: CostModel = field(default_factory=CostModel)
    symbols: dict[str, str] = field(default_factory=dict)

    def symbol_for(self, venue: str) -> str | None:
        """venue in {yfinance, ccxt, mt5, tradingview}."""
        return self.symbols.get(venue)


class AssetUniverse:
    """In-memory view of config/assets.yaml."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._defaults = dict(raw.get("defaults") or {})
        self._assets: dict[str, Asset] = {}
        for entry in raw.get("assets") or []:
            asset = self._parse(entry)
            self._assets[asset.key] = asset

    def _parse(self, entry: dict[str, Any]) -> Asset:
        cost_raw = dict(entry.get("cost") or {})
        timeframes = entry.get("timeframes") or self._defaults.get("timeframes") or ["H1", "D1"]
        return Asset(
            key=str(entry["key"]).upper(),
            name=entry.get("name", entry["key"]),
            asset_class=entry.get("class", "unknown"),
            tick_size=float(entry.get("tick_size", 0.01)),
            point_value=float(entry.get("point_value", 1.0)),
            session=entry.get("session") or self._defaults.get("session", "24h"),
            timeframes=tuple(timeframes),
            cost=CostModel(
                spread_points=float(cost_raw.get("spread_points", 10.0)),
                commission_per_lot=float(cost_raw.get("commission_per_lot", 0.0)),
                slippage_points=float(cost_raw.get("slippage_points", 5.0)),
            ),
            symbols={
                venue: str(entry[venue])
                for venue in ("yfinance", "ccxt", "mt5", "tradingview")
                if entry.get(venue)
            },
        )

    @property
    def history_years(self) -> int:
        return int(self._defaults.get("history_years", 8))

    def keys(self) -> list[str]:
        return list(self._assets)

    def all(self) -> list[Asset]:
        return list(self._assets.values())

    def get(self, key: str) -> Asset:
        try:
            return self._assets[key.upper()]
        except KeyError as exc:
            raise KeyError(f"unknown asset '{key}'. known: {self.keys()}") from exc

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.upper() in self._assets

    def __len__(self) -> int:
        return len(self._assets)


_universe: AssetUniverse | None = None


def universe(*, refresh: bool = False) -> AssetUniverse:
    global _universe
    if _universe is None or refresh:
        _universe = AssetUniverse(load_yaml("assets", refresh=refresh))
    return _universe


def add_asset(entry: dict[str, Any]) -> Asset:
    """Append an asset to config/assets.yaml and reload. Used by the UI."""
    raw = load_yaml("assets", refresh=True)
    key = str(entry["key"]).upper()
    assets = list(raw.get("assets") or [])
    if any(str(a.get("key", "")).upper() == key for a in assets):
        raise ValueError(f"asset '{key}' already exists")
    entry = {**entry, "key": key}
    assets.append(entry)
    raw["assets"] = assets
    save_yaml("assets", raw)
    return universe(refresh=True).get(key)


def remove_asset(key: str) -> None:
    raw = load_yaml("assets", refresh=True)
    key = key.upper()
    raw["assets"] = [a for a in (raw.get("assets") or []) if str(a.get("key", "")).upper() != key]
    save_yaml("assets", raw)
    universe(refresh=True)


# Bars per year, used to annualise metrics per timeframe.
BARS_PER_YEAR: dict[str, float] = {
    "M1": 372_000,
    "M5": 74_400,
    "M15": 24_800,
    "M30": 12_400,
    "H1": 6_200,
    "H4": 1_550,
    "D1": 252,
    "W1": 52,
}

TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


def bars_per_year(timeframe: str, *, session: str = "24h") -> float:
    base = BARS_PER_YEAR.get(timeframe.upper(), 252.0)
    if session == "24x7" and timeframe.upper() not in {"D1", "W1"}:
        base *= 7 / 5
    elif session == "24x7":
        base = 365.0 if timeframe.upper() == "D1" else base
    return base
