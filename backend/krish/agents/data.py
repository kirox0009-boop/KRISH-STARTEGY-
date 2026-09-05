"""Data squad: the agents that own market context.

``market_data`` is the factory's help desk for prices — nobody else touches the
provider layer. Everything is request/response over the bus, so testers can be
moved to another machine and still get their bars.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from ..assets import universe
from ..data.providers import DataError, cache_status, fetch_ohlcv
from ..messages import Message, Topic
from .base import BaseAgent

# Process-local frame cache so N testers hitting the same asset/timeframe within
# one cycle do not each pay the parse cost.
_FRAMES: dict[tuple[str, str], pd.DataFrame] = {}


def frame_cache_get(asset: str, timeframe: str) -> pd.DataFrame | None:
    return _FRAMES.get((asset.upper(), timeframe.upper()))


class MarketDataAgent(BaseAgent):
    name = "market_data"
    role = "Market Data"
    squad = "data"
    description = "Fetches, cleans, caches and serves price history for every asset."
    subscribes = (Topic.DATA_REQUEST, Topic.CYCLE_START)
    handler_timeout = 600.0

    async def setup(self) -> None:
        self._warmed: set[tuple[str, str]] = set()

    async def handle(self, msg: Message) -> None:
        if msg.topic == Topic.DATA_REQUEST:
            await self._serve(msg)
        elif msg.topic == Topic.CYCLE_START:
            await self._warm(msg)

    # ------------------------------------------------------------------ #

    async def _serve(self, msg: Message) -> None:
        asset = str(msg.payload.get("asset", "")).upper()
        timeframe = str(msg.payload.get("timeframe", "H1")).upper()
        refresh = bool(msg.payload.get("refresh", False))
        self.progress(f"serving {asset} {timeframe}")
        try:
            frame = await self._get(asset, timeframe, refresh=refresh)
        except Exception as exc:
            self.log(f"data request failed for {asset} {timeframe}: {exc}", level="error", msg=msg)
            await self.bus.publish(msg.responds_to(self.name, {"error": str(exc)}))
            return

        # The frame itself stays in this process; the response carries a handle
        # plus enough metadata for the caller to decide what to do.
        await self.bus.publish(
            msg.responds_to(
                self.name,
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "bars": len(frame),
                    "start": str(frame.index[0]),
                    "end": str(frame.index[-1]),
                    "synthetic": bool(frame.attrs.get("synthetic", False)),
                    "handle": f"{asset}:{timeframe}",
                },
            )
        )

    async def _warm(self, msg: Message) -> None:
        """Pre-load whatever this cycle is about, so testers never wait on I/O."""
        asset = str(msg.payload.get("asset", "")).upper()
        timeframe = str(msg.payload.get("timeframe", "H1")).upper()
        if not asset:
            return
        key = (asset, timeframe)
        if key in self._warmed:
            return
        try:
            frame = await self._get(asset, timeframe)
            self._warmed.add(key)
            await self.emit(
                Topic.DATA_UPDATED,
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "bars": len(frame),
                    "start": str(frame.index[0]),
                    "end": str(frame.index[-1]),
                },
                parent=msg,
            )
        except DataError as exc:
            self.log(f"cannot warm {asset} {timeframe}: {exc}", level="warn", msg=msg)

    async def _get(self, asset: str, timeframe: str, *, refresh: bool = False) -> pd.DataFrame:
        key = (asset, timeframe)
        if not refresh and key in _FRAMES:
            return _FRAMES[key]
        universe().get(asset)  # raises for unknown assets before any network work
        frame = await asyncio.to_thread(fetch_ohlcv, asset, timeframe, refresh=refresh)
        _FRAMES[key] = frame
        return frame

    async def on_reload(self) -> None:
        universe(refresh=True)
        self._warmed.clear()
        self.log("asset universe reloaded")


def data_health() -> list[dict[str, Any]]:
    """Used by the API's /health/data endpoint and the dashboard data panel."""
    return cache_status()


__all__ = ["MarketDataAgent", "data_health", "frame_cache_get"]
