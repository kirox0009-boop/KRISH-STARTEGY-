"""System squad: orchestrator, memory, monitor.

* **orchestrator** keeps the factory moving: it decides what to work on next and
  never lets the pipeline go idle.
* **memory** is the learning core: it turns the experiment ledger into priors
  that steer the next generation.
* **monitor** watches the machine, not the markets.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import platform
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..assets import universe
from ..config import ARTIFACT_DIR, CACHE_DIR, DATA_DIR, LOG_DIR, PACKAGE_DIR, factory_section
from ..genome import RECIPES
from ..messages import Message, Topic, new_id
from ..registry import registry
from ..store import (
    counts,
    database_bytes,
    delete_old_rejected,
    in_db,
    judged_strategies,
    priors_for,
    prune_events,
    strip_rejected_details,
    upsert_prior,
    vacuum,
)
from .base import BaseAgent


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    role = "Orchestrator"
    squad = "system"
    description = "Plans cycles, spreads work across assets and timeframes, keeps the loop alive."
    subscribes = (Topic.CONFIG_RELOAD, Topic.AGENT_CONTROL)

    async def setup(self) -> None:
        self._cycles = 0
        self._pairs = self._build_rotation()
        self._rotation = itertools.cycle(self._pairs) if self._pairs else None
        # KRISH_SCHEDULER=off runs the factory on demand only (used by `krish cycle`
        # and by anyone who wants to drive it purely from the control room).
        if os.getenv("KRISH_SCHEDULER", "on").lower() != "off":
            self._tasks.append(asyncio.create_task(self._ticker(), name="orchestrator-ticker"))
        else:
            self.log("scheduler disabled; cycles run on request only")

    @staticmethod
    def _build_rotation() -> list[tuple[str, str]]:
        """Round-robin over every configured asset x timeframe."""
        pairs: list[tuple[str, str]] = []
        for asset in universe().all():
            for timeframe in asset.timeframes:
                pairs.append((asset.key, timeframe))
        return pairs

    async def handle(self, msg: Message) -> None:
        if msg.topic == Topic.CONFIG_RELOAD:
            self._pairs = self._build_rotation()
            self._rotation = itertools.cycle(self._pairs) if self._pairs else None
            self.log(f"rotation rebuilt: {len(self._pairs)} asset/timeframe pairs")
            return
        # Manual "run a cycle now" from the control room.
        if str(msg.payload.get("action", "")).lower() == "cycle":
            await self._start_cycle(
                asset=msg.payload.get("asset"),
                timeframe=msg.payload.get("timeframe"),
                count=int(msg.payload.get("count", 0)) or None,
                trigger="manual",
            )

    async def _ticker(self) -> None:
        cfg = factory_section("cycle")
        interval = float(cfg.get("interval_seconds", 60))
        # Small initial delay so every agent has finished subscribing first.
        await asyncio.sleep(3.0)
        while not self._stop.is_set():
            await self._paused.wait()
            try:
                await self._start_cycle(trigger="schedule")
            except Exception:
                self.state.errors += 1
                self.log("cycle start failed", level="error")
            interval = float(factory_section("cycle").get("interval_seconds", interval))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval)

    async def _start_cycle(
        self,
        *,
        asset: str | None = None,
        timeframe: str | None = None,
        count: int | None = None,
        trigger: str = "schedule",
    ) -> None:
        if self._rotation is None:
            self.log("no assets configured; nothing to do", level="warn")
            return
        pair = (asset.upper(), timeframe.upper()) if asset and timeframe else next(self._rotation)

        cfg = factory_section("cycle")
        count = count or int(cfg.get("strategies_per_cycle", 4))
        self._cycles += 1
        cycle_id = new_id("cyc")

        self.progress(f"cycle {self._cycles}: {pair[0]} {pair[1]} x{count}")
        self.log(f"cycle {self._cycles} ({trigger}): {pair[0]} {pair[1]}, {count} ideas")
        await self.emit(
            Topic.CYCLE_START,
            {
                "cycle_id": cycle_id,
                "cycle": self._cycles,
                "asset": pair[0],
                "timeframe": pair[1],
                "count": count,
                "trigger": trigger,
            },
        )


class MemoryAgent(BaseAgent):
    name = "memory"
    role = "Memory / Learning"
    squad = "system"
    description = "Turns every result, pass or fail, into priors that steer the next generation."
    subscribes = (Topic.STRATEGY_JUDGED, Topic.STRATEGY_INVALID, Topic.STRATEGY_TEST_FAILED)

    #: recompute priors at most this often, per scope
    REFRESH_SECONDS = 30.0

    async def setup(self) -> None:
        self._last_refresh: dict[str, float] = {}

    async def handle(self, msg: Message) -> None:
        asset = str(msg.payload.get("asset") or "").upper()
        timeframe = str(msg.payload.get("timeframe") or "").upper()
        if not asset:
            return
        scope = f"{asset}:{timeframe}" if timeframe else asset

        last = self._last_refresh.get(scope, 0.0)
        if time.time() - last < self.REFRESH_SECONDS:
            return
        self._last_refresh[scope] = time.time()

        self.progress(f"learning from results on {scope}")
        ledger = await in_db(judged_strategies, asset, 500)
        if len(ledger) < 5:
            return

        priors = self._derive(ledger, timeframe)
        for key, value in priors.items():
            await in_db(
                upsert_prior,
                scope,
                key,
                value if isinstance(value, dict) else {"value": value},
                samples=len(ledger),
                confidence=min(1.0, len(ledger) / 100.0),
            )

        merged = await in_db(priors_for, scope)
        self.log(
            f"priors for {scope} updated from {len(ledger)} judged strategies "
            f"(best style: {priors.get('best_style', {}).get('style', 'n/a')})",
            msg=msg,
        )
        await self.emit(
            Topic.MEMORY_PRIORS,
            {
                "scope": scope,
                "asset": asset,
                "timeframe": timeframe,
                "priors": self._flatten(merged),
                "samples": len(ledger),
            },
            parent=msg,
        )

    # ------------------------------------------------------------------ #

    def _derive(self, ledger: list[dict[str, Any]], timeframe: str) -> dict[str, Any]:
        rows = [r for r in ledger if not timeframe or r.get("timeframe") == timeframe] or ledger

        by_recipe: dict[str, list[float]] = defaultdict(list)
        by_style: dict[str, list[float]] = defaultdict(list)
        by_origin: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            score = float(row.get("score") or 0.0)
            recipe = self._recipe_of(row)
            if recipe:
                by_recipe[recipe].append(score)
            by_style[str(row.get("style") or "unknown")].append(score)
            by_origin[str(row.get("origin") or "fresh")].append(score)

        weights = self._weights(by_recipe)
        best_style = max(
            ((s, sum(v) / len(v)) for s, v in by_style.items() if v),
            key=lambda kv: kv[1],
            default=("unknown", 0.0),
        )
        pass_rate = sum(1 for r in rows if r.get("verdict") == "PASS") / max(len(rows), 1)

        return {
            "recipe_weights": weights,
            "best_style": {"style": best_style[0], "mean_score": round(best_style[1], 4)},
            "origin_performance": {
                origin: round(sum(v) / len(v), 4) for origin, v in by_origin.items() if v
            },
            "pass_rate": {"value": round(pass_rate, 4), "n": len(rows)},
        }

    @staticmethod
    def _recipe_of(row: dict[str, Any]) -> str | None:
        notes = str(row.get("recipe") or "")
        if "recipe=" in notes:
            return notes.split("recipe=")[1].split()[0].strip(" |,")
        return None

    @staticmethod
    def _weights(by_recipe: dict[str, list[float]]) -> dict[str, float]:
        """Weight = mean score relative to the field, floored so nothing dies out.

        Exploration never stops: a recipe that has failed 20 times still gets a
        small share of the dice, because market regimes change.
        """
        if not by_recipe:
            return {}
        means = {r: (sum(v) / len(v)) for r, v in by_recipe.items() if v}
        overall = (sum(means.values()) / len(means)) or 1e-9
        weights = {}
        for recipe in RECIPES:
            mean = means.get(recipe)
            if mean is None:
                weights[recipe] = 1.2  # untried recipes get a curiosity bonus
            else:
                weights[recipe] = round(max(0.15, min(3.0, mean / overall)), 3)
        return weights

    @staticmethod
    def _flatten(priors: dict[str, Any]) -> dict[str, Any]:
        """Unwrap single-value priors so consumers can read them directly."""
        out: dict[str, Any] = {}
        for key, value in priors.items():
            if isinstance(value, dict) and set(value) == {"value"}:
                out[key] = value["value"]
            else:
                out[key] = value
        return out


class MonitorAgent(BaseAgent):
    name = "monitor"
    role = "System Monitor"
    squad = "system"
    description = "Watches VPS resources, agent health and error rates; alerts before things break."
    subscribes = (Topic.DELIVERY_FAILED,)

    CHECK_SECONDS = 30.0

    async def setup(self) -> None:
        self._alerted: set[str] = set()
        self._tasks.append(asyncio.create_task(self._loop(), name="monitor-loop"))

    async def handle(self, msg: Message) -> None:
        self.log(f"delivery failed for '{msg.payload.get('name')}'", level="error", msg=msg)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                await self._check()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.CHECK_SECONDS)

    async def _check(self) -> None:
        health = self.snapshot()
        for agent in health["agents"]:
            if not agent["alive"] and agent["status"] not in {"stopped", "paused"}:
                key = f"dead:{agent['name']}"
                if key not in self._alerted:
                    self._alerted.add(key)
                    self.log(
                        f"agent '{agent['name']}' has not heartbeated in "
                        f"{agent['last_heartbeat_age']}s",
                        level="error",
                    )
            elif agent["alive"]:
                self._alerted.discard(f"dead:{agent['name']}")

        if health["disk"]["free_pct"] < 10:
            self.log(f"disk almost full: {health['disk']['free_pct']}% free", level="error")
        self.progress(
            f"{health['registry']['alive']}/{health['registry']['total']} agents alive, "
            f"{health['db']['strategies']} strategies"
        )

    @staticmethod
    def snapshot() -> dict[str, Any]:
        """System health, also served by the API's /api/health endpoint.

        Cross-platform on purpose: the VPS may well be Windows, because that is
        where MetaTrader 5 runs.
        """
        # os.getloadavg does not exist on Windows at all (not just fail).
        if hasattr(os, "getloadavg"):
            try:
                load1, load5, load15 = os.getloadavg()
            except OSError:  # pragma: no cover
                load1 = load5 = load15 = 0.0
        else:
            load1 = load5 = load15 = 0.0

        usage = shutil.disk_usage(str(DATA_DIR))  # works on Windows and POSIX
        return {
            "registry": registry().summary(),
            "agents": registry().snapshot(),
            "db": counts(),
            "load": {"1m": round(load1, 2), "5m": round(load5, 2), "15m": round(load15, 2)},
            "load_available": hasattr(os, "getloadavg"),
            "cpus": os.cpu_count() or 1,
            "platform": platform.system(),
            "disk": {
                "free_pct": round(usage.free / max(usage.total, 1) * 100, 1),
                "free_gb": round(usage.free / 1e9, 2),
                "total_gb": round(usage.total / 1e9, 2),
            },
        }


def storage_report() -> dict[str, Any]:
    """Where the disk is going. Served by /api/health/storage and the dashboard."""

    def folder_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    db = database_bytes()
    cache = folder_bytes(CACHE_DIR)
    packages = folder_bytes(PACKAGE_DIR)
    logs = folder_bytes(LOG_DIR)
    artifacts = folder_bytes(ARTIFACT_DIR)
    usage = shutil.disk_usage(str(DATA_DIR))
    row_counts = counts()
    strategies = max(row_counts.get("strategies", 0), 1)

    return {
        "database_mb": round(db / 1e6, 2),
        "price_cache_mb": round(cache / 1e6, 2),
        "packages_mb": round(packages / 1e6, 2),
        "logs_mb": round(logs / 1e6, 2),
        "artifacts_mb": round(artifacts / 1e6, 2),
        "total_mb": round((db + cache + packages + logs + artifacts) / 1e6, 2),
        "kb_per_strategy": round(db / strategies / 1e3, 1),
        "disk_free_gb": round(usage.free / 1e9, 2),
        "disk_total_gb": round(usage.total / 1e9, 2),
        # The price cache is bounded: one file per asset+timeframe, overwritten
        # on refresh. The database is the part that grows with every experiment.
        "notes": {
            "price_cache": "bounded — one file per asset/timeframe, overwritten on refresh",
            "logs": "bounded — rotates at 20 MB, keeps 5 files (~120 MB ceiling)",
            "database": "grows with every experiment; the librarian agent prunes it",
        },
    }


class LibrarianAgent(BaseAgent):
    name = "librarian"
    role = "Librarian"
    squad = "system"
    description = "Keeps disk use flat: prunes old detail so the factory can run for years."
    subscribes = (Topic.CONFIG_RELOAD,)

    async def setup(self) -> None:
        self._tasks.append(asyncio.create_task(self._loop(), name="librarian-loop"))

    async def handle(self, msg: Message) -> None:
        self.log("retention settings reloaded")

    async def _loop(self) -> None:
        # First pass shortly after boot, so a box that has been off for a while
        # gets tidied before it starts producing again.
        await asyncio.sleep(60.0)
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                await self._sweep()
            minutes = float(factory_section("retention").get("prune_interval_minutes", 30))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=max(minutes, 1.0) * 60.0)

    async def _sweep(self) -> None:
        cfg = factory_section("retention")
        before = await in_db(storage_report)

        events = await in_db(prune_events, float(cfg.get("events_keep_days", 7)))
        stripped = await in_db(
            strip_rejected_details, float(cfg.get("rejected_detail_keep_hours", 12))
        )
        deleted = await in_db(delete_old_rejected, float(cfg.get("rejected_keep_days", 60)))

        did_work = events or stripped or deleted.get("strategies")
        if did_work and cfg.get("vacuum", True):
            await in_db(vacuum)

        after = await in_db(storage_report)
        freed = round(before["database_mb"] - after["database_mb"], 2)

        if did_work:
            self.log(
                f"pruned {events} events, stripped detail from {stripped} rejected runs, "
                f"deleted {deleted.get('strategies', 0)} old rejected strategies; "
                f"database {before['database_mb']}MB -> {after['database_mb']}MB "
                f"(freed {freed}MB)"
            )
        self.progress(
            f"disk: db {after['database_mb']}MB, cache {after['price_cache_mb']}MB, "
            f"{after['disk_free_gb']}GB free"
        )

        if after["disk_free_gb"] < 2.0:
            self.log(
                f"only {after['disk_free_gb']}GB of disk left — lower "
                f"retention.rejected_keep_days in config/factory.yaml",
                level="error",
            )


__all__ = [
    "LibrarianAgent",
    "MemoryAgent",
    "MonitorAgent",
    "OrchestratorAgent",
    "storage_report",
]
