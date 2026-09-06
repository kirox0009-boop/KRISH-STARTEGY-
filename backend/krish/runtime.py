"""Factory runtime: build the squad, supervise it, keep it alive 24/7.

Every agent runs as a supervised asyncio task. If one crashes it is restarted
with exponential backoff while the rest keep working, because a factory that
stops when one worker trips is not a factory.

Scaling on a VPS: CPU-bound agents (tester, tuner, robustness) can be started
multiple times via ``instances``; each replica gets its own bus cursor, so work
is shared rather than duplicated.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any

from .agents.base import BaseAgent
from .agents.build import DeveloperAgent
from .agents.champion import ChampionAgent
from .agents.data import MarketDataAgent
from .agents.deliver import DeliveryAgent, DocWriterAgent, PackagerAgent
from .agents.research import ArchitectAgent, QuantAnalystAgent, ResearcherAgent
from .agents.system import LibrarianAgent, MemoryAgent, MonitorAgent, OrchestratorAgent
from .agents.validate import (
    JudgeAgent,
    RiskAgent,
    RobustnessAgent,
    TesterAgent,
    TunerAgent,
)
from .bus import Bus
from .bus import bus as default_bus
from .config import LOG_DIR, factory_section, settings
from .storage import minimise_local_disk
from .store import init_db

log = logging.getLogger("krish.runtime")

#: The Phase 0 squad, in pipeline order. Agents from later phases
#: (news_sentiment, regime, portfolio, mt5_deploy, live_monitor) plug in here
#: without touching anyone else - that is the point of the bus.
AGENT_CLASSES: tuple[type[BaseAgent], ...] = (
    OrchestratorAgent,
    MarketDataAgent,
    ResearcherAgent,
    QuantAnalystAgent,
    ArchitectAgent,
    ChampionAgent,
    DeveloperAgent,
    TesterAgent,
    TunerAgent,
    RobustnessAgent,
    RiskAgent,
    JudgeAgent,
    DocWriterAgent,
    PackagerAgent,
    DeliveryAgent,
    MemoryAgent,
    MonitorAgent,
    LibrarianAgent,
)

#: Agents worth replicating when the box has cores to spare.
REPLICABLE = {"tester", "tuner", "robustness"}

MAX_BACKOFF = 60.0


def configure_logging() -> None:
    level = getattr(logging, settings().log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    with contextlib.suppress(OSError):
        # Rotating, not plain: this process is meant to run for months, and an
        # unbounded log file is the classic way to fill a VPS disk.
        # 20 MB x 5 normally; a tenth of that when the operator has asked for the
        # smallest possible disk footprint. stdout still carries everything, and
        # the service wrapper captures that.
        small = minimise_local_disk()
        handlers.append(
            RotatingFileHandler(
                LOG_DIR / "krish.log",
                maxBytes=(2 if small else 20) * 1024 * 1024,
                backupCount=1 if small else 5,
                encoding="utf-8",
            )
        )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        handlers=handlers,
        force=True,
    )
    # yfinance and friends are chatty
    for noisy in ("yfinance", "peewee", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Factory:
    def __init__(self, bus: Bus | None = None, *, replicas: int | None = None) -> None:
        self.bus = bus or default_bus()
        self.agents: list[BaseAgent] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()
        if replicas is None:
            budget = int(factory_section("cycle").get("max_parallel_backtests", 2))
            replicas = max(1, min(budget, (os.cpu_count() or 2)))
        self.replicas = replicas

    # ------------------------------------------------------------------ #

    def build(self) -> list[BaseAgent]:
        agents: list[BaseAgent] = []
        for cls in AGENT_CLASSES:
            copies = self.replicas if cls.name in REPLICABLE else 1
            for index in range(copies):
                instance = str(index + 1) if copies > 1 else None
                agents.append(cls(self.bus, instance=instance))
        self.agents = agents
        return agents

    async def start(self) -> None:
        init_db()
        await self.bus.start()
        if not self.agents:
            self.build()
        for agent in self.agents:
            self._tasks.append(
                asyncio.create_task(self._supervise(agent), name=f"supervise-{agent.name}")
            )
        log.info(
            "factory started: %d agents (%d replicas for %s), bus=%s",
            len(self.agents),
            self.replicas,
            ",".join(sorted(REPLICABLE)),
            settings().bus_backend,
        )

    async def _supervise(self, agent: BaseAgent) -> None:
        backoff = 1.0
        while not self._stopping.is_set():
            try:
                await agent.run()
                if self._stopping.is_set() or agent._stop.is_set():
                    return
                log.warning("agent %s exited cleanly; restarting", agent.name)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("agent %s crashed; restarting in %.0fs", agent.name, backoff)
                agent.state.errors += 1
            if self._stopping.is_set():
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    async def stop(self) -> None:
        self._stopping.set()
        for agent in self.agents:
            with contextlib.suppress(Exception):
                await agent.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self.bus.close()
        log.info("factory stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            await self._stopping.wait()
        finally:
            await self.stop()

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": a.name,
                "role": a.role,
                "squad": a.squad,
                "description": a.description,
                "subscribes": [str(t) for t in a.subscribes],
            }
            for a in (self.agents or self.build())
        ]


__all__ = ["AGENT_CLASSES", "Factory", "configure_logging"]
