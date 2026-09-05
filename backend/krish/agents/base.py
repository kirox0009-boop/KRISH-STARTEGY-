"""BaseAgent — everything an agent gets for free.

Subclass, declare ``name`` / ``role`` / ``subscribes``, implement ``handle()``.
You get: durable subscription, heartbeat, live status for the dashboard, ask()
for cross-agent help, structured logging to the blackboard, pause/resume/kill
control from the UI, and crash isolation (one bad message never kills the agent).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, ClassVar

from ..bus import Bus, BusTimeout
from ..bus import bus as default_bus
from ..messages import Message, MsgKind, Topic
from ..registry import AgentStatus, registry
from ..store import log_event

log = logging.getLogger("krish.agent")

HEARTBEAT_SECONDS = 5.0


class BaseAgent:
    #: unique, lowercase, stable — used as consumer-group name
    name: ClassVar[str] = "agent"
    #: human label for the dashboard
    role: ClassVar[str] = "Agent"
    description: ClassVar[str] = ""
    squad: ClassVar[str] = "system"
    #: topics this agent consumes
    subscribes: ClassVar[tuple[str, ...]] = ()
    #: hard cap on one message's processing time; prevents a stuck agent
    handler_timeout: ClassVar[float] = 900.0

    def __init__(self, bus: Bus | None = None, *, instance: str | None = None) -> None:
        self.bus = bus or default_bus()
        # Instance name shadows the class name, so the same agent class can be run
        # N times (e.g. four testers on a 4-core VPS) with independent cursors.
        if instance:
            self.name = f"{type(self).name}-{instance}"
        self.state = registry().register(
            self.name,
            self.role,
            description=self.description,
            squad=self.squad,
            subscriptions=[str(t) for t in self.subscribes],
        )
        self._stop = asyncio.Event()
        self._paused = asyncio.Event()
        self._paused.set()  # set == not paused
        self._tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    async def setup(self) -> None:
        """Optional one-time init (open connections, warm caches)."""

    async def teardown(self) -> None:
        """Optional cleanup."""

    async def run(self) -> None:
        await self.setup()
        self._tasks.append(asyncio.create_task(self._heartbeat(), name=f"{self.name}-hb"))
        self._tasks.append(asyncio.create_task(self._control_loop(), name=f"{self.name}-ctl"))
        self._set_status(AgentStatus.IDLE)
        self.log("started", level="info")
        try:
            topics = [str(t) for t in self.subscribes]
            async for msg in self.bus.subscribe(self.name, topics):
                if self._stop.is_set():
                    break
                await self._paused.wait()
                await self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        finally:
            for task in self._tasks:
                task.cancel()
            for task in self._tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self.teardown()
            self._set_status(AgentStatus.STOPPED)
            self.log("stopped", level="info")

    async def stop(self) -> None:
        self._stop.set()
        self._paused.set()

    async def _heartbeat(self) -> None:
        while not self._stop.is_set():
            self.state.last_heartbeat = time.time()
            await asyncio.sleep(HEARTBEAT_SECONDS)

    async def _control_loop(self) -> None:
        """React to pause/resume/kill/reload sent from the control room."""
        async for msg in self.bus.subscribe(f"{self.name}-ctl", [Topic.AGENT_CONTROL]):
            target = msg.payload.get("agent")
            if target not in (None, "*", self.name):
                continue
            action = str(msg.payload.get("action", "")).lower()
            if action == "pause":
                self._paused.clear()
                self._set_status(AgentStatus.PAUSED)
                self.log("paused by operator", level="warn")
            elif action == "resume":
                self._paused.set()
                self._set_status(AgentStatus.IDLE)
                self.log("resumed by operator", level="info")
            elif action in {"stop", "kill"}:
                self.log("stop requested by operator", level="warn")
                await self.stop()
                return
            elif action == "reload":
                await self.on_reload()

    async def on_reload(self) -> None:
        """Called when config changed. Override to re-read settings."""

    # ------------------------------------------------------------------ #
    # message handling
    # ------------------------------------------------------------------ #

    async def _dispatch(self, msg: Message) -> None:
        started = time.time()
        self.state.task = self._describe(msg)
        self.state.project_id = msg.project_id
        self.state.strategy_id = msg.strategy_id
        self.state.task_started_at = started
        self._set_status(AgentStatus.WORKING)
        try:
            await asyncio.wait_for(self.handle(msg), timeout=self.handler_timeout)
            self.state.handled += 1
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self.state.errors += 1
            self.state.last_error = f"timeout after {self.handler_timeout}s on {msg.topic}"
            self.log(self.state.last_error, level="error", topic=msg.topic, msg=msg)
        except Exception as exc:  # one poison message must not kill the agent
            self.state.errors += 1
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("[%s] failed handling %s", self.name, msg.topic)
            self.log(
                f"error handling {msg.topic}: {self.state.last_error}",
                level="error",
                topic=msg.topic,
                msg=msg,
            )
        finally:
            self.state.task = ""
            self.state.task_started_at = None
            self.state.eta_seconds = None
            self.state.project_id = None
            self.state.strategy_id = None
            if self.state.status is AgentStatus.WORKING:
                self._set_status(AgentStatus.IDLE)

    def _describe(self, msg: Message) -> str:
        """Short human label shown on the dashboard card."""
        bits = [msg.topic]
        for key in ("asset", "name", "strategy_name"):
            if msg.payload.get(key):
                bits.append(str(msg.payload[key]))
                break
        return " · ".join(bits)

    async def handle(self, msg: Message) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # helpers for subclasses
    # ------------------------------------------------------------------ #

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        parent: Message | None = None,
        kind: MsgKind = MsgKind.EVENT,
        project_id: str | None = None,
        strategy_id: str | None = None,
    ) -> Message:
        if parent is not None:
            msg = parent.child(
                topic=topic,
                sender=self.name,
                kind=kind,
                payload=payload,
                project_id=project_id or parent.project_id,
                strategy_id=strategy_id or parent.strategy_id,
            )
        else:
            msg = Message(
                topic=topic,
                sender=self.name,
                kind=kind,
                payload=payload or {},
                project_id=project_id,
                strategy_id=strategy_id,
                trace=[self.name],
            )
        await self.bus.publish(msg)
        return msg

    async def ask(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        reply_topic: str,
        timeout: float = 60.0,
        parent: Message | None = None,
    ) -> dict[str, Any]:
        """Ask another agent for help and wait for the answer.

        Raises :class:`BusTimeout` if nobody answers — callers decide whether
        that is fatal or just a degraded path.
        """
        req = Message(
            topic=topic,
            sender=self.name,
            kind=MsgKind.REQUEST,
            payload=payload,
            reply_topic=reply_topic,
            project_id=parent.project_id if parent else None,
            strategy_id=parent.strategy_id if parent else None,
            trace=[*(parent.trace if parent else []), self.name],
        )
        previous = self.state.status
        self._set_status(AgentStatus.WAITING)
        try:
            reply = await self.bus.request(req, timeout=timeout)
            if reply.payload.get("error"):
                raise RuntimeError(f"{topic} failed: {reply.payload['error']}")
            return reply.payload
        except BusTimeout:
            self.log(f"no answer for {topic} in {timeout}s", level="warn")
            raise
        finally:
            self._set_status(previous)

    def progress(self, task: str, *, eta_seconds: float | None = None) -> None:
        """Update the label + ETA the dashboard shows for this agent."""
        self.state.task = task
        self.state.eta_seconds = eta_seconds
        self.state.last_heartbeat = time.time()

    def log(
        self,
        message: str,
        *,
        level: str = "info",
        topic: str = "",
        msg: Message | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        log_event(
            agent=self.name,
            topic=topic or (msg.topic if msg else "agent.lifecycle"),
            kind=str(msg.kind) if msg else "log",
            level=level,
            message=message,
            project_id=msg.project_id if msg else None,
            strategy_id=msg.strategy_id if msg else None,
            payload=payload or {},
        )
        getattr(log, "warning" if level == "warn" else level, log.info)(
            "[%s] %s", self.name, message
        )

    def _set_status(self, status: AgentStatus) -> None:
        if self.state.status is AgentStatus.PAUSED and status is not AgentStatus.IDLE:
            return
        self.state.status = status
        self.state.last_heartbeat = time.time()
