"""Message bus — the nervous system of the factory.

Two interchangeable backends:

* ``memory``  : zero dependencies, single process. Perfect for dev and tests.
* ``redis``   : Redis Streams with consumer groups. Survives restarts, lets you
                split agents across processes/machines on the VPS.

Both expose the same three primitives:
    publish(msg)                  fire and forget
    subscribe(group, topics)      durable consumer loop
    request(msg, timeout)         ask another agent and await its answer
plus ``tap()``, a firehose every message passes through, which is what feeds the
live WebSocket view in the control room.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Any

from .config import settings
from .messages import Message, MsgKind

log = logging.getLogger("krish.bus")

STREAM_KEY = "krish:bus"
_MAX_STREAM_LEN = 50_000


class BusTimeout(TimeoutError):
    """A request got no response in time."""


class Bus(ABC):
    """Shared plumbing: firehose taps and request/response correlation."""

    def __init__(self) -> None:
        self._taps: set[asyncio.Queue[Message]] = set()
        self._waiters: dict[str, asyncio.Future[Message]] = {}
        self._closed = False

    # ---------------------------- lifecycle ---------------------------- #

    async def start(self) -> None:  # pragma: no cover - backend specific
        return None

    async def close(self) -> None:
        self._closed = True
        for fut in self._waiters.values():
            if not fut.done():
                fut.cancel()
        self._waiters.clear()

    # ---------------------------- publishing --------------------------- #

    async def publish(self, msg: Message) -> None:
        await self._deliver(msg)
        self._fanout_local(msg)

    @abstractmethod
    async def _deliver(self, msg: Message) -> None:
        """Backend-specific write."""

    def _fanout_local(self, msg: Message) -> None:
        """Feed taps and resolve pending requests inside this process."""
        for q in list(self._taps):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(msg)
        if msg.kind is MsgKind.RESPONSE and msg.reply_to:
            fut = self._waiters.pop(msg.reply_to, None)
            if fut is not None and not fut.done():
                fut.set_result(msg)

    # ---------------------------- consuming ---------------------------- #

    @abstractmethod
    def subscribe(
        self, group: str, topics: Iterable[str], *, consumer: str = "main"
    ) -> AsyncIterator[Message]:
        """Yield messages on ``topics``. One durable cursor per ``group``."""

    async def tap(self, maxsize: int = 2000) -> AsyncIterator[Message]:
        """Every message, in order, for observability. Drops on backpressure."""
        q: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)
        self._taps.add(q)
        try:
            while not self._closed:
                yield await q.get()
        finally:
            self._taps.discard(q)

    # ------------------------ request / response ----------------------- #

    async def request(self, msg: Message, *, timeout: float = 30.0) -> Message:
        """Send a REQUEST and wait for the matching RESPONSE.

        This is how "jisko jo chahiye" works: any agent can ask any other agent
        for help without importing it.
        """
        if msg.kind is not MsgKind.REQUEST:
            msg = msg.model_copy(update={"kind": MsgKind.REQUEST})
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Message] = loop.create_future()
        self._waiters[msg.id] = fut
        try:
            await self.publish(msg)
            return await asyncio.wait_for(fut, timeout)
        except TimeoutError as exc:
            raise BusTimeout(f"no response to {msg.topic} ({msg.id}) in {timeout}s") from exc
        finally:
            self._waiters.pop(msg.id, None)


# --------------------------------------------------------------------------- #
# In-memory backend
# --------------------------------------------------------------------------- #


class MemoryBus(Bus):
    def __init__(self) -> None:
        super().__init__()
        self._queues: dict[str, asyncio.Queue[Message]] = {}
        self._refs: dict[str, int] = {}

    async def _deliver(self, msg: Message) -> None:
        for q in list(self._queues.values()):
            await q.put(msg)

    async def subscribe(  # type: ignore[override]
        self, group: str, topics: Iterable[str], *, consumer: str = "main"
    ) -> AsyncIterator[Message]:
        """Competing consumers: every member of ``group`` shares one queue.

        This is what makes replicas useful. Four testers in the group "tester"
        take one strategy each; they do not each run the same backtest. The queue
        is keyed by group only, and refcounted so one replica stopping does not
        pull the queue out from under its siblings.
        """
        wanted = {str(t) for t in topics}
        q: asyncio.Queue[Message] = self._queues.setdefault(group, asyncio.Queue())
        self._refs[group] = self._refs.get(group, 0) + 1
        try:
            while not self._closed:
                msg = await q.get()
                if not wanted or msg.topic in wanted:
                    yield msg
        finally:
            self._refs[group] = max(0, self._refs.get(group, 1) - 1)
            if not self._refs[group]:
                self._queues.pop(group, None)
                self._refs.pop(group, None)


# --------------------------------------------------------------------------- #
# Redis Streams backend
# --------------------------------------------------------------------------- #


class RedisBus(Bus):
    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url
        self._redis: Any = None
        self._tap_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        import redis.asyncio as aioredis  # imported lazily so dev needs no redis

        self._redis = aioredis.from_url(self._url, decode_responses=True)
        await self._redis.ping()
        # Mirror the stream into local taps/waiters so cross-process responses
        # and the dashboard firehose both work.
        self._tap_task = asyncio.create_task(self._mirror(), name="bus-mirror")
        log.info("redis bus connected: %s", self._url)

    async def close(self) -> None:
        await super().close()
        if self._tap_task:
            self._tap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tap_task
        if self._redis:
            await self._redis.aclose()

    async def _deliver(self, msg: Message) -> None:
        assert self._redis is not None, "call start() first"
        await self._redis.xadd(
            STREAM_KEY,
            {"data": msg.model_dump_json()},
            maxlen=_MAX_STREAM_LEN,
            approximate=True,
        )

    def _fanout_local(self, msg: Message) -> None:
        # With redis, fan-out happens in _mirror() to avoid duplicates.
        return None

    async def _mirror(self) -> None:
        last = "$"
        while not self._closed:
            try:
                batches = await self._redis.xread({STREAM_KEY: last}, count=200, block=2000)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - transient redis errors
                log.exception("bus mirror read failed; retrying")
                await asyncio.sleep(1.0)
                continue
            for _stream, entries in batches or []:
                for entry_id, fields in entries:
                    last = entry_id
                    msg = _decode(fields)
                    if msg is not None:
                        super()._fanout_local(msg)

    async def subscribe(  # type: ignore[override]
        self, group: str, topics: Iterable[str], *, consumer: str = "main"
    ) -> AsyncIterator[Message]:
        assert self._redis is not None, "call start() first"
        wanted = {str(t) for t in topics}
        with contextlib.suppress(Exception):  # BUSYGROUP if it already exists
            await self._redis.xgroup_create(STREAM_KEY, group, id="$", mkstream=True)
        while not self._closed:
            try:
                batches = await self._redis.xreadgroup(
                    group, consumer, {STREAM_KEY: ">"}, count=50, block=2000
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover
                log.exception("xreadgroup failed for %s; retrying", group)
                await asyncio.sleep(1.0)
                continue
            for _stream, entries in batches or []:
                ids = [entry_id for entry_id, _ in entries]
                for _entry_id, fields in entries:
                    msg = _decode(fields)
                    if msg is not None and (not wanted or msg.topic in wanted):
                        yield msg
                if ids:
                    # Ack everything read: our work is checkpointed in the
                    # blackboard, so redelivery is not how we get reliability.
                    await self._redis.xack(STREAM_KEY, group, *ids)


def _decode(fields: dict[str, Any]) -> Message | None:
    raw = fields.get("data")
    if not raw:
        return None
    try:
        return Message.model_validate(json.loads(raw))
    except Exception:  # pragma: no cover - poison message
        log.warning("dropping undecodable bus message")
        return None


# --------------------------------------------------------------------------- #

_bus: Bus | None = None


def make_bus(backend: str | None = None) -> Bus:
    backend = (backend or settings().bus_backend).lower()
    if backend == "redis":
        return RedisBus(settings().redis_url)
    if backend == "memory":
        return MemoryBus()
    raise ValueError(f"unknown bus backend '{backend}' (use 'memory' or 'redis')")


def bus() -> Bus:
    """Process-wide bus singleton."""
    global _bus
    if _bus is None:
        _bus = make_bus()
    return _bus


def set_bus(instance: Bus | None) -> None:
    global _bus
    _bus = instance
