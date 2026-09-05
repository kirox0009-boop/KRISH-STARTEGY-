"""Live agent roster — who is alive, what they hold, and for how long.

This is exactly what the control room's agent cards render: role, status, current
project, elapsed time, ETA, throughput, last error.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class AgentStatus(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"  # blocked on another agent's answer
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentState:
    name: str
    role: str
    description: str = ""
    squad: str = "system"
    status: AgentStatus = AgentStatus.STARTING
    task: str = ""
    project_id: str | None = None
    strategy_id: str | None = None
    task_started_at: float | None = None
    eta_seconds: float | None = None
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    handled: int = 0
    errors: int = 0
    last_error: str = ""
    subscriptions: list[str] = field(default_factory=list)

    @property
    def elapsed_on_task(self) -> float | None:
        if self.task_started_at is None:
            return None
        return round(time.time() - self.task_started_at, 2)

    @property
    def uptime(self) -> float:
        return round(time.time() - self.started_at, 2)

    @property
    def alive(self) -> bool:
        return (
            self.status not in {AgentStatus.STOPPED, AgentStatus.ERROR}
            and (time.time() - self.last_heartbeat) < 30
        )

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = str(self.status)
        data["elapsed_on_task"] = self.elapsed_on_task
        data["uptime"] = self.uptime
        data["alive"] = self.alive
        data["last_heartbeat_age"] = round(time.time() - self.last_heartbeat, 2)
        return data


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentState] = {}

    def register(
        self,
        name: str,
        role: str,
        *,
        description: str = "",
        squad: str = "system",
        subscriptions: list[str] | None = None,
    ) -> AgentState:
        state = AgentState(
            name=name,
            role=role,
            description=description,
            squad=squad,
            subscriptions=subscriptions or [],
        )
        self._agents[name] = state
        return state

    def get(self, name: str) -> AgentState | None:
        return self._agents.get(name)

    def all(self) -> list[AgentState]:
        return list(self._agents.values())

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            a.snapshot() for a in sorted(self._agents.values(), key=lambda a: (a.squad, a.name))
        ]

    def summary(self) -> dict[str, Any]:
        states = self.all()
        by_status: dict[str, int] = {}
        for s in states:
            by_status[str(s.status)] = by_status.get(str(s.status), 0) + 1
        return {
            "total": len(states),
            "alive": sum(1 for s in states if s.alive),
            "working": sum(1 for s in states if s.status is AgentStatus.WORKING),
            "handled": sum(s.handled for s in states),
            "errors": sum(s.errors for s in states),
            "by_status": by_status,
        }


_registry: AgentRegistry | None = None


def registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
