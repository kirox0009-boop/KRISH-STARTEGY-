"""The only way agents talk to each other: a typed message envelope.

Keeping this small and explicit is what lets agents be swapped, duplicated or
moved to another machine without touching anyone else's code.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class MsgKind(StrEnum):
    EVENT = "event"  # broadcast: "this happened"
    REQUEST = "request"  # "mujhe ye chahiye" -> expects a RESPONSE
    RESPONSE = "response"  # answer to a REQUEST
    TASK = "task"  # orchestrator assigning work
    CONTROL = "control"  # pause / resume / kill / reload, usually from the UI


class Topic(StrEnum):
    """Every channel in the system. Adding a topic is a deliberate act."""

    # system / lifecycle
    CYCLE_START = "cycle.start"
    TASK_ASSIGNED = "task.assigned"
    AGENT_CONTROL = "agent.control"
    CONFIG_RELOAD = "config.reload"

    # data & context
    DATA_REQUEST = "data.request"
    DATA_RESPONSE = "data.response"
    DATA_UPDATED = "data.updated"
    MACRO_EVENT = "macro.event"
    MACRO_REGIME = "macro.regime_changed"
    NEWS_SIGNAL = "news.signal"
    REGIME_UPDATED = "regime.labels_updated"

    # research & build
    HYPOTHESIS_CREATED = "hypothesis.created"
    FEATURE_SPEC_CREATED = "feature_spec.created"
    STRATEGY_CREATED = "strategy.created"
    STRATEGY_BUILT = "strategy.built"
    STRATEGY_INVALID = "strategy.invalid"
    EVOLUTION_REQUEST = "evolution.request"

    # validation
    BACKTEST_REQUEST = "backtest.request"
    STRATEGY_TESTED = "strategy.tested"
    STRATEGY_TEST_FAILED = "strategy.test_failed"
    TUNE_REQUEST = "tune.request"
    STRATEGY_TUNED = "strategy.tuned"
    ROBUSTNESS_REPORT = "robustness.report"
    RISK_REPORT = "risk.report"
    STRATEGY_JUDGED = "strategy.judged"

    # delivery
    DOCS_READY = "docs.ready"
    PACKAGE_REQUEST = "package.request"
    PACKAGE_READY = "package.ready"
    DELIVERY_COMPLETED = "delivery.completed"
    DELIVERY_FAILED = "delivery.failed"

    # deployment (only on explicit user command)
    AUTOMATE_REQUEST = "automate.request"
    TRADINGVIEW_REQUEST = "tradingview.request"
    DEPLOY_STATUS = "deploy.status"

    # learning
    MEMORY_PRIORS = "memory.priors_updated"
    CHAMPION_CAMPAIGN = "champion.campaign"

    # observability (fan-out to the dashboard)
    AGENT_STATUS = "agent.status"
    LOG = "system.log"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    ts: datetime = Field(default_factory=_now)
    kind: MsgKind = MsgKind.EVENT
    topic: str
    sender: str
    payload: dict[str, Any] = Field(default_factory=dict)

    # correlation
    project_id: str | None = None  # groups a whole strategy's journey
    strategy_id: str | None = None
    reply_to: str | None = None  # id of the REQUEST this answers
    reply_topic: str | None = None  # where the responder should send the answer
    trace: list[str] = Field(default_factory=list)  # agent hops, for the UI graph

    def child(
        self,
        *,
        topic: str,
        sender: str,
        kind: MsgKind = MsgKind.EVENT,
        payload: dict[str, Any] | None = None,
        **extra: Any,
    ) -> Message:
        """Derive a message that keeps this one's correlation ids and trace."""
        return Message(
            topic=topic,
            sender=sender,
            kind=kind,
            payload=payload or {},
            project_id=extra.get("project_id", self.project_id),
            strategy_id=extra.get("strategy_id", self.strategy_id),
            reply_to=extra.get("reply_to"),
            reply_topic=extra.get("reply_topic"),
            trace=[*self.trace, sender],
        )

    def responds_to(self, sender: str, payload: dict[str, Any]) -> Message:
        """Build the RESPONSE for this REQUEST."""
        return Message(
            topic=self.reply_topic or Topic.DATA_RESPONSE,
            sender=sender,
            kind=MsgKind.RESPONSE,
            payload=payload,
            project_id=self.project_id,
            strategy_id=self.strategy_id,
            reply_to=self.id,
            trace=[*self.trace, sender],
        )
