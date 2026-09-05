"""The blackboard: long-lived facts every agent can read and write.

The bus carries *what just happened*; this carries *what is true*. Because every
stage checkpoints here, a crashed or restarted agent never loses a project — it
picks the work back up from the last recorded stage.

SQLite by default (zero setup), Postgres on the VPS via ``DATABASE_URL``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import settings
from .messages import new_id


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Project(Base):
    """One journey: idea -> strategy -> tested -> judged -> delivered."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("prj"))
    title: Mapped[str] = mapped_column(String(200), default="")
    asset: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    stage: Mapped[str] = mapped_column(String(32), default="created", index=True)
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)
    owner_agent: Mapped[str] = mapped_column(String(40), default="")
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    strategy_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Strategy(Base):
    """A strategy *is* its IR. Everything else is derived from this row."""

    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("stg"))
    project_id: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    style: Mapped[str] = mapped_column(String(40), default="unknown")
    asset: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    generation: Mapped[int] = mapped_column(Integer, default=0, index=True)
    parents: Mapped[list[Any]] = mapped_column(default=list)
    origin: Mapped[str] = mapped_column(String(24), default="fresh")  # fresh|mutation|crossover
    ir: Mapped[dict[str, Any]] = mapped_column(default=dict)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True, default="")
    status: Mapped[str] = mapped_column(String(24), default="created", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class BacktestRun(Base):
    """Every backtest ever run — pass or fail. This is the learning dataset."""

    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("run"))
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), index=True)
    project_id: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(24), default="full")  # full|is|oos|walkforward|tuned
    asset: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    equity: Mapped[list[Any]] = mapped_column(default=list)  # downsampled curve for the UI
    trades: Mapped[list[Any]] = mapped_column(default=list)  # capped sample for the UI
    bars: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("vdt"))
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), index=True)
    project_id: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    verdict: Mapped[str] = mapped_column(String(16), index=True)  # PASS|BORDERLINE|REJECT
    score: Mapped[float] = mapped_column(Float, default=0.0)
    long_term_viable: Mapped[bool] = mapped_column(default=False)
    checks: Mapped[dict[str, Any]] = mapped_column(default=dict)
    reasons: Mapped[list[Any]] = mapped_column(default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Delivery(Base):
    __tablename__ = "deliveries"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("dlv"))
    strategy_id: Mapped[str] = mapped_column(String(40), index=True)
    package_name: Mapped[str] = mapped_column(String(200), default="")
    local_path: Mapped[str] = mapped_column(Text, default="")
    checksum: Mapped[str] = mapped_column(String(64), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    channels: Mapped[dict[str, Any]] = mapped_column(default=dict)  # channel -> url/status
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AgentEvent(Base):
    """Flight recorder. Feeds the dashboard timeline and post-mortems."""

    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    agent: Mapped[str] = mapped_column(String(40), index=True)
    topic: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="event")
    level: Mapped[str] = mapped_column(String(10), default="info")
    project_id: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    strategy_id: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Prior(Base):
    """Aggregated learning: "ATR stops beat fixed stops on GOLD H1 (n=340)"."""

    __tablename__ = "priors"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("pri"))
    scope: Mapped[str] = mapped_column(String(64), index=True)  # e.g. "GOLD:H1"
    key: Mapped[str] = mapped_column(String(120), index=True)  # e.g. "stop.atr_vs_fixed"
    value: Mapped[dict[str, Any]] = mapped_column(default=dict)
    samples: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# --------------------------------------------------------------------------- #
# engine / session
# --------------------------------------------------------------------------- #

_engine = None
_Session: sessionmaker[Session] | None = None


def init_db(url: str | None = None, *, echo: bool = False) -> None:
    global _engine, _Session
    url = url or settings().database_url
    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    _engine = create_engine(url, **kwargs)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)


@contextmanager
def session() -> Iterator[Session]:
    if _Session is None:
        init_db()
    assert _Session is not None
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


async def in_db(fn, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking DB function off the event loop."""
    return await asyncio.to_thread(fn, *args, **kwargs)


# --------------------------------------------------------------------------- #
# small query helpers used across agents and the API
# --------------------------------------------------------------------------- #


def save(obj: Base) -> Base:
    with session() as s:
        s.add(obj)
        s.flush()
        s.refresh(obj)
        return obj


def get_strategy(strategy_id: str) -> Strategy | None:
    with session() as s:
        return s.get(Strategy, strategy_id)


def get_project(project_id: str) -> Project | None:
    with session() as s:
        return s.get(Project, project_id)


def update_project(project_id: str, **fields: Any) -> None:
    with session() as s:
        proj = s.get(Project, project_id)
        if proj is None:
            return
        for key, value in fields.items():
            setattr(proj, key, value)


def update_strategy(strategy_id: str, **fields: Any) -> None:
    with session() as s:
        stg = s.get(Strategy, strategy_id)
        if stg is None:
            return
        for key, value in fields.items():
            setattr(stg, key, value)


def recent_projects(limit: int = 50) -> Sequence[Project]:
    with session() as s:
        return list(s.scalars(select(Project).order_by(Project.updated_at.desc()).limit(limit)))


def recent_strategies(limit: int = 100, asset: str | None = None) -> Sequence[Strategy]:
    with session() as s:
        stmt = select(Strategy).order_by(Strategy.created_at.desc()).limit(limit)
        if asset:
            stmt = stmt.where(Strategy.asset == asset.upper())
        return list(s.scalars(stmt))


def runs_for_strategy(strategy_id: str) -> Sequence[BacktestRun]:
    with session() as s:
        return list(
            s.scalars(
                select(BacktestRun)
                .where(BacktestRun.strategy_id == strategy_id)
                .order_by(BacktestRun.created_at.asc())
            )
        )


def verdict_for_strategy(strategy_id: str) -> Verdict | None:
    with session() as s:
        return s.scalars(
            select(Verdict)
            .where(Verdict.strategy_id == strategy_id)
            .order_by(Verdict.created_at.desc())
            .limit(1)
        ).first()


def elite_strategies(asset: str, limit: int = 20) -> Sequence[Strategy]:
    """Best scoring strategies for an asset — the breeding stock."""
    with session() as s:
        rows = s.execute(
            select(Strategy, Verdict.score)
            .join(Verdict, Verdict.strategy_id == Strategy.id)
            .where(Strategy.asset == asset.upper())
            .order_by(Verdict.score.desc())
            .limit(limit)
        ).all()
        return [row[0] for row in rows]


def fingerprint_exists(fingerprint: str) -> bool:
    if not fingerprint:
        return False
    with session() as s:
        return (
            s.scalar(
                select(func.count())
                .select_from(Strategy)
                .where(Strategy.fingerprint == fingerprint)
            )
            or 0
        ) > 0


def log_event(
    *,
    agent: str,
    topic: str,
    kind: str = "event",
    level: str = "info",
    message: str = "",
    project_id: str | None = None,
    strategy_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    with session() as s:
        s.add(
            AgentEvent(
                agent=agent,
                topic=topic,
                kind=kind,
                level=level,
                message=message,
                project_id=project_id,
                strategy_id=strategy_id,
                payload=payload or {},
            )
        )


def recent_events(limit: int = 200, project_id: str | None = None) -> Sequence[AgentEvent]:
    with session() as s:
        stmt = select(AgentEvent).order_by(AgentEvent.ts.desc()).limit(limit)
        if project_id:
            stmt = stmt.where(AgentEvent.project_id == project_id)
        return list(s.scalars(stmt))


def counts() -> dict[str, int]:
    with session() as s:

        def _count(model: Any, *where: Any) -> int:
            stmt = select(func.count()).select_from(model)
            for clause in where:
                stmt = stmt.where(clause)
            return int(s.scalar(stmt) or 0)

        return {
            "projects": _count(Project),
            "projects_running": _count(Project, Project.status == "running"),
            "strategies": _count(Strategy),
            "backtests": _count(BacktestRun),
            "passed": _count(Verdict, Verdict.verdict == "PASS"),
            "borderline": _count(Verdict, Verdict.verdict == "BORDERLINE"),
            "rejected": _count(Verdict, Verdict.verdict == "REJECT"),
            "deliveries": _count(Delivery),
        }


def priors_for(scope: str) -> dict[str, Any]:
    """Flatten the memory agent's learnings for one scope, e.g. ``"GOLD:H1"``.

    Falls back to the asset-wide scope, then global, so a brand-new timeframe
    still starts from whatever the factory already knows.
    """
    scopes = [scope]
    if ":" in scope:
        scopes.append(scope.split(":")[0])
    scopes.append("*")

    merged: dict[str, Any] = {}
    with session() as s:
        for sc in reversed(scopes):  # most specific wins
            rows = s.scalars(select(Prior).where(Prior.scope == sc)).all()
            for row in rows:
                merged[row.key] = row.value
    return merged


def upsert_prior(
    scope: str, key: str, value: dict[str, Any], *, samples: int, confidence: float
) -> None:
    with session() as s:
        row = s.scalars(
            select(Prior).where(Prior.scope == scope, Prior.key == key).limit(1)
        ).first()
        if row is None:
            s.add(Prior(scope=scope, key=key, value=value, samples=samples, confidence=confidence))
        else:
            row.value = value
            row.samples = samples
            row.confidence = confidence


def all_priors() -> list[dict[str, Any]]:
    with session() as s:
        rows = s.scalars(select(Prior).order_by(Prior.scope, Prior.key)).all()
        return [
            {
                "scope": r.scope,
                "key": r.key,
                "value": r.value,
                "samples": r.samples,
                "confidence": r.confidence,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in rows
        ]


def judged_strategies(asset: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    """Experiment ledger view: strategy + its verdict, newest first."""
    with session() as s:
        stmt = (
            select(Strategy, Verdict)
            .join(Verdict, Verdict.strategy_id == Strategy.id)
            .order_by(Verdict.created_at.desc())
            .limit(limit)
        )
        if asset:
            stmt = stmt.where(Strategy.asset == asset.upper())
        out = []
        for strategy, verdict in s.execute(stmt).all():
            out.append(
                {
                    "strategy_id": strategy.id,
                    "name": strategy.name,
                    "style": strategy.style,
                    "asset": strategy.asset,
                    "timeframe": strategy.timeframe,
                    "generation": strategy.generation,
                    "origin": strategy.origin,
                    "recipe": str(strategy.ir.get("notes", "")),
                    "verdict": verdict.verdict,
                    "score": verdict.score,
                    "checks": verdict.checks,
                }
            )
        return out
