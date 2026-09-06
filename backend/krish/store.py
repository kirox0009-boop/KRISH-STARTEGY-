"""The blackboard: long-lived facts every agent can read and write.

The bus carries *what just happened*; this carries *what is true*. Because every
stage checkpoints here, a crashed or restarted agent never loses a project — it
picks the work back up from the last recorded stage.

SQLite by default (zero setup), Postgres on the VPS via ``DATABASE_URL``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    cast,
    create_engine,
    delete,
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


# --------------------------------------------------------------------------- #
# housekeeping / retention
#
# A factory that runs forever writes forever. Left alone, the blackboard grows by
# roughly 130 KB per strategy - mostly equity curves and trade lists - which on a
# busy box is several GB a month. The librarian agent calls these to keep the disk
# flat without throwing away the numbers that matter: metrics and verdicts are
# always kept, only the bulky per-bar detail of *rejected* work is discarded.
# --------------------------------------------------------------------------- #


def _cutoff(*, days: float = 0.0, hours: float = 0.0) -> datetime:
    return _utcnow() - timedelta(days=days, hours=hours)


def prune_events(keep_days: float) -> int:
    """Drop flight-recorder rows older than ``keep_days``."""
    if keep_days <= 0:
        return 0
    with session() as s:
        result = s.execute(delete(AgentEvent).where(AgentEvent.ts < _cutoff(days=keep_days)))
        return int(result.rowcount or 0)


def strip_rejected_details(keep_hours: float) -> int:
    """Blank the heavy columns on runs belonging to rejected strategies.

    Keeps the row and its metrics, so the experiment ledger and everything the
    memory agent learns from stay intact. Only the equity curve and trade list -
    the parts nobody re-reads for a rejected idea - are released.
    """
    if keep_hours <= 0:
        return 0
    cutoff = _cutoff(hours=keep_hours)
    with session() as s:
        rejected = select(Verdict.strategy_id).where(Verdict.verdict == "REJECT")
        rows = s.scalars(
            select(BacktestRun).where(
                BacktestRun.strategy_id.in_(rejected),
                BacktestRun.created_at < cutoff,
                func.length(func.coalesce(cast(BacktestRun.equity, String), "")) > 2,
            )
        ).all()
        for row in rows:
            row.equity = []
            row.trades = []
        return len(rows)


def delete_old_rejected(keep_days: float) -> dict[str, int]:
    """Remove rejected strategies entirely once they are old enough to be noise."""
    if keep_days <= 0:
        return {"strategies": 0, "runs": 0, "verdicts": 0}
    cutoff = _cutoff(days=keep_days)
    with session() as s:
        ids = list(
            s.scalars(
                select(Verdict.strategy_id).where(
                    Verdict.verdict == "REJECT", Verdict.created_at < cutoff
                )
            ).all()
        )
        if not ids:
            return {"strategies": 0, "runs": 0, "verdicts": 0}
        runs = s.execute(delete(BacktestRun).where(BacktestRun.strategy_id.in_(ids)))
        verdicts = s.execute(delete(Verdict).where(Verdict.strategy_id.in_(ids)))
        strategies = s.execute(delete(Strategy).where(Strategy.id.in_(ids)))
        return {
            "strategies": int(strategies.rowcount or 0),
            "runs": int(runs.rowcount or 0),
            "verdicts": int(verdicts.rowcount or 0),
        }


def vacuum() -> bool:
    """Return freed pages to the filesystem. SQLite only; a no-op elsewhere."""
    if _engine is None:
        init_db()
    assert _engine is not None
    if _engine.dialect.name != "sqlite":
        return False
    with _engine.connect() as conn:
        conn.exec_driver_sql("VACUUM")
    return True


def database_bytes() -> int:
    url = settings().database_url
    if not url.startswith("sqlite"):
        return 0
    path = Path(url.split("sqlite:///")[-1])
    return path.stat().st_size if path.exists() else 0


def accepted_strategies(
    *, include_borderline: bool = True, limit: int = 60
) -> list[dict[str, Any]]:
    """The vault: strategies that cleared the judge, with their download link.

    This is the answer to "it says 8 passed - where are they?". A verdict on its
    own is not a deliverable, so each row carries the package, its checksum and a
    download URL when one exists, and says so plainly when one does not.
    """
    wanted = ["PASS", "BORDERLINE"] if include_borderline else ["PASS"]
    with session() as s:
        # A strategy can be judged more than once - re-running a backtest from the
        # UI produces a fresh verdict. Only the newest one counts, otherwise the
        # same strategy appears in the vault several times.
        newest = (
            select(
                Verdict.strategy_id.label("sid"),
                func.max(Verdict.created_at).label("latest"),
            )
            .group_by(Verdict.strategy_id)
            .subquery()
        )
        rows = s.execute(
            select(Strategy, Verdict)
            .join(Verdict, Verdict.strategy_id == Strategy.id)
            .join(
                newest,
                (newest.c.sid == Verdict.strategy_id) & (newest.c.latest == Verdict.created_at),
            )
            .where(Verdict.verdict.in_(wanted))
            .order_by(Verdict.score.desc(), Verdict.created_at.desc())
            .limit(limit)
        ).all()

        # Belt and braces: identical timestamps on two verdicts would still slip
        # a duplicate through the join above.
        seen: set[str] = set()
        rows = [
            (st, v)
            for st, v in rows
            if not (st.id in seen or seen.add(st.id))  # type: ignore[func-returns-value]
        ]

        out: list[dict[str, Any]] = []
        for strategy, verdict in rows:
            delivery = s.scalars(
                select(Delivery)
                .where(Delivery.strategy_id == strategy.id)
                .order_by(Delivery.created_at.desc())
                .limit(1)
            ).first()

            # Prefer the out-of-sample numbers - the only ones that mean anything.
            run = s.scalars(
                select(BacktestRun)
                .where(
                    BacktestRun.strategy_id == strategy.id,
                    BacktestRun.kind.in_(["tuned_oos", "oos"]),
                )
                .order_by(BacktestRun.created_at.desc())
                .limit(1)
            ).first()
            metrics = dict(run.metrics or {}) if run else {}

            remote = (delivery.channels or {}).get("object_store") if delivery else None

            # Label the horizon from the trade log, so the card can say "swing
            # trend-following" rather than leaving the reader to work it out.
            kind: dict[str, Any] = {}
            try:
                from .classify import classify
                from .ir.schema import StrategyIR

                kind = classify(StrategyIR.model_validate(strategy.ir), metrics)
            except Exception:
                kind = {}

            out.append(
                {
                    "strategy_id": strategy.id,
                    "name": strategy.name,
                    "style": strategy.style,
                    "kind": kind,
                    "asset": strategy.asset,
                    "timeframe": strategy.timeframe,
                    "generation": strategy.generation,
                    "origin": strategy.origin,
                    "verdict": verdict.verdict,
                    "score": verdict.score,
                    "long_term_viable": verdict.long_term_viable,
                    "summary": verdict.summary,
                    "reasons": verdict.reasons,
                    "judged_at": verdict.created_at.isoformat() if verdict.created_at else None,
                    "metrics": {
                        k: metrics.get(k)
                        for k in (
                            "trades",
                            "sharpe",
                            "profit_factor",
                            "win_rate_pct",
                            "expectancy_r",
                            "max_drawdown_pct",
                            "total_return_pct",
                        )
                    },
                    "package": (
                        {
                            "name": delivery.package_name,
                            "size_bytes": delivery.size_bytes,
                            "checksum": delivery.checksum,
                            "status": delivery.status,
                            "download": f"/api/deliveries/{delivery.id}/download",
                            "remote_url": (remote or {}).get("url"),
                        }
                        if delivery
                        else None
                    ),
                }
            )
        return out


def near_misses(limit: int = 12) -> list[dict[str, Any]]:
    """Strategies that did NOT pass, ranked by how close they came.

    Exists because "nothing has passed for three hours" is an unanswerable
    complaint without it. This turns silence into a diagnosis: which gate is
    blocking, by how much, and whether the bar is set somewhere reachable.
    """
    with session() as s:
        newest = (
            select(
                Verdict.strategy_id.label("sid"),
                func.max(Verdict.created_at).label("latest"),
            )
            .group_by(Verdict.strategy_id)
            .subquery()
        )
        rows = s.execute(
            select(Strategy, Verdict)
            .join(Verdict, Verdict.strategy_id == Strategy.id)
            .join(
                newest,
                (newest.c.sid == Verdict.strategy_id) & (newest.c.latest == Verdict.created_at),
            )
            .where(Verdict.verdict != "PASS")
            .order_by(Verdict.created_at.desc())
            .limit(400)
        ).all()

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for strategy, verdict in rows:
            if strategy.id in seen:
                continue
            seen.add(strategy.id)
            checks = verdict.checks or {}
            failed = [c for c in checks.values() if not c.get("pass")]
            if not checks:
                continue
            # ratio < 1 means "this far short of the bar"; the worst one is the
            # binding constraint, which is the number worth showing.
            worst = min((float(c.get("ratio", 0)) for c in failed), default=1.0)
            blocker = min(failed, key=lambda c: float(c.get("ratio", 0)), default=None)
            out.append(
                {
                    "strategy_id": strategy.id,
                    "name": strategy.name,
                    "asset": strategy.asset,
                    "timeframe": strategy.timeframe,
                    "style": strategy.style,
                    "verdict": verdict.verdict,
                    "score": verdict.score,
                    "failed_count": len(failed),
                    "passed_count": len(checks) - len(failed),
                    "total_checks": len(checks),
                    "closeness": round(worst, 3),
                    "blocker": (
                        {
                            "label": blocker.get("label"),
                            "value": blocker.get("value"),
                            "threshold": blocker.get("threshold"),
                            "ratio": blocker.get("ratio"),
                        }
                        if blocker
                        else None
                    ),
                    "failed": [
                        {
                            "label": c.get("label"),
                            "value": c.get("value"),
                            "threshold": c.get("threshold"),
                        }
                        for c in failed
                    ],
                }
            )
        # fewest failures first, then whichever came closest on its worst gate
        out.sort(key=lambda r: (r["failed_count"], -r["closeness"]))
        return out[:limit]


def gate_stats(limit: int = 500) -> dict[str, Any]:
    """How often each individual threshold is cleared.

    The single most useful number when nothing passes: if one gate has a 0% pass
    rate, that gate is the problem, not the strategies.
    """
    with session() as s:
        verdicts = list(s.scalars(select(Verdict).order_by(Verdict.created_at.desc()).limit(limit)))
    tally: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        for key, c in (v.checks or {}).items():
            slot = tally.setdefault(
                key,
                {
                    "label": c.get("label", key),
                    "passed": 0,
                    "total": 0,
                    "threshold": c.get("threshold"),
                    "best": None,
                    "best_ratio": -1.0,
                },
            )
            slot["total"] += 1
            if c.get("pass"):
                slot["passed"] += 1
            # `ratio` is already normalised so that higher means closer to (or
            # further past) the bar, whichever direction the gate points. Picking
            # the best by ratio therefore works for "max drawdown" too, where a
            # lower raw value is better.
            ratio = float(c.get("ratio", 0) or 0)
            if c.get("value") is not None and ratio >= slot["best_ratio"]:
                slot["best_ratio"] = ratio
                slot["best"] = c.get("value")
    return {
        "judged": len(verdicts),
        "gates": sorted(
            (
                {**v, "key": k, "pass_rate": round(v["passed"] / max(v["total"], 1), 3)}
                for k, v in tally.items()
            ),
            key=lambda r: r["pass_rate"],
        ),
    }


def purge_rejected_strategy(strategy_id: str) -> dict[str, int]:
    """Delete a failed strategy's bulk the moment it is rejected. No waiting.

    What goes immediately: every backtest run (equity curves and trade lists -
    about 125 KB, roughly 97% of what a strategy costs) and its event log.

    What is deliberately kept, at roughly 5 KB: the strategy row with its
    fingerprint, and the verdict with its check results. Those are not sentiment -
    they are load-bearing:

      * the fingerprint stops the factory re-inventing and re-testing the same
        failing idea forever, which would waste far more than it saves
      * the memory agent builds its priors from failures as much as successes;
        delete them and the factory stops learning what does not work
      * the "why nothing is passing" panel is built from these check results

    So this keeps the diagnosis and throws away the evidence locker.
    """
    with session() as s:
        runs = s.execute(delete(BacktestRun).where(BacktestRun.strategy_id == strategy_id))
        events = s.execute(delete(AgentEvent).where(AgentEvent.strategy_id == strategy_id))
        row = s.get(Strategy, strategy_id)
        ir_freed = 0
        if row is not None:
            ir_freed = len(json.dumps(row.ir or {}))
            row.ir = {}
            row.status = "purged_reject"
    return {
        "runs_deleted": int(runs.rowcount or 0),
        "events_deleted": int(events.rowcount or 0),
        "ir_bytes_freed": ir_freed,
    }
