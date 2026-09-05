"""FastAPI surface: read the factory, and control it.

Everything the control room does goes through here. Control actions are turned
into bus messages rather than direct calls, so the UI has exactly the same
authority as an agent — no privileged back door, and every action shows up in the
same event stream you are watching.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .agents.data import data_health
from .agents.system import MonitorAgent, storage_report
from .assets import add_asset, remove_asset, universe
from .bus import bus
from .compilers.pine import PineUnsupported, to_pine
from .config import PACKAGE_DIR, factory_config, load_yaml, save_yaml
from .indicators import indicators_by_family
from .ir.schema import StrategyIR
from .messages import Message, MsgKind, Topic
from .registry import registry
from .runtime import Factory

log = logging.getLogger("krish.api")

WEB_DIR = Path(__file__).parent / "web"


# --------------------------------------------------------------------------- #
# Request bodies.
#
# These MUST live at module level. This file uses `from __future__ import
# annotations`, so every annotation is a string that FastAPI resolves against the
# module's globals. A model defined inside create_app() is invisible there, and
# FastAPI silently downgrades it to a *query* parameter - which makes every
# control endpoint reject its JSON body with 422.
# --------------------------------------------------------------------------- #


class ConfigPatch(BaseModel):
    data: dict[str, Any]


class AssetIn(BaseModel):
    key: str
    name: str
    asset_class: str = "unknown"
    yfinance: str | None = None
    ccxt: str | None = None
    mt5: str | None = None
    tradingview: str | None = None
    tick_size: float = 0.01
    point_value: float = 1.0
    session: str = "24h"
    timeframes: list[str] = ["H1", "D1"]
    spread_points: float = 10.0
    commission_per_lot: float = 0.0
    slippage_points: float = 5.0


class AgentControl(BaseModel):
    agent: str = "*"
    action: str  # pause | resume | stop | reload


class CycleRequest(BaseModel):
    asset: str | None = None
    timeframe: str | None = None
    count: int = 0


class AutomateRequest(BaseModel):
    target: str = "mt5"  # mt5 | tradingview
    mode: str = "demo"  # demo | live


def create_app(factory: Factory | None = None) -> FastAPI:
    app = FastAPI(
        title="KRISH Control API",
        version="0.1.0",
        description="Autonomous multi-agent trading strategy factory.",
    )
    app.state.factory = factory

    # ------------------------------------------------------------------ #
    # overview / health
    # ------------------------------------------------------------------ #

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        return {
            "counts": await store.in_db(store.counts),
            "agents": registry().summary(),
            "assets": [
                {
                    "key": a.key,
                    "name": a.name,
                    "class": a.asset_class,
                    "timeframes": list(a.timeframes),
                    "session": a.session,
                }
                for a in universe().all()
            ],
            "config": factory_config(),
        }

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return MonitorAgent.snapshot()

    @app.get("/api/health/data")
    async def health_data() -> list[dict[str, Any]]:
        return await asyncio.to_thread(data_health)

    @app.get("/api/health/storage")
    async def health_storage() -> dict[str, Any]:
        """Where the disk is going, and what is bounded vs growing."""
        return await asyncio.to_thread(storage_report)

    @app.get("/api/agents")
    async def agents() -> list[dict[str, Any]]:
        return registry().snapshot()

    @app.get("/api/agents/roster")
    async def roster() -> list[dict[str, Any]]:
        fac: Factory | None = app.state.factory
        return fac.describe() if fac else []

    @app.get("/api/indicators")
    async def indicators() -> dict[str, list[str]]:
        return indicators_by_family()

    # ------------------------------------------------------------------ #
    # projects / strategies
    # ------------------------------------------------------------------ #

    @app.get("/api/projects")
    async def projects(limit: int = Query(50, le=500)) -> list[dict[str, Any]]:
        rows = await store.in_db(store.recent_projects, limit)
        return [
            {
                "id": p.id,
                "title": p.title,
                "asset": p.asset,
                "timeframe": p.timeframe,
                "stage": p.stage,
                "status": p.status,
                "hypothesis": p.hypothesis,
                "strategy_id": p.strategy_id,
                "meta": p.meta,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                "elapsed_seconds": (
                    round((p.updated_at - p.created_at).total_seconds(), 1)
                    if p.created_at and p.updated_at
                    else None
                ),
            }
            for p in rows
        ]

    @app.get("/api/strategies")
    async def strategies(
        limit: int = Query(100, le=1000), asset: str | None = None
    ) -> list[dict[str, Any]]:
        rows = await store.in_db(store.recent_strategies, limit, asset)
        out: list[dict[str, Any]] = []
        for s in rows:
            verdict = await store.in_db(store.verdict_for_strategy, s.id)
            out.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "style": s.style,
                    "asset": s.asset,
                    "timeframe": s.timeframe,
                    "generation": s.generation,
                    "origin": s.origin,
                    "status": s.status,
                    "parents": s.parents,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "verdict": verdict.verdict if verdict else None,
                    "score": verdict.score if verdict else None,
                    "long_term_viable": verdict.long_term_viable if verdict else None,
                }
            )
        return out

    @app.get("/api/strategies/{strategy_id}")
    async def strategy_detail(strategy_id: str) -> dict[str, Any]:
        row = await store.in_db(store.get_strategy, strategy_id)
        if row is None:
            raise HTTPException(404, "strategy not found")
        runs = await store.in_db(store.runs_for_strategy, strategy_id)
        verdict = await store.in_db(store.verdict_for_strategy, strategy_id)
        try:
            described = StrategyIR.model_validate(row.ir).describe()
        except Exception as exc:
            described = f"(IR could not be rendered: {exc})"
        return {
            "id": row.id,
            "name": row.name,
            "style": row.style,
            "asset": row.asset,
            "timeframe": row.timeframe,
            "generation": row.generation,
            "origin": row.origin,
            "parents": row.parents,
            "status": row.status,
            "ir": row.ir,
            "description": described,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "runs": [
                {
                    "id": r.id,
                    "kind": r.kind,
                    "metrics": r.metrics,
                    "equity": r.equity,
                    "trades": r.trades,
                    "params": r.params,
                    "bars": r.bars,
                    "duration_ms": r.duration_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in runs
            ],
            "verdict": (
                {
                    "verdict": verdict.verdict,
                    "score": verdict.score,
                    "long_term_viable": verdict.long_term_viable,
                    "checks": verdict.checks,
                    "reasons": verdict.reasons,
                    "summary": verdict.summary,
                }
                if verdict
                else None
            ),
        }

    @app.get("/api/strategies/{strategy_id}/pine", response_class=PlainTextResponse)
    async def strategy_pine(strategy_id: str) -> str:
        row = await store.in_db(store.get_strategy, strategy_id)
        if row is None:
            raise HTTPException(404, "strategy not found")
        try:
            return to_pine(StrategyIR.model_validate(row.ir))
        except PineUnsupported as exc:
            raise HTTPException(422, f"not exportable to Pine Script: {exc}") from exc

    @app.get("/api/ledger")
    async def ledger(asset: str | None = None, limit: int = Query(200, le=1000)):
        return await store.in_db(store.judged_strategies, asset, limit)

    @app.get("/api/priors")
    async def priors() -> list[dict[str, Any]]:
        return await store.in_db(store.all_priors)

    @app.get("/api/events")
    async def events(
        limit: int = Query(200, le=2000), project_id: str | None = None
    ) -> list[dict[str, Any]]:
        rows = await store.in_db(store.recent_events, limit, project_id)
        return [
            {
                "ts": e.ts.isoformat() if e.ts else None,
                "agent": e.agent,
                "topic": e.topic,
                "level": e.level,
                "message": e.message,
                "project_id": e.project_id,
                "strategy_id": e.strategy_id,
            }
            for e in rows
        ]

    # ------------------------------------------------------------------ #
    # deliveries
    # ------------------------------------------------------------------ #

    @app.get("/api/deliveries")
    async def deliveries(limit: int = Query(50, le=500)) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            from sqlalchemy import select

            with store.session() as s:
                rows = s.scalars(
                    select(store.Delivery).order_by(store.Delivery.created_at.desc()).limit(limit)
                ).all()
                return [
                    {
                        "id": d.id,
                        "strategy_id": d.strategy_id,
                        "package_name": d.package_name,
                        "checksum": d.checksum,
                        "size_bytes": d.size_bytes,
                        "channels": d.channels,
                        "status": d.status,
                        "created_at": d.created_at.isoformat() if d.created_at else None,
                        "download": f"/api/deliveries/{d.id}/download",
                    }
                    for d in rows
                ]

        return await store.in_db(_query)

    @app.get("/api/deliveries/{delivery_id}/download")
    async def download(delivery_id: str) -> FileResponse:
        def _get() -> Any:
            with store.session() as s:
                return s.get(store.Delivery, delivery_id)

        record = await store.in_db(_get)
        if record is None:
            raise HTTPException(404, "delivery not found")
        path = Path(record.local_path)
        if not path.exists():
            raise HTTPException(410, "package file is no longer on disk")
        return FileResponse(path, filename=path.name, media_type="application/zip")

    # ------------------------------------------------------------------ #
    # configuration (editable from the UI)
    # ------------------------------------------------------------------ #

    @app.get("/api/config/{name}")
    async def get_config(name: str) -> dict[str, Any]:
        if name not in {"factory", "assets"}:
            raise HTTPException(404, "unknown config file")
        return load_yaml(name, refresh=True)

    @app.put("/api/config/{name}")
    async def put_config(name: str, patch: ConfigPatch) -> dict[str, Any]:
        if name not in {"factory", "assets"}:
            raise HTTPException(404, "unknown config file")
        save_yaml(name, patch.data)
        if name == "assets":
            universe(refresh=True)
        await bus().publish(
            Message(
                topic=Topic.CONFIG_RELOAD,
                sender="api",
                kind=MsgKind.CONTROL,
                payload={"file": name},
            )
        )
        return {"ok": True, "file": name}

    @app.get("/api/assets")
    async def get_assets() -> list[dict[str, Any]]:
        return [
            {
                "key": a.key,
                "name": a.name,
                "class": a.asset_class,
                "timeframes": list(a.timeframes),
                "session": a.session,
                "tick_size": a.tick_size,
                "point_value": a.point_value,
                "symbols": a.symbols,
                "cost": {
                    "spread_points": a.cost.spread_points,
                    "commission_per_lot": a.cost.commission_per_lot,
                    "slippage_points": a.cost.slippage_points,
                },
            }
            for a in universe().all()
        ]

    @app.post("/api/assets")
    async def post_asset(payload: AssetIn) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "key": payload.key,
            "name": payload.name,
            "class": payload.asset_class,
            "tick_size": payload.tick_size,
            "point_value": payload.point_value,
            "session": payload.session,
            "timeframes": payload.timeframes,
            "cost": {
                "spread_points": payload.spread_points,
                "commission_per_lot": payload.commission_per_lot,
                "slippage_points": payload.slippage_points,
            },
        }
        for venue in ("yfinance", "ccxt", "mt5", "tradingview"):
            value = getattr(payload, venue)
            if value:
                entry[venue] = value
        try:
            asset = await asyncio.to_thread(add_asset, entry)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        await bus().publish(
            Message(
                topic=Topic.CONFIG_RELOAD,
                sender="api",
                kind=MsgKind.CONTROL,
                payload={"file": "assets", "added": asset.key},
            )
        )
        return {"ok": True, "asset": asset.key}

    @app.delete("/api/assets/{key}")
    async def delete_asset(key: str) -> dict[str, Any]:
        await asyncio.to_thread(remove_asset, key)
        await bus().publish(
            Message(
                topic=Topic.CONFIG_RELOAD,
                sender="api",
                kind=MsgKind.CONTROL,
                payload={"file": "assets", "removed": key.upper()},
            )
        )
        return {"ok": True, "removed": key.upper()}

    # ------------------------------------------------------------------ #
    # control
    # ------------------------------------------------------------------ #

    @app.post("/api/control/agent")
    async def control_agent(body: AgentControl) -> dict[str, Any]:
        if body.action not in {"pause", "resume", "stop", "kill", "reload"}:
            raise HTTPException(400, "action must be pause|resume|stop|reload")
        await bus().publish(
            Message(
                topic=Topic.AGENT_CONTROL,
                sender="api",
                kind=MsgKind.CONTROL,
                payload={"agent": body.agent, "action": body.action},
            )
        )
        return {"ok": True, **body.model_dump()}

    @app.post("/api/control/cycle")
    async def control_cycle(body: CycleRequest) -> dict[str, Any]:
        await bus().publish(
            Message(
                topic=Topic.AGENT_CONTROL,
                sender="api",
                kind=MsgKind.CONTROL,
                payload={"agent": "orchestrator", "action": "cycle", **body.model_dump()},
            )
        )
        return {"ok": True, "queued": body.model_dump()}

    @app.post("/api/control/retest/{strategy_id}")
    async def retest(strategy_id: str) -> dict[str, Any]:
        row = await store.in_db(store.get_strategy, strategy_id)
        if row is None:
            raise HTTPException(404, "strategy not found")
        await bus().publish(
            Message(
                topic=Topic.STRATEGY_BUILT,
                sender="api",
                kind=MsgKind.TASK,
                strategy_id=strategy_id,
                project_id=row.project_id,
                payload={
                    "strategy_id": strategy_id,
                    "name": row.name,
                    "asset": row.asset,
                    "timeframe": row.timeframe,
                    "ir": row.ir,
                    "repairs": [],
                    "requested_by": "operator",
                },
            )
        )
        return {"ok": True, "strategy_id": strategy_id, "note": "re-entered the pipeline at tester"}

    @app.post("/api/control/deliver/{strategy_id}")
    async def deliver(strategy_id: str) -> dict[str, Any]:
        row = await store.in_db(store.get_strategy, strategy_id)
        if row is None:
            raise HTTPException(404, "strategy not found")
        verdict = await store.in_db(store.verdict_for_strategy, strategy_id)
        await bus().publish(
            Message(
                topic=Topic.PACKAGE_REQUEST,
                sender="api",
                kind=MsgKind.TASK,
                strategy_id=strategy_id,
                project_id=row.project_id,
                payload={
                    "strategy_id": strategy_id,
                    "ir": row.ir,
                    "verdict": verdict.verdict if verdict else "MANUAL",
                    "score": verdict.score if verdict else None,
                    "summary": verdict.summary if verdict else "Packaged on operator request.",
                    "checks": verdict.checks if verdict else {},
                    "reasons": verdict.reasons if verdict else [],
                    "dossier_md": verdict.summary if verdict else "",
                    "instructions_md": "",
                    "metrics": {},
                },
            )
        )
        return {"ok": True, "strategy_id": strategy_id}

    @app.post("/api/control/automate/{strategy_id}")
    async def automate(strategy_id: str, body: AutomateRequest) -> dict[str, Any]:
        row = await store.in_db(store.get_strategy, strategy_id)
        if row is None:
            raise HTTPException(404, "strategy not found")
        topic = (
            Topic.TRADINGVIEW_REQUEST if body.target == "tradingview" else Topic.AUTOMATE_REQUEST
        )
        await bus().publish(
            Message(
                topic=topic,
                sender="api",
                kind=MsgKind.TASK,
                strategy_id=strategy_id,
                payload={"strategy_id": strategy_id, "ir": row.ir, **body.model_dump()},
            )
        )
        return {
            "ok": True,
            "strategy_id": strategy_id,
            "target": body.target,
            "mode": body.mode,
            "note": (
                "Request published. The MT5 bridge and mt5_deploy agent land in Phase 7; "
                "TradingView Pine is available now via /api/strategies/{id}/pine."
            ),
        }

    # ------------------------------------------------------------------ #
    # live stream
    # ------------------------------------------------------------------ #

    @app.websocket("/ws/live")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        stop = asyncio.Event()

        async def pump_bus() -> None:
            async for msg in bus().tap():
                if stop.is_set():
                    return
                with contextlib.suppress(Exception):
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "message",
                                "ts": msg.ts.isoformat(),
                                "topic": msg.topic,
                                "kind": str(msg.kind),
                                "sender": msg.sender,
                                "project_id": msg.project_id,
                                "strategy_id": msg.strategy_id,
                                "trace": msg.trace,
                                "summary": _summarise(msg),
                            }
                        )
                    )

        async def pump_agents() -> None:
            while not stop.is_set():
                with contextlib.suppress(Exception):
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "agents",
                                "agents": registry().snapshot(),
                                "summary": registry().summary(),
                                "counts": store.counts(),
                            }
                        )
                    )
                await asyncio.sleep(2.0)

        tasks = [asyncio.create_task(pump_bus()), asyncio.create_task(pump_agents())]
        try:
            while True:
                await ws.receive_text()  # keepalive / client pings
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    # ------------------------------------------------------------------ #
    # control room (temporary UI; Phase 6 replaces it with the Next.js app)
    # ------------------------------------------------------------------ #

    if (WEB_DIR / "static").exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        page = WEB_DIR / "control_room.html"
        if not page.exists():
            return "<h1>KRISH</h1><p>Control room page missing.</p>"
        return page.read_text(encoding="utf-8")

    @app.get("/api/packages")
    async def packages() -> list[dict[str, Any]]:
        return [
            {"name": p.name, "size_bytes": p.stat().st_size, "modified": p.stat().st_mtime}
            for p in sorted(
                PACKAGE_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        ]

    return app


def _summarise(msg: Message) -> str:
    """One-line human description for the live feed."""
    p = msg.payload
    name = p.get("name") or p.get("title") or ""
    match msg.topic:
        case Topic.CYCLE_START:
            return (
                f"cycle {p.get('cycle')} · {p.get('asset')} {p.get('timeframe')} · "
                f"{p.get('count')} ideas"
            )
        case Topic.HYPOTHESIS_CREATED:
            return f"{p.get('mode')} idea: {name}"
        case Topic.STRATEGY_CREATED:
            return f"designed {name} ({p.get('origin')}, gen {p.get('generation')})"
        case Topic.STRATEGY_BUILT:
            repairs = p.get("repairs") or []
            return f"built {name}" + (f" ({len(repairs)} repairs)" if repairs else "")
        case Topic.STRATEGY_INVALID:
            return f"rejected {name}: {'; '.join((p.get('errors') or [])[:2])}"
        case Topic.STRATEGY_TESTED:
            oos = (p.get("metrics") or {}).get("oos") or {}
            return (
                f"tested {name}: OOS {oos.get('trades', 0)} trades, "
                f"Sharpe {oos.get('sharpe', 0)}, PF {oos.get('profit_factor', 0)}"
            )
        case Topic.STRATEGY_TUNED:
            return (
                f"tuned {name}: IS score {p.get('is_score_before')} -> {p.get('is_score_after')}"
                if p.get("tuned")
                else f"left {name} untuned ({p.get('skip_reason', 'no gain')})"
            )
        case Topic.ROBUSTNESS_REPORT:
            flags = (p.get("robustness") or {}).get("flags") or []
            return f"robustness on {name}: {len(flags)} concern(s)"
        case Topic.RISK_REPORT:
            r = p.get("risk_report") or {}
            return f"risk on {name}: plan {r.get('planning_loss_streak')} losses"
        case Topic.STRATEGY_JUDGED:
            return f"{name} -> {p.get('verdict')} (score {p.get('score')})"
        case Topic.PACKAGE_READY:
            return f"packaged {p.get('package_name')}"
        case Topic.DELIVERY_COMPLETED:
            return f"delivered {p.get('package_name')}"
        case Topic.DATA_UPDATED:
            return f"{p.get('asset')} {p.get('timeframe')}: {p.get('bars')} bars"
        case Topic.MEMORY_PRIORS:
            return f"priors updated for {p.get('scope')} from {p.get('samples')} results"
        case _:
            return name or msg.topic


__all__ = ["create_app"]
