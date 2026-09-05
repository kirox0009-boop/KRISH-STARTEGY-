"""Research squad: where new strategies are conceived.

Flow: ``cycle.start`` -> researcher (hypothesis + project) -> quant_analyst
(feasibility + recipe + priors) -> architect (Strategy IR).

The explore/exploit split lives in the researcher: it decides whether this idea
should be brand new, a mutation of something that already scored well, or a
crossover of two winners. That single decision is what keeps the factory from
either stagnating on four strategies or forever wandering at random.
"""

from __future__ import annotations

import random
from typing import Any

from .. import llm
from ..bus import BusTimeout
from ..genome import RECIPES, StrategyFactory
from ..ir.schema import StrategyIR
from ..messages import Message, Topic
from ..store import (
    Project,
    Strategy,
    elite_strategies,
    fingerprint_exists,
    get_strategy,
    in_db,
    priors_for,
    save,
    update_project,
)
from .base import BaseAgent

EXPLORE_FRESH = 0.5  # brand-new recipes
EXPLORE_MUTATE = 0.35  # refine a winner
EXPLORE_CROSS = 0.15  # breed two winners


class ResearcherAgent(BaseAgent):
    name = "researcher"
    role = "Researcher"
    squad = "research"
    description = "Turns market context and past results into testable hypotheses."
    subscribes = (Topic.CYCLE_START, Topic.MEMORY_PRIORS)

    async def setup(self) -> None:
        self.rng = random.Random()
        self._priors: dict[str, dict[str, Any]] = {}

    async def handle(self, msg: Message) -> None:
        if msg.topic == Topic.MEMORY_PRIORS:
            scope = str(msg.payload.get("scope", "*"))
            self._priors[scope] = dict(msg.payload.get("priors") or {})
            self.log(f"priors updated for {scope}")
            return

        asset = str(msg.payload.get("asset", "")).upper()
        timeframe = str(msg.payload.get("timeframe", "H1")).upper()
        count = int(msg.payload.get("count", 1))
        scope = f"{asset}:{timeframe}"
        priors = self._priors.get(scope) or await in_db(priors_for, scope)

        elites = await in_db(elite_strategies, asset, 20)
        for _ in range(count):
            mode, parents = self._choose_mode(elites)
            hypothesis = await self._write_hypothesis(asset, timeframe, mode, parents, priors)

            project = await in_db(
                save,
                Project(
                    title=hypothesis["title"],
                    asset=asset,
                    timeframe=timeframe,
                    stage="hypothesis",
                    status="running",
                    owner_agent=self.name,
                    hypothesis=hypothesis["text"],
                    meta={
                        "mode": mode,
                        "parents": [p.id for p in parents],
                        "cycle": msg.payload.get("cycle_id"),
                    },
                ),
            )
            self.progress(f"hypothesis for {asset} {timeframe} ({mode})")
            await self.emit(
                Topic.HYPOTHESIS_CREATED,
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "mode": mode,
                    "parents": [p.id for p in parents],
                    "title": hypothesis["title"],
                    "hypothesis": hypothesis["text"],
                    "priors": priors,
                },
                parent=msg,
                project_id=project.id,
            )

    def _choose_mode(self, elites: list[Strategy]) -> tuple[str, list[Strategy]]:
        """Explore vs exploit. With no elites yet, everything is exploration."""
        if not elites:
            return "fresh", []
        roll = self.rng.random()
        if roll < EXPLORE_FRESH:
            return "fresh", []
        if roll < EXPLORE_FRESH + EXPLORE_MUTATE or len(elites) < 2:
            return "mutation", [self._pick_elite(elites)]
        a = self._pick_elite(elites)
        b = self._pick_elite([e for e in elites if e.id != a.id])
        return "crossover", [a, b]

    def _pick_elite(self, elites: list[Strategy]) -> Strategy:
        """Rank-weighted choice: the best is favoured but never guaranteed."""
        weights = [1.0 / (i + 1.5) for i in range(len(elites))]
        return self.rng.choices(elites, weights=weights, k=1)[0]

    async def _write_hypothesis(
        self,
        asset: str,
        timeframe: str,
        mode: str,
        parents: list[Strategy],
        priors: dict[str, Any],
    ) -> dict[str, str]:
        if mode == "mutation" and parents:
            title = f"Refine '{parents[0].name}' on {asset} {timeframe}"
            text = (
                f"'{parents[0].name}' ({parents[0].style}) already scores well on {asset} "
                f"{timeframe}. Perturbing its parameters, exits and filters should find a more "
                "robust plateau rather than a sharper peak."
            )
        elif mode == "crossover" and len(parents) >= 2:
            title = f"Cross '{parents[0].name}' with '{parents[1].name}'"
            text = (
                f"'{parents[0].name}' ({parents[0].style}) and '{parents[1].name}' "
                f"({parents[1].style}) succeed for different reasons on {asset}. Combining one's "
                "entry timing with the other's exit and risk handling may keep both edges."
            )
        else:
            theme = self._fresh_theme(asset, timeframe, priors)
            title = theme["title"]
            text = theme["text"]

        # An LLM, when configured, only rewrites the prose — never the decision.
        if llm.available():
            polished = await llm.complete(
                f"Rewrite this trading research hypothesis in 2 precise sentences. "
                f"Keep it falsifiable, no hype, no advice:\n\n{text}",
                system="You are a quantitative researcher writing terse internal notes.",
                max_tokens=200,
                temperature=0.6,
            )
            if polished:
                text = polished
        return {"title": title, "text": text}

    def _fresh_theme(self, asset: str, timeframe: str, priors: dict[str, Any]) -> dict[str, str]:
        best_family = priors.get("best_style", {}).get("style") if priors else None
        themes = [
            (
                f"Volatility expansion on {asset}",
                f"On {asset} {timeframe}, moves that begin from compressed volatility travel "
                "further than moves that begin from already-elevated volatility.",
            ),
            (
                f"Trend persistence on {asset}",
                f"{asset} {timeframe} trends persist long enough that a trailing exit captures "
                "more than a fixed target, despite a lower win rate.",
            ),
            (
                f"Overshoot reversion on {asset}",
                f"Short-term {asset} {timeframe} overshoots beyond a statistical band revert "
                "before the prevailing trend resumes.",
            ),
            (
                f"Session structure on {asset}",
                f"{asset} behaves differently by session; restricting entries to the most liquid "
                "hours should improve expectancy per trade.",
            ),
            (
                f"Momentum confirmation on {asset}",
                f"Requiring momentum confirmation before entry on {asset} {timeframe} filters out "
                "false starts that dominate raw crossover signals.",
            ),
        ]
        if best_family:
            themes.insert(
                0,
                (
                    f"Extend the {best_family} edge on {asset}",
                    f"{best_family} strategies have been the strongest family on {asset} "
                    f"{timeframe} so far; a differently-structured member of that family should "
                    "share the edge without sharing its specific weaknesses.",
                ),
            )
        title, text = self.rng.choice(themes)
        return {"title": title, "text": text}


class QuantAnalystAgent(BaseAgent):
    name = "quant_analyst"
    role = "Quant Analyst"
    squad = "research"
    description = "Checks feasibility, picks the structural recipe and the parameter priors."
    subscribes = (Topic.HYPOTHESIS_CREATED,)

    async def setup(self) -> None:
        self.rng = random.Random()

    async def handle(self, msg: Message) -> None:
        asset = str(msg.payload["asset"])
        timeframe = str(msg.payload["timeframe"])

        # Feasibility first: no point designing anything if the data is not there.
        try:
            info = await self.ask(
                Topic.DATA_REQUEST,
                {"asset": asset, "timeframe": timeframe},
                reply_topic=Topic.DATA_RESPONSE,
                timeout=300.0,
                parent=msg,
            )
        except BusTimeout:
            await self._reject(msg, "market_data did not answer in time")
            return
        except RuntimeError as exc:
            await self._reject(msg, str(exc))
            return

        bars = int(info.get("bars", 0))
        if bars < 800:
            await self._reject(msg, f"only {bars} bars available for {asset} {timeframe}")
            return

        priors = dict(msg.payload.get("priors") or {})
        mode = str(msg.payload.get("mode", "fresh"))
        recipe = self._pick_recipe(mode, priors)

        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="design")

        await self.emit(
            Topic.FEATURE_SPEC_CREATED,
            {
                "asset": asset,
                "timeframe": timeframe,
                "mode": mode,
                "parents": msg.payload.get("parents") or [],
                "recipe": recipe,
                "priors": priors,
                "hypothesis": msg.payload.get("hypothesis", ""),
                "data": {
                    "bars": bars,
                    "start": info.get("start"),
                    "end": info.get("end"),
                    "synthetic": info.get("synthetic", False),
                },
            },
            parent=msg,
        )

    def _pick_recipe(self, mode: str, priors: dict[str, Any]) -> str | None:
        if mode != "fresh":
            return None
        weights = priors.get("recipe_weights") or {}
        names = list(RECIPES)
        if not weights:
            return self.rng.choice(names)
        return self.rng.choices(
            names, weights=[max(0.15, float(weights.get(n, 1.0))) for n in names], k=1
        )[0]

    async def _reject(self, msg: Message, reason: str) -> None:
        self.log(f"hypothesis rejected early: {reason}", level="warn", msg=msg)
        if msg.project_id:
            await in_db(
                update_project,
                msg.project_id,
                stage="rejected_early",
                status="stopped",
                meta={"reason": reason},
            )


class ArchitectAgent(BaseAgent):
    name = "architect"
    role = "Strategy Architect"
    squad = "research"
    description = "Invents the Strategy IR: fresh recipes, mutations and crossovers."
    subscribes = (Topic.FEATURE_SPEC_CREATED, Topic.EVOLUTION_REQUEST)

    #: how many times to retry when the dice produce an already-tested strategy
    DEDUPE_ATTEMPTS = 6

    async def setup(self) -> None:
        self.factory = StrategyFactory()

    async def handle(self, msg: Message) -> None:
        asset = str(msg.payload["asset"])
        timeframe = str(msg.payload["timeframe"])
        mode = str(msg.payload.get("mode", "fresh"))
        priors = dict(msg.payload.get("priors") or {})
        parent_ids = list(msg.payload.get("parents") or [])
        recipe = msg.payload.get("recipe")

        ir: StrategyIR | None = None
        for attempt in range(self.DEDUPE_ATTEMPTS):
            candidate = await self._build(asset, timeframe, mode, parent_ids, recipe, priors)
            if candidate is None:
                break
            exists = await in_db(fingerprint_exists, candidate.fingerprint())
            if not exists:
                ir = candidate
                break
            self.progress(f"duplicate on attempt {attempt + 1}, re-rolling")
            mode, recipe = "fresh", None  # break out of a stuck lineage

        if ir is None:
            self.log(
                f"could not produce a novel strategy for {asset} {timeframe} after "
                f"{self.DEDUPE_ATTEMPTS} attempts",
                level="warn",
                msg=msg,
            )
            if msg.project_id:
                await in_db(update_project, msg.project_id, stage="duplicate", status="stopped")
            return

        if msg.payload.get("hypothesis"):
            ir.hypothesis = str(msg.payload["hypothesis"])

        row = await in_db(
            save,
            Strategy(
                id=ir.id,
                project_id=msg.project_id,
                name=ir.name,
                style=ir.style,
                asset=ir.asset,
                timeframe=ir.timeframe,
                generation=ir.generation,
                parents=ir.parents,
                origin=ir.origin,
                ir=ir.model_dump(mode="json"),
                fingerprint=ir.fingerprint(),
                status="created",
            ),
        )
        if msg.project_id:
            await in_db(
                update_project,
                msg.project_id,
                stage="created",
                strategy_id=row.id,
                title=f"{ir.name} · {ir.asset} {ir.timeframe}",
            )

        self.log(f"designed '{ir.name}' ({ir.origin}, gen {ir.generation})", msg=msg)
        await self.emit(
            Topic.STRATEGY_CREATED,
            {
                "strategy_id": ir.id,
                "name": ir.name,
                "asset": ir.asset,
                "timeframe": ir.timeframe,
                "style": ir.style,
                "origin": ir.origin,
                "generation": ir.generation,
                "ir": ir.model_dump(mode="json"),
                "summary": ir.describe(),
            },
            parent=msg,
            strategy_id=ir.id,
        )

    async def _build(
        self,
        asset: str,
        timeframe: str,
        mode: str,
        parent_ids: list[str],
        recipe: str | None,
        priors: dict[str, Any],
    ) -> StrategyIR | None:
        if mode == "mutation" and parent_ids:
            parent = await self._load_ir(parent_ids[0])
            if parent is not None:
                return self.factory.mutate(parent, priors=priors)
        elif mode == "crossover" and len(parent_ids) >= 2:
            a = await self._load_ir(parent_ids[0])
            b = await self._load_ir(parent_ids[1])
            if a is not None and b is not None:
                return self.factory.crossover(a, b)
        return self.factory.fresh(asset, timeframe, recipe=recipe, priors=priors)

    async def _load_ir(self, strategy_id: str) -> StrategyIR | None:
        row = await in_db(get_strategy, strategy_id)
        if row is None:
            return None
        try:
            return StrategyIR.model_validate(row.ir)
        except Exception as exc:
            self.log(f"stored IR for {strategy_id} is unreadable: {exc}", level="warn")
            return None


__all__ = ["ArchitectAgent", "QuantAnalystAgent", "ResearcherAgent"]
