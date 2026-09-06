"""Champion agent: depth instead of breadth.

Generating ten thousand throwaway strategies is motion, not progress. When an
idea shows any real promise, this agent stops the conveyor belt and works the idea
properly:

    1. sweep it across every asset and timeframe        (in-sample only)
    2. keep the cells where it genuinely performs
    3. tune each survivor hard - far more trials than the routine tuner spends
    4. push the results back into the normal pipeline for full judgement

Everything the campaign does happens on in-sample data, so the out-of-sample tail
is still untouched when the judge sees the result. And because a best-of-N winner
flatters itself, the campaign records the deflated score alongside the raw one.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pandas as pd

from ..backtest.engine import BacktestConfig, run_backtest
from ..backtest.metrics import robust_score
from ..config import factory_section
from ..data.providers import fetch_ohlcv
from ..ir.schema import StrategyIR
from ..ir.validate import audit_ir
from ..messages import Message, Topic
from ..store import Strategy, fingerprint_exists, in_db, save
from ..sweep import MIN_IS_TRADES, SweepResult, in_sample_slice, relocate, sweep
from .base import BaseAgent
from .data import frame_cache_get


class ChampionAgent(BaseAgent):
    name = "champion"
    role = "Champion Trainer"
    squad = "research"
    description = "Takes a promising strategy and works it hard: every market, every timeframe."
    subscribes = (Topic.STRATEGY_JUDGED,)
    handler_timeout = 3600.0

    async def setup(self) -> None:
        self.rng = random.Random()
        #: shapes already put through a campaign, so one idea is not re-swept
        #: every time one of its children is judged
        self._campaigned: set[str] = set()

    # ------------------------------------------------------------------ #

    async def handle(self, msg: Message) -> None:
        cfg = factory_section("champion")
        if not cfg.get("enabled", True):
            return

        ir = StrategyIR.model_validate(msg.payload["ir"])
        score = float(msg.payload.get("score") or 0.0)
        min_score = float(cfg.get("min_score", 0.35))
        max_gen = int(cfg.get("max_generation", 3))

        if score < min_score:
            return
        if ir.generation >= max_gen:
            self.log(
                f"'{ir.name}' scored {score:.3f} but is already generation "
                f"{ir.generation}; stopping the lineage here",
                msg=msg,
            )
            return
        shape = ir.shape_fingerprint()
        if shape in self._campaigned:
            return
        self._campaigned.add(shape)

        self.progress(f"campaign for '{ir.name}' (score {score:.2f})")
        self.log(
            f"promoting '{ir.name}' (score {score:.3f}): sweeping every market and timeframe",
            msg=msg,
        )

        config = BacktestConfig.from_factory_config()
        result: SweepResult = await asyncio.to_thread(
            sweep,
            ir,
            self._load_frame,
            config=config,
            max_cells=int(cfg.get("sweep_max_cells", 24)),
        )

        await self.emit(
            Topic.CHAMPION_CAMPAIGN,
            {
                "strategy_id": ir.id,
                "name": ir.name,
                "origin_market": f"{ir.asset} {ir.timeframe}",
                "sweep": result.as_dict(),
            },
            parent=msg,
            strategy_id=ir.id,
        )

        top = result.top(int(cfg.get("promote_top_k", 2)))
        if not top:
            self.log(
                f"'{ir.name}' did not clear {MIN_IS_TRADES} in-sample trades on any "
                f"of {result.tried} market/timeframe combinations - not promoting",
                msg=msg,
            )
            return

        usable_cells = sum(1 for c in result.cells if c.usable)
        best_cell = result.best
        self.log(
            f"sweep result for '{ir.name}': best {best_cell.asset} {best_cell.timeframe} "
            f"raw {result.raw_best_score:.3f} / deflated {result.deflated_best_score:.3f} "
            f"({usable_cells} usable of {result.tried} tried)",
            msg=msg,
        )

        for cell in top:
            candidate = relocate(ir, cell)
            self.progress(f"deep-tuning {candidate.name}")
            tuned, gain = await asyncio.to_thread(
                self._deep_tune, candidate, config, int(cfg.get("tune_trials", 120))
            )
            if not audit_ir(tuned).ok:
                continue
            if await in_db(fingerprint_exists, tuned.fingerprint()):
                continue

            row = await in_db(
                save,
                Strategy(
                    id=tuned.id,
                    project_id=msg.project_id,
                    name=tuned.name,
                    style=tuned.style,
                    asset=tuned.asset,
                    timeframe=tuned.timeframe,
                    generation=tuned.generation,
                    parents=tuned.parents,
                    origin="promoted",
                    ir=tuned.model_dump(mode="json"),
                    fingerprint=tuned.fingerprint(),
                    status="created",
                ),
            )
            self.log(
                f"promoted '{tuned.name}' to the pipeline "
                f"(in-sample score {cell.score:.3f} -> {gain:.3f} after deep tuning)",
                msg=msg,
            )
            await self.emit(
                Topic.STRATEGY_CREATED,
                {
                    "strategy_id": row.id,
                    "name": tuned.name,
                    "asset": tuned.asset,
                    "timeframe": tuned.timeframe,
                    "style": tuned.style,
                    "origin": "promoted",
                    "generation": tuned.generation,
                    "ir": tuned.model_dump(mode="json"),
                    "summary": tuned.describe(),
                    "promoted_from": ir.id,
                    "sweep_cell": cell.as_dict(),
                },
                parent=msg,
                strategy_id=row.id,
            )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_frame(asset: str, timeframe: str) -> pd.DataFrame:
        cached = frame_cache_get(asset, timeframe)
        if cached is not None:
            return cached
        return fetch_ohlcv(asset, timeframe)

    def _deep_tune(
        self, ir: StrategyIR, config: BacktestConfig, trials: int
    ) -> tuple[StrategyIR, float]:
        """Random search on in-sample data only, with a far bigger budget than the
        routine tuner spends. Returns the best variant and its score."""
        if not ir.param_space:
            return ir, 0.0
        frame = in_sample_slice(self._load_frame(ir.asset, ir.timeframe), config.oos_fraction)
        if len(frame) < 400:
            return ir, 0.0

        def score_of(candidate: StrategyIR) -> float:
            run = run_backtest(candidate, frame, config=config, kind="champion_tune")
            if not run.ok or int(run.metrics.get("trades", 0)) < MIN_IS_TRADES:
                return 0.0
            return robust_score(
                run.metrics,
                min_trades=MIN_IS_TRADES,
                n_params=len(candidate.param_space),
                complexity_penalty=float(factory_section("tuner").get("complexity_penalty", 0.02)),
            )

        best_ir, best = ir, score_of(ir)
        for i in range(trials):
            values: dict[str, Any] = {}
            for spec in ir.param_space:
                raw = self.rng.uniform(spec.low, spec.high)
                values[spec.path] = float(round(raw)) if spec.is_int else round(raw, 3)
            trial = ir.with_params(values)
            s = score_of(trial)
            if s > best:
                best, best_ir = s, trial
            if (i + 1) % 30 == 0:
                self.progress(f"deep-tuning {ir.name}: {i + 1}/{trials}, best {best:.3f}")

        if best_ir is not ir:
            best_ir.origin = "promoted"
            best_ir.notes = f"{ir.notes} | deep-tuned over {trials} trials"
        return best_ir, best


__all__ = ["ChampionAgent"]
