"""Validation squad: tester, tuner, robustness, risk, judge.

This is where most strategies die, and that is the point. The factory's value is
not how many strategies it invents — it is how ruthlessly it throws them away.

Order matters:
  tester      honest backtest, IS/OOS split
  tuner       optimises on IS only; never sees the OOS tail
  robustness  walk-forward, cost stress, Monte Carlo, parameter sensitivity
  risk        survivability: streaks, ruin, sizing
  judge       one verdict, with reasons, against configurable thresholds
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..assets import universe
from ..backtest.engine import BacktestConfig, run_backtest, split_backtest, walk_forward
from ..backtest.metrics import robust_score
from ..bus import BusTimeout
from ..config import factory_section
from ..data.providers import fetch_ohlcv
from ..ir.schema import StrategyIR
from ..messages import Message, Topic
from ..store import (
    BacktestRun,
    Verdict,
    in_db,
    purge_rejected_strategy,
    save,
    update_project,
    update_strategy,
)
from .base import BaseAgent
from .data import frame_cache_get


class _NeedsPrices:
    """Shared helper: get a price frame via the market_data agent."""

    async def get_frame(self: BaseAgent, msg: Message, asset: str, timeframe: str) -> pd.DataFrame:
        # Ask market_data first. That both guarantees freshness and records the
        # dependency on the bus, which is what the dashboard graph draws.
        try:
            await self.ask(
                Topic.DATA_REQUEST,
                {"asset": asset, "timeframe": timeframe},
                reply_topic=Topic.DATA_RESPONSE,
                timeout=300.0,
                parent=msg,
            )
        except (BusTimeout, RuntimeError) as exc:
            self.log(f"market_data unavailable ({exc}); falling back to local fetch", level="warn")

        frame = frame_cache_get(asset, timeframe)
        if frame is None:
            # Running in a separate process from market_data: read the shared cache.
            frame = await asyncio.to_thread(fetch_ohlcv, asset, timeframe)
        return frame


# --------------------------------------------------------------------------- #
# tester
# --------------------------------------------------------------------------- #


class TesterAgent(BaseAgent, _NeedsPrices):
    name = "tester"
    role = "Tester"
    squad = "validation"
    description = "Runs honest backtests with real costs and an untouched out-of-sample tail."
    subscribes = (Topic.STRATEGY_BUILT, Topic.BACKTEST_REQUEST)
    handler_timeout = 1800.0

    async def handle(self, msg: Message) -> None:
        if msg.topic == Topic.BACKTEST_REQUEST:
            await self._serve_request(msg)
            return

        ir = StrategyIR.model_validate(msg.payload["ir"])
        frame = await self.get_frame(msg, ir.asset, ir.timeframe)
        self.progress(f"backtesting {ir.name} on {len(frame)} bars")

        config = BacktestConfig.from_factory_config()
        result = await asyncio.to_thread(split_backtest, ir, frame, config=config)

        if result.get("error"):
            await self._failed(msg, ir, str(result["error"]))
            return

        oos = result["oos"]["metrics"]
        is_m = result["is"]["metrics"]
        full = result["full"]["metrics"]

        run_ids: dict[str, str] = {}
        for kind in ("full", "is", "oos"):
            payload = result[kind]
            row = await in_db(
                save,
                BacktestRun(
                    strategy_id=ir.id,
                    project_id=msg.project_id,
                    kind=kind,
                    asset=ir.asset,
                    timeframe=ir.timeframe,
                    params=ir.current_params(),
                    metrics=payload["metrics"],
                    equity=payload["equity_curve"],
                    trades=payload["trades"][-200:],
                    bars=int(payload["metrics"].get("bars", 0)),
                    duration_ms=int(payload.get("duration_ms", 0)),
                ),
            )
            run_ids[kind] = row.id

        await in_db(update_strategy, ir.id, status="tested")
        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="tested")

        self.log(
            f"'{ir.name}': OOS trades={oos.get('trades')} sharpe={oos.get('sharpe')} "
            f"pf={oos.get('profit_factor')} dd={oos.get('max_drawdown_pct')}%",
            msg=msg,
        )
        await self.emit(
            Topic.STRATEGY_TESTED,
            {
                "strategy_id": ir.id,
                "name": ir.name,
                "asset": ir.asset,
                "timeframe": ir.timeframe,
                "ir": ir.model_dump(mode="json"),
                "run_ids": run_ids,
                "metrics": {"full": full, "is": is_m, "oos": oos},
                "robust_score": result["robust_score"],
                "sharpe_retention": result["sharpe_retention"],
                "bars": len(frame),
            },
            parent=msg,
            strategy_id=ir.id,
        )

    async def _serve_request(self, msg: Message) -> None:
        """Ad-hoc backtest for the tuner, the robustness agent, or the UI."""
        try:
            ir = StrategyIR.model_validate(msg.payload["ir"])
            frame = await self.get_frame(msg, ir.asset, ir.timeframe)
            cfg = BacktestConfig.from_factory_config()
            if msg.payload.get("cost_multiplier"):
                cfg.cost_multiplier = float(msg.payload["cost_multiplier"])
            result = await asyncio.to_thread(
                run_backtest, ir, frame, config=cfg, kind=str(msg.payload.get("kind", "adhoc"))
            )
            await self.bus.publish(msg.responds_to(self.name, result.as_dict()))
        except Exception as exc:
            await self.bus.publish(msg.responds_to(self.name, {"error": str(exc)}))

    async def _failed(self, msg: Message, ir: StrategyIR, error: str) -> None:
        self.log(f"backtest failed for '{ir.name}': {error}", level="warn", msg=msg)
        await in_db(update_strategy, ir.id, status="test_failed")
        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="test_failed", status="stopped")
        await self.emit(
            Topic.STRATEGY_TEST_FAILED,
            {
                "strategy_id": ir.id,
                "name": ir.name,
                "error": error,
                "ir": ir.model_dump(mode="json"),
            },
            parent=msg,
            strategy_id=ir.id,
        )


# --------------------------------------------------------------------------- #
# tuner
# --------------------------------------------------------------------------- #


class TunerAgent(BaseAgent, _NeedsPrices):
    name = "tuner"
    role = "Tuner"
    squad = "validation"
    description = "Optimises parameters on in-sample data only, and reports the stability region."
    subscribes = (Topic.STRATEGY_TESTED, Topic.TUNE_REQUEST)
    handler_timeout = 3600.0

    async def setup(self) -> None:
        self.rng = random.Random()

    async def handle(self, msg: Message) -> None:
        ir = StrategyIR.model_validate(msg.payload["ir"])
        cfg = factory_section("tuner")
        trials = int(cfg.get("trials", 40))
        penalty = float(cfg.get("complexity_penalty", 0.02))

        baseline_oos = dict(msg.payload.get("metrics", {}).get("oos") or {})
        baseline_is = dict(msg.payload.get("metrics", {}).get("is") or {})

        if not ir.param_space:
            await self._pass_through(msg, ir, "no tunable parameters", baseline_oos)
            return

        # Do not spend CPU tuning something with no signal at all in-sample.
        if int(baseline_is.get("trades", 0)) < 10:
            await self._pass_through(msg, ir, "too few in-sample trades to tune", baseline_oos)
            return

        frame = await self.get_frame(msg, ir.asset, ir.timeframe)
        oos_fraction = float(factory_section("backtest").get("oos_fraction", 0.3))
        split_idx = int(len(frame) * (1 - oos_fraction))
        is_frame = frame.iloc[:split_idx]  # the tuner never sees past this
        config = BacktestConfig.from_factory_config()

        base_score = await self._score(ir, is_frame, config, penalty)
        best_score, best_params = base_score, ir.current_params()
        history: list[dict[str, Any]] = []

        for trial in range(trials):
            candidate_params = self._sample(ir)
            candidate = ir.with_params(candidate_params)
            score = await self._score(candidate, is_frame, config, penalty)
            history.append({"trial": trial + 1, "score": score, "params": candidate_params})
            if score > best_score:
                best_score, best_params = score, candidate_params
            if (trial + 1) % 10 == 0:
                self.progress(
                    f"tuning {ir.name}: {trial + 1}/{trials} trials, best {best_score:.3f}",
                    eta_seconds=None,
                )

        improved = best_score > base_score * 1.05
        tuned_ir = ir.with_params(best_params) if improved else ir
        if improved:
            tuned_ir.origin = "tuned"
            tuned_ir.notes = f"{ir.notes} | tuned: IS score {base_score:.3f} -> {best_score:.3f}"

        stability = self._stability(history, best_score)

        # Final verdict input is always a fresh split on the *unseen* tail.
        result = await asyncio.to_thread(split_backtest, tuned_ir, frame, config=config)
        oos = (result.get("oos") or {}).get("metrics") or {}

        if improved:
            await in_db(
                update_strategy,
                ir.id,
                ir=tuned_ir.model_dump(mode="json"),
                fingerprint=tuned_ir.fingerprint(),
                status="tuned",
            )
            await in_db(
                save,
                BacktestRun(
                    strategy_id=ir.id,
                    project_id=msg.project_id,
                    kind="tuned_oos",
                    asset=ir.asset,
                    timeframe=ir.timeframe,
                    params=best_params,
                    metrics=oos,
                    equity=(result.get("oos") or {}).get("equity_curve") or [],
                    trades=((result.get("oos") or {}).get("trades") or [])[-200:],
                ),
            )
        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="tuned")

        self.log(
            f"'{ir.name}': {'tuned' if improved else 'left alone'} "
            f"(IS {base_score:.3f}->{best_score:.3f}, stability {stability['plateau_ratio']})",
            msg=msg,
        )
        await self.emit(
            Topic.STRATEGY_TUNED,
            {
                "strategy_id": ir.id,
                "name": tuned_ir.name,
                "asset": ir.asset,
                "timeframe": ir.timeframe,
                "ir": tuned_ir.model_dump(mode="json"),
                "tuned": improved,
                "trials": trials,
                "is_score_before": base_score,
                "is_score_after": best_score,
                "best_params": best_params,
                "stability": stability,
                "metrics": {
                    "full": (result.get("full") or {}).get("metrics") or {},
                    "is": (result.get("is") or {}).get("metrics") or {},
                    "oos": oos,
                },
                "robust_score": result.get("robust_score", 0.0),
                "sharpe_retention": result.get("sharpe_retention", 0.0),
            },
            parent=msg,
            strategy_id=ir.id,
        )

    def _sample(self, ir: StrategyIR) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for spec in ir.param_space:
            raw = self.rng.uniform(spec.low, spec.high)
            values[spec.path] = float(round(raw)) if spec.is_int else round(raw, 3)
        return values

    async def _score(
        self, ir: StrategyIR, frame: pd.DataFrame, config: BacktestConfig, penalty: float
    ) -> float:
        result = await asyncio.to_thread(run_backtest, ir, frame, config=config, kind="tune")
        if not result.ok:
            return 0.0
        return robust_score(
            result.metrics,
            min_trades=config.min_trades,
            n_params=len(ir.param_space),
            complexity_penalty=penalty,
        )

    @staticmethod
    def _stability(history: list[dict[str, Any]], best: float) -> dict[str, Any]:
        """A good optimum is a plateau. A lone spike is curve fitting."""
        scores = [h["score"] for h in history if h["score"] > 0]
        if not scores or best <= 0:
            return {"plateau_ratio": 0.0, "positive_trials": 0, "trials": len(history)}
        near_best = sum(1 for s in scores if s >= best * 0.8)
        # Ratio is measured against trials that had *any* edge, not against all
        # trials: random search wastes most draws on dead parameter regions, and
        # counting those would flag every strategy as fragile.
        return {
            "plateau_ratio": round(near_best / max(len(scores), 1), 3),
            "positive_trials": len(scores),
            "trials": len(history),
            "median_score": round(float(np.median(scores)), 4),
            "best_score": round(best, 4),
        }

    async def _pass_through(
        self, msg: Message, ir: StrategyIR, reason: str, oos: dict[str, Any]
    ) -> None:
        self.log(f"'{ir.name}': skipping tuning ({reason})", msg=msg)
        await self.emit(
            Topic.STRATEGY_TUNED,
            {
                **{k: v for k, v in msg.payload.items() if k != "ir"},
                "ir": ir.model_dump(mode="json"),
                "tuned": False,
                "skip_reason": reason,
                "stability": {"plateau_ratio": 0.0, "trials": 0},
            },
            parent=msg,
            strategy_id=ir.id,
        )


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #


class RobustnessAgent(BaseAgent, _NeedsPrices):
    name = "robustness"
    role = "Robustness Analyst"
    squad = "validation"
    description = "Walk-forward, cost stress, Monte Carlo and parameter sensitivity."
    subscribes = (Topic.STRATEGY_TUNED,)
    handler_timeout = 3600.0

    async def setup(self) -> None:
        self.rng = random.Random()

    async def handle(self, msg: Message) -> None:
        ir = StrategyIR.model_validate(msg.payload["ir"])
        frame = await self.get_frame(msg, ir.asset, ir.timeframe)
        config = BacktestConfig.from_factory_config()
        folds = int(factory_section("tuner").get("walk_forward_folds", 4))

        self.progress(f"walk-forward on {ir.name}")
        wf = await asyncio.to_thread(walk_forward, ir, frame, folds=folds, config=config)

        self.progress(f"cost stress on {ir.name}")
        stress = await self._cost_stress(ir, frame, config)

        self.progress(f"monte carlo on {ir.name}")
        mc = await self._monte_carlo(ir, frame, config)

        self.progress(f"parameter sensitivity on {ir.name}")
        sensitivity = await self._sensitivity(ir, frame, config)

        flags = self._flags(wf, stress, mc, sensitivity, msg.payload.get("stability") or {})
        overfit = bool(flags)

        self.log(
            f"'{ir.name}': wf consistency {wf.get('consistency')}, "
            f"mc p(profit) {mc.get('prob_profitable')}, flags={flags or 'none'}",
            msg=msg,
        )
        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="robustness")

        await self.emit(
            Topic.ROBUSTNESS_REPORT,
            {
                **{k: v for k, v in msg.payload.items() if k not in {"stability"}},
                "stability": msg.payload.get("stability") or {},
                "robustness": {
                    "walk_forward": wf,
                    "cost_stress": stress,
                    "monte_carlo": mc,
                    "sensitivity": sensitivity,
                    "flags": flags,
                    "overfit_suspected": overfit,
                },
            },
            parent=msg,
            strategy_id=ir.id,
        )

    async def _cost_stress(
        self, ir: StrategyIR, frame: pd.DataFrame, config: BacktestConfig
    ) -> dict[str, Any]:
        """Does the edge survive worse spreads than the broker quotes today?"""
        out: dict[str, Any] = {}
        for label, multiplier in (("1.0x", 1.0), ("1.5x", 1.5), ("2.0x", 2.0)):
            cfg = replace(config, cost_multiplier=multiplier)
            result = await asyncio.to_thread(
                run_backtest, ir, frame, config=cfg, kind=f"cost{label}"
            )
            out[label] = {
                "sharpe": result.metrics.get("sharpe", 0.0),
                "profit_factor": result.metrics.get("profit_factor", 0.0),
                "return_pct": result.metrics.get("total_return_pct", 0.0),
                "trades": result.metrics.get("trades", 0),
            }
        survives = float(out["2.0x"]["profit_factor"] or 0) > 1.0
        out["survives_double_costs"] = survives
        return out

    async def _monte_carlo(
        self, ir: StrategyIR, frame: pd.DataFrame, config: BacktestConfig, runs: int = 400
    ) -> dict[str, Any]:
        """Bootstrap the trade sequence: how much of the result was ordering luck?"""
        result = await asyncio.to_thread(run_backtest, ir, frame, config=config, kind="mc_base")
        rs = [float(t.get("r", 0.0)) for t in result.trades]
        if len(rs) < 10:
            return {"runs": 0, "note": "too few trades for a meaningful Monte Carlo"}

        arr = np.asarray(rs, dtype=float)
        rng = np.random.default_rng(12345)
        finals = np.empty(runs)
        max_dds = np.empty(runs)
        for i in range(runs):
            shuffled = rng.choice(arr, size=arr.size, replace=True)
            curve = np.cumsum(shuffled)
            finals[i] = curve[-1]
            peaks = np.maximum.accumulate(np.concatenate([[0.0], curve]))
            max_dds[i] = float(np.max(peaks - np.concatenate([[0.0], curve])))
        return {
            "runs": runs,
            "trades": int(arr.size),
            "mean_total_r": round(float(finals.mean()), 2),
            "prob_profitable": round(float((finals > 0).mean()), 3),
            "r_5th_percentile": round(float(np.percentile(finals, 5)), 2),
            "median_max_dd_r": round(float(np.median(max_dds)), 2),
            "worst_max_dd_r": round(float(np.max(max_dds)), 2),
        }

    async def _sensitivity(
        self, ir: StrategyIR, frame: pd.DataFrame, config: BacktestConfig
    ) -> dict[str, Any]:
        """Perturb each parameter +-15%: a real edge degrades gently, not off a cliff."""
        if not ir.param_space:
            return {"tested": 0, "note": "no tunable parameters"}
        base = await asyncio.to_thread(run_backtest, ir, frame, config=config, kind="sens_base")
        base_score = robust_score(base.metrics, min_trades=config.min_trades)
        if base_score <= 0:
            return {"tested": 0, "base_score": 0.0, "note": "no baseline edge to perturb"}

        rows: list[dict[str, Any]] = []
        for spec in ir.param_space[:4]:  # keep the CPU budget sane
            current = float(ir.get_param(spec.path))
            scores = []
            for factor in (0.85, 1.15):
                value = current * factor
                value = float(round(value)) if spec.is_int else round(value, 3)
                value = min(max(value, spec.low), spec.high)
                variant = ir.with_params({spec.path: value})
                res = await asyncio.to_thread(
                    run_backtest, variant, frame, config=config, kind="sens"
                )
                scores.append(robust_score(res.metrics, min_trades=config.min_trades))
            retention = min(scores) / base_score if base_score > 0 else 0.0
            rows.append(
                {
                    "param": spec.label or spec.path,
                    "base": round(current, 3),
                    "scores": [round(s, 4) for s in scores],
                    "worst_retention": round(retention, 3),
                }
            )
        worst = min((r["worst_retention"] for r in rows), default=0.0)
        return {
            "tested": len(rows),
            "base_score": round(base_score, 4),
            "params": rows,
            "worst_retention": round(worst, 3),
            "is_plateau": worst >= 0.6,
        }

    @staticmethod
    def _flags(
        wf: dict[str, Any],
        stress: dict[str, Any],
        mc: dict[str, Any],
        sensitivity: dict[str, Any],
        stability: dict[str, Any],
    ) -> list[str]:
        flags: list[str] = []
        if wf.get("folds") and float(wf.get("consistency", 0)) < 0.5:
            flags.append(
                f"only {wf.get('profitable_folds')}/{wf.get('fold_count')} walk-forward folds "
                "were profitable"
            )
        if wf.get("worst_sharpe") is not None and float(wf.get("worst_sharpe", 0)) < -1.0:
            flags.append(f"one walk-forward era was badly negative (Sharpe {wf['worst_sharpe']})")
        if stress and not stress.get("survives_double_costs", False):
            flags.append("edge disappears at 2x the configured spread/slippage")
        if mc.get("runs") and float(mc.get("prob_profitable", 0)) < 0.7:
            flags.append(
                f"Monte Carlo says only {float(mc['prob_profitable']) * 100:.0f}% of trade "
                "orderings finish profitable"
            )
        if sensitivity.get("tested") and not sensitivity.get("is_plateau", False):
            flags.append(
                f"parameters are a spike, not a plateau (worst retention "
                f"{sensitivity.get('worst_retention')})"
            )
        if (
            int(stability.get("positive_trials", 0)) >= 5
            and float(stability.get("plateau_ratio", 1.0)) < 0.15
        ):
            flags.append("tuning found a single lucky combination rather than a stable region")
        return flags


# --------------------------------------------------------------------------- #
# risk
# --------------------------------------------------------------------------- #


class RiskAgent(BaseAgent):
    name = "risk"
    role = "Risk Manager"
    squad = "validation"
    description = "Survivability: streaks, ruin probability, sizing and required capital."
    subscribes = (Topic.ROBUSTNESS_REPORT,)

    async def handle(self, msg: Message) -> None:
        ir = StrategyIR.model_validate(msg.payload["ir"])
        oos = dict(msg.payload.get("metrics", {}).get("oos") or {})
        mc = dict(msg.payload.get("robustness", {}).get("monte_carlo") or {})
        asset = universe().get(ir.asset)

        win_rate = float(oos.get("win_rate_pct", 0.0)) / 100.0
        expectancy_r = float(oos.get("expectancy_r", 0.0))
        trades = int(oos.get("trades", 0))
        risk_pct = ir.risk.risk_per_trade_pct

        # Worst streak we should plan for, from observed and theoretical angles.
        observed_streak = int(oos.get("max_consecutive_losses", 0))
        modelled_streak = self._expected_streak(win_rate, trades)
        planning_streak = max(observed_streak, modelled_streak)
        streak_drawdown_pct = round(planning_streak * risk_pct, 2)

        kelly = self._kelly(oos)
        recommended_risk = round(min(max(kelly * 0.25, 0.25), 2.0), 2) if kelly > 0 else 0.5

        report = {
            "risk_per_trade_pct": risk_pct,
            "recommended_risk_per_trade_pct": recommended_risk,
            "kelly_fraction_pct": round(kelly, 2),
            "observed_loss_streak": observed_streak,
            "modelled_loss_streak": modelled_streak,
            "planning_loss_streak": planning_streak,
            "streak_drawdown_pct": streak_drawdown_pct,
            "observed_max_drawdown_pct": float(oos.get("max_drawdown_pct", 0.0)),
            "monte_carlo_worst_dd_r": mc.get("worst_max_dd_r"),
            "expectancy_r": expectancy_r,
            "exposure_pct": float(oos.get("exposure_pct", 0.0)),
            "stop_style": ir.risk.label(),
            "asset_cost_round_trip_points": asset.cost.round_trip_points(),
        }

        # `warnings` question whether the strategy is survivable at all - the judge
        # treats them as quality evidence. `advisories` are configuration tips that
        # the operator can simply apply, so they must not sink a good strategy.
        warnings: list[str] = []
        advisories: list[str] = []
        if expectancy_r <= 0:
            warnings.append("expectancy per trade is not positive after costs")
        if streak_drawdown_pct > 25:
            warnings.append(
                f"a {planning_streak}-loss streak at {risk_pct:g}% risk is a "
                f"{streak_drawdown_pct:g}% drawdown"
            )
        if trades and trades < 30:
            warnings.append(f"only {trades} out-of-sample trades: statistically thin")
        if recommended_risk < risk_pct:
            advisories.append(
                f"reduce risk per trade to about {recommended_risk:g}% (currently {risk_pct:g}%)"
            )
        if float(oos.get("exposure_pct", 0)) > 80:
            advisories.append("position is open more than 80% of the time; overnight/weekend risk")
        report["warnings"] = warnings
        report["advisories"] = advisories

        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="risk_reviewed")

        self.log(
            f"'{ir.name}': plan for {planning_streak} losses "
            f"({streak_drawdown_pct:g}% dd), suggest {recommended_risk:g}% risk",
            msg=msg,
        )
        await self.emit(
            Topic.RISK_REPORT,
            {**msg.payload, "risk_report": report},
            parent=msg,
            strategy_id=ir.id,
        )

    @staticmethod
    def _expected_streak(win_rate: float, trades: int) -> int:
        """Longest losing run you should expect to see in ``trades`` samples."""
        loss_rate = 1.0 - win_rate
        if trades <= 0 or loss_rate <= 0 or loss_rate >= 1:
            return 0
        return int(np.ceil(np.log(max(trades, 1)) / -np.log(loss_rate)))

    @staticmethod
    def _kelly(metrics: dict[str, Any]) -> float:
        """Kelly fraction in percent of equity, from win rate and payoff ratio."""
        win_rate = float(metrics.get("win_rate_pct", 0.0)) / 100.0
        avg_win = abs(float(metrics.get("avg_win", 0.0)))
        avg_loss = abs(float(metrics.get("avg_loss", 0.0)))
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        payoff = avg_win / avg_loss
        edge = win_rate - (1 - win_rate) / payoff
        return max(0.0, edge * 100.0)


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #


class JudgeAgent(BaseAgent):
    name = "judge"
    role = "Judge"
    squad = "validation"
    description = "Final verdict: PASS, BORDERLINE or REJECT, with written reasoning."
    subscribes = (Topic.RISK_REPORT,)

    async def handle(self, msg: Message) -> None:
        ir = StrategyIR.model_validate(msg.payload["ir"])
        thresholds = factory_section("judge")
        oos = dict(msg.payload.get("metrics", {}).get("oos") or {})
        is_m = dict(msg.payload.get("metrics", {}).get("is") or {})
        robustness = dict(msg.payload.get("robustness") or {})
        risk_report = dict(msg.payload.get("risk_report") or {})

        checks = self._run_checks(oos, is_m, thresholds)
        flags = list(robustness.get("flags") or [])
        risk_warnings = list(risk_report.get("warnings") or [])

        failed = [name for name, c in checks.items() if not c["pass"]]
        band = float(thresholds.get("borderline_band", 0.15))
        near_miss = [name for name in failed if checks[name].get("ratio", 0.0) >= (1 - band)]

        score = robust_score(
            oos,
            min_trades=int(thresholds.get("min_oos_trades", 30)),
            is_metrics=is_m,
            n_params=len(ir.param_space),
            complexity_penalty=float(factory_section("tuner").get("complexity_penalty", 0.02)),
        )

        hard_fail = [name for name in failed if name not in near_miss]

        # A flag from the robustness agent is never cosmetic. The ones below mean
        # the edge is probably not real, so they veto outright; any *other* flag
        # still costs the strategy its PASS.
        # These two mean the edge is probably an artefact, so they veto outright.
        # Softer concerns (fragile parameters, one bad era) cost the strategy its
        # PASS and land it in BORDERLINE, where it still gets packaged - clearly
        # labelled as a candidate rather than a finished product.
        blocking_flags = [f for f in flags if "2x the configured spread" in f or "orderings" in f]

        if hard_fail or blocking_flags:
            verdict = "REJECT"
        elif failed or flags or risk_warnings:
            verdict = "BORDERLINE"
        else:
            verdict = "PASS"

        long_term = (
            verdict == "PASS"
            and not flags
            and float(robustness.get("walk_forward", {}).get("consistency", 0)) >= 0.5
            and bool(robustness.get("cost_stress", {}).get("survives_double_costs", False))
            and float(risk_report.get("expectancy_r", 0)) > 0
        )

        reasons = self._reasons(checks, failed, near_miss, flags, risk_warnings, verdict)
        summary = self._summary(ir, verdict, oos, robustness, risk_report, long_term)

        await in_db(
            save,
            Verdict(
                strategy_id=ir.id,
                project_id=msg.project_id,
                verdict=verdict,
                score=score,
                long_term_viable=long_term,
                checks=checks,
                reasons=reasons,
                summary=summary,
            ),
        )
        await in_db(update_strategy, ir.id, status=f"judged_{verdict.lower()}")
        if msg.project_id:
            await in_db(
                update_project,
                msg.project_id,
                stage=f"judged_{verdict.lower()}",
                status="running" if verdict == "PASS" else "done",
            )

        # A rejected strategy is dead weight from this instant. Nothing downstream
        # reads its backtests - only PASS and BORDERLINE continue to the doc writer
        # - so the bulk goes now rather than sitting on disk waiting for a sweep.
        if verdict == "REJECT" and factory_section("retention").get(
            "purge_rejected_immediately", True
        ):
            freed = await in_db(purge_rejected_strategy, ir.id)
            self.log(
                f"purged '{ir.name}' on rejection: {freed['runs_deleted']} runs, "
                f"{freed['events_deleted']} events, {freed['ir_bytes_freed']}B of IR "
                "(fingerprint and verdict kept for dedupe and learning)",
                msg=msg,
            )

        self.log(f"'{ir.name}' -> {verdict} (score {score:.3f})", msg=msg)
        await self.emit(
            Topic.STRATEGY_JUDGED,
            {
                **msg.payload,
                "verdict": verdict,
                "score": score,
                "long_term_viable": long_term,
                "checks": checks,
                "reasons": reasons,
                "summary": summary,
            },
            parent=msg,
            strategy_id=ir.id,
        )

    @staticmethod
    def _run_checks(
        oos: dict[str, Any], is_m: dict[str, Any], t: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        def check(
            name: str, value: float, threshold: float, *, higher_is_better: bool = True
        ) -> dict[str, Any]:
            if higher_is_better:
                ok = value >= threshold
                ratio = value / threshold if threshold else 1.0
            else:
                ok = value <= threshold
                ratio = threshold / value if value else 1.0
            return {
                "label": name,
                "value": round(float(value), 3),
                "threshold": float(threshold),
                "pass": bool(ok),
                "ratio": round(float(max(0.0, min(ratio, 3.0))), 3),
            }

        is_sharpe = float(is_m.get("sharpe", 0.0))
        oos_sharpe = float(oos.get("sharpe", 0.0))
        retention = oos_sharpe / is_sharpe if is_sharpe > 0 else 0.0

        checks: dict[str, dict[str, Any]] = {
            "oos_sharpe": check(
                "Out-of-sample Sharpe", oos_sharpe, float(t.get("min_oos_sharpe", 0.8))
            ),
            "oos_profit_factor": check(
                "Out-of-sample profit factor",
                float(oos.get("profit_factor", 0.0)),
                float(t.get("min_oos_profit_factor", 1.15)),
            ),
            "oos_drawdown": check(
                "Out-of-sample max drawdown",
                float(oos.get("max_drawdown_pct", 100.0)),
                float(t.get("max_oos_drawdown_pct", 25.0)),
                higher_is_better=False,
            ),
            "oos_trades": check(
                "Out-of-sample trades",
                float(oos.get("trades", 0)),
                float(t.get("min_oos_trades", 30)),
            ),
            "expectancy_r": check(
                "Expectancy per trade (R)",
                float(oos.get("expectancy_r", 0.0)),
                float(t.get("min_expectancy_r", 0.05)),
            ),
            "is_oos_retention": check(
                "IS->OOS Sharpe retention",
                retention,
                float(t.get("max_is_oos_degradation", 0.5)),
            ),
        }

        # Optional gates. Zero means "do not test this", so the operator can turn
        # them on without editing code.
        min_win = float(t.get("min_win_rate_pct", 0) or 0)
        if min_win > 0:
            checks["win_rate"] = check("Win rate %", float(oos.get("win_rate_pct", 0.0)), min_win)

        min_monthly = float(t.get("min_monthly_return_pct", 0) or 0)
        if min_monthly > 0:
            checks["monthly_growth"] = check(
                "Avg monthly return %",
                float(oos.get("avg_monthly_return_pct", 0.0)),
                min_monthly,
            )

        return checks

    @staticmethod
    def _reasons(
        checks: dict[str, dict[str, Any]],
        failed: list[str],
        near_miss: list[str],
        flags: list[str],
        risk_warnings: list[str],
        verdict: str,
    ) -> list[str]:
        reasons: list[str] = []
        for name in failed:
            c = checks[name]
            comparator = "below" if c["threshold"] >= c["value"] else "above"
            note = " (near miss)" if name in near_miss else ""
            reasons.append(
                f"{c['label']} is {c['value']}, {comparator} the required {c['threshold']}{note}"
            )
        reasons.extend(flags)
        reasons.extend(f"risk: {w}" for w in risk_warnings)
        if verdict == "PASS" and not reasons:
            reasons.append("cleared every threshold and every robustness check")
        return reasons

    @staticmethod
    def _summary(
        ir: StrategyIR,
        verdict: str,
        oos: dict[str, Any],
        robustness: dict[str, Any],
        risk_report: dict[str, Any],
        long_term: bool,
    ) -> str:
        wf = robustness.get("walk_forward") or {}
        mc = robustness.get("monte_carlo") or {}
        lines = [
            f"{ir.name} ({ir.style}) on {ir.asset} {ir.timeframe}: {verdict}.",
            f"Out-of-sample: {oos.get('trades', 0)} trades, Sharpe {oos.get('sharpe', 0)}, "
            f"profit factor {oos.get('profit_factor', 0)}, max drawdown "
            f"{oos.get('max_drawdown_pct', 0)}%, expectancy {oos.get('expectancy_r', 0)}R.",
        ]
        if wf.get("fold_count"):
            lines.append(
                f"Walk-forward: {wf.get('profitable_folds')}/{wf.get('fold_count')} eras "
                f"profitable, mean Sharpe {wf.get('mean_sharpe')}."
            )
        if mc.get("runs"):
            lines.append(
                f"Monte Carlo ({mc['runs']} resamples): "
                f"{float(mc.get('prob_profitable', 0)) * 100:.0f}% of trade orderings profitable, "
                f"5th percentile {mc.get('r_5th_percentile')}R."
            )
        if risk_report:
            lines.append(
                f"Risk: plan for a {risk_report.get('planning_loss_streak')}-loss streak "
                f"(~{risk_report.get('streak_drawdown_pct')}% drawdown); suggested risk "
                f"{risk_report.get('recommended_risk_per_trade_pct')}% per trade."
            )
        lines.append(
            "Long-term viability: "
            + (
                "plausible — the edge held out of sample, across eras, and at double costs."
                if long_term
                else "not established by this evidence."
            )
        )
        return " ".join(lines)


__all__ = [
    "JudgeAgent",
    "RiskAgent",
    "RobustnessAgent",
    "TesterAgent",
    "TunerAgent",
]
