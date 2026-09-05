"""Build squad: the developer agent.

The architect is deliberately allowed to be reckless — variety is its job. The
developer is the one that has to make the thing actually work: it audits the IR,
*repairs* what it can, rejects what it cannot, and only then lets the expensive
backtester see it.

Repair matters more than rejection. A strategy with one nonsense condition and
four good ones should lose the nonsense condition, not the whole idea.
"""

from __future__ import annotations

from typing import Any

from ..ir.compiler import compile_ir
from ..ir.schema import Condition, ConditionOp, Direction, Operand, RuleGroup, StrategyIR
from ..ir.validate import AuditReport, audit_ir
from ..messages import Message, Topic
from ..store import in_db, update_project, update_strategy
from .base import BaseAgent
from .data import frame_cache_get


class DeveloperAgent(BaseAgent):
    name = "developer"
    role = "Developer"
    squad = "build"
    description = "Audits, repairs and compiles Strategy IR into runnable signals."
    subscribes = (Topic.STRATEGY_CREATED, Topic.STRATEGY_TEST_FAILED)

    MAX_REPAIR_ROUNDS = 4

    async def handle(self, msg: Message) -> None:
        raw = msg.payload.get("ir")
        if not raw:
            self.log("strategy message carried no IR", level="error", msg=msg)
            return
        try:
            ir = StrategyIR.model_validate(raw)
        except Exception as exc:
            await self._invalid(msg, [f"IR failed schema validation: {exc}"], None)
            return

        report = audit_ir(ir)
        repairs: list[str] = []
        rounds = 0
        while not report.ok and rounds < self.MAX_REPAIR_ROUNDS:
            rounds += 1
            fixed = self._repair(ir, report)
            if not fixed:
                break
            repairs.extend(fixed)
            self.progress(f"repairing {ir.name} (round {rounds})")
            report = audit_ir(ir)

        if not report.ok:
            await self._invalid(msg, report.errors, ir, repairs=repairs)
            return

        # Cheap compile smoke test when the frame is already in this process:
        # catches indicator/aliasing problems before we pay for a full backtest.
        compile_note: dict[str, Any] = {}
        frame = frame_cache_get(ir.asset, ir.timeframe)
        if frame is not None:
            try:
                compiled = compile_ir(ir, frame.iloc[-3000:])
                counts = compiled.signal_counts()
                compile_note = {"signals_sample": counts, "warmup": compiled.warmup}
                if counts["entry_long"] + counts["entry_short"] == 0:
                    await self._invalid(
                        msg,
                        ["compiles cleanly but produces zero entry signals on recent data"],
                        ir,
                        repairs=repairs,
                    )
                    return
            except Exception as exc:
                await self._invalid(msg, [f"compile failed: {exc}"], ir, repairs=repairs)
                return

        if repairs:
            self.log(f"repaired '{ir.name}': {'; '.join(repairs)}", msg=msg)
            await in_db(
                update_strategy,
                ir.id,
                ir=ir.model_dump(mode="json"),
                fingerprint=ir.fingerprint(),
            )
        await in_db(update_strategy, ir.id, status="built")
        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="built")

        await self.emit(
            Topic.STRATEGY_BUILT,
            {
                "strategy_id": ir.id,
                "name": ir.name,
                "asset": ir.asset,
                "timeframe": ir.timeframe,
                "ir": ir.model_dump(mode="json"),
                "repairs": repairs,
                "warnings": report.warnings,
                **compile_note,
            },
            parent=msg,
            strategy_id=ir.id,
        )

    # ------------------------------------------------------------------ #
    # repair
    # ------------------------------------------------------------------ #

    def _repair(self, ir: StrategyIR, report: AuditReport) -> list[str]:
        """Apply the narrowest fix for each error. Returns what it changed."""
        actions: list[str] = []
        for error in report.errors:
            action = self._repair_one(ir, error)
            if action:
                actions.append(action)
        return actions

    def _repair_one(self, ir: StrategyIR, error: str) -> str | None:
        lowered = error.lower()

        if "no way to ever exit" in lowered:
            ir.risk.target_kind = "rr"
            ir.risk.target_value = max(ir.risk.target_value, 1.5)
            return "added a 1.5R+ target so trades can close"

        if "no stop loss" in lowered:
            ir.risk.stop_kind = "atr"
            ir.risk.stop_value = max(ir.risk.stop_value, 1.5)
            return "added a 1.5xATR stop (risk was unbounded)"

        if "min_atr_pct >= max_atr_pct" in lowered:
            ir.filters.max_atr_pct = None
            return "cleared max_atr_pct (volatility filter blocked every bar)"

        if "trend filter" in lowered:
            ir.filters.trend_filter_mode = "off"
            ir.filters.trend_filter_alias = None
            return "disabled a dangling trend filter"

        if "param_space path" in lowered:
            before = len(ir.param_space)
            ir.param_space = [s for s in ir.param_space if self._param_ok(ir, s.path)]
            return f"dropped {before - len(ir.param_space)} invalid tuner parameter(s)"

        if "direction is long but entry_long is empty" in lowered:
            if not ir.entry_short.is_empty():
                ir.direction = Direction.SHORT
                return "flipped direction to short (only short rules existed)"
            return None

        if "direction is short but entry_short is empty" in lowered:
            if not ir.entry_long.is_empty():
                ir.direction = Direction.LONG
                return "flipped direction to long (only long rules existed)"
            return None

        # Everything below is "drop the offending condition from this group".
        group_name = next(
            (
                g
                for g in ("entry_long", "entry_short", "exit_long", "exit_short")
                if error.startswith(f"{g}:")
            ),
            None,
        )
        if group_name is None:
            return None
        group: RuleGroup = getattr(ir, group_name)

        if (
            "scale mismatch" in lowered
            or "compares an operand to itself" in lowered
            or ("can never fire" in lowered)
        ):
            label = self._label_in(error)
            removed = self._drop_by_label(group, label)
            if removed:
                self._ensure_group_usable(ir, group_name)
                return f"removed nonsense condition from {group_name}: {label}"
            return None

        if "contradictory pair" in lowered and len(group.conditions) > 1:
            group.conditions.pop()
            self._ensure_group_usable(ir, group_name)
            return f"removed a contradictory condition from {group_name}"

        if "unknown indicator alias" in lowered:
            alias = error.rsplit("'", 2)[-2] if "'" in error else ""
            before = len(group.conditions)
            group.conditions = [c for c in group.conditions if not self._references(c, alias)]
            if len(group.conditions) != before:
                self._ensure_group_usable(ir, group_name)
                return f"removed {group_name} condition(s) referencing missing '{alias}'"
        return None

    @staticmethod
    def _param_ok(ir: StrategyIR, path: str) -> bool:
        try:
            ir.get_param(path)
            return True
        except (AttributeError, KeyError, TypeError):
            return False

    @staticmethod
    def _label_in(error: str) -> str:
        parts = error.split("'")
        return parts[1] if len(parts) >= 2 else ""

    @staticmethod
    def _drop_by_label(group: RuleGroup, label: str) -> bool:
        before = len(group.conditions)
        group.conditions = [c for c in group.conditions if c.label() != label]
        return len(group.conditions) != before

    @staticmethod
    def _references(cond: Condition, alias: str) -> bool:
        return any(
            op is not None and op.kind.value == "indicator" and op.ref == alias
            for op in (cond.left, cond.right, cond.right2)
        )

    def _ensure_group_usable(self, ir: StrategyIR, group_name: str) -> None:
        """If repairs emptied an entry group, retire that side rather than leave it dead."""
        group: RuleGroup = getattr(ir, group_name)
        if group.conditions or not group_name.startswith("entry_"):
            return
        if group_name == "entry_long" and not ir.entry_short.is_empty():
            ir.direction = Direction.SHORT
            ir.exit_long = RuleGroup()
        elif group_name == "entry_short" and not ir.entry_long.is_empty():
            ir.direction = Direction.LONG
            ir.exit_short = RuleGroup()
        else:
            # Nothing left to trade on either side: fall back to a minimal,
            # honest momentum entry rather than emitting a dead strategy.
            alias = next(iter(ir.indicators), None)
            if alias:
                setattr(
                    ir,
                    group_name,
                    RuleGroup(
                        conditions=[
                            Condition(op=ConditionOp.RISING, left=Operand.ind(alias), lookback=3)
                        ]
                    ),
                )

    # ------------------------------------------------------------------ #

    async def _invalid(
        self,
        msg: Message,
        errors: list[str],
        ir: StrategyIR | None,
        *,
        repairs: list[str] | None = None,
    ) -> None:
        name = ir.name if ir else msg.payload.get("name", "unknown")
        self.log(f"rejected '{name}': {'; '.join(errors)}", level="warn", msg=msg)
        if ir is not None:
            await in_db(update_strategy, ir.id, status="invalid")
        if msg.project_id:
            await in_db(
                update_project,
                msg.project_id,
                stage="invalid",
                status="stopped",
                meta={"errors": errors, "repairs": repairs or []},
            )
        await self.emit(
            Topic.STRATEGY_INVALID,
            {
                "strategy_id": ir.id if ir else msg.strategy_id,
                "name": name,
                "errors": errors,
                "repairs": repairs or [],
            },
            parent=msg,
        )


__all__ = ["DeveloperAgent"]
