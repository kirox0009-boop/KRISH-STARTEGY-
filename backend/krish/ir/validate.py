"""Static audit of a Strategy IR — run by the developer agent before any backtest.

Cheap checks that kill garbage early, so the expensive backtester only ever sees
strategies that could plausibly work:

* structural sanity (dangling aliases, unused indicators, empty entries)
* scale mismatches (comparing an RSI to a price, comparing price to a constant)
* degenerate logic (always-true / contradictory conditions, self-comparison)
* risk sanity (no stop at all, target smaller than costs)
* look-ahead audit (any negative shift, any non-causal construct)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..assets import universe
from ..indicators import REGISTRY
from .schema import Condition, ConditionOp, Operand, OperandKind, RuleGroup, StrategyIR


class IRValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(slots=True)
class AuditReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


_PRICE_LIKE = {"price", "unbounded"}


def _operand_scale(op: Operand, ir: StrategyIR) -> str:
    if op.kind is OperandKind.CONST:
        return "const"
    if op.kind is OperandKind.PRICE:
        return "price"
    spec = ir.indicators.get(str(op.ref))
    return REGISTRY[spec.type].scale if spec else "unknown"


def _check_condition(cond: Condition, ir: StrategyIR, where: str, report: AuditReport) -> None:
    operands = [cond.left, cond.right, cond.right2]
    for op in operands:
        if op is None:
            continue
        if op.shift < 0:  # pragma: no cover - schema blocks it, kept as defence
            report.errors.append(f"{where}: negative shift is look-ahead ({op.label()})")
        if op.kind is OperandKind.INDICATOR and str(op.ref) not in ir.indicators:
            report.errors.append(f"{where}: unknown indicator alias '{op.ref}'")

    if cond.right is not None and cond.op not in {ConditionOp.RISING, ConditionOp.FALLING}:
        left, right = cond.left, cond.right
        if left.kind == right.kind and left.ref == right.ref and left.shift == right.shift:
            report.errors.append(f"{where}: '{cond.label()}' compares an operand to itself")

        ls, rs = _operand_scale(left, ir), _operand_scale(right, ir)
        if "unknown" not in (ls, rs) and ls != rs:
            price_side = {ls, rs} & _PRICE_LIKE
            if ls == "const" or rs == "const":
                other = rs if ls == "const" else ls
                if other == "price":
                    report.warnings.append(
                        f"{where}: comparing price to a raw constant "
                        f"('{cond.label()}') will not generalise across regimes"
                    )
            elif price_side and ({ls, rs} - _PRICE_LIKE):
                report.errors.append(f"{where}: scale mismatch in '{cond.label()}' ({ls} vs {rs})")

        # oscillator thresholds that can never trigger
        if right.kind is OperandKind.CONST and _operand_scale(left, ir) == "0-100":
            value = float(right.value or 0)
            if not -1 <= value <= 101:
                report.errors.append(
                    f"{where}: '{cond.label()}' can never fire (0-100 scale vs {value:g})"
                )


def _check_group(group: RuleGroup, ir: StrategyIR, where: str, report: AuditReport) -> None:
    seen: set[str] = set()
    for cond in group.conditions:
        label = cond.label()
        if label in seen:
            report.warnings.append(f"{where}: duplicate condition '{label}'")
        seen.add(label)
        _check_condition(cond, ir, where, report)

    if group.logic == "and" and len(group.conditions) >= 2:
        # x > y AND x < y can never be true
        for i, a in enumerate(group.conditions):
            for b in group.conditions[i + 1 :]:
                if _contradicts(a, b):
                    report.errors.append(
                        f"{where}: contradictory pair '{a.label()}' AND '{b.label()}'"
                    )


_OPPOSITES = {
    (ConditionOp.GT, ConditionOp.LT),
    (ConditionOp.GTE, ConditionOp.LTE),
    (ConditionOp.CROSS_ABOVE, ConditionOp.CROSS_BELOW),
    (ConditionOp.RISING, ConditionOp.FALLING),
}


def _same_operand(a: Operand | None, b: Operand | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.kind == b.kind and a.ref == b.ref and a.shift == b.shift and a.value == b.value


def _contradicts(a: Condition, b: Condition) -> bool:
    pair = (a.op, b.op)
    if pair not in _OPPOSITES and tuple(reversed(pair)) not in _OPPOSITES:
        return False
    return _same_operand(a.left, b.left) and _same_operand(a.right, b.right)


def audit_ir(ir: StrategyIR) -> AuditReport:
    report = AuditReport(ok=True)

    # --- universe -------------------------------------------------------
    try:
        asset = universe().get(ir.asset)
        if ir.timeframe not in asset.timeframes:
            report.warnings.append(
                f"timeframe {ir.timeframe} is not in {asset.key}'s configured "
                f"timeframes {list(asset.timeframes)}"
            )
    except KeyError as exc:
        report.errors.append(str(exc))

    # --- entries exist --------------------------------------------------
    if ir.entry_long.is_empty() and ir.entry_short.is_empty():
        report.errors.append("strategy has no entry rules at all")
    if ir.direction.value == "long" and ir.entry_long.is_empty():
        report.errors.append("direction is long but entry_long is empty")
    if ir.direction.value == "short" and ir.entry_short.is_empty():
        report.errors.append("direction is short but entry_short is empty")

    # --- indicators -----------------------------------------------------
    if not ir.indicators:
        report.warnings.append("strategy uses no indicators (pure price action)")
    referenced: set[str] = set()
    for group in (ir.entry_long, ir.entry_short, ir.exit_long, ir.exit_short):
        for cond in group.conditions:
            for op in (cond.left, cond.right, cond.right2):
                if op is not None and op.kind is OperandKind.INDICATOR and op.ref:
                    referenced.add(op.ref)
    if ir.filters.trend_filter_alias:
        referenced.add(ir.filters.trend_filter_alias)
    for alias in set(ir.indicators) - referenced:
        report.warnings.append(f"indicator '{alias}' is computed but never used")
    if ir.filters.trend_filter_mode != "off" and not ir.filters.trend_filter_alias:
        report.errors.append("trend filter enabled without an alias")
    if ir.filters.trend_filter_alias and ir.filters.trend_filter_alias not in ir.indicators:
        report.errors.append(f"trend filter alias '{ir.filters.trend_filter_alias}' is not defined")

    # --- rule groups ----------------------------------------------------
    _check_group(ir.entry_long, ir, "entry_long", report)
    _check_group(ir.entry_short, ir, "entry_short", report)
    _check_group(ir.exit_long, ir, "exit_long", report)
    _check_group(ir.exit_short, ir, "exit_short", report)

    # --- exits exist in some form --------------------------------------
    has_rule_exit = not ir.exit_long.is_empty() or not ir.exit_short.is_empty()
    has_stop = ir.risk.stop_kind != "none"
    has_target = ir.risk.target_kind != "none"
    if not (has_rule_exit or has_target or ir.risk.max_bars_in_trade):
        report.errors.append("no way to ever exit a trade (no rule exit, no target, no time stop)")
    if not has_stop and not ir.risk.max_bars_in_trade:
        report.errors.append("no stop loss and no time stop: unbounded risk")

    # --- risk sanity ----------------------------------------------------
    if has_stop and ir.risk.stop_kind == "atr" and ir.risk.stop_value < 0.3:
        report.warnings.append(
            f"stop of {ir.risk.stop_value:g}xATR is inside typical noise and will bleed on costs"
        )
    if has_target and ir.risk.target_kind == "rr" and ir.risk.target_value < 0.5:
        report.warnings.append(
            f"reward:risk of {ir.risk.target_value:g} needs a very high win rate to survive costs"
        )
    if ir.risk.risk_per_trade_pct > 3:
        report.warnings.append(
            f"risk per trade {ir.risk.risk_per_trade_pct:g}% is aggressive for long-term survival"
        )

    # --- filters over-restricting --------------------------------------
    if ir.filters.allowed_hours is not None and len(ir.filters.allowed_hours) <= 1:
        report.warnings.append("session filter allows <= 1 hour per day; expect very few trades")
    if (
        ir.filters.min_atr_pct is not None
        and ir.filters.max_atr_pct is not None
        and ir.filters.min_atr_pct >= ir.filters.max_atr_pct
    ):
        report.errors.append("min_atr_pct >= max_atr_pct: volatility filter blocks everything")

    # --- tuner space ----------------------------------------------------
    for spec in ir.param_space:
        try:
            ir.get_param(spec.path)
        except (AttributeError, KeyError, TypeError):
            report.errors.append(f"param_space path '{spec.path}' does not exist in this IR")

    report.ok = not report.errors
    return report


def validate_ir(ir: StrategyIR, *, strict: bool = False) -> AuditReport:
    """Raise :class:`IRValidationError` if the IR is unusable."""
    report = audit_ir(ir)
    if not report.ok or (strict and report.warnings):
        raise IRValidationError(report.errors or report.warnings)
    return report
