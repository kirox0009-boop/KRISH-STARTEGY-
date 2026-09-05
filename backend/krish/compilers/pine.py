"""Strategy IR -> TradingView Pine Script v5.

Proof that the IR is genuinely multi-target: the same JSON that the Python
backtester runs is mechanically translated here, with no strategy logic
re-implemented by hand. Every indicator maps to a ``ta.*`` builtin chosen to
match this project's Python implementation (same smoothing, same causality).

Emitted script includes: inputs for every tunable parameter, the entry/exit
logic, ATR-based stops/targets/trailing, the session and volatility filters, and
``alert()`` calls carrying a JSON webhook payload so it can drive automation.
"""

from __future__ import annotations

from ..indicators import REGISTRY
from ..ir.schema import (
    Condition,
    ConditionOp,
    IndicatorSpec,
    Operand,
    OperandKind,
    RuleGroup,
    StrategyIR,
)


class PineUnsupported(NotImplementedError):
    """This IR uses something Pine Script cannot express faithfully."""


_PRICE = {
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "hl2": "hl2",
    "hlc3": "hlc3",
    "ohlc4": "ohlc4",
}


def _num(value: float) -> str:
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


def _var(alias: str) -> str:
    return f"i_{alias}"


def _param(alias: str, param: str) -> str:
    return f"p_{alias}_{param}"


def _indicator_expr(alias: str, spec: IndicatorSpec) -> list[str]:
    """Return the Pine lines that define ``_var(alias)``."""
    src = _PRICE[spec.source]
    p = {k: _param(alias, k) for k in spec.params}
    v = _var(alias)
    t = spec.type

    if t in {"sma", "ema", "wma"}:
        return [f"{v} = ta.{t}({src}, {p['period']})"]
    if t == "rsi":
        return [f"{v} = ta.rsi({src}, {p['period']})"]
    if t == "atr":
        return [f"{v} = ta.atr({p['period']})"]
    if t == "atr_pct":
        return [f"{v} = ta.atr({p['period']}) / close * 100.0"]
    if t == "stddev":
        return [f"{v} = ta.stdev({src}, {p['period']})"]
    if t == "bb_upper":
        return [
            f"{v} = ta.sma({src}, {p['period']}) + {p['mult']} * ta.stdev({src}, {p['period']})"
        ]
    if t == "bb_lower":
        return [
            f"{v} = ta.sma({src}, {p['period']}) - {p['mult']} * ta.stdev({src}, {p['period']})"
        ]
    if t == "bb_percent":
        return [
            f"{v}_basis = ta.sma({src}, {p['period']})",
            f"{v}_dev = {p['mult']} * ta.stdev({src}, {p['period']})",
            f"{v} = ({src} - ({v}_basis - {v}_dev)) / math.max({v}_dev * 2, 1e-10) * 100.0",
        ]
    if t == "macd":
        return [f"{v} = ta.ema({src}, {p['fast']}) - ta.ema({src}, {p['slow']})"]
    if t == "macd_signal":
        return [
            f"{v}_line = ta.ema({src}, {p['fast']}) - ta.ema({src}, {p['slow']})",
            f"{v} = ta.ema({v}_line, {p['signal']})",
        ]
    if t == "macd_hist":
        return [
            f"{v}_line = ta.ema({src}, {p['fast']}) - ta.ema({src}, {p['slow']})",
            f"{v} = {v}_line - ta.ema({v}_line, {p['signal']})",
        ]
    if t == "donchian_high":
        return [f"{v} = ta.highest(high, {p['period']})"]
    if t == "donchian_low":
        return [f"{v} = ta.lowest(low, {p['period']})"]
    if t == "donchian_mid":
        return [f"{v} = (ta.highest(high, {p['period']}) + ta.lowest(low, {p['period']})) / 2.0"]
    if t == "highest":
        return [f"{v} = ta.highest({src}, {p['period']})"]
    if t == "lowest":
        return [f"{v} = ta.lowest({src}, {p['period']})"]
    if t == "roc":
        return [f"{v} = ta.roc({src}, {p['period']})"]
    if t == "momentum":
        return [f"{v} = ta.mom({src}, {p['period']})"]
    if t == "stoch_k":
        return [f"{v} = ta.sma(ta.stoch(close, high, low, {p['period']}), {p['smooth']})"]
    if t == "cci":
        return [f"{v} = ta.cci({src}, {p['period']})"]
    if t == "adx":
        return [f"[_dip_{alias}, _dim_{alias}, {v}] = ta.dmi({p['period']}, {p['period']})"]
    if t == "zscore":
        return [
            f"{v} = ({src} - ta.sma({src}, {p['period']})) / "
            f"math.max(ta.stdev({src}, {p['period']}), 1e-10)"
        ]
    if t == "keltner_upper":
        return [f"{v} = ta.ema(close, {p['period']}) + {p['mult']} * ta.atr({p['period']})"]
    if t == "keltner_lower":
        return [f"{v} = ta.ema(close, {p['period']}) - {p['mult']} * ta.atr({p['period']})"]
    if t == "vwap":
        return [f"{v} = ta.vwap"]
    raise PineUnsupported(f"indicator '{t}' has no faithful Pine Script equivalent yet")


def _operand(op: Operand) -> str:
    if op.kind is OperandKind.CONST:
        return _num(float(op.value or 0.0))
    base = _PRICE[str(op.ref)] if op.kind is OperandKind.PRICE else _var(str(op.ref))
    return f"{base}[{op.shift}]" if op.shift else base


def _condition(cond: Condition) -> str:
    left = _operand(cond.left)
    if cond.op is ConditionOp.RISING:
        return f"ta.rising({left}, {cond.lookback})"
    if cond.op is ConditionOp.FALLING:
        return f"ta.falling({left}, {cond.lookback})"
    right = _operand(cond.right)  # type: ignore[arg-type]
    match cond.op:
        case ConditionOp.CROSS_ABOVE:
            return f"ta.crossover({left}, {right})"
        case ConditionOp.CROSS_BELOW:
            return f"ta.crossunder({left}, {right})"
        case ConditionOp.GT:
            return f"({left} > {right})"
        case ConditionOp.LT:
            return f"({left} < {right})"
        case ConditionOp.GTE:
            return f"({left} >= {right})"
        case ConditionOp.LTE:
            return f"({left} <= {right})"
        case ConditionOp.BETWEEN:
            upper = _operand(cond.right2)  # type: ignore[arg-type]
            return (
                f"({left} >= math.min({right}, {upper}) and {left} <= math.max({right}, {upper}))"
            )
    raise PineUnsupported(f"operator '{cond.op}' is not translatable")


def _group(group: RuleGroup, *, default: str = "false") -> str:
    if group.is_empty():
        return default
    joiner = " and " if group.logic == "and" else " or "
    return joiner.join(_condition(c) for c in group.conditions)


def to_pine(ir: StrategyIR) -> str:
    """Compile ``ir`` to a Pine Script v5 strategy."""
    lines: list[str] = []
    add = lines.append

    add("// ---------------------------------------------------------------------------")
    add(f"// {ir.name} - generated by KRISH from Strategy IR {ir.id}")
    add(f"// {ir.asset} {ir.timeframe} | style: {ir.style} | generation {ir.generation}")
    add("// Machine-generated. Edit the IR, not this file: regeneration overwrites it.")
    add("// ---------------------------------------------------------------------------")
    add("//@version=5")
    safe_name = ir.name.replace('"', "'")
    add(
        f'strategy("{safe_name} [KRISH]", overlay=true, '
        "default_qty_type=strategy.percent_of_equity, default_qty_value=10, "
        "commission_type=strategy.commission.percent, commission_value=0.02, "
        "slippage=2, calc_on_every_tick=false, process_orders_on_close=false)"
    )
    add("")

    # ---- inputs ----------------------------------------------------------
    add("// === Inputs =========================================================")
    for alias, spec in ir.indicators.items():
        meta = REGISTRY[spec.type]
        for param, value in spec.params.items():
            is_int = meta.params.get(param, (0, 0, True))[2]
            fn = "input.int" if is_int else "input.float"
            value_str = str(int(value)) if is_int else _num(value)
            step = "" if is_int else ", step=0.1"
            add(
                f'{_param(alias, param)} = {fn}({value_str}, "{alias} {param}", '
                f'minval={1 if is_int else 0.1}{step}, group="Indicators")'
            )
    add(
        f'p_atr_period = input.int({ir.risk.atr_period}, "ATR period (risk)", minval=2, '
        'group="Risk")'
    )
    add(
        f'p_stop = input.float({_num(ir.risk.stop_value)}, "Stop ({ir.risk.stop_kind})", '
        'minval=0.1, step=0.1, group="Risk")'
    )
    add(
        f"p_target = input.float({_num(ir.risk.target_value)}, "
        f'"Target ({ir.risk.target_kind})", minval=0.1, step=0.1, group="Risk")'
    )
    add(f'p_trailing = input.bool({str(ir.risk.trailing).lower()}, "Trailing stop", group="Risk")')
    add(
        f'p_trail_mult = input.float({_num(ir.risk.trail_atr_mult)}, "Trail (xATR)", '
        'minval=0.1, step=0.1, group="Risk")'
    )
    add("")

    # ---- indicators ------------------------------------------------------
    add("// === Indicators =====================================================")
    for alias, spec in ir.indicators.items():
        for line in _indicator_expr(alias, spec):
            add(line)
    add("atrValue = ta.atr(p_atr_period)")
    add("")

    # ---- filters ---------------------------------------------------------
    add("// === Filters ========================================================")
    filters: list[str] = []
    f = ir.filters
    if f.allowed_hours:
        hours = ", ".join(str(h) for h in f.allowed_hours)
        add(f"var allowedHours = array.from({hours})")
        add('hourOk = array.includes(allowedHours, hour(time, "UTC"))')
        filters.append("hourOk")
    if f.allowed_weekdays is not None:
        days = ", ".join(str((d + 1) % 7 + 1) for d in f.allowed_weekdays)
        add(f"var allowedDays = array.from({days})")
        add('dayOk = array.includes(allowedDays, dayofweek(time, "UTC"))')
        filters.append("dayOk")
    if f.min_atr_pct is not None:
        add(f"volMinOk = (atrValue / close * 100.0) >= {_num(f.min_atr_pct)}")
        filters.append("volMinOk")
    if f.max_atr_pct is not None:
        add(f"volMaxOk = (atrValue / close * 100.0) <= {_num(f.max_atr_pct)}")
        filters.append("volMaxOk")
    add(f"baseFilter = {' and '.join(filters) if filters else 'true'}")

    trend_long = trend_short = "true"
    if f.trend_filter_mode != "off" and f.trend_filter_alias:
        line = _var(f.trend_filter_alias)
        if f.trend_filter_mode == "with_slope":
            trend_long, trend_short = f"({line} > {line}[1])", f"({line} < {line}[1])"
        elif f.trend_filter_mode == "above":
            trend_long, trend_short = f"(close > {line})", f"(close < {line})"
        else:
            trend_long, trend_short = f"(close < {line})", f"(close > {line})"
    add(f"trendOkLong = {trend_long}")
    add(f"trendOkShort = {trend_short}")
    add("")

    # ---- signals ---------------------------------------------------------
    add("// === Signals ========================================================")
    long_ok = ir.trades_long()
    short_ok = ir.trades_short()
    add(f"rawLong = {_group(ir.entry_long) if long_ok else 'false'}")
    add(f"rawShort = {_group(ir.entry_short) if short_ok else 'false'}")
    add("enterLong = rawLong and baseFilter and trendOkLong")
    add("enterShort = rawShort and baseFilter and trendOkShort")
    add(f"ruleExitLong = {_group(ir.exit_long)}")
    add(f"ruleExitShort = {_group(ir.exit_short)}")
    add("")

    # ---- risk maths ------------------------------------------------------
    add("// === Stop / target distances ========================================")
    if ir.risk.stop_kind == "atr":
        add("stopDist = atrValue * p_stop")
    elif ir.risk.stop_kind == "percent":
        add("stopDist = close * p_stop / 100.0")
    elif ir.risk.stop_kind == "points":
        add("stopDist = p_stop * syminfo.mintick")
    else:
        add("stopDist = atrValue * 3.0  // no explicit stop in the IR; used for sizing only")

    if ir.risk.target_kind == "rr":
        add("targetDist = stopDist * p_target")
    elif ir.risk.target_kind == "atr":
        add("targetDist = atrValue * p_target")
    elif ir.risk.target_kind == "percent":
        add("targetDist = close * p_target / 100.0")
    elif ir.risk.target_kind == "points":
        add("targetDist = p_target * syminfo.mintick")
    else:
        add("targetDist = 0.0")
    add("")

    # ---- orders ----------------------------------------------------------
    add("// === Orders =========================================================")
    add('webhookLong = \'{"strategy":"' + safe_name + '","action":"buy"}\'')
    add('webhookShort = \'{"strategy":"' + safe_name + '","action":"sell"}\'')
    add('webhookFlat = \'{"strategy":"' + safe_name + '","action":"close"}\'')
    add("")
    if long_ok:
        add("if enterLong and strategy.position_size == 0")
        add('    strategy.entry("Long", strategy.long)')
        add("    alert(webhookLong, alert.freq_once_per_bar_close)")
    if short_ok:
        add("if enterShort and strategy.position_size == 0")
        add('    strategy.entry("Short", strategy.short)')
        add("    alert(webhookShort, alert.freq_once_per_bar_close)")
    add("")
    add("longStop = strategy.position_avg_price - stopDist")
    add("shortStop = strategy.position_avg_price + stopDist")
    add("longTarget = targetDist > 0 ? strategy.position_avg_price + targetDist : na")
    add("shortTarget = targetDist > 0 ? strategy.position_avg_price - targetDist : na")
    add("")
    add("if strategy.position_size > 0")
    add("    trailStop = p_trailing ? close - atrValue * p_trail_mult : na")
    add("    finalStop = na(trailStop) ? longStop : math.max(longStop, trailStop)")
    add('    strategy.exit("Long exit", from_entry="Long", stop=finalStop, limit=longTarget)')
    add("    if ruleExitLong")
    add('        strategy.close("Long", comment="rule exit")')
    add("        alert(webhookFlat, alert.freq_once_per_bar_close)")
    add("")
    add("if strategy.position_size < 0")
    add("    trailStop = p_trailing ? close + atrValue * p_trail_mult : na")
    add("    finalStop = na(trailStop) ? shortStop : math.min(shortStop, trailStop)")
    add('    strategy.exit("Short exit", from_entry="Short", stop=finalStop, limit=shortTarget)')
    add("    if ruleExitShort")
    add('        strategy.close("Short", comment="rule exit")')
    add("        alert(webhookFlat, alert.freq_once_per_bar_close)")
    add("")
    if ir.risk.max_bars_in_trade:
        add(f"// time stop: {ir.risk.max_bars_in_trade} bars")
        add(
            f"if strategy.position_size != 0 and "
            f"(bar_index - strategy.opentrades.entry_bar_index(0)) >= "
            f"{ir.risk.max_bars_in_trade}"
        )
        add('    strategy.close_all(comment="time stop")')
        add("    alert(webhookFlat, alert.freq_once_per_bar_close)")
        add("")
    add("// === Plots ==========================================================")
    for alias, spec in ir.indicators.items():
        if REGISTRY[spec.type].scale == "price":
            add(f'plot({_var(alias)}, "{alias}", linewidth=1)')
    add("")
    add("// Notes")
    for note_line in ir.describe().splitlines():
        add(f"// {note_line}")

    return "\n".join(lines) + "\n"


__all__ = ["PineUnsupported", "to_pine"]
