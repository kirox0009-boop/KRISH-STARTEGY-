"""Strategy genome: how new strategies come into existence.

Three creation modes, all producing valid :class:`StrategyIR`:

* :meth:`StrategyFactory.fresh`      — instantiate a *recipe* with random params
* :meth:`StrategyFactory.mutate`     — perturb one elite strategy
* :meth:`StrategyFactory.crossover`  — splice two parents

Recipes are structural templates (trend cross, breakout, mean reversion, ...),
not finished strategies. Because parameters, filters, risk blocks, exits and
directions are all randomised and then evolved, the search space is effectively
unbounded — the factory never runs out of new things to try.

``priors`` (produced by the memory agent from past results) bias the dice:
"on GOLD H1, ATR stops and 2.5+ R:R have been working" nudges generation without
hard-coding it.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .assets import universe
from .indicators import REGISTRY
from .ir.schema import (
    Condition,
    ConditionOp,
    Direction,
    FilterBlock,
    IndicatorSpec,
    Operand,
    ParamSpec,
    RiskBlock,
    RuleGroup,
    StrategyIR,
)

# --------------------------------------------------------------------------- #
# naming — every strategy gets a memorable, file-safe name
# --------------------------------------------------------------------------- #

_ADJECTIVES = (
    "Iron",
    "Silent",
    "Rapid",
    "Patient",
    "Cobalt",
    "Crimson",
    "Northern",
    "Quantum",
    "Steady",
    "Feral",
    "Lucid",
    "Obsidian",
    "Solar",
    "Tidal",
    "Vector",
    "Zenith",
    "Amber",
    "Granite",
    "Hollow",
    "Kinetic",
    "Nimbus",
    "Prime",
    "Stellar",
    "Vigil",
)
_NOUNS = (
    "Falcon",
    "Anvil",
    "Compass",
    "Current",
    "Drift",
    "Ember",
    "Forge",
    "Gale",
    "Harbor",
    "Lantern",
    "Meridian",
    "Orbit",
    "Pulse",
    "Ridge",
    "Sentry",
    "Tempo",
    "Thread",
    "Tide",
    "Vault",
    "Wake",
    "Beacon",
    "Cascade",
    "Helix",
    "Summit",
)


def strategy_name(rng: random.Random, style: str) -> str:
    return f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}"


# --------------------------------------------------------------------------- #
# recipe plumbing
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Blueprint:
    """What a recipe returns before risk/filters/params are attached."""

    style: str
    indicators: dict[str, IndicatorSpec]
    entry_long: RuleGroup
    entry_short: RuleGroup = field(default_factory=RuleGroup)
    exit_long: RuleGroup = field(default_factory=RuleGroup)
    exit_short: RuleGroup = field(default_factory=RuleGroup)
    hypothesis: str = ""
    #: recipe's preferred risk shape; the factory still randomises around it
    prefers_trailing: bool = False
    prefers_rr: tuple[float, float] = (1.2, 3.5)
    prefers_stop_atr: tuple[float, float] = (1.2, 3.5)
    trend_filter_hint: str | None = None


Recipe = Callable[[random.Random, str], Blueprint]


def _int_param(rng: random.Random, indicator: str, param: str, lo: float, hi: float) -> float:
    """Sample a parameter, respecting the indicator's declared bounds."""
    meta = REGISTRY[indicator]
    bounds = meta.params.get(param)
    if bounds:
        b_lo, b_hi, is_int = bounds
        lo, hi = max(lo, b_lo), min(hi, b_hi)
        value = rng.uniform(lo, hi)
        return float(round(value)) if is_int else round(value, 2)
    return round(rng.uniform(lo, hi), 2)


def _ma_type(rng: random.Random) -> str:
    return rng.choice(("ema", "sma", "wma"))


def _mirror(group: RuleGroup) -> RuleGroup:
    """Build the opposite-side rule group by flipping each condition."""
    flip = {
        ConditionOp.CROSS_ABOVE: ConditionOp.CROSS_BELOW,
        ConditionOp.CROSS_BELOW: ConditionOp.CROSS_ABOVE,
        ConditionOp.GT: ConditionOp.LT,
        ConditionOp.LT: ConditionOp.GT,
        ConditionOp.GTE: ConditionOp.LTE,
        ConditionOp.LTE: ConditionOp.GTE,
        ConditionOp.RISING: ConditionOp.FALLING,
        ConditionOp.FALLING: ConditionOp.RISING,
    }
    mirrored: list[Condition] = []
    for cond in group.conditions:
        if cond.op is ConditionOp.BETWEEN:
            mirrored.append(cond.model_copy(deep=True))
            continue
        new = cond.model_copy(deep=True)
        new.op = flip.get(cond.op, cond.op)
        # oscillator thresholds mirror around their midpoint
        if (
            new.right is not None
            and new.right.kind.value == "const"
            and new.right.value is not None
            and 0 <= new.right.value <= 100
            and cond.op in {ConditionOp.GT, ConditionOp.LT, ConditionOp.GTE, ConditionOp.LTE}
        ):
            new.right.value = round(100.0 - new.right.value, 2)
        mirrored.append(new)
    return RuleGroup(logic=group.logic, conditions=mirrored)


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #


def _recipe_ma_cross(rng: random.Random, timeframe: str) -> Blueprint:
    kind = _ma_type(rng)
    fast_p = _int_param(rng, kind, "period", 8, 40)
    slow_p = _int_param(rng, kind, "period", max(fast_p * 1.8, 30), 200)
    inds = {
        "fast": IndicatorSpec(type=kind, params={"period": fast_p}),
        "slow": IndicatorSpec(type=kind, params={"period": slow_p}),
    }
    conds = [
        Condition(op=ConditionOp.CROSS_ABOVE, left=Operand.ind("fast"), right=Operand.ind("slow"))
    ]
    if rng.random() < 0.6:
        inds["strength"] = IndicatorSpec(
            type="adx", params={"period": _int_param(rng, "adx", "period", 10, 24)}
        )
        conds.append(
            Condition(
                op=ConditionOp.GT,
                left=Operand.ind("strength"),
                right=Operand.const(round(rng.uniform(15, 30), 1)),
            )
        )
    entry_long = RuleGroup(logic="and", conditions=conds)
    return Blueprint(
        style="trend_following",
        indicators=inds,
        entry_long=entry_long,
        entry_short=_mirror(entry_long),
        hypothesis=(
            f"A {kind.upper()}({fast_p:g}) / {kind.upper()}({slow_p:g}) crossover captures "
            f"persistent {timeframe} trends; an ADX gate avoids chop."
        ),
        prefers_trailing=rng.random() < 0.6,
        prefers_rr=(1.5, 4.0),
    )


def _recipe_donchian_breakout(rng: random.Random, timeframe: str) -> Blueprint:
    period = _int_param(rng, "donchian_high", "period", 20, 120)
    inds = {
        "hi": IndicatorSpec(type="donchian_high", params={"period": period}),
        "lo": IndicatorSpec(type="donchian_low", params={"period": period}),
    }
    entry_long = RuleGroup(
        logic="and",
        conditions=[
            Condition(
                op=ConditionOp.GT, left=Operand.price("close"), right=Operand.ind("hi", shift=1)
            )
        ],
    )
    entry_short = RuleGroup(
        logic="and",
        conditions=[
            Condition(
                op=ConditionOp.LT, left=Operand.price("close"), right=Operand.ind("lo", shift=1)
            )
        ],
    )
    if rng.random() < 0.5:
        inds["vol"] = IndicatorSpec(type="atr_pct", params={"period": 14})
        gate = Condition(
            op=ConditionOp.GT,
            left=Operand.ind("vol"),
            right=Operand.const(round(rng.uniform(0.2, 0.8), 2)),
        )
        entry_long.conditions.append(gate)
        entry_short.conditions.append(gate.model_copy(deep=True))
    return Blueprint(
        style="breakout",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        hypothesis=(
            f"Closing beyond the {period:g}-bar range on {timeframe} marks a genuine "
            "expansion that continues for several bars."
        ),
        prefers_trailing=rng.random() < 0.7,
        prefers_rr=(1.5, 4.5),
        prefers_stop_atr=(1.5, 4.0),
    )


def _recipe_bb_reversion(rng: random.Random, timeframe: str) -> Blueprint:
    period = _int_param(rng, "bb_lower", "period", 14, 60)
    mult = _int_param(rng, "bb_lower", "mult", 1.6, 3.0)
    inds = {
        "lower": IndicatorSpec(type="bb_lower", params={"period": period, "mult": mult}),
        "upper": IndicatorSpec(type="bb_upper", params={"period": period, "mult": mult}),
        "mid": IndicatorSpec(type="sma", params={"period": period}),
    }
    entry_long = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.LT, left=Operand.price("close"), right=Operand.ind("lower"))
        ]
    )
    entry_short = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.GT, left=Operand.price("close"), right=Operand.ind("upper"))
        ]
    )
    exit_long = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_ABOVE, left=Operand.price("close"), right=Operand.ind("mid")
            )
        ]
    )
    exit_short = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_BELOW, left=Operand.price("close"), right=Operand.ind("mid")
            )
        ]
    )
    return Blueprint(
        style="mean_reversion",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        exit_long=exit_long,
        exit_short=exit_short,
        hypothesis=(
            f"{timeframe} closes outside a {period:g}/{mult:g} Bollinger band overshoot and "
            "revert to the mean."
        ),
        prefers_rr=(0.8, 2.0),
        prefers_stop_atr=(1.5, 3.5),
    )


def _recipe_rsi_pullback(rng: random.Random, timeframe: str) -> Blueprint:
    rsi_p = _int_param(rng, "rsi", "period", 7, 21)
    trend_p = _int_param(rng, "ema", "period", 100, 250)
    oversold = round(rng.uniform(20, 40), 1)
    inds = {
        "osc": IndicatorSpec(type="rsi", params={"period": rsi_p}),
        "trend": IndicatorSpec(type="ema", params={"period": trend_p}),
    }
    entry_long = RuleGroup(
        logic="and",
        conditions=[
            Condition(
                op=ConditionOp.CROSS_ABOVE,
                left=Operand.ind("osc"),
                right=Operand.const(oversold),
            )
        ],
    )
    entry_short = _mirror(entry_long)
    return Blueprint(
        style="trend_pullback",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        hypothesis=(
            f"Buying RSI({rsi_p:g}) recoveries from {oversold:g} only in the direction of the "
            f"EMA({trend_p:g}) trend beats buying every dip."
        ),
        prefers_rr=(1.2, 3.0),
        trend_filter_hint="trend",
    )


def _recipe_macd_momentum(rng: random.Random, timeframe: str) -> Blueprint:
    fast = _int_param(rng, "macd", "fast", 8, 20)
    slow = _int_param(rng, "macd", "slow", max(fast * 1.5, 21), 60)
    signal = _int_param(rng, "macd_signal", "signal", 5, 15)
    inds = {
        "line": IndicatorSpec(type="macd", params={"fast": fast, "slow": slow}),
        "sig": IndicatorSpec(
            type="macd_signal", params={"fast": fast, "slow": slow, "signal": signal}
        ),
    }
    entry_long = RuleGroup(
        logic="and",
        conditions=[
            Condition(
                op=ConditionOp.CROSS_ABOVE, left=Operand.ind("line"), right=Operand.ind("sig")
            ),
            Condition(op=ConditionOp.GT, left=Operand.ind("line"), right=Operand.const(0.0)),
        ],
    )
    return Blueprint(
        style="momentum",
        indicators=inds,
        entry_long=entry_long,
        entry_short=_mirror(entry_long),
        hypothesis=(
            f"MACD({fast:g},{slow:g},{signal:g}) signal-line crossings above zero identify "
            f"momentum bursts that persist on {timeframe}."
        ),
        prefers_trailing=rng.random() < 0.5,
    )


def _recipe_zscore_reversion(rng: random.Random, timeframe: str) -> Blueprint:
    period = _int_param(rng, "zscore", "period", 30, 150)
    threshold = round(rng.uniform(1.5, 3.0), 2)
    inds = {"z": IndicatorSpec(type="zscore", params={"period": period})}
    entry_long = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.LT, left=Operand.ind("z"), right=Operand.const(-threshold))
        ]
    )
    entry_short = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.GT, left=Operand.ind("z"), right=Operand.const(threshold))
        ]
    )
    exit_long = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.CROSS_ABOVE, left=Operand.ind("z"), right=Operand.const(0.0))
        ]
    )
    exit_short = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.CROSS_BELOW, left=Operand.ind("z"), right=Operand.const(0.0))
        ]
    )
    return Blueprint(
        style="mean_reversion",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        exit_long=exit_long,
        exit_short=exit_short,
        hypothesis=(
            f"{period:g}-bar z-score extremes beyond {threshold:g}s mean-revert to zero "
            f"on {timeframe}."
        ),
        prefers_rr=(0.8, 2.2),
    )


def _recipe_keltner_trend(rng: random.Random, timeframe: str) -> Blueprint:
    period = _int_param(rng, "keltner_upper", "period", 15, 60)
    mult = _int_param(rng, "keltner_upper", "mult", 1.2, 3.0)
    inds = {
        "kc_up": IndicatorSpec(type="keltner_upper", params={"period": period, "mult": mult}),
        "kc_dn": IndicatorSpec(type="keltner_lower", params={"period": period, "mult": mult}),
        "basis": IndicatorSpec(type="ema", params={"period": period}),
    }
    entry_long = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_ABOVE, left=Operand.price("close"), right=Operand.ind("kc_up")
            )
        ]
    )
    entry_short = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_BELOW, left=Operand.price("close"), right=Operand.ind("kc_dn")
            )
        ]
    )
    exit_long = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_BELOW, left=Operand.price("close"), right=Operand.ind("basis")
            )
        ]
    )
    exit_short = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_ABOVE, left=Operand.price("close"), right=Operand.ind("basis")
            )
        ]
    )
    return Blueprint(
        style="volatility_breakout",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        exit_long=exit_long,
        exit_short=exit_short,
        hypothesis=(
            f"Closing outside a {mult:g}xATR Keltner channel signals volatility expansion "
            "that trends until price loses the basis."
        ),
        prefers_trailing=True,
        prefers_rr=(1.5, 4.0),
    )


def _recipe_stoch_reversal(rng: random.Random, timeframe: str) -> Blueprint:
    period = _int_param(rng, "stoch_k", "period", 8, 30)
    smooth = _int_param(rng, "stoch_k", "smooth", 2, 6)
    level = round(rng.uniform(15, 30), 1)
    inds = {"k": IndicatorSpec(type="stoch_k", params={"period": period, "smooth": smooth})}
    entry_long = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.CROSS_ABOVE, left=Operand.ind("k"), right=Operand.const(level))
        ]
    )
    return Blueprint(
        style="oscillator_reversal",
        indicators=inds,
        entry_long=entry_long,
        entry_short=_mirror(entry_long),
        hypothesis=(
            f"Stochastic({period:g},{smooth:g}) exiting the {level:g} zone marks exhaustion "
            f"of short-term {timeframe} moves."
        ),
        prefers_rr=(1.0, 2.5),
    )


def _recipe_squeeze_expansion(rng: random.Random, timeframe: str) -> Blueprint:
    look = _int_param(rng, "highest", "period", 15, 60)
    vol_p = _int_param(rng, "atr_pct", "period", 10, 30)
    quiet = round(rng.uniform(0.15, 0.6), 2)
    inds = {
        "hh": IndicatorSpec(type="highest", params={"period": look}, source="high"),
        "ll": IndicatorSpec(type="lowest", params={"period": look}, source="low"),
        "vol": IndicatorSpec(type="atr_pct", params={"period": vol_p}),
    }
    entry_long = RuleGroup(
        logic="and",
        conditions=[
            Condition(
                op=ConditionOp.GT, left=Operand.price("close"), right=Operand.ind("hh", shift=1)
            ),
            Condition(
                op=ConditionOp.LT, left=Operand.ind("vol", shift=2), right=Operand.const(quiet)
            ),
        ],
    )
    entry_short = RuleGroup(
        logic="and",
        conditions=[
            Condition(
                op=ConditionOp.LT, left=Operand.price("close"), right=Operand.ind("ll", shift=1)
            ),
            Condition(
                op=ConditionOp.LT, left=Operand.ind("vol", shift=2), right=Operand.const(quiet)
            ),
        ],
    )
    return Blueprint(
        style="squeeze_breakout",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        hypothesis=(
            f"Breakouts that follow a quiet period (ATR% below {quiet:g}) run further than "
            "breakouts from already-volatile conditions."
        ),
        prefers_trailing=True,
        prefers_rr=(2.0, 5.0),
    )


def _recipe_roc_trend(rng: random.Random, timeframe: str) -> Blueprint:
    period = _int_param(rng, "roc", "period", 5, 40)
    trend_p = _int_param(rng, "sma", "period", 80, 250)
    threshold = round(rng.uniform(0.2, 2.0), 2)
    inds = {
        "mom": IndicatorSpec(type="roc", params={"period": period}),
        "trend": IndicatorSpec(type="sma", params={"period": trend_p}),
    }
    entry_long = RuleGroup(
        logic="and",
        conditions=[
            Condition(op=ConditionOp.GT, left=Operand.ind("mom"), right=Operand.const(threshold)),
            Condition(op=ConditionOp.RISING, left=Operand.ind("mom"), lookback=rng.randint(2, 4)),
        ],
    )
    return Blueprint(
        style="momentum",
        indicators=inds,
        entry_long=entry_long,
        entry_short=_mirror(entry_long),
        hypothesis=(
            f"Accelerating {period:g}-bar rate of change above {threshold:g}% in the direction "
            "of the long trend precedes continuation."
        ),
        trend_filter_hint="trend",
        prefers_trailing=rng.random() < 0.5,
    )


# --------------------------------------------------------------------------- #
# Market structure / SMC recipes
#
# These cannot use _mirror(): flipping "close crosses above the swing HIGH" gives
# "close crosses below the swing HIGH", which is meaningless. The short side has to
# reference the swing LOW, the bearish gap, the bearish order block. So each of
# these writes both sides explicitly.
# --------------------------------------------------------------------------- #


def _recipe_bos_continuation(rng: random.Random, timeframe: str) -> Blueprint:
    left = _int_param(rng, "swing_high_level", "left", 2, 6)
    right = _int_param(rng, "swing_high_level", "right", 2, 5)
    disp = round(rng.uniform(0.6, 1.6), 2)
    inds = {
        "sw_hi": IndicatorSpec(type="swing_high_level", params={"left": left, "right": right}),
        "sw_lo": IndicatorSpec(type="swing_low_level", params={"left": left, "right": right}),
        "disp": IndicatorSpec(type="displacement", params={"period": 14}),
    }
    long_conds = [
        Condition(
            op=ConditionOp.CROSS_ABOVE, left=Operand.price("close"), right=Operand.ind("sw_hi")
        ),
        Condition(op=ConditionOp.GT, left=Operand.ind("disp"), right=Operand.const(disp)),
    ]
    short_conds = [
        Condition(
            op=ConditionOp.CROSS_BELOW, left=Operand.price("close"), right=Operand.ind("sw_lo")
        ),
        Condition(op=ConditionOp.GT, left=Operand.ind("disp"), right=Operand.const(disp)),
    ]
    return Blueprint(
        style="market_structure",
        indicators=inds,
        entry_long=RuleGroup(logic="and", conditions=long_conds),
        entry_short=RuleGroup(logic="and", conditions=short_conds),
        hypothesis=(
            f"A break of structure on {timeframe} - price closing beyond the last confirmed "
            f"swing with a body of at least {disp:g} ATR - marks a genuine shift that "
            "continues, rather than another failed poke at the level."
        ),
        prefers_trailing=rng.random() < 0.6,
        prefers_rr=(1.5, 4.0),
        prefers_stop_atr=(1.0, 2.8),
    )


def _recipe_fvg_retrace(rng: random.Random, timeframe: str) -> Blueprint:
    inds = {
        "fvg_up": IndicatorSpec(type="fvg_bull_level", params={}),
        "fvg_dn": IndicatorSpec(type="fvg_bear_level", params={}),
    }
    # tagged the gap with the wick, closed back on the right side of it
    entry_long = RuleGroup(
        logic="and",
        conditions=[
            Condition(op=ConditionOp.LT, left=Operand.price("low"), right=Operand.ind("fvg_up")),
            Condition(op=ConditionOp.GT, left=Operand.price("close"), right=Operand.ind("fvg_up")),
        ],
    )
    entry_short = RuleGroup(
        logic="and",
        conditions=[
            Condition(op=ConditionOp.GT, left=Operand.price("high"), right=Operand.ind("fvg_dn")),
            Condition(op=ConditionOp.LT, left=Operand.price("close"), right=Operand.ind("fvg_dn")),
        ],
    )
    return Blueprint(
        style="fair_value_gap",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        hypothesis=(
            f"Price that skipped a band on the way up returns to fill it. Entering when "
            f"{timeframe} wicks into the fair value gap but closes back above it should "
            "catch the continuation rather than the fill."
        ),
        prefers_rr=(1.4, 3.5),
        prefers_stop_atr=(0.8, 2.2),
    )


def _recipe_order_block_reclaim(rng: random.Random, timeframe: str) -> Blueprint:
    inds = {
        "ob_up": IndicatorSpec(type="ob_bull_level", params={}),
        "ob_dn": IndicatorSpec(type="ob_bear_level", params={}),
    }
    entry_long = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_ABOVE, left=Operand.price("close"), right=Operand.ind("ob_up")
            )
        ]
    )
    entry_short = RuleGroup(
        conditions=[
            Condition(
                op=ConditionOp.CROSS_BELOW, left=Operand.price("close"), right=Operand.ind("ob_dn")
            )
        ]
    )
    if rng.random() < 0.55:
        gate = IndicatorSpec(type="displacement", params={"period": 14})
        inds["disp"] = gate
        level = round(rng.uniform(0.5, 1.4), 2)
        entry_long.conditions.append(
            Condition(op=ConditionOp.GT, left=Operand.ind("disp"), right=Operand.const(level))
        )
        entry_long.logic = "and"
        entry_short.conditions.append(
            Condition(op=ConditionOp.GT, left=Operand.ind("disp"), right=Operand.const(level))
        )
        entry_short.logic = "and"
    return Blueprint(
        style="order_block",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        hypothesis=(
            "The last opposing candle before an impulse marks where size entered. "
            f"Reclaiming that level on {timeframe} should resume the move."
        ),
        prefers_trailing=rng.random() < 0.5,
        prefers_rr=(1.5, 4.0),
        prefers_stop_atr=(0.9, 2.5),
    )


def _recipe_liquidity_sweep(rng: random.Random, timeframe: str) -> Blueprint:
    left = _int_param(rng, "liquidity_sweep_low", "left", 2, 6)
    right = _int_param(rng, "liquidity_sweep_low", "right", 2, 5)
    inds = {
        "sweep_lo": IndicatorSpec(
            type="liquidity_sweep_low", params={"left": left, "right": right}
        ),
        "sweep_hi": IndicatorSpec(
            type="liquidity_sweep_high", params={"left": left, "right": right}
        ),
    }
    entry_long = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.GT, left=Operand.ind("sweep_lo"), right=Operand.const(50.0))
        ]
    )
    entry_short = RuleGroup(
        conditions=[
            Condition(op=ConditionOp.GT, left=Operand.ind("sweep_hi"), right=Operand.const(50.0))
        ]
    )
    if rng.random() < 0.5:
        inds["wick_dn"] = IndicatorSpec(type="wick_down_pct", params={})
        inds["wick_up"] = IndicatorSpec(type="wick_up_pct", params={})
        thr = round(rng.uniform(35, 60), 1)
        entry_long.conditions.append(
            Condition(op=ConditionOp.GT, left=Operand.ind("wick_dn"), right=Operand.const(thr))
        )
        entry_long.logic = "and"
        entry_short.conditions.append(
            Condition(op=ConditionOp.GT, left=Operand.ind("wick_up"), right=Operand.const(thr))
        )
        entry_short.logic = "and"
    return Blueprint(
        style="liquidity_sweep",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        hypothesis=(
            f"On {timeframe}, a wick that takes out the last swing and closes back inside "
            "is stops being run rather than a real break, so the move reverses."
        ),
        prefers_rr=(1.3, 3.2),
        prefers_stop_atr=(0.8, 2.0),
    )


def _recipe_premium_discount(rng: random.Random, timeframe: str) -> Blueprint:
    period = _int_param(rng, "equilibrium", "period", 30, 140)
    wick = round(rng.uniform(30, 55), 1)
    inds = {
        "eq": IndicatorSpec(type="equilibrium", params={"period": period}),
        "wick_dn": IndicatorSpec(type="wick_down_pct", params={}),
        "wick_up": IndicatorSpec(type="wick_up_pct", params={}),
    }
    entry_long = RuleGroup(
        logic="and",
        conditions=[
            Condition(op=ConditionOp.LT, left=Operand.price("close"), right=Operand.ind("eq")),
            Condition(op=ConditionOp.GT, left=Operand.ind("wick_dn"), right=Operand.const(wick)),
        ],
    )
    entry_short = RuleGroup(
        logic="and",
        conditions=[
            Condition(op=ConditionOp.GT, left=Operand.price("close"), right=Operand.ind("eq")),
            Condition(op=ConditionOp.GT, left=Operand.ind("wick_up"), right=Operand.const(wick)),
        ],
    )
    return Blueprint(
        style="premium_discount",
        indicators=inds,
        entry_long=entry_long,
        entry_short=entry_short,
        hypothesis=(
            f"Buying only in the discount half of the {period:g}-bar range, and only when the "
            f"bar shows a rejection wick of {wick:g}% or more, beats buying dips indiscriminately."
        ),
        prefers_rr=(1.4, 3.5),
        prefers_stop_atr=(0.9, 2.4),
    )


RECIPES: dict[str, Recipe] = {
    "ma_cross": _recipe_ma_cross,
    "donchian_breakout": _recipe_donchian_breakout,
    "bb_reversion": _recipe_bb_reversion,
    "rsi_pullback": _recipe_rsi_pullback,
    "macd_momentum": _recipe_macd_momentum,
    "zscore_reversion": _recipe_zscore_reversion,
    "keltner_trend": _recipe_keltner_trend,
    "stoch_reversal": _recipe_stoch_reversal,
    "squeeze_expansion": _recipe_squeeze_expansion,
    "roc_trend": _recipe_roc_trend,
    # market structure / SMC
    "bos_continuation": _recipe_bos_continuation,
    "fvg_retrace": _recipe_fvg_retrace,
    "order_block_reclaim": _recipe_order_block_reclaim,
    "liquidity_sweep": _recipe_liquidity_sweep,
    "premium_discount": _recipe_premium_discount,
}


# --------------------------------------------------------------------------- #
# the factory
# --------------------------------------------------------------------------- #


class StrategyFactory:
    """Creates, mutates and crossbreeds strategies."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    # ------------------------------ fresh ------------------------------ #

    def fresh(
        self,
        asset: str,
        timeframe: str,
        *,
        recipe: str | None = None,
        priors: dict[str, Any] | None = None,
    ) -> StrategyIR:
        rng = self.rng
        priors = priors or {}
        recipe_name = recipe or self._pick_recipe(priors)
        blueprint = RECIPES[recipe_name](rng, timeframe)

        ir = StrategyIR(
            name=strategy_name(rng, blueprint.style),
            style=blueprint.style,
            asset=asset.upper(),
            timeframe=timeframe,
            direction=self._pick_direction(priors),
            indicators=blueprint.indicators,
            entry_long=blueprint.entry_long,
            entry_short=blueprint.entry_short,
            exit_long=blueprint.exit_long,
            exit_short=blueprint.exit_short,
            risk=self._risk_block(blueprint, priors),
            filters=self._filters(blueprint, asset, priors),
            hypothesis=blueprint.hypothesis,
            origin="fresh",
            generation=0,
            notes=f"recipe={recipe_name}",
        )
        # A long-only or short-only strategy must not carry a dead rule group.
        if ir.direction is Direction.LONG:
            ir.entry_short = RuleGroup()
            ir.exit_short = RuleGroup()
        elif ir.direction is Direction.SHORT:
            ir.entry_long = RuleGroup()
            ir.exit_long = RuleGroup()
        ir.param_space = self.build_param_space(ir)
        return ir

    def _pick_recipe(self, priors: dict[str, Any]) -> str:
        """Weight recipes by past performance when the memory agent has data."""
        weights = priors.get("recipe_weights") or {}
        names = list(RECIPES)
        if not weights:
            return self.rng.choice(names)
        # floor of 0.15 keeps exploration alive: nothing is ever fully abandoned
        w = [max(0.15, float(weights.get(n, 1.0))) for n in names]
        return self.rng.choices(names, weights=w, k=1)[0]

    def _pick_direction(self, priors: dict[str, Any]) -> Direction:
        preferred = priors.get("direction")
        if preferred in {"long", "short", "both"} and self.rng.random() < 0.7:
            return Direction(preferred)
        roll = self.rng.random()
        if roll < 0.72:
            return Direction.BOTH
        return Direction.LONG if roll < 0.9 else Direction.SHORT

    def _risk_block(self, blueprint: Blueprint, priors: dict[str, Any]) -> RiskBlock:
        rng = self.rng
        stop_lo, stop_hi = blueprint.prefers_stop_atr
        rr_lo, rr_hi = blueprint.prefers_rr

        stop_kind = "atr"
        if rng.random() < 0.12:
            stop_kind = "percent"
        if priors.get("stop_kind") in {"atr", "percent", "points"} and rng.random() < 0.6:
            stop_kind = str(priors["stop_kind"])

        if stop_kind == "atr":
            stop_value = round(rng.uniform(stop_lo, stop_hi), 2)
        elif stop_kind == "percent":
            stop_value = round(rng.uniform(0.4, 3.0), 2)
        else:
            stop_value = round(rng.uniform(50, 500), 0)

        target_kind = "rr" if rng.random() < 0.75 else "atr"
        target_value = (
            round(rng.uniform(rr_lo, rr_hi), 2)
            if target_kind == "rr"
            else round(rng.uniform(1.5, 6.0), 2)
        )

        trailing = blueprint.prefers_trailing if rng.random() < 0.7 else rng.random() < 0.4

        return RiskBlock(
            stop_kind=stop_kind,  # type: ignore[arg-type]
            stop_value=stop_value,
            target_kind=target_kind,  # type: ignore[arg-type]
            target_value=target_value,
            trailing=trailing,
            trail_atr_mult=round(rng.uniform(1.5, 4.5), 2),
            breakeven_at_r=round(rng.uniform(0.6, 1.6), 2) if rng.random() < 0.35 else None,
            risk_per_trade_pct=float(priors.get("risk_per_trade_pct", 1.0)),
            max_bars_in_trade=int(rng.choice((72, 120, 200, 400))) if rng.random() < 0.5 else None,
            atr_period=int(rng.choice((10, 14, 20))),
        )

    def _filters(self, blueprint: Blueprint, asset: str, priors: dict[str, Any]) -> FilterBlock:
        rng = self.rng
        block = FilterBlock()
        if blueprint.trend_filter_hint and blueprint.trend_filter_hint in blueprint.indicators:
            block.trend_filter_alias = blueprint.trend_filter_hint
            block.trend_filter_mode = "above"
        elif rng.random() < 0.25:
            candidates = [
                alias
                for alias, spec in blueprint.indicators.items()
                if REGISTRY[spec.type].family in {"trend", "level"}
            ]
            if candidates:
                block.trend_filter_alias = rng.choice(candidates)
                block.trend_filter_mode = rng.choice(("above", "with_slope"))

        if rng.random() < 0.3:
            block.min_atr_pct = round(rng.uniform(0.05, 0.4), 3)

        session = universe().get(asset).session
        if session != "24x7" and rng.random() < 0.3:
            start = rng.choice((6, 7, 8, 12, 13))
            block.allowed_hours = list(range(start, min(start + rng.choice((6, 8, 10)), 24)))

        if rng.random() < 0.25:
            block.cooldown_bars = int(rng.choice((3, 5, 10, 20)))
        return block

    # -------------------------- parameter space ------------------------ #

    def build_param_space(self, ir: StrategyIR) -> list[ParamSpec]:
        """Expose a *small* set of knobs. Fewer knobs = less curve fitting."""
        specs: list[ParamSpec] = []
        for alias, spec in ir.indicators.items():
            meta = REGISTRY[spec.type]
            for param, (lo, hi, is_int) in meta.params.items():
                if param not in spec.params:
                    continue
                current = float(spec.params[param])
                low = max(lo, current * 0.5)
                high = min(hi, max(current * 1.8, current + 5))
                if high <= low:
                    continue
                specs.append(
                    ParamSpec(
                        path=f"indicators.{alias}.params.{param}",
                        low=round(low, 3),
                        high=round(high, 3),
                        is_int=is_int,
                        label=f"{alias}.{param}",
                    )
                )
        if ir.risk.stop_kind == "atr":
            specs.append(
                ParamSpec(
                    path="risk.stop_value",
                    low=max(0.5, ir.risk.stop_value * 0.5),
                    high=min(6.0, ir.risk.stop_value * 1.8),
                    label="stop (xATR)",
                )
            )
        if ir.risk.target_kind == "rr":
            specs.append(
                ParamSpec(
                    path="risk.target_value",
                    low=max(0.6, ir.risk.target_value * 0.5),
                    high=min(8.0, ir.risk.target_value * 1.8),
                    label="target (R:R)",
                )
            )
        if ir.risk.trailing:
            specs.append(
                ParamSpec(
                    path="risk.trail_atr_mult",
                    low=max(0.8, ir.risk.trail_atr_mult * 0.5),
                    high=min(7.0, ir.risk.trail_atr_mult * 1.8),
                    label="trail (xATR)",
                )
            )
        # Hard cap: more than 6 tunable knobs on this little data is fitting noise.
        self.rng.shuffle(specs)
        return specs[:6]

    # ----------------------------- mutation ---------------------------- #

    def mutate(self, parent: StrategyIR, *, priors: dict[str, Any] | None = None) -> StrategyIR:
        rng = self.rng
        child = parent.model_copy(deep=True)
        child.id = StrategyIR.model_fields["id"].default_factory()  # type: ignore[misc]
        child.generation = parent.generation + 1
        child.parents = [parent.id]
        child.origin = "mutation"
        child.name = f"{parent.name.split(' Mk')[0]} Mk{child.generation + 1}"

        operators = [
            self._mut_param,
            self._mut_risk,
            self._mut_threshold,
            self._mut_filter,
            self._mut_add_condition,
            self._mut_drop_condition,
            self._mut_swap_indicator,
            self._mut_direction,
            self._mut_exit_style,
        ]
        applied: list[str] = []
        for _ in range(rng.randint(1, 3)):
            op = rng.choice(operators)
            label = op(child)
            if label:
                applied.append(label)
        child.notes = f"{parent.notes} | mutation: {', '.join(applied) or 'none'}"
        child.param_space = self.build_param_space(child)
        return child

    def _mut_param(self, ir: StrategyIR) -> str | None:
        candidates = [
            (alias, param) for alias, spec in ir.indicators.items() for param in spec.params
        ]
        if not candidates:
            return None
        alias, param = self.rng.choice(candidates)
        meta = REGISTRY[ir.indicators[alias].type]
        bounds = meta.params.get(param)
        current = float(ir.indicators[alias].params[param])
        scale = self.rng.uniform(0.6, 1.6)
        value = current * scale
        if bounds:
            lo, hi, is_int = bounds
            value = min(max(value, lo), hi)
            value = float(round(value)) if is_int else round(value, 2)
        ir.indicators[alias].params[param] = value
        return f"{alias}.{param} {current:g}->{value:g}"

    def _mut_risk(self, ir: StrategyIR) -> str:
        rng = self.rng
        choice = rng.choice(("stop", "target", "trail", "breakeven", "time"))
        if choice == "stop":
            ir.risk.stop_value = round(max(0.3, ir.risk.stop_value * rng.uniform(0.6, 1.6)), 2)
            return f"stop->{ir.risk.stop_value:g}"
        if choice == "target":
            ir.risk.target_value = round(max(0.4, ir.risk.target_value * rng.uniform(0.6, 1.7)), 2)
            return f"target->{ir.risk.target_value:g}"
        if choice == "trail":
            ir.risk.trailing = not ir.risk.trailing
            return f"trailing->{ir.risk.trailing}"
        if choice == "breakeven":
            ir.risk.breakeven_at_r = (
                None if ir.risk.breakeven_at_r else round(rng.uniform(0.6, 1.8), 2)
            )
            return f"breakeven->{ir.risk.breakeven_at_r}"
        ir.risk.max_bars_in_trade = (
            None if ir.risk.max_bars_in_trade else int(rng.choice((72, 120, 200, 400)))
        )
        return f"time_stop->{ir.risk.max_bars_in_trade}"

    def _mut_threshold(self, ir: StrategyIR) -> str | None:
        consts = [
            (group_name, cond)
            for group_name in ("entry_long", "entry_short", "exit_long", "exit_short")
            for cond in getattr(ir, group_name).conditions
            if cond.right is not None and cond.right.kind.value == "const"
        ]
        if not consts:
            return None
        group_name, cond = self.rng.choice(consts)
        old = float(cond.right.value or 0.0)  # type: ignore[union-attr]
        delta = self.rng.uniform(-0.25, 0.25)
        new = old * (1 + delta) if abs(old) > 1e-9 else self.rng.uniform(-1, 1)
        cond.right.value = round(new, 3)  # type: ignore[union-attr]
        return f"{group_name} threshold {old:g}->{new:g}"

    def _mut_filter(self, ir: StrategyIR) -> str:
        rng = self.rng
        choice = rng.choice(("trend", "vol", "session", "cooldown"))
        if choice == "trend":
            if ir.filters.trend_filter_mode != "off":
                ir.filters.trend_filter_mode = "off"
                ir.filters.trend_filter_alias = None
                return "trend filter off"
            candidates = [
                alias
                for alias, spec in ir.indicators.items()
                if REGISTRY[spec.type].family in {"trend", "level"}
            ]
            if candidates:
                ir.filters.trend_filter_alias = rng.choice(candidates)
                ir.filters.trend_filter_mode = rng.choice(("above", "with_slope"))
                return f"trend filter on ({ir.filters.trend_filter_alias})"
            return "trend filter unchanged"
        if choice == "vol":
            ir.filters.min_atr_pct = (
                None if ir.filters.min_atr_pct else round(rng.uniform(0.05, 0.5), 3)
            )
            return f"min_atr_pct->{ir.filters.min_atr_pct}"
        if choice == "session":
            if ir.filters.allowed_hours:
                ir.filters.allowed_hours = None
                return "session filter off"
            start = rng.choice((6, 7, 8, 12, 13))
            ir.filters.allowed_hours = list(range(start, min(start + rng.choice((6, 8, 10)), 24)))
            return f"session {ir.filters.allowed_hours[0]}-{ir.filters.allowed_hours[-1]} UTC"
        ir.filters.cooldown_bars = int(rng.choice((0, 3, 5, 10, 20)))
        return f"cooldown->{ir.filters.cooldown_bars}"

    def _mut_add_condition(self, ir: StrategyIR) -> str | None:
        """Add a confirmation gate — and mirror it on the short side."""
        rng = self.rng
        if len(ir.entry_long.conditions) >= 4:
            return None
        alias = f"gate{len(ir.indicators)}"
        pick = rng.choice(("adx", "rsi", "atr_pct", "zscore", "stoch_k"))
        params = {"period": float(rng.choice((10, 14, 20, 30)))}
        if pick == "stoch_k":
            params["smooth"] = 3.0
        ir.indicators[alias] = IndicatorSpec(type=pick, params=params)
        scale = REGISTRY[pick].scale
        if scale == "0-100":
            level = round(rng.uniform(20, 60), 1)
        elif pick == "atr_pct":
            level = round(rng.uniform(0.05, 0.8), 3)
        else:
            level = round(rng.uniform(-1.5, 1.5), 2)
        cond = Condition(
            op=rng.choice((ConditionOp.GT, ConditionOp.LT)),
            left=Operand.ind(alias),
            right=Operand.const(level),
        )
        if not ir.entry_long.is_empty():
            ir.entry_long.conditions.append(cond)
            ir.entry_long.logic = "and"
        if not ir.entry_short.is_empty():
            ir.entry_short.conditions.append(cond.model_copy(deep=True))
            ir.entry_short.logic = "and"
        return f"+gate {pick}{cond.op.value}{level:g}"

    def _mut_drop_condition(self, ir: StrategyIR) -> str | None:
        for group_name in ("entry_long", "entry_short"):
            group: RuleGroup = getattr(ir, group_name)
            if len(group.conditions) > 1:
                dropped = group.conditions.pop(self.rng.randrange(len(group.conditions)))
                self._prune_indicators(ir)
                return f"-{group_name}: {dropped.label()}"
        return None

    def _mut_swap_indicator(self, ir: StrategyIR) -> str | None:
        """Swap an MA family member for another (ema <-> sma <-> wma)."""
        swappable = [
            alias for alias, spec in ir.indicators.items() if spec.type in {"ema", "sma", "wma"}
        ]
        if not swappable:
            return None
        alias = self.rng.choice(swappable)
        old = ir.indicators[alias].type
        new = self.rng.choice([t for t in ("ema", "sma", "wma") if t != old])
        ir.indicators[alias].type = new
        return f"{alias}: {old}->{new}"

    def _mut_direction(self, ir: StrategyIR) -> str | None:
        if ir.direction is Direction.BOTH:
            side = self.rng.choice((Direction.LONG, Direction.SHORT))
            ir.direction = side
            if side is Direction.LONG:
                ir.entry_short, ir.exit_short = RuleGroup(), RuleGroup()
            else:
                ir.entry_long, ir.exit_long = RuleGroup(), RuleGroup()
            return f"direction->{side.value}"
        return None

    def _mut_exit_style(self, ir: StrategyIR) -> str:
        if ir.risk.target_kind == "none":
            ir.risk.target_kind = "rr"
            ir.risk.target_value = round(self.rng.uniform(1.2, 3.5), 2)
            return "target on"
        if self.rng.random() < 0.35 and (
            not ir.exit_long.is_empty() or not ir.exit_short.is_empty()
        ):
            ir.risk.target_kind = "none"
            return "target off (rule exit only)"
        ir.risk.target_kind = "atr" if ir.risk.target_kind == "rr" else "rr"
        ir.risk.target_value = round(self.rng.uniform(1.2, 4.0), 2)
        return f"target kind->{ir.risk.target_kind}"

    # ---------------------------- crossover ---------------------------- #

    def crossover(self, a: StrategyIR, b: StrategyIR) -> StrategyIR:
        """Splice two parents: entries from one, exits/risk/filters from the other.

        Aliases are namespaced per parent so the two halves cannot collide.
        """
        rng = self.rng
        child = a.model_copy(deep=True)
        child.id = StrategyIR.model_fields["id"].default_factory()  # type: ignore[misc]
        child.generation = max(a.generation, b.generation) + 1
        child.parents = [a.id, b.id]
        child.origin = "crossover"
        child.name = f"{a.name.split()[0]} {b.name.split()[-1]}"
        child.style = f"{a.style}+{b.style}" if a.style != b.style else a.style

        take_entry_from_b = rng.random() < 0.5
        source = b if take_entry_from_b else a
        other = a if take_entry_from_b else b

        child.indicators = {f"a_{k}": v.model_copy(deep=True) for k, v in source.indicators.items()}
        child.entry_long = _rename_aliases(source.entry_long, "a_")
        child.entry_short = _rename_aliases(source.entry_short, "a_")

        # exits and filters come from the other parent, when it has them
        if not other.exit_long.is_empty() or not other.exit_short.is_empty():
            for k, v in other.indicators.items():
                child.indicators[f"b_{k}"] = v.model_copy(deep=True)
            child.exit_long = _rename_aliases(other.exit_long, "b_")
            child.exit_short = _rename_aliases(other.exit_short, "b_")
        else:
            child.exit_long = _rename_aliases(source.exit_long, "a_")
            child.exit_short = _rename_aliases(source.exit_short, "a_")

        child.risk = (b if rng.random() < 0.5 else a).risk.model_copy(deep=True)
        child.filters = other.filters.model_copy(deep=True)
        if child.filters.trend_filter_alias:
            renamed = f"b_{child.filters.trend_filter_alias}"
            if renamed in child.indicators:
                child.filters.trend_filter_alias = renamed
            elif f"a_{child.filters.trend_filter_alias}" in child.indicators:
                child.filters.trend_filter_alias = f"a_{child.filters.trend_filter_alias}"
            else:
                child.filters.trend_filter_alias = None
                child.filters.trend_filter_mode = "off"

        child.direction = source.direction
        if child.direction is Direction.LONG:
            child.entry_short, child.exit_short = RuleGroup(), RuleGroup()
        elif child.direction is Direction.SHORT:
            child.entry_long, child.exit_long = RuleGroup(), RuleGroup()

        child.hypothesis = (
            f"Combine the entry logic of '{source.name}' with the exit and risk management of "
            f"'{other.name}'."
        )
        child.notes = f"crossover of {a.id} x {b.id}"
        self._prune_indicators(child)
        child.param_space = self.build_param_space(child)
        return child

    # ----------------------------- helpers ----------------------------- #

    @staticmethod
    def _prune_indicators(ir: StrategyIR) -> None:
        """Drop indicators nothing references any more."""
        used: set[str] = set()
        for group_name in ("entry_long", "entry_short", "exit_long", "exit_short"):
            for cond in getattr(ir, group_name).conditions:
                for op in (cond.left, cond.right, cond.right2):
                    if op is not None and op.kind.value == "indicator" and op.ref:
                        used.add(op.ref)
        if ir.filters.trend_filter_alias:
            used.add(ir.filters.trend_filter_alias)
        ir.indicators = {k: v for k, v in ir.indicators.items() if k in used}
        if ir.filters.trend_filter_alias not in ir.indicators:
            ir.filters.trend_filter_alias = None
            ir.filters.trend_filter_mode = "off"


def _rename_aliases(group: RuleGroup, prefix: str) -> RuleGroup:
    clone = group.model_copy(deep=True)
    for cond in clone.conditions:
        for op in (cond.left, cond.right, cond.right2):
            if op is not None and op.kind.value == "indicator" and op.ref:
                op.ref = f"{prefix}{op.ref}"
    return clone
