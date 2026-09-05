"""Strategy IR schema.

Deliberately small and closed: a fixed set of operators and operand kinds means
(a) the architect agent cannot emit something uncompilable, and (b) the MQL5 and
Pine compilers only ever have a finite number of cases to translate.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..indicators import PRICE_SOURCES, REGISTRY
from ..messages import new_id

SCHEMA_VERSION = 1


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class OperandKind(StrEnum):
    INDICATOR = "indicator"  # reference an alias from ir.indicators
    PRICE = "price"  # close / open / high / low / hl2 / hlc3 / ohlc4
    CONST = "const"  # a number


class ConditionOp(StrEnum):
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    RISING = "rising"  # left has risen for `lookback` bars
    FALLING = "falling"
    BETWEEN = "between"  # low <= left <= high


class Operand(BaseModel):
    kind: OperandKind
    ref: str | None = None  # indicator alias or price field
    value: float | None = None  # for CONST
    shift: int = Field(0, ge=0, le=50)  # bars back; never negative (no peeking)

    @model_validator(mode="after")
    def _check(self) -> Operand:
        if self.kind is OperandKind.CONST:
            if self.value is None:
                raise ValueError("const operand needs a value")
        elif not self.ref:
            raise ValueError(f"{self.kind} operand needs a ref")
        if self.kind is OperandKind.PRICE and self.ref not in PRICE_SOURCES:
            raise ValueError(f"price ref must be one of {PRICE_SOURCES}")
        return self

    def label(self) -> str:
        if self.kind is OperandKind.CONST:
            return f"{self.value:g}"
        base = str(self.ref)
        return f"{base}[{self.shift}]" if self.shift else base

    @classmethod
    def ind(cls, alias: str, shift: int = 0) -> Operand:
        return cls(kind=OperandKind.INDICATOR, ref=alias, shift=shift)

    @classmethod
    def price(cls, field_name: str = "close", shift: int = 0) -> Operand:
        return cls(kind=OperandKind.PRICE, ref=field_name, shift=shift)

    @classmethod
    def const(cls, value: float) -> Operand:
        return cls(kind=OperandKind.CONST, value=float(value))


class Condition(BaseModel):
    op: ConditionOp
    left: Operand
    right: Operand | None = None
    right2: Operand | None = None  # BETWEEN upper bound
    lookback: int = Field(3, ge=1, le=50)  # RISING / FALLING window

    @model_validator(mode="after")
    def _check(self) -> Condition:
        if self.op in {ConditionOp.RISING, ConditionOp.FALLING}:
            return self
        if self.right is None:
            raise ValueError(f"{self.op} needs a right operand")
        if self.op is ConditionOp.BETWEEN and self.right2 is None:
            raise ValueError("between needs right2")
        return self

    def label(self) -> str:
        if self.op in {ConditionOp.RISING, ConditionOp.FALLING}:
            return f"{self.left.label()} {self.op}({self.lookback})"
        if self.op is ConditionOp.BETWEEN:
            return f"{self.right.label()} <= {self.left.label()} <= {self.right2.label()}"  # type: ignore[union-attr]
        symbol = {
            ConditionOp.CROSS_ABOVE: "crosses above",
            ConditionOp.CROSS_BELOW: "crosses below",
            ConditionOp.GT: ">",
            ConditionOp.LT: "<",
            ConditionOp.GTE: ">=",
            ConditionOp.LTE: "<=",
        }[self.op]
        return f"{self.left.label()} {symbol} {self.right.label()}"  # type: ignore[union-attr]


class RuleGroup(BaseModel):
    logic: Literal["and", "or"] = "and"
    conditions: list[Condition] = Field(default_factory=list, max_length=8)

    def is_empty(self) -> bool:
        return not self.conditions

    def label(self) -> str:
        joiner = f" {self.logic.upper()} "
        return joiner.join(c.label() for c in self.conditions)


class IndicatorSpec(BaseModel):
    type: str
    params: dict[str, float] = Field(default_factory=dict)
    source: str = "close"

    @field_validator("type")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in REGISTRY:
            raise ValueError(f"unknown indicator '{v}'")
        return v

    @field_validator("source")
    @classmethod
    def _source_ok(cls, v: str) -> str:
        if v not in PRICE_SOURCES:
            raise ValueError(f"source must be one of {PRICE_SOURCES}")
        return v

    @property
    def scale(self) -> str:
        return REGISTRY[self.type].scale

    @property
    def family(self) -> str:
        return REGISTRY[self.type].family


class RiskBlock(BaseModel):
    """How the position is protected and sized. Survivability lives here."""

    stop_kind: Literal["atr", "points", "percent", "none"] = "atr"
    stop_value: float = Field(2.0, gt=0)
    target_kind: Literal["atr", "points", "percent", "rr", "none"] = "rr"
    target_value: float = Field(2.0, gt=0)
    trailing: bool = False
    trail_atr_mult: float = Field(2.0, gt=0)
    breakeven_at_r: float | None = Field(None, ge=0.1, le=5.0)
    risk_per_trade_pct: float = Field(1.0, gt=0, le=5.0)
    max_bars_in_trade: int | None = Field(None, ge=1, le=2000)
    atr_period: int = Field(14, ge=5, le=100)

    def label(self) -> str:
        stop = "none" if self.stop_kind == "none" else f"{self.stop_value:g} {self.stop_kind}"
        target = (
            "none" if self.target_kind == "none" else f"{self.target_value:g} {self.target_kind}"
        )
        extra = []
        if self.trailing:
            extra.append(f"trail {self.trail_atr_mult:g}xATR")
        if self.breakeven_at_r:
            extra.append(f"BE at {self.breakeven_at_r:g}R")
        if self.max_bars_in_trade:
            extra.append(f"time stop {self.max_bars_in_trade} bars")
        suffix = f" ({', '.join(extra)})" if extra else ""
        return f"SL {stop} / TP {target}{suffix}"


class FilterBlock(BaseModel):
    """Gates applied on top of entry rules — regime, session, news, volatility."""

    trend_filter_alias: str | None = None
    trend_filter_mode: Literal["above", "below", "with_slope", "off"] = "off"
    min_atr_pct: float | None = Field(None, ge=0, le=20)
    max_atr_pct: float | None = Field(None, ge=0, le=50)
    allowed_hours: list[int] | None = None  # UTC hours, e.g. [7..16]
    allowed_weekdays: list[int] | None = None  # 0=Mon
    avoid_high_impact_news: bool = False
    news_blackout_minutes: int = Field(30, ge=0, le=240)
    max_positions: int = Field(1, ge=1, le=5)
    cooldown_bars: int = Field(0, ge=0, le=200)

    @field_validator("allowed_hours")
    @classmethod
    def _hours(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return None
        bad = [h for h in v if not 0 <= h <= 23]
        if bad:
            raise ValueError(f"invalid hours: {bad}")
        return sorted(set(v))


class ParamSpec(BaseModel):
    """A knob the tuner is allowed to turn.

    ``path`` is a dotted path into the IR, e.g.
    ``indicators.fast.params.period`` or ``risk.stop_value``.
    """

    path: str
    low: float
    high: float
    step: float | None = None
    is_int: bool = False
    label: str = ""

    @model_validator(mode="after")
    def _range(self) -> ParamSpec:
        if self.high <= self.low:
            raise ValueError(f"param {self.path}: high must exceed low")
        return self


class StrategyIR(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("stg"))
    name: str
    style: str = "unclassified"  # trend_following | mean_reversion | breakout | ...
    asset: str
    timeframe: str = "H1"
    direction: Direction = Direction.BOTH

    indicators: dict[str, IndicatorSpec] = Field(default_factory=dict)
    entry_long: RuleGroup = Field(default_factory=RuleGroup)
    entry_short: RuleGroup = Field(default_factory=RuleGroup)
    exit_long: RuleGroup = Field(default_factory=RuleGroup)
    exit_short: RuleGroup = Field(default_factory=RuleGroup)

    risk: RiskBlock = Field(default_factory=RiskBlock)
    filters: FilterBlock = Field(default_factory=FilterBlock)
    param_space: list[ParamSpec] = Field(default_factory=list)

    # lineage — how the learning loop tracks descent
    generation: int = 0
    parents: list[str] = Field(default_factory=list)
    origin: Literal["fresh", "mutation", "crossover", "manual", "tuned"] = "fresh"
    hypothesis: str = ""
    notes: str = ""

    # ------------------------------------------------------------------ #
    # structural identity
    # ------------------------------------------------------------------ #

    def structural_dict(self) -> dict[str, Any]:
        """Everything that defines behaviour; excludes ids, names, lineage."""
        return {
            "asset": self.asset,
            "timeframe": self.timeframe,
            "direction": str(self.direction),
            "indicators": {
                alias: {"type": spec.type, "params": spec.params, "source": spec.source}
                for alias, spec in sorted(self.indicators.items())
            },
            "entry_long": self.entry_long.model_dump(mode="json"),
            "entry_short": self.entry_short.model_dump(mode="json"),
            "exit_long": self.exit_long.model_dump(mode="json"),
            "exit_short": self.exit_short.model_dump(mode="json"),
            "risk": self.risk.model_dump(mode="json"),
            "filters": self.filters.model_dump(mode="json"),
        }

    def fingerprint(self) -> str:
        """Stable hash used to avoid re-testing the same strategy forever."""
        blob = json.dumps(self.structural_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def shape_fingerprint(self) -> str:
        """Hash of the *shape* only (params ignored) — for novelty checks."""
        shape = {
            "indicators": sorted((s.type, s.source) for s in self.indicators.values()),
            "entry_long": [(c.op, c.left.kind, c.left.ref) for c in self.entry_long.conditions],
            "entry_short": [(c.op, c.left.kind, c.left.ref) for c in self.entry_short.conditions],
            "stop": self.risk.stop_kind,
            "target": self.risk.target_kind,
            "direction": str(self.direction),
        }
        blob = json.dumps(shape, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    # parameter access (used by the tuner)
    # ------------------------------------------------------------------ #

    def get_param(self, path: str) -> Any:
        node: Any = self
        for part in path.split("."):
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        return node

    def set_param(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        last = parts[-1]
        if isinstance(node, dict):
            node[last] = value
        else:
            setattr(node, last, value)

    def with_params(self, values: dict[str, Any]) -> StrategyIR:
        """Return a copy with tuner-chosen parameter values applied."""
        clone = self.model_copy(deep=True)
        for path, value in values.items():
            clone.set_param(path, value)
        return clone

    def current_params(self) -> dict[str, Any]:
        return {spec.path: self.get_param(spec.path) for spec in self.param_space}

    # ------------------------------------------------------------------ #
    # human summary (used by doc_writer and the UI)
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        lines = [
            f"{self.name} [{self.style}] on {self.asset} {self.timeframe} ({self.direction})",
            "Indicators:",
        ]
        for alias, spec in self.indicators.items():
            params = ", ".join(f"{k}={v:g}" for k, v in spec.params.items()) or "default"
            lines.append(f"  - {alias} = {spec.type}({params}) on {spec.source}")
        if not self.entry_long.is_empty():
            lines.append(f"Long entry : {self.entry_long.label()}")
        if not self.entry_short.is_empty():
            lines.append(f"Short entry: {self.entry_short.label()}")
        if not self.exit_long.is_empty():
            lines.append(f"Long exit  : {self.exit_long.label()}")
        if not self.exit_short.is_empty():
            lines.append(f"Short exit : {self.exit_short.label()}")
        lines.append(f"Risk       : {self.risk.label()}")
        if self.filters.trend_filter_mode != "off":
            lines.append(
                f"Filter     : price {self.filters.trend_filter_mode} "
                f"{self.filters.trend_filter_alias}"
            )
        if self.filters.allowed_hours:
            lines.append(f"Session    : UTC hours {self.filters.allowed_hours}")
        if self.filters.avoid_high_impact_news:
            lines.append(f"News       : flat {self.filters.news_blackout_minutes}m around events")
        return "\n".join(lines)

    def trades_long(self) -> bool:
        return self.direction in {Direction.LONG, Direction.BOTH} and not self.entry_long.is_empty()

    def trades_short(self) -> bool:
        return (
            self.direction in {Direction.SHORT, Direction.BOTH} and not self.entry_short.is_empty()
        )
