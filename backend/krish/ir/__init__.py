"""Strategy IR — the single source of truth for every strategy.

One JSON document describes a strategy completely. From it we compile:
    Python signals  (backtesting, tuning)
    MQL5 EA         (MT5 live automation)
    Pine Script     (TradingView alerts)

Because strategies are data, agents can invent, mutate and crossbreed them
without writing code, and the system can learn which *structures* work.
"""

from .compiler import CompiledStrategy, IRCompileError, compile_ir
from .schema import (
    Condition,
    ConditionOp,
    Direction,
    FilterBlock,
    IndicatorSpec,
    Operand,
    OperandKind,
    ParamSpec,
    RiskBlock,
    RuleGroup,
    StrategyIR,
)
from .validate import IRValidationError, validate_ir

__all__ = [
    "CompiledStrategy",
    "Condition",
    "ConditionOp",
    "Direction",
    "FilterBlock",
    "IRCompileError",
    "IRValidationError",
    "IndicatorSpec",
    "Operand",
    "OperandKind",
    "ParamSpec",
    "RiskBlock",
    "RuleGroup",
    "StrategyIR",
    "compile_ir",
    "validate_ir",
]
