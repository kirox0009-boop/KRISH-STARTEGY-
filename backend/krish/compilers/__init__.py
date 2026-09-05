"""IR -> target-language compilers.

The Strategy IR is the single source of truth. Each compiler is a pure function
``StrategyIR -> str`` so a new target (MQL5, cTrader, NinjaTrader) is an additive
change that cannot break the others.

Implemented:
  * ``python_export`` — self-contained reproducible backtest runner
  * ``pine``          — TradingView Pine Script v5 strategy with alert webhooks
  * ``mql5``          — MetaTrader 5 Expert Advisor (.mq5)

Still to come (docs/ROADMAP.md, Phase 7): the MT5 *bridge* that installs and
starts the EA for you. Generating the EA is done; pushing it onto a running
terminal is not.
"""

from .mql5 import Mql5Unsupported, to_mql5
from .pine import PineUnsupported, to_pine
from .python_export import to_python_runner

__all__ = [
    "Mql5Unsupported",
    "PineUnsupported",
    "to_mql5",
    "to_pine",
    "to_python_runner",
]
