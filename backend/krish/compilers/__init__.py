"""IR -> target-language compilers.

The Strategy IR is the single source of truth. Each compiler is a pure function
``StrategyIR -> str`` so a new target (MQL5, cTrader, NinjaTrader) is an additive
change that cannot break the others.

Implemented:
  * ``python_export`` — self-contained reproducible backtest runner
  * ``pine``          — TradingView Pine Script v5 strategy with alert webhooks

Planned (see docs/ROADMAP.md, Phase 5):
  * ``mql5``          — MetaTrader 5 Expert Advisor, with backtest-parity tests
"""

from .pine import PineUnsupported, to_pine
from .python_export import to_python_runner

__all__ = ["PineUnsupported", "to_pine", "to_python_runner"]
