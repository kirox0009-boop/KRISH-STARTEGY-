"""What *kind* of strategy is this?

"Profit factor 2.1" tells you nothing about whether you will be watching charts
all day or checking in once a week. The same rules on M15 and on D1 are different
products with different demands, so every delivered strategy is labelled with the
trading horizon measured from its own trade log - not guessed from its recipe.
"""

from __future__ import annotations

from typing import Any

from .assets import TIMEFRAME_MINUTES

#: upper bound of average holding time, in hours, for each horizon
HORIZONS: tuple[tuple[str, float, str], ...] = (
    ("scalping", 2.0, "minutes to a couple of hours per trade"),
    ("intraday", 24.0, "opened and closed within the day"),
    ("swing", 24.0 * 10, "held for days, through overnight gaps"),
    ("position", float("inf"), "held for weeks or months"),
)

#: how the entry logic makes its money, derived from the recipe family
FAMILY_LABELS = {
    "trend_following": "trend following",
    "trend_pullback": "trend pullback",
    "breakout": "breakout",
    "squeeze_breakout": "volatility squeeze breakout",
    "volatility_breakout": "volatility breakout",
    "mean_reversion": "mean reversion",
    "oscillator_reversal": "oscillator reversal",
    "momentum": "momentum",
}


def horizon_for(avg_hold_hours: float) -> tuple[str, str]:
    for name, upper, blurb in HORIZONS:
        if avg_hold_hours <= upper:
            return name, blurb
    return "position", HORIZONS[-1][2]


def classify(ir: Any, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Label a strategy from its IR and its realised trade statistics."""
    metrics = metrics or {}
    tf = str(getattr(ir, "timeframe", "H1")).upper()
    minutes = TIMEFRAME_MINUTES.get(tf, 60)

    avg_bars = float(metrics.get("avg_bars_held", 0) or 0)
    hold_hours = round(avg_bars * minutes / 60.0, 2)
    horizon, blurb = horizon_for(hold_hours) if avg_bars > 0 else ("unknown", "no trades recorded")

    style = str(getattr(ir, "style", "") or "unclassified")
    family = FAMILY_LABELS.get(style.split("+")[0], style.replace("_", " "))

    trades_per_year = float(metrics.get("trades_per_year", 0) or 0)
    if trades_per_year >= 500:
        cadence = "very active"
    elif trades_per_year >= 150:
        cadence = "active"
    elif trades_per_year >= 40:
        cadence = "selective"
    else:
        cadence = "rare"

    direction = str(getattr(ir, "direction", "both"))
    side = {"long": "long only", "short": "short only", "both": "both directions"}.get(
        direction, direction
    )

    risk = getattr(ir, "risk", None)
    exit_style = "unknown"
    if risk is not None:
        if getattr(risk, "trailing", False):
            exit_style = "trailing stop"
        elif getattr(risk, "target_kind", "none") != "none":
            exit_style = "fixed target"
        elif getattr(risk, "max_bars_in_trade", None):
            exit_style = "time exit"

    label = f"{horizon.title()} {family}" if horizon != "unknown" else family.title()

    return {
        "horizon": horizon,
        "horizon_note": blurb,
        "label": label,
        "family": family,
        "style": style,
        "timeframe": tf,
        "avg_hold_bars": round(avg_bars, 1),
        "avg_hold_hours": hold_hours,
        "avg_hold_readable": _readable_hours(hold_hours) if avg_bars > 0 else "—",
        "trades_per_year": round(trades_per_year, 1),
        "cadence": cadence,
        "side": side,
        "exit_style": exit_style,
        "summary": (
            f"{label}, {side}, {cadence} ({round(trades_per_year)} trades/year), "
            f"average hold {_readable_hours(hold_hours)}, exits via {exit_style}."
            if avg_bars > 0
            else f"{label}, {side} — not enough trades to characterise."
        ),
    }


def _readable_hours(hours: float) -> str:
    if hours < 1:
        return f"{round(hours * 60)} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


__all__ = ["HORIZONS", "classify", "horizon_for"]
