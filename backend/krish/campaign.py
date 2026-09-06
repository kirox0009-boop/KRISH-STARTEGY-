"""Operator-defined campaigns: you decide what to build, not the factory.

The default behaviour is a conveyor belt - the orchestrator picks assets and
timeframes on rotation and never stops. That produces volume, which is exactly
what the operator did not want.

A campaign is the opposite: a single, explicit brief you fill in from the UI -
these assets, these timeframes, these styles, this target profit factor - and the
factory works only on that until it is done. Every strategy it invents for the
campaign is reviewed against the brief:

    * good enough as-is                     -> delivered
    * has an edge but misses the target     -> deep-tuned, then re-judged
    * no edge at all                        -> dropped

The target profit factor is the operator's dial. It floors at 1.7 (below that a
strategy barely covers costs) and defaults to 2.0.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .assets import universe
from .genome import RECIPES

#: recipe -> the style label a strategy built from it carries
STYLE_OF_RECIPE = {
    "ma_cross": "trend_following",
    "donchian_breakout": "breakout",
    "bb_reversion": "mean_reversion",
    "rsi_pullback": "trend_pullback",
    "macd_momentum": "momentum",
    "zscore_reversion": "mean_reversion",
    "keltner_trend": "volatility_breakout",
    "stoch_reversal": "oscillator_reversal",
    "squeeze_expansion": "squeeze_breakout",
    "roc_trend": "momentum",
    "bos_continuation": "market_structure",
    "fvg_retrace": "fair_value_gap",
    "order_block_reclaim": "order_block",
    "liquidity_sweep": "liquidity_sweep",
    "premium_discount": "premium_discount",
}

MIN_ALLOWED_PF = 1.7
DEFAULT_PF = 2.0


class CampaignState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Campaign:
    """One operator brief. Held in memory; the strategies it produces persist."""

    id: str
    name: str
    assets: list[str]
    timeframes: list[str]
    recipes: list[str]
    target_profit_factor: float = DEFAULT_PF
    max_strategies: int = 40  # stop generating after this many ideas
    state: CampaignState = CampaignState.RUNNING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    generated: int = 0
    delivered: int = 0
    tuned: int = 0
    dropped: int = 0
    strategy_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "assets": self.assets,
            "timeframes": self.timeframes,
            "recipes": self.recipes,
            "target_profit_factor": self.target_profit_factor,
            "max_strategies": self.max_strategies,
            "state": str(self.state),
            "created_at": self.created_at,
            "progress": {
                "generated": self.generated,
                "delivered": self.delivered,
                "tuned": self.tuned,
                "dropped": self.dropped,
                "remaining": max(0, self.max_strategies - self.generated),
            },
            "strategy_ids": self.strategy_ids,
        }

    @property
    def complete(self) -> bool:
        return self.generated >= self.max_strategies


def validate_brief(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn raw UI input into a clean, validated brief or raise ValueError."""
    uni = universe()
    known_assets = set(uni.keys())

    assets = [a.upper() for a in (payload.get("assets") or []) if a.upper() in known_assets]
    if not assets:
        raise ValueError(f"pick at least one asset from {sorted(known_assets)}")

    # union of the chosen assets' own timeframes, so we never ask for a
    # timeframe an asset is not configured for
    allowed_tf: set[str] = set()
    for key in assets:
        allowed_tf.update(uni.get(key).timeframes)
    requested_tf = [t.upper() for t in (payload.get("timeframes") or [])]
    timeframes = [t for t in requested_tf if t in allowed_tf] or sorted(allowed_tf)

    requested_recipes = payload.get("recipes") or []
    recipes = [r for r in requested_recipes if r in RECIPES] or list(RECIPES)

    pf = float(payload.get("target_profit_factor", DEFAULT_PF))
    if pf < MIN_ALLOWED_PF:
        pf = MIN_ALLOWED_PF

    max_strategies = int(payload.get("max_strategies", 40))
    max_strategies = max(1, min(max_strategies, 500))

    return {
        "name": str(payload.get("name") or "Untitled campaign")[:80],
        "assets": assets,
        "timeframes": timeframes,
        "recipes": recipes,
        "target_profit_factor": round(pf, 2),
        "max_strategies": max_strategies,
    }


class CampaignBook:
    """In-memory registry of campaigns, newest first."""

    def __init__(self) -> None:
        self._campaigns: dict[str, Campaign] = {}

    def create(self, brief: dict[str, Any]) -> Campaign:
        cid = f"cmp_{uuid.uuid4().hex[:10]}"
        campaign = Campaign(id=cid, **brief)
        self._campaigns[cid] = campaign
        return campaign

    def get(self, cid: str) -> Campaign | None:
        return self._campaigns.get(cid)

    def all(self) -> list[Campaign]:
        return sorted(self._campaigns.values(), key=lambda c: c.created_at, reverse=True)

    def active(self) -> list[Campaign]:
        return [c for c in self._campaigns.values() if c.state is CampaignState.RUNNING]

    def next_target(self, campaign: Campaign, rng: Any) -> tuple[str, str, str]:
        """Pick the next (asset, timeframe, recipe) to generate for a campaign."""
        return (
            rng.choice(campaign.assets),
            rng.choice(campaign.timeframes),
            rng.choice(campaign.recipes),
        )


_book: CampaignBook | None = None


def book() -> CampaignBook:
    global _book
    if _book is None:
        _book = CampaignBook()
    return _book


def recipe_catalogue() -> list[dict[str, str]]:
    """Everything the UI needs to build its style picker."""
    return [{"recipe": r, "style": STYLE_OF_RECIPE.get(r, r)} for r in RECIPES]


__all__ = [
    "DEFAULT_PF",
    "MIN_ALLOWED_PF",
    "STYLE_OF_RECIPE",
    "Campaign",
    "CampaignBook",
    "CampaignState",
    "book",
    "recipe_catalogue",
    "validate_brief",
]
