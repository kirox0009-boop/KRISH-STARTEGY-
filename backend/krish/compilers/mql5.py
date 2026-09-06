"""Strategy IR -> MetaTrader 5 Expert Advisor (.mq5).

The third target from the same JSON. Design rules:

* **Use MetaTrader's own indicators** (``iMA``, ``iATR``, ``iBands``, ``iADX`` …)
  wherever one exists, rather than reimplementing maths. Fewer lines, and the
  numbers match what the user sees on their own chart.
* Where MT5's definition differs from this project's Python (MACD signal uses SMA
  in MT5, EMA here), compute it manually so the EA matches the backtest rather
  than the platform default. A silent definition mismatch is how an EA ends up
  behaving nothing like its backtest.
* **Signals are read from the last closed bar** (shift 1) and orders are sent on
  the new bar - exactly the rule the Python engine enforces. No repainting.
* Sizing comes from the account balance, the configured risk percent and the real
  stop distance, using the symbol's own tick value. Never a fixed lot.
* Every generated input is exposed so the strategy can be re-optimised in MT5's
  own Strategy Tester and compared against the Python numbers.
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


class Mql5Unsupported(NotImplementedError):
    """This IR uses something the MQL5 generator cannot express faithfully."""


#: price source -> MT5 applied price constant
APPLIED = {
    "close": "PRICE_CLOSE",
    "open": "PRICE_OPEN",
    "high": "PRICE_HIGH",
    "low": "PRICE_LOW",
    "hl2": "PRICE_MEDIAN",
    "hlc3": "PRICE_TYPICAL",
    "ohlc4": "PRICE_WEIGHTED",
}
SRC_FN = {
    "close": "CloseAt",
    "open": "OpenAt",
    "high": "HighAt",
    "low": "LowAt",
    "hl2": "Hl2At",
    "hlc3": "Hlc3At",
    "ohlc4": "Ohlc4At",
}


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _inp(alias: str, param: str) -> str:
    return f"Inp_{alias}_{param}"


def _fn(alias: str) -> str:
    return f"IND_{alias}"


def _handle(alias: str, suffix: str = "") -> str:
    return f"h_{alias}{suffix}"


# --------------------------------------------------------------------------- #
# per-indicator emitters: (handle creation lines, accessor body lines)
# --------------------------------------------------------------------------- #


def _emit_indicator(alias: str, spec: IndicatorSpec) -> tuple[list[str], list[str]]:
    t = spec.type
    p = {k: _inp(alias, k) for k in spec.params}
    applied = APPLIED[spec.source]
    src = SRC_FN[spec.source]
    h = _handle(alias)
    init: list[str] = []
    body: list[str] = []

    def simple(create: str, buf: int = 0) -> None:
        init.append(f"   {h} = {create};")
        init.append(
            f'   if({h} == INVALID_HANDLE) {{ Print("failed: {alias}"); return INIT_FAILED; }}'
        )
        body.append(f"   return Buf({h}, {buf}, shift);")

    if t in {"sma", "ema", "wma"}:
        method = {"sma": "MODE_SMA", "ema": "MODE_EMA", "wma": "MODE_LWMA"}[t]
        simple(f"iMA(_Symbol, PERIOD_CURRENT, (int){p['period']}, 0, {method}, {applied})")
    elif t == "rsi":
        simple(f"iRSI(_Symbol, PERIOD_CURRENT, (int){p['period']}, {applied})")
    elif t == "atr":
        simple(f"iATR(_Symbol, PERIOD_CURRENT, (int){p['period']})")
    elif t == "atr_pct":
        init.append(f"   {h} = iATR(_Symbol, PERIOD_CURRENT, (int){p['period']});")
        init.append(
            f'   if({h} == INVALID_HANDLE) {{ Print("failed: {alias}"); return INIT_FAILED; }}'
        )
        body += [
            f"   double a = Buf({h}, 0, shift);",
            "   double c = CloseAt(shift);",
            "   if(c == 0.0) return 0.0;",
            "   return a / c * 100.0;",
        ]
    elif t == "stddev":
        simple(f"iStdDev(_Symbol, PERIOD_CURRENT, (int){p['period']}, 0, MODE_SMA, {applied})")
    elif t in {"bb_upper", "bb_lower", "bb_percent"}:
        init.append(
            f"   {h} = iBands(_Symbol, PERIOD_CURRENT, (int){p['period']}, 0, "
            f"{p['mult']}, {applied});"
        )
        init.append(
            f'   if({h} == INVALID_HANDLE) {{ Print("failed: {alias}"); return INIT_FAILED; }}'
        )
        if t == "bb_upper":
            body.append(f"   return Buf({h}, 1, shift);")
        elif t == "bb_lower":
            body.append(f"   return Buf({h}, 2, shift);")
        else:
            body += [
                f"   double up = Buf({h}, 1, shift);",
                f"   double lo = Buf({h}, 2, shift);",
                "   double w  = up - lo;",
                "   if(w <= 0.0) return 50.0;",
                f"   return ({src}(shift) - lo) / w * 100.0;",
            ]
    elif t == "macd":
        _macd_handles(init, alias, p, applied)
        body.append(f"   return MacdLine_{alias}(shift);")
    elif t in {"macd_signal", "macd_hist"}:
        _macd_handles(init, alias, p, applied)
        if t == "macd_signal":
            body.append(f"   return MacdSignal_{alias}(shift);")
        else:
            body.append(f"   return MacdLine_{alias}(shift) - MacdSignal_{alias}(shift);")
    elif t in {"donchian_high", "donchian_low", "donchian_mid"}:
        if t == "donchian_high":
            body.append(f"   return HighestHigh((int){p['period']}, shift);")
        elif t == "donchian_low":
            body.append(f"   return LowestLow((int){p['period']}, shift);")
        else:
            body.append(
                f"   return (HighestHigh((int){p['period']}, shift) + "
                f"LowestLow((int){p['period']}, shift)) / 2.0;"
            )
    elif t == "highest":
        body.append(f'   return HighestOf("{spec.source}", (int){p["period"]}, shift);')
    elif t == "lowest":
        body.append(f'   return LowestOf("{spec.source}", (int){p["period"]}, shift);')
    elif t == "roc":
        body += [
            f"   double a = {src}(shift);",
            f"   double b = {src}(shift + (int){p['period']});",
            "   if(b == 0.0) return 0.0;",
            "   return (a / b - 1.0) * 100.0;",
        ]
    elif t == "momentum":
        body.append(f"   return {src}(shift) - {src}(shift + (int){p['period']});")
    elif t == "stoch_k":
        simple(
            f"iStochastic(_Symbol, PERIOD_CURRENT, (int){p['period']}, 3, "
            f"(int){p['smooth']}, MODE_SMA, STO_LOWHIGH)"
        )
    elif t == "cci":
        simple(f"iCCI(_Symbol, PERIOD_CURRENT, (int){p['period']}, {applied})")
    elif t == "adx":
        simple(f"iADX(_Symbol, PERIOD_CURRENT, (int){p['period']})", buf=0)
    elif t == "zscore":
        init += [
            f"   {h} = iMA(_Symbol, PERIOD_CURRENT, (int){p['period']}, 0, MODE_SMA, {applied});",
            f"   {_handle(alias, '_sd')} = iStdDev(_Symbol, PERIOD_CURRENT, (int){p['period']}, "
            f"0, MODE_SMA, {applied});",
            f"   if({h} == INVALID_HANDLE || {_handle(alias, '_sd')} == INVALID_HANDLE)"
            f' {{ Print("failed: {alias}"); return INIT_FAILED; }}',
        ]
        body += [
            f"   double mean = Buf({h}, 0, shift);",
            f"   double sd   = Buf({_handle(alias, '_sd')}, 0, shift);",
            "   if(sd <= 0.0) return 0.0;",
            f"   return ({src}(shift) - mean) / sd;",
        ]
    elif t in {"keltner_upper", "keltner_lower"}:
        init += [
            f"   {h} = iMA(_Symbol, PERIOD_CURRENT, (int){p['period']}, 0, MODE_EMA, PRICE_CLOSE);",
            f"   {_handle(alias, '_atr')} = iATR(_Symbol, PERIOD_CURRENT, (int){p['period']});",
            f"   if({h} == INVALID_HANDLE || {_handle(alias, '_atr')} == INVALID_HANDLE)"
            f' {{ Print("failed: {alias}"); return INIT_FAILED; }}',
        ]
        sign = "+" if t == "keltner_upper" else "-"
        body.append(
            f"   return Buf({h}, 0, shift) {sign} {p['mult']} * "
            f"Buf({_handle(alias, '_atr')}, 0, shift);"
        )
    elif t == "swing_high_level":
        body.append(f"   return SwingHighLevel((int){p['left']}, (int){p['right']}, shift);")
    elif t == "swing_low_level":
        body.append(f"   return SwingLowLevel((int){p['left']}, (int){p['right']}, shift);")
    elif t == "fvg_bull_level":
        body.append("   return FvgBullLevel(shift);")
    elif t == "fvg_bear_level":
        body.append("   return FvgBearLevel(shift);")
    elif t == "ob_bull_level":
        body.append("   return ObBullLevel(shift);")
    elif t == "ob_bear_level":
        body.append("   return ObBearLevel(shift);")
    elif t == "liquidity_sweep_high":
        body.append(f"   return SweepHigh((int){p['left']}, (int){p['right']}, shift);")
    elif t == "liquidity_sweep_low":
        body.append(f"   return SweepLow((int){p['left']}, (int){p['right']}, shift);")
    elif t == "equilibrium":
        body.append(f"   return Equilibrium((int){p['period']}, shift);")
    elif t == "displacement":
        init.append(f"   {h} = iATR(_Symbol, PERIOD_CURRENT, (int){p['period']});")
        init.append(
            f'   if({h} == INVALID_HANDLE) {{ Print("failed: {alias}"); return INIT_FAILED; }}'
        )
        body += [
            f"   double a = Buf({h}, 0, shift);",
            "   if(a <= 0.0) return 0.0;",
            "   return MathAbs(CloseAt(shift) - OpenAt(shift)) / a;",
        ]
    elif t == "wick_up_pct":
        body.append("   return WickUpPct(shift);")
    elif t == "wick_down_pct":
        body.append("   return WickDownPct(shift);")
    elif t == "vwap":
        raise Mql5Unsupported(
            "session VWAP has no MetaTrader equivalent that matches this project's "
            "definition; regenerate the strategy without it"
        )
    else:
        raise Mql5Unsupported(f"indicator '{t}' is not supported by the MQL5 generator yet")

    return init, body


#: indicator types that need the market-structure helper block
STRUCTURE_TYPES = {
    "swing_high_level",
    "swing_low_level",
    "fvg_bull_level",
    "fvg_bear_level",
    "ob_bull_level",
    "ob_bear_level",
    "liquidity_sweep_high",
    "liquidity_sweep_low",
    "equilibrium",
    "wick_up_pct",
    "wick_down_pct",
}

#: MetaTrader has no built-ins for any of this, so it is implemented directly on
#: the rate arrays. The scans walk BACKWARDS from `shift` into older bars and a
#: pivot is only considered from `shift + right` onwards - the same "you cannot
#: know a swing until `right` bars later" rule the Python side enforces by
#: shifting. Getting that wrong here would make the EA trade signals the backtest
#: could never have seen.
_STRUCTURE_HELPERS = """
//--- market structure helpers ---------------------------------------------
#define KRISH_SCAN 400

double SwingHighLevel(int left, int right, int shift)
{
   for(int i = shift + right; i < shift + right + KRISH_SCAN; i++)
   {
      double h = HighAt(i);
      if(h <= 0.0) break;
      bool pivot = true;
      for(int k = 1; k <= left  && pivot; k++) if(HighAt(i + k) > h) pivot = false;
      for(int k = 1; k <= right && pivot; k++) if(HighAt(i - k) > h) pivot = false;
      if(pivot) return h;
   }
   return 0.0;
}

double SwingLowLevel(int left, int right, int shift)
{
   for(int i = shift + right; i < shift + right + KRISH_SCAN; i++)
   {
      double l = LowAt(i);
      if(l <= 0.0) break;
      bool pivot = true;
      for(int k = 1; k <= left  && pivot; k++) if(LowAt(i + k) < l) pivot = false;
      for(int k = 1; k <= right && pivot; k++) if(LowAt(i - k) < l) pivot = false;
      if(pivot) return l;
   }
   return 0.0;
}

double FvgBullLevel(int shift)
{
   for(int i = shift; i < shift + KRISH_SCAN; i++)
   {
      double lo = LowAt(i), h2 = HighAt(i + 2);
      if(lo <= 0.0 || h2 <= 0.0) break;
      if(lo > h2) return (h2 + lo) / 2.0;
   }
   return 0.0;
}

double FvgBearLevel(int shift)
{
   for(int i = shift; i < shift + KRISH_SCAN; i++)
   {
      double hi = HighAt(i), l2 = LowAt(i + 2);
      if(hi <= 0.0 || l2 <= 0.0) break;
      if(hi < l2) return (l2 + hi) / 2.0;
   }
   return 0.0;
}

double ObBullLevel(int shift)
{
   for(int i = shift; i < shift + KRISH_SCAN; i++)
   {
      double c = CloseAt(i);
      if(c <= 0.0) break;
      if(c > HighAt(i + 1) && CloseAt(i + 1) < OpenAt(i + 1)) return LowAt(i + 1);
   }
   return 0.0;
}

double ObBearLevel(int shift)
{
   for(int i = shift; i < shift + KRISH_SCAN; i++)
   {
      double c = CloseAt(i);
      if(c <= 0.0) break;
      if(c < LowAt(i + 1) && CloseAt(i + 1) > OpenAt(i + 1)) return HighAt(i + 1);
   }
   return 0.0;
}

double SweepHigh(int left, int right, int shift)
{
   double lvl = SwingHighLevel(left, right, shift);
   if(lvl <= 0.0) return 0.0;
   return (HighAt(shift) > lvl && CloseAt(shift) < lvl) ? 100.0 : 0.0;
}

double SweepLow(int left, int right, int shift)
{
   double lvl = SwingLowLevel(left, right, shift);
   if(lvl <= 0.0) return 0.0;
   return (LowAt(shift) < lvl && CloseAt(shift) > lvl) ? 100.0 : 0.0;
}

double Equilibrium(int period, int shift)
{
   return (HighestHigh(period, shift) + LowestLow(period, shift)) / 2.0;
}

double WickUpPct(int shift)
{
   double h = HighAt(shift), l = LowAt(shift);
   double rng = h - l;
   if(rng <= 0.0) return 0.0;
   return (h - MathMax(OpenAt(shift), CloseAt(shift))) / rng * 100.0;
}

double WickDownPct(int shift)
{
   double h = HighAt(shift), l = LowAt(shift);
   double rng = h - l;
   if(rng <= 0.0) return 0.0;
   return (MathMin(OpenAt(shift), CloseAt(shift)) - l) / rng * 100.0;
}
"""


def _macd_handles(init: list[str], alias: str, p: dict[str, str], applied: str) -> None:
    fast, slow = _handle(alias, "_f"), _handle(alias, "_s")
    init += [
        f"   {fast} = iMA(_Symbol, PERIOD_CURRENT, (int){p['fast']}, 0, MODE_EMA, {applied});",
        f"   {slow} = iMA(_Symbol, PERIOD_CURRENT, (int){p['slow']}, 0, MODE_EMA, {applied});",
        f"   if({fast} == INVALID_HANDLE || {slow} == INVALID_HANDLE)"
        f' {{ Print("failed: {alias}"); return INIT_FAILED; }}',
    ]


def _macd_helpers(alias: str, spec: IndicatorSpec) -> list[str]:
    """MT5's MACD signal line uses SMA; this project uses EMA. Match the project."""
    fast, slow = _handle(alias, "_f"), _handle(alias, "_s")
    signal = _inp(alias, "signal") if "signal" in spec.params else "9"
    return [
        f"double MacdLine_{alias}(int shift)",
        "{",
        f"   return Buf({fast}, 0, shift) - Buf({slow}, 0, shift);",
        "}",
        "",
        f"double MacdSignal_{alias}(int shift)",
        "{",
        f"   int n = (int){signal};",
        "   int warm = n * 5;",
        "   double k = 2.0 / (n + 1.0);",
        f"   double ema = MacdLine_{alias}(shift + warm);",
        "   for(int i = shift + warm - 1; i >= shift; i--)",
        f"      ema = MacdLine_{alias}(i) * k + ema * (1.0 - k);",
        "   return ema;",
        "}",
        "",
    ]


# --------------------------------------------------------------------------- #
# conditions
# --------------------------------------------------------------------------- #

BASE_SHIFT = 1  # signals are read from the last CLOSED bar


def _operand(op: Operand, extra: int = 0) -> str:
    shift = BASE_SHIFT + op.shift + extra
    if op.kind is OperandKind.CONST:
        return _num(float(op.value or 0.0))
    if op.kind is OperandKind.PRICE:
        return f"{SRC_FN[str(op.ref)]}({shift})"
    return f"{_fn(str(op.ref))}({shift})"


def _condition(cond: Condition) -> str:
    left = _operand(cond.left)
    if cond.op in {ConditionOp.RISING, ConditionOp.FALLING}:
        cmp = ">" if cond.op is ConditionOp.RISING else "<"
        parts = [
            f"({_operand(cond.left, i)} {cmp} {_operand(cond.left, i + 1)})"
            for i in range(cond.lookback)
        ]
        return "(" + " && ".join(parts) + ")"

    right = _operand(cond.right)  # type: ignore[arg-type]
    match cond.op:
        case ConditionOp.GT:
            return f"({left} > {right})"
        case ConditionOp.LT:
            return f"({left} < {right})"
        case ConditionOp.GTE:
            return f"({left} >= {right})"
        case ConditionOp.LTE:
            return f"({left} <= {right})"
        case ConditionOp.CROSS_ABOVE:
            return (
                f"({left} > {right} && {_operand(cond.left, 1)} <= "
                f"{_operand(cond.right, 1)})"  # type: ignore[arg-type]
            )
        case ConditionOp.CROSS_BELOW:
            return (
                f"({left} < {right} && {_operand(cond.left, 1)} >= "
                f"{_operand(cond.right, 1)})"  # type: ignore[arg-type]
            )
        case ConditionOp.BETWEEN:
            upper = _operand(cond.right2)  # type: ignore[arg-type]
            return f"({left} >= MathMin({right}, {upper}) && {left} <= MathMax({right}, {upper}))"
    raise Mql5Unsupported(f"operator '{cond.op}' is not translatable to MQL5")


def _group(group: RuleGroup, default: str = "false") -> str:
    if group.is_empty():
        return default
    joiner = " && " if group.logic == "and" else " || "
    return joiner.join(_condition(c) for c in group.conditions)


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #


def to_mql5(ir: StrategyIR, *, magic: int | None = None) -> str:
    L: list[str] = []
    add = L.append
    safe = ir.name.replace('"', "'")
    magic = magic or (abs(hash(ir.id)) % 900_000_000 + 10_000_000)

    add("//+------------------------------------------------------------------+")
    add(f"//| {safe}")
    add(f"//| Generated by KRISH from Strategy IR {ir.id}")
    add(f"//| {ir.asset} {ir.timeframe} | style: {ir.style} | generation {ir.generation}")
    add("//|")
    add("//| MACHINE-GENERATED. Edit the IR and regenerate; do not patch this file.")
    add("//| Run on a DEMO account first and compare the Strategy Tester result")
    add("//| against REPORT.md before it goes anywhere near real money.")
    add("//+------------------------------------------------------------------+")
    add('#property copyright "KRISH"')
    add('#property version   "1.00"')
    add(f'#property description "{safe} - {ir.asset} {ir.timeframe} ({ir.style})"')
    add("")
    add("#include <Trade\\Trade.mqh>")
    add("#include <Trade\\PositionInfo.mqh>")
    add("")

    # ---------------- inputs ----------------
    add('input group             "=== Execution ==="')
    add(f"input ulong  InpMagic          = {magic};      // magic number")
    add("input int    InpSlippagePoints = 20;            // max deviation (points)")
    add("input bool   InpNewBarOnly     = true;          // act once per closed bar")
    add("")
    add('input group             "=== Risk ==="')
    add(
        f"input double InpRiskPercent    = {_num(ir.risk.risk_per_trade_pct)};"
        "            // % of balance risked per trade"
    )
    add(f"input int    InpAtrPeriod      = {ir.risk.atr_period};            // ATR period for risk")
    add(
        f"input double InpStopValue      = {_num(ir.risk.stop_value)};"
        f"            // stop ({ir.risk.stop_kind})"
    )
    add(
        f"input double InpTargetValue    = {_num(ir.risk.target_value)};"
        f"            // target ({ir.risk.target_kind})"
    )
    add(
        f"input bool   InpTrailing       = {str(ir.risk.trailing).lower()};"
        "         // trailing stop"
    )
    add(
        f"input double InpTrailMult      = {_num(ir.risk.trail_atr_mult)};            // trail xATR"
    )
    add(
        f"input double InpBreakevenR     = {_num(ir.risk.breakeven_at_r or 0)};"
        "            // move to breakeven at R (0 = off)"
    )
    add(
        f"input int    InpMaxBarsInTrade = {ir.risk.max_bars_in_trade or 0};"
        "            // time stop in bars (0 = off)"
    )
    add(
        f"input int    InpCooldownBars   = {ir.filters.cooldown_bars};            // bars to wait after a close"
    )
    add("")
    if ir.indicators:
        add('input group             "=== Indicators ==="')
        for alias, spec in ir.indicators.items():
            meta = REGISTRY[spec.type]
            for param, value in spec.params.items():
                is_int = meta.params.get(param, (0, 0, True))[2]
                kind = "int   " if is_int else "double"
                val = str(int(value)) if is_int else _num(value)
                add(
                    f"input {kind} {_inp(alias, param):<16} = {val};"
                    f"            // {alias}.{param} ({spec.type})"
                )
        add("")

    # ---------------- globals ----------------
    add("CTrade        trade;")
    add("CPositionInfo pos;")
    add("")
    add("int h_atr_risk = INVALID_HANDLE;")
    handles: list[str] = []
    for alias, spec in ir.indicators.items():
        handles.append(_handle(alias))
        if spec.type == "zscore":
            handles.append(_handle(alias, "_sd"))
        if spec.type in {"keltner_upper", "keltner_lower"}:
            handles.append(_handle(alias, "_atr"))
        if spec.type in {"macd", "macd_signal", "macd_hist"}:
            handles += [_handle(alias, "_f"), _handle(alias, "_s")]
    for name in dict.fromkeys(handles):
        add(f"int {name} = INVALID_HANDLE;")
    add("")
    add("datetime g_lastBar   = 0;")
    add("datetime g_lastClose = 0;")
    add("")

    # ---------------- price helpers ----------------
    add("//--- price + buffer helpers ------------------------------------------")
    add("double Buf(int handle, int buffer, int shift)")
    add("{")
    add("   double v[];")
    add("   if(CopyBuffer(handle, buffer, shift, 1, v) < 1) return 0.0;")
    add("   return v[0];")
    add("}")
    add("")
    for name, fn in (
        ("CopyClose", "CloseAt"),
        ("CopyOpen", "OpenAt"),
        ("CopyHigh", "HighAt"),
        ("CopyLow", "LowAt"),
    ):
        add(f"double {fn}(int shift)")
        add("{")
        add("   double v[];")
        add(f"   if({name}(_Symbol, PERIOD_CURRENT, shift, 1, v) < 1) return 0.0;")
        add("   return v[0];")
        add("}")
        add("")
    add("double Hl2At(int shift)   { return (HighAt(shift) + LowAt(shift)) / 2.0; }")
    add(
        "double Hlc3At(int shift)  { return (HighAt(shift) + LowAt(shift) + CloseAt(shift)) / 3.0; }"
    )
    add(
        "double Ohlc4At(int shift) { return (OpenAt(shift) + HighAt(shift) + LowAt(shift) + CloseAt(shift)) / 4.0; }"
    )
    add("")
    add("double HighestHigh(int period, int shift)")
    add("{")
    add("   double v[];")
    add("   if(CopyHigh(_Symbol, PERIOD_CURRENT, shift, period, v) < period) return 0.0;")
    add("   return v[ArrayMaximum(v)];")
    add("}")
    add("")
    add("double LowestLow(int period, int shift)")
    add("{")
    add("   double v[];")
    add("   if(CopyLow(_Symbol, PERIOD_CURRENT, shift, period, v) < period) return 0.0;")
    add("   return v[ArrayMinimum(v)];")
    add("}")
    add("")
    add("double HighestOf(string src, int period, int shift)")
    add("{")
    add('   if(src == "high") return HighestHigh(period, shift);')
    add("   double v[];")
    add("   if(CopyClose(_Symbol, PERIOD_CURRENT, shift, period, v) < period) return 0.0;")
    add("   return v[ArrayMaximum(v)];")
    add("}")
    add("")
    add("double LowestOf(string src, int period, int shift)")
    add("{")
    add('   if(src == "low") return LowestLow(period, shift);')
    add("   double v[];")
    add("   if(CopyClose(_Symbol, PERIOD_CURRENT, shift, period, v) < period) return 0.0;")
    add("   return v[ArrayMinimum(v)];")
    add("}")
    add("")

    # Only emitted when the strategy actually uses market structure, so a simple
    # moving-average EA does not carry 120 lines it never calls.
    if any(spec.type in STRUCTURE_TYPES for spec in ir.indicators.values()):
        L.extend(_STRUCTURE_HELPERS.splitlines())
        add("")

    # ---------------- indicator accessors ----------------
    init_lines: list[str] = []
    for alias, spec in ir.indicators.items():
        init, body = _emit_indicator(alias, spec)
        init_lines += init
        if spec.type in {"macd", "macd_signal", "macd_hist"}:
            L.extend(_macd_helpers(alias, spec))
        add(f"double {_fn(alias)}(int shift)")
        add("{")
        L.extend(body)
        add("}")
        add("")

    # ---------------- OnInit ----------------
    add("//--- lifecycle ------------------------------------------------------")
    add("int OnInit()")
    add("{")
    add("   trade.SetExpertMagicNumber(InpMagic);")
    add("   trade.SetDeviationInPoints(InpSlippagePoints);")
    add("   trade.SetTypeFillingBySymbol(_Symbol);")
    add("   h_atr_risk = iATR(_Symbol, PERIOD_CURRENT, InpAtrPeriod);")
    add('   if(h_atr_risk == INVALID_HANDLE) { Print("failed: risk ATR"); return INIT_FAILED; }')
    L.extend(init_lines)
    add(
        f'   PrintFormat("KRISH EA loaded: {safe} | %s %s", _Symbol, EnumToString(PERIOD_CURRENT));'
    )
    add("   return INIT_SUCCEEDED;")
    add("}")
    add("")
    add("void OnDeinit(const int reason) { }")
    add("")

    # ---------------- filters ----------------
    add("//--- filters --------------------------------------------------------")
    add("bool SessionOk()")
    add("{")
    if ir.filters.allowed_hours or ir.filters.allowed_weekdays is not None:
        add("   // NOTE: this uses BROKER SERVER TIME, which is usually not UTC.")
        add("   // The backtest filtered on UTC hours, so check your broker's offset")
        add("   // and shift these numbers if the two disagree.")
        add("   MqlDateTime t;")
        add("   TimeToStruct(TimeCurrent(), t);")
        if ir.filters.allowed_hours:
            hours = ", ".join(str(h) for h in ir.filters.allowed_hours)
            add(f"   int hours[] = {{{hours}}};")
            add("   bool okHour = false;")
            add(
                "   for(int i = 0; i < ArraySize(hours); i++) if(t.hour == hours[i]) okHour = true;"
            )
            add("   if(!okHour) return false;")
        if ir.filters.allowed_weekdays is not None:
            days = ", ".join(str((d + 1) % 7) for d in ir.filters.allowed_weekdays)
            add(f"   int days[] = {{{days}}};")
            add("   bool okDay = false;")
            add(
                "   for(int i = 0; i < ArraySize(days); i++) if(t.day_of_week == days[i]) okDay = true;"
            )
            add("   if(!okDay) return false;")
    add("   return true;")
    add("}")
    add("")
    add("bool VolatilityOk()")
    add("{")
    if ir.filters.min_atr_pct is not None or ir.filters.max_atr_pct is not None:
        add("   double c = CloseAt(1);")
        add("   if(c == 0.0) return false;")
        add("   double ap = Buf(h_atr_risk, 0, 1) / c * 100.0;")
        if ir.filters.min_atr_pct is not None:
            add(f"   if(ap < {_num(ir.filters.min_atr_pct)}) return false;")
        if ir.filters.max_atr_pct is not None:
            add(f"   if(ap > {_num(ir.filters.max_atr_pct)}) return false;")
    add("   return true;")
    add("}")
    add("")

    trend_long, trend_short = "true", "true"
    if ir.filters.trend_filter_mode != "off" and ir.filters.trend_filter_alias:
        line = f"{_fn(ir.filters.trend_filter_alias)}(1)"
        prev = f"{_fn(ir.filters.trend_filter_alias)}(2)"
        if ir.filters.trend_filter_mode == "with_slope":
            trend_long, trend_short = f"({line} > {prev})", f"({line} < {prev})"
        elif ir.filters.trend_filter_mode == "above":
            trend_long, trend_short = f"(CloseAt(1) > {line})", f"(CloseAt(1) < {line})"
        else:
            trend_long, trend_short = f"(CloseAt(1) < {line})", f"(CloseAt(1) > {line})"
    add(f"bool TrendOkLong()  {{ return {trend_long}; }}")
    add(f"bool TrendOkShort() {{ return {trend_short}; }}")
    add("")

    # ---------------- signals ----------------
    add("//--- signals (read from the last CLOSED bar) -------------------------")
    add(f"bool EntryLong()  {{ return {_group(ir.entry_long) if ir.trades_long() else 'false'}; }}")
    add(
        f"bool EntryShort() {{ return {_group(ir.entry_short) if ir.trades_short() else 'false'}; }}"
    )
    add(f"bool ExitLong()   {{ return {_group(ir.exit_long)}; }}")
    add(f"bool ExitShort()  {{ return {_group(ir.exit_short)}; }}")
    add("")

    # ---------------- risk maths ----------------
    add("//--- risk -----------------------------------------------------------")
    add("double StopDistance()")
    add("{")
    if ir.risk.stop_kind == "atr":
        add("   return Buf(h_atr_risk, 0, 1) * InpStopValue;")
    elif ir.risk.stop_kind == "percent":
        add("   return CloseAt(1) * InpStopValue / 100.0;")
    elif ir.risk.stop_kind == "points":
        add("   return InpStopValue * _Point;")
    else:
        add("   // IR has no stop; ATR*3 is used for sizing and R maths only")
        add("   return Buf(h_atr_risk, 0, 1) * 3.0;")
    add("}")
    add("")
    add("double TargetDistance(double stopDist)")
    add("{")
    if ir.risk.target_kind == "rr":
        add("   return stopDist * InpTargetValue;")
    elif ir.risk.target_kind == "atr":
        add("   return Buf(h_atr_risk, 0, 1) * InpTargetValue;")
    elif ir.risk.target_kind == "percent":
        add("   return CloseAt(1) * InpTargetValue / 100.0;")
    elif ir.risk.target_kind == "points":
        add("   return InpTargetValue * _Point;")
    else:
        add("   return 0.0;")
    add("}")
    add("")
    add("double LotsFor(double stopDist)")
    add("{")
    add("   if(stopDist <= 0.0) return 0.0;")
    add("   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);")
    add("   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);")
    add("   if(tickSize <= 0.0 || tickValue <= 0.0) return 0.0;")
    add("   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;")
    add("   double lossPerLot = (stopDist / tickSize) * tickValue;")
    add("   if(lossPerLot <= 0.0) return 0.0;")
    add("   double lots = riskMoney / lossPerLot;")
    add("   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);")
    add("   double minL = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);")
    add("   double maxL = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);")
    add("   if(step > 0.0) lots = MathFloor(lots / step) * step;")
    add("   lots = MathMax(minL, MathMin(maxL, lots));")
    add("   if(lots < minL) return 0.0;")
    add("   return NormalizeDouble(lots, 2);")
    add("}")
    add("")

    # ---------------- position management ----------------
    add("bool HasPosition()")
    add("{")
    add("   for(int i = PositionsTotal() - 1; i >= 0; i--)")
    add("      if(pos.SelectByIndex(i) && pos.Symbol() == _Symbol && pos.Magic() == InpMagic)")
    add("         return true;")
    add("   return false;")
    add("}")
    add("")
    add("void ManagePosition()")
    add("{")
    add("   if(!HasPosition()) return;")
    add("   bool isLong = (pos.PositionType() == POSITION_TYPE_BUY);")
    add("   double entry = pos.PriceOpen();")
    add("   double sl    = pos.StopLoss();")
    add("   double tp    = pos.TakeProfit();")
    add("   double price = isLong ? SymbolInfoDouble(_Symbol, SYMBOL_BID)")
    add("                         : SymbolInfoDouble(_Symbol, SYMBOL_ASK);")
    add("   double atr   = Buf(h_atr_risk, 0, 1);")
    add("   double risk  = MathAbs(entry - sl);")
    add("   double newSl = sl;")
    add("")
    add("   // rule-based exit takes priority over stop management")
    add("   if((isLong && ExitLong()) || (!isLong && ExitShort()))")
    add("   {")
    add("      trade.PositionClose(pos.Ticket());")
    add("      g_lastClose = TimeCurrent();")
    add("      return;")
    add("   }")
    add("")
    if ir.risk.max_bars_in_trade:
        add("   if(InpMaxBarsInTrade > 0)")
        add("   {")
        add("      long held = (long)(TimeCurrent() - pos.Time()) / PeriodSeconds(PERIOD_CURRENT);")
        add("      if(held >= InpMaxBarsInTrade)")
        add("      {")
        add("         trade.PositionClose(pos.Ticket());")
        add("         g_lastClose = TimeCurrent();")
        add("         return;")
        add("      }")
        add("   }")
        add("")
    add("   if(InpBreakevenR > 0.0 && risk > 0.0)")
    add("   {")
    add("      double gained = isLong ? (price - entry) : (entry - price);")
    add("      if(gained >= risk * InpBreakevenR)")
    add("         newSl = isLong ? MathMax(newSl, entry) : MathMin(newSl, entry);")
    add("   }")
    add("   if(InpTrailing && atr > 0.0)")
    add("   {")
    add("      double trail = isLong ? price - atr * InpTrailMult : price + atr * InpTrailMult;")
    add("      newSl = isLong ? MathMax(newSl, trail) : MathMin(newSl, trail);")
    add("   }")
    add("   // stops only ever move in our favour, and only if the move is meaningful")
    add("   if(MathAbs(newSl - sl) > _Point && newSl > 0.0)")
    add("      trade.PositionModify(pos.Ticket(), NormalizeDouble(newSl, _Digits),")
    add("                           NormalizeDouble(tp, _Digits));")
    add("}")
    add("")

    # ---------------- entries ----------------
    add("void TryEntries()")
    add("{")
    add("   if(InpCooldownBars > 0 && g_lastClose > 0)")
    add("   {")
    add("      long since = (long)(TimeCurrent() - g_lastClose) / PeriodSeconds(PERIOD_CURRENT);")
    add("      if(since < InpCooldownBars) return;")
    add("   }")
    add("   if(!SessionOk() || !VolatilityOk()) return;")
    add("")
    add("   bool wantLong  = EntryLong()  && TrendOkLong();")
    add("   bool wantShort = EntryShort() && TrendOkShort();")
    add("   if(!wantLong && !wantShort) return;")
    add("")
    add("   double stopDist = StopDistance();")
    add("   double lots     = LotsFor(stopDist);")
    add("   if(lots <= 0.0 || stopDist <= 0.0)")
    add("   {")
    add('      Print("KRISH: skipping entry - lot size resolved to zero");')
    add("      return;")
    add("   }")
    add("   double targetDist = TargetDistance(stopDist);")
    add("")
    add("   if(wantLong)")
    add("   {")
    add("      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);")
    stop_none = ir.risk.stop_kind == "none"
    add(f"      double sl = {'0.0' if stop_none else 'NormalizeDouble(ask - stopDist, _Digits)'};")
    add("      double tp = targetDist > 0.0 ? NormalizeDouble(ask + targetDist, _Digits) : 0.0;")
    add(f'      trade.Buy(lots, _Symbol, 0.0, sl, tp, "{safe}");')
    add("   }")
    add("   else if(wantShort)")
    add("   {")
    add("      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);")
    add(f"      double sl = {'0.0' if stop_none else 'NormalizeDouble(bid + stopDist, _Digits)'};")
    add("      double tp = targetDist > 0.0 ? NormalizeDouble(bid - targetDist, _Digits) : 0.0;")
    add(f'      trade.Sell(lots, _Symbol, 0.0, sl, tp, "{safe}");')
    add("   }")
    add("}")
    add("")

    # ---------------- OnTick ----------------
    add("void OnTick()")
    add("{")
    add("   // One decision per closed bar: this is what keeps the EA aligned with")
    add("   // the backtest, which also decided on close and acted on the next bar.")
    add("   if(InpNewBarOnly)")
    add("   {")
    add("      datetime t[];")
    add("      if(CopyTime(_Symbol, PERIOD_CURRENT, 0, 1, t) < 1) return;")
    add("      if(t[0] == g_lastBar) return;")
    add("      g_lastBar = t[0];")
    add("   }")
    add("")
    add("   if(Bars(_Symbol, PERIOD_CURRENT) < 300) return;")
    add("")
    add("   if(HasPosition()) ManagePosition();")
    add("   else              TryEntries();")
    add("}")
    add("")
    add("//--- strategy summary ------------------------------------------------")
    for line in ir.describe().splitlines():
        add(f"// {line}")

    return "\n".join(L) + "\n"


__all__ = ["Mql5Unsupported", "to_mql5"]
