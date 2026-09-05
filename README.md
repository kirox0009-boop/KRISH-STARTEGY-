# KRISH

An autonomous **strategy factory**: a squad of AI agents that research, build,
backtest, tune, stress-test, judge, document and deliver trading strategies —
continuously, 24/7, on your VPS, without being told what to do next.

It is not a trading bot. Its product is *validated strategies*, shipped as source
code you can run on MetaTrader 5, TradingView, or Python.

```
 MarketData ─┐                                     ┌─► Packager ─► Delivery
 MacroEcon ──┤                                     │   (ZIP → Drive + Telegram)
 Researcher ─┴─► Architect ─► Developer ─► Tester ──┤
        ▲            ▲            │          │      └─► Rejected
        │            │            └──fix─────┘
        │            │                       ▼
        │            │            Tuner ─► Robustness ─► Risk ─► Judge
        │            │                                            │
        └────────────┴──────────── Memory / Learning ◄─────────────┘
```

## The idea that makes it work

Every strategy is a single JSON document — the **Strategy IR**. From that one
document KRISH compiles Python (for backtesting), Pine Script (TradingView), and
MQL5 (MT5, Phase 5).

Because strategies are *data*, agents can invent them, mutate them and crossbreed
them without writing code, and the system can learn which **structures** work
rather than just which numbers happened to fit. That is why it never runs out of
new strategies to try, and why "now automate this on MT5" is a compile step
instead of a rewrite.

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design,
[`docs/AGENTS.md`](docs/AGENTS.md) for every agent's job, and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what is built and what is next.

## Status: Phase 0 complete and running

Working today, verified end to end on real GOLD H1 data:

- 16 agent types (25 worker instances) on a message bus with a shared blackboard,
  heartbeats, pause/resume/kill, and crash-restart supervision
- Strategy IR + 26-indicator library + IR→Python compiler with a look-ahead audit
- event-driven backtester: real spread/slippage/commission per asset, in-sample
  vs out-of-sample split, walk-forward, Monte Carlo, cost stress, parameter
  sensitivity
- 10 structural strategy recipes plus mutation and crossover, with duplicate
  detection so the same idea is never re-tested
- a judge that produces PASS / BORDERLINE / REJECT with written reasoning and an
  explicit long-term-viability call
- automatic packaging into a named ZIP (source, IR, Pine Script, reproducible
  runner, full reports, trade and equity CSVs, checksum) and delivery
- live control room at `http://localhost:8000` — watch every agent, inspect any
  strategy, and drive the factory from the browser

A representative run: 8 ideas → 6 rejected, 1 borderline (auto-packaged and
delivered), 1 filtered before testing. **That ratio is the point.** The factory's
value is how ruthlessly it discards, not how much it produces.

## Quick start (local)

```bash
make install          # needs python3.11+
make run              # → http://localhost:8000
```

Or run exactly one cycle and watch it happen in the terminal:

```bash
make cycle ASSET=GOLD TF=H1 N=4
```

Useful commands:

```bash
make roster                                  # who the agents are
make backtest FILE=var/packages/.../strategy.json
```

## Deploy on your VPS (24/7)

```bash
git clone <this repo> /opt/krish && cd /opt/krish
cp .env.example .env && $EDITOR .env         # optional keys; nothing is mandatory
docker compose up -d
make logs
```

Then open `http://<vps-ip>:8000`. Redis carries the bus, Postgres holds the
blackboard, and the factory restarts itself on crash or reboot. For a bare-metal
install instead of Docker, use [`deploy/krish.service`](deploy/krish.service).

> Put this behind a reverse proxy with TLS and auth before exposing it to the
> internet. Authentication is Phase 6.

## Assets

`config/assets.yaml` defines the tradable universe — GOLD, US30, US100, BITCOIN,
OIL to start, each with its own cost model. Add any instrument there (or from the
control room) and the whole factory picks it up on the next cycle. No code change.

## Configuration

`config/factory.yaml` holds everything that decides *how strict* the factory is:
cycle cadence, backtest costs and balance, tuner budget, and the thresholds a
strategy must clear to be delivered. Both config files are editable live from the
control room; agents reload on change.

## What needs your input next

| Needed | Unlocks |
|---|---|
| LLM API key | richer hypotheses and dossiers (deterministic fallback works without it) |
| VPS specs | how many parallel backtests to run |
| Telegram bot token + chat id | strategy delivery straight to your phone |
| Google Drive service-account JSON | delivery to your Drive folder |
| MT5 broker + demo account | Phase 5/7: MQL5 export and one-command automation |
| Your thresholds (min OOS Sharpe, max drawdown, min trades) | tune `config/factory.yaml` to your risk appetite |

---

Everything here is historical analysis and generated software. It is **not
financial advice**. Backtests are not results. Run anything this produces on a
demo account first, and treat the out-of-sample numbers as the optimistic case.
