# KRISH — Architecture

KRISH is a **self-improving strategy factory**. It is not a trading bot. It is a
factory of workers (agents) whose product is *validated trading strategies*,
delivered as source code for MT5, TradingView and Python.

## 1. The one idea that makes everything work: Strategy IR

Every strategy lives as a single JSON document called the **Strategy IR**
(Intermediate Representation): indicators, entry/exit logic, risk block,
parameter ranges, and lineage (who its parents were, which generation it is).

```
                    ┌──────────────────┐
                    │  Strategy IR     │   (one JSON = one strategy)
                    │  (JSON / DSL)    │
                    └────────┬─────────┘
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
     Python compiler   MQL5 compiler   Pine Script compiler
     (backtest/tune)   (MT5 live EA)   (TradingView alerts)
```

Because of this:

* An agent can **invent** a strategy by emitting JSON — it never has to write
  syntactically correct code by hand.
* Two strategies can be **crossbred** (take indicators from A, risk from B) →
  infinite new combinations, which is exactly the "kuch na kuch naya banata
  rahega" requirement.
* "Isko MT5 pe automate karo" becomes a *compile step*, not a rewrite. Same for
  TradingView.
* The IR is versioned, diffable and storable, so the system can learn which
  *structures* work, not just which numbers work.

## 2. Agents communicate, they do not call each other

Agents are independent async workers. They never import each other. They talk
over a **message bus** (Redis Streams in production, in-memory for dev) using a
typed envelope:

```json
{
  "id": "msg_...", "ts": "...", "sender": "tester",
  "topic": "strategy.tested", "type": "event",
  "project_id": "prj_...", "reply_to": "msg_...",
  "payload": { }
}
```

Three interaction patterns:

| Pattern | Use |
|---|---|
| **event** (`strategy.created`) | broadcast — anyone interested reacts |
| **request / response** (`data.request` → `data.response`) | "mujhe ye chahiye" — any agent can ask any other agent for help |
| **task** (assigned by Orchestrator) | work queue, load balanced |

Plus a shared **Blackboard** (Postgres/SQLite): the long-lived facts — assets,
strategies, backtest runs, verdicts, deliveries, agent heartbeats. The bus is
the nervous system; the blackboard is the memory.

## 3. The factory loop (runs forever, 24/7)

```
 MarketData ─┐                                     ┌─► Packager ─► Delivery
 MacroEcon ──┤                                     │   (zip + Drive + Telegram)
 Researcher ─┴─► Architect ─► Developer ─► Tester ──┤
        ▲            ▲            │          │      └─► Rejected
        │            │            └──fix─────┘
        │            │                       │
        │            │                       ▼
        │            │                    Tuner ─► Robustness ─► Risk ─► Judge
        │            │                                                    │
        └────────────┴──────────── Memory / Learning ◄─────────────────────┘
                     (every result, pass or fail, feeds the next generation)
```

Nothing in this loop stops. When a strategy is rejected, the *reason* is stored
and becomes a constraint/hint for the next generation. When one passes, its IR
becomes breeding stock.

## 4. Learning — how it actually gets smarter

Three concrete mechanisms (no hand-waving):

1. **Experiment ledger.** Every backtest ever run is stored with the full IR,
   the market regime it was tested in, and its out-of-sample scores. This is the
   training set.
2. **Evolutionary search.** Each cycle keeps an elite population, then mutates
   (change a param / swap an indicator) and crosses over (A's entry + B's exit).
   Fitness = *robust out-of-sample* score, not in-sample profit.
3. **Structural priors.** The Memory agent aggregates the ledger into stats like
   "on GOLD H1, ATR-based stops beat fixed stops 71% of the time (n=340)". The
   Architect receives these priors as its prompt/config, so generation N+1
   starts from what generation N proved.

Overfitting is treated as the enemy: in-sample / out-of-sample split,
walk-forward validation, Monte Carlo trade shuffling, parameter-sensitivity
plateau check, and a deflated performance score that penalises how many
variations were tried. A strategy only gets a PASS verdict if it survives all of
them — that is the answer to "long term me profitable ho sakta hai ya nahi".

## 5. Runtime shape (single VPS, 24/7)

```
docker compose
├── redis            message bus + agent heartbeats + locks
├── postgres         blackboard (strategies, runs, verdicts, deliveries)
├── krish-core       orchestrator + all agents (asyncio, one process, scalable to N)
├── krish-api        FastAPI: REST + WebSocket live feed + control endpoints
└── krish-web        Next.js control room (animated dashboard)
```

Agents are supervised: crash → restart with backoff, heartbeat → dashboard shows
who is alive, what task they hold, and for how long. Work is idempotent and
checkpointed in the blackboard, so a restart never loses a project.

MT5 execution is the one thing that cannot live in Linux Docker: a companion
**MT5 bridge** runs on a Windows VPS (or Wine) and receives compiled EAs +
commands from the bus. Until you say "automate karo", nothing goes live.

## 6. Control from the website

The dashboard is not read-only. Over WebSocket + REST you can:

* watch every agent live — role, status, current project, elapsed time, ETA
* see the strategy pipeline as a live board (created → tested → tuned → judged →
  delivered) with full backtest detail, equity curves, walk-forward tables
* add/remove assets and timeframes, change risk limits and pass thresholds
* pause/resume/kill any agent, force a new research cycle, re-run a backtest
* one-click **Export → MT5 / TradingView** and **Deliver → Drive / Telegram**
* one-click **Automate on MT5** (demo first, live behind an explicit confirm)

## 7. Assets

Configured in `config/assets.yaml`, not hardcoded — GOLD, US30, US100, BITCOIN,
OIL to start; add any symbol later with its spread/commission/tick model and the
whole factory picks it up on next cycle.
