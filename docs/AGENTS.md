# KRISH — Agent Roster

Every agent has: a single responsibility, topics it listens to, topics it emits,
and services it offers to other agents ("help desk"). No agent calls another
directly — everything is messages, so any agent can be replaced, duplicated, or
moved to another machine.

Legend: **S** = subscribes, **E** = emits, **H** = help it offers on request.

## Data & context squad

### `market_data`
Owns all price history. Fetches, cleans, caches (Parquet), and serves bars.
- **S** `data.request`, `asset.added`, `schedule.tick`
- **E** `data.response`, `data.updated`, `data.gap_detected`
- **H** OHLCV for any asset/timeframe/date range; tradable-hours calendar

### `macro_econ`
Economic calendar, rate decisions, CPI/NFP, DXY context, risk-on/risk-off
regime. Turns fundamentals into machine-readable features and blackout windows.
- **S** `schedule.tick`, `macro.request`
- **E** `macro.event`, `macro.regime_changed`, `macro.response`
- **H** "is there high-impact news in this window?", regime label per date

### `news_sentiment`
Headlines and sentiment for the configured assets; feeds features and warns the
live layer about shock events.
- **E** `news.signal`, `news.shock`

### `regime`
Statistical market-state labelling (trend / range / high-vol / low-vol) per
asset+timeframe. Used both as a strategy filter and to check whether a strategy
was only lucky in one regime.
- **E** `regime.labels_updated`
- **H** regime label series for any asset/timeframe

## Research & build squad

### `researcher`
Idea generation. Mines papers, forums, public repos and the internal experiment
ledger for hypotheses; writes them as structured *hypothesis* records with a
testable claim, not prose.
- **S** `cycle.start`, `memory.priors_updated`
- **E** `hypothesis.created`
- **H** literature/context lookup for any concept

### `quant_analyst`
Turns hypotheses into concrete features: indicator choice, transforms, lookback
bounds, statistical pre-checks (does the edge even exist before costs?).
- **S** `hypothesis.created`
- **E** `feature_spec.created`, `hypothesis.rejected_early`

### `architect`
The strategy inventor. Emits **Strategy IR** JSON. Three modes: (a) fresh idea
from a hypothesis, (b) mutation of an elite strategy, (c) crossover of two
parents. Uses Memory's structural priors so each generation starts smarter.
- **S** `feature_spec.created`, `evolution.request`, `memory.priors_updated`
- **E** `strategy.created` (IR + lineage)

### `developer`
Compiles IR → runnable Python signal module, validates schema, catches
degenerate logic (always-true entries, look-ahead bias, unused indicators), and
repairs broken IR emitted by the Architect.
- **S** `strategy.created`, `strategy.test_failed`
- **E** `strategy.built`, `strategy.invalid`
- **H** IR validation / static look-ahead audit for anyone

## Validation squad

### `tester`
Runs the backtest: realistic costs (spread, commission, slippage), in-sample vs
out-of-sample split, per-asset and per-regime breakdown, trade list, equity
curve.
- **S** `strategy.built`, `backtest.request`
- **E** `strategy.tested`, `strategy.test_failed`
- **H** ad-hoc backtest for any IR (used by Tuner and by you from the UI)

### `robustness`
The overfitting police. Walk-forward validation, Monte Carlo trade shuffling,
parameter-sensitivity surface (is the result a plateau or a needle?), deflated
performance score accounting for how many variants were tried, noise/latency
stress tests.
- **S** `strategy.tested`, `strategy.tuned`
- **E** `robustness.report`, `strategy.overfit_flagged`

### `risk`
Position sizing, drawdown limits, exposure and correlation rules, worst-case
sequence, margin math per asset. Rewrites the IR risk block to be survivable.
- **S** `strategy.tested`
- **E** `risk.report`, `strategy.risk_adjusted`

### `tuner`
Parameter optimisation done honestly: walk-forward optimisation with Optuna,
objective = robust OOS score with a complexity penalty, and it must report the
*stability region*, not just the best point. Decides what to adjust and what to
leave alone.
- **S** `strategy.tested`, `tune.request`
- **E** `strategy.tuned`, `tune.no_improvement`

### `portfolio`
Looks across all passing strategies: correlation, overlap, combined drawdown,
capital allocation. Rejects a good strategy that only duplicates an existing one.
- **S** `strategy.judged`
- **E** `portfolio.allocation_updated`

### `judge`
The final verdict authority: PASS / BORDERLINE / REJECT with written reasoning
and an explicit long-term-viability call, based on Tester + Robustness + Risk +
Portfolio reports against configurable thresholds.
- **S** `robustness.report`, `risk.report`
- **E** `strategy.judged`

## Delivery squad

### `doc_writer`
Generates the human package: what the strategy does, why it works, parameters,
recommended asset/timeframe/session, risk instructions, known failure modes,
install steps for MT5 and TradingView, and the full backtest report.
- **S** `strategy.judged` (PASS)
- **E** `docs.ready`

### `packager`
Builds the deliverable: Python source, MQL5 `.mq5`, Pine `.pine`, IR JSON,
backtest CSV/PNG, README/instructions → one named, versioned ZIP with a manifest
and checksum.
- **S** `docs.ready`, `package.request`
- **E** `package.ready`

### `delivery`
Ships it: uploads to Google Drive (foldered by asset/date), posts to Telegram
with a summary card and the link, records the delivery in the blackboard.
- **S** `package.ready`
- **E** `delivery.completed`, `delivery.failed`

## Deployment squad (on your command only)

### `mt5_deploy`
On "automate karo": compiles the EA, pushes to the MT5 bridge, attaches to
chart, starts on **demo** first, verifies live-vs-backtest fill agreement, and
only then offers the live switch.
- **S** `automate.request`
- **E** `deploy.status`, `deploy.parity_report`

### `tradingview_deploy`
Emits Pine Script v5 with alert hooks and webhook payloads, plus setup steps.
- **S** `tradingview.request`
- **E** `pine.ready`

### `live_monitor`
Watches deployed strategies for drift: live vs expected metrics, slippage
creep, regime change, drawdown breach → can auto-pause and page you.
- **E** `live.drift_alert`, `live.paused`

## System squad

### `orchestrator`
Plans cycles, assigns work, sets priorities and budgets (how much CPU per
asset), enforces concurrency limits, restarts dead agents, keeps projects moving.
- **E** `cycle.start`, `task.assigned`

### `memory`
The learning core. Ingests every experiment (pass *and* fail), maintains the
vector + statistical knowledge base, and publishes **structural priors** that
steer the next generation.
- **S** everything terminal (`strategy.judged`, `tune.*`, `robustness.*`)
- **E** `memory.priors_updated`
- **H** "have we tried something like this before?" dedupe lookup

### `monitor`
System health: VPS CPU/RAM/disk, queue depth, agent latency, error rates, data
freshness. Emits alerts to Telegram before things break.

### `librarian` (optional, later)
Housekeeping: prunes dead strategy branches, archives old runs, keeps the DB and
Parquet cache from eating the VPS disk.
