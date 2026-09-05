# KRISH — Build Roadmap

Each phase ends with something that **runs**, not a half-finished layer. Order is
chosen so the risky/foundational parts get proven first.

---

## Phase 0 — Foundation ✅ (this PR)

Skeleton that already breathes: agents, bus, blackboard, IR, backtester, API,
live control room.

- repo layout, `.env.example`, docker-compose (redis + postgres + core + api)
- `config/assets.yaml` — GOLD, US30, US100, BITCOIN, OIL (fully customizable)
- message bus (Redis Streams + in-memory dev fallback), typed envelopes
- agent base class: heartbeat, status, current-task, elapsed time, crash restart
- agent registry + blackboard models (project, strategy, run, verdict, delivery)
- **Strategy IR schema** + indicator library + IR→Python compiler
- event-driven backtest engine: costs, IS/OOS split, full metric set
- first working squad: `market_data → architect → developer → tester → tuner →
  judge → packager`
- FastAPI REST + WebSocket live feed + temporary control room page

**Exit test:** `make run` boots the factory, it pulls real GOLD/BTC data, invents
strategies, backtests them, judges them, and you watch it happen live.

---

## Phase 1 — Real data layer

- providers: yfinance (indices/metals/oil), ccxt (crypto, minute data), MT5
  history import, Dukascopy tick option
- Parquet cache with gap detection + auto-repair, timezone/session handling
- `macro_econ`: economic calendar ingest, high-impact blackout windows,
  regime labels; `news_sentiment` feed
- data quality gate — no strategy is tested on dirty data

**Exit test:** 5+ years of clean H1 + D1 for all 5 assets, with news windows
tagged, reproducible from a cold start.

---

## Phase 2 — Validation you can trust

This is the phase that decides whether the whole system is worth anything.

- `robustness`: walk-forward, Monte Carlo, parameter-sensitivity surface,
  deflated Sharpe / multiple-testing penalty
- `risk`: sizing, drawdown & exposure limits, worst-case sequence, margin model
- `tuner`: Optuna walk-forward optimisation, stability-region reporting
- `judge`: threshold config + written verdict + long-term viability call
- per-regime and per-asset breakdown in every report

**Exit test:** feed it a known-overfit strategy — it must get flagged and
rejected with the correct reason.

---

## Phase 3 — The learning loop

- experiment ledger + `memory` agent (vector + statistical knowledge base)
- structural priors published to `architect`
- evolutionary engine: elite population, mutation, crossover, novelty/dedupe
  check so it does not rediscover the same strategy forever
- `researcher` + `quant_analyst` with LLM-backed hypothesis generation
- `portfolio` agent: correlation-aware acceptance

**Exit test:** generation 10 has measurably better OOS scores than generation 1,
and the ledger shows *why*.

---

## Phase 4 — Delivery

- `doc_writer`: strategy dossier + install instructions
- `packager`: named versioned ZIP (source + IR + reports + README + checksum)
- `delivery`: Google Drive upload (OAuth service account) + Telegram bot post
- delivery log and re-delivery from the UI

**Exit test:** a PASS strategy lands in your Drive and Telegram with everything
needed to run it, no manual step.

---

## Phase 5 — Export compilers ✅ (compilers done)

- ✅ **Pine Script v5 compiler**: IR → strategy + alert webhooks
- ✅ **MQL5 compiler**: IR → `.mq5` Expert Advisor — entries, exits, ATR/percent/
  point stops, R:R targets, trailing, breakeven, time stop, session and
  volatility filters, risk-based lot sizing from the symbol's own tick value,
  magic number, one decision per closed bar. Every parameter is an `input`, so it
  can be re-optimised in MT5's own Strategy Tester.
- ⏳ **still to do:** golden-file parity tests, so MT5 Strategy Tester results are
  automatically compared against the Python backtest within tolerance. Until that
  exists, the parity check is a manual step and the instructions say so.

**Exit test:** same strategy, three engines, matching equity curves.

---

## Phase 6 — Control room (the website)

- Next.js + TypeScript + Tailwind + Framer Motion / GSAP, dark "mission control"
  aesthetic, animated agent graph with live message flow
- live agent cards: role, status, project, elapsed, ETA
- strategy pipeline board (created → tested → tuned → judged → delivered)
- strategy detail: equity curve, drawdown, walk-forward table, trade list,
  parameter surface, verdict reasoning
- controls: assets, thresholds, risk, pause/resume/kill agent, force cycle,
  re-run backtest, export MT5/Pine, deliver, automate
- auth + audit log (it can move money — it gets a login)

**Exit test:** you can run the entire factory from the browser without touching
a terminal.

---

## Phase 7 — Automation & live

- MT5 bridge on Windows VPS: EA push, chart attach, demo-first, live-vs-backtest
  parity check, kill switch
- TradingView webhook receiver → broker relay (optional)
- `live_monitor`: drift detection, auto-pause, Telegram alerts

**Exit test:** "isko automate karo" → running on demo within minutes, with a
parity report and a working kill switch.

---

## Phase 8 — 24/7 hardening on your VPS

- systemd + docker restart policies, healthchecks, log rotation, backups
- resource budgeting so backtests never starve the API
- Prometheus/Grafana or lightweight metrics + Telegram alerting
- one-command deploy/update script

**Exit test:** kill any container at random — the factory keeps working and
nothing is lost.

---

## What I need from you (blockers, in order of urgency)

| # | Needed | For |
|---|---|---|
| 1 | LLM API key (Anthropic / OpenAI / local model) | researcher, architect, doc_writer reasoning |
| 2 | VPS specs + OS (cores, RAM, Linux or Windows) | how many parallel backtests, MT5 bridge plan |
| 3 | Data preference: free (yfinance/ccxt) to start, or paid (Polygon/Databento/Dukascopy) | Phase 1 quality ceiling |
| 4 | Google Drive service-account JSON + target folder | Phase 4 delivery |
| 5 | Telegram bot token + your chat ID | Phase 4 delivery + alerts |
| 6 | Broker + MT5 account (demo first) and typical spreads | Phase 5 cost model, Phase 7 automation |
| 7 | Your pass thresholds — min OOS Sharpe, max drawdown you accept, min trades | judge configuration |

Nothing in Phases 0–3 is blocked by these except the LLM key, and even that has a
deterministic fallback so the factory keeps producing without it.
