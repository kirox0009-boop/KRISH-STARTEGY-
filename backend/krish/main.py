"""KRISH entrypoint.

python -m krish.main run              factory + API + control room (the normal way)
python -m krish.main run --no-api     agents only (e.g. a second worker box)
python -m krish.main api              API only
python -m krish.main cycle --asset GOLD --timeframe H1
                                      run exactly one cycle end to end and exit
python -m krish.main roster           print the agent roster
python -m krish.main backtest FILE    backtest a Strategy IR JSON file
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from pathlib import Path

import uvicorn

from .api import create_app
from .backtest.engine import BacktestConfig, split_backtest, walk_forward
from .bus import bus
from .config import settings
from .data.providers import fetch_ohlcv
from .ir.schema import StrategyIR
from .messages import Message, MsgKind, Topic
from .runtime import Factory, configure_logging
from .store import counts, init_db

log = logging.getLogger("krish.main")


async def _serve_api(app, host: str, port: int) -> None:
    config = uvicorn.Config(app, host=host, port=port, log_level=settings().log_level.lower())
    server = uvicorn.Server(config)
    await server.serve()


async def cmd_run(args: argparse.Namespace) -> int:
    configure_logging()
    factory = Factory(bus(), replicas=args.replicas)
    factory.build()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    tasks: list[asyncio.Task] = []
    if not args.no_agents:
        await factory.start()
    else:
        init_db()
        await bus().start()

    if not args.no_api:
        app = create_app(factory)
        tasks.append(asyncio.create_task(_serve_api(app, settings().api_host, settings().api_port)))
        log.info("control room: http://localhost:%d  (API docs at /docs)", settings().api_port)

    await stop.wait()
    log.info("shutdown requested")
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    if not args.no_agents:
        await factory.stop()
    return 0


async def cmd_api(args: argparse.Namespace) -> int:
    configure_logging()
    init_db()
    await bus().start()
    await _serve_api(create_app(None), settings().api_host, settings().api_port)
    return 0


async def cmd_cycle(args: argparse.Namespace) -> int:
    """Run one full cycle and report what happened. The fastest way to sanity check."""
    os.environ["KRISH_SCHEDULER"] = "off"  # exactly one cycle, no background ticking
    configure_logging()
    factory = Factory(bus(), replicas=1)
    factory.build()
    await factory.start()

    await asyncio.sleep(2.0)
    await bus().publish(
        Message(
            topic=Topic.AGENT_CONTROL,
            sender="cli",
            kind=MsgKind.CONTROL,
            payload={
                "agent": "orchestrator",
                "action": "cycle",
                "asset": args.asset,
                "timeframe": args.timeframe,
                "count": args.count,
            },
        )
    )

    log.info("waiting up to %ds for the cycle to finish...", args.timeout)
    deadline = asyncio.get_running_loop().time() + args.timeout
    seen_judgements = 0
    try:
        async for msg in bus().tap():
            if msg.topic == Topic.STRATEGY_JUDGED:
                seen_judgements += 1
                log.info(
                    "verdict %d/%d: %s -> %s",
                    seen_judgements,
                    args.count,
                    msg.payload.get("name"),
                    msg.payload.get("verdict"),
                )
                if seen_judgements >= args.count:
                    break
            if asyncio.get_running_loop().time() > deadline:
                log.warning("timed out waiting for verdicts")
                break
    finally:
        await factory.stop()

    summary = counts()
    print(json.dumps(summary, indent=2))
    return 0 if seen_judgements else 1


async def cmd_roster(args: argparse.Namespace) -> int:
    factory = Factory(bus())
    rows = factory.describe()
    width = max(len(r["name"]) for r in rows)
    current = ""
    for row in rows:
        if row["squad"] != current:
            current = row["squad"]
            print(f"\n[{current}]")
        print(f"  {row['name']:<{width}}  {row['role']:<22} {row['description']}")
    print(f"\n{len(rows)} agents")
    return 0


async def cmd_backtest(args: argparse.Namespace) -> int:
    configure_logging()
    path = Path(args.file)
    ir = StrategyIR.model_validate(json.loads(path.read_text()))
    print(ir.describe())
    frame = await asyncio.to_thread(fetch_ohlcv, ir.asset, ir.timeframe)
    print(f"\n{len(frame)} bars: {frame.index[0]} -> {frame.index[-1]}\n")

    config = BacktestConfig.from_factory_config()
    result = await asyncio.to_thread(split_backtest, ir, frame, config=config)
    if result.get("error"):
        print(f"error: {result['error']}")
        return 1
    print(f"{'segment':<8} {'trades':>7} {'sharpe':>8} {'PF':>7} {'maxDD%':>8} {'ret%':>9}")
    for segment in ("full", "is", "oos"):
        m = result[segment]["metrics"]
        print(
            f"{segment:<8} {m.get('trades', 0):>7} {m.get('sharpe', 0):>8.3f} "
            f"{m.get('profit_factor', 0):>7.2f} {m.get('max_drawdown_pct', 0):>8.2f} "
            f"{m.get('total_return_pct', 0):>9.2f}"
        )
    print(f"\nrobust score (OOS): {result['robust_score']}")
    if args.walk_forward:
        wf = await asyncio.to_thread(walk_forward, ir, frame, folds=4, config=config)
        print(
            f"walk-forward consistency: {wf.get('consistency')} ({wf.get('profitable_folds')}"
            f"/{wf.get('fold_count')} eras)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="krish", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the factory and the control room")
    run.add_argument("--no-api", action="store_true", help="agents only")
    run.add_argument("--no-agents", action="store_true", help="API only")
    run.add_argument(
        "--replicas",
        type=int,
        default=None,
        help="replicas of CPU-bound agents (default: from config/factory.yaml)",
    )
    run.set_defaults(func=cmd_run)

    api = sub.add_parser("api", help="run only the API + control room")
    api.set_defaults(func=cmd_api)

    cycle = sub.add_parser("cycle", help="run a single cycle end to end, then exit")
    cycle.add_argument("--asset", default=None)
    cycle.add_argument("--timeframe", default=None)
    cycle.add_argument("--count", type=int, default=1)
    cycle.add_argument("--timeout", type=int, default=900)
    cycle.set_defaults(func=cmd_cycle)

    roster = sub.add_parser("roster", help="list the agents")
    roster.set_defaults(func=cmd_roster)

    bt = sub.add_parser("backtest", help="backtest a Strategy IR JSON file")
    bt.add_argument("file")
    bt.add_argument("--walk-forward", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
