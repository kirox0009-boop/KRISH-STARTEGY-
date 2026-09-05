"""Delivery squad: doc_writer -> packager -> delivery.

A strategy is not "done" when it passes. It is done when you have, in your hands,
a named ZIP containing the source, the exact parameters, the evidence, and
instructions clear enough to run it without asking anyone anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import hashlib
import io
import json
import logging
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .. import llm
from ..compilers.mql5 import Mql5Unsupported, to_mql5
from ..compilers.pine import PineUnsupported, to_pine
from ..compilers.python_export import to_python_runner
from ..config import PACKAGE_DIR, factory_section, settings
from ..ir.schema import StrategyIR
from ..messages import Message, Topic
from ..storage import keep_local_packages, package_key, store, upload_packages
from ..store import Delivery, in_db, runs_for_strategy, save, update_project
from .base import BaseAgent

log = logging.getLogger("krish.deliver")

#: verdicts that earn a delivery. BORDERLINE ships too, clearly labelled.
DELIVERABLE = {"PASS", "BORDERLINE"}


def safe_filename(name: str) -> str:
    """Filesystem-safe name. Shared: the docs must name the same file the packager
    writes, so this cannot live on one of the two agents."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name).strip("_")


class DocWriterAgent(BaseAgent):
    name = "doc_writer"
    role = "Documentation Writer"
    squad = "delivery"
    description = "Writes the strategy dossier: what it does, why, how to run it, how it fails."
    #: PACKAGE_REQUEST is the operator asking for a package by hand. It arrives
    #: here, not at the packager, so a hand-made package still gets a real dossier
    #: instead of an empty REPORT.md.
    subscribes = (Topic.STRATEGY_JUDGED, Topic.PACKAGE_REQUEST)

    async def handle(self, msg: Message) -> None:
        verdict = str(msg.payload.get("verdict", "REJECT"))
        manual = msg.topic == Topic.PACKAGE_REQUEST
        if not manual and verdict not in DELIVERABLE:
            return

        ir = StrategyIR.model_validate(msg.payload["ir"])
        dossier = await self._write(ir, msg.payload)
        instructions = self._instructions(ir, msg.payload)

        self.log(
            f"dossier written for '{ir.name}'" + (" (operator request)" if manual else ""),
            msg=msg,
        )
        await self.emit(
            Topic.DOCS_READY,
            {**msg.payload, "dossier_md": dossier, "instructions_md": instructions},
            parent=msg,
            strategy_id=ir.id,
        )

    async def _write(self, ir: StrategyIR, payload: dict[str, Any]) -> str:
        oos = dict(payload.get("metrics", {}).get("oos") or {})
        is_m = dict(payload.get("metrics", {}).get("is") or {})
        full = dict(payload.get("metrics", {}).get("full") or {})
        rob = dict(payload.get("robustness") or {})
        risk = dict(payload.get("risk_report") or {})
        wf = dict(rob.get("walk_forward") or {})
        mc = dict(rob.get("monte_carlo") or {})
        stress = dict(rob.get("cost_stress") or {})
        sens = dict(rob.get("sensitivity") or {})

        rows = [
            ("Trades", "trades"),
            ("Sharpe", "sharpe"),
            ("Sortino", "sortino"),
            ("Profit factor", "profit_factor"),
            ("Win rate %", "win_rate_pct"),
            ("Expectancy (R)", "expectancy_r"),
            ("Max drawdown %", "max_drawdown_pct"),
            ("Total return %", "total_return_pct"),
            ("CAGR %", "cagr_pct"),
            ("Exposure %", "exposure_pct"),
        ]
        table = ["| Metric | Full | In-sample | Out-of-sample |", "|---|---|---|---|"]
        for label, key in rows:
            table.append(
                f"| {label} | {full.get(key, '-')} | {is_m.get(key, '-')} | {oos.get(key, '-')} |"
            )

        why = ir.hypothesis or "No hypothesis recorded."
        if llm.available():
            better = await llm.complete(
                "Explain, in 3-4 sentences, the market behaviour this trading strategy is "
                "trying to exploit. Be concrete and sceptical. No hype, no financial advice.\n\n"
                f"Strategy:\n{ir.describe()}\n\nOriginal hypothesis: {why}",
                system="You are a quantitative researcher documenting an internal strategy.",
                max_tokens=350,
                temperature=0.5,
            )
            if better:
                why = better

        parts = [
            f"# {ir.name}",
            "",
            f"**{ir.asset} · {ir.timeframe} · {ir.style}** — verdict "
            f"**{payload.get('verdict')}** (score {payload.get('score')})",
            "",
            f"> {payload.get('summary', '')}",
            "",
            "## What it is trying to exploit",
            "",
            why,
            "",
            "## Logic",
            "",
            "```",
            ir.describe(),
            "```",
            "",
            "## Performance",
            "",
            *table,
            "",
            "The out-of-sample column is the only one that matters. It comes from the tail of "
            "history that the tuner was never allowed to see.",
            "",
            "## Robustness evidence",
            "",
        ]

        if wf.get("fold_count"):
            parts += [
                f"- **Walk-forward:** {wf.get('profitable_folds')}/{wf.get('fold_count')} eras "
                f"profitable (consistency {wf.get('consistency')}), mean Sharpe "
                f"{wf.get('mean_sharpe')}, worst era Sharpe {wf.get('worst_sharpe')}.",
            ]
        if stress:
            parts += [
                f"- **Cost stress:** profit factor {stress.get('1.0x', {}).get('profit_factor')} "
                f"at quoted costs, {stress.get('2.0x', {}).get('profit_factor')} at double costs "
                f"({'survives' if stress.get('survives_double_costs') else 'does NOT survive'}).",
            ]
        if mc.get("runs"):
            parts += [
                f"- **Monte Carlo ({mc['runs']} resamples):** "
                f"{float(mc.get('prob_profitable', 0)) * 100:.0f}% of trade orderings end "
                f"profitable; 5th percentile outcome {mc.get('r_5th_percentile')}R; "
                f"typical worst drawdown {mc.get('median_max_dd_r')}R.",
            ]
        if sens.get("tested"):
            parts += [
                f"- **Parameter sensitivity:** worst retention "
                f"{sens.get('worst_retention')} across ±15% perturbations "
                f"({'plateau' if sens.get('is_plateau') else 'SPIKE - treat with suspicion'}).",
            ]
        flags = rob.get("flags") or []
        if flags:
            parts += ["", "### Open concerns", ""] + [f"- {f}" for f in flags]

        parts += ["", "## Risk instructions", ""]
        if risk:
            parts += [
                f"- Configured risk per trade: **{risk.get('risk_per_trade_pct')}%**; "
                f"recommended: **{risk.get('recommended_risk_per_trade_pct')}%** "
                f"(quarter-Kelly of {risk.get('kelly_fraction_pct')}%).",
                f"- Plan for a losing streak of **{risk.get('planning_loss_streak')} trades** "
                f"≈ **{risk.get('streak_drawdown_pct')}%** account drawdown.",
                f"- Stop/target style: {risk.get('stop_style')}.",
                f"- Observed out-of-sample max drawdown: {risk.get('observed_max_drawdown_pct')}%.",
            ]
            for warning in risk.get("warnings") or []:
                parts.append(f"- ⚠️ {warning}")
            for advisory in risk.get("advisories") or []:
                parts.append(f"- {advisory}")

        parts += [
            "",
            "## Known failure modes",
            "",
            "- Every number here comes from historical bars with a modelled spread, slippage and "
            "commission. Real fills, requotes and weekend gaps are not fully modelled.",
            "- The strategy was selected from many candidates. Selection alone inflates apparent "
            "quality; that is why the out-of-sample and walk-forward numbers are the ones quoted.",
            "- Run it on **demo first**, then at the smallest live size, and compare live fills "
            "against these numbers before increasing risk.",
            "",
            "## Lineage",
            "",
            f"- IR id: `{ir.id}`",
            f"- Origin: {ir.origin}, generation {ir.generation}",
            f"- Parents: {', '.join(ir.parents) if ir.parents else 'none (fresh idea)'}",
            f"- Notes: {ir.notes}",
        ]
        return "\n".join(parts) + "\n"

    def _instructions(self, ir: StrategyIR, payload: dict[str, Any]) -> str:
        risk_report = dict(payload.get("risk_report") or {})
        suggested_risk = risk_report.get("recommended_risk_per_trade_pct", 0.5)
        params = (
            "\n".join(
                f"  - `{spec.label or spec.path}` = {ir.get_param(spec.path)}"
                for spec in ir.param_space
            )
            or "  - (no tunable parameters exposed)"
        )
        return (
            "\n".join(
                [
                    f"# How to run {ir.name}",
                    "",
                    f"Instrument: **{ir.asset}** · Timeframe: **{ir.timeframe}** · Direction: "
                    f"**{ir.direction}**",
                    "",
                    "## 1. Reproduce the backtest (recommended first step)",
                    "",
                    "```bash",
                    "pip install -e /path/to/KRISH/backend",
                    "python run_backtest.py --walk-forward",
                    "```",
                    "",
                    "You should see the same out-of-sample numbers as in REPORT.md. If you do not, "
                    "stop and investigate before anything else.",
                    "",
                    "## 2. TradingView",
                    "",
                    "1. Open the chart for this instrument and set the timeframe to "
                    f"**{ir.timeframe}**.",
                    "2. Pine Editor → paste `strategy.pine` → Add to chart.",
                    "3. Check the Strategy Tester numbers are in the same ballpark as REPORT.md "
                    "(TradingView's cost model differs, so exact equality is not expected).",
                    "4. For automation: create an alert on the strategy, tick *Webhook "
                    "URL*, and use the JSON the script already emits.",
                    "",
                    "## 3. MetaTrader 5",
                    "",
                    f"The `{safe_filename(ir.name)}.mq5` file in this package is a complete "
                    "Expert Advisor generated from the same IR the backtest ran.",
                    "",
                    "1. In MT5: **File → Open Data Folder → MQL5 → Experts**, copy the "
                    "`.mq5` file there.",
                    "2. Back in MT5, open **MetaEditor** (F4), open the file, press "
                    "**Compile** (F7). It should compile with no errors.",
                    "3. Open the chart for this instrument at the "
                    f"**{ir.timeframe}** timeframe and drag the EA onto it.",
                    "4. **Run the Strategy Tester on it first** and compare the result with "
                    "REPORT.md. They will not match exactly — your broker's spread, "
                    "commission and swap differ from the model — but the shape of the "
                    "equity curve and the trade count should be recognisably similar. If "
                    "they are not, do not run it.",
                    "5. Attach to a **demo** account and leave it for at least 30 trades "
                    "before considering anything else.",
                    "",
                    "Two things to check before you trust it:",
                    "",
                    "- **Session filter uses broker server time**, which is usually not UTC. "
                    "The backtest filtered on UTC. If this strategy has an hours filter, "
                    "shift the `hours[]` values by your broker's offset.",
                    "- **Symbol name** must match your broker's "
                    f"(this was built for `{ir.asset}`; your broker may call it something "
                    "else, e.g. `XAUUSD.m`).",
                    "",
                    "## 4. Parameters as delivered",
                    "",
                    params,
                    "",
                    "## 5. Risk",
                    "",
                    f"Use at most {suggested_risk}% of equity per trade. Do not scale up "
                    "until at least 30 live trades agree with the expected distribution.",
                    "",
                    "_This package is generated software and historical analysis, not financial "
                    "advice._",
                ]
            )
            + "\n"
        )


class PackagerAgent(BaseAgent):
    name = "packager"
    role = "Packager"
    squad = "delivery"
    description = "Builds the named, versioned ZIP: source, IR, reports, instructions, checksum."
    #: Only DOCS_READY: everything reaches the packager through the doc writer, so
    #: there is exactly one path into a package and it always carries documentation.
    subscribes = (Topic.DOCS_READY,)

    async def handle(self, msg: Message) -> None:
        ir = StrategyIR.model_validate(msg.payload["ir"])
        verdict = str(msg.payload.get("verdict", "MANUAL"))
        cfg = factory_section("delivery")
        template = str(cfg.get("package_name_template", "KRISH_{asset}_{style}_{name}_v{version}"))
        stamp = datetime.now(UTC).strftime("%Y%m%d")
        package_name = safe_filename(
            template.format(
                asset=ir.asset,
                style=ir.style.split("+")[0],
                name=ir.name,
                version=f"{ir.generation + 1}.{stamp}",
            )
        )

        self.progress(f"packaging {package_name}")
        path, checksum, size, manifest = await asyncio.to_thread(
            self._build, ir, msg.payload, package_name, verdict
        )

        # Push to object storage before anyone is told the package exists, so a
        # recorded remote URL is always real.
        remote_url: str | None = None
        if upload_packages():
            self.progress(f"uploading {package_name}")
            remote_url = await asyncio.to_thread(store().put, path, package_key(package_name))

        record = await in_db(
            save,
            Delivery(
                strategy_id=ir.id,
                package_name=package_name,
                local_path=str(path),
                checksum=checksum,
                size_bytes=size,
                channels=(
                    {"object_store": {"status": "ok", "url": remote_url}} if remote_url else {}
                ),
                status="packaged",
            ),
        )
        if msg.project_id:
            await in_db(update_project, msg.project_id, stage="packaged")

        self.log(f"packaged '{ir.name}' -> {path.name} ({size / 1024:.0f} KB)", msg=msg)
        await self.emit(
            Topic.PACKAGE_READY,
            {
                "strategy_id": ir.id,
                "delivery_id": record.id,
                "name": ir.name,
                "asset": ir.asset,
                "timeframe": ir.timeframe,
                "verdict": verdict,
                "package_name": package_name,
                "path": str(path),
                "checksum": checksum,
                "size_bytes": size,
                "remote_url": remote_url,
                "manifest": manifest,
                "summary": msg.payload.get("summary", ""),
                "metrics": msg.payload.get("metrics", {}),
                "score": msg.payload.get("score"),
            },
            parent=msg,
            strategy_id=ir.id,
        )

    # ------------------------------------------------------------------ #

    def _build(
        self, ir: StrategyIR, payload: dict[str, Any], package_name: str, verdict: str
    ) -> tuple[Path, str, int, dict[str, Any]]:
        staging = PACKAGE_DIR / package_name
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        files: dict[str, str] = {
            "strategy.json": json.dumps(ir.model_dump(mode="json"), indent=2),
            "run_backtest.py": to_python_runner(
                ir,
                balance=float(factory_section("backtest").get("initial_balance", 10_000)),
                oos_fraction=float(factory_section("backtest").get("oos_fraction", 0.3)),
            ),
            "REPORT.md": payload.get("dossier_md", ""),
            "INSTRUCTIONS.md": payload.get("instructions_md", ""),
            "verdict.json": json.dumps(
                {
                    "verdict": verdict,
                    "score": payload.get("score"),
                    "long_term_viable": payload.get("long_term_viable"),
                    "checks": payload.get("checks", {}),
                    "reasons": payload.get("reasons", []),
                    "robustness": payload.get("robustness", {}),
                    "risk_report": payload.get("risk_report", {}),
                    "metrics": payload.get("metrics", {}),
                },
                indent=2,
                default=str,
            ),
        }

        try:
            files["strategy.pine"] = to_pine(ir)
        except PineUnsupported as exc:
            files["strategy.pine.SKIPPED.txt"] = (
                f"Pine Script export unavailable for this strategy: {exc}\n"
            )

        try:
            files[f"{safe_filename(ir.name)}.mq5"] = to_mql5(ir)
        except Mql5Unsupported as exc:
            files["strategy.mq5.SKIPPED.txt"] = (
                f"MetaTrader 5 export unavailable for this strategy: {exc}\n"
            )

        runs = runs_for_strategy(ir.id)
        for run in runs:
            if run.trades:
                files[f"trades_{run.kind}.csv"] = self._csv(run.trades)
            if run.equity:
                files[f"equity_{run.kind}.csv"] = self._csv(run.equity)

        files["README.md"] = self._readme(ir, payload, verdict, sorted(files))

        for rel, content in files.items():
            (staging / rel).write_text(content, encoding="utf-8")

        manifest = {
            "package": package_name,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_id": ir.id,
            "name": ir.name,
            "asset": ir.asset,
            "timeframe": ir.timeframe,
            "style": ir.style,
            "verdict": verdict,
            "score": payload.get("score"),
            "files": sorted(files),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        zip_path = PACKAGE_DIR / f"{package_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(staging.rglob("*")):
                if file.is_file():
                    zf.write(file, arcname=f"{package_name}/{file.relative_to(staging)}")

        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        (PACKAGE_DIR / f"{package_name}.zip.sha256").write_text(
            f"{digest}  {zip_path.name}\n", encoding="utf-8"
        )
        shutil.rmtree(staging)
        return zip_path, digest, zip_path.stat().st_size, manifest

    @staticmethod
    def _csv(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    @staticmethod
    def _readme(ir: StrategyIR, payload: dict[str, Any], verdict: str, files: list[str]) -> str:
        oos = dict(payload.get("metrics", {}).get("oos") or {})
        banner = (
            "This strategy cleared every threshold and robustness check."
            if verdict == "PASS"
            else "This strategy is BORDERLINE: it missed at least one threshold narrowly. "
            "Treat it as a candidate, not a finished product."
        )
        return (
            "\n".join(
                [
                    f"# {ir.name} — {ir.asset} {ir.timeframe}",
                    "",
                    f"**Verdict: {verdict}.** {banner}",
                    "",
                    f"Out-of-sample: {oos.get('trades', 0)} trades · "
                    f"Sharpe {oos.get('sharpe', 0)} · PF {oos.get('profit_factor', 0)} · "
                    f"max DD {oos.get('max_drawdown_pct', 0)}%",
                    "",
                    "## Contents",
                    "",
                    *[f"- `{f}`" for f in files],
                    "- `manifest.json` — file list, ids and checksum context",
                    "",
                    "Start with `INSTRUCTIONS.md`. Read `REPORT.md` before risking money.",
                    "",
                    "Generated by KRISH. Historical analysis and generated code — not financial "
                    "advice.",
                ]
            )
            + "\n"
        )


class DeliveryAgent(BaseAgent):
    name = "delivery"
    role = "Delivery"
    squad = "delivery"
    description = "Ships the ZIP to Telegram and Google Drive, and records where it went."
    subscribes = (Topic.PACKAGE_READY,)
    handler_timeout = 600.0

    async def handle(self, msg: Message) -> None:
        cfg = factory_section("delivery")
        path = Path(str(msg.payload["path"]))
        remote_url = msg.payload.get("remote_url")
        channels: dict[str, Any] = {
            "local": {"status": "ok", "path": str(path), "checksum": msg.payload.get("checksum")}
        }
        if remote_url:
            channels["object_store"] = {"status": "ok", "url": remote_url}

        if cfg.get("telegram_enabled") and settings().telegram_bot_token:
            channels["telegram"] = await self._telegram(path, msg.payload)
        else:
            channels["telegram"] = {"status": "disabled"}

        if cfg.get("drive_enabled") and settings().gdrive_credentials_file:
            channels["gdrive"] = await self._gdrive(path)
        else:
            channels["gdrive"] = {"status": "disabled"}

        # Only now, once every channel has had its turn, is it safe to reclaim the
        # local copy — and only if it genuinely lives somewhere else.
        if remote_url and not keep_local_packages():
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
                channels["local"] = {
                    "status": "offloaded",
                    "note": "local copy removed; served from object storage",
                }

        ok = any(c.get("status") in {"ok", "offloaded"} for c in channels.values())
        await in_db(
            self._record,
            str(msg.payload.get("delivery_id", "")),
            channels,
            "delivered" if ok else "failed",
        )
        if msg.project_id:
            await in_db(
                update_project,
                msg.project_id,
                stage="delivered",
                status="done",
                finished_at=datetime.now(UTC),
            )

        self.log(
            f"delivered '{msg.payload.get('name')}': "
            + ", ".join(f"{k}={v.get('status')}" for k, v in channels.items()),
            msg=msg,
        )
        await self.emit(
            Topic.DELIVERY_COMPLETED if ok else Topic.DELIVERY_FAILED,
            {**msg.payload, "channels": channels},
            parent=msg,
        )

    @staticmethod
    def _record(delivery_id: str, channels: dict[str, Any], status: str) -> None:
        from ..store import Delivery as DeliveryRow
        from ..store import session

        if not delivery_id:
            return
        with session() as s:
            row = s.get(DeliveryRow, delivery_id)
            if row is not None:
                row.channels = channels
                row.status = status

    async def _telegram(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = settings()
        oos = dict(payload.get("metrics", {}).get("oos") or {})
        caption = (
            f"✅ {payload.get('name')} — {payload.get('asset')} {payload.get('timeframe')}\n"
            f"Verdict: {payload.get('verdict')} (score {payload.get('score')})\n"
            f"OOS: {oos.get('trades', 0)} trades · Sharpe {oos.get('sharpe', 0)} · "
            f"PF {oos.get('profit_factor', 0)} · DD {oos.get('max_drawdown_pct', 0)}%\n"
            f"sha256 {str(payload.get('checksum', ''))[:16]}…"
        )
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
                with path.open("rb") as fh:
                    resp = await client.post(
                        f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendDocument",
                        data={"chat_id": cfg.telegram_chat_id, "caption": caption[:1000]},
                        files={"document": (path.name, fh, "application/zip")},
                    )
            resp.raise_for_status()
            return {"status": "ok", "message_id": resp.json().get("result", {}).get("message_id")}
        except Exception as exc:
            self.log(f"telegram delivery failed: {exc}", level="error")
            return {"status": "failed", "error": str(exc)}

    async def _gdrive(self, path: Path) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._gdrive_sync, path)
        except Exception as exc:
            self.log(f"drive delivery failed: {exc}", level="error")
            return {"status": "failed", "error": str(exc)}

    @staticmethod
    def _gdrive_sync(path: Path) -> dict[str, Any]:
        """Service-account upload. Requires the optional ``delivery`` extra."""
        try:
            from google.oauth2 import service_account  # type: ignore
            from googleapiclient.discovery import build  # type: ignore
            from googleapiclient.http import MediaFileUpload  # type: ignore
        except ImportError:
            return {
                "status": "unavailable",
                "error": "install the optional extra: pip install -e 'backend[delivery]'",
            }

        cfg = settings()
        creds = service_account.Credentials.from_service_account_file(
            cfg.gdrive_credentials_file, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        metadata: dict[str, Any] = {"name": path.name}
        if cfg.gdrive_folder_id:
            metadata["parents"] = [cfg.gdrive_folder_id]
        created = (
            service.files()
            .create(
                body=metadata,
                media_body=MediaFileUpload(str(path), mimetype="application/zip", resumable=True),
                fields="id, webViewLink",
            )
            .execute()
        )
        return {
            "status": "ok",
            "file_id": created.get("id"),
            "url": created.get("webViewLink"),
        }


__all__ = ["DeliveryAgent", "DocWriterAgent", "PackagerAgent"]
