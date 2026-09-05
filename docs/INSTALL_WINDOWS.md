# Installing KRISH on a Windows VPS

Windows is a good host for this: MetaTrader 5 only runs on Windows, so the same
box can later run the MT5 bridge for live automation.

On Windows, KRISH runs **natively** — no Docker, no Redis, no Postgres. It uses an
in-process message bus and a SQLite database, which is plenty for one machine.

---

## Step 1 — Install Python

1. Download **Python 3.12 (64-bit)**: <https://www.python.org/downloads/windows/>
2. Run the installer and **tick "Add python.exe to PATH"** on the first screen.
   This is the step everyone misses, and nothing works without it.
3. Choose *Install Now*.

Verify — open a **new** PowerShell window:

```powershell
python --version
```

You should see `Python 3.12.x` (3.11 or newer is fine).

## Step 2 — Install Git

1. Download: <https://git-scm.com/download/win>
2. Install with all the defaults.

Verify:

```powershell
git --version
```

## Step 3 — Get the code

```powershell
cd C:\
git clone https://github.com/kirox0009-boop/KRISH-STARTEGY-.git krish
cd C:\krish
```

## Step 4 — Install KRISH

```powershell
scripts\windows\install.bat
```

This creates a virtual environment in `.venv`, installs every dependency, copies
`.env.example` to `.env`, and prints the agent roster to prove it works. It takes
2–5 minutes.

## Step 5 — First run

```powershell
scripts\windows\start.bat
```

Then open <http://localhost:8000> in the VPS browser.

You should see the control room: 25 agents, and within a minute the first cycle
starting — data being fetched, strategies designed, backtested and judged live.

Press `Ctrl+C` in the terminal to stop.

Prefer to watch one cycle in the terminal instead?

```powershell
scripts\windows\start.bat cycle GOLD H1
```

## Step 6 — Run it 24/7

Right-click **`scripts\windows\install-service.bat`** → **Run as administrator**.

That registers a Windows scheduled task which:

- starts KRISH automatically on boot, before anyone logs in
- restarts it if it ever crashes
- logs to `C:\krish\var\logs\service.log`

Manage it:

```powershell
schtasks /query  /tn KRISH        # status
schtasks /end    /tn KRISH        # stop
schtasks /run    /tn KRISH        # start
schtasks /delete /tn KRISH /f     # remove
```

## Step 7 — Reach it from your own computer (optional)

By default the control room is only reachable on the VPS itself. To open it from
your laptop:

1. Allow the port through Windows Firewall (PowerShell as administrator):

   ```powershell
   New-NetFirewallRule -DisplayName "KRISH" -Direction Inbound -LocalPort 8000 `
     -Protocol TCP -Action Allow
   ```

2. Browse to `http://<your-vps-ip>:8000`.

> **Security warning.** There is no login on the control room yet
> (authentication is Phase 6 in [ROADMAP.md](ROADMAP.md)). Anyone who reaches
> that port can control the factory. Until auth exists, prefer either:
> - leaving the port closed and using the VPS's own browser, or
> - restricting the firewall rule to your own IP:
>   `-RemoteAddress <your.home.ip>`

---

## Configuration

| File | What it controls |
|---|---|
| `.env` | secrets and infrastructure (Telegram, Drive, LLM key, ports) |
| `config\factory.yaml` | how strict the factory is — thresholds, cycle cadence, costs |
| `config\assets.yaml` | which instruments to trade |

Both YAML files are editable live from the control room; agents reload on change.

### Turn on Telegram delivery

1. Message **@BotFather** on Telegram, send `/newbot`, copy the token.
2. Message your new bot once, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
3. In `.env`:

   ```ini
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=987654321
   ```

4. In `config\factory.yaml`, set `delivery.telegram_enabled: true`.
5. Restart: `schtasks /end /tn KRISH` then `schtasks /run /tn KRISH`.

Every strategy that passes now arrives on your phone as a ZIP with a summary.

---

## Troubleshooting

**`'python' is not recognized`**
Python is not on PATH. Re-run the Python installer, choose *Modify*, and enable
"Add python.exe to PATH". Then open a **new** terminal.

**`scripts\windows\install.bat` fails while installing packages**
Usually a network/proxy issue. Re-run it — pip resumes. If it mentions a compiler
error, install the
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**Port 8000 already in use**
Set a different port in `.env`: `KRISH_API_PORT=8080`.

**No strategies passing**
That is normal and correct — most random strategies are worthless, and the judge
is deliberately harsh. It is also partly a data limit: free Yahoo intraday history
caps at about 729 days, which keeps out-of-sample trade counts under the 30-trade
minimum. Two options:

- lower the bar in `config\factory.yaml` (`judge.min_oos_trades`, `min_oos_sharpe`)
  while you are experimenting, or
- use `D1` timeframes, which have decades of history, or
- wait for Phase 1, which adds proper data sources.

**Where do finished strategies land?**
`C:\krish\var\packages\*.zip`, and in the control room's Deliveries panel with a
download link.

**How do I see what went wrong?**
`C:\krish\var\logs\krish.log`, or the live bus panel in the control room.
