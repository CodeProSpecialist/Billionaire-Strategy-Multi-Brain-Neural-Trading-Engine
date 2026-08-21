# Billionaire Alpaca Robot

An automated day-trading bot for Alpaca that runs a rules-based dip-buy scanner on the NASDAQ-100 + S&P 500 large caps, gated by a **four-brain neural committee** whose votes are debated live on a WebSocket dashboard.

---

New Upgrade on August 21, 2026. 

## What it does

- Scans a curated NASDAQ-100 + S&P 500 large-cap universe for dip-buy setups (RSI oversold, MACD bullish crossover, above 200-day SMA, multi-timeframe confirmation).
- Detects the current market regime (bull / sideways / bear / panic) from SPY + VIX and adapts sizing, thresholds, and exit rules per regime.
- Feeds every candidate through a committee of four brains that debate on a shared "Brain Trading Floor" before a Chair issues the final approve / deny decision.
- Exits positions via a layered stack: **chandelier ATR trailing stop** (dynamic, volatility-adjusted) → **profit monitor** (peak-giveback fraction) → **scale-out stages** (fixed % milestones) → **hard stop** (2 × entry ATR).
- Streams live state to a browser dashboard where you can pause, resume, sell-all, emergency-stop, adjust settings, and manage the two-tier blacklist.

---

## Architecture at a glance

```
                     ┌────────────────────────────────────────┐
                     │   Rules scanner (dip-buy universe)     │
                     └─────────────────┬──────────────────────┘
                                       │ candidate symbol
                                       ▼
     ┌────────────────────────────────────────────────────────────────┐
     │                    Brain Trading Floor debate                  │
     │                                                                │
     │   TRADING_A     BACKTEST_C     RISK_B        PORTFOLIO_D       │
     │   (neural,      (5y sim,       (neural,      (neural,          │
     │    P(win))       win-rate)      P(safe))      size×/aggr)      │
     │        │             │             │             │             │
     │        └─────────────┴──────┬──────┴─────────────┘             │
     │                             ▼                                  │
     │                     CHAIR (Brain E)                            │
     │                aggregates votes, applies rules,                │
     │                posts every vote + final decision               │
     └─────────────────────────────┬──────────────────────────────────┘
                                   │ APPROVE + adjusted notional
                                   ▼
                        Alpaca API: submit_order
                                   │
                                   ▼
                            Position opens
                                   │
                                   ▼
     ┌────────────────────────────────────────────────────────────────┐
     │  Exit stack: Chandelier → Profit Monitor → Scale-out → Hard    │
     └────────────────────────────────────────────────────────────────┘
```

---

## The four brains

| Brain | Type | Answers | Authority |
|---|---|---|---|
| **A — Trading** | Neural net (Conv1D + LSTM on frozen expert foundation) | "Does this symbol look like a good buy right now?" | Advisory (nudges score) |
| **B — Risk** | Neural net (12-feature Dense on frozen risk-detector foundation) | "Can the account safely take another position right now?" | Veto (armed mode) |
| **C — Backtest** | Statistical (per-symbol 5-year replay of the buy signal under actual exit rules) | "Historically, does this signal make money on this symbol?" | Veto (armed by default) |
| **D — Portfolio Manager** | Neural net (10-feature Dense on frozen PM heuristic foundation) | "Given recent trajectory, should we scale up, down, or hold neutral?" | Advisory (sizing nudges) |
| **E — Chair** | Deterministic aggregator | Reads all votes, applies rules, issues final decision | Executes |

**Brain A** is pretrained on 20,000 historical setups drawn from NASDAQ-100 + S&P 500 large caps, then fine-tuned continuously on the bot's own closed trades.

**Brain B and D** are pretrained on 20,000 synthetic scenarios that encode the ground-truth "safe / dangerous" (B) and "correct action" (D) rules. The head learns a smooth interpolation of the frozen foundation's expert detectors.

**Brain C** is not a neural network — it's an honest statistical calculation. It walks 5 years of daily bars per symbol, replays the buy signal on each bar, simulates the outcome under the actual exit rules, and reports the historical win rate.

**Brain E (Chair)** is deterministic aggregation, not an LLM conversation. The floor discussion is a human-readable audit log of a math-based decision, not the mechanism by which the decision is reached.

---

## Safety layers (in order of trigger sensitivity)

1. **Broker/arithmetic checks** — buying power, cash buffer, exposure cap, margin health, PDT accounting. Cannot be overridden.
2. **Trade Governor** — pauses new entries after N consecutive losses; locks the trading day once a profit target is hit.
3. **Blacklist** — losing trades auto-add to a 72-hour temporary blacklist. Permanent blacklist for symbols you never want to trade.
4. **Per-symbol mute** — statistical soft-mute for symbols with poor personal track record.
5. **Chair vetoes** — Brain B (armed), Brain C (armed).
6. **Chandelier ATR stop** — dynamic trailing stop, tightens as volatility drops.
7. **Profit monitor** — peak-giveback fraction locks in profits after +0.5%.
8. **Scale-out stages** — fixed milestones close position tranches.
9. **Hard stop** — 2 × entry ATR from entry, the deep floor.

---

## Dashboard

Real-time WebSocket dashboard at `ws://localhost:8765`. Open `alpaca_dashboard.html` in any browser to connect.

**Tiles:**
- 💰 Account (equity, buying power, cash, margin health)
- 📈 Market Regime (regime, VIX, weak-bull downgrade flag)
- 🛑 Trade Governor (can-trade, consecutive losses, cooldown, day-lock)
- 🛡️ Risk Brain B (P(safe), min threshold, top risk features)
- 💼 Portfolio Manager D (size ×, aggression, concentration ×, scale up/down tag)
- 🧠 Brains (backtest mode + threshold, chandelier mode + ATR ×, brain trust scores)
- 📊 Live Positions (per-symbol tiles with entry, live price, gain, chandelier stop)
- 💎 Profit Monitor (per-symbol peak price, armed state, last seen)
- 📋 Recent Trades (last 25 buys/sells)
- 🧠 Brain Trading Floor (scrolling live debate feed + operator terminal)
- ⏳ 72h Auto Blacklist (chips + add form)
- 🚫 Permanent Blacklist (chips + confirm-to-add)
- 🔇 Muted (auto) — statistical soft-mute list
- ⚙️ Settings (14 dashboard-editable knobs, apply immediately)
- 🤖 Robot Control (pause, resume, sell-all, emergency stop)

**Operator terminal (bottom of Brain Trading Floor):**
```
status | pause | resume | sell_all | blacklist AAPL 72h | blacklist AAPL perm |
unblacklist AAPL | clear | help
```
Arrow keys navigate command history.

---

## Files

- `billionaire-strategy-buy-lowest-price-stock-market-robot.py` — main bot (~11,400 lines, single file by design)
- `alpaca_dashboard.html` — dashboard client (~900 lines, no framework, pure HTML/CSS/JS)

**State files** (auto-created under the bot's directory):
- `ml_brain_model/` — Brain A weights + metadata
- `brain_b_risk_model/` — Brain B weights + metadata
- `brain_d_pm_model/` — Brain D weights + metadata
- `persymbol_stats.json` — per-symbol win rate + P&L (survives restart)
- `blacklist.json` — permanent + temporary blacklist entries

---

## Setup

### 1. Alpaca API credentials (required)

The bot reads three environment variables at startup:

| Variable | Purpose | Where to get it |
|---|---|---|
| `APCA_API_KEY_ID` | Alpaca API key ID | [alpaca.markets](https://alpaca.markets) → Account → API Keys |
| `APCA_API_SECRET_KEY` | Alpaca API secret key | Same page (shown once when key is generated) |
| `APCA_API_BASE_URL` | API endpoint | `https://paper-api.alpaca.markets` for paper trading, `https://api.alpaca.markets` for live |

### 2. Add them to `~/.bashrc` (Linux/macOS)

Open your `~/.bashrc` in an editor:

```bash
nano ~/.bashrc
```

Append these lines to the bottom of the file (replace the placeholder values with your real credentials):

```bash
# ─── Alpaca API credentials for the Billionaire Robot ─────────────
export APCA_API_KEY_ID="PK_YOUR_KEY_ID_HERE"
export APCA_API_SECRET_KEY="YOUR_SECRET_KEY_HERE"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
```

Save and exit (`Ctrl-O`, `Enter`, `Ctrl-X` in nano).

Reload your current shell so the new variables take effect:

```bash
source ~/.bashrc
```

Verify they're set:

```bash
echo $APCA_API_KEY_ID
echo $APCA_API_BASE_URL
```

Both should print the values you set. `APCA_API_SECRET_KEY` will also print if you check it, but treat that value like a password — don't paste it into terminals other people can see.

**Security tips:**
- `chmod 600 ~/.bashrc` restricts read access to just your user.
- Never commit `~/.bashrc` or any file containing these values to git.
- Start with `paper-api.alpaca.markets`. Only switch to `api.alpaca.markets` (live money) after you've watched the bot run for several sessions and understand its behavior.

### 3. System dependencies

The bot's auto-bootstrap installs Python packages via `pip install --quiet` for anything missing, but **TA-Lib requires the system C library** which pip can't install:

```bash
# Debian/Ubuntu
sudo apt-get install build-essential wget
wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# macOS (Homebrew)
brew install ta-lib
```

### 4. First run

```bash
python billionaire-strategy-buy-lowest-price-stock-market-robot.py
```

First-run behavior:
- Bootstrap installs any missing Python packages (a few seconds to a minute).
- **Brain A pretrains on 20,000 historical setups** — takes 15–45 minutes depending on network speed and CPU. Fetches ~5 years of daily bars for ~160 tickers, generates 20k labeled sequences, trains 6 epochs.
- **Brain B and D pretrain on 20,000 synthetic scenarios each** — seconds on CPU.
- Once pretraining completes, the trading loop starts.

Subsequent runs skip all pretraining (models load from disk instantly).

### 5. Connect the dashboard

Once the bot is running, open `alpaca_dashboard.html` in any browser. It auto-connects to `ws://localhost:8765` and auto-reconnects on drop.

If running the bot on a remote server, either:
- Use SSH port forwarding: `ssh -L 8765:localhost:8765 you@server`
- Or change `DASHBOARD_WS_HOST` from `'127.0.0.1'` to `'0.0.0.0'` in the bot source (⚠️ then anyone on your network can reach the dashboard — no authentication is built in).

---

## Default configuration

| Setting | Default | What it does |
|---|---|---|
| `CHANDELIER_MODE` | `armed` | Dynamic ATR trailing stop is active |
| `CHANDELIER_ATR_MULT` | `3.0` | Chuck LeBeau's classic constant |
| `BACKTEST_BRAIN_MODE` | `armed` | Backtest vetoes buys with poor historical win rate |
| `BACKTEST_MIN_WIN_RATE` | `0.50` | Coin-flip threshold — blocks worse-than-random |
| `BACKTEST_MIN_SAMPLES` | `20` | Below this, brain abstains (no veto) |
| `BRAIN_B_MODE` | `shadow` | Risk brain observes only — flip to `armed` after review |
| `BRAIN_B_MIN_SAFE` | `0.40` | Veto threshold when armed |
| `BRAIN_D_MODE` | `shadow` | Portfolio manager observes only — flip to `armed` after review |
| `CONSEC_LOSS_COOLDOWN` | `5` | Losses in a row before cooldown |
| `COOLDOWN_SECONDS` | `1800` | 30-minute pause after loss streak |
| `DAILY_PROFIT_LOCK_PCT` | `0.03` | +3% session gain locks the day |
| `DASHBOARD_WS_PORT` | `8765` | WebSocket port for the dashboard |

**All settings are live-editable from the dashboard's Settings tile** — no restart needed. Changes are applied to the running process.

---

## Design principles

1. **Multiple safety layers, no single override.** Broker/arithmetic checks always run, regardless of what the brains say. A neural net bug can't bypass margin protection.
2. **Neural nets are advisors, not black-box deciders.** Every brain prints its reasoning trail. Every trade decision is a debate on the floor with the Chair's vote visible.
3. **New capabilities ship in shadow mode.** New neural brains default to observation-only. You watch them for a session, then flip to armed if the numbers look sane.
4. **Rules-based scanning as the foundation.** The bot's core buy scanner is deterministic rules (RSI, MACD, SMA, multi-timeframe). Brains refine and gate; they don't originate signals.
5. **Statistical, not aspirational.** Where a real calculation exists (backtest brain, per-symbol performance), it stays statistical. We don't wrap math in neural nets just to add symmetry.

# License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

Copyright (C) 2026 CodeProSpecialist

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).

---

**Disclaimer:** This software is provided for informational and educational purposes only. The author is not liable for any losses, damages, or other consequences resulting from the use of this software or any trades made with it. **Trade at your own risk.**

```

For
