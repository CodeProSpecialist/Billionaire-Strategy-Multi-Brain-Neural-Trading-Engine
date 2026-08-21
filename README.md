New Upgrade on August 21, 2026. 

# Billionaire Alpaca Robot

An automated, single-file NASDAQ / S&P 500 daytrading bot that trades through the Alpaca API. Six neural-network "brains" collaborate on every entry, a live web dashboard shows what the bot is thinking, and every risky action is gated by a safety-first ensemble that can be tightened, loosened, or shut off from the browser without restarting the bot.

**Version 10** — ~12,400 lines of Python in one file, plus a static HTML dashboard.

> ⚠️ This software trades real money. Paper-trade first (`APCA_API_BASE_URL=https://paper-api.alpaca.markets`). Nothing in this repo is investment advice.

---

## What it does

Every ~60 seconds the bot:

1. **Scans** the S&P 500 + NASDAQ universe for dip-buy setups (RSI, MACD, SMA structure, volume, VWAP, multi-timeframe intraday pullback).
2. **Scores** each candidate with a rule-based signal, then adjusts the score with up to six neural brains (see below).
3. **Votes** on every candidate through the Chair (Brain E) — any brain can veto, the portfolio manager can resize.
4. **Enters** the top-ranked survivors up to `MAX_NEW_POSITIONS_PER_CYCLE` (default 3) using ATR-based position sizing.
5. **Exits** open positions through a stack of protective rules: hard stop-loss, chandelier ATR trailing stop, bad-news stop tightener, scaled profit exits, and a profit-monitor peak follower.

A **bull-market strategy** runs alongside the ML-scored path during confirmed uptrends (10:02 AM – 3:35 PM ET), buying momentum breakouts with a 1% trailing stop and 0.5% take-profit.

---

## The six brains

| Brain | Role | Type | Authority |
|---|---|---|---|
| **A · ML Trading Brain** | Predicts P(this dip-buy wins) from a 20-day sequence of 32 daily features (10 stock + 20 NASDAQ proxy + 2 relative-strength). Conv1D → 2× LSTM head on a frozen hand-weighted foundation. | Sequence model, focal loss γ=1.2 | Advisory (± 1.5 score points) |
| **B · Risk Brain** | Portfolio-safety gate. 12 features (margin, exposure, VIX, consec losses, session P&L, drawdown concentration, churn, regime risk) → P(safe). | Feedforward NN, frozen foundation + trainable head | **Veto when armed** if P(safe) < 0.40 |
| **C · Backtest Brain** | Walks 5 years of daily history, replays the simplified dip-buy on this symbol, simulates exits under the bot's actual rules, vetoes if regime-weighted win rate < 50%. | Deterministic historical replay, 4-hour per-symbol cache | **Veto when armed** |
| **D · Portfolio Manager** | Reads trajectory of the account (7d/30d return, Sharpe, max drawdown, win rate, profit factor, exposure) → outputs `size_multiplier`, `aggression`, `concentration_multiplier`. | Feedforward NN, 3-head sigmoid output | Nudge only (never vetoes) |
| **E · Chair** | Deterministic ensemble aggregator. Collects every brain's vote, posts them all to the Brain Trading Floor, applies Portfolio_D sizing when armed, blocks the trade on any DENY. | Rule-based aggregation (not an LLM) | Final say |
| **F · Bullish-Trend Picker** | Independent second opinion on trend quality. 12 features including RSI, MACD, SMA slope/structure, ATR%, volume, and Alpaca news sentiment → P(bullish). | Feedforward NN, 9-unit frozen foundation | Additive bump ± 0.5 score points |

Plus a **RISK_TIME** gate (early-morning entry restriction, see Config below) that votes DENY when armed and out of window.

---

## Safety systems

* **Hard stop-loss** at 2× ATR from entry (min −3%), sized so `RISK_PER_TRADE_PCT = 1%` of equity is the actual maximum loss per position.
* **Chandelier ATR trailing stop** — `peak_since_entry − 3× entry_ATR`, ratchets up only, fires between the hard stop and the profit monitor.
* **Bad-news stop tightener** — when Brain F detects negative Alpaca news on an owned position (rolling neg-only sentiment ≥ 0.20), the stop tightens to 0.3% below current price. Only *raises* the stop, never lowers it.
* **Profit monitor** — arms after a position is +profit, follows the peak, exits on a configurable retracement.
* **Scaled exits** at profit milestones, moving the floor to breakeven after the first stage.
* **Trade Governor** — 5 consecutive losses trip a 30-minute cooldown; 3% daily profit lock ends the session.
* **Auto blacklist** — losing closes add the symbol to a 72-hour timeout; a permanent list is also supported.
* **Per-symbol mute** — a symbol with ≥5 trades, negative P&L, and win rate < 30% is muted for 24 hours.
* **Early-morning gate** — no new entries before 10:05 AM ET unless the stock is down ≥ 2% from yesterday's close (skip the thin, fake-out-prone open).
* **Margin-health floor** — buys suspended when equity / long_market_value < 30%.

---

## Live web dashboard

A tile-based, dark-themed dashboard runs on `127.0.0.1:8765` (WebSocket, 1 Hz). Open `alpaca_dashboard.html` in a browser.

The dashboard shows:

* **Account, Market Regime, Trade Governor** — equity, buying power, VIX, current regime, cooldown state.
* **Risk Brain (B), Portfolio Mgr (D), Bullish Picker (F)** — mode, key output (P(safe) / size × / P(bullish)), model status, top features driving the current call.
* **Brains** tile — Backtest / Chandelier mode + thresholds, Brain Trust rolling accuracy.
* **Live Positions** — entry, current, gain %, chandelier stop.
* **Profit Monitor** — armed positions with peak, floor, and last-seen tick.
* **Recent Trades** — last 25 fills.
* **Brain Trading Floor** — running feed of every brain's decisions, user commands, lessons from closed trades. Free-text terminal at the bottom accepts `status | pause | resume | sell_all | blacklist SYM 72h | unblacklist SYM | clear | help`.
* **72-hour auto blacklist**, **permanent blacklist**, **muted symbols**.
* **Settings** — 15+ live-mutable knobs (all three brain modes, thresholds, chandelier ATR multiplier, cooldown seconds, mute win-rate, etc.). Changes apply immediately, no restart.
* **Robot Control** — pause / resume / sell-all / emergency-stop, all with browser-side confirmation.

---

## Requirements

**Python** 3.10+ (tested on 3.12).

The bot **bootstraps its own dependencies** on first launch — it runs `pip list --format=freeze` and quietly installs anything missing. You do not need to run `pip install` yourself. The auto-installed packages are:

```
alpaca-trade-api    alpaca-py           pytz                numpy
TA-Lib              yfinance            SQLAlchemy          ratelimit
pandas-market-calendars                 pandas              schedule
websockets                              requests
tensorflow *
```

\* TensorFlow is optional but recommended. Without it, all six brains degrade gracefully — the rule-based signal still trades, dashboards still work. With it, the bot auto-detects an NVIDIA GPU and swaps in the CPU / GPU wheel accordingly, restarting once if the wheel changed.

**TA-Lib** requires the system C library. On Debian/Ubuntu:

```bash
sudo apt-get install libta-lib-dev
```

macOS:

```bash
brew install ta-lib
```

---

## Install & run

```bash
git clone https://github.com/<you>/billionaire-alpaca-robot
cd billionaire-alpaca-robot

# 1. Set your Alpaca credentials (paper first!)
export APCA_API_KEY_ID='PK...'
export APCA_API_SECRET_KEY='...'
export APCA_API_BASE_URL='https://paper-api.alpaca.markets'   # or live: https://api.alpaca.markets

# 2. Just run it — deps install on first launch
python3 billionaire-strategy-buy-lowest-price-stock-market-robot.py
```

First launch will:

1. Bootstrap missing pip packages (silent, one-time).
2. Detect GPU vs CPU and swap the TensorFlow wheel if needed (`execv` restart, one time).
3. Pretrain **Brain B** (risk, ~seconds), **Brain D** (portfolio, ~seconds), **Brain A** (ML trading, ~minutes on 20k historical examples), and **Brain F** (bullish picker, ~5s on 20k synthetic examples).
4. Scan the S&P 500 for the initial watchlist (~7 seconds via Alpaca IEX).
5. Wait for market open, then start trading.

Open the dashboard in a separate browser tab:

```bash
firefox alpaca_dashboard.html
# or just open the file — it connects to ws://127.0.0.1:8765 automatically
```

---

## Configuration

Everything is a module-level constant near the top of the .py file. The most-tuned knobs:

### Position sizing & risk

```python
RISK_PER_TRADE_PCT              = 0.01     # 1% of equity risked per position
MAX_ALLOCATION_PER_SYMBOL       = 600.0    # $ cap
MAX_NEW_POSITIONS_PER_CYCLE     = 3        # rank all candidates, buy top-N
MAX_PORTFOLIO_EXPOSURE_PCT      = 0.98     # of buying power
MAX_LEVERAGE                    = 1.0      # 2.0 to enable Reg-T intraday
MAINTENANCE_MARGIN_FLOOR_PCT    = 0.30     # halt buys below this
```

### Exit rules

```python
HARD_STOP_ATR_MULTIPLIER   = 2.0    # entry − 2×ATR
HARD_STOP_MIN_PCT          = 0.03   # floor at −3% for low-ATR stocks
CHANDELIER_ATR_MULT        = 3.0    # LeBeau default
CHANDELIER_MIN_HOLD_SECS   = 300    # ignore entry-noise stop-outs
BAD_NEWS_STOP_THRESHOLD    = 0.20   # neg-news sentiment that triggers tightener
BAD_NEWS_TIGHTEN_STOP_PCT  = 0.003  # 0.3% below current on bad news
```

### Brain modes (all live-toggle from dashboard)

```python
BRAIN_B_MODE           = 'shadow'   # 'armed' | 'shadow'
BRAIN_B_MIN_SAFE       = 0.40
BRAIN_D_MODE           = 'shadow'
BRAIN_F_MODE           = 'armed'    # 'armed' | 'shadow' | 'off'
BRAIN_F_MIN_PROB_TO_BUMP = 0.55
BRAIN_F_MAX_SCORE_BUMP = 0.5
BACKTEST_BRAIN_MODE    = 'armed'    # 'armed' | 'advisory'
BACKTEST_MIN_WIN_RATE  = 0.50
CHANDELIER_MODE        = 'armed'    # 'armed' | 'shadow' | 'off'
EARLY_MORNING_GATE_MODE = 'armed'   # 'armed' | 'shadow' | 'off'
EARLY_MORNING_CUTOFF_ET = time(10, 5)
EARLY_MORNING_MIN_DROP_PCT = 0.02
```

### Governor

```python
CONSEC_LOSS_COOLDOWN         = 5
COOLDOWN_SECONDS             = 1800     # 30 min
DAILY_PROFIT_LOCK_PCT        = 0.03     # 3% session gain ends the day
PERSYMBOL_MUTE_WINRATE       = 0.30
PERSYMBOL_MIN_TRADES         = 5
PERSYMBOL_MUTE_DURATION_SECONDS = 86400  # 24h
BLACKLIST_TEMP_DURATION_SECONDS  = 259200  # 72h
```

### Dashboard

```python
DASHBOARD_WS_ENABLED   = True
DASHBOARD_WS_HOST      = '127.0.0.1'   # '0.0.0.0' to expose on LAN
DASHBOARD_WS_PORT      = 8765
```

---

## Data sources

| What | Where | Latency / cost |
|---|---|---|
| Daily bars (scanner, indicators, ML features) | Alpaca IEX free plan (primary) → yfinance (fallback) | S&P 500 in ~7s vs ~540s on yfinance |
| Latest trade for bull-strategy live sampling | Alpaca IEX `get_stock_latest_trade` | ~100 ms |
| Intraday 1m / 5m / 60m VWAP | yfinance | Not yet migrated to Alpaca |
| Order execution, positions, cash, portfolio history | Alpaca trading API | Real-time |
| News sentiment (Brain F, bad-news stop) | Alpaca News API v1beta1 (Benzinga) | Free with existing keys, 15-min per-symbol cache |
| VIX / SPY regime | yfinance (`^VIX`, `SPY`) | Cached per session |

Ticker convention: yfinance uses `BRK-B` (dash), Alpaca uses `BRK.B` (dot). The bot keeps yfinance form internally and converts at the Alpaca boundary via `to_alpaca()` / `to_yf()`.

---

## File layout

```
billionaire-strategy-buy-lowest-price-stock-market-robot.py   # the bot (~12.4k lines, single file)
alpaca_dashboard.html                                         # dashboard UI (open in browser)
ml_brain/                                                     # created at runtime
  brain_a_model/                                              #   ML trading brain (Brain A)
  brain_b_risk_model/                                         #   Risk brain (Brain B)
  brain_d_portfolio_model/                                    #   Portfolio manager (Brain D)
  brain_f_bullish_model/                                      #   Bullish picker (Brain F)
schedule_state.json                                           # last-run bookkeeping
blacklist.json                                                # persistent blacklist
trades.db                                                     # SQLite: positions, history, features
```

Model files (`model.keras`) and their `meta.json` are auto-detected as stale (wrong feature count, wrong architecture) and rebuilt without operator intervention.

---

## Operating notes

* **Paper first.** Every knob works the same on paper — validate for a week before pointing at a live account.
* **Portfolio summary every hour.** The bot prints 24h / 7d / 14d / 30d equity gain in the terminal, cross-referenced against NYSE trading sessions (skips weekends and holidays).
* **Losing closes teach.** Every losing close blacklists the symbol for 72 hours and fine-tunes the ML brain overnight (17:00 ET daily retrain on the last 15k trades). Wins just log observations.
* **Weak-bull downgrade.** If regime is BULL past 10:35 ET and no trades have fired today, the regime downgrades to SIDEWAYS for the rest of the day (avoids "bull that never actually pulls back to buy").
* **Emergency stop is a flag, not a kill.** It stops new entries and cancels open orders; positions still exit via their normal rules unless you click sell-all.

---

## License

GNU General Public License v3.0. This program is free software: you can redistribute it and/or modify it under the terms of the GNU GPL v3 as published by the Free Software Foundation. Distributed WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See `LICENSE` for the full text, or <https://www.gnu.org/licenses/gpl-3.0.html>.

## Disclaimer

Automated trading involves substantial risk of loss. The bot's brains are trained on limited data, its assumptions may be wrong, and market conditions change faster than any backtest can capture. Use paper trading extensively. Understand every rule before enabling live trading. You are responsible for anything this software does to your account.
