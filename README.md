# Billionaire Strategy Stock Market Trading Robot

***** New Upgraded Version 17 on 8-29-2026 *****

Single-file automated Alpaca daytrading bot for NASDAQ + S&P 500 equities, driven by an ensemble of neural-net "brains" and a rules-based dip-buy / bull-momentum strategy. Ships with a live browser dashboard.

**Files**
- `billionaire_strategy_buy_lowest_price_stock_market_robot.py` — the bot (~12.5k lines, one file, banner "Version 17")
- `alpaca_dashboard.html` — the dashboard UI it serves at `ws://127.0.0.1:8765`

---

## Quick start

```bash
export APCA_API_KEY_ID=your_key
export APCA_API_SECRET_KEY=your_secret
export APCA_API_BASE_URL=https://paper-api.alpaca.markets   # or live
python billionaire_strategy_buy_lowest_price_stock_market_robot.py
```

Then open `alpaca_dashboard.html` in a browser. It connects to the bot over WebSocket on `<hostname>:8080` after 
you setup a seperate webserver of your choice on port 8080. 

---

## What's new in Version 17 (Aug 28, 2026)

### Entry gates
- **Hard 10:05 AM Eastern buy-cycle block.** `_buy_stocks_body` returns immediately if the wall clock is before `EARLY_MORNING_CUTOFF_ET`. Never buy before 10:05 AM Eastern time because the stock market is nothing but confusing noise for the first 35 minutes every day. No per-symbol overnight-drop escape hatch, and the KSB per-symbol fetches are skipped too so rate-limit budget isn't spent on candidates we can't act on.
- **2-day momentum gate (hard).** Buy candidates now must show strictly positive 2-day close-to-close momentum on daily bars (`close[-1] > close[-3]`) — enforced right after the 90d history fetch and before `compute_buy_score`. Symbols failing the check are skipped with the failing % printed.

### Position sizing
- **Half-Kelly per-symbol sizing.** `PerSymbolPerformance` now also records `sum_win` / `sum_loss` on each close and exposes `kelly_fraction()` — the classic Kelly `f* = (p·b − q)/b` halved for small-sample robustness and clamped to `KELLY_MAX_FRACTION` (5% of equity). When a symbol has ≥ `KELLY_MIN_TRADES = 10` closed trades (with at least one win **and** one loss), the non-bull, non-$1 sizing path uses `kelly_fraction × total_equity` as the notional; otherwise it falls back to the existing ATR risk sizing. Toggle: `USE_KELLY_SIZING`.

### Chandelier-Exit scoring (both sides)
- **Buy side (`compute_buy_score`).** A daily-bar close cross **up** through the 22-day / 3×ATR Chandelier long-stop within the last 5 bars now scores as `chandelier_buy_signal` (weight 2 in BULL/SIDEWAYS, 3 in BEAR/PANIC). A 3-of-N confirmation guard (rising short-term price, volume support, positive momentum; ≥ 2 required) keeps a bare cross-up on a downtrend bar from adding score.
- **Sell side (`evaluate_chandelier_sell_signal` + `_sell_stocks_body`).** A close cross **down** through the same stop is scored symmetrically (falling short-term price, volume support on the down move, negative momentum; ≥ 2 confirms). Policy on a confirmed daily SELL:
  - **RSI(14) falling** (last < prev **and** last < value 3 bars back) → exit **now** via the standard escalation chain (label `CHANDELIER SELL + RSI FALLING`).
  - **RSI not falling** → do **not** market-sell. Instead place an **extremely tight** stop at `current_price × (1 − CHANDELIER_DAILY_SELL_TIGHTEN_PCT)` (default `0.25%`). If already breached, exit (label `CHANDELIER TIGHT STOP`); otherwise hold and let the intraday ATR trailing chandelier keep running.

### Per-symbol scanner surface
- New `_DayTradeBrain.symbol_scan(symbol)`, `get_technical_snapshot()` and `format_technical_snapshot()` produce a compact per-symbol readout: MACD line/signal/hist, RSI(14), 10-day momentum, trend, bull/bear flags, chandelier state, last BUY/SELL timestamps, and p(win).
- `get_monitored_symbols_snapshot(max_symbols=25)` batches the whole watchlist behind a 60-second cache so the dashboard can broadcast live TA cheaply.
- Dashboard adds two full-row cards, **🧠 Daytrade Brain — Symbol Scans** and **👁 Monitored Symbols**, wired through `renderDaytradeScans()` / `renderMonitoredSymbols()`.

### Naming
- The bot's Python file is now snake_case: **`billionaire_strategy_buy_lowest_price_stock_market_robot.py`**.

---

## Earlier upgrade highlights (Aug 2026)

The dashboard ↔ bot bridge and buy-cycle stability got a substantial overhaul:

### Dashboard / WebSocket
- **WebSocket transport rewritten from scratch** on both sides (bot's `DashboardWSEngine` + browser `WS` module). Per-client 3s send timeout so a slow browser can never stall the broadcast; 8s connect-timeout on the client so the UI never sits in "connecting…" forever.
- **Web-server hosting**: bot binds `0.0.0.0:8080` by default (override with `BOT_DASHBOARD_HOST` / `BOT_DASHBOARD_PORT`) 
It connects to the bot over WebSocket on `<hostname>:8080` after 
you setup a seperate webserver of your choice on port 8080. 
The dashboard connects direct to `ws://<hostname>:8765` by default — works from `file://`, any static HTTP server, or Caddy.
  - `?proxy=1` — same-origin `/ws` (Caddy reverse-proxy)
  - `?ws=ws://host:port` — explicit URL override
  - `?token=...` — auth when `BOT_DASHBOARD_TOKEN` is set on the bot
- **Live connection banner** shows the exact WS URL and close code so misconfigured proxies are diagnosable at a glance.
- **Brain B / D / F settings UI** wired up in the Settings card so the whitelist keys the bot already persists actually have controls.
- **Dead `renderBrains()`** removed; `--cyan` palette token added.

### Bot stability
- **KSB brain-suite serializer** (fixes catastrophic thread leak — py-spy showed 200+ daemon threads all wedged on `KSB.snapshot()`): every KSB entry from this bot now funnels through a single-worker `ThreadPoolExecutor` gated by an inflight mutex. If a KSB call is already running (training, retrain, slow snapshot) callers get `(False, None)` immediately and fall back to cache / skip. Worst-case ever-leaked thread count is 1, not hundreds.
- **Trade during training** — the KSB brain suite trains in-process and holds its internal lock for tens of seconds. Both hot paths now degrade gracefully:
  - `buy_stocks` KSB feed block skips itself when `_KSB_INFLIGHT` shows a call in flight (`[KSB feed] brain suite busy — skipping feed`).
  - `unified_buy_gate` (score gate) uses `_ksb_call(..., timeout=2s)`; if KSB is training, it fail-opens with `{'approved': True, 'reason': 'KSB busy (training) — fail-open'}` so the strategy score alone decides the trade — same posture as `KSB_GATE_MODE='off'`.
- **Pretrain crash under KSB fixed**: `pretrain_all_brains()` short-circuits when KSB owns the brains; error-log path is `getattr`-safe. No more `'RiskSentinel' object has no attribute 'name'`.
- **Network-timeout guards** on the KSB feed loop (45s wall-clock deadline, capped at 25 symbols) and on the SPY/VIX `yf.download` calls (`timeout=15` / `timeout=10`).
- **Settings persistence**: `manager_strictness` now round-trips through `~/.alpaca_bot_settings.json` and re-applies at startup via `MANAGER_BRAIN.set_strictness()`, and appears in the WS `state.settings` payload so the dropdown actually shows the current value.
- **Dashboard WS broadcast interval** raised 1.0 → 3.0 s so ticks can never outrun KSB call rate.

### Environment overrides

| Env var | Default | Purpose |
|---|---|---|
| `BOT_DASHBOARD_HOST` | `0.0.0.0` | WS bind address (`127.0.0.1` for loopback-only) |
| `BOT_DASHBOARD_PORT` | `8765` | WS bind port |
| `BOT_DASHBOARD_TOKEN` | *(unset)* | If set, clients must present `?token=<value>` |

On first launch the bot will:
1. Bootstrap-install any missing dependencies via `pip list` + `pip install` (stdlib-only, no `--break-system-packages`).
2. Auto-detect an NVIDIA GPU and swap `tensorflow` ↔ `tensorflow-cpu` accordingly, then `os.execv`-restart itself once (guarded against loops).
3. Scan the current S&P 500 universe (~504 tickers, 2026 list).
4. Pretrain all brains that don't yet have a saved model on disk — Brain B (risk) → Brain D (portfolio) → ML Brain (Brain A, ~20k synthetic examples) → Brain F (bullish picker).

Total cold start is a few minutes; subsequent starts skip pretraining.

---

## Strategy in one paragraph

The bot runs two parallel entry paths. The **dip-buy path** scores every candidate on RSI/MACD/SMA/ATR structure, then routes the score through a chain of brains (ML win-probability adjustment, backtest veto, risk veto, bullish-picker bump, early-morning gate) before the "Chair" ensemble makes a single go/no-go decision and Portfolio Manager sizes the notional. The **bull-momentum path** only fires in a confirmed bull regime during 10:02–15:35 ET, monitors a candidate live for 180s, and buys when strict momentum + volume + MACD/RSI conditions all fire. Exits are a layered stack: broker-side trailing stop, chandelier ATR stop that ratchets with peak price, profit-monitor round-trip rescue, hard 2×ATR stop, and a bad-news stop-tightener that fires when Brain F's negative news sentiment on an owned position crosses a threshold.

---

## The brain suite

| Brain | Role | Mode |
|---|---|---|
| **A — ML Brain (TFBrain)** | Conv1D + 2×LSTM on frozen hand-weighted foundation. 32 features (10 stock + 20 NASDAQ + 2 relative strength). Adjusts buy score ±1.5. | armed |
| **B — Risk Brain** | 12-feature NN → P(safe). Vetoes new entries when portfolio state is dangerous. | shadow (default) |
| **C — Backtest Brain** | Replays 5y history under actual exit rules, vetoes if regime-weighted win rate < 50%. | armed |
| **D — Portfolio Manager** | 10 trajectory features → size_multiplier, aggression, concentration. Nudges Chair only. | shadow (default) |
| **E — Chair** | Deterministic ensemble aggregator. Any DENY → skip; else D's multiplier scales notional. | always on |
| **F — Bullish Picker** | 12-feature NN → P(bullish), +news sentiment. Bumps buy score up to +0.5. | armed |
| **RISK_TIME** | Blocks new entries before 10:05 ET unless candidate is ≥2% below yesterday's close. | armed |

All brains post to the **Brain Trading Floor** — a shared message bus that logs every vote, lesson, and observation. The dashboard exposes it as a live terminal.

---

## Data plumbing

- **Alpaca-py `StockHistoricalDataClient`** is primary for daily bars (IEX free feed) — ~78× faster than yfinance for the S&P 500 batch (7s vs 540s).
- **yfinance** is fallback for daily and stays primary for intraday (1m/5m/60m) and `^`-prefixed indices (VIX).
- Symbol convention: yfinance-form internally (`BRK-B`); `to_alpaca()` / `to_yf()` swap the dot when talking to Alpaca.
- **`alpaca-trade-api`** (legacy) still handles orders, positions, cash, and portfolio history.
- **News**: Alpaca News API v1beta1 (Benzinga, free with existing keys), rolling 3-day window, 15 min cache, ~80 positive / ~70 negative keywords with negation lookback.

---

## Safety layers

- **BlacklistManager** — two-tier (permanent + 72h temporary), auto-adds losing closes, persisted to `blacklist.json`.
- **TradeGovernor** — 5 consecutive losses → 30 min cooldown; 3% daily profit locks the day; auto-resets on ET date roll.
- **PerSymbolPerformance** — mutes a symbol for 24h after 5+ trades with <30% win rate and negative PnL.
- **BrainTrustTracker** — rolling accuracy per brain (windows 50/200/1000); observation-only for now.
- Early-morning gate blocks thin/volatile 9:30–10:05 ET entries unless there's a real ≥2% overnight dislocation.

---

## Dashboard

Open `alpaca_dashboard.html` directly in a browser — no server needed, it's a static file that connects to the bot's WebSocket. Tiles:

- **Header**: Account, Regime, Governor, Brains A–F, Risk (B), Portfolio Mgr (D)
- Live Positions · Profit Monitor (ARMED / TOUCHED / waiting) · Recent Trades
- Brain Trading Floor terminal (accepts commands: `status`, `pause`, `resume`, `sell_all`, `blacklist <SYM> [Nh|perm]`, `unblacklist`, `clear`, `help`)
- 72h Auto Blacklist · Permanent Blacklist · Muted
- Settings (14 whitelisted keys apply immediately, no restart)
- Robot Control (pause / resume / sell-all / emergency stop)

If the `websockets` package is missing, the dashboard is silently disabled and the bot keeps trading.

---

## Configuration

Runtime knobs live as module-level constants near the top of each section. Key ones:

```python
# Bull-momentum path
BULL_BUY_WINDOW = (time(10, 2), time(15, 35))
BULL_MONITOR_SECONDS = 180
BULL_MIN_NET_RETURN = 0.001
BULL_MAX_ALLOCATION_PER_SYMBOL = 600

# Chandelier ATR trailing stop
CHANDELIER_MODE = 'armed'
CHANDELIER_ATR_MULT = 3.0
CHANDELIER_MIN_HOLD_SECS = 300

# Profit-monitor round-trip rescue
PROFIT_TOUCHED_PCT = 0.002       # +0.2%
ROUND_TRIP_EXIT_GAIN = 0.0       # breakeven-or-below

# Bad-news stop tightener
BAD_NEWS_STOP_MODE = 'armed'
BAD_NEWS_STOP_THRESHOLD = 0.20
BAD_NEWS_TIGHTEN_STOP_PCT = 0.003

# Early-morning gate (Version 17: hard block, no per-symbol escape)
EARLY_MORNING_CUTOFF_ET = time(10, 5)
EARLY_MORNING_MIN_DROP_PCT = 0.02   # legacy escape hatch (bypassed by the hard cycle-level block)

# Chandelier daily SELL (Version 17)
CHANDELIER_DAILY_SELL_TIGHTEN_PCT = 0.0025   # 0.25% below current — extremely tight

# Kelly position sizing (Version 17)
USE_KELLY_SIZING   = True
KELLY_MIN_TRADES   = 10
KELLY_MAX_FRACTION = 0.05           # cap at 5% of equity per position

# Governor
CONSEC_LOSS_COOLDOWN = 5
COOLDOWN_SECONDS = 1800
DAILY_PROFIT_LOCK_PCT = 0.03
```

Most brain modes are `'armed'` / `'shadow'` / `'off'` and can be toggled live from the dashboard Settings tile.

---

## Dependencies

Auto-installed on first launch, but for reference:

```
alpaca-trade-api  alpaca-py  pytz  numpy  TA-Lib  yfinance
SQLAlchemy  ratelimit  pandas-market-calendars  pandas
schedule  websockets  requests  tensorflow (or tensorflow-cpu)
```

`TA-Lib` requires the system `libta-lib` C library be installed first.

---

## Notes

- Trading path and market-data path use two separate Alpaca clients on purpose.
- The ML brain retrains 15k examples on live trades daily at 17:00 ET (20k lifetime pretrain cap).
- All exit decisions remain rule-based; brains only influence *entries* and *position sizing*.

---

## Third-Party Libraries and Attributions

The following third-party program libraries are used by this software. Each library is the property of its respective copyright holders and authors, and is used here under the terms of its own upstream license. Their inclusion does not imply endorsement of this software by the library authors. This attribution list is provided to satisfy the Lennox GNU program license requirement that program library names be referenced together with their respective copyright holders and authors.

Only the third-party libraries actually imported by `billionaire_strategy_buy_lowest_price_stock_market_robot.py` are listed. Python standard-library modules (`os`, `sys`, `time`, `json`, `csv`, `logging`, `threading`, `datetime`, `math`, `pickle`, `sqlite3`, `random`, `concurrent.futures`, `collections`, `importlib.metadata`, `subprocess`, `shutil`, etc.) are part of the Python distribution and are not repeated here.

1. **alpaca-trade-api-python** (imported as `alpaca_trade_api`). Copyright (c) Alpaca Securities LLC and the alpaca-trade-api-python contributors. License: Apache License 2.0. Purpose: Legacy Python client for the Alpaca brokerage REST and streaming APIs; used for account, position, order placement/cancellation, and real-time trade updates.

2. **alpaca-py** (imported as `alpaca`, specifically `alpaca.data.historical.StockHistoricalDataClient`, `alpaca.data.requests`, `alpaca.data.enums.DataFeed`, and `alpaca.data.timeframe.TimeFrame`). Copyright (c) Alpaca Securities LLC and the alpaca-py contributors. License: Apache License 2.0. Purpose: Newer official Alpaca SDK; used specifically for the historical stock market-data client that fetches OHLCV bars for the scanner and scoring engine.

3. **NumPy** (imported as `numpy`). Copyright (c) 2005-present, NumPy Developers. Original author: Travis E. Oliphant. Maintained by the NumPy development team and contributors. License: BSD 3-Clause License. Purpose: N-dimensional array math, vectorized numerical operations, and array infrastructure used by the scanner, indicator math, and ML feature builders.

4. **pandas** (imported as `pandas`). Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team. Copyright (c) 2011-present, Open source contributors. Original author: Wes McKinney. Maintained by the pandas development team. License: BSD 3-Clause License. Purpose: DataFrame and time-series manipulation used throughout the scanner, the ML feature pipeline, and price-history processing.

5. **pandas-market-calendars** (imported as `pandas_market_calendars`). Copyright (c) Ryan Sheftel and the pandas-market-calendars contributors. License: MIT License. Purpose: Exchange trading calendars used to determine NYSE/NASDAQ session open/close times, holidays, and early-close days.

6. **pytz** (imported as `pytz`). Copyright (c) 2003-present Stuart Bishop and contributors. License: MIT License. Purpose: IANA time-zone database bindings used for US/Eastern market-time conversions and scheduling.

7. **ratelimit** (imported as `ratelimit`, specifically `limits` and `sleep_and_retry`). Copyright (c) Tomas Basham and contributors. License: MIT License. Purpose: Decorators used to throttle outbound broker and market-data API calls.

8. **schedule** (imported as `schedule`). Copyright (c) 2013 Daniel Bader (https://dbader.org) and contributors. License: MIT License. Purpose: Human-friendly job-scheduling library declared as a runtime dependency for the recurring background tasks (daily S&P 500 scanner refresh, the overnight ML training run, the pre-market and pre-close profit sweeps, and the weekly Sunday-night backtest); some of these recurring jobs are dispatched by inline wall-clock checks against `pandas_market_calendars` and `pytz` alongside this library.

9. **SQLAlchemy** (imported as `sqlalchemy`; symbols `create_engine`, `Column`, `Integer`, `String`, `Float`, `event`, `sessionmaker`, `scoped_session`, `SQLAlchemyError`, `NoResultFound`). Copyright (c) 2005-present Michael Bayer and contributors. License: MIT License. Purpose: Object-relational mapping and connection handling for the local SQLite trading database (positions, trade history, feature snapshots, adaptive parameters).

10. **TA-Lib Python wrapper** (imported as `talib`). Python wrapper copyright (c) John Benediktsson (mrjbq7) and contributors. License (Python wrapper): BSD 2-Clause License. The wrapper links against the underlying TA-Lib C library, originally authored by Mario Fortier and now maintained by the TA-Lib community, distributed under its own BSD-style license. Purpose: Technical-analysis indicators (RSI, MACD, ATR, Bollinger Bands, Stochastic, ADX, OBV, candlestick pattern recognition, moving averages, etc.) used by both the scanner and the live scoring engine.

11. **yfinance** (imported as `yfinance`). Copyright (c) Ran Aroussi and the yfinance contributors. License: Apache License 2.0. Purpose: Retrieval of historical and recent OHLCV price/volume data used by the scanner, ML feature builders, market-regime analysis, and the weekly backtester. This library is an independent, community-maintained project and is not affiliated with, endorsed by, or sponsored by Yahoo, Inc.

12. **TensorFlow and Keras** (imported lazily as `tensorflow` / `tensorflow.keras`). Copyright (c) The TensorFlow Authors and Google LLC. Keras originally authored by François Chollet; now developed as part of the TensorFlow project by the Keras and TensorFlow authors and contributors. License: Apache License 2.0. Purpose: Construction, training, saving, loading, and inference of the shared Conv1D + LSTM sequence brain (with a frozen hand-weighted technical-analysis foundation) that produces the ML buy-score adjustment, and of the smaller Brain B risk-gating network. Loaded lazily; the robot runs without ML if TensorFlow is not installed. On NVIDIA CUDA GPU-equipped systems the `tensorflow` wheel is used; on CPU-only systems the `tensorflow-cpu` wheel is used, and the robot will auto-swap the installed variant at startup to match the detected hardware.

13. **Backtrader** (imported lazily as `backtrader`). Copyright (c) 2015-present Daniel Rodriguez (`mementum`) and the Backtrader contributors. License: GNU General Public License, version 3 (GPLv3). Purpose: Event-driven backtesting engine used by the integrated weekly Sunday-night historical backtest (scheduled for 22:00 ET on Sundays), which replays approximately two years of daily OHLCV bars through an approximation of the live strategy — reusing the same module-level trading constants (entry rules, stops, scale-out targets, position caps, commission and slippage assumptions) — against either the current in-memory scanner candidate list or a hardcoded fallback universe. The backtest never runs during trading hours and never blocks the live loop. Loaded lazily; the live robot runs without it if the package is not installed.

14. **importlib_metadata** (optional fallback import as `importlib_metadata`). Copyright (c) Jason R. Coombs and the Python Packaging Authority (PyPA) contributors. License: Apache License 2.0. Purpose: Fallback package-metadata inspection used only on Python installations where `importlib.metadata` from the standard library is unavailable or incomplete, during the TensorFlow / TensorFlow-CPU wheel detection step.

### Additional runtime dependencies

- **SQLite**: The trading database is stored using SQLite, which is in the public domain (authored by D. Richard Hipp and the SQLite consortium). SQLite is accessed indirectly through SQLAlchemy and the Python standard library's `sqlite3` module.
- **CPython**: The Python interpreter itself is copyright (c) the Python Software Foundation and is distributed under the PSF License Agreement.

All trademarks, product names, and brand names referenced in this document (including but not limited to Alpaca, NASDAQ, S&P 500, QQQ, SPY, VIX, TensorFlow, Keras, Yahoo, and SQLite) are the property of their respective owners and are used here for identification and interoperability purposes only.

---

## Risk Disclaimer

This software is not affiliated with or endorsed by Alpaca Securities, LLC.

Automated trading involves substantial financial risk. Past performance, backtest results, machine-learning predictions, technical indicators, or historical expectancy do not guarantee future results.

Backtests are approximations and cannot reproduce every aspect of live execution, including market impact, liquidity changes, slippage, latency, order rejection, gaps, and intraday price sequencing.

Use paper trading and thorough testing before risking real capital. Never risk money you cannot afford to lose.

This software is provided on an "as-is" and "use-at-your-own-risk" basis without guarantees of profitability or uninterrupted operation. The developer is not responsible for financial losses or damages resulting from use of the software, to the fullest extent permitted by law.

Trade with discipline. Protect capital first.
