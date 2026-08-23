# Billionaire Strategy Stock Market Trading Robot

Single-file automated Alpaca daytrading bot for NASDAQ + S&P 500 equities, driven by an ensemble of neural-net "brains" and a rules-based dip-buy / bull-momentum strategy. Ships with a live browser dashboard.

**Files**
- `billionaire-strategy-buy-lowest-price-stock-market-robot.py` — the bot (~12.5k lines, one file, banner "Version 10")
- `alpaca_dashboard.html` — the dashboard UI it serves at `ws://127.0.0.1:8765`

---

## Quick start

```bash
export APCA_API_KEY_ID=your_key
export APCA_API_SECRET_KEY=your_secret
export APCA_API_BASE_URL=https://paper-api.alpaca.markets   # or live
python billionaire-strategy-buy-lowest-price-stock-market-robot.py
```

Then open `alpaca_dashboard.html` in a browser. It connects to the bot over WebSocket on `127.0.0.1:8765` at 1 Hz.

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

# Early-morning gate
EARLY_MORNING_CUTOFF_ET = time(10, 5)
EARLY_MORNING_MIN_DROP_PCT = 0.02

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
