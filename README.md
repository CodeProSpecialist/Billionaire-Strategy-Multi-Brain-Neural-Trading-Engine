Absolutely. Below is the **complete README**, with the warnings consolidated into **one short paragraph at the very bottom**, and the project licensed under **GNU General Public License v3.0 (GPL-3.0)**. GPLv3 is a copyleft free-software license published by the Free Software Foundation. ([GNU][1])

````markdown
# Billionaire Strategy Stock Market Trading Robot

An algorithmic stock-trading system for **Alpaca** designed around multi-factor stock selection, technical analysis, market-regime detection, machine-learning-assisted scoring, adaptive parameters, automated risk controls, and continuous position management.

---

## Overview

The **Billionaire Strategy Stock Market Trading Robot — Version 10** is a Python-based automated trading system that continuously scans a large stock universe, ranks potential opportunities, applies multiple layers of confirmation and risk management, and manages open positions.

The system combines:

- Alpaca trading and market data
- S&P 500 stock scanning
- RSI, MACD, ATR, ADX, Bollinger Bands, stochastic and volume analysis
- Market-regime detection
- VIX-aware trading decisions
- Risk-based position sizing
- Hard ATR-based stop losses
- Profit monitoring and scale-outs
- Per-symbol performance monitoring
- Historical backtesting
- Adaptive parameter optimization
- A shared TensorFlow sequence-model "brain"
- Historical analog analysis
- Bull-market strategy
- Trade-governor safeguards
- Real-time WebSocket dashboard
- Emergency-stop and sell-all controls
- Persistent trade and model state

The bot's scanner operates in-process and ranks an S&P 500 candidate universe rather than relying on a separate stock-list generation process.

---

# Architecture

```text
                    ┌─────────────────────────┐
                    │      Alpaca Account     │
                    │ Positions / Cash / BP   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Market Data Layer     │
                    │ Alpaca IEX + yfinance   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     S&P 500 Scanner      │
                    │ Technical + Fundamental  │
                    │      Candidate Ranking   │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌────────────────────────────────────┐
              │         Trading Decision Layer      │
              │                                    │
              │ • Buy Score                         │
              │ • Market Regime                     │
              │ • ML Brain                          │
              │ • Historical Analogs                │
              │ • Backtest Brain                    │
              │ • Per-Symbol Performance            │
              │ • Trade Governor                    │
              └────────────────┬───────────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │    Risk / Position      │
                    │       Sizing Layer      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Alpaca Orders       │
                    │     Buy / Sell / Stops   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Position Manager     │
                    │ Stops / TP / Scale-Out   │
                    │      / Exit Logic        │
                    └─────────────────────────┘

                         ┌────────────────┐
                         │ Web Dashboard   │
                         │ WebSocket :8787 │
                         └────────────────┘
````

---

# Core Trading System

## Alpaca Integration

The bot uses Alpaca for:

* Account information
* Buying power
* Positions
* Order submission
* Order status
* Current prices
* Historical stock bars
* Latest trade data

The implementation uses both the legacy `alpaca_trade_api` trading client and `alpaca-py` for read-only market data. Alpaca's IEX feed is used as the primary source for supported daily stock data, with `yfinance` used as a fallback or for data types not available through that path.

### Environment Variables

Set the Alpaca credentials before starting the bot:

```bash
export APCA_API_KEY_ID="YOUR_ALPACA_API_KEY"
export APCA_API_SECRET_KEY="YOUR_ALPACA_SECRET_KEY"
export APCA_API_BASE_URL="YOUR_ALPACA_BASE_URL"
```

For Alpaca paper trading, use the appropriate paper-trading API base URL.

---

# Stock Scanner

The integrated scanner evaluates an S&P 500 candidate universe and creates a ranked list of potential purchases.

The scanner uses historical data and multiple technical measurements, including:

* RSI
* MACD
* ATR
* ADX
* Bollinger Bands
* Stochastic
* Volume
* Moving averages
* VWAP-related measurements
* Seasonal performance
* Relative strength versus SPY
* Historical price behavior

The scanner configuration includes a 14-period RSI, 12/26/9 MACD configuration, 20-period Bollinger Bands, stochastic calculations, and a 14-period ADX.

Relative strength versus SPY can provide an additional positive score adjustment when a stock is outperforming the benchmark.

---

# Market-Regime Detection

Before entering new positions, the robot evaluates the current market regime.

The regime system influences:

* Whether new positions are permitted
* Buy-score thresholds
* Strategy weighting
* Bull-market strategy activation
* Risk decisions

The dashboard exposes the current regime, VIX value, and whether a weak bullish environment has been downgraded.

---

# Bull-Market Strategy

When the market regime qualifies as bullish, the robot can activate an additional bull-market buying path alongside the normal ranked strategy.

The bull strategy requires multiple confirmations, including:

* Short-term live price monitoring
* Positive price movement
* Daily MACD confirmation
* RSI confirmation
* Volume confirmation
* A defined time-of-day trading window

Bull candidates are merged with the normal candidate list and remain subject to the maximum number of new positions per cycle.

Bull-market positions have their own exit behavior, including a flat profit target and a broker-side trailing stop.

---

# Machine Learning Brain

Version 10 contains a shared TensorFlow sequence-model brain.

There is intentionally **one shared brain** rather than a separate neural network for every symbol.

The model uses a rolling sequence of **20 daily feature snapshots** with **10 features per day**.

## Model Architecture

```text
20 × 10 Daily Feature Sequence
              │
              ▼
     Frozen Foundation
              │
       Dense(8, tanh)
              │
       Dense(6, tanh)
              │
              ▼
      Conv1D(64, causal)
              │
       Batch Normalization
              │
          Dropout
              │
        LSTM(128)
              │
          Dropout
              │
        LSTM(64)
              │
          Dropout
              │
        Dense(32, ReLU)
              │
        Dense(1, Sigmoid)
              │
              ▼
        Estimated Win Probability
```

The trainable portion uses a Conv1D → LSTM → LSTM → Dense architecture. The first two foundation layers are intentionally frozen so the learned model cannot completely overwrite the predefined technical-analysis foundation.

## ML Features

The model receives:

1. RSI
2. MACD
3. MACD signal
4. MACD-above-signal indicator
5. ATR percentage
6. Daily return
7. Volume relative to SMA
8. Distance from SMA20
9. Distance from SMA50
10. SMA20 > SMA50 indicator

These features are constructed directly from the daily OHLCV data.

## ML Guardrails

The ML component is deliberately prevented from becoming the sole trading decision.

The model:

* Has a win-probability threshold
* Requires sufficient live trade history before influencing live decisions
* Has a maximum score adjustment
* Can gracefully disable itself if TensorFlow is unavailable
* Uses focal loss to emphasize difficult examples
* Gives additional weight to losing examples

The current configuration requires 60 live trades before the ML adjustment can affect live decisions and caps its score adjustment at ±1.5 points.

---

# ML Training

Historical pretraining uses daily market data.

The current configuration includes:

* 20-day sequence length
* 10 features
* 0.0006 learning rate
* Focal-loss gamma of 1.2
* Batch size of 64
* 20,000-example historical pretraining lifetime cap
* 15,000-example scheduled training runs
* Daily maintenance training after the historical pretraining cap

The model is scheduled around the trading day, with the normal training window beginning at **5:00 PM ET** and designed to finish before **7:45 AM ET**.

TensorFlow is loaded lazily so that failure of the ML subsystem does not necessarily prevent the trading bot from operating. The program also detects NVIDIA CUDA availability and selects the appropriate TensorFlow package.

---

# Historical Analog Analysis

In addition to the neural-network model, the robot can search historical data for situations resembling the current market setup.

Analog results can adjust the buy score when there is sufficient evidence.

The analog system is intentionally bounded and does not act as an independent hard buy gate.

---

# Backtest Brain

The system contains an integrated historical backtesting subsystem.

Backtests use the same major trading parameters used by the live system, including:

* ATR hard stops
* Profit-arm settings
* Giveback settings
* Scale-out stages
* Risk per trade
* Allocation limits
* Buy-score threshold

The backtest subsystem is configured for a two-year lookback and saves timestamped JSON reports to the `backtest_results` directory.

Backtest reports include metrics such as:

* Final portfolio value
* Net P&L
* Total return
* Total trades
* Wins
* Losses
* Win rate
* Average winning trade
* Average losing trade
* Sharpe ratio
* Maximum drawdown
* SQN

## Backtest Limitations

The backtester uses daily bars rather than tick-by-tick market data.

Therefore, it does not perfectly reproduce:

* Intraday order sequencing
* Actual order escalation
* Live slippage
* Cancel-before-sell behavior
* Existing broker order interactions

Backtest results should therefore be considered an approximation of live behavior rather than an exact simulation.

---

# Risk Management

Risk controls are applied before new positions are opened.

Current configuration includes:

```python
ACCOUNT_MODE = 'margin'
MAX_PORTFOLIO_EXPOSURE_PCT = 0.98
MAX_LEVERAGE = 1.0
RISK_PER_TRADE_PCT = 0.01
MAX_ALLOCATION_PER_SYMBOL = 600.0
MAX_NEW_POSITIONS_PER_CYCLE = 3
MIN_ORDER_NOTIONAL = 1.00
CASH_BUFFER = 1.00
```

Before buying, the bot checks:

* Broker account status
* Trading restrictions
* Margin health
* Portfolio exposure
* Effective buying power
* Cash buffer
* Trade-governor state
* Position limits
* Per-symbol performance
* Historical backtest veto
* Candidate score

If margin health falls below the configured maintenance floor, new purchases are suspended.

---

# Position Sizing

Position sizing is based on ATR-derived risk.

The system calculates the risk per share using the same ATR multiplier used by the actual hard stop.

```text
Risk Amount
     ÷
Risk Per Share
     =
Position Size
```

The current configuration uses:

```python
RISK_PER_TRADE_PCT = 0.01
HARD_STOP_ATR_MULTIPLIER = 2.0
```

The sizing calculation is aligned with the actual hard-stop distance rather than using a different theoretical stop distance.

---

# Stop Loss and Exit System

The robot uses several layers of exit protection.

## Hard Stop

The hard stop is ATR-based:

```python
USE_HARD_STOP_LOSS = True
HARD_STOP_ATR_MULTIPLIER = 2.0
HARD_STOP_MIN_PCT = 0.03
```

## Profit Monitoring

The exit system can incorporate:

* Profit targets
* Peak tracking
* Giveback protection
* Scale-out stages
* Hard stops
* Trailing stops
* Strategy-specific exits

## Bull Strategy Exit

Bull-market positions have a separate +0.5% flat target and their own 1% broker-side trailing stop.

---

# Trade Governor

The trade governor provides another layer of protection against uncontrolled trading.

It monitors conditions such as:

* Consecutive losses
* Trading cooldowns
* Daily profit locks
* Whether trading is currently permitted

The buy cycle checks the governor before creating new positions.

---

# Per-Symbol Performance Mute

The robot can temporarily mute symbols that have demonstrated poor performance in the bot's own trading history.

This prevents the system from repeatedly trading a symbol that has become consistently unfavorable.

The performance system evaluates:

* Win rate
* Net P&L
* Number of trades

and can skip muted symbols during candidate evaluation.

---

# Score-Based Historical Expectancy

The system can also analyze previous closed trades according to their buy score.

When enough historical trades exist for a particular score level, the robot calculates the average historical outcome associated with that score.

The current minimum sample size is 15 trades per score bucket.

This allows the ranking system to become increasingly informed by the robot's own historical results.

---

# Adaptive Parameters

Version 10 includes an adaptive parameter subsystem.

The adaptive system can automatically adjust selected live parameters, but only within defined guardrails.

Controls include:

* Minimum sample requirements
* Maximum adjustment step sizes
* Hard parameter bounds
* Audit logging

The main loop periodically runs the adaptive parameter pass alongside trade-history analysis and expectancy calculations.

---

# Trading Loop

The primary trading cycle operates approximately every **60 seconds**.

During the main loop the system can:

1. Analyze existing positions
2. Evaluate new candidates
3. Check account health
4. Evaluate the market regime
5. Rank candidates
6. Apply ML adjustments
7. Apply historical/analog analysis
8. Apply risk controls
9. Submit orders
10. Manage exits
11. Analyze trading history
12. Run adaptive parameters
13. Run scheduled ML training
14. Run scheduled backtesting
15. Wait for the next cycle

---

# Real-Time Dashboard

The repository includes a separate HTML dashboard:

```text
alpaca_dashboard.html
```

The dashboard provides a real-time view of the trading system.

It displays:

* Account equity
* Buying power
* Cash
* Exposure
* Effective buying power
* Margin health
* Market regime
* VIX
* Trade governor status
* Consecutive losses
* Cooldown
* ML brain status
* Brain trust
* Open positions
* Current prices
* Gain/loss percentage
* ATR
* Chandelier stop
* Recent trades
* Muted symbols
* Brain thinking log
* System flags

---

# Dashboard Controls

The dashboard provides four operator controls.

## Pause

Stops new entries while allowing normal position-management logic to continue.

## Resume

Re-enables normal entry decisions.

## Sell All

Requests that all open positions be closed at market.

## Emergency Stop

Combines:

* Pause new entries
* Sell all positions

The dashboard commands are implemented through WebSocket messages.

---

# Dashboard WebSocket

The trading robot runs a WebSocket server on:

```text
ws://127.0.0.1:8787
```

The default configuration binds to localhost:

```python
DASHBOARD_WS_HOST = '127.0.0.1'
DASHBOARD_WS_PORT = 8787
DASHBOARD_BROADCAST_INTERVAL_SECS = 1.0
```

The dashboard receives a state snapshot approximately once per second.

The HTML dashboard automatically reconnects if the WebSocket connection is lost.

---

# Project Structure

A typical deployment can look like:

```text
billionaire-trading-bot/
│
├── billionaire-strategy-buy-lowest-price-stock-market-robot.py
├── alpaca_dashboard.html
│
├── ml_brain_model/
│   ├── model.keras
│   ├── meta.json
│   └── schedule_state.json
│
├── backtest_results/
│   └── backtest_YYYYMMDD_HHMMSS.json
│
└── README.md
```

The ML model and metadata are stored under `ml_brain_model`.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/billionaire-trading-bot.git
cd billionaire-trading-bot
```

## 2. Install Python

Python 3.x is required.

A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

The bot contains an automatic dependency bootstrap system that checks installed packages and attempts to install missing dependencies at startup.

The dependency list includes packages such as:

```text
alpaca-trade-api
alpaca-py
numpy
TA-Lib
yfinance
SQLAlchemy
ratelimit
pandas-market-calendars
pandas
schedule
websockets
```

You may also install dependencies manually if preferred.

---

# Configure Alpaca

Set:

```bash
export APCA_API_KEY_ID="YOUR_KEY"
export APCA_API_SECRET_KEY="YOUR_SECRET"
export APCA_API_BASE_URL="YOUR_BASE_URL"
```

Verify that the credentials have the appropriate permissions before starting the program.

---

# Start the Robot

```bash
python billionaire-strategy-buy-lowest-price-stock-market-robot.py
```

The program will initialize its dependencies and trading components.

---

# Start the Dashboard

The dashboard HTML file can be served through any simple static HTTP server.

For example:

```bash
python -m http.server 8080
```

Then open:

```text
http://127.0.0.1:8080/alpaca_dashboard.html
```

The dashboard connects to the trading bot's WebSocket endpoint:

```text
ws://127.0.0.1:8787
```

The dashboard itself does not provide the trading engine; it is an operator interface connected to the bot's WebSocket server.

---

# Monitoring

The dashboard provides real-time visibility into the system's internal state.

Example monitoring areas:

```text
ACCOUNT
├── Equity
├── Buying Power
├── Cash
├── Exposure
└── Margin Health

MARKET
├── Regime
├── VIX
└── Regime Downgrade

GOVERNOR
├── Can Trade
├── Consecutive Losses
├── Cooldown
└── Day Lock

BRAINS
├── Backtest Mode
├── ML Threshold
├── Cached Symbols
├── Chandelier Mode
└── Brain Trust

POSITIONS
├── Average Price
├── Current Price
├── Gain %
├── Entry ATR
└── Chandelier Stop

OPERATIONS
├── Recent Trades
├── Muted Symbols
├── Thinking Log
└── Emergency Flags
```

---

# Current Risk Configuration

| Parameter                     | Current Configuration |
| ----------------------------- | --------------------: |
| Account mode                  |                Margin |
| Maximum portfolio exposure    |                   98% |
| Maximum leverage              |                  1.0× |
| Risk per trade                |                    1% |
| Maximum allocation / symbol   |                  $600 |
| Maximum new positions / cycle |                     3 |
| Minimum order                 |                    $1 |
| Cash buffer                   |                    $1 |
| Hard stop                     |               Enabled |
| Hard stop ATR multiplier      |                  2.0× |
| Hard stop minimum             |                    3% |
| Bull-market target            |                 +0.5% |
| Bull trailing stop            |                    1% |

---

# Performance Evaluation

The robot should be evaluated using more than raw win rate.

Recommended metrics include:

* Net P&L
* Profit factor
* Average win
* Average loss
* Expectancy
* Maximum drawdown
* Sharpe ratio
* SQN
* Number of trades
* Win rate
* Exposure
* Return on capital
* Performance by score
* Performance by symbol
* Performance by market regime

The integrated backtesting system already records several of these metrics for historical evaluation.

---

# Development Status

**Version:** 10

**Primary language:** Python

**Broker:** Alpaca

**Market:** U.S. equities

**Primary candidate universe:** S&P 500

**Machine learning:** TensorFlow

**Technical analysis:** TA-Lib

**Historical data:** Alpaca IEX + yfinance fallback

**Dashboard:** HTML + WebSocket

**Backtesting:** Integrated scheduled backtesting subsystem

---

# Contributing

Contributions, bug reports, strategy improvements, testing results, and performance-analysis tools are welcome.

When submitting changes to trading logic, include:

* The reason for the change
* The affected strategy component
* Backtest results where applicable
* Risk implications
* Any changes to position sizing
* Any changes to exit behavior
* Any new dependencies

For trading-strategy changes, avoid making claims of profitability without reproducible evidence.

---

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
