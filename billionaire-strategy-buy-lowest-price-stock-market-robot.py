"""Billionaire Strategy Stock Market Trading Robot, Version 10.

Startup ordering:
  1. Bootstrap dependencies (this block) -- runs pip list, silently
     installs anything missing, then proceeds. Uses stdlib only so it
     can run before any 3rd-party import.
  2. Normal imports.
  3. Main program.
"""

# ---------------- Dependency bootstrap (must run before other imports) ------
import subprocess as _boot_subprocess
import sys as _boot_sys


def _bootstrap_dependencies():
    """Check for required pip packages via `pip list` and silently install
    any that are missing. Runs at import time so downstream imports below
    always succeed.

    Package tuples are (import_name, pip_name) because they don't always
    match (e.g. `import talib` from pip package `TA-Lib`; `import alpaca`
    from pip package `alpaca-py`).
    """
    required = [
        # (import name, pip name)
        ("alpaca_trade_api",        "alpaca-trade-api"),
        ("alpaca",                  "alpaca-py"),
        ("pytz",                    "pytz"),
        ("numpy",                   "numpy"),
        ("talib",                   "TA-Lib"),          # requires system libta-lib
        ("yfinance",                "yfinance"),
        ("sqlalchemy",              "SQLAlchemy"),
        ("ratelimit",               "ratelimit"),
        ("pandas_market_calendars", "pandas-market-calendars"),
        ("pandas",                  "pandas"),
        ("schedule",                "schedule"),
        ("websockets",              "websockets"),        # dashboard WS server
        ("requests",                "requests"),          # brain_f Alpaca News fetch
    ]
    # Fast path: one `pip list` call, parse output, install only missing.
    try:
        r = _boot_subprocess.run(
            [_boot_sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=60,
        )
        installed = set()
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "==" in line:
                    installed.add(line.split("==", 1)[0].strip().lower())
    except Exception:
        installed = set()

    missing = [pip_name for (imp, pip_name) in required
               if pip_name.lower() not in installed]
    if not missing:
        return

    # Install missing packages via a plain pip install.
    for pkg in missing:
        installed_ok = False
        try:
            r = _boot_subprocess.run(
                [_boot_sys.executable, "-m", "pip", "install", "--quiet", pkg],
                capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                installed_ok = True
        except Exception:
            pass
        if not installed_ok:
            print(f"[bootstrap] Warning: could not install {pkg} automatically. "
                  f"Please run: pip install {pkg}")


_bootstrap_dependencies()
# ---------------- End dependency bootstrap ----------------------------------


import threading
import logging
import csv
import os
import time
import schedule
from datetime import datetime, timedelta, date, timezone
from datetime import time as time2
import alpaca_trade_api as tradeapi
import pytz
import math
import numpy as np
from collections import deque
import talib
import yfinance as yf
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import SQLAlchemyError
from ratelimit import limits, sleep_and_retry
import pandas_market_calendars as mcal
import pandas as pd  # needed by the inlined stock scanner (and harmless elsewhere)

# =================================================================================
# ---------------- ML BRAIN (sequence-model TFBrain style, inlined) ----------------------
# =================================================================================
# A sequence-model TFBrain architecture, with the
# per-symbol / per-coin / per-ETF-fund brains INTENTIONALLY removed per instruction:
# there is exactly ONE brain here, shared across every symbol the bot trades. All
# training data comes from yfinance -- no crypto exchange feeds, no per-asset
# specialization.
#
# ARCHITECTURE (sequence-model TFBrain shape):
#     Input: (batch, ML_BRAIN_SEQ_LEN=20, ML_BRAIN_FEATURES) -- a rolling window
#            of the last 20 daily feature snapshots ending at "now".
#          |
#     Conv1D(64, kernel=3, causal) -> BatchNorm -> Dropout(0.25)
#          |  captures short-term local patterns (3-day motifs)
#     LSTM(128, return_sequences=True) -> Dropout(0.25)
#          |  learns temporal dependencies across the full 20-day window
#     LSTM(64) -> Dropout(0.20)
#          |  compresses the sequence into a fixed-length context vector
#     Dense(32, relu) -> Dense(1, sigmoid) = "will this trade be profitable?"
#
# TRAINING SIGNAL:
#   - Focal loss: concentrates gradient on trades the model is currently
#     getting WRONG, instead of spending capacity on already-confident-correct
#     examples ("extreme desire to win" expressed as a loss function).
#   - Per-sample loss penalty: losers weighted heavier than winners.
#   - Win-probability threshold at inference time: only trades when confident.
#
# TRAINING SCHEDULE (per instruction):
#   - First run (on-startup, if no model exists yet OR forced): 2,500 examples,
#     lightweight -- enough to seed a working model without overfitting on
#     limited data.
#   - Every day at 17:00 ET: full training run at 15,000 examples. Must
#     complete before 07:45 ET the next morning so it doesn't collide with the
#     08:00-ish trading day starting up (leaves ~14.5 hours of overnight
#     runway, which is plenty for a 15k-example run on this small model).
#
# NO Runpod, NO websocket feed, NO web server, NO per-symbol brains, NO
# per-ETF brains, NO crypto price sources. Everything runs in-process using
# yfinance for all historical data.
# =================================================================================

import json as _ml_json
from collections import deque as _ml_deque

_ml_base_dir = os.path.dirname(os.path.abspath(__file__))
ML_BRAIN_DIR = os.path.join(_ml_base_dir, 'ml_brain_model')
ML_MODEL_PATH = os.path.join(ML_BRAIN_DIR, 'model.keras')
ML_META_PATH = os.path.join(ML_BRAIN_DIR, 'meta.json')
ML_STATE_PATH = os.path.join(ML_BRAIN_DIR, 'schedule_state.json')

# Kept for external status callers -- points at the same meta file the new
# code writes to, so get_ml_status() etc. keep working.
ML_WEIGHTS_PATH = ML_MODEL_PATH
ML_MODEL_DIR = ML_BRAIN_DIR

# ---- Model hyperparameters (sequence-model TFBrain constants, tuned for the
# smaller daily-bar feature vector we build below) ----
ML_BRAIN_SEQ_LEN = 20              # rolling window of 20 daily bars
ML_BRAIN_FEATURES = 10             # per-day feature count: see _ml_build_feature_row()
ML_BRAIN_LEARNING_RATE = 0.0006
ML_BRAIN_FOCAL_GAMMA = 1.2         # focal loss: focus on hard examples
ML_BRAIN_LOSS_PENALTY = 1.2        # losers weighted 1.2x winners
ML_BRAIN_BATCH_SIZE = 64
ML_BRAIN_WIN_THRESHOLD = 0.55      # inference: only nudge score up when P(win) >= this

# ---- Training-schedule config (per instruction) ----
ML_FIRST_RUN_EXAMPLES = 20000      # fill the 20k lifetime cap in one first-run pass
ML_DAILY_RUN_EXAMPLES = 15000      # midpoint of 10,000-20,000
# Hard lifetime cap on HISTORICAL PRETRAINING examples (per instruction).
# Once cumulative n_updates reaches this, historical pretraining stops
# entirely and only daily MAINTENANCE training on live win/loss outcomes
# runs from that point forward. The cap is on cumulative pretraining
# examples across all prior runs, tracked in meta.json's n_updates field.
ML_PRETRAIN_LIFETIME_CAP = 20000
# Once pretraining is capped, the daily maintenance pass fine-tunes the
# model on the last MAINTENANCE_LOOKBACK_DAYS of closed live trades.
ML_MAINTENANCE_LOOKBACK_DAYS = 1
ML_MAINTENANCE_MIN_TRADES = 5   # skip maintenance run if fewer live trades closed than this
ML_DAILY_TRAIN_HOUR = 17           # 5:00 PM ET start
ML_DAILY_TRAIN_MUST_FINISH_HOUR = 7    # done before 7:45 AM ET next day
ML_DAILY_TRAIN_MUST_FINISH_MINUTE = 45
ML_HIST_LOOKBACK_YEARS = 2
ML_HIST_FORWARD_LABEL_DAYS = 5      # label = 1 if close[N+5] > close[N]
ML_HIST_MIN_ROWS_PER_SYMBOL = ML_BRAIN_SEQ_LEN + ML_HIST_FORWARD_LABEL_DAYS + 40

# ---- Live-inference guardrails, unchanged in spirit from the prior version ----
ML_MIN_TRADES_FOR_LIVE_ADJUSTMENT = 60  # gate on LIVE trades before we adjust LIVE decisions
ML_MAX_SCORE_ADJUSTMENT = 1.5           # cap on how many buy-score points the model can add/subtract

_ml_lock = threading.RLock()  # RLock so nested calls (e.g. train -> load_cached_model) don't self-deadlock
_ml_model_cache = {'model': None, 'trained_at': None, 'n_updates': 0}
_ml_tf_availability_cache = {'checked': False, 'available': False}
_ml_tf_variant_ensured = False  # one-shot guard for the CPU/GPU package swap


def _ml_detect_nvidia_cuda_gpu():
    """
    Return True iff a working NVIDIA CUDA GPU is visible to this process.

    Strategy: shell out to `nvidia-smi` and require exit code 0 with at least
    one GPU name in the output. This deliberately does NOT import tensorflow
    (that's the whole point -- we're deciding which tensorflow package to
    install BEFORE importing it). Any failure -- binary missing, driver
    broken, no GPU, permission error, timeout -- is treated as "no GPU".
    """
    import shutil, subprocess
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    if r.returncode != 0:
        return False
    return any(line.strip() for line in r.stdout.splitlines())


def _ml_ensure_tf_variant():
    """
    Make sure the installed tensorflow package matches the hardware:
      * NVIDIA CUDA GPU present -> leave tensorflow (GPU build) alone.
      * No NVIDIA CUDA GPU      -> uninstall tensorflow, install tensorflow-cpu.

    Runs at most once per process. Any failure here is logged and swallowed
    so a broken pip environment can't take down the trading bot -- the
    subsequent `import tensorflow` will either succeed with whatever is
    installed or fail cleanly in _ml_lazy_import_tf().
    """
    global _ml_tf_variant_ensured
    if _ml_tf_variant_ensured:
        return
    _ml_tf_variant_ensured = True

    try:
        has_gpu = _ml_detect_nvidia_cuda_gpu()

        # Use `pip list --format=freeze` as the authoritative source of installed
        # packages: importlib.metadata sometimes reports the wheel name with an
        # underscore (tensorflow_cpu) instead of the dash form pip publishes,
        # which made the old lookup miss an already-installed tensorflow-cpu and
        # trigger a needless `pip install` every launch (visible in the console
        # as a wall of "Requirement already satisfied" lines). Freeze output is
        # PEP 503-normalized, so we lowercase and treat '_' == '-' when matching.
        import subprocess, sys
        installed = set()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze",
                 "--disable-pip-version-check"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            for line in (proc.stdout or "").splitlines():
                name = line.split("==", 1)[0].strip().lower().replace("_", "-")
                if name:
                    installed.add(name)
        except Exception as e:
            logging.warning(f"ml_brain: `pip list` failed ({e}); falling back to importlib.metadata.")
            try:
                from importlib.metadata import distributions
            except Exception:
                from importlib_metadata import distributions  # type: ignore
            for d in distributions():
                try:
                    n = (d.metadata['Name'] or "").strip().lower().replace("_", "-")
                    if n:
                        installed.add(n)
                except Exception:
                    continue

        has_cpu_pkg = 'tensorflow-cpu' in installed
        has_gpu_pkg = 'tensorflow' in installed

        if has_gpu:
            if not has_gpu_pkg:
                logging.warning(
                    "ml_brain: NVIDIA CUDA GPU detected but tensorflow not installed; installing tensorflow."
                )
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "tensorflow"],
                    check=False,
                )
                import sys as _sys
                for mod_name in [m for m in list(_sys.modules) if m == 'tensorflow' or m.startswith('tensorflow.')]:
                    _sys.modules.pop(mod_name, None)
            else:
                logging.info("ml_brain: NVIDIA CUDA GPU detected; using GPU tensorflow build.")
            return

        # No GPU. Only swap if we currently have the GPU-flavored `tensorflow`
        # package and not already `tensorflow-cpu`.
        if has_cpu_pkg and not has_gpu_pkg:
            logging.info("ml_brain: no NVIDIA GPU; tensorflow-cpu already installed (skipping pip).")
            return

        logging.warning(
            "ml_brain: no NVIDIA CUDA GPU detected; swapping tensorflow -> tensorflow-cpu."
        )
        swapped = False
        if has_gpu_pkg:
            rc = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"],
                check=False,
            ).returncode
            swapped = swapped or (rc == 0)
        if not has_cpu_pkg:
            rc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "tensorflow-cpu"],
                check=False,
            ).returncode
            swapped = swapped or (rc == 0)

        # Drop any stale tensorflow modules from sys.modules so the next
        # `import tensorflow` picks up the freshly installed package.
        import sys as _sys
        for mod_name in [m for m in list(_sys.modules) if m == 'tensorflow' or m.startswith('tensorflow.')]:
            _sys.modules.pop(mod_name, None)

        # If we actually swapped the tensorflow binary in this process, we CAN'T
        # trust an in-process `import tensorflow` next -- pip changed the on-disk
        # package but any cached shared-object handles, .pyc's, or partial C
        # extension state from the old build can crash the interpreter or load
        # a mismatched libtensorflow.so. Re-exec the same Python with the same
        # argv so the next run starts clean with the CPU wheel. We set a guard
        # env var so we can't loop if the swap somehow "succeeds" every launch.
        import os
        if swapped and os.environ.get("BOT_TF_SWAP_RESTARTED") != "1":
            os.environ["BOT_TF_SWAP_RESTARTED"] = "1"
            logging.warning(
                "ml_brain: tensorflow-cpu installed; restarting Python so the "
                "new binary is loaded (first-run ML training will begin after restart)."
            )
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as _e:
                logging.warning(f"ml_brain: os.execv restart failed ({_e}); continuing in-process.")
    except Exception as e:
        logging.warning(f"ml_brain: tensorflow variant check failed ({e}); continuing with whatever is installed.")


def _ml_lazy_import_tf():
    """
    TensorFlow is a heavy, optional dependency. Import it lazily and let any
    failure (not installed, wrong platform, etc.) disable the ML signal
    gracefully rather than crashing the live trading bot at startup.
    """
    try:
        _ml_ensure_tf_variant()
        import tensorflow as tf
        return tf
    except Exception as e:
        logging.warning(f"ml_brain: TensorFlow unavailable ({e}); ML scoring disabled.")
        return None


ML_BRAIN_AVAILABLE = False   # set on first _ml_brain_is_available() call


def _ml_brain_is_available():
    """
    Lazily resolves and caches TensorFlow availability on first call.
    TensorFlow's import alone can take 30-60 seconds -- doing it eagerly at
    bot.py startup would badly delay position reconciliation and the first
    trading cycle.
    """
    global ML_BRAIN_AVAILABLE
    if not _ml_tf_availability_cache['checked']:
        _ml_tf_availability_cache['available'] = _ml_lazy_import_tf() is not None
        _ml_tf_availability_cache['checked'] = True
        ML_BRAIN_AVAILABLE = _ml_tf_availability_cache['available']
        if not _ml_tf_availability_cache['available']:
            print("ML brain: TensorFlow unavailable; buy scoring will run without the ML adjustment.")
    return _ml_tf_availability_cache['available']


def _ml_make_focal_loss(gamma):
    """Focal loss, adapted:
    gamma=0 reduces to ordinary binary cross-entropy. Higher gamma => sharper
    focus on hard (currently-misclassified) examples. Works alongside the
    per-sample weight multiplication for the loss-penalty term.
    """
    def focal(y_true, y_pred):
        import tensorflow as _tf
        eps = _tf.keras.backend.epsilon()
        y_pred = _tf.clip_by_value(y_pred, eps, 1.0 - eps)
        y_true = _tf.cast(y_true, y_pred.dtype)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        ce = -_tf.math.log(p_t)
        focal_factor = _tf.pow(1.0 - p_t, gamma)
        return focal_factor * ce
    focal.__name__ = "focal_loss"
    return focal


def _ml_build_model(tf):
    """Frozen-foundation + trainable-head sequence model.

    Architecture:
        Input: (batch, 20, 10) -- rolling window of daily feature snapshots
              |
        TimeDistributed(Dense(8, tanh))  <-- FOUNDATION 1: per-timestep prior
              |     [FROZEN -- hand-specified weights, trainable=False]
        TimeDistributed(Dense(6, tanh))  <-- FOUNDATION 2: refined per-timestep prior
              |     [FROZEN -- hand-specified weights, trainable=False]
              |
        Conv1D(64, k=3, causal, relu)    <-- TRAINABLE HEAD FROM HERE DOWN
              |
        BatchNorm -> Dropout(0.25)
              |
        LSTM(128, return_sequences=True) -> Dropout(0.25)
              |
        LSTM(64) -> Dropout(0.20)
              |
        Dense(32, relu) -> Dense(1, sigmoid) = "will this trade be profitable?"

    Why frozen layers, and why here:
        The foundation is a hand-specified, read-only encoding of "what a
        good dip-buy setup looks like at a single day's feature snapshot"
        -- low RSI + MACD confirmation + healthy uptrend + reasonable
        volatility + volume holding. Those weights are set once, at
        construction time, and Keras is told `trainable=False` on the whole
        foundation layer so training never modifies them no matter what
        the loss gradient wants. The trainable Conv1D/LSTM head can only
        learn temporal patterns and interactions on top of that fixed
        per-day interpretation. This is the "extreme desire to not forget
        the basic prior" mechanism -- a small or skewed training set can
        still bend the head somewhat, but it can never rewrite the
        foundation into believing (say) that RSI=90 is a dip-buy signal.

    Foundation weights are set from a separate helper (see
    _ml_set_foundation_weights) so the numbers are auditable in one place.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(ML_BRAIN_SEQ_LEN, ML_BRAIN_FEATURES), name='features_seq')

    # ---- FROZEN FOUNDATION (per-timestep) ----
    foundation_1 = layers.TimeDistributed(
        layers.Dense(8, activation='tanh', name='foundation_1_inner'),
        name='foundation_1',
    )
    foundation_1.trainable = False

    foundation_2 = layers.TimeDistributed(
        layers.Dense(6, activation='tanh', name='foundation_2_inner'),
        name='foundation_2',
    )
    foundation_2.trainable = False

    x = foundation_1(inputs)
    x = foundation_2(x)

    # ---- TRAINABLE HEAD (Conv1D -> LSTM -> LSTM -> Dense) ----
    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4),
                      name='head_conv1d')(x)
    x = layers.BatchNormalization(name='head_batchnorm')(x)
    x = layers.Dropout(0.25, name='head_dropout_1')(x)
    x = layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    name='head_lstm_1')(x)
    x = layers.Dropout(0.25, name='head_dropout_2')(x)
    x = layers.LSTM(64, name='head_lstm_2')(x)
    x = layers.Dropout(0.20, name='head_dropout_3')(x)
    x = layers.Dense(32, activation='relu', name='head_dense')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='win_prob')(x)

    model = keras.Model(inputs, outputs, name='ml_brain')

    # Foundation weights MUST be set AFTER the model is built (so the layers
    # have concrete weight shapes), but before compile so training never
    # sees the untouched random-init foundation values on first fit.
    _ml_set_foundation_weights(model)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=ML_BRAIN_LEARNING_RATE),
                  loss=_ml_make_focal_loss(ML_BRAIN_FOCAL_GAMMA),
                  metrics=['accuracy'])
    return model


def _ml_set_foundation_weights(model):
    """Hand-specified, non-trainable weights for the two frozen foundation
    layers. Auditable numeric constants -- NOT learned from any data.

    Feature index reminder (see _ml_build_feature_row):
        0 RSI/100           -- lower = more oversold (good for dip-buy)
        1 MACD raw
        2 MACD signal raw
        3 MACD > signal     -- 1.0 when bullish crossover
        4 ATR%              -- higher = more volatile (both risk and opportunity)
        5 day's return
        6 volume vs SMA     -- 0 = at average, >0 = above average
        7 close vs SMA20    -- 0 = at SMA, >0 = above short-term trend
        8 close vs SMA50    -- 0 = at SMA, >0 = above medium-term trend
        9 SMA20 > SMA50     -- 1.0 when short-term trend > long-term trend

    ---- foundation_1 (8 hidden units, tanh) ----
    Each unit is a hand-built rule detector. Weights below are set so the
    unit's pre-activation is meaningfully large (>0.5 in magnitude) when
    the corresponding condition is TRUE, and near zero otherwise. tanh
    then squashes to roughly [-1, +1] per unit.

        unit 0: OVERSOLD_DETECTOR       -- fires on low RSI (RSI/100 near 0)
        unit 1: MACD_BULL_DETECTOR      -- fires on MACD > signal
        unit 2: UPTREND_DETECTOR        -- fires on SMA20 > SMA50 AND close > SMA50
        unit 3: DIP_BUY_COMPOSITE       -- fires when oversold AND in uptrend
        unit 4: VOLATILITY_DETECTOR     -- fires on high ATR%
        unit 5: HIGH_VOLUME_DETECTOR    -- fires on volume > SMA
        unit 6: SHORT_TERM_STRENGTH     -- fires on close > SMA20
        unit 7: BEARISH_DETECTOR        -- fires on MACD<signal AND SMA20<SMA50
                                            (negative signal, used to counter-weight)

    ---- foundation_2 (6 hidden units, tanh) ----
    Combines those 8 rule detectors into broader per-day setup summaries.
        unit 0: "clean dip-buy setup"     -- OVERSOLD + UPTREND + MACD_BULL
        unit 1: "confirmed reversal"      -- OVERSOLD + MACD_BULL + HIGH_VOLUME
        unit 2: "healthy uptrend"         -- UPTREND + SHORT_TERM_STRENGTH
        unit 3: "high-risk environment"   -- VOLATILITY (positive weight so head sees it)
        unit 4: "bearish setup"           -- BEARISH_DETECTOR (negative caution signal)
        unit 5: "raw uptrend passthrough" -- direct SHORT_TERM_STRENGTH signal
    """
    import numpy as np
    n_features = ML_BRAIN_FEATURES   # 10
    n_hidden_1 = 8
    n_hidden_2 = 6

    # ---- Foundation 1 weights: (n_features, n_hidden_1) = (10, 8) ----
    W1 = np.zeros((n_features, n_hidden_1), dtype=np.float32)
    b1 = np.zeros((n_hidden_1,), dtype=np.float32)

    # unit 0: OVERSOLD_DETECTOR
    #   Pre-activation = -6*(RSI/100) + 2. RSI=30 (oversold) -> +0.2 -> tanh~0.2.
    #   RSI=70 (overbought) -> -2.2 -> tanh~-0.98. Weakly positive for oversold,
    #   strongly negative for overbought.
    W1[0, 0] = -6.0
    b1[0] = 2.0

    # unit 1: MACD_BULL_DETECTOR -- MACD>signal indicator directly
    W1[3, 1] = 3.0
    b1[1] = -1.0   # so it needs the indicator TRUE to activate positively

    # unit 2: UPTREND_DETECTOR -- SMA20>SMA50 AND close>SMA50
    W1[9, 2] = 2.0    # SMA20>SMA50 flag
    W1[8, 2] = 5.0    # close>SMA50 distance (0 to a few percent)
    b1[2] = -1.5

    # unit 3: DIP_BUY_COMPOSITE -- oversold (low RSI) AND in uptrend
    W1[0, 3] = -4.0   # low RSI is good
    W1[9, 3] = 2.0    # AND uptrend
    b1[3] = 0.5

    # unit 4: VOLATILITY_DETECTOR -- ATR% (typical ~0.01-0.05)
    W1[4, 4] = 30.0   # ATR%=0.03 -> +0.9
    b1[4] = -0.5

    # unit 5: HIGH_VOLUME_DETECTOR -- volume vs SMA (0=avg, +0.5=50% above)
    W1[6, 5] = 3.0
    b1[5] = 0.0

    # unit 6: SHORT_TERM_STRENGTH -- close vs SMA20 (positive when above)
    W1[7, 6] = 15.0   # close 2% above SMA20 -> +0.3
    b1[6] = 0.0

    # unit 7: BEARISH_DETECTOR -- MACD_below_signal AND SMA20<SMA50
    W1[3, 7] = -3.0   # MACD<signal (indicator is 0) -> no penalty here, but strong penalty when 1
    W1[9, 7] = -2.0   # SMA20<SMA50 -> stays 0 (no negation), positive when 1 -> -2
    b1[7] = 2.5       # baseline high; MACD_bull+uptrend suppress it toward zero

    # Set foundation_1 weights. The layer wraps a Dense in TimeDistributed,
    # so we address the inner Dense's weights.
    f1_layer = model.get_layer('foundation_1').layer
    f1_layer.set_weights([W1, b1])

    # ---- Foundation 2 weights: (n_hidden_1, n_hidden_2) = (8, 6) ----
    # Input is the tanh-activated output of foundation_1, in ~[-1, +1] per unit.
    W2 = np.zeros((n_hidden_1, n_hidden_2), dtype=np.float32)
    b2 = np.zeros((n_hidden_2,), dtype=np.float32)

    # foundation_1 unit indices for reference:
    #   0=OVERSOLD, 1=MACD_BULL, 2=UPTREND, 3=DIP_BUY, 4=VOLATILITY,
    #   5=HIGH_VOL, 6=SHORT_STRENGTH, 7=BEARISH

    # unit 0: clean dip-buy setup = OVERSOLD + UPTREND + MACD_BULL
    W2[0, 0] = 0.8
    W2[2, 0] = 0.8
    W2[1, 0] = 0.8
    b2[0] = -0.5   # requires at least two to activate

    # unit 1: confirmed reversal = OVERSOLD + MACD_BULL + HIGH_VOLUME
    W2[0, 1] = 0.7
    W2[1, 1] = 0.9
    W2[5, 1] = 0.6
    b2[1] = -0.6

    # unit 2: healthy uptrend = UPTREND + SHORT_STRENGTH
    W2[2, 2] = 1.0
    W2[6, 2] = 0.8
    b2[2] = -0.3

    # unit 3: high-risk environment = VOLATILITY (positive so head sees it)
    W2[4, 3] = 1.5
    b2[3] = -0.2

    # unit 4: bearish setup = BEARISH_DETECTOR (negative caution)
    W2[7, 4] = -1.5   # negate: when foundation_1 bearish unit fires positive, unit 4 fires negative
    b2[4] = 0.0

    # unit 5: raw uptrend passthrough
    W2[6, 5] = 1.2
    W2[2, 5] = 0.5
    b2[5] = -0.2

    f2_layer = model.get_layer('foundation_2').layer
    f2_layer.set_weights([W2, b2])



def _ml_build_feature_row(rsi_val, macd_val, macd_sig_val, atr_val, close_val,
                          volume_val, vol_sma_val, close_sma20, close_sma50,
                          bar_return):
    """One day's feature snapshot (10 features), used both when generating
    historical training rows and (eventually) when scoring at inference time.
    Kept intentionally small so the model is fast to train even at the
    15k-example daily budget."""
    def _fin(x, default=0.0):
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    close = _fin(close_val, 1.0)
    return [
        _fin(rsi_val, 50.0) / 100.0,                           # 0: RSI normalized [0,1]
        _fin(macd_val, 0.0),                                    # 1: MACD raw
        _fin(macd_sig_val, 0.0),                                # 2: MACD signal raw
        1.0 if _fin(macd_val) > _fin(macd_sig_val) else 0.0,   # 3: MACD-above-signal indicator
        _fin(atr_val, 0.0) / max(close, 0.01),                 # 4: ATR%
        _fin(bar_return, 0.0),                                  # 5: day's return
        (_fin(volume_val, 0.0) / max(_fin(vol_sma_val, 1.0), 1.0)) - 1.0,  # 6: volume vs SMA
        (close / max(_fin(close_sma20, close), 0.01)) - 1.0,   # 7: distance from 20-SMA
        (close / max(_fin(close_sma50, close), 0.01)) - 1.0,   # 8: distance from 50-SMA
        1.0 if _fin(close_sma20, 0) > _fin(close_sma50, 0) else 0.0,  # 9: short-trend > long-trend
    ]


def _ml_build_examples_for_symbol(symbol, df, n_examples_wanted):
    """Turns one symbol's daily OHLCV history into (sequence, label) training
    pairs -- a sequence is ML_BRAIN_SEQ_LEN=20 consecutive daily feature rows
    ending at day N; label is 1 if close[N + ML_HIST_FORWARD_LABEL_DAYS] >
    close[N], else 0. Only anchor points where all indicators are valid are
    used. Caller specifies how many examples they want to try to draw; if the
    symbol doesn't have that many valid anchor points, returns fewer.
    """
    if df is None or df.empty or len(df) < ML_HIST_MIN_ROWS_PER_SYMBOL:
        return []

    # BUGFIX: previously we did `np.any(np.isnan(close))` and returned [] on
    # a single NaN. yfinance frequently returns an all-NaN row for the current
    # in-progress trading day when the batch call runs while the market is
    # open (which is exactly what happens on first-run training right after
    # market open). A single NaN row was killing every symbol -- log showed
    # "0 examples from 0/27 symbols". Drop rows where OHLC is NaN first, then
    # proceed. The per-anchor NaN guard below (rsi/macd/atr/sma50 not-NaN)
    # still keeps bad indicator rows out of the training set.
    df = df.dropna(subset=['Close', 'High', 'Low'])
    if len(df) < ML_HIST_MIN_ROWS_PER_SYMBOL:
        return []

    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    volume = df['Volume'].values.astype(np.float64) if 'Volume' in df else np.zeros(len(close))

    if len(close) < ML_HIST_MIN_ROWS_PER_SYMBOL:
        return []

    try:
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        atr = talib.ATR(high, low, close, timeperiod=14)
        vol_sma = talib.SMA(volume, timeperiod=14) if volume.sum() > 0 else np.full(len(close), 1.0)
        sma20 = talib.SMA(close, timeperiod=20)
        sma50 = talib.SMA(close, timeperiod=50)
    except Exception as e:
        logging.warning(f"ml_brain hist: indicator calc failed for {symbol}: {e}")
        return []

    daily_return = np.diff(close, prepend=close[0]) / np.maximum(close, 0.01)

    # Pre-build every day's feature row once, then slice sequences from it.
    n_days = len(close)
    feature_rows = np.zeros((n_days, ML_BRAIN_FEATURES), dtype=np.float32)
    for i in range(n_days):
        feature_rows[i] = _ml_build_feature_row(
            rsi[i], macd[i], macd_signal[i], atr[i], close[i],
            volume[i], vol_sma[i], sma20[i], sma50[i], daily_return[i])

    # Valid anchor points: need SEQ_LEN days behind for the input window,
    # and enough forward days for the exit simulator to reach a terminal.
    # Bumping the forward window from ML_HIST_FORWARD_LABEL_DAYS (5) to a
    # longer horizon so the profit monitor and hard stop have room to
    # actually fire the way they would in live trading.
    ML_HIST_SIM_MAX_DAYS = max(20, ML_HIST_FORWARD_LABEL_DAYS)
    first_valid = 50  # after SMA50 stabilizes
    last_valid = n_days - ML_HIST_SIM_MAX_DAYS - 1
    if last_valid <= first_valid:
        return []

    candidate_anchors = []
    for i in range(max(first_valid, ML_BRAIN_SEQ_LEN), last_valid + 1):
        if (not np.isnan(rsi[i]) and not np.isnan(macd[i]) and
            not np.isnan(atr[i]) and not np.isnan(sma50[i])):
            candidate_anchors.append(i)

    if not candidate_anchors:
        return []

    # Sample uniformly across the symbol's history -- avoids over-weighting
    # any particular regime the symbol happened to spend a lot of time in.
    rng = np.random.default_rng(hash(symbol) & 0xFFFFFFFF)
    if len(candidate_anchors) > n_examples_wanted:
        picked = rng.choice(candidate_anchors, n_examples_wanted, replace=False)
    else:
        picked = candidate_anchors

    examples = []
    for i in picked:
        seq = feature_rows[i - ML_BRAIN_SEQ_LEN + 1 : i + 1]
        if seq.shape != (ML_BRAIN_SEQ_LEN, ML_BRAIN_FEATURES):
            continue

        # NEW LABEL (item #6 from the latest review): instead of the naive
        # "close[i+5] > close[i]" question, simulate what the bot's actual
        # exit rules would have done starting from this entry, and label
        # 1 iff the SIMULATED EXIT was profitable. This trains the ML brain
        # on the same question the live bot is trying to answer, not a
        # cheaper proxy that doesn't reflect the exit machinery.
        entry_price = close[i]
        entry_atr = atr[i]
        label = _ml_simulate_exit_label(
            entry_price=entry_price, entry_atr=entry_atr,
            forward_close=close[i + 1 : i + 1 + ML_HIST_SIM_MAX_DAYS],
            forward_high=high[i + 1 : i + 1 + ML_HIST_SIM_MAX_DAYS],
            forward_low=low[i + 1 : i + 1 + ML_HIST_SIM_MAX_DAYS])
        # Anchor date is the calendar date of the last bar in this sequence
        # -- needed downstream so walk-forward evaluation can sort examples
        # chronologically across ALL symbols instead of the previous
        # symbol-chunked order (which meant "walk-forward" was really just
        # walking forward through whichever symbol happened to be first in
        # the download batch). Falls back to None if the DataFrame's index
        # isn't a DatetimeIndex, so mocks/tests without dates still work.
        try:
            anchor_date = df.index[i].to_pydatetime() if hasattr(df.index[i], 'to_pydatetime') else None
        except Exception:
            anchor_date = None
        examples.append((seq.astype(np.float32), float(label), anchor_date))
    return examples


def _ml_simulate_exit_label(entry_price, entry_atr, forward_close, forward_high, forward_low):
    """Simulate the bot's actual exit logic day-by-day forward from an entry,
    and return 1.0 if the simulated exit was profitable (>0% net), else 0.0.

    Uses the SAME constants the live bot uses -- HARD_STOP_ATR_MULTIPLIER,
    HARD_STOP_MIN_PCT, ARM_PROFIT_PCT, PEAK_GIVEBACK_FRACTION, SCALE_OUT_STAGES.
    Approximations documented inline:

      1. Daily bars only. Real intraday paths could hit both the stop AND
         the peak on the same day; we assume "worst first" for the label
         (if low[k] hits the hard stop, we exit at the stop, period).
      2. Scale-out tranches take the day's CLOSE as the fill price (real
         fills would land somewhere between high and low intraday).
      3. Slippage/spread ignored -- consistent with the training data being
         theoretical rather than tick-accurate.

    None of these approximations invalidate the label direction (profitable
    vs. not); they just add some noise around the exact realized return.
    """
    if entry_price <= 0 or entry_atr is None or entry_atr <= 0:
        return 0.0
    if len(forward_close) == 0:
        return 0.0

    stop_distance_pct = max(HARD_STOP_ATR_MULTIPLIER * (entry_atr / entry_price),
                            HARD_STOP_MIN_PCT)
    atr_pct = entry_atr / entry_price
    arm_pct = max(ARM_PROFIT_PCT, ATR_ARM_MULTIPLIER * atr_pct)

    # Track scale-out state: fraction of position still open, and cumulative
    # locked-in profit from tranches already sold.
    remaining_frac = 1.0
    realized = 0.0
    fired_stages = set()
    armed = False
    peak = entry_price
    floor_pct = HARD_FLOOR_PCT

    for k in range(len(forward_close)):
        day_low = forward_low[k]
        day_high = forward_high[k]
        day_close = forward_close[k]

        gain_low = (day_low - entry_price) / entry_price
        gain_high = (day_high - entry_price) / entry_price

        # 1. Hard stop -- fires FIRST any day the low breaches it. Everything
        # still open exits at the stop price.
        if gain_low <= -stop_distance_pct:
            realized += remaining_frac * (-stop_distance_pct)
            return 1.0 if realized > 0 else 0.0

        # 2. Scale-out tranches on the high side (uses day_high as trigger,
        # day_close as fill price -- see docstring approx #2).
        for idx, (trigger_pct, tranche_frac) in enumerate(SCALE_OUT_STAGES):
            if idx in fired_stages: continue
            if gain_high >= trigger_pct:
                tranche_realized = tranche_frac * ((day_close - entry_price) / entry_price)
                realized += tranche_realized
                remaining_frac -= tranche_frac
                fired_stages.add(idx)
                # Live bot moves the floor to breakeven after the first tranche.
                floor_pct = max(floor_pct, 0.0005)

        # 3. Profit monitor: arm above arm_pct, then follow peak, exit on
        # giveback. Peak tracks the intraday high.
        if not armed and gain_high >= arm_pct:
            armed = True
            peak = day_high
        elif armed:
            peak = max(peak, day_high)

        if armed and remaining_frac > 0.0001:
            peak_gain = (peak - entry_price) / entry_price
            giveback_pct = max(PEAK_GIVEBACK_PCT,
                               ATR_GIVEBACK_FRACTION * arm_pct,
                               PEAK_GIVEBACK_FRACTION * peak_gain)
            gain_close = (day_close - entry_price) / entry_price
            giveback = (peak - day_close) / peak if peak > 0 else 0.0
            if giveback >= giveback_pct and gain_close >= floor_pct:
                realized += remaining_frac * gain_close
                remaining_frac = 0.0
                return 1.0 if realized > 0 else 0.0

        if remaining_frac <= 0.0001:
            return 1.0 if realized > 0 else 0.0

    # End of the forward window without a terminal exit: mark to market.
    final_close = forward_close[-1]
    realized += remaining_frac * (final_close - entry_price) / entry_price
    return 1.0 if realized > 0 else 0.0


def _ml_simulate_exit_return(entry_price, entry_atr, forward_close, forward_high, forward_low):
    """Numeric-outcome variant of _ml_simulate_exit_label. Returns the
    simulated realized return as a float (positive = profitable) rather than
    a binary label. Used by analog-based pre-trade analysis where we want
    the actual expected P&L across analogs, not just the win rate.

    Approximations (identical to _ml_simulate_exit_label): daily bars only,
    scale-outs fill at day's close price, no slippage/spread. Consistent
    with the training data being theoretical rather than tick-accurate.
    """
    if entry_price <= 0 or entry_atr is None or entry_atr <= 0:
        return 0.0
    if len(forward_close) == 0:
        return 0.0

    stop_distance_pct = max(HARD_STOP_ATR_MULTIPLIER * (entry_atr / entry_price),
                            HARD_STOP_MIN_PCT)
    atr_pct = entry_atr / entry_price
    arm_pct = max(ARM_PROFIT_PCT, ATR_ARM_MULTIPLIER * atr_pct)

    remaining_frac = 1.0
    realized = 0.0
    fired_stages = set()
    armed = False
    peak = entry_price
    floor_pct = HARD_FLOOR_PCT

    for k in range(len(forward_close)):
        day_low = forward_low[k]
        day_high = forward_high[k]
        day_close = forward_close[k]

        gain_low = (day_low - entry_price) / entry_price
        gain_high = (day_high - entry_price) / entry_price

        if gain_low <= -stop_distance_pct:
            realized += remaining_frac * (-stop_distance_pct)
            return realized

        for idx, (trigger_pct, tranche_frac) in enumerate(SCALE_OUT_STAGES):
            if idx in fired_stages: continue
            if gain_high >= trigger_pct:
                tranche_realized = tranche_frac * ((day_close - entry_price) / entry_price)
                realized += tranche_realized
                remaining_frac -= tranche_frac
                fired_stages.add(idx)
                floor_pct = max(floor_pct, 0.0005)

        if not armed and gain_high >= arm_pct:
            armed = True
            peak = day_high
        elif armed:
            peak = max(peak, day_high)

        if armed and remaining_frac > 0.0001:
            peak_gain = (peak - entry_price) / entry_price
            giveback_pct = max(PEAK_GIVEBACK_PCT,
                               ATR_GIVEBACK_FRACTION * arm_pct,
                               PEAK_GIVEBACK_FRACTION * peak_gain)
            gain_close = (day_close - entry_price) / entry_price
            giveback = (peak - day_close) / peak if peak > 0 else 0.0
            if giveback >= giveback_pct and gain_close >= floor_pct:
                realized += remaining_frac * gain_close
                return realized

        if remaining_frac <= 0.0001:
            return realized

    final_close = forward_close[-1]
    realized += remaining_frac * (final_close - entry_price) / entry_price
    return realized


# =====================================================================
# Pre-trade analog analyzer ("would this trade have worked historically?")
# =====================================================================
#
# For every candidate the bot considers buying, this looks up historical days
# where the SAME symbol had similar feature values, simulates the bot's actual
# exit rules from each analog day using intraday-high/low approximation, and
# reports what percentage of those historical analogs were profitable + what
# the mean realized return was.
#
# This is NOT a full walk-forward strategy backtest -- it's a targeted lookup
# that answers "how did this exact setup do in this symbol's own past?" at
# scoring time. Much cheaper than a full simulator, and it uses the same
# _ml_simulate_exit_return function so the exit rules being tested against
# are the exact ones the live bot will apply if we do buy.
#
# Cached to disk so re-runs on the same symbol within a 24-hour window skip
# the yfinance download entirely -- a full daily scan of dozens of candidates
# then only pays the network cost once per symbol per day.

ANALOG_CACHE_DIR = os.path.join(_ml_base_dir, 'analog_cache')
ANALOG_CACHE_TTL_HOURS = 24
ANALOG_LOOKBACK_YEARS = 2
ANALOG_DEFAULT_K_NEIGHBORS = 25    # k nearest historical days by feature distance
ANALOG_MIN_HISTORY_ROWS = 100      # skip symbol if less than this much daily history
ANALOG_MIN_ANALOGS = 5             # skip symbol if fewer than this many neighbors within threshold

# Buy-pipeline integration toggles. USE_ANALOG_ADJUSTMENT=False completely
# disables the analog analyzer at scoring time -- everything else in the
# analog section still works if called directly, but no yfinance downloads
# fire during buy scans and no score adjustments happen.
USE_ANALOG_ADJUSTMENT = True
USE_ANALOG_MAX_ADJUSTMENT = 1.0    # cap on how many buy-score points the analog signal can add/subtract


def _analog_cache_path(symbol):
    # .pkl not .parquet because parquet requires pyarrow, which is an
    # extra ~100MB dep. Pandas' native pickle serializer is stdlib-only
    # and handles DatetimeIndex + typed columns just fine for our purposes.
    return os.path.join(ANALOG_CACHE_DIR, f"{symbol.replace('/', '-')}.pkl")


def _analog_load_or_fetch_history(symbol, days_lookback):
    """Returns a daily-OHLCV DataFrame for the symbol, using a local parquet
    cache if it's less than ANALOG_CACHE_TTL_HOURS old. Otherwise fetches via
    yf_download_batch (shared rate limiter) and updates the cache. Returns
    None if fetch fails and no cache exists.
    """
    os.makedirs(ANALOG_CACHE_DIR, exist_ok=True)
    cache_path = _analog_cache_path(symbol)

    try:
        if os.path.exists(cache_path):
            cache_age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600.0
            if cache_age_hours < ANALOG_CACHE_TTL_HOURS:
                import pandas as pd
                return pd.read_pickle(cache_path)
    except Exception as e:
        logging.warning(f"analog cache read failed for {symbol}: {e}")

    # Cache stale or missing -- fetch fresh
    end_date = datetime.now(eastern).date()
    start_date = end_date - timedelta(days=days_lookback)
    try:
        data_by_symbol = yf_download_batch([symbol],
                                          start=start_date.isoformat(),
                                          end=end_date.isoformat(),
                                          interval='1d')
        df = data_by_symbol.get(symbol)
        if df is None or df.empty:
            # Fall back to stale cache if fetch returned empty
            if os.path.exists(cache_path):
                import pandas as pd
                return pd.read_pickle(cache_path)
            return None
        try:
            df.to_pickle(cache_path)
        except Exception as e:
            logging.warning(f"analog cache write failed for {symbol}: {e}")
        return df
    except Exception as e:
        logging.warning(f"analog history fetch failed for {symbol}: {e}")
        # Fall back to any cached copy, even if stale
        if os.path.exists(cache_path):
            try:
                import pandas as pd
                return pd.read_pickle(cache_path)
            except Exception:
                pass
        return None


def _analog_build_all_feature_rows(df):
    """Build one feature row per day in the symbol's history using the SAME
    _ml_build_feature_row helper the ML training pipeline uses. Returns
    (feature_matrix, close, high, low, atr_series) or None if data is too
    short or indicator calc fails.
    """
    if df is None or df.empty or len(df) < ANALOG_MIN_HISTORY_ROWS:
        return None

    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    volume = df['Volume'].values.astype(np.float64) if 'Volume' in df else np.zeros(len(close))

    if np.any(np.isnan(close)):
        return None

    try:
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        atr = talib.ATR(high, low, close, timeperiod=14)
        vol_sma = talib.SMA(volume, timeperiod=14) if volume.sum() > 0 else np.full(len(close), 1.0)
        sma20 = talib.SMA(close, timeperiod=20)
        sma50 = talib.SMA(close, timeperiod=50)
    except Exception as e:
        logging.warning(f"analog indicator calc failed: {e}")
        return None

    daily_return = np.diff(close, prepend=close[0]) / np.maximum(close, 0.01)

    n_days = len(close)
    feature_rows = np.zeros((n_days, ML_BRAIN_FEATURES), dtype=np.float32)
    for i in range(n_days):
        feature_rows[i] = _ml_build_feature_row(
            rsi[i], macd[i], macd_signal[i], atr[i], close[i],
            volume[i], vol_sma[i], sma20[i], sma50[i], daily_return[i])

    return feature_rows, close, high, low, atr


def analyze_trade_analog(symbol, k_neighbors=ANALOG_DEFAULT_K_NEIGHBORS,
                          forward_days=None):
    """Pre-trade analog analyzer. For the given symbol, finds the k historical
    days whose feature vector most closely matches TODAY (last row of the
    symbol's daily history), simulates the bot's actual exit rules from each
    of those historical days, and returns aggregate stats.

    Returns dict with:
        n_analogs           -- how many analogs actually simulated
        win_rate            -- fraction of analogs that were profitable
        mean_return         -- average simulated realized return
        median_return       -- median simulated realized return
        best_return         -- best analog outcome
        worst_return        -- worst analog outcome (usually the hard-stop distance)
        std_return          -- std dev of simulated returns
        expectancy          -- win_rate * avg_win + (1-win_rate) * avg_loss
        mean_feature_dist   -- how similar the analogs are, on average (0=identical)
    or None if not enough history or no analogs found.

    Uses ANALOG_LOOKBACK_YEARS of daily history (cached to disk with
    ANALOG_CACHE_TTL_HOURS TTL). Reuses _ml_simulate_exit_return so the exit
    rules being tested are exactly the ones the live bot will apply.
    """
    if forward_days is None:
        forward_days = max(20, ML_HIST_FORWARD_LABEL_DAYS)

    days_lookback = int(365 * ANALOG_LOOKBACK_YEARS)
    df = _analog_load_or_fetch_history(symbol, days_lookback)
    if df is None:
        return None

    built = _analog_build_all_feature_rows(df)
    if built is None:
        return None
    feature_rows, close, high, low, atr = built

    # Today's feature row is the LAST row of the history. Analogs must have
    # enough forward data for the simulator to reach a real exit, so exclude
    # the last `forward_days + 5` rows from the analog pool (they'd terminate
    # early on end-of-window mark-to-market with too little info).
    if len(feature_rows) < 50 + forward_days + 5:
        return None

    today_features = feature_rows[-1]
    if np.any(np.isnan(today_features)) or np.any(np.isinf(today_features)):
        return None

    valid_end = len(feature_rows) - forward_days - 5
    if valid_end < 50:
        return None

    # Feature distance (L2, normalized per-feature by its own std across history).
    hist_rows = feature_rows[50:valid_end]
    if len(hist_rows) < ANALOG_MIN_ANALOGS:
        return None

    stds = hist_rows.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)   # avoid div-by-zero on constant features
    diffs = (hist_rows - today_features) / stds
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    # k nearest neighbors
    k = min(k_neighbors, len(dists))
    nearest_indices = np.argpartition(dists, k - 1)[:k]
    # Map back to global indices (we sliced away first 50)
    nearest_global = nearest_indices + 50

    if len(nearest_global) < ANALOG_MIN_ANALOGS:
        return None

    # Simulate each analog through the bot's actual exit rules
    outcomes = []
    for idx in nearest_global:
        entry_price = close[idx]
        entry_atr = atr[idx]
        if not np.isfinite(entry_price) or not np.isfinite(entry_atr):
            continue
        fwd_close = close[idx + 1 : idx + 1 + forward_days]
        fwd_high = high[idx + 1 : idx + 1 + forward_days]
        fwd_low = low[idx + 1 : idx + 1 + forward_days]
        if len(fwd_close) < forward_days // 2:   # too little forward data
            continue
        ret = _ml_simulate_exit_return(entry_price, entry_atr, fwd_close, fwd_high, fwd_low)
        outcomes.append(ret)

    if len(outcomes) < ANALOG_MIN_ANALOGS:
        return None

    arr = np.array(outcomes)
    winners = arr[arr > 0]
    losers = arr[arr < 0]
    win_rate = float((arr > 0).sum()) / len(arr)
    avg_win = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(losers.mean()) if len(losers) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        'symbol': symbol,
        'n_analogs': len(arr),
        'win_rate': win_rate,
        'mean_return': float(arr.mean()),
        'median_return': float(np.median(arr)),
        'best_return': float(arr.max()),
        'worst_return': float(arr.min()),
        'std_return': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'expectancy': expectancy,
        'mean_feature_dist': float(dists[nearest_indices].mean()),
    }





# NASDAQ-100 focus + S&P 500 large-cap supplement. Kept as a hand-curated
# constant rather than a web-scrape so pretraining is deterministic across runs
# and never blocks on Wikipedia parsing. Overlap between the two is fine --
# we deduplicate before use. Bias is intentionally NASDAQ-heavy (this is a
# NASDAQ-focused bot); the S&P adds sector breadth (energy, financials,
# industrials, staples) so the model sees regimes NASDAQ names don't produce
# on their own.
_ML_PRETRAIN_NASDAQ100 = [
    'AAPL','MSFT','NVDA','GOOGL','GOOG','AMZN','META','TSLA','AVGO','COST',
    'NFLX','ADBE','PEP','CSCO','AMD','TMUS','INTC','CMCSA','QCOM','AMGN',
    'TXN','HON','INTU','AMAT','ISRG','BKNG','SBUX','MU','ADI','LRCX',
    'ADP','GILD','MDLZ','VRTX','REGN','PANW','KLAC','SNPS','CDNS','MELI',
    'ASML','CSX','MAR','ORLY','ABNB','CHTR','PYPL','FTNT','MNST','NXPI',
    'ADSK','WDAY','CTAS','ROP','PCAR','MRVL','KDP','AEP','PAYX','ODFL',
    'FAST','KHC','ROST','DXCM','EXC','BKR','CPRT','MCHP','IDXX','EA',
    'CTSH','XEL','GEHC','LULU','VRSK','CSGP','CCEP','ANSS','FANG','ON',
    'TTD','CDW','TEAM','DDOG','BIIB','ZS','MDB','ILMN','SIRI','WBD',
    'SMCI','ARM','CRWD','APP','PLTR','GFS','MRNA','TTWO','WBA','DASH',
]
_ML_PRETRAIN_SP500_LARGE = [
    # Large-caps outside the NASDAQ-100 (mostly NYSE) for sector breadth.
    'BRK-B','JPM','V','WMT','XOM','UNH','JNJ','MA','PG','HD',
    'CVX','LLY','ABBV','BAC','MRK','KO','PFE','TMO','ORCL','ACN',
    'MCD','DHR','ABT','WFC','CRM','LIN','DIS','VZ','NKE','PM',
    'NEE','RTX','UPS','MS','CAT','BMY','LOW','T','GS','SPGI',
    'BLK','AXP','SCHW','DE','ELV','LMT','C','MDT','SYK','PLD',
    'GE','TJX','BA','MMC','CB','ADI','ADP','ETN','SO','DUK',
]

def _ml_hist_fetch_symbol_universe():
    """Universe for pretraining the ML brain. NASDAQ-100 focused with S&P 500
    large-cap supplement for sector breadth. Deduplicated. This is broader
    than the live watchlist by design: pretraining should teach the brain
    what winning setups look like ACROSS the tape, not just the ~25-40
    symbols the bot happens to be scanning today. Live fine-tuning on the
    bot's own closed trades happens separately in
    _ml_maintenance_train_on_live_trades and specializes the model to what
    it actually trades.
    """
    seen = set()
    universe = []
    for sym in _ML_PRETRAIN_NASDAQ100 + _ML_PRETRAIN_SP500_LARGE:
        if sym not in seen:
            seen.add(sym)
            universe.append(sym)
    return universe


def _ml_gather_training_examples(target_count):
    """Downloads the current candidate-list universe's historical data
    (yfinance, batched to respect the shared rate limiter) and builds up to
    `target_count` (sequence, label) examples across all symbols.

    Returns (examples_list, symbols_used, symbols_attempted).
    """
    symbols = _ml_hist_fetch_symbol_universe()
    if not symbols:
        return [], 0, 0

    end_date = datetime.now(eastern).date()
    start_date = end_date - timedelta(days=int(365 * ML_HIST_LOOKBACK_YEARS))

    try:
        data_by_symbol = yf_download_batch(symbols, start=start_date.isoformat(),
                                           end=end_date.isoformat(), interval='1d')
    except Exception as e:
        logging.error(f"ml_brain: batch download failed: {e}")
        return [], 0, len(symbols)

    if not data_by_symbol:
        return [], 0, len(symbols)

    # Even split of the target budget across symbols that returned data --
    # prevents any single very-long-history symbol from dominating.
    per_symbol_budget = max(20, target_count // max(1, len(data_by_symbol)))

    all_examples = []
    symbols_used = 0
    for sym, df in data_by_symbol.items():
        examples = _ml_build_examples_for_symbol(sym, df, per_symbol_budget)
        if examples:
            symbols_used += 1
            all_examples.extend(examples)
        if len(all_examples) >= target_count:
            break

    # If we're way over target, trim; if under, take what we got.
    if len(all_examples) > target_count:
        rng = np.random.default_rng(0)
        picks = rng.choice(len(all_examples), target_count, replace=False)
        all_examples = [all_examples[i] for i in picks]

    return all_examples, symbols_used, len(symbols)


def _ml_walk_forward_evaluation(examples, tf, n_folds=5):
    """
    Rolling walk-forward evaluation: sort examples chronologically by their
    anchor date (from _ml_build_examples_for_symbol), then run N sliding
    folds. Each fold's training window is calendar-EARLIER than its
    validation window, so per-fold accuracy degradation across folds reveals
    a model that's overfitting to older market regimes instead of
    generalizing forward through time. Prior versions of this function
    walked forward through example ORDER (which was symbol-chunked, not
    chronological); this one walks forward through calendar time.

    This is DIAGNOSTIC ONLY -- the final model that gets saved to disk is
    still trained on the full example set by _ml_train_on_examples's main
    fit call after this returns. Walk-forward here just tells us how much
    to trust that final model. Returns a short status string for logging.

    NOTE: this is a small-sample walk-forward, not a proper backtest. Real
    walk-forward against genuinely out-of-sample market regimes requires a
    price-tick simulator that replays the bot's actual exit logic against
    historical bars. That's a separate, much larger piece of infrastructure.
    """
    if len(examples) < 200:
        return f"not enough examples for walk-forward (need 200+, got {len(examples)})"

    # Sort examples chronologically by anchor_date (third element of each
    # tuple). This is the real walk-forward: each fold's training window is
    # calendar-earlier than its validation window, so fold N->N+1 accuracy
    # degradation reveals a model that's overfitting to older market regimes
    # instead of generalizing forward through time. Examples with no
    # anchor_date (mocks, or a df without a DatetimeIndex) sink to the end
    # so real-dated examples dominate the earlier folds -- the alternative
    # (drop them) would silently shrink the training set on tests.
    def _sort_key(e):
        d = e[2] if len(e) >= 3 else None
        # Sentinel far-future value so None-dated examples sort last
        return d if d is not None else datetime(9999, 1, 1)

    sorted_examples = sorted(examples, key=_sort_key)
    dated_count = sum(1 for e in sorted_examples if len(e) >= 3 and e[2] is not None)

    X = np.stack([e[0] for e in sorted_examples]).astype(np.float32)
    y = np.array([e[1] for e in sorted_examples], dtype=np.float32)

    fold_size = len(X) // n_folds
    accs = []
    losses = []
    for fold in range(n_folds - 1):
        train_end = fold_size * (fold + 1)
        val_end = min(train_end + fold_size, len(X))
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        if len(X_val) < 10 or len(X_train) < 50:
            continue
        model = _ml_build_model(tf)
        try:
            model.fit(X_train, y_train, epochs=4,
                     batch_size=min(ML_BRAIN_BATCH_SIZE, len(X_train)),
                     verbose=0, shuffle=True)
            val_out = model.evaluate(X_val, y_val, verbose=0)
            # val_out is [loss, accuracy] from compile's metrics=['accuracy']
            fold_loss, fold_acc = float(val_out[0]), float(val_out[1])
            accs.append(fold_acc)
            losses.append(fold_loss)
        except Exception as e:
            logging.warning(f"walk-forward fold {fold} failed: {e}")
            continue

    if not accs:
        return "no walk-forward folds completed"

    accs_arr = np.array(accs)
    mean_acc = float(accs_arr.mean())
    std_acc = float(accs_arr.std(ddof=1)) if len(accs_arr) > 1 else 0.0
    # Trend: is the model getting worse or better in later folds? Simple
    # linear-fit slope of accuracy vs fold index.
    if len(accs_arr) >= 3:
        slope = float(np.polyfit(range(len(accs_arr)), accs_arr, 1)[0])
    else:
        slope = 0.0
    per_fold_str = ", ".join(f"fold{i+1}={a:.3f}" for i, a in enumerate(accs))
    if dated_count == len(sorted_examples):
        date_note = "calendar-time sorted"
    elif dated_count > 0:
        date_note = f"{dated_count}/{len(sorted_examples)} calendar-dated"
    else:
        date_note = "no dates (sorted by insertion order)"
    return (f"walk-forward ({len(accs)} folds, {date_note}): mean_acc={mean_acc:.3f}, "
           f"std={std_acc:.3f}, slope={slope:+.4f}/fold; per fold: {per_fold_str}")


def _ml_train_on_examples(examples, run_kind):
    """Actually runs model.fit() on a list of (sequence, label) examples.
    focal loss + per-sample loss penalty + class-balance
    weighting so a wrong forecast on a "loser" costs the model more than a
    wrong forecast on a "winner". Saves the trained model + meta.json under
    ML_BRAIN_DIR when done.
    """
    tf = _ml_lazy_import_tf()
    if tf is None:
        return "tensorflow unavailable; skipped"

    if not examples:
        return "no examples supplied; skipped"

    from tensorflow import keras

    X = np.stack([e[0] for e in examples]).astype(np.float32)
    y = np.array([e[1] for e in examples], dtype=np.float32)

    # Walk-forward diagnostic BEFORE the main fit. This trains multiple small
    # models on rolling folds just to log per-fold accuracy. It's slow (~1
    # extra minute for a 15k-example daily run) but only runs once per
    # scheduled training, and the alternative -- no walk-forward at all --
    # is what let last quarter's win-rate look great even as the model was
    # slowly forgetting how earlier market regimes behaved.
    try:
        wf_status = _ml_walk_forward_evaluation(examples, tf, n_folds=5)
        print(f"ML brain [{run_kind}] {wf_status}")
        logging.info(f"ml_brain [{run_kind}] {wf_status}")
    except Exception as e:
        logging.warning(f"walk-forward evaluation failed: {e}")

    n_pos = max(1, int(y.sum()))
    n_neg = max(1, len(y) - n_pos)
    w_pos = len(y) / (2.0 * n_pos)
    w_neg = len(y) / (2.0 * n_neg)
    sample_weight = np.where(y > 0.5, w_pos, w_neg * ML_BRAIN_LOSS_PENALTY).astype(np.float32)

    n_val = max(20, int(len(X) * 0.15))
    # Chronological ordering isn't meaningful here (examples span many
    # symbols and dates); random shuffle before split is the right choice.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y, sw = X[idx], y[idx], sample_weight[idx]
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    sw_train = sw[:-n_val]

    with _ml_lock:
        model = _ml_load_cached_model(tf) or _ml_build_model(tf)
        epochs = 6 if run_kind == 'first_run' else 12
        try:
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                batch_size=min(ML_BRAIN_BATCH_SIZE, len(X_train)),
                                sample_weight=sw_train,
                                validation_data=(X_val, y_val),
                                verbose=0, shuffle=True)
        except Exception as e:
            logging.error(f"ml_brain: model.fit failed: {e}")
            return f"model.fit failed: {e}"

        val_acc = history.history.get('val_accuracy', [None])[-1]
        val_loss = history.history.get('val_loss', [None])[-1]
        train_loss = history.history.get('loss', [None])[-1]

        os.makedirs(ML_BRAIN_DIR, exist_ok=True)
        try:
            model.save(ML_MODEL_PATH)
        except Exception as e:
            logging.error(f"ml_brain: model.save failed: {e}")
            return f"model.save failed: {e}"

        meta = {
            'trained_at': datetime.utcnow().isoformat(),
            'trained_from': f'historical_yfinance ({run_kind})',
            'n_examples': len(X),
            'n_train': int(len(X_train)),
            'n_val': int(len(X_val)),
            'val_accuracy': float(val_acc) if val_acc is not None else None,
            'val_loss': float(val_loss) if val_loss is not None else None,
            'train_loss': float(train_loss) if train_loss is not None else None,
            'n_updates': _ml_model_cache.get('n_updates', 0) + len(X),
        }
        with open(ML_META_PATH, 'w') as f:
            _ml_json.dump(meta, f, indent=2)

        _ml_model_cache['model'] = model
        _ml_model_cache['trained_at'] = meta['trained_at']
        _ml_model_cache['n_updates'] = meta['n_updates']

        return (f"[{run_kind}] trained on {len(X_train)} sequences, validated on {len(X_val)} "
                f"(val_accuracy={val_acc:.3f}, val_loss={val_loss:.3f}, "
                f"train_loss={train_loss:.3f})" if val_acc is not None else
                f"[{run_kind}] trained on {len(X_train)} sequences, validated on {len(X_val)}")


def _ml_load_cached_model(tf):
    """Load the most recently saved model from disk, if not already in the
    process cache. Returns None if nothing has been saved yet or if load
    fails (caller falls back to _ml_build_model()).
    """
    with _ml_lock:
        if _ml_model_cache['model'] is not None:
            return _ml_model_cache['model']
        if not os.path.exists(ML_MODEL_PATH):
            return None
        try:
            from tensorflow import keras
            # custom_objects lets Keras deserialize the focal loss function
            # by name when re-loading a saved model.
            model = keras.models.load_model(
                ML_MODEL_PATH,
                custom_objects={'focal_loss': _ml_make_focal_loss(ML_BRAIN_FOCAL_GAMMA)},
                compile=False,
            )
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=ML_BRAIN_LEARNING_RATE),
                          loss=_ml_make_focal_loss(ML_BRAIN_FOCAL_GAMMA),
                          metrics=['accuracy'])
            _ml_model_cache['model'] = model
            if os.path.exists(ML_META_PATH):
                with open(ML_META_PATH) as f:
                    meta = _ml_json.load(f)
                _ml_model_cache['trained_at'] = meta.get('trained_at')
                _ml_model_cache['n_updates'] = meta.get('n_updates', 0)
            return model
        except Exception as e:
            logging.warning(f"ml_brain: failed to load saved model ({e}); will rebuild.")
            return None


# =================================================================================
# Training schedule + entry points
# =================================================================================

def _ml_load_schedule_state():
    try:
        if os.path.exists(ML_STATE_PATH):
            with open(ML_STATE_PATH) as f:
                return _ml_json.load(f)
    except Exception:
        pass
    return {}


def _ml_save_schedule_state(state):
    try:
        os.makedirs(ML_BRAIN_DIR, exist_ok=True)
        with open(ML_STATE_PATH, 'w') as f:
            _ml_json.dump(state, f, indent=2)
    except Exception as e:
        logging.warning(f"ml_brain: failed to save schedule state ({e}).")


def run_ml_first_training_if_needed():
    """First-run training: fires ONCE, the first time this bot ever calls
    maybe_run_scheduled_ml_training(), if no model exists on disk yet. Uses
    ML_FIRST_RUN_EXAMPLES (2,500) -- lighter than the daily budget so the bot
    has a working model as soon as possible instead of waiting for the first
    17:00 slot. Returns a status string if it actually ran, None otherwise.
    """
    if os.path.exists(ML_MODEL_PATH):
        return None  # already have a model from a prior day

    state = _ml_load_schedule_state()
    if state.get('first_run_completed'):
        return None

    if not _ml_brain_is_available():
        return None

    print(f"ML brain: no existing model on disk. Running first-time training "
          f"({ML_FIRST_RUN_EXAMPLES} examples).")
    logging.info(f"ml_brain: first-time training starting ({ML_FIRST_RUN_EXAMPLES} examples).")

    first_budget = min(ML_FIRST_RUN_EXAMPLES, ML_PRETRAIN_LIFETIME_CAP)
    examples, symbols_used, symbols_attempted = _ml_gather_training_examples(first_budget)
    if len(examples) < 100:
        msg = (f"first-time training: only gathered {len(examples)} examples from "
              f"{symbols_used}/{symbols_attempted} symbols (need 100+); "
              f"deferring to next scheduled run.")
        logging.warning(f"ml_brain: {msg}")
        return msg

    status = _ml_train_on_examples(examples, run_kind='first_run')

    state['first_run_completed'] = True
    state['first_run_completed_at'] = datetime.now(eastern).isoformat()
    _ml_save_schedule_state(state)

    return f"first-time training ({len(examples)} examples from {symbols_used} symbols): {status}"


def _ml_in_daily_train_window(now):
    """True from 17:00 ET (today) through 07:45 ET (next day), i.e. the
    off-hours window in which the daily 15k-example run must both START and
    FINISH. The intent per instruction is 'begin at 17:00 ET, finish before
    07:45 ET' -- this returns True as long as we're inside that overnight
    window, so a bot that started up during it can still catch the run.
    """
    hour, minute = now.hour, now.minute
    # After 17:00 through end of day
    if hour >= ML_DAILY_TRAIN_HOUR:
        return True
    # Or before 07:45 the next morning
    if hour < ML_DAILY_TRAIN_MUST_FINISH_HOUR:
        return True
    if hour == ML_DAILY_TRAIN_MUST_FINISH_HOUR and minute < ML_DAILY_TRAIN_MUST_FINISH_MINUTE:
        return True
    return False


def _ml_daily_train_key_for(now):
    """
    Runs are identified by the DATE OF THE 17:00 START, not the calendar day
    the training might spill into. A run that starts at 23:00 on Tuesday and
    a run that starts at 01:00 on Wednesday morning are the SAME logical
    "Tuesday overnight" run and shouldn't both fire.
    """
    if now.hour >= ML_DAILY_TRAIN_HOUR:
        return now.date().isoformat()
    # It's between midnight and 07:45 -- attribute to yesterday's window.
    return (now.date() - timedelta(days=1)).isoformat()


def _ml_maintenance_train_on_live_trades(sess, tf_model):
    """Post-cap daily maintenance: once cumulative pretraining has hit
    ML_PRETRAIN_LIFETIME_CAP, historical pretraining stops entirely. In
    its place, each scheduled 17:00 ET window fine-tunes the model on
    the last ML_MAINTENANCE_LOOKBACK_DAYS of the bot's OWN closed live
    trades -- literal 'win/loss of the market for that 24 hour time' per
    instruction.

    Sequences are rebuilt from each trade's symbol using yfinance, ending
    at the trade's entry date, labeled by whether the trade actually won
    (outcome_pct > 0). No new example count is added to n_updates -- the
    lifetime cap is on PRETRAINING, not on maintenance -- so maintenance
    passes can keep running indefinitely without ever re-opening the
    pretraining pipeline.
    """
    if not _ml_brain_is_available():
        return "tensorflow unavailable; maintenance skipped"

    cutoff = datetime.now(eastern) - timedelta(days=ML_MAINTENANCE_LOOKBACK_DAYS)
    try:
        rows = (sess.query(tf_model)
                .filter(tf_model.outcome_pct.isnot(None))
                .filter(tf_model.buy_score.isnot(None))
                .all())
        # Filter to recent by exit_date if the field exists on the row
        recent = []
        for r in rows:
            exit_date_str = getattr(r, 'exit_date', None)
            if not exit_date_str:
                continue
            try:
                exit_dt = datetime.fromisoformat(exit_date_str).replace(tzinfo=eastern) \
                    if 'T' in exit_date_str else \
                    datetime.strptime(exit_date_str, '%Y-%m-%d').replace(tzinfo=eastern)
            except Exception:
                continue
            if exit_dt >= cutoff:
                recent.append(r)
    except Exception as e:
        return f"maintenance: live-trade query failed ({e}); skipping"

    if len(recent) < ML_MAINTENANCE_MIN_TRADES:
        return (f"maintenance: only {len(recent)} live trades closed in the "
               f"last {ML_MAINTENANCE_LOOKBACK_DAYS} day(s) (need "
               f"{ML_MAINTENANCE_MIN_TRADES}+); skipping this window.")

    # Build one training sequence per closed trade using its symbol's
    # daily history ending AT (or just before) the trade's entry date.
    # This is much cheaper than a batch pretraining pull.
    examples = []
    for r in recent:
        symbol = getattr(r, 'symbols', None) or getattr(r, 'symbol', None)
        entry_date_str = getattr(r, 'entry_date', None)
        if not symbol or not entry_date_str:
            continue
        try:
            entry_dt = datetime.fromisoformat(entry_date_str) \
                if 'T' in entry_date_str else \
                datetime.strptime(entry_date_str, '%Y-%m-%d')
        except Exception:
            continue
        try:
            start = (entry_dt - timedelta(days=90)).date().isoformat()
            end = (entry_dt + timedelta(days=1)).date().isoformat()
            df_map = yf_download_batch([symbol], start=start, end=end, interval='1d')
        except Exception:
            continue
        df = df_map.get(symbol)
        if df is None or df.empty or len(df) < ML_BRAIN_SEQ_LEN + 5:
            continue
        seq = _ml_current_feature_row_for_symbol(symbol, df, None, None,
                                                  None, None, None, None)
        if seq is None:
            continue
        label = 1.0 if float(r.outcome_pct) > 0 else 0.0
        examples.append((seq, label))

    if len(examples) < ML_MAINTENANCE_MIN_TRADES:
        return (f"maintenance: could only rebuild {len(examples)} usable sequences "
               f"from {len(recent)} closed trades; skipping.")

    return _ml_train_on_examples(examples, run_kind='maintenance')


def maybe_run_scheduled_ml_training():
    """Called every cycle from the main loop. Runs the first-time bootstrap
    if a model doesn't exist yet, then otherwise checks whether we're inside
    a daily-training window that hasn't fired yet. Returns a status string
    when it actually did something, None otherwise (so the caller only logs
    on real activity).

    Cheap to call every cycle: file-existence check + a small JSON read.
    """
    if not _ml_brain_is_available():
        return None

    # First-time bootstrap has priority. If a model already exists on disk,
    # this returns None immediately.
    first = run_ml_first_training_if_needed()
    if first is not None:
        return first

    now = datetime.now(eastern)
    if not _ml_in_daily_train_window(now):
        return None

    state = _ml_load_schedule_state()
    last_daily_key = state.get('last_daily_run_window_key')
    todays_key = _ml_daily_train_key_for(now)
    if last_daily_key == todays_key:
        return None  # already fired for this window

    # Claim the slot BEFORE the slow download starts, so a slow run doesn't
    # cause a second thread on the next tick to see 'not yet fired' and
    # start a duplicate training pass.
    state['last_daily_run_window_key'] = todays_key
    state['last_daily_run_started_at'] = now.isoformat()
    _ml_save_schedule_state(state)

    # Lifetime pretraining cap check (per instruction: no brain model
    # should train more than 20,000 times for pre-training, then just
    # daily maintenance training on the win/loss of the market for that
    # 24-hour time). n_updates in meta.json is the cumulative count of
    # PRETRAINING examples across every historical run this brain has
    # ever seen; once it clears ML_PRETRAIN_LIFETIME_CAP, we permanently
    # switch to daily maintenance on live win/loss outcomes only.
    lifetime_n = _ml_model_cache.get('n_updates', 0)
    if lifetime_n == 0 and os.path.exists(ML_META_PATH):
        try:
            with open(ML_META_PATH) as f:
                lifetime_n = int(_ml_json.load(f).get('n_updates', 0))
        except Exception:
            lifetime_n = 0

    if lifetime_n >= ML_PRETRAIN_LIFETIME_CAP:
        print(f"ML brain: cumulative pretraining n_updates={lifetime_n} has reached "
              f"the lifetime cap of {ML_PRETRAIN_LIFETIME_CAP}. Switching to daily "
              f"MAINTENANCE training on live win/loss only.")
        logging.info(f"ml_brain: maintenance mode (lifetime cap {ML_PRETRAIN_LIFETIME_CAP} reached).")
        status = _ml_maintenance_train_on_live_trades(session, TradeFeatures)
        state['last_daily_run_completed_at'] = datetime.now(eastern).isoformat()
        _ml_save_schedule_state(state)
        return status

    # Cap the request size so a single run can't blow past the lifetime
    # limit -- if we're 3,000 away from the cap and someone set the
    # daily budget to 15,000, only pull 3,000 this time.
    daily_budget = min(ML_DAILY_RUN_EXAMPLES, ML_PRETRAIN_LIFETIME_CAP - lifetime_n)
    print(f"ML brain: daily training window (17:00-07:45 ET), starting "
          f"{daily_budget}-example run (lifetime n_updates so far: {lifetime_n}).")
    logging.info(f"ml_brain: daily training starting ({daily_budget} examples, "
                f"lifetime n_updates={lifetime_n}).")

    examples, symbols_used, symbols_attempted = _ml_gather_training_examples(daily_budget)
    if len(examples) < 500:
        msg = (f"daily training: only gathered {len(examples)} examples from "
              f"{symbols_used}/{symbols_attempted} symbols (need 500+); skipping this window.")
        logging.warning(f"ml_brain: {msg}")
        return msg

    status = _ml_train_on_examples(examples, run_kind='daily')

    state['last_daily_run_completed_at'] = datetime.now(eastern).isoformat()
    _ml_save_schedule_state(state)

    return f"daily training ({len(examples)} examples from {symbols_used} symbols): {status}"


# =================================================================================
# Live-inference adjustment (called from buy_stocks)
# =================================================================================

def _ml_current_feature_row_for_symbol(symbol, df, rsi_val, macd_val,
                                        macd_sig_val, atr_val, volume_val,
                                        current_price):
    """Build a live feature row from the latest bar of `df` (which
    compute_buy_score already has in hand). Used by get_ml_adjustment() to
    turn the current market state into a sequence input.

    Returns None if there's not enough recent data to build a full ML_BRAIN_SEQ_LEN
    window -- caller should fall back to no adjustment.
    """
    if df is None or df.empty or len(df) < ML_BRAIN_SEQ_LEN + 5:
        return None

    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    volume = df['Volume'].values.astype(np.float64) if 'Volume' in df else np.zeros(len(close))

    try:
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        atr_series = talib.ATR(high, low, close, timeperiod=14)
        vol_sma = talib.SMA(volume, timeperiod=14) if volume.sum() > 0 else np.full(len(close), 1.0)
        sma20 = talib.SMA(close, timeperiod=20)
        sma50 = talib.SMA(close, timeperiod=50)
    except Exception:
        return None

    daily_return = np.diff(close, prepend=close[0]) / np.maximum(close, 0.01)

    n_needed = ML_BRAIN_SEQ_LEN
    start = len(close) - n_needed
    if start < 50:
        return None

    seq = np.zeros((n_needed, ML_BRAIN_FEATURES), dtype=np.float32)
    for j, i in enumerate(range(start, start + n_needed)):
        if np.isnan(rsi[i]) or np.isnan(macd[i]) or np.isnan(atr_series[i]):
            return None
        seq[j] = _ml_build_feature_row(
            rsi[i], macd[i], macd_signal[i], atr_series[i], close[i],
            volume[i], vol_sma[i], sma20[i], sma50[i], daily_return[i])
    return seq


# ---------------- ML brain "thinking" explainer ----------------
# On the first buy cycle after startup, print WHY the brain scored each
# candidate the way it did: raw win-probability, confidence, and the top
# feature contributions computed via gradient x input attribution on the
# last timestep. This is not chain-of-thought (a neural net doesn't have
# any); it's the closest fast, honest analog -- which of the 10 inputs
# most swung the output. After the first cycle the flag flips off so
# steady-state logs stay clean.
_ML_FEATURE_LABELS = [
    'RSI/100 (lower=more oversold)',
    'MACD raw',
    'MACD signal raw',
    'MACD > signal (bullish x-over)',
    'ATR% (volatility)',
    'day return',
    'volume vs SMA (excess)',
    'close vs SMA20 (short trend)',
    'close vs SMA50 (medium trend)',
    'SMA20 > SMA50 (uptrend)',
]

_ml_first_cycle_thinking_printed = False


def _ml_explain_prediction(model, seq, tf):
    """Gradient x input attribution on the LAST timestep of the sequence.
    Returns (prob, [(feature_label, value, contribution), ...] sorted by
    |contribution| descending). Cheap: one backward pass per call.

    Attribution isn't a proof of reasoning -- it's a first-order tangent.
    But for a 10-feature dip-buy model it lines up well with what the frozen
    foundation rules the head sits on top of would flag as decisive.
    """
    try:
        x = tf.constant(seq.reshape((1, ML_BRAIN_SEQ_LEN, ML_BRAIN_FEATURES)))
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, training=False)
        grads = tape.gradient(y, x).numpy()[0]   # (SEQ_LEN, FEATURES)
        vals = seq                                # (SEQ_LEN, FEATURES)
        # Take the last timestep -- that's "right now" from the brain's POV.
        last_grad = grads[-1]
        last_vals = vals[-1]
        contribs = last_grad * last_vals
        prob = float(y.numpy()[0, 0])
        ranked = sorted(
            [(_ML_FEATURE_LABELS[i], float(last_vals[i]), float(contribs[i]))
             for i in range(ML_BRAIN_FEATURES)],
            key=lambda t: abs(t[2]), reverse=True)
        return prob, ranked
    except Exception as e:
        logging.warning(f"ml_brain: explainer failed ({e}); no thinking output.")
        return None, None


def _ml_print_thinking(symbol, prob, ranked, adjustment, side='BUY'):
    if side == 'BUY':
        verdict = ('LEAN BUY' if prob >= ML_BRAIN_WIN_THRESHOLD
                   else 'LEAN AVOID' if prob <= (1.0 - ML_BRAIN_WIN_THRESHOLD)
                   else 'NEUTRAL')
    else:  # SELL side: high P(win) = brain expects further upside = LEAN HOLD
        verdict = ('LEAN HOLD' if prob >= ML_BRAIN_WIN_THRESHOLD
                   else 'LEAN SELL' if prob <= (1.0 - ML_BRAIN_WIN_THRESHOLD)
                   else 'NEUTRAL')
    print(f"  [brain thinking {side}] {symbol}: P(win)={prob:.3f} -> {verdict} "
          f"(score adj {adjustment:+.2f})")
    for label, val, contrib in ranked[:5]:
        arrow = '+' if contrib >= 0 else '-'
        print(f"      {arrow} {label:38s} val={val:+.3f}  contrib={contrib:+.4f}")
    # Mirror to dashboard thinking log if the engine is loaded.
    try:
        dashboard_record_thinking(side, symbol, prob, verdict, adjustment)
    except NameError:
        pass  # dashboard defined later in the file; NameError on the very first
              # call before its section loaded — safe to skip.


def explain_sell_decision(symbol, df, current_price):
    """Public helper for sell_stocks: builds the brain's current feature
    sequence for `symbol`, runs inference + attribution, and prints the
    reasoning trail. Returns P(win) so the caller can log it alongside
    whatever rule-based sell logic it uses. Returns None if the model or
    the input isn't ready -- caller falls back to rules only.
    """
    if not _ml_brain_is_available():
        return None
    tf = _ml_lazy_import_tf()
    if tf is None:
        return None
    model = _ml_load_cached_model(tf)
    if model is None:
        return None
    seq = _ml_current_feature_row_for_symbol(symbol, df, None, None, None,
                                              None, None, current_price)
    if seq is None:
        return None
    try:
        prob, ranked = _ml_explain_prediction(model, seq, tf)
        if ranked is not None:
            # Sell-side "adjustment" is diagnostic-only: shows how much the
            # brain's view of forward P(win) leans away from neutral. The
            # actual sell decision stays with the rule-based sell_stocks
            # logic (trailing stop, hard stop, timeout, take-profit).
            pseudo_adj = round((prob - 0.5) * 2 * ML_MAX_SCORE_ADJUSTMENT, 3)
            _ml_print_thinking(symbol, prob, ranked, pseudo_adj, side='SELL')
        return prob
    except Exception as e:
        logging.debug(f"ml_brain: sell-side thinking skipped for {symbol} ({e}).")
        return None


def get_ml_adjustment(sess, tf_model, buy_score=None, rsi=None, atr_pct=None,
                      macd_above_signal=None, volume_holding=None, regime=None,
                      symbol=None, df=None, current_price=None):
    """Returns a small buy-score adjustment (float, can be negative) or None
    if the model isn't trained/eligible yet. Callers must treat None as "no
    opinion" and fall back entirely to the rule-based score.

    Signature preserved from the previous ML wiring so buy_stocks doesn't
    need changing. The new sequence-based model wants a rolling window of
    daily bars, so it looks at `df` (which buy_stocks already fetches via
    get_cached_data + yf_history for compute_buy_score) to build the sequence.
    If `df` isn't supplied (older callers) or is too short, returns None.
    """
    if not _ml_brain_is_available():
        return None

    # LIVE-trust gate: even a well-trained historical model shouldn't touch
    # LIVE decisions until this bot has enough of its own outcomes on record
    # for the operator to have some basis for trusting it. Same philosophy
    # AdaptiveParams uses elsewhere in this file.
    try:
        live_rows = (sess.query(tf_model)
                     .filter(tf_model.outcome_pct.isnot(None))
                     .filter(tf_model.buy_score.isnot(None))
                     .all())
        if len(live_rows) < ML_MIN_TRADES_FOR_LIVE_ADJUSTMENT:
            return None
    except Exception as e:
        logging.warning(f"ml_brain: live-trades gate query failed ({e}); no adjustment.")
        return None

    tf = _ml_lazy_import_tf()
    if tf is None:
        return None

    model = _ml_load_cached_model(tf)
    if model is None:
        return None

    if df is None:
        return None
    seq = _ml_current_feature_row_for_symbol(symbol, df, rsi, None, None,
                                              atr_pct, None, current_price)
    if seq is None:
        return None

    try:
        x = seq.reshape((1, ML_BRAIN_SEQ_LEN, ML_BRAIN_FEATURES))
        prob = float(model(x, training=False).numpy()[0, 0])
    except Exception as e:
        logging.warning(f"ml_brain: inference failed ({e}); returning no adjustment.")
        return None

    adjustment = round((prob - 0.5) * 2 * ML_MAX_SCORE_ADJUSTMENT, 3)

    # Print the brain's reasoning trail every cycle so the operator can see
    # WHAT the model is keying on for each candidate. Gradient x input
    # attribution is cheap (one backward pass per symbol).
    try:
        expl_prob, ranked = _ml_explain_prediction(model, seq, tf)
        if ranked is not None and symbol:
            _ml_print_thinking(symbol, expl_prob, ranked, adjustment, side='BUY')
    except Exception as e:
        logging.debug(f"ml_brain: thinking print skipped ({e}).")

    return adjustment


def _ml_mark_first_cycle_thinking_done():
    """Kept as a no-op for callers that still invoke it. Thinking prints
    now fire every cycle."""
    return


# ============================================================================
# BACKTEST BRAIN (Brain C)
# ============================================================================
# Pre-trade check: for each buy candidate, walks the symbol's 5-year daily
# history, replays a simplified dip-buy signal on each bar, simulates the
# outcome under the bot's actual exit rules (via _ml_simulate_exit_label),
# and vetoes the live buy if the historical win rate is below threshold.
#
# HONESTY NOTES:
#   1. The signal replayed on historical bars is a DAILY PROXY of the live
#      buy criteria (RSI oversold, MACD bull, above 200-SMA). The live
#      compute_buy_score also uses intraday inputs (5m/30m pullbacks, VWAP
#      distance) that daily bars can't reproduce. The proxy captures the
#      core setup shape but is not tick-accurate.
#   2. Regime-weighting: signals are tagged with the SPY+VIX regime that
#      was live on that historical date. When a current-regime win rate
#      exists (>= REGIME_MIN_SAMPLES within the current regime), it is
#      used as the primary metric; otherwise falls back to the overall
#      win rate.
#   3. Cached per-symbol for 4 hours: the historical bars don't change
#      during a session, so the backtest result is stable within a day.
#      Cache TTL exists only to pick up new bars after a session boundary.
#   4. Armed by default. Vetoes any buy where the historical win rate is
#      below BACKTEST_MIN_WIN_RATE (0.50 = coin flip). Abstains (does not
#      block) when the symbol has fewer than BACKTEST_MIN_SAMPLES (20)
#      historical signal fires -- statistics on a handful of trades are
#      noise, not evidence.

BACKTEST_BRAIN_MODE = 'armed'           # 'armed' (vetos) or 'advisory' (prints only)
BACKTEST_MIN_WIN_RATE = 0.50            # veto if historical win rate below this
BACKTEST_MIN_SAMPLES = 20               # abstain (no veto) if fewer historical fires than this
BACKTEST_REGIME_MIN_SAMPLES = 10        # use regime-specific win rate if this many fires in-regime
BACKTEST_LOOKBACK_DAYS = 365 * 5        # 5 years of history per symbol
BACKTEST_CACHE_TTL_SECONDS = 4 * 3600   # 4 hours
BACKTEST_MAX_FORWARD_DAYS = 20          # how far forward each simulated trade runs

_backtest_cache = {}                    # {symbol: (fetched_at_epoch, result_dict)}
_backtest_cache_lock = threading.Lock()


def _backtest_daily_signal_fires(rsi_i, macd_i, macd_sig_i, close_i, sma200_i):
    """Simplified daily-bar buy signal for the backtest brain. Fires when
    all three classic dip-buy conditions hold on the daily bar. This is a
    PROXY of live compute_buy_score -- it can't include intraday pullback
    inputs. Documented approximation.
    """
    if np.isnan(rsi_i) or np.isnan(macd_i) or np.isnan(macd_sig_i) or np.isnan(sma200_i):
        return False
    if rsi_i >= 40:              # RSI oversold-ish
        return False
    if macd_i <= macd_sig_i:     # MACD bullish crossover / above signal
        return False
    if close_i < sma200_i:       # above 200-day SMA (regime filter)
        return False
    return True


def _backtest_classify_regime_for_bar(spy_close_i, spy_sma20_i, spy_sma50_i,
                                       vix_close_i):
    """Approximate SPY+VIX regime classification for a historical bar. Mirrors
    the live _fetch_market_regime logic so historical regime labels line up
    with what get_market_regime returns today. Returns one of the REGIME_*
    constants, or 'unknown' if inputs are NaN.
    """
    if np.isnan(spy_close_i) or np.isnan(spy_sma20_i) or np.isnan(spy_sma50_i):
        return 'unknown'
    vix = vix_close_i if not np.isnan(vix_close_i) else None
    if vix is not None and vix >= VIX_PANIC_LEVEL:
        return REGIME_PANIC
    if spy_close_i > spy_sma20_i > spy_sma50_i:
        regime = REGIME_BULL
    elif spy_close_i < spy_sma20_i < spy_sma50_i:
        regime = REGIME_BEAR
    else:
        regime = REGIME_SIDEWAYS
    if vix is not None and vix >= VIX_ELEVATED_LEVEL and regime != REGIME_BULL:
        regime = REGIME_BEAR
    return regime


def _backtest_load_regime_series():
    """Load SPY + VIX daily history for the backtest lookback window and
    return aligned arrays for regime classification. Cached on the module
    for the duration of a session (SPY/VIX history doesn't change; only
    growing at the tail).
    """
    if hasattr(_backtest_load_regime_series, '_cache'):
        cached_at, data = _backtest_load_regime_series._cache
        if time.time() - cached_at < BACKTEST_CACHE_TTL_SECONDS:
            return data
    try:
        spy = yf_history('SPY', period=f'{int(BACKTEST_LOOKBACK_DAYS/365) + 1}y',
                         interval='1d')
        vix = yf_history('^VIX', period=f'{int(BACKTEST_LOOKBACK_DAYS/365) + 1}y',
                         interval='1d')
    except Exception as e:
        logging.warning(f"backtest brain: SPY/VIX regime series fetch failed ({e})")
        return None
    if spy is None or spy.empty or vix is None or vix.empty:
        return None
    try:
        spy_close = spy['Close'].values.astype(np.float64)
        spy_sma20 = talib.SMA(spy_close, timeperiod=20)
        spy_sma50 = talib.SMA(spy_close, timeperiod=50)
        # Align VIX to SPY dates: reindex on SPY's index. Missing VIX days
        # forward-fill (VIX is very rarely missing on a trading day).
        vix_aligned = vix['Close'].reindex(spy.index).ffill()
        vix_close = vix_aligned.values.astype(np.float64)
        data = {
            'dates': spy.index,
            'spy_close': spy_close,
            'spy_sma20': spy_sma20,
            'spy_sma50': spy_sma50,
            'vix_close': vix_close,
        }
        _backtest_load_regime_series._cache = (time.time(), data)
        return data
    except Exception as e:
        logging.warning(f"backtest brain: SPY/VIX preprocessing failed ({e})")
        return None


def _backtest_lookup_regime_for_date(regime_series, target_date):
    """Given the SPY/VIX regime series and a pandas Timestamp / date, return
    the regime label for that trading day. Uses searchsorted on the SPY date
    index so missing dates (weekends/holidays) fall back to the most recent
    prior trading day.
    """
    if regime_series is None:
        return 'unknown'
    try:
        dates = regime_series['dates']
        idx = dates.searchsorted(target_date, side='right') - 1
        if idx < 0 or idx >= len(dates):
            return 'unknown'
        return _backtest_classify_regime_for_bar(
            regime_series['spy_close'][idx],
            regime_series['spy_sma20'][idx],
            regime_series['spy_sma50'][idx],
            regime_series['vix_close'][idx])
    except Exception:
        return 'unknown'


def _backtest_run_for_symbol(symbol):
    """Walk this symbol's daily history, find every bar where the proxy
    dip-buy signal fires, simulate the exit under the bot's actual exit
    rules, and tally win rate + regime breakdown.

    Returns dict:
        {
            'symbol': str, 'n_signals': int,
            'overall_win_rate': float,
            'overall_avg_return': float,
            'by_regime': {regime_name: {'n': int, 'win_rate': float, 'avg_return': float}},
            'first_bar': date, 'last_bar': date,
        }
    or None on data / calculation failure.
    """
    try:
        df = yf_history(symbol,
                        period=f'{int(BACKTEST_LOOKBACK_DAYS/365) + 1}y',
                        interval='1d')
    except Exception as e:
        logging.debug(f"backtest brain: {symbol} history fetch failed ({e})")
        return None
    if df is None or df.empty or len(df) < 260:
        return None

    df = df.dropna(subset=['Close', 'High', 'Low'])
    if len(df) < 260:
        return None

    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)

    try:
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        atr_series = talib.ATR(high, low, close, timeperiod=14)
        sma200 = talib.SMA(close, timeperiod=200)
    except Exception as e:
        logging.debug(f"backtest brain: {symbol} indicator calc failed ({e})")
        return None

    regime_series = _backtest_load_regime_series()

    n_days = len(close)
    first_valid = 200                                # need SMA200
    last_valid = n_days - BACKTEST_MAX_FORWARD_DAYS - 1
    if last_valid <= first_valid:
        return None

    outcomes = []           # (return_pct, regime_label)
    for i in range(first_valid, last_valid + 1):
        if not _backtest_daily_signal_fires(rsi[i], macd[i], macd_signal[i],
                                             close[i], sma200[i]):
            continue
        entry_price = close[i]
        entry_atr = atr_series[i]
        if np.isnan(entry_atr) or entry_atr <= 0:
            continue

        # Reuse the exact exit simulator Brain A uses for its training labels,
        # but capture the realized return (not just the win/loss label) so we
        # can report avg return alongside win rate. We call the simulator to
        # get the win/loss bit, and separately compute realized return by
        # rolling the same simulator inline (cheap; same forward window).
        forward_close = close[i + 1 : i + 1 + BACKTEST_MAX_FORWARD_DAYS]
        forward_high = high[i + 1 : i + 1 + BACKTEST_MAX_FORWARD_DAYS]
        forward_low = low[i + 1 : i + 1 + BACKTEST_MAX_FORWARD_DAYS]

        # Fast inline realized-return simulation. Mirrors _ml_simulate_exit_label
        # but returns the numeric return instead of a 0/1 label.
        realized = _backtest_realized_return(entry_price, entry_atr,
                                              forward_close, forward_high,
                                              forward_low)
        if realized is None:
            continue

        bar_date = df.index[i]
        regime_label = _backtest_lookup_regime_for_date(regime_series, bar_date)
        outcomes.append((realized, regime_label))

    if not outcomes:
        return {
            'symbol': symbol, 'n_signals': 0,
            'overall_win_rate': 0.0, 'overall_avg_return': 0.0,
            'by_regime': {},
            'first_bar': df.index[0], 'last_bar': df.index[-1],
        }

    returns = np.array([r for r, _ in outcomes])
    wins = (returns > 0).sum()
    result = {
        'symbol': symbol,
        'n_signals': len(outcomes),
        'overall_win_rate': float(wins / len(outcomes)),
        'overall_avg_return': float(returns.mean()),
        'by_regime': {},
        'first_bar': df.index[0],
        'last_bar': df.index[-1],
    }
    for regime_name in (REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR, REGIME_PANIC, 'unknown'):
        subset = [r for r, reg in outcomes if reg == regime_name]
        if subset:
            arr = np.array(subset)
            result['by_regime'][regime_name] = {
                'n': int(len(arr)),
                'win_rate': float((arr > 0).sum() / len(arr)),
                'avg_return': float(arr.mean()),
            }
    return result


def _backtest_realized_return(entry_price, entry_atr, forward_close,
                                forward_high, forward_low):
    """Same exit machinery as _ml_simulate_exit_label but returns the
    realized return (float) instead of a 0/1 win label. Kept separate so
    the label function's contract (returns 0.0/1.0) doesn't drift.
    """
    if entry_price <= 0 or entry_atr is None or entry_atr <= 0:
        return None
    if len(forward_close) == 0:
        return None
    stop_distance_pct = max(HARD_STOP_ATR_MULTIPLIER * (entry_atr / entry_price),
                            HARD_STOP_MIN_PCT)
    atr_pct = entry_atr / entry_price
    arm_pct = max(ARM_PROFIT_PCT, ATR_ARM_MULTIPLIER * atr_pct)

    remaining_frac = 1.0
    realized = 0.0
    fired_stages = set()
    armed = False
    peak = entry_price
    floor_pct = HARD_FLOOR_PCT

    for k in range(len(forward_close)):
        day_low = forward_low[k]
        day_high = forward_high[k]
        day_close = forward_close[k]
        gain_low = (day_low - entry_price) / entry_price
        gain_high = (day_high - entry_price) / entry_price

        if gain_low <= -stop_distance_pct:
            realized += remaining_frac * (-stop_distance_pct)
            return realized

        for idx, (trigger_pct, tranche_frac) in enumerate(SCALE_OUT_STAGES):
            if idx in fired_stages:
                continue
            if gain_high >= trigger_pct:
                tranche_realized = tranche_frac * ((day_close - entry_price) / entry_price)
                realized += tranche_realized
                remaining_frac -= tranche_frac
                fired_stages.add(idx)
                floor_pct = max(floor_pct, 0.0005)

        if gain_high >= arm_pct:
            armed = True
        peak = max(peak, day_high)
        if armed:
            peak_gain = (peak - entry_price) / entry_price
            giveback = (peak - day_close) / peak if peak > 0 else 0
            if giveback >= PEAK_GIVEBACK_FRACTION * peak_gain and remaining_frac > 0:
                realized += remaining_frac * ((day_close - entry_price) / entry_price)
                return realized

    # Timed out: exit at final close.
    if remaining_frac > 0:
        final_close = forward_close[-1]
        realized += remaining_frac * ((final_close - entry_price) / entry_price)
    return realized


def get_backtest_verdict(symbol, current_regime=None):
    """Public entry point used by buy_stocks. Returns:
        (verdict, result_dict)
    verdict is one of: 'PASS', 'VETO', 'ABSTAIN', 'ERROR'.
    result_dict has the raw numbers for printing (or None on ERROR).

    Cached per-symbol for BACKTEST_CACHE_TTL_SECONDS. Thread-safe.
    """
    now = time.time()
    with _backtest_cache_lock:
        cached = _backtest_cache.get(symbol)
        if cached and now - cached[0] < BACKTEST_CACHE_TTL_SECONDS:
            result = cached[1]
        else:
            result = None

    if result is None:
        try:
            result = _backtest_run_for_symbol(symbol)
        except Exception as e:
            logging.debug(f"backtest brain: {symbol} simulation failed ({e})")
            return 'ERROR', None
        if result is None:
            return 'ERROR', None
        with _backtest_cache_lock:
            _backtest_cache[symbol] = (now, result)

    if result['n_signals'] < BACKTEST_MIN_SAMPLES:
        return 'ABSTAIN', result

    # Prefer regime-specific win rate if we have enough same-regime signals.
    win_rate = result['overall_win_rate']
    used = 'overall'
    if current_regime and current_regime in result['by_regime']:
        rg = result['by_regime'][current_regime]
        if rg['n'] >= BACKTEST_REGIME_MIN_SAMPLES:
            win_rate = rg['win_rate']
            used = f"regime={current_regime}"
    result['decision_win_rate'] = win_rate
    result['decision_basis'] = used

    if win_rate < BACKTEST_MIN_WIN_RATE:
        return 'VETO', result
    return 'PASS', result


def print_backtest_verdict(symbol, verdict, result):
    """Formats and prints the backtest brain's decision for one candidate.
    Every buy candidate gets a line -- vetos, abstains, and passes -- so
    the operator sees the full reasoning trail every cycle."""
    if verdict == 'ERROR':
        print(f"  [backtest brain] {symbol}: data/simulation error -> ABSTAIN (no veto)")
        return
    if verdict == 'ABSTAIN':
        print(f"  [backtest brain] {symbol}: {result['n_signals']} historical signals "
              f"(need {BACKTEST_MIN_SAMPLES}+) -> ABSTAIN (no veto)")
        return
    win_pct = result['decision_win_rate'] * 100
    avg = result['overall_avg_return'] * 100
    n = result['n_signals']
    basis = result['decision_basis']
    regime_bits = []
    for r_name in (REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR, REGIME_PANIC):
        if r_name in result['by_regime']:
            rg = result['by_regime'][r_name]
            regime_bits.append(f"{r_name}={rg['n']}/{rg['win_rate']*100:.0f}%")
    regime_summary = ' | '.join(regime_bits) if regime_bits else 'none'
    verdict_str = f"{GREEN}PASS{RESET}" if verdict == 'PASS' else f"{RED}VETO{RESET}"
    mode_note = '' if BACKTEST_BRAIN_MODE == 'armed' else f' (mode={BACKTEST_BRAIN_MODE}, veto disabled)'
    print(f"  [backtest brain] {symbol}: {n} sigs -> {win_pct:.0f}% win "
          f"({basis}), avg {avg:+.2f}% -> {verdict_str}{mode_note}")
    print(f"      regime breakdown: {regime_summary}")


# ============================================================================
# BLACKLIST MANAGER — 72h temporary + permanent
# ============================================================================
# Two-tier blacklist system that hard-blocks buys for listed symbols.
# Different semantic from the soft PerSymbolPerformance mute (which is a
# statistical auto-gate on losing streaks) — the blacklist is explicit,
# operator-controlled, and persists to disk.
#
#   TEMPORARY (72h auto): a symbol is added the moment it closes a losing
#   trade; the entry expires 72 hours later and the symbol becomes tradable
#   again automatically. The dashboard can add/remove temp entries too.
#
#   PERMANENT: symbol stays blocked until explicitly removed via the
#   dashboard. Use for tickers you never want the bot to touch (planned
#   delisting, thin liquidity, personal aversion).
#
# Positions currently held are NOT force-sold when a symbol is blacklisted;
# only NEW buys are blocked. This preserves operator control over exits.

BLACKLIST_FILE = os.path.join(_ml_base_dir, 'blacklist.json')
BLACKLIST_TEMP_DURATION_SECONDS = 72 * 3600   # 72 hours


class BlacklistManager:
    """Two-tier symbol blacklist, persisted to disk. Thread-safe."""
    def __init__(self):
        self._lock = threading.Lock()
        self.permanent = set()
        self.temporary = {}   # {symbol: expiry_epoch}
        self._load()

    def _load(self):
        try:
            if os.path.exists(BLACKLIST_FILE):
                with open(BLACKLIST_FILE) as f:
                    data = _ml_json.load(f) or {}
                self.permanent = set((data.get('permanent') or []))
                self.temporary = {k: float(v) for k, v in (data.get('temporary') or {}).items()}
                # Prune already-expired temp entries on load
                now = time.time()
                self.temporary = {k: v for k, v in self.temporary.items() if v > now}
        except Exception as e:
            logging.warning(f"BlacklistManager: load failed ({e})")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(BLACKLIST_FILE), exist_ok=True)
            with open(BLACKLIST_FILE, 'w') as f:
                _ml_json.dump({
                    'permanent': sorted(self.permanent),
                    'temporary': self.temporary,
                }, f, indent=2)
        except Exception as e:
            logging.warning(f"BlacklistManager: save failed ({e})")

    def is_blocked(self, symbol):
        sym = (symbol or '').upper()
        if not sym:
            return False, ''
        with self._lock:
            if sym in self.permanent:
                return True, 'permanent'
            expiry = self.temporary.get(sym)
            if expiry is not None:
                if time.time() >= expiry:
                    self.temporary.pop(sym, None)
                    self._save()
                else:
                    remaining_hrs = (expiry - time.time()) / 3600
                    return True, f'72h auto ({remaining_hrs:.1f}h left)'
            return False, ''

    def add_temporary(self, symbol, duration_secs=None):
        sym = (symbol or '').upper()
        if not sym:
            return
        dur = duration_secs if duration_secs else BLACKLIST_TEMP_DURATION_SECONDS
        with self._lock:
            self.temporary[sym] = time.time() + dur
            self._save()
        logging.info(f"BlacklistManager: {sym} added to 72h auto-blacklist "
                     f"(expires in {dur/3600:.1f}h)")

    def add_permanent(self, symbol):
        sym = (symbol or '').upper()
        if not sym:
            return
        with self._lock:
            self.permanent.add(sym)
            # Remove from temp if present (permanent supersedes)
            self.temporary.pop(sym, None)
            self._save()
        logging.info(f"BlacklistManager: {sym} added to permanent blacklist")

    def remove(self, symbol):
        """Removes from BOTH lists (permanent + temporary)."""
        sym = (symbol or '').upper()
        if not sym:
            return
        with self._lock:
            self.permanent.discard(sym)
            self.temporary.pop(sym, None)
            self._save()
        logging.info(f"BlacklistManager: {sym} removed from blacklist")

    def snapshot(self):
        """Dashboard payload. Prunes expired temporary entries as it reads."""
        now = time.time()
        with self._lock:
            # Auto-prune expired temp entries
            expired = [s for s, e in self.temporary.items() if e <= now]
            for s in expired:
                self.temporary.pop(s, None)
            if expired:
                self._save()
            return {
                'permanent': sorted(self.permanent),
                'temporary': [
                    {'symbol': s, 'expiry_epoch': e,
                     'hours_remaining': (e - now) / 3600}
                    for s, e in sorted(self.temporary.items())
                ],
            }


BLACKLIST = BlacklistManager()


# ============================================================================
# BRAIN TRADING FLOOR — inter-brain message feed + operator terminal
# ============================================================================
# Ring buffer of typed messages the brains and safety layer publish to a
# shared "trading floor" feed. Mirrors Kalshi's brainFeed pattern: every
# significant event (buy decision, veto, chandelier fire, governor action,
# regime change) posts a message with {ts, from, type, detail}. The
# dashboard renders these as a live scrolling log the operator can watch.
#
# The operator can also send messages back via `trading_floor_msg` from the
# dashboard (e.g. `status`, `blacklist AAPL 72h`, `pause`, etc.) which are
# executed by _handle_floor_command in the dashboard engine.

FLOOR_MAX_MESSAGES = 500

_floor_lock = threading.Lock()
_floor_feed = deque(maxlen=FLOOR_MAX_MESSAGES)
_floor_counts = {'instructions': 0, 'lessons': 0, 'observations': 0, 'training': 0}


def floor_post(from_name, msg_type, detail):
    """Publish a message to the Brain Trading Floor. Non-blocking, safe from
    any thread. `msg_type` is one of: instruction, lesson, observation,
    training, alert, discuss, block, approve, learn, info.
    """
    ts = time.time()
    ts_str = time.strftime('%H:%M:%S', time.localtime(ts))
    with _floor_lock:
        _floor_feed.append({
            'ts': ts, 'ts_str': ts_str,
            'from': from_name or 'BOT',
            'type': msg_type or 'info',
            'detail': str(detail)[:500],
        })
        # Update counters (Kalshi mirrors these on the floor bar)
        counter_key = None
        if msg_type in ('instruction', 'buy', 'sell'):
            counter_key = 'instructions'
        elif msg_type in ('lesson', 'learn'):
            counter_key = 'lessons'
        elif msg_type in ('observation', 'observe'):
            counter_key = 'observations'
        elif msg_type == 'training':
            counter_key = 'training'
        if counter_key:
            _floor_counts[counter_key] += 1


def floor_snapshot(max_msgs=100):
    with _floor_lock:
        # Newest last for the dashboard's prepend behavior
        msgs = list(_floor_feed)[-max_msgs:]
        counts = dict(_floor_counts)
    return {'feed': msgs, 'counts': counts}


def floor_clear():
    with _floor_lock:
        _floor_feed.clear()


# ============================================================================
# PER-SYMBOL PERFORMANCE (ported from Kalshi bot's PerCoinPerformance)
# ============================================================================
# Tracks realized win-rate and net PnL per symbol from THIS bot's actual
# closed trades, and auto-mutes symbols that lose money after enough tries.
# Muting is soft (data-driven, checked as a gate at buy time) and separate
# from any hard blacklist. Data survives restart via a small JSON file.

PERSYMBOL_STATS_FILE = os.path.join(_ml_base_dir, 'persymbol_stats.json')
PERSYMBOL_MIN_TRADES = 5              # need at least this many closed trades before muting is possible
PERSYMBOL_MUTE_WINRATE = 0.30         # mute if win rate below this AND net PnL negative
PERSYMBOL_MUTE_DURATION_SECONDS = 24 * 3600   # muted symbols get re-tried after this


class PerSymbolPerformance:
    """Live per-symbol win-rate + PnL tracker. Feeds a soft "muted" gate that
    buy_stocks checks before submitting an order. A symbol is muted when it
    has ≥ PERSYMBOL_MIN_TRADES closed trades AND net PnL is negative AND win
    rate is below PERSYMBOL_MUTE_WINRATE. Muted symbols get automatically
    re-tried after PERSYMBOL_MUTE_DURATION_SECONDS so a bad week doesn't
    permanently exile them.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.stats = self._load()
        # {symbol: mute_until_epoch}. Populated when a symbol trips the
        # mute condition; cleared when the timestamp passes.
        self.mute_until = {}

    def _load(self):
        try:
            if os.path.exists(PERSYMBOL_STATS_FILE):
                with open(PERSYMBOL_STATS_FILE) as f:
                    return _ml_json.load(f) or {}
        except Exception as e:
            logging.warning(f"PerSymbolPerformance: load failed ({e})")
        return {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(PERSYMBOL_STATS_FILE), exist_ok=True)
            with open(PERSYMBOL_STATS_FILE, 'w') as f:
                _ml_json.dump(self.stats, f, indent=2)
        except Exception as e:
            logging.warning(f"PerSymbolPerformance: save failed ({e})")

    def record(self, symbol, won, pnl):
        """Called when a position closes. `won` is True/False; `pnl` is the
        realized dollar profit (can be negative)."""
        sym = (symbol or '').upper()
        if not sym:
            return
        with self._lock:
            s = self.stats.setdefault(sym, {'wins': 0, 'losses': 0, 'pnl': 0.0, 'n': 0})
            s['n'] += 1
            s['pnl'] += float(pnl)
            if won:
                s['wins'] += 1
            else:
                s['losses'] += 1
            self._save()

    def win_rate(self, symbol):
        s = self.stats.get((symbol or '').upper())
        if not s or s['n'] == 0:
            return 1.0
        return s['wins'] / s['n']

    def net_pnl(self, symbol):
        s = self.stats.get((symbol or '').upper())
        return float(s['pnl']) if s else 0.0

    def is_muted(self, symbol):
        """Soft mute check for buy_stocks. Returns True to block a buy."""
        sym = (symbol or '').upper()
        if not sym:
            return False
        with self._lock:
            # Auto-unmute after PERSYMBOL_MUTE_DURATION_SECONDS.
            expiry = self.mute_until.get(sym)
            if expiry is not None:
                if time.time() >= expiry:
                    self.mute_until.pop(sym, None)
                else:
                    return True
            s = self.stats.get(sym)
            if not s or s['n'] < PERSYMBOL_MIN_TRADES:
                return False
            if s['pnl'] < 0 and (s['wins'] / s['n']) < PERSYMBOL_MUTE_WINRATE:
                # Trip the mute for the standard duration.
                self.mute_until[sym] = time.time() + PERSYMBOL_MUTE_DURATION_SECONDS
                logging.info(f"PerSymbolPerformance: muting {sym} for "
                             f"{PERSYMBOL_MUTE_DURATION_SECONDS//3600}h "
                             f"(win_rate={s['wins']/s['n']:.2f}, pnl=${s['pnl']:.2f})")
                return True
            return False

    def snapshot(self):
        with self._lock:
            return {k: dict(v) for k, v in self.stats.items()}


PERSYMBOL = PerSymbolPerformance()


# ============================================================================
# TRADE GOVERNOR (ported from Kalshi bot's TradeGovernor)
# ============================================================================
# Global circuit-breaker: after N consecutive losses, pause new entries for
# a cooldown period. Also locks the trading day once a profit target is hit.
# Complements the per-symbol mute above with account-wide protection against
# bleed on bad days.

CONSEC_LOSS_COOLDOWN = 5              # trip after this many losses in a row
COOLDOWN_SECONDS = 30 * 60            # 30-minute cool-off
DAILY_PROFIT_LOCK_PCT = 0.03          # +3% of session-start equity locks the day


class TradeGovernor:
    """Account-wide entry gate. Trips on loss streaks; also stops entries for
    the rest of the trading day when the session gain hits a profit lock.
    Locks reset when the ET date rolls over.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.consec_losses = 0
        self.cooldown_until = 0.0
        self.day_locked = False
        self._locked_date = None
        self._session_start_equity = None
        self._session_date = None

    def _reset_if_new_day(self):
        today = datetime.now(eastern).date().isoformat()
        if self._session_date != today:
            self._session_date = today
            self._session_start_equity = None
            if self._locked_date != today:
                self.day_locked = False
                self._locked_date = None

    def note_session_equity(self, equity):
        """Called at the start of each buy cycle so daily-lock math has a
        session baseline. Only captures the first non-None reading of the
        ET day; subsequent updates same-day are ignored."""
        with self._lock:
            self._reset_if_new_day()
            if self._session_start_equity is None and equity and equity > 0:
                self._session_start_equity = float(equity)

    def on_close(self, won):
        """Called every time a position closes."""
        with self._lock:
            if won:
                self.consec_losses = 0
            else:
                self.consec_losses += 1
                if self.consec_losses >= CONSEC_LOSS_COOLDOWN:
                    self.cooldown_until = time.time() + COOLDOWN_SECONDS
                    logging.warning(f"TradeGovernor: {self.consec_losses} straight "
                                    f"losses — cooldown {COOLDOWN_SECONDS//60}min.")
                    print(f"  [governor] {self.consec_losses} consecutive losses — "
                          f"pausing new entries for {COOLDOWN_SECONDS//60} min.")

    def in_cooldown(self):
        return time.time() < self.cooldown_until

    def can_trade(self, current_equity=None):
        """Called from buy_stocks before considering any candidate. Returns
        (True, '') to allow trading, or (False, reason_string) to block."""
        with self._lock:
            self._reset_if_new_day()
            if self.day_locked:
                return False, f"day locked (session profit ≥ {DAILY_PROFIT_LOCK_PCT*100:.1f}%)"
            if self.in_cooldown():
                remaining = int(self.cooldown_until - time.time())
                return False, f"cooldown after loss streak ({remaining}s left)"
            if (current_equity and self._session_start_equity
                    and current_equity >= self._session_start_equity * (1 + DAILY_PROFIT_LOCK_PCT)):
                self.day_locked = True
                self._locked_date = self._session_date
                gain = (current_equity / self._session_start_equity - 1) * 100
                logging.info(f"TradeGovernor: daily profit lock hit (+{gain:.2f}%) "
                             f"— stopping new entries for the day.")
                print(f"  [governor] daily profit lock hit (+{gain:.2f}%) — "
                      f"no new entries for the rest of the day.")
                return False, f"day locked (+{gain:.2f}%)"
            return True, ''


GOVERNOR = TradeGovernor()


# ============================================================================
# BRAIN TRUST TRACKER (ported from Kalshi bot's BrainTrustTracker)
# ============================================================================
# Rolling accuracy tracker per brain. Every time a brain makes a P(win)
# prediction for a trade, and that trade later closes, we compare
# predicted-direction to realized-direction. Each brain gets a rolling
# trust score from 0.30 (untrustworthy) to 1.50 (hot). Currently reported
# for observation only; can be wired to weight brain votes in a future
# turn once we have enough live samples to trust the trust score itself.

TRUST_WINDOW_SHORT = 50               # last 50 predictions — most responsive
TRUST_WINDOW_MED = 200                # last 200 predictions
TRUST_WINDOW_LONG = 1000              # last 1000 predictions — stability anchor
TRUST_CACHE_TTL_SECONDS = 30


class BrainTrustTracker:
    """Per-brain rolling direction-accuracy + magnitude-correlation tracker.
    Public API:
      - record(brain_name, predicted_prob, actual_won): call when a trade closes
      - trust_score(brain_name) -> float in [0.30, 1.50]
      - is_hot(brain_name), is_cold(brain_name)
      - snapshot() -> dict for dashboards/logs
    Predicted_prob is what the brain output (0-1 probability); actual_won
    is 1 if the trade was profitable, else 0. We convert to signed values
    (pp - 0.5, aw - 0.5) so direction and magnitude can be scored.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._history = {}    # {brain_name: deque of (predicted_prob, actual_won, ts)}
        self._trust_cache = {}
        self._cache_time = {}

    def record(self, brain_name, predicted_prob, actual_won):
        if not brain_name:
            return
        with self._lock:
            h = self._history.get(brain_name)
            if h is None:
                h = deque(maxlen=TRUST_WINDOW_LONG)
                self._history[brain_name] = h
            h.append((float(predicted_prob), 1 if actual_won else 0, time.time()))
            self._cache_time.pop(brain_name, None)

    def trust_score(self, brain_name):
        with self._lock:
            ct = self._cache_time.get(brain_name, 0)
            if time.time() - ct < TRUST_CACHE_TTL_SECONDS and brain_name in self._trust_cache:
                return self._trust_cache[brain_name]
            h = self._history.get(brain_name)
            if not h or len(h) < 10:
                return 0.80   # neutral until we have data
            records = list(h)

            def _accuracy(subset):
                if not subset:
                    return 0.5
                correct = sum(1 for pp, aw, _ in subset
                              if (pp > 0.5 and aw == 1) or (pp <= 0.5 and aw == 0))
                return correct / len(subset)

            def _magnitude_correlation(subset):
                if len(subset) < 5:
                    return 0.5
                preds = [pp - 0.5 for pp, _, _ in subset]
                acts = [aw - 0.5 for _, aw, _ in subset]
                p_mean = sum(preds) / len(preds)
                a_mean = sum(acts) / len(acts)
                num = sum((p - p_mean) * (a - a_mean) for p, a in zip(preds, acts))
                d_p = math.sqrt(max(1e-12, sum((p - p_mean) ** 2 for p in preds)))
                d_a = math.sqrt(max(1e-12, sum((a - a_mean) ** 2 for a in acts)))
                if d_p * d_a < 1e-12:
                    return 0.5
                corr = num / (d_p * d_a)
                return max(0.0, (corr + 1) / 2)  # normalize -1..1 -> 0..1

            short = records[-TRUST_WINDOW_SHORT:] if len(records) >= TRUST_WINDOW_SHORT else records
            med = records[-TRUST_WINDOW_MED:] if len(records) >= TRUST_WINDOW_MED else records
            lng = records

            acc_s = _accuracy(short)
            acc_m = _accuracy(med)
            acc_l = _accuracy(lng)
            corr_s = _magnitude_correlation(short)

            blended = (
                (acc_s * 0.7 + corr_s * 0.3) * 0.50 +
                acc_m * 0.30 +
                acc_l * 0.20
            )
            # 0.5 accuracy -> 0.80 trust (neutral); 0.7 -> 1.3 (hot); 0.3 -> 0.42 (cold)
            trust = max(0.30, min(1.50, 0.30 + blended * 1.40))
            self._trust_cache[brain_name] = trust
            self._cache_time[brain_name] = time.time()
            return trust

    def is_hot(self, brain_name):
        return self.trust_score(brain_name) >= 1.10

    def is_cold(self, brain_name):
        return self.trust_score(brain_name) <= 0.60

    def snapshot(self):
        with self._lock:
            return {name: {
                'n_predictions': len(h),
                'trust': self.trust_score(name),
                'hot': self.is_hot(name),
                'cold': self.is_cold(name),
            } for name, h in self._history.items()}


BRAIN_TRUST = BrainTrustTracker()


# Convenience: pending predictions awaiting outcome. Keyed by (brain_name,
# symbol, entry_epoch). When a position closes we look up matching pending
# entries and call BRAIN_TRUST.record(). Buy_stocks / sell_stocks feed this.
_pending_brain_predictions_lock = threading.Lock()
_pending_brain_predictions = {}   # {(brain_name, symbol): [predicted_prob, ...]}


def brain_trust_record_prediction(brain_name, symbol, predicted_prob):
    """Called at BUY time by any brain that made a prediction, so we can
    score it later when the position closes. Multiple predictions from the
    same brain on the same symbol before close (e.g. re-buys) all queue up
    and get consumed FIFO."""
    if not brain_name or not symbol or predicted_prob is None:
        return
    key = (brain_name, (symbol or '').upper())
    with _pending_brain_predictions_lock:
        _pending_brain_predictions.setdefault(key, []).append(float(predicted_prob))


def brain_trust_settle_symbol(symbol, won):
    """Called when a position closes: drains all pending brain predictions
    for this symbol and records the outcome against each brain's rolling
    accuracy."""
    sym = (symbol or '').upper()
    if not sym:
        return
    with _pending_brain_predictions_lock:
        to_settle = []
        for (brain_name, s), preds in list(_pending_brain_predictions.items()):
            if s == sym and preds:
                for pp in preds:
                    to_settle.append((brain_name, pp))
                _pending_brain_predictions.pop((brain_name, s), None)
    for brain_name, pp in to_settle:
        BRAIN_TRUST.record(brain_name, pp, won)


def get_backtest_status():
    with _backtest_cache_lock:
        n_cached = len(_backtest_cache)
    return f"mode={BACKTEST_BRAIN_MODE}, threshold={BACKTEST_MIN_WIN_RATE:.2f}, cached={n_cached}"


# ============================================================================
# RISK BRAIN (Brain B) — genuine second neural network
# ============================================================================
# A dedicated neural network that evaluates PORTFOLIO/ACCOUNT-level risk,
# separate from Brain A's per-symbol trading decision. Where Brain A asks
# "does this SYMBOL look like a good buy," Brain B asks "does the ACCOUNT
# look like it can safely take another buy right now."
#
# Architecture (deliberately much smaller than Brain A — risk decisions are
# functions of current state, not sequences):
#
#   Input:  12-feature account/portfolio/market-stress vector (not sequence)
#   Layer 1: Dense(12 → 8, tanh)   [FROZEN — hand-weighted risk detectors]
#   Layer 2: Dense(8 → 4, tanh)    [trainable]
#   Layer 3: Dense(4 → 1, sigmoid) [trainable]   → P(safe) in [0, 1]
#
# The frozen foundation encodes classic risk-manager heuristics as fixed
# detector units (margin danger, over-exposure, loss streak, VIX shock,
# drawdown concentration), so P(safe) is meaningful even before any
# training data has accumulated.
#
# Pretraining: 20k SYNTHETIC snapshots covering the 12-feature space with
# deterministic safety labels I know are correct because I chose the label
# rule. Purpose: teach the trainable head a smooth continuous "safety
# surface" so novel real states get sensible interpolations, and to
# validate the frozen detectors are wired correctly. Live fine-tuning on
# actual account-drawdown outcomes can be added later once we have data.
#
# Modes: 'shadow' (default — prints P(safe) every cycle, never vetoes)
#        'armed'  (vetoes buys when P(safe) < BRAIN_B_MIN_SAFE)

BRAIN_B_DIR = os.path.join(_ml_base_dir, 'brain_b_risk_model')
BRAIN_B_MODEL_PATH = os.path.join(BRAIN_B_DIR, 'model.keras')
BRAIN_B_META_PATH = os.path.join(BRAIN_B_DIR, 'meta.json')
BRAIN_B_FEATURES = 12                    # portfolio state features (see below)
BRAIN_B_MODE = 'shadow'                  # 'shadow' | 'armed'
BRAIN_B_MIN_SAFE = 0.40                  # veto when P(safe) below this and mode is armed
BRAIN_B_PRETRAIN_SAMPLES = 20000         # synthetic snapshots for first-run pretrain
BRAIN_B_PRETRAIN_EPOCHS = 12
BRAIN_B_INFER_CACHE_TTL = 5              # cache P(safe) for N seconds so multiple per-cycle
                                          # candidate checks reuse one inference

_brain_b_model = None                    # lazy-loaded singleton
_brain_b_infer_cache = {'ts': 0, 'p_safe': None, 'features': None}


BRAIN_B_FEATURE_LABELS = [
    'margin_ratio',              # equity / long_mv (higher = safer)
    'exposure_frac',             # long_mv / equity (lower = safer)
    'bp_headroom_frac',          # buying_power / equity
    'cash_frac',                 # cash / equity
    'position_count_norm',       # min(n_positions, 10) / 10
    'avg_unrealized_pct',        # avg P/L across open positions
    'worst_unrealized_pct',      # worst P/L (most negative or least positive)
    'session_pnl_pct',           # today's session gain/loss
    'consec_losses_norm',        # min(consec_losses, 10) / 10
    'vix_norm',                  # (vix - 15) / 20 (0=calm, 1=VIX 35)
    'regime_risk',               # 0.0 bull, 0.5 sideways, 0.75 bear, 1.0 panic
    'recent_churn_norm',         # positions closed last hour / 5 (clamped)
]


def _brain_b_build_model(tf):
    """Build Brain B model. Called by first-run pretrain and by
    _brain_b_load_or_build when disk cache is missing."""
    keras = tf.keras
    inputs = keras.Input(shape=(BRAIN_B_FEATURES,), name='risk_features')
    x = keras.layers.Dense(8, activation='tanh', name='risk_foundation',
                            use_bias=True)(inputs)
    x = keras.layers.Dense(4, activation='tanh', name='risk_head_1')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid', name='p_safe')(x)
    model = keras.Model(inputs, outputs, name='brain_b_risk')
    _brain_b_set_foundation_weights(model)
    # Freeze the foundation layer — hand-specified expertise, never learned.
    model.get_layer('risk_foundation').trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _brain_b_set_foundation_weights(model):
    """Hand-specified frozen risk-detector weights. Each of the 8 units
    fires on a specific danger pattern in the 12-feature input. Tanh
    activation squashes to [-1, 1]; positive output = danger detected.
    Column order matches BRAIN_B_FEATURE_LABELS above.
    """
    # Weights: shape (12, 8). Column j = unit j's weights across all 12 features.
    W = np.zeros((BRAIN_B_FEATURES, 8), dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)

    # Unit 0 — MARGIN_DANGER: fires when margin_ratio drops toward floor.
    #   High weight on margin_ratio (index 0), negated: low ratio = high danger.
    W[0, 0] = -2.5   # margin_ratio (lower = more danger)
    b[0] = +2.5      # bias so unit fires when margin_ratio ~ 1.0 or lower

    # Unit 1 — OVER_EXPOSURE: fires when exposure_frac approaches or exceeds 1.
    W[1, 1] = +2.0   # exposure_frac (higher = more danger)
    b[1] = -1.5      # bias so unit crosses zero around exposure_frac = 0.75

    # Unit 2 — LIQUIDITY_STRESS: low buying-power headroom AND low cash.
    W[2, 2] = -1.5   # bp_headroom
    W[3, 2] = -1.0   # cash_frac
    b[2] = +1.5

    # Unit 3 — LOSS_STREAK: consec_losses high AND session_pnl negative.
    W[8, 3] = +2.5   # consec_losses_norm
    W[7, 3] = -1.5   # session_pnl (negative pnl → positive contribution)
    b[3] = -0.8

    # Unit 4 — VIX_SHOCK: VIX elevated AND regime not bull.
    W[9, 4] = +2.0   # vix_norm
    W[10, 4] = +1.5  # regime_risk
    b[4] = -1.5

    # Unit 5 — DRAWDOWN_CONCENTRATION: worst position deep red AND
    #          high exposure means one big loser can drag the account.
    W[6, 5] = -3.0   # worst_unrealized_pct (very negative = danger)
    W[1, 5] = +1.5   # exposure_frac
    b[5] = -1.0

    # Unit 6 — CHURN: many recent closes plus loss streak = something's off.
    W[11, 6] = +2.0  # recent_churn_norm
    W[8, 6] = +1.5   # consec_losses_norm
    b[6] = -1.5

    # Unit 7 — POSITIVE_ENVIRONMENT: HIGH margin ratio + LOW VIX + bull regime.
    #   Included as a counter-balance signal (negative activation → subtracts
    #   from downstream danger score). Everything above fires POSITIVE for
    #   danger; this one fires POSITIVE for safety.
    W[0, 7] = +1.5   # margin_ratio (higher = safer)
    W[9, 7] = -1.5   # vix_norm (lower = safer)
    W[10, 7] = -1.5  # regime_risk (lower = safer)
    b[7] = -0.5

    model.get_layer('risk_foundation').set_weights([W, b])


def _brain_b_build_portfolio_features(margin_state, position_snapshots,
                                       regime_info, governor_state,
                                       recent_closes_last_hour):
    """Build the 12-feature vector for Brain B from live account state.
    All accessors are wrapped so a missing datum falls back to neutral
    (0.5 for regime, 0.0 for pct/count-style features).
    """
    st = margin_state or {}
    equity = max(1.0, float(st.get('equity') or 0))
    long_mv = float(st.get('long_market_value') or 0)
    bp = float(st.get('buying_power') or 0)
    cash = float(st.get('cash') or 0)
    margin_ratio = float(st.get('margin_ratio') or 1.0)

    # Position stats
    positions = position_snapshots or []
    n_pos = len(positions)
    unrealized_pcts = []
    for p in positions:
        avg = p.get('avg_price') or 0
        cur = p.get('current_price') or 0
        if avg > 0 and cur > 0:
            unrealized_pcts.append((cur - avg) / avg)
    avg_unreal = sum(unrealized_pcts) / len(unrealized_pcts) if unrealized_pcts else 0.0
    worst_unreal = min(unrealized_pcts) if unrealized_pcts else 0.0

    # Session P/L. If governor has a session baseline, use it; else neutral 0.
    session_pnl_pct = 0.0
    try:
        base = getattr(GOVERNOR, '_session_start_equity', None)
        if base and base > 0:
            session_pnl_pct = (equity - base) / base
    except Exception:
        pass

    consec_losses = int(governor_state.get('consec_losses', 0)) if governor_state else 0

    # Regime → risk score
    regime_str = (regime_info or {}).get('regime', 'sideways').lower()
    regime_risk = {'bull': 0.0, 'sideways': 0.5, 'bear': 0.75, 'panic': 1.0}.get(regime_str, 0.5)
    vix = (regime_info or {}).get('vix')
    vix_norm = ((float(vix) - 15) / 20) if vix is not None else 0.5
    vix_norm = max(0.0, min(1.5, vix_norm))

    features = np.array([
        float(np.clip(margin_ratio, 0.0, 3.0)),
        float(np.clip(long_mv / equity, 0.0, 2.0)),
        float(np.clip(bp / equity, 0.0, 3.0)),
        float(np.clip(cash / equity, 0.0, 2.0)),
        float(min(n_pos, 10) / 10.0),
        float(np.clip(avg_unreal, -0.5, 0.5)),
        float(np.clip(worst_unreal, -0.5, 0.5)),
        float(np.clip(session_pnl_pct, -0.2, 0.2)),
        float(min(consec_losses, 10) / 10.0),
        float(vix_norm),
        float(regime_risk),
        float(min(recent_closes_last_hour, 5) / 5.0),
    ], dtype=np.float32)
    return features


def _brain_b_synth_generate(n_samples, seed=0):
    """Generate synthetic (features, label) training pairs. Labels are
    deterministic from a hand-designed 'true safety' function: this teaches
    the trainable head to interpolate the frozen foundation's danger
    detectors into a smooth continuous safety surface.

    The synthetic labeling is a SIMPLIFIED version of the danger rules the
    frozen foundation encodes — the head learns to combine detector outputs
    into a probability that matches the ground-truth safety function.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, BRAIN_B_FEATURES), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        # Sample each feature from its natural range
        margin_ratio = rng.uniform(0.3, 2.5)
        exposure_frac = rng.uniform(0.0, 1.5)
        bp_headroom = rng.uniform(0.0, 3.0)
        cash_frac = rng.uniform(0.0, 2.0)
        pos_norm = rng.uniform(0.0, 1.0)
        avg_unreal = rng.uniform(-0.15, 0.15)
        worst_unreal = rng.uniform(-0.30, 0.10)
        session_pnl = rng.uniform(-0.10, 0.10)
        consec_norm = rng.uniform(0.0, 1.0)
        vix_norm = rng.uniform(0.0, 1.5)
        regime_risk = rng.choice([0.0, 0.5, 0.75, 1.0])
        churn_norm = rng.uniform(0.0, 1.0)

        X[i] = [margin_ratio, exposure_frac, bp_headroom, cash_frac,
                pos_norm, avg_unreal, worst_unreal, session_pnl,
                consec_norm, vix_norm, regime_risk, churn_norm]

        # "True" safety function: combines the same signals the foundation
        # detectors fire on. Weights chosen to reflect risk-management
        # priorities (margin > everything else, exposure and VIX shocks
        # next, then drawdown/streak, then churn).
        danger = 0.0
        # Margin danger (dominant)
        if margin_ratio < 1.0:
            danger += (1.0 - margin_ratio) * 3.0
        # Over-exposure
        if exposure_frac > 0.85:
            danger += (exposure_frac - 0.85) * 4.0
        # Liquidity stress
        if bp_headroom < 0.3 and cash_frac < 0.2:
            danger += 1.5
        # Loss streak in the red
        if consec_norm > 0.3 and session_pnl < -0.01:
            danger += consec_norm * 2.0
        # VIX shock in non-bull regime
        if vix_norm > 0.5 and regime_risk >= 0.5:
            danger += (vix_norm - 0.5) * 2.0 * regime_risk
        # Drawdown concentration
        if worst_unreal < -0.05 and exposure_frac > 0.5:
            danger += (-worst_unreal - 0.05) * 4.0 * exposure_frac
        # Churn
        if churn_norm > 0.6:
            danger += (churn_norm - 0.6) * 1.5

        # P(safe) = sigmoid(-danger + 2.0). Higher danger → lower P(safe).
        # +2.0 offset means the baseline (zero danger) starts at P(safe) ≈ 0.88.
        p_safe = 1.0 / (1.0 + math.exp(danger - 2.0))
        # Label is a Bernoulli draw so the head learns a probability,
        # not a step function.
        y[i] = 1.0 if rng.random() < p_safe else 0.0
    return X, y


def brain_b_pretrain_if_needed():
    """Called at startup. If no on-disk Brain B model exists, run one-shot
    synthetic pretraining (~seconds on any CPU). Idempotent."""
    if os.path.exists(BRAIN_B_MODEL_PATH):
        return  # already have a trained model
    tf = _ml_lazy_import_tf()
    if tf is None:
        logging.info("brain_b: tensorflow not available — Brain B disabled.")
        return
    try:
        os.makedirs(BRAIN_B_DIR, exist_ok=True)
        print(f"[brain_b] first-run: pretraining risk brain on "
              f"{BRAIN_B_PRETRAIN_SAMPLES} synthetic snapshots...")
        floor_post('BRAIN_B', 'training',
                   f"pretraining on {BRAIN_B_PRETRAIN_SAMPLES} synthetic snapshots")
        X, y = _brain_b_synth_generate(BRAIN_B_PRETRAIN_SAMPLES)
        model = _brain_b_build_model(tf)
        model.fit(X, y, epochs=BRAIN_B_PRETRAIN_EPOCHS, batch_size=256,
                  validation_split=0.15, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=3, restore_best_weights=True)])
        model.save(BRAIN_B_MODEL_PATH)
        try:
            with open(BRAIN_B_META_PATH, 'w') as f:
                _ml_json.dump({
                    'trained_from': 'synthetic',
                    'trained_at': datetime.now(eastern).isoformat(),
                    'n_examples': int(BRAIN_B_PRETRAIN_SAMPLES),
                    'feature_count': BRAIN_B_FEATURES,
                }, f, indent=2)
        except Exception:
            pass
        print(f"[brain_b] pretrain complete → {BRAIN_B_MODEL_PATH}")
        floor_post('BRAIN_B', 'training', 'pretraining complete')
    except Exception as e:
        logging.warning(f"brain_b: pretraining failed ({e})")


def _brain_b_load(tf):
    """Load Brain B into module-level singleton; rebuild if load fails."""
    global _brain_b_model
    if _brain_b_model is not None:
        return _brain_b_model
    try:
        if os.path.exists(BRAIN_B_MODEL_PATH):
            _brain_b_model = tf.keras.models.load_model(BRAIN_B_MODEL_PATH)
            return _brain_b_model
    except Exception as e:
        logging.warning(f"brain_b: load failed, rebuilding ({e})")
    _brain_b_model = _brain_b_build_model(tf)
    return _brain_b_model


def brain_b_evaluate(margin_state=None, position_snapshots=None,
                      regime_info=None, governor_state=None,
                      recent_closes_last_hour=0, use_cache=True):
    """Public entry point. Returns dict:
        {p_safe: float, features: [...], top_dangers: [(label, contribution)]}
    Cached for BRAIN_B_INFER_CACHE_TTL seconds so multiple per-cycle
    candidate checks reuse one inference."""
    tf = _ml_lazy_import_tf()
    if tf is None:
        return None
    now = time.time()
    if use_cache and _brain_b_infer_cache['p_safe'] is not None:
        if now - _brain_b_infer_cache['ts'] < BRAIN_B_INFER_CACHE_TTL:
            return {
                'p_safe': _brain_b_infer_cache['p_safe'],
                'features': _brain_b_infer_cache['features'],
                'cached': True,
            }
    try:
        if margin_state is None:
            margin_state = get_margin_state()
        if regime_info is None:
            regime_info = get_market_regime()
        if governor_state is None:
            governor_state = {'consec_losses': GOVERNOR.consec_losses,
                              'day_locked': GOVERNOR.day_locked}
        features = _brain_b_build_portfolio_features(
            margin_state, position_snapshots or [], regime_info,
            governor_state, recent_closes_last_hour)
        model = _brain_b_load(tf)
        p_safe = float(model(features.reshape(1, -1), training=False).numpy()[0, 0])
        _brain_b_infer_cache['ts'] = now
        _brain_b_infer_cache['p_safe'] = p_safe
        _brain_b_infer_cache['features'] = features.tolist()
        return {'p_safe': p_safe, 'features': features.tolist(), 'cached': False}
    except Exception as e:
        logging.debug(f"brain_b: evaluation failed ({e})")
        return None


def brain_b_check_veto(margin_state=None, position_snapshots=None,
                        regime_info=None, governor_state=None,
                        recent_closes_last_hour=0):
    """Wraps brain_b_evaluate and applies BRAIN_B_MODE logic. Returns
    (should_veto: bool, reason: str, p_safe: float or None). In shadow
    mode, always returns should_veto=False but still populates reason
    so the caller can print the observation."""
    result = brain_b_evaluate(margin_state=margin_state,
                               position_snapshots=position_snapshots,
                               regime_info=regime_info,
                               governor_state=governor_state,
                               recent_closes_last_hour=recent_closes_last_hour)
    if result is None:
        return False, 'brain_b unavailable', None
    p_safe = result['p_safe']
    mode_note = '' if BRAIN_B_MODE == 'armed' else ' (SHADOW — no veto)'
    if p_safe < BRAIN_B_MIN_SAFE:
        reason = f'P(safe)={p_safe:.2f} < {BRAIN_B_MIN_SAFE:.2f}{mode_note}'
        return (BRAIN_B_MODE == 'armed'), reason, p_safe
    return False, f'P(safe)={p_safe:.2f} OK{mode_note}', p_safe


def get_brain_b_status():
    try:
        exists = os.path.exists(BRAIN_B_MODEL_PATH)
        return (f"mode={BRAIN_B_MODE}, min_safe={BRAIN_B_MIN_SAFE:.2f}, "
                f"model={'loaded' if exists else 'not yet trained'}")
    except Exception as e:
        return f"status unavailable ({e})"


# ============================================================================
# PORTFOLIO MANAGER BRAIN (Brain D)
# ============================================================================
# Third neural network. Different question from A/B/C: given the account's
# RECENT TRAJECTORY (not current state, not per-symbol pattern, not
# historical simulation), what should we DO to grow this account? Should we
# scale size UP or DOWN? Take fewer trades or more? Concentrate or spread?
#
# Architecture:
#   Input:  10 trajectory features (flat vector)
#   Frozen: Dense(10 → 8, tanh)     hand-weighted PM heuristics
#   Head:   Dense(8 → 6, tanh) → Dense(6 → 3, sigmoid)
#   Output: 3-vector [size_mult_raw, aggression, concentration_raw]
#           mapped by _brain_d_scale_outputs into:
#             size_multiplier    ∈ [0.5, 1.5]   → multiplies notional
#             aggression         ∈ [0, 1]        → soft signal
#             concentration_mult ∈ [0.5, 2.0]   → per-symbol cap multiplier
#
# NOT a veto brain. Emits nudges the Chair uses when sizing an approved
# trade. Prints per cycle. Advisory in 'shadow' mode; applied in 'armed'.

BRAIN_D_DIR = os.path.join(_ml_base_dir, 'brain_d_pm_model')
BRAIN_D_MODEL_PATH = os.path.join(BRAIN_D_DIR, 'model.keras')
BRAIN_D_META_PATH = os.path.join(BRAIN_D_DIR, 'meta.json')
BRAIN_D_FEATURES = 10
BRAIN_D_MODE = 'shadow'                  # 'shadow' | 'armed'
BRAIN_D_PRETRAIN_SAMPLES = 20000
BRAIN_D_PRETRAIN_EPOCHS = 12
BRAIN_D_INFER_CACHE_TTL = 5              # seconds

_brain_d_model = None
_brain_d_infer_cache = {'ts': 0, 'outputs': None, 'features': None}


BRAIN_D_FEATURE_LABELS = [
    'ret_7d_pct',          # rolling 7-day equity return
    'ret_30d_pct',          # rolling 30-day equity return
    'vol_7d',              # daily-return stddev over last 7 sessions
    'winrate_last30',      # win rate over last 30 closed trades
    'profit_factor_norm',  # sum(wins) / abs(sum(losses)), clamped/log-normalized
    'max_dd_last30',       # worst peak-to-trough over last 30 trades
    'trade_freq_norm',     # trades per day last 7 days / 10 (clamped)
    'tp_vs_stop_frac',     # frac of trades exited at take-profit vs stop
    'sharpe_21d',          # annualized Sharpe over 21 sessions
    'exposure_frac',       # long_mv / equity
]


def _brain_d_build_model(tf):
    keras = tf.keras
    inputs = keras.Input(shape=(BRAIN_D_FEATURES,), name='pm_features')
    x = keras.layers.Dense(8, activation='tanh', name='pm_foundation',
                            use_bias=True)(inputs)
    x = keras.layers.Dense(6, activation='tanh', name='pm_head_1')(x)
    outputs = keras.layers.Dense(3, activation='sigmoid', name='pm_actions')(x)
    model = keras.Model(inputs, outputs, name='brain_d_pm')
    _brain_d_set_foundation_weights(model)
    model.get_layer('pm_foundation').trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse', metrics=['mae'])
    return model


def _brain_d_set_foundation_weights(model):
    """Hand-specified portfolio-manager heuristics. 8 detector units:
      0 COMPOUNDING_WELL        (positive ret_7d + ret_30d + high winrate)
      1 IN_DRAWDOWN            (negative ret_30d + max_dd severe)
      2 HIGH_SHARPE            (sharpe_21d positive and consistent)
      3 OVER_ACTIVE            (trade_freq high + winrate low)
      4 UNDER_ACTIVE           (trade_freq low)
      5 HIGH_VOL_ENVIRONMENT   (vol_7d elevated)
      6 STOP_HEAVY             (tp_vs_stop_frac low — losing on stops)
      7 CONCENTRATION_HIGH     (exposure fraction high)
    """
    W = np.zeros((BRAIN_D_FEATURES, 8), dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)

    # 0 COMPOUNDING_WELL
    W[0, 0] = +1.5   # ret_7d
    W[1, 0] = +2.0   # ret_30d
    W[3, 0] = +1.5   # winrate
    W[4, 0] = +1.0   # profit_factor
    b[0] = -1.0

    # 1 IN_DRAWDOWN
    W[1, 1] = -2.5   # ret_30d (negative -> danger)
    W[5, 1] = +2.5   # max_dd (larger drawdown -> danger)
    b[1] = -1.0

    # 2 HIGH_SHARPE
    W[8, 2] = +2.5   # sharpe_21d
    W[2, 2] = -1.0   # penalize high daily vol
    b[2] = -0.5

    # 3 OVER_ACTIVE
    W[6, 3] = +2.0   # trade_freq
    W[3, 3] = -1.5   # penalize by winrate (high freq + high winrate is fine)
    b[3] = -0.5

    # 4 UNDER_ACTIVE
    W[6, 4] = -2.0
    b[4] = +1.0      # fires when trade_freq near zero

    # 5 HIGH_VOL_ENVIRONMENT
    W[2, 5] = +2.5   # vol_7d
    b[5] = -1.0

    # 6 STOP_HEAVY
    W[7, 6] = -2.5   # tp_vs_stop_frac low -> danger
    b[6] = +1.0

    # 7 CONCENTRATION_HIGH
    W[9, 7] = +2.0   # exposure_frac
    b[7] = -1.0

    model.get_layer('pm_foundation').set_weights([W, b])


def _brain_d_scale_outputs(raw):
    """Map three sigmoid outputs in [0,1] to their target action ranges.
    raw is a numpy array of shape (3,).
    """
    size_mult = 0.5 + float(raw[0])                # [0.5, 1.5]
    aggression = float(raw[1])                     # [0, 1]
    concentration_mult = 0.5 + 1.5 * float(raw[2]) # [0.5, 2.0]
    return {
        'size_multiplier': size_mult,
        'aggression': aggression,
        'concentration_multiplier': concentration_mult,
    }


def _brain_d_build_trajectory_features():
    """Assembles the 10-feature trajectory vector from live sources:
    api.get_portfolio_history for equity curve, TradeHistory for trade
    outcomes, and the live margin state for exposure. All accessors
    wrapped so missing data falls back to neutral values.
    """
    # Defaults (neutral portfolio)
    ret_7d, ret_30d, vol_7d = 0.0, 0.0, 0.01
    winrate, profit_factor_norm = 0.5, 0.0
    max_dd, trade_freq_norm, tp_vs_stop = 0.0, 0.0, 0.5
    sharpe_21d, exposure_frac = 0.0, 0.0

    try:
        hist = api.get_portfolio_history(period='2M', timeframe='1D')
        eq = list(hist.equity or [])
        eq = [float(e) for e in eq if e is not None and e > 0]
        if len(eq) >= 8:
            ret_7d = (eq[-1] - eq[-8]) / eq[-8]
        if len(eq) >= 22:
            ret_30d = (eq[-1] - eq[-22]) / eq[-22]
        if len(eq) >= 8:
            daily_rets = [(eq[i]-eq[i-1])/eq[i-1] for i in range(-7, 0)]
            vol_7d = float(np.std(daily_rets)) if daily_rets else 0.01
        if len(eq) >= 22:
            r21 = [(eq[i]-eq[i-1])/eq[i-1] for i in range(-21, 0)]
            mean_r = np.mean(r21) if r21 else 0
            sd_r = np.std(r21) if r21 else 1
            sharpe_21d = float((mean_r / sd_r) * math.sqrt(252)) if sd_r > 0 else 0
    except Exception:
        pass

    # Trade stats from last 30 closed trades
    try:
        sess = Session()
        rows = (sess.query(TradeFeatures)
                .filter(TradeFeatures.outcome_pct.isnot(None))
                .order_by(TradeFeatures.id.desc())
                .limit(30).all())
        sess.close()
        if rows:
            outcomes = [float(r.outcome_pct) for r in rows]
            wins = [o for o in outcomes if o > 0]
            losses = [o for o in outcomes if o <= 0]
            winrate = len(wins) / len(outcomes)
            if losses:
                pf = sum(wins) / abs(sum(losses))
                profit_factor_norm = float(np.tanh((pf - 1) * 0.5))
            elif wins:
                profit_factor_norm = 1.0
            # Max drawdown across the 30-trade sequence (chronological reverse of desc query)
            seq = list(reversed(outcomes))
            cum = 0.0; peak = 0.0; max_dd_val = 0.0
            for o in seq:
                cum += o
                peak = max(peak, cum)
                dd = peak - cum
                max_dd_val = max(max_dd_val, dd)
            max_dd = float(np.clip(max_dd_val, 0, 0.5))
    except Exception:
        pass

    # Trade frequency over last 7 days
    try:
        sess = Session()
        cutoff = (datetime.now(eastern).date() - timedelta(days=7)).strftime("%Y-%m-%d")
        n_recent = (sess.query(TradeHistory)
                    .filter(TradeHistory.date >= cutoff)
                    .filter(TradeHistory.action == 'sell')
                    .count())
        sess.close()
        trade_freq_norm = float(min(n_recent, 70) / 70.0)
    except Exception:
        pass

    # Take-profit vs stop-out fraction: approximate from TradeFeatures rows
    # by treating outcome_pct > 0 as "made money", <= 0 as "stopped out"
    # (rough; refine later if we tag exit reason)
    try:
        tp_vs_stop = winrate  # same signal for now
    except Exception:
        pass

    # Exposure
    try:
        st = get_margin_state()
        eq = max(1.0, float(st.get('equity') or 0))
        exposure_frac = float(np.clip(float(st.get('long_market_value') or 0) / eq, 0, 2))
    except Exception:
        pass

    features = np.array([
        float(np.clip(ret_7d, -0.3, 0.3)),
        float(np.clip(ret_30d, -0.3, 0.3)),
        float(np.clip(vol_7d, 0.0, 0.1)),
        float(np.clip(winrate, 0.0, 1.0)),
        float(np.clip(profit_factor_norm, -1.0, 1.0)),
        float(np.clip(max_dd, 0.0, 0.5)),
        float(np.clip(trade_freq_norm, 0.0, 1.0)),
        float(np.clip(tp_vs_stop, 0.0, 1.0)),
        float(np.clip(sharpe_21d, -3.0, 3.0)),
        float(exposure_frac),
    ], dtype=np.float32)
    return features


def _brain_d_synth_generate(n_samples, seed=1):
    """Synthetic trajectory → correct-action training pairs. Encodes what a
    good portfolio manager would do: scale UP when compounding well +
    high Sharpe + low DD, scale DOWN when in drawdown + high vol.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, BRAIN_D_FEATURES), dtype=np.float32)
    y = np.empty((n_samples, 3), dtype=np.float32)
    for i in range(n_samples):
        r7 = rng.uniform(-0.15, 0.15)
        r30 = rng.uniform(-0.20, 0.25)
        vol = rng.uniform(0.001, 0.04)
        wr = rng.uniform(0.2, 0.85)
        pf = rng.uniform(-1.0, 1.0)
        dd = rng.uniform(0.0, 0.3)
        tfreq = rng.uniform(0.0, 1.0)
        tp = wr + rng.uniform(-0.1, 0.1)
        tp = float(np.clip(tp, 0.0, 1.0))
        sharpe = rng.uniform(-2.0, 3.0)
        exp = rng.uniform(0.0, 1.2)

        X[i] = [r7, r30, vol, wr, pf, dd, tfreq, tp, sharpe, exp]

        # SIZE MULTIPLIER target (0=×0.5, 1=×1.5)
        size_score = 0.5
        if r30 > 0: size_score += r30 * 1.5
        if sharpe > 0.5: size_score += (sharpe - 0.5) * 0.15
        if wr > 0.55: size_score += (wr - 0.55) * 0.8
        if dd > 0.05: size_score -= (dd - 0.05) * 1.5
        if r30 < -0.05: size_score -= (-r30 - 0.05) * 2.0
        if vol > 0.02: size_score -= (vol - 0.02) * 5.0
        size_target = float(np.clip(size_score, 0.0, 1.0))

        # AGGRESSION target
        agg_score = 0.5
        if r7 > 0: agg_score += r7 * 1.5
        if wr > 0.5: agg_score += (wr - 0.5) * 0.6
        if dd > 0.1: agg_score -= (dd - 0.1) * 2.0
        if pf > 0: agg_score += pf * 0.15
        if vol > 0.025: agg_score -= (vol - 0.025) * 6.0
        agg_target = float(np.clip(agg_score, 0.0, 1.0))

        # CONCENTRATION MULTIPLIER target (0=×0.5, 1=×2.0)
        conc_score = 0.5
        if sharpe > 1.0: conc_score += (sharpe - 1.0) * 0.15
        if wr > 0.6: conc_score += (wr - 0.6) * 0.5
        if dd > 0.1: conc_score -= (dd - 0.1) * 2.0
        if exp > 0.8: conc_score -= (exp - 0.8) * 1.0
        conc_target = float(np.clip(conc_score, 0.0, 1.0))

        y[i] = [size_target, agg_target, conc_target]
    return X, y


def brain_d_pretrain_if_needed():
    if os.path.exists(BRAIN_D_MODEL_PATH):
        return
    tf = _ml_lazy_import_tf()
    if tf is None:
        logging.info("brain_d: tensorflow not available — Brain D disabled.")
        return
    try:
        os.makedirs(BRAIN_D_DIR, exist_ok=True)
        print(f"[brain_d] first-run: pretraining portfolio manager on "
              f"{BRAIN_D_PRETRAIN_SAMPLES} synthetic trajectories...")
        floor_post('BRAIN_D', 'training',
                   f"pretraining on {BRAIN_D_PRETRAIN_SAMPLES} synthetic trajectories")
        X, y = _brain_d_synth_generate(BRAIN_D_PRETRAIN_SAMPLES)
        model = _brain_d_build_model(tf)
        model.fit(X, y, epochs=BRAIN_D_PRETRAIN_EPOCHS, batch_size=256,
                  validation_split=0.15, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=3, restore_best_weights=True)])
        model.save(BRAIN_D_MODEL_PATH)
        try:
            with open(BRAIN_D_META_PATH, 'w') as f:
                _ml_json.dump({
                    'trained_from': 'synthetic',
                    'trained_at': datetime.now(eastern).isoformat(),
                    'n_examples': int(BRAIN_D_PRETRAIN_SAMPLES),
                    'feature_count': BRAIN_D_FEATURES,
                }, f, indent=2)
        except Exception:
            pass
        print(f"[brain_d] pretrain complete → {BRAIN_D_MODEL_PATH}")
        floor_post('BRAIN_D', 'training', 'pretraining complete')
    except Exception as e:
        logging.warning(f"brain_d: pretraining failed ({e})")


def _brain_d_load(tf):
    global _brain_d_model
    if _brain_d_model is not None:
        return _brain_d_model
    try:
        if os.path.exists(BRAIN_D_MODEL_PATH):
            _brain_d_model = tf.keras.models.load_model(BRAIN_D_MODEL_PATH)
            return _brain_d_model
    except Exception as e:
        logging.warning(f"brain_d: load failed, rebuilding ({e})")
    _brain_d_model = _brain_d_build_model(tf)
    return _brain_d_model


def brain_d_evaluate(use_cache=True):
    """Returns {size_multiplier, aggression, concentration_multiplier,
                features, feature_labels} or None on failure."""
    tf = _ml_lazy_import_tf()
    if tf is None:
        return None
    now = time.time()
    if use_cache and _brain_d_infer_cache['outputs'] is not None:
        if now - _brain_d_infer_cache['ts'] < BRAIN_D_INFER_CACHE_TTL:
            out = _brain_d_infer_cache['outputs']
            return {
                **out,
                'features': _brain_d_infer_cache['features'],
                'feature_labels': BRAIN_D_FEATURE_LABELS,
                'cached': True,
            }
    try:
        features = _brain_d_build_trajectory_features()
        model = _brain_d_load(tf)
        raw = model(features.reshape(1, -1), training=False).numpy()[0]
        scaled = _brain_d_scale_outputs(raw)
        _brain_d_infer_cache['ts'] = now
        _brain_d_infer_cache['outputs'] = scaled
        _brain_d_infer_cache['features'] = features.tolist()
        return {**scaled, 'features': features.tolist(),
                'feature_labels': BRAIN_D_FEATURE_LABELS, 'cached': False}
    except Exception as e:
        logging.debug(f"brain_d: evaluation failed ({e})")
        return None


def get_brain_d_status():
    exists = os.path.exists(BRAIN_D_MODEL_PATH)
    return f"mode={BRAIN_D_MODE}, model={'loaded' if exists else 'not yet trained'}"


# ============================================================================
# BULLISH-TREND PICKER BRAIN (Brain F)
# ============================================================================
# Small, additive "is this symbol looking favorable to buy right now?" brain.
# Feeds a bounded score bump into the same site as get_ml_adjustment() so it
# never overrides the rule-based score -- it just nudges strong bullish setups
# a little higher when the trainable head agrees with the frozen expert layer.
#
# Design mirrors Brain B/D: frozen 8-unit expert foundation (hand-weighted
# bullish-pattern detectors) + trainable Dense head + sigmoid P(bullish),
# pretrained on 20k synthetic examples at first-run boot.
#
# Feature vector (12 numbers; features 0-10 derived from the SAME daily bars
# compute_buy_score already fetched -- no extra HTTP; feature 11 fetched from
# Alpaca News with a 15-min per-symbol cache):
#   0  rsi14_norm         RSI(14) / 100
#   1  macd_norm          MACD line / close
#   2  macd_hist_norm     (MACD - signal) / close
#   3  ret_5d             5-day return
#   4  ret_20d            20-day return
#   5  dist_sma20         (close - SMA20) / SMA20
#   6  dist_sma50         (close - SMA50) / SMA50
#   7  sma20_slope_5d     SMA20 today / SMA20 5d ago - 1
#   8  atr_pct            ATR14 / close
#   9  vol_vs_sma20       (volume / vol_sma20) - 1, clipped +/-5
#  10  sma20_gt_sma50     0/1
#  11  news_sent_pos      rolling positive-only news sentiment, 0..+1
#                         (negative headlines are CLAMPED TO ZERO before
#                         averaging -- only positive news counts for buying
#                         decisions; no news / neutral / negative -> 0.0)
BRAIN_F_DIR = os.path.join(_ml_base_dir, 'brain_f_bullish_model')
BRAIN_F_MODEL_PATH = os.path.join(BRAIN_F_DIR, 'model.keras')
BRAIN_F_META_PATH = os.path.join(BRAIN_F_DIR, 'meta.json')
BRAIN_F_FEATURES = 12
BRAIN_F_MODE = 'armed'                    # 'off' | 'shadow' | 'armed'
BRAIN_F_MAX_SCORE_BUMP = 0.5              # points added when P(bullish) is very high
BRAIN_F_MIN_PROB_TO_BUMP = 0.55           # only bump when P(bullish) exceeds this
BRAIN_F_PRETRAIN_SAMPLES = 20000
BRAIN_F_PRETRAIN_EPOCHS = 12
BRAIN_F_INFER_CACHE_TTL = 300             # 5 min cache per symbol

# News sentiment (Alpaca News API, free with existing keys). POSITIVE-ONLY:
# per-headline sentiment is clamped to max(0, s) before averaging into the
# feature -- negative news is IGNORED for buying decisions (a design choice:
# a bullish-trend brain should not be nudged upward by negative news, and
# shouldn't be nudged downward either -- other brains handle risk/exit).
BRAIN_F_NEWS_ROLL_DAYS = 3
BRAIN_F_NEWS_MAX_PER_SYMBOL = 50
BRAIN_F_NEWS_CACHE_TTL_SEC = 900          # 15 min in-process cache

_brain_f_model = None
_brain_f_infer_cache = {}                 # {sym: (ts, p_bullish)}
_brain_f_news_cache = {}                  # {sym: (ts, pos_avg, neg_avg)} shared
_brain_f_word_re = None                   # lazy-compiled regex

# Dashboard tracker: last activity + session counters for the Bullish Picker
# tile. Written from brain_f_get_score_bump and brain_f_score_symbol; read
# from the WS snapshot builder. Thread-safety is best-effort (single writer
# per cycle, single reader on the WS loop) — no lock; a torn read at worst
# shows the tile a slightly-inconsistent frame that self-corrects next tick.
_brain_f_tracker = {
    'last_symbol': None,
    'last_p_bullish': None,
    'last_bump': None,
    'last_features': None,        # list[float], last full feature vector
    'last_ts': None,              # epoch of last score
    'bumps_session': 0,           # count of nonzero bumps this session
    'positive_bumps_session': 0,  # subset: bumps that pushed score UP
}

BRAIN_F_FEATURE_LABELS = [
    'rsi14_norm', 'macd_norm', 'macd_hist_norm', 'ret_5d', 'ret_20d',
    'dist_sma20', 'dist_sma50', 'sma20_slope_5d', 'atr_pct',
    'vol_vs_sma20', 'sma20_gt_sma50', 'news_sent_pos',
]


# ---------------------------------------------------------------------------
# Brain F news sentiment (Alpaca News API v1beta1, Benzinga-sourced, free
# with existing APCA keys). Lexicon-based scorer with negation + boosters.
# Ported from the standalone bullish_stock_brain.py; adapted here to clamp
# per-headline sentiment to POSITIVE ONLY before averaging -- so only
# positive news contributes to buying decisions.
# ---------------------------------------------------------------------------
_BRAIN_F_LEX_POS = {
    "beat": 0.7, "beats": 0.7, "beating": 0.6, "surge": 0.8, "surges": 0.8,
    "surged": 0.7, "soar": 0.8, "soars": 0.8, "jump": 0.6, "jumps": 0.6,
    "jumped": 0.5, "rally": 0.6, "rallies": 0.6, "rallied": 0.5,
    "rise": 0.4, "rises": 0.4, "rising": 0.4, "gain": 0.4, "gains": 0.4,
    "record": 0.5, "high": 0.3, "highs": 0.3, "outperform": 0.7,
    "outperforms": 0.7, "upgrade": 0.7, "upgraded": 0.7, "upgrades": 0.7,
    "raises": 0.5, "raised": 0.5, "boost": 0.6, "boosts": 0.6, "boosted": 0.5,
    "strong": 0.5, "stronger": 0.5, "strongest": 0.6, "growth": 0.4,
    "grow": 0.3, "grows": 0.3, "expand": 0.4, "expands": 0.4, "expanding": 0.4,
    "profit": 0.4, "profits": 0.4, "profitable": 0.5, "positive": 0.4,
    "bullish": 0.8, "buy": 0.5, "buying": 0.4, "approval": 0.6, "approved": 0.6,
    "wins": 0.5, "win": 0.4, "won": 0.4, "breakthrough": 0.7, "milestone": 0.5,
    "surpass": 0.6, "surpasses": 0.6, "surpassed": 0.5, "exceed": 0.5,
    "exceeds": 0.5, "exceeded": 0.5, "momentum": 0.4, "acceleration": 0.5,
    "accelerating": 0.5, "innovate": 0.3, "innovation": 0.3, "success": 0.5,
    "successful": 0.5, "reward": 0.4, "rewarded": 0.4, "dividend": 0.3,
    "buyback": 0.5, "buybacks": 0.5, "repurchase": 0.4,
}
_BRAIN_F_LEX_NEG = {
    "miss": -0.7, "misses": -0.7, "missed": -0.6, "plunge": -0.8, "plunges": -0.8,
    "plunged": -0.7, "tumble": -0.7, "tumbles": -0.7, "tumbled": -0.6,
    "crash": -0.9, "crashes": -0.9, "crashed": -0.8, "slump": -0.6,
    "slumps": -0.6, "slumped": -0.6, "drop": -0.5, "drops": -0.5, "dropped": -0.4,
    "fall": -0.4, "falls": -0.4, "fell": -0.4, "decline": -0.5, "declines": -0.5,
    "declined": -0.4, "loss": -0.5, "losses": -0.5, "losing": -0.5, "lost": -0.4,
    "low": -0.3, "lows": -0.3, "downgrade": -0.7, "downgraded": -0.7,
    "downgrades": -0.7, "cut": -0.4, "cuts": -0.4, "slash": -0.6, "slashes": -0.6,
    "slashed": -0.6, "warn": -0.6, "warns": -0.6, "warning": -0.6, "warned": -0.5,
    "weak": -0.5, "weaker": -0.5, "weakness": -0.5, "concern": -0.4,
    "concerns": -0.4, "worry": -0.4, "worries": -0.4, "risk": -0.3, "risks": -0.3,
    "risky": -0.4, "bearish": -0.8, "sell": -0.5, "selling": -0.4, "sellers": -0.3,
    "layoff": -0.7, "layoffs": -0.7, "lawsuit": -0.6, "sued": -0.5,
    "investigation": -0.5, "probe": -0.5, "fraud": -0.9, "scandal": -0.8,
    "recall": -0.6, "recalled": -0.5, "delay": -0.4, "delays": -0.4, "delayed": -0.4,
    "disappoint": -0.6, "disappoints": -0.6, "disappointing": -0.6,
    "underperform": -0.7, "underperforms": -0.7, "bankruptcy": -0.9, "default": -0.7,
    "debt": -0.2, "outage": -0.5, "hack": -0.6, "breach": -0.6, "halted": -0.5,
    "suspend": -0.4, "suspended": -0.4,
}
_BRAIN_F_NEG_MODIFIERS = {"not", "no", "never", "without", "nor"}
_BRAIN_F_BOOSTERS = {"very": 1.3, "highly": 1.3, "extremely": 1.5,
                     "significantly": 1.3, "sharply": 1.3, "sharp": 1.3}


def _brain_f_score_text(text):
    """Lexicon sentiment in [-1, +1] with simple negation + booster handling.
    Note: this returns the RAW signed sentiment. The positive-only clamp is
    applied in _brain_f_rolling_positive_sentiment() before averaging."""
    global _brain_f_word_re
    if not text:
        return 0.0
    import re
    if _brain_f_word_re is None:
        _brain_f_word_re = re.compile(r"[A-Za-z']+")
    tokens = [w.lower() for w in _brain_f_word_re.findall(text)]
    if not tokens:
        return 0.0
    total = 0.0
    hits = 0
    for i, w in enumerate(tokens):
        base = _BRAIN_F_LEX_POS.get(w, 0.0) + _BRAIN_F_LEX_NEG.get(w, 0.0)
        if base == 0.0:
            continue
        prev = tokens[max(0, i - 3): i]
        if any(p in _BRAIN_F_NEG_MODIFIERS for p in prev):
            base = -base
        if i > 0 and tokens[i - 1] in _BRAIN_F_BOOSTERS:
            base *= _BRAIN_F_BOOSTERS[tokens[i - 1]]
        total += base
        hits += 1
    if hits == 0:
        return 0.0
    avg = total / hits
    return float(max(-1.0, min(1.0, avg)))


def _brain_f_alpaca_news_headers():
    """Reuse the same APCA keys the bot already reads for market data."""
    try:
        # These are already set at module scope for the alpaca_trade_api / alpaca-py
        # clients elsewhere in the file. Fall back to env just in case.
        key = os.environ.get("APCA_API_KEY_ID") or globals().get("APCA_API_KEY_ID")
        sec = os.environ.get("APCA_API_SECRET_KEY") or globals().get("APCA_API_SECRET_KEY")
    except Exception:
        key = sec = None
    if not key or not sec:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def _brain_f_fetch_news(symbol, limit=None):
    """Fetch recent Alpaca News headlines for `symbol`. Returns a list of
    dicts with per-headline SIGNED sentiment. Positive-only clamping happens
    at the aggregation step, not here."""
    import requests
    headers = _brain_f_alpaca_news_headers()
    if headers is None:
        return []
    if limit is None:
        limit = BRAIN_F_NEWS_MAX_PER_SYMBOL
    # Look back BRAIN_F_NEWS_ROLL_DAYS + a small buffer so the rolling
    # average has coverage even after weekends.
    try:
        start_iso = (datetime.now(timezone.utc)
                     - timedelta(days=BRAIN_F_NEWS_ROLL_DAYS + 4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        start_iso = None
    params = {"symbols": symbol.upper(), "limit": min(50, limit), "sort": "desc"}
    if start_iso:
        params["start"] = start_iso
    try:
        r = requests.get("https://data.alpaca.markets/v1beta1/news",
                         headers=headers, params=params, timeout=8)
        if r.status_code != 200:
            return []
        data = r.json().get("news", [])
    except Exception:
        return []
    out = []
    for item in data:
        headline = item.get("headline", "") or ""
        summary = item.get("summary", "") or ""
        sent = _brain_f_score_text(headline + ". " + summary)
        out.append({
            "created_at": item.get("created_at", ""),
            "headline": headline,
            "sentiment": sent,   # signed; caller clamps to positive-only
        })
    return out


def _brain_f_compute_news_rollups(symbol):
    """Fetch recent Alpaca News for `symbol`, filter to the rolling window,
    and return (pos_avg, neg_avg) where:
      * pos_avg in [0, +1]: mean of max(0, s) across headlines in window
      * neg_avg in [0, +1]: mean of max(0, -s) across headlines in window
    (Symmetric averages: for pos_avg only positive headlines lift the value;
    for neg_avg only negative headlines lift the value; neutral -> 0 in both.)

    Cached BRAIN_F_NEWS_CACHE_TTL_SEC seconds per symbol -- one Alpaca fetch
    serves both the bullish-brain feature and the negative-news stop-tightener.
    Returns (0.0, 0.0) on any failure so callers degrade gracefully."""
    now = time.time()
    hit = _brain_f_news_cache.get(symbol)
    if hit and (now - hit[0]) < BRAIN_F_NEWS_CACHE_TTL_SEC:
        return hit[1], hit[2]

    items = _brain_f_fetch_news(symbol)
    if not items:
        _brain_f_news_cache[symbol] = (now, 0.0, 0.0)
        return 0.0, 0.0

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=BRAIN_F_NEWS_ROLL_DAYS)
    except Exception:
        cutoff = None

    pos_scores = []
    neg_scores = []
    for it in items:
        ts_str = it.get("created_at", "")
        if cutoff is not None and ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts < cutoff:
                    continue
            except Exception:
                pass  # if parse fails, still include it
        s = float(it.get("sentiment", 0.0))
        pos_scores.append(max(0.0, s))
        neg_scores.append(max(0.0, -s))

    pos_avg = float(sum(pos_scores) / len(pos_scores)) if pos_scores else 0.0
    neg_avg = float(sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0
    pos_avg = max(0.0, min(1.0, pos_avg))
    neg_avg = max(0.0, min(1.0, neg_avg))
    _brain_f_news_cache[symbol] = (now, pos_avg, neg_avg)
    return pos_avg, neg_avg


def _brain_f_rolling_positive_sentiment(symbol):
    """Rolling positive-only news sentiment in [0, +1] for Brain F's bullish
    feature. Only positive headlines contribute; negative and neutral -> 0.
    Uses the shared _brain_f_compute_news_rollups cache."""
    pos, _neg = _brain_f_compute_news_rollups(symbol)
    return pos


def brain_f_rolling_negative_sentiment(symbol):
    """Rolling negative-only news sentiment in [0, +1] used by the negative-news
    stop-loss tightener on OWNED positions. Only negative headlines contribute
    (each headline's sentiment is clamped to max(0, -s) before averaging);
    positive and neutral -> 0. This is the exit-side counterpart to the
    positive-only feature Brain F uses for buy decisions -- same cache, same
    fetch, so a symbol we already looked up for scoring costs zero extra HTTP
    when we later check it for stop tightening. Public (no leading underscore)
    because the chandelier evaluator lives elsewhere in the file."""
    _pos, neg = _brain_f_compute_news_rollups(symbol)
    return neg


def _brain_f_set_foundation_weights(model):
    """Hand-weighted 9-unit frozen foundation. Each unit fires on a specific
    bullish pattern; tanh activation squashes to [-1, +1]; positive output =
    bullish-setup signal detected. Column order matches BRAIN_F_FEATURE_LABELS."""
    W = np.zeros((BRAIN_F_FEATURES, 9), dtype=np.float32)
    b = np.zeros(9, dtype=np.float32)

    RSI, MACD, MACDH, R5, R20, D20, D50, SLOPE, ATRP, VOL, XUP, NEWS = range(12)

    # Unit 0 -- OVERSOLD_SNAPBACK: RSI low + recent 5d not disastrous.
    W[RSI, 0] = -3.0
    W[R5, 0] = 2.0
    b[0] = +1.0

    # Unit 1 -- BULLISH_MACD: MACD > signal AND MACD line positive.
    W[MACD, 1] = +6.0
    W[MACDH, 1] = +8.0

    # Unit 2 -- UPTREND_STRUCTURE: above SMA20/50, SMA20 sloping up, golden cross.
    W[D20, 2] = +3.0
    W[D50, 2] = +2.0
    W[SLOPE, 2] = +6.0
    W[XUP, 2] = +1.5
    b[2] = -0.4

    # Unit 3 -- MOMENTUM_STRONG: positive 5d and 20d returns.
    W[R5, 3] = +5.0
    W[R20, 3] = +3.0

    # Unit 4 -- VOLUME_CONFIRMS: above-average volume backing the move.
    W[VOL, 4] = +1.5
    b[4] = -0.2

    # Unit 5 -- NOT_OVEREXTENDED: penalize price too far above SMA20 (parabolic).
    W[D20, 5] = -4.0
    b[5] = +0.6

    # Unit 6 -- LOW_CHOP: penalize very high ATR% (whippy tape).
    W[ATRP, 6] = -8.0
    b[6] = +0.8

    # Unit 7 -- BULL_COMPOSITE: broad mix of the good stuff.
    W[MACD, 7] = +3.0
    W[MACDH, 7] = +3.0
    W[SLOPE, 7] = +3.0
    W[R5, 7] = +2.0
    W[R20, 7] = +1.5
    W[D20, 7] = +1.0
    W[XUP, 7] = +0.5
    W[RSI, 7] = -0.5
    b[7] = -0.3

    # Unit 8 -- POSITIVE_NEWS_TAILWIND: fires when positive-only rolling news
    # sentiment is elevated. Because the news feature is already clamped
    # to [0, +1] upstream, this unit can ONLY provide upward pressure --
    # never a bearish signal. That matches the design rule: only positive
    # news counts for buying decisions.
    W[NEWS, 8] = +2.5
    b[8] = -0.2

    model.get_layer('bullish_foundation').set_weights([W, b])


def _brain_f_build_model(tf):
    keras = tf.keras
    inputs = keras.Input(shape=(BRAIN_F_FEATURES,), name='bullish_features')
    x = keras.layers.Dense(9, activation='tanh', name='bullish_foundation',
                           use_bias=True)(inputs)
    x = keras.layers.Dense(6, activation='tanh', name='bullish_head_1')(x)
    x = keras.layers.Dense(4, activation='tanh', name='bullish_head_2')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid', name='p_bullish')(x)
    model = keras.Model(inputs, outputs, name='brain_f_bullish')
    _brain_f_set_foundation_weights(model)
    model.get_layer('bullish_foundation').trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _brain_f_synth_generate(n):
    """Generate n synthetic (feature, label) training examples.
    Label = 1 if the sample is a "genuinely bullish setup" (mix of positive
    MACD, uptrend, momentum, volume confirmation), else 0. Deterministic
    scoring function drawn from the same intuitions as the frozen weights
    so training teaches the head to interpolate smoothly, not to fight
    the expert layer."""
    X = np.zeros((n, BRAIN_F_FEATURES), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        bullish = np.random.rand() < 0.5
        # Base drift determines all correlated features
        if bullish:
            drift = np.random.uniform(0.0005, 0.004)
            rsi = np.clip(np.random.normal(0.60, 0.12), 0.05, 0.95)
        else:
            drift = np.random.uniform(-0.004, -0.0005)
            rsi = np.clip(np.random.normal(0.40, 0.12), 0.05, 0.95)
        macd_n = drift * 6 + np.random.normal(0, 0.002)
        macd_h = drift * 4 + np.random.normal(0, 0.001)
        r5 = drift * 5 + np.random.normal(0, 0.02)
        r20 = drift * 20 + np.random.normal(0, 0.05)
        d20 = drift * 10 + np.random.normal(0, 0.02)
        d50 = drift * 15 + np.random.normal(0, 0.03)
        slope = drift * 5 + np.random.normal(0, 0.003)
        atrp = abs(np.random.normal(0.015, 0.008))
        vol = np.random.normal(0.15 if bullish else -0.05, 0.4)
        xup = 1.0 if drift > 0 else 0.0
        # Synthetic positive-only news sentiment. Bullish setups more often
        # coincide with positive news; the feature is 0..+1 (already clamped
        # in the live path). Most samples get 0 (no news / all-negative day).
        if bullish:
            if np.random.rand() < 0.55:
                news = float(np.clip(np.random.beta(2.0, 3.5), 0.0, 1.0))
            else:
                news = 0.0
        else:
            if np.random.rand() < 0.20:
                news = float(np.clip(np.random.beta(1.5, 5.0) * 0.6, 0.0, 1.0))
            else:
                news = 0.0
        X[i] = [rsi, macd_n, macd_h, r5, r20, d20, d50, slope, atrp, vol, xup, news]
        # Ground-truth label: bullish + not overextended + not too choppy.
        # Strong positive news can rescue a marginal setup (small boost),
        # matching the "only positive news counts for buying" rule.
        overextended = d20 > 0.15
        too_choppy = atrp > 0.05
        base_bull = bullish and not overextended and not too_choppy
        news_rescue = (not base_bull) and bullish and news > 0.4 and not too_choppy
        y[i] = 1.0 if (base_bull or news_rescue) else 0.0
    return X, y


def brain_f_pretrain_if_needed():
    """Run once at startup. Idempotent: no-op if a VALID (correct feature
    count) model already exists on disk. If an on-disk model has the wrong
    feature count -- e.g. a pre-news 11-feature model still sitting from
    before the news feature was added -- we delete it and retrain, rather
    than silently building an untrained fresh model at inference time."""
    if BRAIN_F_MODE == 'off':
        return
    tf = _ml_lazy_import_tf()
    if tf is None:
        logging.warning('brain_f: tensorflow not available; skipping pretrain')
        return

    # Validate any existing on-disk model. If it exists but has the wrong
    # feature count, wipe it so we retrain fresh below.
    if os.path.exists(BRAIN_F_MODEL_PATH):
        try:
            existing = tf.keras.models.load_model(BRAIN_F_MODEL_PATH)
            if (existing.input_shape
                    and existing.input_shape[-1] == BRAIN_F_FEATURES):
                return  # valid model already trained; nothing to do
            logging.warning(
                f'brain_f: on-disk model has {existing.input_shape[-1]} '
                f'features but code expects {BRAIN_F_FEATURES}; deleting '
                'stale model and retraining.'
            )
            print(f'[brain_f] on-disk model is stale '
                  f'({existing.input_shape[-1]} features vs code '
                  f'{BRAIN_F_FEATURES}); deleting and retraining')
            try:
                os.remove(BRAIN_F_MODEL_PATH)
                if os.path.exists(BRAIN_F_META_PATH):
                    os.remove(BRAIN_F_META_PATH)
            except Exception as _re:
                logging.warning(f'brain_f: could not remove stale model files: {_re}')
        except Exception as e:
            logging.warning(f'brain_f: could not inspect on-disk model ({e}); '
                            'assuming stale and retraining')
            try:
                os.remove(BRAIN_F_MODEL_PATH)
            except Exception:
                pass

    os.makedirs(BRAIN_F_DIR, exist_ok=True)
    print(f'[brain_f] no valid model on disk -- pretraining {BRAIN_F_PRETRAIN_SAMPLES:,} synthetic examples')
    t0 = time.time()
    model = _brain_f_build_model(tf)
    X, y = _brain_f_synth_generate(BRAIN_F_PRETRAIN_SAMPLES)
    # Chronological-ish split (last 15% val)
    n_val = max(200, int(len(X) * 0.15))
    model.fit(X[:-n_val], y[:-n_val],
              validation_data=(X[-n_val:], y[-n_val:]),
              epochs=BRAIN_F_PRETRAIN_EPOCHS, batch_size=128,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss', patience=3, restore_best_weights=True)],
              verbose=0)
    model.save(BRAIN_F_MODEL_PATH)
    with open(BRAIN_F_META_PATH, 'w') as f:
        json.dump({
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'n_examples': int(len(X)),
            'n_features': BRAIN_F_FEATURES,
            'feature_labels': BRAIN_F_FEATURE_LABELS,
            'pretrain_epochs': BRAIN_F_PRETRAIN_EPOCHS,
        }, f, indent=2)
    # Reset in-process cached model so the next inference reloads from disk
    # (rather than continuing to use an old cached _brain_f_model with the
    # wrong feature count).
    global _brain_f_model
    _brain_f_model = None
    dt = time.time() - t0
    print(f'[brain_f] pretrain complete in {dt:.1f}s -> {BRAIN_F_MODEL_PATH}')


def _brain_f_load_or_build():
    global _brain_f_model
    if _brain_f_model is not None:
        return _brain_f_model
    tf = _ml_lazy_import_tf()
    if tf is None:
        return None
    if os.path.exists(BRAIN_F_MODEL_PATH):
        try:
            _brain_f_model = tf.keras.models.load_model(BRAIN_F_MODEL_PATH)
            # Sanity: input feature count must match
            if _brain_f_model.input_shape and _brain_f_model.input_shape[-1] == BRAIN_F_FEATURES:
                return _brain_f_model
            logging.warning('brain_f: on-disk model feature-count mismatch; '
                            'triggering full retrain (not just an in-memory rebuild)')
            _brain_f_model = None
            # Delete the stale file so pretrain sees it as missing and retrains.
            try:
                os.remove(BRAIN_F_MODEL_PATH)
                if os.path.exists(BRAIN_F_META_PATH):
                    os.remove(BRAIN_F_META_PATH)
            except Exception:
                pass
        except Exception as e:
            logging.warning(f'brain_f: load failed ({e}); triggering full retrain')
            _brain_f_model = None
            try:
                os.remove(BRAIN_F_MODEL_PATH)
            except Exception:
                pass
    # No valid model on disk. Two paths:
    #   1) Retrain synchronously (correct, but expensive -- 5s+ blocking).
    #   2) Fall back to build-fresh (fast but head is UNTRAINED / random weights).
    # Path 2 was the historical behavior and produces meaningless P(bullish)
    # scores. Path 1 is safer; a brain producing garbage is worse than a
    # brain that's briefly slow at startup.
    try:
        brain_f_pretrain_if_needed()
        if os.path.exists(BRAIN_F_MODEL_PATH):
            _brain_f_model = tf.keras.models.load_model(BRAIN_F_MODEL_PATH)
            if (_brain_f_model.input_shape
                    and _brain_f_model.input_shape[-1] == BRAIN_F_FEATURES):
                return _brain_f_model
    except Exception as e:
        logging.warning(f'brain_f: synchronous retrain failed ({e}); '
                        'falling back to untrained fresh model')
    # Last-resort fallback -- head is untrained; log loudly so the operator
    # notices something is wrong and can trigger a manual retrain.
    logging.warning('brain_f: using UNTRAINED fresh model; P(bullish) values '
                    'will be near-random until a proper retrain completes')
    _brain_f_model = _brain_f_build_model(tf)
    return _brain_f_model


def _brain_f_build_features_from_df(df, current_price=None, symbol=None):
    """Extract the 12-feature vector from a daily-bar DataFrame. Features
    0-10 use the SAME df compute_buy_score already fetched -- no extra HTTP.
    Feature 11 (news_sent_pos) is a positive-only rolling Alpaca News
    sentiment score for `symbol` (0.0 when symbol is None, keys are missing,
    the fetch fails, or all recent headlines are neutral/negative).
    Returns None if the df is too short or missing required columns."""
    try:
        if df is None or len(df) < 50:
            return None
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float) if 'Volume' in df.columns else pd.Series(np.zeros(len(df)))
        if current_price is None or not np.isfinite(current_price):
            current_price = float(close.iloc[-1])

        # RSI(14) via EWMA (no talib dep at this depth)
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)
        rsi_norm = float(rsi.iloc[-1]) / 100.0

        # MACD 12/26/9
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        px = max(1e-6, float(close.iloc[-1]))
        macd_norm = float(np.clip(macd.iloc[-1] / px, -0.2, 0.2))
        macd_hist_norm = float(np.clip((macd.iloc[-1] - signal.iloc[-1]) / px, -0.1, 0.1))

        # Returns
        def _ret(n):
            if len(close) <= n:
                return 0.0
            return float(np.clip(close.iloc[-1] / close.iloc[-1-n] - 1.0, -0.6, 0.6))
        r5 = _ret(5)
        r20 = _ret(20)

        # SMAs
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        s20 = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else px
        s50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else px
        dist20 = float(np.clip((current_price - s20) / max(1e-6, s20), -0.3, 0.3))
        dist50 = float(np.clip((current_price - s50) / max(1e-6, s50), -0.5, 0.5))
        if len(sma20) > 5 and pd.notna(sma20.iloc[-6]) and sma20.iloc[-6] > 0:
            slope = float(np.clip(s20 / float(sma20.iloc[-6]) - 1.0, -0.2, 0.2))
        else:
            slope = 0.0
        xup = 1.0 if s20 > s50 else 0.0

        # ATR(14) as EWMA of true range
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(),
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        atr_pct = float(np.clip(atr.iloc[-1] / px, 0.0, 0.2))

        # Volume vs 20d avg
        vol_sma = volume.rolling(20).mean()
        if pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
            vol_vs = float(np.clip(volume.iloc[-1] / vol_sma.iloc[-1] - 1.0, -5.0, 5.0))
        else:
            vol_vs = 0.0

        # Positive-only rolling news sentiment (0.0 if no symbol, no keys,
        # fetch fails, or no positive headlines in the window).
        if symbol:
            try:
                news_pos = float(_brain_f_rolling_positive_sentiment(symbol))
            except Exception:
                news_pos = 0.0
        else:
            news_pos = 0.0

        feats = np.array([rsi_norm, macd_norm, macd_hist_norm, r5, r20,
                          dist20, dist50, slope, atr_pct, vol_vs, xup, news_pos],
                         dtype=np.float32)
        if not np.all(np.isfinite(feats)):
            return None
        return feats
    except Exception:
        return None


def brain_f_score_symbol(symbol, df, current_price=None):
    """Return P(bullish) in [0, 1] for `symbol` given its daily-bar df.
    Cached BRAIN_F_INFER_CACHE_TTL seconds per symbol. Returns None on any
    failure -- caller must handle None as "no bump this cycle"."""
    if BRAIN_F_MODE == 'off':
        return None
    now = time.time()
    hit = _brain_f_infer_cache.get(symbol)
    if hit and (now - hit[0]) < BRAIN_F_INFER_CACHE_TTL:
        # Cache hit: still update last_symbol/last_p_bullish/last_ts for the
        # dashboard so the tile reflects the most recent symbol scored this
        # cycle, but leave last_features alone (we don't have them at hand).
        _brain_f_tracker['last_symbol'] = symbol
        _brain_f_tracker['last_p_bullish'] = hit[1]
        _brain_f_tracker['last_ts'] = now
        return hit[1]
    feats = _brain_f_build_features_from_df(df, current_price=current_price, symbol=symbol)
    if feats is None:
        return None
    model = _brain_f_load_or_build()
    if model is None:
        return None
    try:
        p = float(model.predict(feats.reshape(1, -1), verbose=0).reshape(-1)[0])
    except Exception:
        return None
    _brain_f_infer_cache[symbol] = (now, p)
    # Dashboard tracker: record every fresh score.
    _brain_f_tracker['last_symbol'] = symbol
    _brain_f_tracker['last_p_bullish'] = p
    _brain_f_tracker['last_features'] = [float(v) for v in feats.tolist()]
    _brain_f_tracker['last_ts'] = now
    return p


def brain_f_get_score_bump(symbol, df, current_price=None):
    """Compute the bounded additive buy-score bump for this symbol.
    Returns (bump, p_bullish) where bump is in
    [-BRAIN_F_MAX_SCORE_BUMP/2, +BRAIN_F_MAX_SCORE_BUMP]. Never bumps in
    'off' mode. In 'shadow' mode always returns bump=0 (for observation
    logging only). Only bumps positively when P(bullish) > BRAIN_F_MIN_PROB_TO_BUMP;
    small negative bump when P(bullish) < 0.35 to gently deprioritize weak setups."""
    p = brain_f_score_symbol(symbol, df, current_price=current_price)
    if p is None:
        return (0.0, None)
    if BRAIN_F_MODE == 'shadow':
        _brain_f_tracker['last_bump'] = 0.0
        return (0.0, p)
    # Positive bump scaled with confidence above threshold; negative bump for weak setups
    if p >= BRAIN_F_MIN_PROB_TO_BUMP:
        # Linear ramp from 0 at threshold to MAX_BUMP at P=1.0
        span = max(1e-6, 1.0 - BRAIN_F_MIN_PROB_TO_BUMP)
        bump = BRAIN_F_MAX_SCORE_BUMP * (p - BRAIN_F_MIN_PROB_TO_BUMP) / span
    elif p <= 0.35:
        # Gentle negative bump for clearly weak setups (half the positive cap)
        bump = -BRAIN_F_MAX_SCORE_BUMP * 0.5 * (0.35 - p) / 0.35
    else:
        bump = 0.0
    bump = round(bump, 3)
    # Dashboard tracker: record bump + increment session counters.
    _brain_f_tracker['last_bump'] = bump
    if bump != 0.0:
        _brain_f_tracker['bumps_session'] += 1
        if bump > 0:
            _brain_f_tracker['positive_bumps_session'] += 1
    return (bump, p)


def get_brain_f_status():
    exists = os.path.exists(BRAIN_F_MODEL_PATH)
    return f"mode={BRAIN_F_MODE}, max_bump={BRAIN_F_MAX_SCORE_BUMP}, model={'loaded' if exists else 'not yet trained'}"


# ============================================================================
# TRADE APPROVAL BRAIN (Brain E — the Chair)
# ============================================================================
# The Chair is the SINGLE POINT OF DECISION for every trade. It:
#   1. Queries every other brain for its verdict on this candidate
#   2. Posts each brain's verdict to the Brain Trading Floor
#   3. Applies deterministic aggregation rules (documented below)
#   4. Posts its own final decision to the floor
#   5. Returns the decision (approve/deny + adjusted notional) to buy_stocks
#
# This is NOT an LLM conversation between agents. Brains are neural nets or
# statistical evaluators that emit structured BrainVote records; the Chair
# combines them with explicit weighted rules. The floor discussion is a
# HUMAN-READABLE AUDIT LOG of a deterministic decision, not the mechanism
# by which the decision is reached.
#
# Aggregation rules (highest to lowest priority):
#   VETO: any brain with veto authority (Backtest_C when armed,
#         Risk_B when armed) returning DENY → SKIP the trade.
#   SIZE: Portfolio_D's size_multiplier scales the notional. In shadow
#         mode the multiplier is printed but not applied.
#   TRADING_A weight: raw trading brain P(win) is advisory; the Chair
#         does not veto on it (Brain A already applies its own score
#         adjustment upstream via ML_MAX_SCORE_ADJUSTMENT).
#
# BrainVote: (name, verdict in {'APPROVE','DENY','ADVISE'}, detail_string,
#             meta_dict with brain-specific numbers)


class BrainVote:
    __slots__ = ('name', 'verdict', 'detail', 'meta')
    def __init__(self, name, verdict, detail, meta=None):
        self.name = name
        self.verdict = verdict          # 'APPROVE' | 'DENY' | 'ADVISE'
        self.detail = detail
        self.meta = meta or {}
    def to_floor_msg(self):
        emoji = ('✅' if self.verdict == 'APPROVE' else
                 '❌' if self.verdict == 'DENY' else '💡')
        return f"{emoji} {self.verdict} — {self.detail}"


def check_early_morning_entry_gate(current_price, df):
    """Early-morning entry gate. Returns (allow, reason).

    Before EARLY_MORNING_CUTOFF_ET (10:05 AM Eastern) new entries are
    blocked UNLESS the candidate has already dropped >=EARLY_MORNING_MIN_DROP_PCT
    below yesterday's daily close (2% by default) AND is currently trading
    strictly BELOW yesterday's close. Both must hold -- the "and lower than
    yesterday's close" clause is redundant with the drop check for positive
    thresholds, but it guards against future config changes that set the
    threshold to zero or negative.

    After the cutoff, the gate is fully open regardless of price.

    In 'shadow' mode always returns allow=True but the reason string
    records what the armed decision would have been -- so the buy loop can
    log it without acting.
    """
    if EARLY_MORNING_GATE_MODE == 'off':
        return True, 'early-gate off'

    now_et = datetime.now(eastern).time()
    if now_et >= EARLY_MORNING_CUTOFF_ET:
        return True, (f'post-cutoff ({now_et.strftime("%H:%M")} ET >= '
                      f'{EARLY_MORNING_CUTOFF_ET.strftime("%H:%M")} ET)')

    # Pre-cutoff: need a real drop from yesterday's close.
    try:
        # Yesterday's daily close is the LAST fully-closed daily bar in df.
        # In an intraday-updated daily-bar df, iloc[-1] is today's forming
        # bar; the previous row is yesterday's close. If the df only has
        # one row, we can't gate safely -- default open, tag the reason.
        if df is None or len(df) < 2:
            reason = 'pre-cutoff, no prior daily bar -> allow (insufficient data)'
            return True, reason
        prev_close = float(df['Close'].iloc[-2])
    except Exception as e:
        return True, f'pre-cutoff, prev-close lookup failed ({e}) -> allow'

    if prev_close <= 0 or current_price is None or current_price <= 0:
        return True, 'pre-cutoff, unusable prices -> allow'

    drop_pct = (prev_close - current_price) / prev_close
    lower_than_yesterday = current_price < prev_close
    drop_qualifies = drop_pct >= EARLY_MORNING_MIN_DROP_PCT

    if lower_than_yesterday and drop_qualifies:
        reason = (f'pre-cutoff ({now_et.strftime("%H:%M")} ET) but qualifies: '
                  f'-{drop_pct*100:.2f}% vs prev close ${prev_close:.2f} '
                  f'(need >=-{EARLY_MORNING_MIN_DROP_PCT*100:.1f}%)')
        return True, reason

    # Denied
    reason = (f'pre-cutoff ({now_et.strftime("%H:%M")} ET < '
              f'{EARLY_MORNING_CUTOFF_ET.strftime("%H:%M")} ET): '
              f'{drop_pct*100:+.2f}% vs prev close ${prev_close:.2f} '
              f'does not meet >=-{EARLY_MORNING_MIN_DROP_PCT*100:.1f}% + '
              f'strictly-below-prev-close requirement')
    if EARLY_MORNING_GATE_MODE == 'shadow':
        return True, f'[SHADOW would-deny] {reason}'
    return False, reason


def _chair_gather_votes(symbol, current_regime=None, ml_adjustment=None,
                        bt_verdict=None, bt_result=None,
                        current_price=None, df=None):
    """Query every brain and return list of BrainVote. Brains that fail
    are represented as ADVISE with an error tag so the floor still shows
    what happened.

    `current_price` and `df` are optional and only used by the early-morning
    entry gate (RISK_TIME vote). When omitted, that vote is ADVISE.
    """
    votes = []

    # Trading brain (Brain A) — advisory (already influenced the score)
    if ml_adjustment is not None:
        inferred_pwin = 0.5 + (ml_adjustment / (2 * ML_MAX_SCORE_ADJUSTMENT))
        inferred_pwin = max(0.0, min(1.0, inferred_pwin))
        lean = ('LEAN BUY' if inferred_pwin >= ML_BRAIN_WIN_THRESHOLD
                else 'LEAN AVOID' if inferred_pwin <= (1 - ML_BRAIN_WIN_THRESHOLD)
                else 'NEUTRAL')
        votes.append(BrainVote(
            'TRADING_A', 'ADVISE',
            f"{symbol}: P(win)={inferred_pwin:.2f} → {lean} (adj {ml_adjustment:+.2f})",
            {'p_win': inferred_pwin, 'lean': lean, 'adjustment': ml_adjustment}))
    else:
        votes.append(BrainVote(
            'TRADING_A', 'ADVISE',
            f"{symbol}: no ML adjustment available",
            {'p_win': None}))

    # Backtest brain (Brain C) — veto authority when armed
    if bt_verdict is not None and bt_result is not None:
        if bt_verdict == 'VETO':
            wr = bt_result.get('decision_win_rate', 0) * 100
            n = bt_result.get('n_signals', 0)
            v = 'DENY' if BACKTEST_BRAIN_MODE == 'armed' else 'ADVISE'
            votes.append(BrainVote(
                'BACKTEST_C', v,
                f"{symbol}: {n} sigs, {wr:.0f}% win — below "
                f"{BACKTEST_MIN_WIN_RATE*100:.0f}% threshold",
                {'win_rate': bt_result.get('decision_win_rate'),
                 'n_signals': n, 'mode': BACKTEST_BRAIN_MODE}))
        elif bt_verdict == 'PASS':
            wr = bt_result.get('decision_win_rate', 0) * 100
            n = bt_result.get('n_signals', 0)
            votes.append(BrainVote(
                'BACKTEST_C', 'APPROVE',
                f"{symbol}: {n} sigs, {wr:.0f}% win — clears threshold",
                {'win_rate': bt_result.get('decision_win_rate'), 'n_signals': n}))
        else:  # ABSTAIN or ERROR
            votes.append(BrainVote(
                'BACKTEST_C', 'ADVISE',
                f"{symbol}: {bt_verdict.lower()} — abstaining",
                {'reason': bt_verdict}))
    else:
        votes.append(BrainVote('BACKTEST_C', 'ADVISE',
                                f"{symbol}: unavailable", {}))

    # Risk brain (Brain B) — veto authority when armed. Portfolio-level, not
    # per-symbol; same vote goes on every candidate this cycle.
    try:
        b_veto, b_reason, b_p_safe = brain_b_check_veto()
        if b_p_safe is not None:
            if b_veto:
                votes.append(BrainVote(
                    'RISK_B', 'DENY',
                    f"portfolio risk: {b_reason}",
                    {'p_safe': b_p_safe, 'mode': BRAIN_B_MODE}))
            elif b_p_safe < BRAIN_B_MIN_SAFE:
                # In shadow mode, would deny — mark as ADVISE
                votes.append(BrainVote(
                    'RISK_B', 'ADVISE',
                    f"portfolio risk: {b_reason}",
                    {'p_safe': b_p_safe, 'mode': BRAIN_B_MODE}))
            else:
                votes.append(BrainVote(
                    'RISK_B', 'APPROVE',
                    f"portfolio safe: P(safe)={b_p_safe:.2f}",
                    {'p_safe': b_p_safe, 'mode': BRAIN_B_MODE}))
        else:
            votes.append(BrainVote('RISK_B', 'ADVISE',
                                    'unavailable', {}))
    except Exception as e:
        votes.append(BrainVote('RISK_B', 'ADVISE', f'error: {e}', {}))

    # Risk brain (time gate) -- veto authority when armed. Before 10:05 ET,
    # deny unless the candidate has dropped >=2% below yesterday's close.
    # Portfolio-independent; per-symbol. Same vote goes on every candidate
    # this cycle when only time is checked, but per-symbol when price/df
    # are provided (which the buy loop does).
    try:
        if current_price is None or df is None:
            votes.append(BrainVote('RISK_TIME', 'ADVISE',
                                    'no price/df provided to gather_votes',
                                    {'mode': EARLY_MORNING_GATE_MODE}))
        else:
            allow, reason = check_early_morning_entry_gate(current_price, df)
            if allow:
                votes.append(BrainVote(
                    'RISK_TIME', 'APPROVE', reason,
                    {'mode': EARLY_MORNING_GATE_MODE,
                     'cutoff': EARLY_MORNING_CUTOFF_ET.strftime('%H:%M'),
                     'min_drop_pct': EARLY_MORNING_MIN_DROP_PCT}))
            else:
                votes.append(BrainVote(
                    'RISK_TIME', 'DENY', reason,
                    {'mode': EARLY_MORNING_GATE_MODE,
                     'cutoff': EARLY_MORNING_CUTOFF_ET.strftime('%H:%M'),
                     'min_drop_pct': EARLY_MORNING_MIN_DROP_PCT}))
    except Exception as e:
        votes.append(BrainVote('RISK_TIME', 'ADVISE', f'error: {e}', {}))

    # Portfolio manager (Brain D) — nudge, never vetoes
    try:
        d = brain_d_evaluate()
        if d is not None:
            votes.append(BrainVote(
                'PORTFOLIO_D', 'ADVISE',
                f"size ×{d['size_multiplier']:.2f}, "
                f"aggression {d['aggression']:.2f}, "
                f"concentration ×{d['concentration_multiplier']:.2f}",
                d))
        else:
            votes.append(BrainVote('PORTFOLIO_D', 'ADVISE',
                                    'unavailable', {}))
    except Exception as e:
        votes.append(BrainVote('PORTFOLIO_D', 'ADVISE', f'error: {e}', {}))

    return votes


def chair_decide(symbol, notional_requested, current_regime=None,
                 ml_adjustment=None, bt_verdict=None, bt_result=None,
                 current_price=None, df=None):
    """The single point of decision. Returns dict:
        {approved: bool, notional: float, reason: str, votes: [BrainVote]}
    Every brain vote is posted to the Brain Trading Floor before this
    returns, followed by the Chair's decision line.

    `current_price` and `df` are optional; when supplied they let RISK_TIME
    evaluate the early-morning entry gate per-symbol.
    """
    votes = _chair_gather_votes(symbol, current_regime=current_regime,
                                 ml_adjustment=ml_adjustment,
                                 bt_verdict=bt_verdict, bt_result=bt_result,
                                 current_price=current_price, df=df)

    # Post each brain's vote to the floor.
    for v in votes:
        floor_post(v.name,
                   'block' if v.verdict == 'DENY'
                   else 'approve' if v.verdict == 'APPROVE'
                   else 'discuss',
                   v.to_floor_msg())

    # Aggregate: any DENY blocks the trade.
    deniers = [v for v in votes if v.verdict == 'DENY']
    if deniers:
        deny_names = ', '.join(v.name for v in deniers)
        reason = f"denied by: {deny_names}"
        floor_post('CHAIR', 'block',
                   f"{symbol}: SKIP — {reason}")
        return {'approved': False, 'notional': 0.0, 'reason': reason,
                'votes': votes}

    # Approved. Apply Portfolio D's size multiplier if armed.
    size_mult = 1.0
    for v in votes:
        if v.name == 'PORTFOLIO_D' and v.meta.get('size_multiplier') is not None:
            if BRAIN_D_MODE == 'armed':
                size_mult = float(v.meta['size_multiplier'])
            break
    adjusted_notional = notional_requested * size_mult
    d_note = ('' if size_mult == 1.0
              else f" (Portfolio_D ×{size_mult:.2f}: "
                   f"${notional_requested:.2f}→${adjusted_notional:.2f})")
    floor_post('CHAIR', 'approve',
               f"{symbol}: EXECUTE @ ${adjusted_notional:.2f}{d_note}")
    return {'approved': True, 'notional': adjusted_notional,
            'reason': 'all brains cleared', 'votes': votes}


def get_ml_status():
    """Human-readable status for logging/diagnostics -- never raises."""
    try:
        if os.path.exists(ML_META_PATH):
            with open(ML_META_PATH) as f:
                meta = _ml_json.load(f)
            return (f"trained_from={meta.get('trained_from', 'unknown')}, "
                   f"trained_at={meta.get('trained_at')}, "
                   f"n_examples={meta.get('n_examples')}, "
                   f"val_accuracy={meta.get('val_accuracy')}")
        return "no saved model yet"
    except Exception as e:
        return f"status unavailable ({e})"


# Backward-compat shim: this name was used by the previous ML section, kept
# so the main-loop retrain call site doesn't need editing. It now just wraps
# maybe_run_scheduled_ml_training, since the "retrain on every N cycles"
# concept from the old design is replaced by the 17:00 schedule per instruction.
def train_ml_brain_from_live_trades(sess, tf_model, force=False):
    return "live-trade fine-tuning replaced by daily 17:00 ET schedule; see maybe_run_scheduled_ml_training"

# ANSI color codes for terminal output.
# Green for positive P/L, red for negative, dim for contextual detail.
# Set env var _NO_COLOR=1 to disable all coloring (for log files or terminals
# that don't render ANSI escapes). All constants become empty strings in that
# case, so existing f-strings that interpolate them keep working.
_USE_COLOR = os.environ.get("_NO_COLOR") != "1"
GREEN = "\033[92m" if _USE_COLOR else ""
RED = "\033[91m" if _USE_COLOR else ""
DIM = "\033[2m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""


def pnl_color(value):
    """Return GREEN for non-negative P/L values, RED for negative.
    Use in f-strings: f"{pnl_color(pnl)}{pnl:+.2f}%{RESET}".
    A value of exactly 0 is treated as non-negative (green) since
    flat isn't a loss."""
    try:
        return GREEN if float(value) >= 0 else RED
    except (TypeError, ValueError):
        return ""

APIKEYID = os.getenv('APCA_API_KEY_ID')
APISECRETKEY = os.getenv('APCA_API_SECRET_KEY')
APIBASEURL = os.getenv('APCA_API_BASE_URL')

api = tradeapi.REST(APIKEYID, APISECRETKEY, APIBASEURL)

# ---------------- Alpaca-py market data client (free-plan IEX feed) ----------
# Coexists with the legacy alpaca_trade_api trading client above -- both use
# the same API key/secret. This client is READ-ONLY market data (no order
# capability at all) and is used ONLY for:
#   - S&P 500 scanner daily-bar fetch  (replaces yfinance batch download)
#   - Bull-market monitor latest-trade sampling  (replaces yfinance 1m bars)
# The trading path (orders, positions, cash, portfolio history) stays on the
# original `api` client so working code isn't disturbed.
from alpaca.data.historical import StockHistoricalDataClient as _AlpacaHistClient
from alpaca.data.requests import (
    StockBarsRequest as _AlpacaBarsRequest,
    StockLatestTradeRequest as _AlpacaLatestTradeRequest,
)
from alpaca.data.enums import DataFeed as _AlpacaDataFeed
from alpaca.data.timeframe import TimeFrame as _AlpacaTimeFrame

alpaca_data_client = _AlpacaHistClient(api_key=APIKEYID, secret_key=APISECRETKEY)

# Symbol-form conversion between yfinance ('-') and Alpaca ('.') is handled
# by the existing to_yf() and to_alpaca() helpers defined later in this file
# (kept together with the other symbol utilities). This section just wires
# in the client; usage sites call to_alpaca(sym) inline before making a
# request and to_yf(sym) when normalizing an Alpaca response back to the
# bot's internal symbol form.


def _alpaca_latest_trade_single(sym):
    """Fetch the latest IEX trade price for ONE symbol via Alpaca (free plan).
    Returns float or None. Used as the primary source in _fetch_current_price;
    the caller's yf_history fallback still runs on any failure.

    NOTE: this is ONE HTTP call regardless of caller context. Where the caller
    is looping over N symbols one-at-a-time, that's still N HTTP calls -- but
    each call is ~100ms via Alpaca vs ~1s+ via yfinance, and Alpaca's IEX
    endpoint doesn't throttle or intermittently 404 the way yfinance does.
    """
    try:
        req = _AlpacaLatestTradeRequest(
            symbol_or_symbols=to_alpaca(sym),
            feed=_AlpacaDataFeed.IEX,
        )
        resp = alpaca_data_client.get_stock_latest_trade(req)
        # Response is keyed by the Alpaca-form symbol.
        trade = resp.get(to_alpaca(sym))
        if trade is None:
            return None
        p = float(trade.price)
        return p if p > 0 else None
    except Exception as e:
        logging.debug(f"[alpaca] latest-trade fetch failed for {sym}: {e}")
        return None


def _alpaca_daily_bars_single(sym, days):
    """Fetch the last `days` calendar days of daily bars for ONE symbol from
    Alpaca IEX. Returns a pandas DataFrame with the SAME columns yfinance
    produces (Open, High, Low, Close, Volume) and a DatetimeIndex named
    'Date' so downstream indicator code (talib.ATR / MACD / RSI / etc.)
    can consume it unchanged.

    Returns an empty DataFrame on any failure -- caller checks .empty and
    falls back to yfinance if needed.
    """
    try:
        end = datetime.now(pytz.UTC)
        start = end - timedelta(days=int(days) + 5)  # small pad for weekends
        req = _AlpacaBarsRequest(
            symbol_or_symbols=to_alpaca(sym),
            timeframe=_AlpacaTimeFrame.Day,
            start=start,
            end=end,
            feed=_AlpacaDataFeed.IEX,
        )
        resp = alpaca_data_client.get_stock_bars(req)
        data = resp.data if hasattr(resp, 'data') else resp
        bars = data.get(to_alpaca(sym), []) if hasattr(data, 'get') else []
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame([{
            'Open':   float(b.open),
            'High':   float(b.high),
            'Low':    float(b.low),
            'Close':  float(b.close),
            'Volume': float(b.volume),
        } for b in bars],
            index=pd.DatetimeIndex([b.timestamp for b in bars], name='Date'),
        )
        return df
    except Exception as e:
        logging.debug(f"[alpaca] daily bars fetch failed for {sym}: {e}")
        return pd.DataFrame()
# ---------------- End alpaca-py market data client ---------------------------

global symbols_to_buy
global price_history, last_stored, interval_map

# ---------------- Configuration flags ----------------
PRINT_SYMBOLS_TO_BUY = False
PRINT_ROBOT_STORED_BUY_AND_SELL_LIST_DATABASE = True
PRINT_DATABASE = True
DEBUG = False
ALL_BUY_ORDERS_ARE_1_DOLLAR = False
FRACTIONAL_BUY_ORDERS = True

# ---------------- 2026 Margin Account Rules ----------------
# The legacy FINRA Pattern Day Trader rule (4 round-trips / 5 business days,
# $25k minimum equity) is no longer enforced by this robot.
# Instead we operate under margin-account risk controls:
ACCOUNT_MODE = 'margin'          # 'margin' or 'cash'
UNLIMITED_DAY_TRADES = True      # No PDT round-trip counting
MAX_PORTFOLIO_EXPOSURE_PCT = 0.98    # of equity (buying power aware)
MAX_LEVERAGE = 1.0               # 1.0 = no borrowing. Raise to 2.0 for Reg-T intraday.
RISK_PER_TRADE_PCT = 0.01        # 1% of equity risked per position
MAX_ALLOCATION_PER_SYMBOL = 600.0
MAX_NEW_POSITIONS_PER_CYCLE = 3  # rank all qualifying candidates, only buy the top N (review item #9)

# ---------------- Early-morning entry gate ----------------
# Before this cutoff (Eastern time), the risk brain and buy path together
# require ONE of two conditions to permit a new entry:
#   1. It's already past EARLY_MORNING_CUTOFF_ET, OR
#   2. The candidate's current price is BOTH:
#        a) strictly LOWER than yesterday's daily close, AND
#        b) at least EARLY_MORNING_MIN_DROP_PCT below yesterday's close
# The open (roughly 9:30-10:05 ET) is thin, volatile, and prone to opening-
# range fakeouts. Restricting early buys to real >=2% dislocations from
# yesterday's close means we still catch overnight gap-downs and open-panic
# dips (where the edge is best) while filtering out the routine open-drift
# entries that historically stop out on the reversion at 10:00-10:30.
from datetime import time as _dt_time  # local alias -- 'time' module is used elsewhere in this file
EARLY_MORNING_CUTOFF_ET = _dt_time(10, 5)   # 10:05 AM Eastern
EARLY_MORNING_MIN_DROP_PCT = 0.02           # 2% below yesterday's close
EARLY_MORNING_GATE_MODE = 'armed'           # 'armed' | 'shadow' | 'off'

# ---------------- ML brain (optional buy-score adjustment) ----------------
# Adds a small +/- adjustment to the buy score from the inlined ML brain
# section above, ONLY once it has enough of the bot's own closed-trade
# history to be minimally trustworthy for LIVE decisions (see
# ML_MIN_TRADES_FOR_LIVE_ADJUSTMENT). Below that threshold, or if TensorFlow
# isn't available at all, this contributes NOTHING -- the bot behaves
# exactly as it did before this feature existed.
USE_ML_BRAIN_ADJUSTMENT = True
ML_BRAIN_RETRAIN_EVERY_N_CYCLES = 120   # live-trade fine-tune cadence -- ~2 hours at 60s/cycle, cheap-gated internally too
MIN_ORDER_NOTIONAL = 1.00
CASH_BUFFER = 1.00
MAINTENANCE_MARGIN_FLOOR_PCT = 0.30  # abort new buys if equity/market_value dips below

# ---------------- Hard stop-loss (review items #6/#7) ----------------
# The profit monitor and scaled exits above only ever act once a position is
# ARMED (i.e. already showing a gain) -- there was previously no equivalent
# floor on the downside: a position that went straight from entry to -8%
# with no bounce through the arm threshold had nothing forcing an exit. This
# is a genuine hard stop: checked every sell_stocks() cycle regardless of the
# profit monitor's armed/unarmed state, and fires independently of it.
#
# HARD_STOP_ATR_MULTIPLIER is also the SAME distance position sizing uses to
# compute risk-per-share (see buy_stocks' `risk_per_share = HARD_STOP_ATR_MULTIPLIER * atr`).
# Before this fix, sizing math assumed a 2xATR stop that didn't actually
# exist anywhere in the exit logic -- the "1% risk" the sizing model promised
# was fictional. Now both use the same constant, so a position sized to risk
# RISK_PER_TRADE_PCT of equity is actually capped at that loss by a real stop.
USE_HARD_STOP_LOSS = True
HARD_STOP_ATR_MULTIPLIER = 2.0   # stop at entry_price - (this x current ATR)
HARD_STOP_MIN_PCT = 0.03         # never let a low-ATR stock's stop sit tighter than -3%
                                  # (a near-zero ATR would otherwise produce a razor-thin stop)

# ---------------- Exit strategy ----------------
# Two exits can act on the same shares. A GTC trailing stop RESERVES shares at
# the broker, so a later take-profit sell can only touch the unreserved fraction
# unless the stop is cancelled first (see cancel_open_sell_orders).
#
# IMPORTANT: the trailing stop and the profit monitor are redundant and the stop
# is COARSER. A 1% trailing stop fires long before the monitor's 0.2% giveback
# ever triggers, so leaving both on means the broker-side stop wins every race
# and the peak-following logic never actually runs. USE_TRAILING_STOP therefore
# defaults to False when the monitor is enabled.
USE_TRAILING_STOP = False        # broker-side 1% trailing stop (coarse)
TRAIL_PERCENT = 1.0
TAKE_PROFIT_PCT = 1.005          # +0.5% flat target, only used if monitor is off

# ---------------- Bull-market strategy (gated by regime == REGIME_BULL) ----------------
# Ported from the standalone Bull Market Advanced Stock Market Trading Robot v5.
# When the SPY+VIX regime detector reports REGIME_BULL, this adds a SECOND buy
# path that runs ALONGSIDE the normal ML-scored path (candidates from both
# feeds are merged into the same ranked shortlist; MAX_NEW_POSITIONS_PER_CYCLE
# still applies). Positions opened by this path are tagged strategy_tag='bull'
# in the Position table and use a distinct exit rule:
#   - +0.5% flat take-profit checked in sell_stocks BEFORE the normal exit chain
#   - broker-side 1% trailing stop placed at fill time, whether or not
#     USE_TRAILING_STOP is on globally (this path always places its own)
# Sizing: equal cash allocation across all bull candidates in the cycle,
# capped at BULL_MAX_ALLOCATION_PER_SYMBOL (default $600).
#
# Buy gate (ALL must fire, per v5):
#   - Live tick monitoring: BULL_MONITOR_SECONDS of samples at BULL_SAMPLE_INTERVAL
#     seconds each, then require increases >= BULL_MIN_PRICE_INCREASES AND
#     increases > decreases.
#   - Daily MACD signal line > BULL_MACD_SIGNAL_MIN (0.15)
#   - Daily RSI(14) > BULL_RSI_MIN (70)
#   - Latest daily volume > BULL_VOLUME_MULT_OF_MEAN (0.85) x 90d mean volume
#   - Time of day between BULL_BUY_WINDOW_START and BULL_BUY_WINDOW_END (ET)
# The 3-minute monitoring pass makes a bull cycle take ~3-4 minutes instead of
# the usual ~60s. sell_stocks still runs in parallel on its own thread so exits
# aren't blocked during monitoring.
USE_BULL_MARKET_STRATEGY = True
BULL_MONITOR_SECONDS = 180                 # per-symbol price-tick monitoring window
BULL_SAMPLE_INTERVAL = 10                  # seconds between price samples
BULL_MIN_SAMPLES = 12                      # of BULL_MONITOR_SECONDS/BULL_SAMPLE_INTERVAL = 18 max
BULL_MIN_PRICE_INCREASES = 3               # need at least this many up-ticks
BULL_MIN_NET_RETURN = 0.001                # last must be >= first * (1 + this)   (0.10%)
BULL_MAX_MONITOR_DRAWDOWN = 0.005          # max intra-window drawdown allowed    (0.50%)
BULL_MACD_SIGNAL_MIN = 0.15                # MACD signal line must exceed this
BULL_RSI_MIN = 70.0                        # daily RSI(14) must exceed this
BULL_VOLUME_MULT_OF_MEAN = 0.85            # today's volume > this x 90d mean
BULL_BUY_WINDOW_START = (10, 2)            # (hour, minute) ET
BULL_BUY_WINDOW_END = (15, 35)             # (hour, minute) ET
BULL_TAKE_PROFIT_PCT = 1.005               # +0.5% flat sell target
BULL_TRAIL_PERCENT = 1.0                   # 1% broker-side trailing stop
BULL_MAX_ALLOCATION_PER_SYMBOL = 600.0     # per-symbol cash cap for bull buys
# Note: no dedicated rate-limit config for the bull path -- the batched
# _bull_batch_monitor calls yf_download_batch once per BULL_SAMPLE_INTERVAL
# regardless of symbol count, and yf_download_batch already goes through the
# bot's global yf_gate. At the defaults (180s / 10s) that's 18 HTTP calls per
# bull scan whether we're monitoring 3 symbols or 50.

# ---------------- Profit Monitor (peak-following exit) ----------------
# There is NO holding-period restriction: a position can be sold the same second
# it is bought. PDT is retired, so same-day round trips are unrestricted.
#
# Rather than dumping at the first tick over +0.5%, the monitor ARMS at that
# level and then follows price up, tracking a high-water mark. It sells when
# price pulls back from the peak by GIVEBACK_PCT, so a run to +3% is captured
# instead of being cut at +0.5%.
USE_PROFIT_MONITOR = True
ARM_PROFIT_PCT = 0.005           # +0.5% flat fallback -> monitor arms and begins following
PEAK_GIVEBACK_PCT = 0.002        # 0.2% flat fallback -> sell after a pullback from the peak
HARD_FLOOR_PCT = 0.001           # never sell armed positions below +0.1% net
MONITOR_STALE_SECS = 900         # drop peak state unseen for 15m (position gone)

# ---------------- Round-trip protection (pre-arm profit rescue) ----------------
# BUGFIX: the profit monitor previously did NOTHING until a position reached the
# +0.5% arm threshold. A position that ran to, say, +0.35% and then reversed all
# the way into a loss was invisible to the monitor -- it stayed in the
# 'watching' state until the hard stop / chandelier eventually caught it well
# into the red. The operator's complaint: "the profit monitor failed to sell a
# position that was profitable before it was a loss."
#
# Fix: track the peak_price from tick #1 (even before arming). If the position
# EVER touched at least PROFIT_TOUCHED_PCT above entry and then round-trips to
# at or below entry (i.e. gives back the entire touched gain), exit immediately.
# This is a strict superset of the old behavior -- positions that arm still
# follow the normal peak/giveback logic; this only rescues the sub-arm
# round-trippers that used to slip through.
PROFIT_TOUCHED_PCT = 0.002       # +0.2% counts as "was profitable" for rescue
ROUND_TRIP_EXIT_GAIN = 0.0       # exit when a touched-profit position drops to <= breakeven

# ---------------- Volatility-based profit targets (review item #2) ----------------
# Flat percentages above are used ONLY as a fallback when ATR is unavailable.
# When ATR is available, arm/giveback scale with the stock's own volatility so
# winners can breathe on volatile days while quiet-market trades still bank
# quick, small gains. See ProfitMonitorEngine._arm_threshold_for() and
# ._giveback_for_peak().
ATR_ARM_MULTIPLIER = 0.30        # arm at max(ARM_PROFIT_PCT, 0.30 x ATR%)
ATR_GIVEBACK_FRACTION = 0.20     # giveback floor = 20% of the ARM threshold (small moves)

# REVIEW ITEM #9: the giveback above was originally scaled only off the arm
# threshold, which is fixed once a position arms -- so a stock that ran to
# +1.2% and one that ran to +6% got the SAME tiny giveback room, sacrificing
# a lot of a strong trend's potential move. PEAK_GIVEBACK_FRACTION instead
# scales giveback off the position's ACTUAL peak gain as it grows, so a
# bigger run earns proportionally more room before the monitor sells. The
# arm-based ATR_GIVEBACK_FRACTION above still sets a FLOOR so a position that
# just barely armed doesn't get an unreasonably wide giveback either -- see
# ProfitMonitorEngine._thresholds_for_peak(), which takes max(arm-based floor,
# peak-based scaling).
PEAK_GIVEBACK_FRACTION = 0.20    # giveback = 20% of the peak gain achieved so far

# ---------------- Scaled exits (review item #5) ----------------
# Instead of an all-or-nothing sell, take partial profit in tranches and move
# the effective floor up as each tranche fires, letting the remainder run.
USE_SCALED_EXITS = True
SCALE_OUT_STAGES = [
    # (trigger_gain_pct, fraction_of_ORIGINAL_qty_to_sell)
    (0.010, 0.25),   # +1.0% -> sell 25% of the original position
    (0.020, 0.25),   # +2.0% -> sell another 25% (50% cumulative)
]
# After the LAST configured stage fires, the remaining shares are handed to
# the normal peak-following ProfitMonitorEngine with its stop effectively
# moved to breakeven (HARD_FLOOR_PCT floor already does most of this; the
# scale-out logic additionally refuses to let the remainder exit below
# breakeven once a scale-out stage has fired).

# ---------------- Pre-market open / pre-close profit sweeps ----------------
# At MOO_SWEEP_HOUR:MOO_SWEEP_MINUTE Eastern (default 9:25am) and again at
# CLOSE_SWEEP_HOUR:CLOSE_SWEEP_MINUTE Eastern (default 3:45pm, 15 min before
# the close), check every open position's unrealized P/L and sell any
# position currently showing a profit. Both sweeps use a 3-step escalation
# chain (see _sell_with_escalation) so a sale is not left to chance on a
# single order type: initial order (MOO pre-market / plain market pre-close)
# -> aggressive limit at the bid if that doesn't fill -> plain market order as
# a final guarantee-of-fill fallback. Neither sweep touches or replaces the
# existing intraday profit-monitor / scaled-exit logic, which still runs
# during the regular session as before.
USE_PREMARKET_PROFIT_SWEEP = True
MOO_SWEEP_HOUR = 9
MOO_SWEEP_MINUTE = 25
# Alpaca rejects OPG (market-on-open) orders submitted after approximately
# 9:28am ET. 150 seconds (9:25:00-9:27:30) leaves real margin before that
# cutoff -- covering broker/network latency plus a wait-loop tick that runs
# a bit slow -- rather than running the window right up against 9:29, which
# risks a rejected OPG order on a delayed cycle.
MOO_SWEEP_CUTOFF_SECS = 150
MOO_SWEEP_MIN_PROFIT_PCT = 0.0  # "any profit" -- must be strictly above this

USE_CLOSE_PROFIT_SWEEP = True
CLOSE_SWEEP_HOUR = 15
CLOSE_SWEEP_MINUTE = 45   # 3:45pm ET -- 15 minutes before the 4:00pm close

# ---------------- Portfolio-level liquidation (phase 2 of each sweep) ----------------
# After phase 1 sells whatever individual positions are already profitable,
# check what's LEFT: if the combined unrealized P/L of the remaining
# positions, divided by their combined cost basis, is >= this threshold, sell
# everything that's left -- winners and losers together -- because the
# winners are covering the losers for a net portfolio profit. If the
# remainder doesn't clear the bar, those positions are left alone (this does
# NOT force-sell losers on its own; phase 1's per-position winners still sell
# regardless of what phase 2 decides).
USE_PORTFOLIO_LIQUIDATION_SWEEP = True
PORTFOLIO_LIQUIDATION_MIN_PROFIT_PCT = 0.01   # 1% combined; raise to 0.02 for 2%

# ---------------- Threading ----------------
# Worker threads are joined with a timeout so a hung API call in one thread
# cannot freeze the main loop indefinitely.
THREAD_JOIN_TIMEOUT = 180

eastern = pytz.timezone('US/Eastern')

stock_data = {}
previous_prices = {}
price_changes = {}

price_history = {}
last_stored = {}
interval_map = {
    '1min': 60, '5min': 300, '10min': 600, '15min': 900,
    '30min': 1800, '45min': 2700, '60min': 3600
}

buy_sell_lock = threading.Lock()
yf_lock = threading.Lock()

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(filename=os.path.join(_BASE_DIR, 'trading-bot-program-logging-messages.txt'),
                    level=logging.INFO)

# BUGFIX: relative path meant the trade log followed the launch directory, same
# as the .db issue. Anchor it to the script directory.
csv_filename = os.path.join(_BASE_DIR, 'log-file-of-buy-and-sell-signals.csv')
fieldnames = ['Date', 'Buy', 'Sell', 'Quantity', 'Symbol', 'Price Per Share']

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv.DictWriter(csv_file, fieldnames=fieldnames).writeheader()

Base = sqlalchemy.orm.declarative_base()


class TradeHistory(Base):
    __tablename__ = 'trade_history'
    id = Column(Integer, primary_key=True)
    symbols = Column(String)
    action = Column(String)
    quantity = Column(Float)
    price = Column(Float)
    date = Column(String)


class Position(Base):
    __tablename__ = 'positions'
    symbols = Column(String, primary_key=True)
    quantity = Column(Float)
    avg_price = Column(Float)
    purchase_date = Column(String)
    # Item #1 (risk-per-trade): frozen entry-time ATR captured at buy. The
    # hard stop and position-sizing both use this value for the LIFE of the
    # position -- never recomputed from current market ATR. This prevents a
    # spike in volatility from silently widening an already-live stop.
    entry_atr = Column(Float)
    # Bull-market strategy tag: 'bull' for positions opened by the ported
    # Bull Market Advanced Stock Market Trading Robot v8 buy path; None (or
    # 'default') for the normal ML-scored path. sell_stocks branches on this
    # so bull-strategy positions exit at +0.5% flat instead of running the
    # profit monitor / scaled exit chain.
    strategy_tag = Column(String)


class TradeFeatures(Base):
    """
    REVIEW ITEM #7: record the features present at entry for every buy, plus
    the eventual outcome (filled in when the position closes), so trade
    history can be analyzed to see which feature combinations actually
    produced profitable trades rather than relying only on fixed assumptions.
    """
    __tablename__ = 'trade_features'
    id = Column(Integer, primary_key=True)
    symbols = Column(String)
    entry_date = Column(String)
    entry_price = Column(Float)
    rsi = Column(Float)
    macd_above_signal = Column(Integer)      # 0/1
    atr_pct = Column(Float)
    volume_holding = Column(Integer)         # 0/1
    candlestick_pattern = Column(String)
    buy_score = Column(Float)
    regime = Column(String)
    time_of_day = Column(String)
    exit_date = Column(String)               # filled on close
    exit_price = Column(Float)               # filled on close
    outcome_pct = Column(Float)              # filled on close: (exit-entry)/entry
    entry_atr = Column(Float)                # item #1: raw entry-time ATR (per-share dollars)


class AdaptiveParamState(Base):
    """
    Persisted current value of one auto-adjusted parameter, so a restart
    resumes from the last-learned state instead of resetting to the coded
    default. One row per (param_name, regime).
    """
    __tablename__ = 'adaptive_param_state'
    id = Column(Integer, primary_key=True)
    param_name = Column(String)   # e.g. 'buy_score_threshold'
    regime = Column(String)       # e.g. 'bull' / 'bear' / ... or 'global'
    value = Column(Float)
    updated_at = Column(String)


class AdaptiveParamLog(Base):
    """
    Audit trail: every automatic adjustment the bot makes to its own
    parameters, with the reasoning, so behavior changes are never silent.
    """
    __tablename__ = 'adaptive_param_log'
    id = Column(Integer, primary_key=True)
    timestamp = Column(String)
    param_name = Column(String)
    regime = Column(String)
    old_value = Column(Float)
    new_value = Column(Float)
    sample_size = Column(Integer)
    reason = Column(String)


# ---------------- Database ----------------
# BUGFIX: the path was relative ('sqlite:///trading_bot.db'), so the DB was
# created in whatever directory the program was launched from. Starting the bot
# from a different cwd silently opened a DIFFERENT, EMPTY database -- which
# looks exactly like "the .db stopped working after a restart". Anchor it to the
# script's own directory so it is always the same file.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_bot.db')
print(f"Using database: {DB_PATH}")

engine = create_engine(
    f'sqlite:///{DB_PATH}',
    connect_args={
        'check_same_thread': False,
        # BUGFIX: with two writer threads, lock contention raised
        # "database is locked" and the write was lost. Wait instead of failing.
        'timeout': 30,
    },
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _record):
    """
    BUGFIX: default journal mode gives poor durability and concurrency for a
    two-thread writer. WAL allows a reader alongside a writer and survives an
    ungraceful kill (e.g. Ctrl-C mid-write) without corrupting the file.
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")      # crash-safe, concurrent reads
    cur.execute("PRAGMA synchronous=FULL")      # fsync on commit; survives power loss
    cur.execute("PRAGMA busy_timeout=30000")    # wait 30s for locks, don't error
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


# BUGFIX: buy_stocks and sell_stocks run as concurrent threads and previously
# shared one module-level Session. SQLAlchemy Sessions are NOT thread-safe, and
# both threads call session.query() outside any lock, so this could corrupt the
# identity map or raise intermittent InvalidRequestError once positions existed.
# scoped_session hands each thread its own session behind the same API.
Session = scoped_session(sessionmaker(bind=engine))
session = Session
Base.metadata.create_all(engine)


def _migrate_add_entry_atr_columns():
    """SQLAlchemy's create_all only adds missing TABLES, never missing
    COLUMNS. Users upgrading from an earlier version of the bot will have
    a positions/trade_features table without the new entry_atr column. Add
    it in-place with plain ALTER TABLE ADD COLUMN -- SQLite supports this
    with default NULL, which is exactly what we want for old rows (old
    positions won't have a captured entry ATR available)."""
    with engine.connect() as conn:
        for tbl in ('positions', 'trade_features'):
            try:
                cols = [r[1] for r in conn.execute(sqlalchemy.text(f"PRAGMA table_info({tbl})")).fetchall()]
                if 'entry_atr' not in cols:
                    conn.execute(sqlalchemy.text(f"ALTER TABLE {tbl} ADD COLUMN entry_atr FLOAT"))
                    conn.commit()
                    logging.info(f"Migrated {tbl}: added entry_atr column.")
            except Exception as e:
                logging.warning(f"Migration check for {tbl}.entry_atr failed: {e}")

        # Bull-market strategy tag on positions. Old rows land NULL, which the
        # sell path treats as the default (non-bull) strategy.
        try:
            cols = [r[1] for r in conn.execute(sqlalchemy.text("PRAGMA table_info(positions)")).fetchall()]
            if 'strategy_tag' not in cols:
                conn.execute(sqlalchemy.text("ALTER TABLE positions ADD COLUMN strategy_tag VARCHAR"))
                conn.commit()
                logging.info("Migrated positions: added strategy_tag column.")
        except Exception as e:
            logging.warning(f"Migration check for positions.strategy_tag failed: {e}")


_migrate_add_entry_atr_columns()

data_cache = {}
# BUGFIX: data_cache is read AND written by both worker threads with no
# synchronization. Guard it.
_cache_lock = threading.Lock()
CACHE_EXPIRY = 120                 # default: intraday prices

# Tiered TTLs. Daily-bar data (200-day SMA, daily RSI, 22-period ATR) barely
# moves during a session, but was being refetched every cycle -- the single
# biggest consumer of the yfinance budget. 16 symbols x 5 requests = 80/cycle
# against a 55/min cap would throttle every cold pass. Caching daily series for
# 30 minutes keeps steady-state well under the cap.
CACHE_TTLS = {
    'current_price': 120,          # 2m  - needs to be fresh
    'atr': 1800,                   # 30m - 22-period daily ATR
    'uptrend': 1800,               # 30m - 200-day SMA
    'daily_rsi': 1800,             # 30m - 14-period daily RSI
    'history_90d': 900,            # 15m - daily candles for scoring
    'regime': 900,                 # 15m - VIX + SPY market regime classification
    'mtf_60m': 900,                # 15m - 60-minute intraday trend confirmation
    'mtf_5m': 300,                 # 5m  - 5-minute reversal confirmation
    'intraday_vwap': 60,           # 1m  - real session-anchored intraday VWAP (1-min OHLCV)
    'earnings_date': 21600,        # 6h  - next earnings date per symbol
    'relative_strength': 1800,     # 30m - RS vs SPY / sector proxy
}

CALLS = 60
PERIOD = 60

# ---------------- yfinance rate limiting ----------------
# yfinance guidance: 60 req/min (1/sec) is very safe, 120/min (2/sec) usually
# safe, plus a 0.5-1s delay BETWEEN requests. Batch where practical.
#
# BUGFIX: @limits creates a SEPARATE counter per decorated function -- they do
# NOT share a budget. Five different functions call yfinance
# (_fetch_current_price, _fetch_atr, is_in_uptrend, get_daily_rsi,
# calculate_technical_indicators), each decorated @limits(calls=60, period=60),
# which permitted 5 x 60 = 300 yfinance calls/minute. buy_stocks also called
# yf.Ticker(...).history() directly with NO limit at all.
#
# Every yfinance request now passes through ONE shared gate, so the cap is a real
# 60/min across the whole process regardless of which thread or function calls it.
YF_CALLS_PER_MIN = 55        # under the 60/min "very safe" guidance
YF_MIN_INTERVAL = 0.6        # seconds between requests, per yfinance guidance


class _YFGate:
    """
    Process-wide gate for yfinance. Thread-safe; shared by both workers.

    Enforces BOTH:
      - a rolling 60s ceiling (YF_CALLS_PER_MIN), and
      - a minimum spacing between consecutive requests (YF_MIN_INTERVAL).

    BUGFIX: the rolling window ALONE let all 55 requests fire in the same
    millisecond and then stall 60s (measured: 55 requests in 0.000s). The average
    is technically legal but that burst is exactly what triggers throttling --
    yfinance guidance asks for a 0.5-1s delay BETWEEN requests. Now enforced.
    """

    def __init__(self, calls_per_min, min_interval=0.0):
        self.capacity = calls_per_min
        self.min_interval = min_interval
        self._times = deque()
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        """Block until it is polite to issue the next yfinance request."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()

                since_last = now - self._last_call
                if self.min_interval and since_last < self.min_interval:
                    wait = self.min_interval - since_last          # spacing
                elif len(self._times) >= self.capacity:
                    wait = 60.0 - (now - self._times[0]) + 0.01    # window cap
                    logging.info(f"yfinance gate: at {self.capacity}/min cap, waiting {wait:.1f}s")
                else:
                    self._times.append(now)
                    self._last_call = now
                    return
            # Sleep OUTSIDE the lock so other threads can drain expired slots.
            time.sleep(min(wait, 5.0))

    def used(self):
        with self._lock:
            now = time.monotonic()
            while self._times and now - self._times[0] >= 60.0:
                self._times.popleft()
            return len(self._times)


yf_gate = _YFGate(YF_CALLS_PER_MIN, YF_MIN_INTERVAL)


def _yf_kwargs_to_alpaca_daily_days(kwargs):
    """Translate yfinance-style history kwargs to a plain 'give me N calendar
    days of daily bars' request that Alpaca can handle cleanly. Returns the
    integer day count, or None if the request shape isn't a good fit for
    Alpaca (intraday interval, prepost, custom start/end date pair, etc.).

    We only take the Alpaca path when the caller wants daily bars over a
    simple lookback -- that covers roughly 80% of the bot's yf_history calls
    (RSI/MACD windows, ATR window, uptrend check, prewarm, VIX, SPY regime).
    The remaining ~20% (intraday 1m/5m/60m, prepost, explicit date ranges)
    keep going to yfinance as before.
    """
    # Any intraday interval -> yfinance only.
    interval = kwargs.get('interval')
    if interval and interval != '1d':
        return None
    # Prepost is a yfinance-specific concept; Alpaca has separate feeds.
    if kwargs.get('prepost'):
        return None
    # Explicit start/end date range -> use it directly if both are dates.
    if kwargs.get('start') is not None or kwargs.get('end') is not None:
        return None  # keep on yfinance; caller wants exact date bounds
    # Only 'period' shorthand strings we support: NNd, Ny, ytd -> best-effort
    period = kwargs.get('period')
    if not period:
        return None
    p = str(period).strip().lower()
    if p == 'ytd':
        return (datetime.now().timetuple().tm_yday) + 5
    if p.endswith('d'):
        try:
            return int(p[:-1])
        except ValueError:
            return None
    if p.endswith('mo'):
        try:
            return int(p[:-2]) * 31
        except ValueError:
            return None
    if p.endswith('y'):
        try:
            return int(p[:-1]) * 365
        except ValueError:
            return None
    if p == 'max':
        return None  # let yfinance decide
    return None


def _alpaca_index_or_stock_daily_bars(symbol, days):
    """Wrapper around _alpaca_daily_bars_single that gracefully returns an
    empty DataFrame for symbols Alpaca doesn't serve (indices like ^VIX
    start with ^, which Alpaca doesn't have in its equity feed). Caller
    falls back to yfinance on empty."""
    if not symbol or symbol.startswith('^'):
        return pd.DataFrame()
    return _alpaca_daily_bars_single(symbol, days=days)


def yf_history(symbol, **kwargs):
    """
    The ONLY way this program should make a single-symbol daily-bar call.
    Routing every request through one function makes the shared cap enforceable.
    Prefer yf_download_batch() when fetching the same period for many symbols.

    Alpaca IEX PRIMARY (when the shape is daily bars over a plain lookback);
    yfinance FALLBACK (used directly for intraday, prepost, explicit dates,
    ^VIX / index symbols, and any Alpaca failure or short-return).
    """
    # ---- Primary: Alpaca IEX daily bars, if the request shape fits ----
    days = _yf_kwargs_to_alpaca_daily_days(kwargs)
    if days is not None:
        alp_df = _alpaca_index_or_stock_daily_bars(symbol, days)
        # A caller asking for e.g. 60d wants ~40+ trading bars back; short
        # returns fall back to yfinance so the caller isn't left with a
        # too-small window (talib.MACD needs 35, etc.). Conservative floor:
        # require at least ceil(days * 0.5) trading bars.
        min_bars = max(5, int(days * 0.5))
        if not alp_df.empty and len(alp_df) >= min_bars:
            return alp_df

    # ---- Fallback: yfinance (original path) ----
    yf_gate.acquire()
    with yf_lock:
        return yf.Ticker(to_yf(symbol)).history(**kwargs)


def _fetch_intraday_vwap(symbol):
    """Fetch today's 1-minute OHLCV bars and compute the REAL session-anchored
    VWAP:  Σ(TypicalPrice * Volume) / Σ(Volume),  cumulated from the first bar
    of the current regular-hours session. Returns the latest (current) VWAP as
    a float, or None if bars aren't available yet (pre-open, thin data, or
    yfinance error). Goes through yf_history() so it consumes the shared
    yfinance rate-limit budget just like every other single-symbol call."""
    try:
        df = yf_history(symbol, period='1d', interval='1m', prepost=False,
                        auto_adjust=False)
    except Exception as e:
        logging.debug(f"intraday_vwap: fetch failed for {symbol}: {e}")
        return None
    if df is None or df.empty:
        return None
    try:
        high = df['High'].astype(float).to_numpy()
        low = df['Low'].astype(float).to_numpy()
        close = df['Close'].astype(float).to_numpy()
        volume = df['Volume'].astype(float).to_numpy()
    except Exception:
        return None
    # Guard against all-NaN or all-zero-volume slices (rare, but possible in
    # the very first minute of the session before any prints).
    mask = ~(np.isnan(close) | np.isnan(volume))
    if not mask.any():
        return None
    high = high[mask]; low = low[mask]; close = close[mask]; volume = volume[mask]
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return None
    typical = (high + low + close) / 3.0
    vwap = float((typical * volume).sum() / total_volume)
    if not np.isfinite(vwap) or vwap <= 0:
        return None
    return vwap


def get_intraday_vwap(symbol):
    """Cached wrapper -- the sell/buy loops call this every tick, so the raw
    1-minute fetch is throttled to CACHE_TTLS['intraday_vwap'] (60s) per
    symbol. New minute bars only arrive once a minute anyway."""
    try:
        return get_cached_data(symbol, 'intraday_vwap', _fetch_intraday_vwap, symbol)
    except Exception as e:
        logging.debug(f"get_intraday_vwap({symbol}) cache path raised: {e}")
        return None


def _alpaca_daily_bars_batch(symbols, days):
    """Batched N-symbol Alpaca IEX daily-bar fetch. Returns
    {yf_symbol: DataFrame} in the same shape yf_download_batch produces
    (per-symbol Open/High/Low/Close/Volume DF with DatetimeIndex).

    Skips ^VIX and other index-style symbols (Alpaca doesn't serve them).
    Batches at 100 symbols per request, matching the proven scanner config.
    """
    if not symbols:
        return {}
    # Filter out index symbols Alpaca can't serve; caller falls back to
    # yfinance for those (VIX regime path).
    eligible = [s for s in symbols if s and not s.startswith('^')]
    if not eligible:
        return {}
    alpaca_syms = [to_alpaca(s) for s in eligible]
    yf_by_alpaca = dict(zip(alpaca_syms, eligible))
    end = datetime.now(pytz.UTC)
    start = end - timedelta(days=int(days) + 5)

    out = {}
    BATCH = 100
    for i in range(0, len(alpaca_syms), BATCH):
        chunk = alpaca_syms[i:i + BATCH]
        try:
            req = _AlpacaBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=_AlpacaTimeFrame.Day,
                start=start,
                end=end,
                feed=_AlpacaDataFeed.IEX,
            )
            resp = alpaca_data_client.get_stock_bars(req)
            data = resp.data if hasattr(resp, 'data') else resp
            for asym in chunk:
                bars = data.get(asym, []) if hasattr(data, 'get') else []
                if not bars:
                    continue
                yf_sym = yf_by_alpaca.get(asym, asym)
                try:
                    df = pd.DataFrame([{
                        'Open':   float(b.open),
                        'High':   float(b.high),
                        'Low':    float(b.low),
                        'Close':  float(b.close),
                        'Volume': float(b.volume),
                    } for b in bars],
                        index=pd.DatetimeIndex([b.timestamp for b in bars], name='Date'),
                    )
                    if not df.empty and not df['Close'].isna().all():
                        out[yf_sym] = df
                except Exception:
                    continue
        except Exception as e:
            logging.debug(f"[alpaca] batch daily bars fetch failed: {e}")
            continue
    return out


def yf_download_batch(symbols, **kwargs):
    """
    Batched multi-symbol daily-bar fetch.

    Alpaca IEX PRIMARY (when the request shape is daily bars); yfinance
    FALLBACK (intraday intervals, prepost, explicit dates, index symbols,
    or any Alpaca miss for a specific symbol).

    Returns {symbol: DataFrame}. Symbols that came back empty are omitted.
    Consumes ONE slot from the yfinance gate only if the fallback runs.
    """
    syms = [to_yf(s) for s in symbols]
    if not syms:
        return {}

    # ---- Primary: Alpaca IEX daily bars, if the request shape fits ----
    days = _yf_kwargs_to_alpaca_daily_days(kwargs)
    if days is not None:
        alp_out = _alpaca_daily_bars_batch(symbols, days)
        # Which symbols did Alpaca cover? If ALL of them, we're done and
        # yfinance is not touched. If PARTIAL, fall through to yfinance
        # for the missing ones only.
        missing = [s for s in symbols if s not in alp_out]
        if not missing:
            return alp_out
        # Partial coverage -- run yfinance only for the missing ones, then
        # merge results. This is the "12 partial-coverage ex-index tickers"
        # case observed in testing (Q, DFS, HES, PSKY, WBA, FI, IPG, K, MMC,
        # SNDK, HOLX, CTRA); they get filled in from yfinance.
        yf_gate.acquire()
        with yf_lock:
            raw = yf.download([to_yf(s) for s in missing], group_by='ticker',
                              progress=False, auto_adjust=False, threads=False,
                              **kwargs)
        for orig in missing:
            ys = to_yf(orig)
            try:
                df = raw[ys] if len(missing) > 1 else raw
                if df is not None and not df.empty and not df['Close'].isna().all():
                    alp_out[orig] = df.dropna(how='all')
            except (KeyError, TypeError):
                continue
        return alp_out

    # ---- Fallback: yfinance batched (original path) ----
    yf_gate.acquire()
    with yf_lock:
        raw = yf.download(syms, group_by='ticker', progress=False,
                          auto_adjust=False, threads=False, **kwargs)

    out = {}
    for orig, ys in zip(symbols, syms):
        try:
            # Multi-symbol returns column MultiIndex; single symbol returns flat.
            df = raw[ys] if len(syms) > 1 else raw
            if df is not None and not df.empty and not df['Close'].isna().all():
                out[orig] = df.dropna(how='all')
        except (KeyError, TypeError):
            continue
    return out


def prewarm_daily_cache(symbols):
    """
    Fetch a year of daily bars for every symbol in ONE batched request and seed
    the cache with the derived 200-day SMA, daily RSI and ATR.

    PERF: previously each symbol cost 3 separate yfinance requests for these
    (is_in_uptrend 1y + get_daily_rsi 60d + _fetch_atr 60d). For 16 symbols that
    was 48 requests; batched it is 1. All three are daily series, so one 1y pull
    serves all of them.
    """
    if not symbols:
        return

    # Skip entirely if every symbol still has warm daily entries -- otherwise the
    # batched call itself would waste a gate slot every cycle.
    now = time.time()
    ttl = CACHE_TTLS.get('uptrend', 1800)
    with _cache_lock:
        stale = [s for s in symbols
                 if now - data_cache.get((s, 'uptrend'), {}).get('timestamp', 0) >= ttl]
    if not stale:
        return

    try:
        batch = yf_download_batch(stale, period='1y', interval='1d')
    except Exception as e:
        logging.warning(f"Batched daily prewarm failed ({e}); falling back to per-symbol fetches.")
        return

    seeded = 0
    for sym, df in batch.items():
        try:
            close = df['Close'].values
            entries = {}

            if len(close) >= 200:
                sma = talib.SMA(close, timeperiod=200)[-1]
                if np.isfinite(sma):
                    entries['uptrend'] = float(sma)

            if len(close) >= 15:
                r = talib.RSI(close, timeperiod=14)[-1]
                if np.isfinite(r):
                    entries['daily_rsi'] = round(float(r), 2)

            if len(df) >= 23:
                atr = talib.ATR(df['High'].values, df['Low'].values, close, timeperiod=22)[-1]
                if np.isfinite(atr) and atr > 0:
                    entries['atr'] = float(atr)

            if len(df) >= 40:
                entries['history_90d'] = df.tail(90)

            with _cache_lock:
                for k, v in entries.items():
                    data_cache[(sym, k)] = {'timestamp': now, 'data': v}
            if entries:
                seeded += 1
        except Exception as e:
            logging.warning(f"Prewarm: could not derive indicators for {sym}: {e}")

    print(f"Prewarmed daily cache for {seeded}/{len(stale)} symbols in 1 batched request "
          f"(~{len(stale) * 3} individual requests avoided).")


# ---------------- Symbol helpers (BUGFIX: consistent normalization) ----------------
def to_yf(sym):
    """yfinance uses dashes for share classes: BRK.B -> BRK-B"""
    return sym.strip().upper().replace('.', '-')


def to_alpaca(sym):
    """Alpaca uses dots: BRK-B -> BRK.B"""
    return sym.strip().upper().replace('-', '.')


# BUGFIX: get_cached_data used to be @sleep_and_retry/@limits decorated. That was
# wrong twice over:
#   1. A CACHE HIT -- which makes no network call at all -- still consumed a slot
#      from the shared 60/min budget.
#   2. It NESTED: get_current_price -> get_cached_data -> _fetch_current_price,
#      all three rate-limited, so ONE price lookup burned THREE slots. With ~16
#      symbols the budget was exhausted mid-cycle, and sleep_and_retry then SLEEPS
#      THE CALLING THREAD for up to a full 60s window -- while sell_stocks held a
#      per-symbol claim, locking buy_stocks out of that symbol the entire time.
#      (Verified: 70 calls against the real limiter block for exactly 60.0s.)
# The cache layer does no I/O, so it is no longer rate-limited. Only the real
# network fetchers below are.
def get_cached_data(symbols, data_type, fetch_func, *args, **kwargs):
    key = (symbols, data_type)
    ttl = CACHE_TTLS.get(data_type, CACHE_EXPIRY)
    now = time.time()

    with _cache_lock:
        entry = data_cache.get(key)
        if entry and now - entry['timestamp'] < ttl:
            return entry['data']

    # Fetch OUTSIDE the cache lock: fetch_func is rate-limited and can block for
    # a full window. Holding _cache_lock across it would stall every cache reader
    # in both threads.
    data = fetch_func(*args, **kwargs)

    with _cache_lock:
        data_cache[key] = {'timestamp': time.time(), 'data': data}
    return data


# ---------------- Market regime detection ----------------
# Classifies the overall market as Bull / Sideways / Bear / Panic using VIX
# level plus SPY's position relative to its 20/50-day SMAs. Buy-score signal
# weights and the buy threshold both key off this classification, per the
# review's recommendation to stop treating every signal as equally important
# in every market.
REGIME_BULL = 'bull'
REGIME_SIDEWAYS = 'sideways'
REGIME_BEAR = 'bear'
REGIME_PANIC = 'panic'

# VIX thresholds (approximate historical bands)
VIX_PANIC_LEVEL = 30.0
VIX_ELEVATED_LEVEL = 20.0

# Per-regime signal weights. Keys must match the boolean/points computed in
# compute_buy_score. Unlisted keys default to weight 1.
REGIME_SIGNAL_WEIGHTS = {
    REGIME_BULL: {
        'pattern': 1, 'rsi_below_50': 1, 'rsi_falling': 1, 'volume_holding': 1,
        'macd_above_signal': 1, 'price_decline': 1, 'pattern_bonus': 1,
        'price_stable': 1, 'trend': 3, 'intraday_pullback': 1,
        'pullback_5m': 1, 'pullback_30m': 1, 'swing_high_distance': 1, 'vwap_distance': 1,
    },
    REGIME_SIDEWAYS: {
        'pattern': 2, 'rsi_below_50': 1, 'rsi_falling': 1, 'volume_holding': 1,
        'macd_above_signal': 1, 'price_decline': 1, 'pattern_bonus': 1,
        'price_stable': 1, 'trend': 1, 'intraday_pullback': 1,
        'pullback_5m': 1, 'pullback_30m': 1, 'swing_high_distance': 1, 'vwap_distance': 1,
    },
    REGIME_BEAR: {
        'pattern': 2, 'rsi_below_50': 3, 'rsi_falling': 1, 'volume_holding': 2,
        'macd_above_signal': 3, 'price_decline': 1, 'pattern_bonus': 1,
        'price_stable': 1, 'trend': 1, 'intraday_pullback': 2,
        'pullback_5m': 2, 'pullback_30m': 2, 'swing_high_distance': 2, 'vwap_distance': 2,
    },
    REGIME_PANIC: {
        'pattern': 1, 'rsi_below_50': 3, 'rsi_falling': 1, 'volume_holding': 2,
        'macd_above_signal': 3, 'price_decline': 1, 'pattern_bonus': 1,
        'price_stable': 1, 'trend': 1, 'intraday_pullback': 2,
        'pullback_5m': 2, 'pullback_30m': 2, 'swing_high_distance': 2, 'vwap_distance': 2,
    },
}

# Dynamic buy-score threshold per regime (feature #6 in the review). Higher
# volatility / more danger -> require stronger confirmation before buying.
# These are the STARTING values only. The live values the bot actually trades
# with are auto-adjusted within bounds by AdaptiveParams (see below) and
# persisted to the database, so a restart resumes from the last-learned state
# instead of snapping back to these defaults.
REGIME_BUY_THRESHOLDS_DEFAULT = {
    REGIME_BULL: 3,
    REGIME_SIDEWAYS: 4,
    REGIME_BEAR: 5,
    REGIME_PANIC: 6,
}
# Hard bounds the auto-adjuster can never move outside of, regardless of what
# the trade history seems to suggest. Keeps a run of lucky/unlucky trades from
# ever turning the bot reckless (threshold too low) or completely inert
# (threshold too high).
REGIME_BUY_THRESHOLD_BOUNDS = {
    REGIME_BULL: (2, 5),
    REGIME_SIDEWAYS: (3, 6),
    REGIME_BEAR: (4, 8),
    REGIME_PANIC: (5, 9),
}


def _fetch_market_regime():
    """
    Classify the market using ^VIX level and SPY's close relative to its
    20-day and 50-day SMA. Cached for 15 minutes (CACHE_TTLS['regime']) since
    regime does not flip minute-to-minute.
    """
    try:
        vix_hist = yf_history('^VIX', period='5d', interval='1d')
        vix_level = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else None
    except Exception as e:
        logging.warning(f"Regime: VIX fetch failed: {e}")
        vix_level = None

    try:
        spy_hist = yf_history('SPY', period='90d', interval='1d')
        if spy_hist.empty or len(spy_hist) < 55:
            spy_close = spy_sma20 = spy_sma50 = None
        else:
            close = spy_hist['Close'].values
            spy_close = float(close[-1])
            spy_sma20 = float(talib.SMA(close, timeperiod=20)[-1])
            spy_sma50 = float(talib.SMA(close, timeperiod=50)[-1])
    except Exception as e:
        logging.warning(f"Regime: SPY fetch failed: {e}")
        spy_close = spy_sma20 = spy_sma50 = None

    # Panic overrides everything: extreme VIX means bear-style caution
    # regardless of where SPY sits relative to its averages.
    if vix_level is not None and vix_level >= VIX_PANIC_LEVEL:
        regime = REGIME_PANIC
    elif spy_close is not None and spy_sma20 is not None and spy_sma50 is not None:
        if spy_close > spy_sma20 > spy_sma50:
            regime = REGIME_BULL
        elif spy_close < spy_sma20 < spy_sma50:
            regime = REGIME_BEAR
        else:
            regime = REGIME_SIDEWAYS
        # Elevated VIX in a non-bull SPY posture still tightens things up.
        if vix_level is not None and vix_level >= VIX_ELEVATED_LEVEL and regime != REGIME_BULL:
            regime = REGIME_BEAR
    else:
        # Missing data: default to the conservative middle ground rather than
        # silently trading as if conditions were calm.
        regime = REGIME_SIDEWAYS

    return {
        'regime': regime, 'vix': vix_level,
        'spy_close': spy_close, 'spy_sma20': spy_sma20, 'spy_sma50': spy_sma50,
    }


# ---------------- Auto-adapting parameters with safety guardrails ----------------
# Point-based auto-adjuster for BUY_SCORE_THRESHOLD-per-regime and the
# per-regime signal weights. Every ADAPT_EVERY_N_CYCLES, this scores each
# regime's recent closed trades and nudges parameters toward what's working --
# automatically, no human approval step -- but ONLY within hard guardrails:
#
#   1. MIN SAMPLE SIZE: a regime with too few closed trades is left alone.
#      Small samples are how you get a threshold that "learned" from 3 lucky
#      trades. No sample, no move.
#   2. MAX STEP SIZE: every adjustment moves a parameter by at most one small
#      step per cycle. The bot drifts toward better settings over many
#      windows; it can never jump to an extreme in one adjustment.
#   3. HARD BOUNDS: REGIME_BUY_THRESHOLD_BOUNDS / weight bounds below are
#      ceilings and floors the adjuster can never cross, no matter how
#      strongly the data seems to argue for it.
#   4. FULL AUDIT LOG: every adjustment (or decision NOT to adjust) is written
#      to AdaptiveParamLog with the sample size and reasoning, and printed to
#      the console. Nothing changes silently.
#   5. PERSISTED, NOT RESET: current values live in AdaptiveParamState, so a
#      restart resumes from the last-learned value instead of the coded
#      default -- but also means a bad drift persists until corrected, which
#      is exactly why 1-4 exist.
ADAPT_EVERY_N_CYCLES = 60             # ~1 hour at 60s/cycle
ADAPT_MIN_TRADES_PER_REGIME = 15      # guardrail #1: floor on sample size
ADAPT_MIN_WIN_RATE_SAMPLE = 8         # per-bucket floor when scoring weight point-deltas
THRESHOLD_STEP = 1                    # guardrail #2: max threshold move per cycle
WEIGHT_STEP = 1                       # guardrail #2: max per-signal weight move per cycle
WEIGHT_BOUNDS = (1, 4)                # guardrail #3 for signal weights (pattern, rsi, macd, etc.)
# A regime needs a clearly better/worse out-of-sample result -- not noise --
# before the threshold moves at all. Expressed as a minimum gap in average
# outcome_pct between "trades that would have passed a stricter/looser bar".
ADAPT_MIN_EDGE_PCT = 0.0015           # 0.15 percentage points of average return


class AdaptiveParams:
    """
    Thread-safe, DB-persisted store for the live values of auto-adjusted
    parameters. Reads are cheap (in-memory dict behind a lock); writes go
    through `_persist` and are logged via AdaptiveParamLog.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thresholds = dict(REGIME_BUY_THRESHOLDS_DEFAULT)
        self._weights = {r: dict(w) for r, w in REGIME_SIGNAL_WEIGHTS.items()}
        self._loaded = False

    # ---- loading / persistence ----
    def load_from_db(self):
        """Resume from the last-learned state on startup, if any exists."""
        with self._lock:
            if self._loaded:
                return
            rows = session.query(AdaptiveParamState).all()
            for row in rows:
                if row.param_name == 'buy_score_threshold' and row.regime in self._thresholds:
                    self._thresholds[row.regime] = row.value
                elif row.param_name.startswith('weight:'):
                    signal_key = row.param_name.split('weight:', 1)[1]
                    if row.regime in self._weights:
                        self._weights[row.regime][signal_key] = row.value
            self._loaded = True
            if rows:
                print(f"AdaptiveParams: resumed {len(rows)} persisted parameter values from prior runs.")

    def _persist(self, param_name, regime, value):
        row = (session.query(AdaptiveParamState)
               .filter_by(param_name=param_name, regime=regime).one_or_none())
        now_str = datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S")
        if row is None:
            row = AdaptiveParamState(param_name=param_name, regime=regime,
                                     value=value, updated_at=now_str)
            session.add(row)
        else:
            row.value = value
            row.updated_at = now_str

    def _log(self, param_name, regime, old_value, new_value, sample_size, reason):
        now_str = datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S")
        session.add(AdaptiveParamLog(
            timestamp=now_str, param_name=param_name, regime=regime,
            old_value=old_value, new_value=new_value,
            sample_size=sample_size, reason=reason,
        ))
        arrow = '->' if old_value != new_value else '(unchanged)'
        print(f"  [adapt] {param_name} [{regime}]: {old_value} {arrow} {new_value}  "
              f"(n={sample_size}) {reason}")

    # ---- reads (used by live trading code every cycle) ----
    def get_threshold(self, regime):
        with self._lock:
            return self._thresholds.get(regime, BUY_SCORE_THRESHOLD_DEFAULT)

    def get_weights(self, regime):
        with self._lock:
            return dict(self._weights.get(regime, self._weights[REGIME_SIDEWAYS]))

    # ---- the adjustment pass itself ----
    def run_adjustment_pass(self):
        """
        Called periodically from the main loop. For each regime with enough
        closed trades (guardrail #1), compares outcomes above vs. below the
        current threshold and nudges the threshold by at most THRESHOLD_STEP
        (guardrail #2) toward whichever side has the better average outcome,
        clamped to REGIME_BUY_THRESHOLD_BOUNDS (guardrail #3). Every decision
        -- move or no-move -- is logged (guardrail #4).
        """
        all_rows = (session.query(TradeFeatures)
                    .filter(TradeFeatures.outcome_pct.isnot(None))
                    .all())
        by_regime = {}
        for r in all_rows:
            by_regime.setdefault(r.regime or REGIME_SIDEWAYS, []).append(r)

        print("\n--- Adaptive Parameter Pass (auto-applies within guardrails) ---")
        with self._lock:
            for regime in (REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR, REGIME_PANIC):
                rows = by_regime.get(regime, [])
                self._adjust_threshold_for_regime(regime, rows)
                self._adjust_weights_for_regime(regime, rows)
        session.commit()
        print("--- End adaptive parameter pass ---\n")

    def _adjust_threshold_for_regime(self, regime, rows):
        n = len(rows)
        lo, hi = REGIME_BUY_THRESHOLD_BOUNDS[regime]
        current = self._thresholds[regime]

        if n < ADAPT_MIN_TRADES_PER_REGIME:
            self._log('buy_score_threshold', regime, current, current, n,
                      f"below min sample ({ADAPT_MIN_TRADES_PER_REGIME}); holding steady.")
            return

        # Point-based comparison: trades that scored AT the current threshold
        # vs. trades that scored one point ABOVE it. If the higher bar clearly
        # outperforms, tighten (raise threshold). If trades right at the
        # current bar do just as well or better than stricter ones, and the
        # bar is already above its floor, loosen it one point to trade more.
        at_bar = [r.outcome_pct for r in rows if r.buy_score is not None
                  and current <= r.buy_score < current + 1]
        above_bar = [r.outcome_pct for r in rows if r.buy_score is not None
                    and r.buy_score >= current + 1]

        new_value = current
        reason = "no clear edge either direction; holding steady."

        if len(above_bar) >= ADAPT_MIN_WIN_RATE_SAMPLE and len(at_bar) >= ADAPT_MIN_WIN_RATE_SAMPLE:
            edge = float(np.mean(above_bar)) - float(np.mean(at_bar))
            if edge >= ADAPT_MIN_EDGE_PCT and current + THRESHOLD_STEP <= hi:
                new_value = current + THRESHOLD_STEP
                reason = (f"scores >= {current+1} outperformed scores at {current} by "
                          f"{edge*100:+.2f}pp; tightening.")
            elif edge <= -ADAPT_MIN_EDGE_PCT and current - THRESHOLD_STEP >= lo:
                new_value = current - THRESHOLD_STEP
                reason = (f"scores at {current} outperformed scores >= {current+1} by "
                          f"{-edge*100:+.2f}pp; loosening to trade more of what's working.")
            else:
                reason = f"edge {edge*100:+.2f}pp within noise band (±{ADAPT_MIN_EDGE_PCT*100:.2f}pp); holding."
        else:
            reason = (f"not enough trades on both sides of the bar "
                      f"(at={len(at_bar)}, above={len(above_bar)}, need {ADAPT_MIN_WIN_RATE_SAMPLE}+ each); holding.")

        if new_value != current:
            self._thresholds[regime] = new_value
            self._persist('buy_score_threshold', regime, new_value)
        self._log('buy_score_threshold', regime, current, new_value, n, reason)

    def _adjust_weights_for_regime(self, regime, rows):
        """
        Point-based weight adjustment: for each signal key present in this
        regime's weight table, compare the average outcome of trades where
        that signal fired vs. trades where it didn't. A clearly-helpful
        signal's weight nudges up by WEIGHT_STEP; a clearly-unhelpful one
        nudges down. Both clamped to WEIGHT_BOUNDS. Every signal needs its
        own minimum sample on both sides (fired/not-fired) before it moves.
        """
        n = len(rows)
        if n < ADAPT_MIN_TRADES_PER_REGIME:
            return  # already logged by the threshold check above for this regime

        weights = self._weights[regime]
        # Only these signals are inferable from what's stored per-trade today
        # (TradeFeatures doesn't currently break out every raw sub-signal --
        # macd and pattern presence are the ones we can score point-by-point).
        signal_checks = {
            'macd_above_signal': lambda r: bool(r.macd_above_signal),
            'volume_holding': lambda r: bool(r.volume_holding),
        }
        for signal_key, predicate in signal_checks.items():
            fired = [r.outcome_pct for r in rows if predicate(r)]
            not_fired = [r.outcome_pct for r in rows if not predicate(r)]
            current_w = weights.get(signal_key, 1)
            lo, hi = WEIGHT_BOUNDS

            if len(fired) < ADAPT_MIN_WIN_RATE_SAMPLE or len(not_fired) < ADAPT_MIN_WIN_RATE_SAMPLE:
                continue  # not enough trades on both sides; leave this weight alone

            edge = float(np.mean(fired)) - float(np.mean(not_fired))
            new_w = current_w
            reason = f"edge {edge*100:+.2f}pp within noise band; holding."
            if edge >= ADAPT_MIN_EDGE_PCT and current_w + WEIGHT_STEP <= hi:
                new_w = current_w + WEIGHT_STEP
                reason = f"trades with {signal_key} outperformed by {edge*100:+.2f}pp; raising weight."
            elif edge <= -ADAPT_MIN_EDGE_PCT and current_w - WEIGHT_STEP >= lo:
                new_w = current_w - WEIGHT_STEP
                reason = f"trades with {signal_key} underperformed by {-edge*100:+.2f}pp; lowering weight."

            if new_w != current_w:
                weights[signal_key] = new_w
                self._persist(f'weight:{signal_key}', regime, new_w)
            self._log(f'weight:{signal_key}', regime, current_w, new_w, n, reason)


adaptive_params = AdaptiveParams()


# ---------------- Weak-bull regime downgrade ----------------
# If the SPY+VIX detector says BULL but the bull-buy window has been open for
# ~30 minutes (10:05-10:35 AM ET) with zero fills today, treat the tape as
# "weak bullish" and downgrade the regime to SIDEWAYS for the REST of the
# trading day. Rationale: a real bull tape produces at least one qualifying
# candidate within that half-hour; if nothing has cleared the multi-timeframe
# / RSI / bull-gate stack by then, the day is likely not going to be a
# textbook bull day and the tighter sideways playbook (higher score
# threshold, sideways weights, bull-path disabled) is a better fit than
# sitting out on unmet bull thresholds.
#
# Once downgraded, the flag sticks until the ET date rolls over.
WEAK_BULL_DOWNGRADE_CUTOFF = (10, 35)  # (hour, minute) ET -- decision time
_weak_bull_downgrade_date = None       # 'YYYY-MM-DD' the downgrade is locked in for


def _bought_anything_today_et():
    """Return True if TradeHistory has any 'buy' row with today's ET date."""
    today = datetime.now(eastern).date().strftime("%Y-%m-%d")
    session = Session()
    try:
        row = session.query(TradeHistory).filter(
            TradeHistory.action == 'buy',
            TradeHistory.date == today,
        ).first()
        return row is not None
    except Exception as e:
        logging.warning(f"Weak-bull downgrade: TradeHistory query failed ({e}); "
                        "assuming buys HAVE happened (skips downgrade).")
        return True
    finally:
        session.close()


def _apply_weak_bull_downgrade(regime_info):
    """
    If regime is BULL and we're past 10:35 AM ET with no fills today,
    downgrade to SIDEWAYS for the rest of the day. Idempotent: once the
    date-scoped flag is set, this always downgrades until date changes.
    """
    global _weak_bull_downgrade_date
    if regime_info.get('regime') != REGIME_BULL:
        return regime_info

    today = datetime.now(eastern).date().strftime("%Y-%m-%d")

    # Already downgraded earlier today? Keep it downgraded.
    if _weak_bull_downgrade_date == today:
        info = dict(regime_info)
        info['regime'] = REGIME_SIDEWAYS
        info['weak_bull_downgraded'] = True
        return info

    now_et = datetime.now(eastern)
    cutoff = now_et.replace(hour=WEAK_BULL_DOWNGRADE_CUTOFF[0],
                            minute=WEAK_BULL_DOWNGRADE_CUTOFF[1],
                            second=0, microsecond=0)
    if now_et < cutoff:
        return regime_info

    if _bought_anything_today_et():
        return regime_info

    # Lock in the downgrade for the rest of the day.
    _weak_bull_downgrade_date = today
    print(f"Weak-bull downgrade engaged at {now_et.strftime('%H:%M:%S')} ET: "
          f"BULL regime detected but no buys since bull window opened at "
          f"{BULL_BUY_WINDOW_START[0]:02d}:{BULL_BUY_WINDOW_START[1]:02d}. "
          f"Treating tape as SIDEWAYS for the rest of the day.")
    logging.info(f"Weak-bull downgrade engaged for {today}: BULL -> SIDEWAYS "
                 f"(no fills by {WEAK_BULL_DOWNGRADE_CUTOFF[0]:02d}:"
                 f"{WEAK_BULL_DOWNGRADE_CUTOFF[1]:02d} ET).")
    info = dict(regime_info)
    info['regime'] = REGIME_SIDEWAYS
    info['weak_bull_downgraded'] = True
    return info


def get_market_regime():
    raw = get_cached_data('MARKET', 'regime', _fetch_market_regime)
    return _apply_weak_bull_downgrade(raw)


def get_buy_score_threshold(regime=None):
    regime = regime or get_market_regime()['regime']
    return adaptive_params.get_threshold(regime)


def get_regime_weights(regime=None):
    regime = regime or get_market_regime()['regime']
    return adaptive_params.get_weights(regime)


_moo_sweep_lock = threading.Lock()
_moo_sweep_last_run_date = None   # 'YYYY-MM-DD' of the last date the AM sweep ran
_close_sweep_lock = threading.Lock()
_close_sweep_last_run_date = None  # 'YYYY-MM-DD' of the last date the PM sweep ran

# ---------------- Escalation chain config (both sweeps share this) ----------------
# Step 1: MOO (pre-market) or plain market order (pre-close) is submitted.
# Step 2: if it isn't confirmed/filled within ESCALATION_STEP1_TIMEOUT_SECS,
#         cancel it and submit an aggressive limit order at (or near) the
#         current bid/last price.
# Step 3: if THAT isn't filled within ESCALATION_STEP2_TIMEOUT_SECS, cancel it
#         and submit a plain market order, which will fill at whatever the
#         prevailing price is. This step is what makes the sweep a "sell no
#         matter what" rather than a "try to sell" -- the only things that can
#         still stop it are a trading halt or the broker being unreachable.
ESCALATION_STEP1_TIMEOUT_SECS = 90    # time to wait for the MOO/initial order
ESCALATION_STEP2_TIMEOUT_SECS = 60    # time to wait for the fallback limit order
LIMIT_FALLBACK_DISCOUNT_PCT = 0.0     # 0.0 = limit AT current bid/last price, no discount

# Scale-out stage orders are plain market orders during regular session hours
# and should confirm within a few seconds -- a much shorter budget than the
# sweep escalation timeouts above, so sell_stocks() isn't blocked for up to
# 90s per symbol per cycle while iterating many positions.
SCALE_OUT_FILL_TIMEOUT_SECS = 15


def _in_time_window(now, hour, minute, window_minutes=4):
    """True from hour:minute through the next window_minutes Eastern."""
    start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return start <= now < start + timedelta(minutes=window_minutes)


def _in_moo_sweep_window(now):
    """
    True from MOO_SWEEP_HOUR:MOO_SWEEP_MINUTE through MOO_SWEEP_CUTOFF_SECS
    later (default 9:25:00-9:27:30 Eastern). Alpaca rejects OPG (market-on-
    open) orders submitted after approximately 9:28am ET, so this window ends
    well before that cutoff -- with margin for network/broker latency between
    when the bot decides to submit and when Alpaca actually receives it, plus
    tolerance for the wait loop's 60s tick occasionally running slow. The
    once-per-day guard in run_premarket_profit_sweep() still ensures it only
    fires once even though the window spans multiple loop ticks.
    """
    start = now.replace(hour=MOO_SWEEP_HOUR, minute=MOO_SWEEP_MINUTE, second=0, microsecond=0)
    return start <= now < start + timedelta(seconds=MOO_SWEEP_CUTOFF_SECS)


def _in_close_sweep_window(now):
    """True in the window around CLOSE_SWEEP_HOUR:CLOSE_SWEEP_MINUTE Eastern
    (default 3:45pm, i.e. 15 minutes before the 4:00pm close)."""
    return _in_time_window(now, CLOSE_SWEEP_HOUR, CLOSE_SWEEP_MINUTE)


def _get_bid_or_last_price(symbol, fallback_price):
    """Best-effort current bid for the aggressive fallback limit price; falls
    back to the last-known price (from the position or a fresh quote) if a
    live quote isn't available."""
    try:
        quote = api.get_latest_quote(symbol)
        bid = float(getattr(quote, 'bid_price', 0) or 0)
        if bid > 0:
            return bid
    except Exception as e:
        logging.info(f"{symbol}: latest-quote lookup failed ({e}); using fallback price.")
    return fallback_price


def _poll_order_terminal(order_id, timeout_secs, poll_every=2):
    """
    Poll an order until it reaches a terminal state (filled/canceled/expired/
    rejected) or timeout_secs elapses. Returns (terminal: bool, filled_qty,
    filled_price, status). Mirrors the polling pattern already used for buy
    fills in buy_stocks().
    """
    filled_qty, filled_price, status = 0.0, None, 'unknown'
    elapsed = 0
    while elapsed < timeout_secs:
        try:
            o = api.get_order(order_id)
        except Exception as e:
            logging.warning(f"Order {order_id}: poll error ({e}); retrying.")
            time.sleep(poll_every)
            elapsed += poll_every
            continue

        filled_qty = float(o.filled_qty or 0)
        if o.filled_avg_price:
            filled_price = float(o.filled_avg_price)
        status = o.status

        if status == 'filled':
            return True, filled_qty, filled_price, status
        if status in ('canceled', 'expired', 'rejected'):
            return True, filled_qty, filled_price, status

        time.sleep(poll_every)
        elapsed += poll_every

    return False, filled_qty, filled_price, status


def _cancel_existing_sell_orders(symbol, sweep_label, now_str):
    """
    Cancel any pre-existing open SELL orders for `symbol` (e.g. a resting
    stop-loss or GTC limit placed earlier by the bot's normal intraday exit
    logic) so the profit sweeps always take full, unencumbered ownership of
    the position instead of only working around whatever quantity those
    orders already cover. Always cancels -- per instruction, the sweep should
    always control the whole position rather than leave a fraction tied up in
    an older order, regardless of that order's price.

    Returns the number of orders cancelled (0 if none existed or cancel
    failed -- a failure here is logged but does not block the sweep from
    still trying to sell, since list_orders will simply keep counting that
    quantity as "already selling" if the cancel didn't go through).
    """
    try:
        open_orders = api.list_orders(status='open', symbols=[symbol])
    except Exception as e:
        logging.warning(f"{symbol}: [{sweep_label}] list_orders failed while checking for "
                        f"pre-existing sell orders ({e}); proceeding without cancelling.")
        return 0

    sell_orders = [o for o in open_orders if o.side == 'sell']
    cancelled = 0
    for o in sell_orders:
        try:
            api.cancel_order(o.id)
            cancelled += 1
            print(f"{symbol}: [{sweep_label}] cancelled pre-existing open sell order "
                  f"{o.id} ({o.qty} sh) so the sweep can own the full position.")
            logging.info(f"{now_str} [{sweep_label}] {symbol}: cancelled pre-existing sell "
                        f"order {o.id} ({o.qty} sh).")
        except Exception as e:
            print(f"{symbol}: [{sweep_label}] failed to cancel pre-existing sell order "
                  f"{o.id}: {e}. That quantity may remain tied up.")
            logging.warning(f"{now_str} [{sweep_label}] {symbol}: cancel of pre-existing "
                            f"sell order {o.id} failed: {e}")

    if cancelled:
        # Give the broker a beat to process the cancellations before we re-check
        # qty and submit the sweep's own order, so we don't race a cancel that
        # hasn't landed yet.
        time.sleep(1)

    return cancelled


def _sell_with_escalation(symbol, qty, last_known_price, step1_type, step1_tif,
                          sweep_label, now_str):
    """
    Shared 3-step escalation: step1 (MOO for the AM sweep, plain market for
    the PM sweep) -> aggressive limit at bid/last -> plain market order that
    fills at whatever the prevailing price is. Returns (total_filled_qty,
    total_notional, steps_used: list[str]) for logging/summary purposes.

    Each step only tries to sell whatever quantity is STILL unfilled from the
    previous step, so a partial fill at any stage is never double-sold.
    """
    remaining_qty = qty
    total_filled_qty = 0.0
    total_notional = 0.0
    steps_used = []

    def _log_fill(step_name, fq, fp):
        nonlocal total_filled_qty, total_notional
        if fq <= 0:
            return
        px = fp if fp else last_known_price
        total_filled_qty += fq
        total_notional += fq * px
        steps_used.append(f"{step_name}:{fq:.4f}sh@${px:.2f}")
        with open(csv_filename, mode='a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                'Date': now_str, 'Buy': '', 'Sell': f"Sell ({sweep_label} {step_name})",
                'Quantity': fq, 'Symbol': symbol, 'Price Per Share': px,
            })

    # ---- Step 1: MOO (AM) or plain market (PM) ----
    # Defense-in-depth against Alpaca's real ~9:28am ET OPG cutoff: even
    # though the window check in _in_moo_sweep_window() already stops NEW
    # sweep runs after MOO_SWEEP_CUTOFF_SECS, a sweep already in progress
    # (e.g. slow on an earlier symbol in a long position list) could still
    # reach this point after the cutoff. If so, skip straight past the OPG
    # attempt -- which Alpaca would reject anyway -- to the limit fallback,
    # rather than wasting a submit/reject round trip this close to the open.
    skip_step1_late_opg = (
        step1_tif == 'opg'
        and datetime.now(eastern).replace(second=0, microsecond=0)
        >= datetime.now(eastern).replace(hour=MOO_SWEEP_HOUR, minute=MOO_SWEEP_MINUTE,
                                         second=0, microsecond=0) + timedelta(seconds=MOO_SWEEP_CUTOFF_SECS)
    )
    if skip_step1_late_opg:
        print(f"{symbol}: [{sweep_label}] past the safe OPG submission window "
              f"(cutoff ~9:28am ET); skipping straight to the limit fallback.")
    else:
        try:
            order = api.submit_order(symbol=symbol, qty=str(remaining_qty), side='sell',
                                     type=step1_type, time_in_force=step1_tif)
            print(f"{symbol}: [{sweep_label}] step 1 ({step1_type}/{step1_tif}) submitted "
                  f"for {remaining_qty:.4f} sh (order {getattr(order, 'id', 'n/a')}).")
            terminal, fq, fp, status = _poll_order_terminal(order.id, ESCALATION_STEP1_TIMEOUT_SECS)
            _log_fill('step1', fq, fp)
            remaining_qty = round(remaining_qty - fq, 6)
            if not terminal:
                print(f"{symbol}: [{sweep_label}] step 1 order not terminal after "
                      f"{ESCALATION_STEP1_TIMEOUT_SECS}s (status={status}); cancelling and escalating.")
                try:
                    api.cancel_order(order.id)
                    time.sleep(1)
                    # Pick up anything that filled in the instant before cancel landed.
                    o = api.get_order(order.id)
                    extra_fq = max(0.0, float(o.filled_qty or 0) - fq)
                    if extra_fq > 0:
                        _log_fill('step1_late', extra_fq, float(o.filled_avg_price) if o.filled_avg_price else fp)
                        remaining_qty = round(remaining_qty - extra_fq, 6)
                except Exception as e:
                    logging.warning(f"{symbol}: [{sweep_label}] step 1 cancel/recheck failed: {e}")
            elif status in ('canceled', 'expired', 'rejected') and fq == 0:
                print(f"{symbol}: [{sweep_label}] step 1 order {status} with no fill; escalating.")
        except Exception as e:
            print(f"{symbol}: [{sweep_label}] step 1 submit failed ({e}); escalating to limit fallback.")
            logging.error(f"{now_str} [{sweep_label}] {symbol} step 1 submit failed: {e}")

    if remaining_qty <= 0:
        return total_filled_qty, total_notional, steps_used

    # ---- Step 2: aggressive limit at bid/last ----
    limit_price = _get_bid_or_last_price(symbol, last_known_price)
    if LIMIT_FALLBACK_DISCOUNT_PCT > 0:
        limit_price = round(limit_price * (1 - LIMIT_FALLBACK_DISCOUNT_PCT), 2)
    try:
        order2 = api.submit_order(symbol=symbol, qty=str(remaining_qty), side='sell',
                                  type='limit', limit_price=str(round(limit_price, 2)),
                                  time_in_force='day')
        print(f"{symbol}: [{sweep_label}] step 2 (limit @ ${limit_price:.2f}) submitted "
              f"for {remaining_qty:.4f} sh (order {getattr(order2, 'id', 'n/a')}).")
        terminal2, fq2, fp2, status2 = _poll_order_terminal(order2.id, ESCALATION_STEP2_TIMEOUT_SECS)
        _log_fill('step2', fq2, fp2)
        remaining_qty = round(remaining_qty - fq2, 6)
        if not terminal2:
            print(f"{symbol}: [{sweep_label}] step 2 limit not terminal after "
                  f"{ESCALATION_STEP2_TIMEOUT_SECS}s (status={status2}); cancelling and escalating to market.")
            try:
                api.cancel_order(order2.id)
                time.sleep(1)
                o2 = api.get_order(order2.id)
                extra_fq2 = max(0.0, float(o2.filled_qty or 0) - fq2)
                if extra_fq2 > 0:
                    _log_fill('step2_late', extra_fq2, float(o2.filled_avg_price) if o2.filled_avg_price else fp2)
                    remaining_qty = round(remaining_qty - extra_fq2, 6)
            except Exception as e:
                logging.warning(f"{symbol}: [{sweep_label}] step 2 cancel/recheck failed: {e}")
        elif status2 in ('canceled', 'expired', 'rejected') and fq2 == 0:
            print(f"{symbol}: [{sweep_label}] step 2 limit {status2} with no fill; escalating to market.")
    except Exception as e:
        print(f"{symbol}: [{sweep_label}] step 2 submit failed ({e}); escalating to market fallback.")
        logging.error(f"{now_str} [{sweep_label}] {symbol} step 2 submit failed: {e}")

    if remaining_qty <= 0:
        return total_filled_qty, total_notional, steps_used

    # ---- Step 3: plain market order -- guarantees a fill at whatever the
    # prevailing price is (short of a trading halt or broker outage). This is
    # what makes the sweep "sell no matter what" rather than "try to sell". ----
    try:
        order3 = api.submit_order(symbol=symbol, qty=str(remaining_qty), side='sell',
                                  type='market', time_in_force='day')
        print(f"{symbol}: [{sweep_label}] step 3 (market) submitted for "
              f"{remaining_qty:.4f} sh (order {getattr(order3, 'id', 'n/a')}).")
        terminal3, fq3, fp3, status3 = _poll_order_terminal(order3.id, ESCALATION_STEP2_TIMEOUT_SECS)
        _log_fill('step3', fq3, fp3)
        remaining_qty = round(remaining_qty - fq3, 6)
        if remaining_qty > 0:
            # Still not fully sold -- most likely a trading halt or a broker/
            # network problem. Nothing further to escalate to; log loudly so
            # a human notices, since this is the edge case "100%" can't cover.
            print(f"{symbol}: [{sweep_label}] {RED}WARNING{RESET} {remaining_qty:.4f} sh "
                  f"still unsold after all 3 escalation steps (status={status3}). "
                  f"Likely a trading halt or broker issue -- needs manual attention.")
            logging.error(f"{now_str} [{sweep_label}] {symbol}: {remaining_qty:.4f} sh unsold "
                          f"after full escalation chain (final status={status3}).")
    except Exception as e:
        print(f"{symbol}: [{sweep_label}] step 3 (market) submit failed: {e}. "
              f"{remaining_qty:.4f} sh unsold -- needs manual attention.")
        logging.error(f"{now_str} [{sweep_label}] {symbol} step 3 submit failed: {e}. "
                      f"{remaining_qty:.4f} sh unsold.")

    return total_filled_qty, total_notional, steps_used


def _run_profit_sweep(sweep_label, step1_type, step1_tif):
    """
    Shared body for both the pre-market (MOO) and pre-close (market) profit
    sweeps, run in two phases:

    Phase 1 (per-position): sell every position that is individually
    profitable, via the 3-step escalation chain. Unchanged from before.

    Phase 2 (portfolio-level): look at whatever positions are LEFT after
    phase 1. If their COMBINED unrealized P/L, divided by their combined cost
    basis, is >= PORTFOLIO_LIQUIDATION_MIN_PROFIT_PCT, sell all of them --
    winners and losers together -- because the portfolio nets a profit even
    though some individual legs don't. If the remainder doesn't clear the
    bar, phase 2 does nothing and those positions are left exactly as phase 1
    left them.
    """
    now = datetime.now(eastern)
    now_str = now.strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")
    print(f"\n==================== {sweep_label} ====================")
    try:
        positions = api.list_positions()
    except Exception as e:
        print(f"{sweep_label}: failed to fetch positions from broker: {e}")
        logging.error(f"{now_str} {sweep_label}: list_positions failed: {e}")
        return

    if not positions:
        print(f"{sweep_label}: no open positions.")
        print("=================================================================================\n")
        return

    # ---------------- Phase 1: per-position profitable sells ----------------
    submitted, skipped = [], []
    remaining_after_phase1 = []  # positions phase 1 did NOT sell (unprofitable or partial leftovers)

    for p in positions:
        symbol = p.symbol
        try:
            qty = float(p.qty)
            avg_entry = float(p.avg_entry_price)
            current_price = float(p.current_price) if getattr(p, 'current_price', None) else None
            if current_price is None:
                current_price = get_current_price(to_yf(symbol))

            # BUGFIX: a position with valid qty/avg_entry but an unconfirmed
            # current price used to be dropped from BOTH phase 1 and phase 2
            # entirely -- silently missing from the phase 2 cost-basis/market-
            # value totals, which understates (or overstates) the true
            # combined portfolio return. If qty/avg_entry are sane but price
            # lookup failed, still fold it into phase 2 using avg_entry as a
            # conservative (zero-gain) stand-in for its market value, so the
            # combined math isn't silently wrong -- it just can't be sold in
            # phase 1 without a live price.
            if qty <= 0 or avg_entry <= 0:
                skipped.append((symbol, "missing/invalid qty or avg entry price"))
                continue
            if current_price is None or current_price <= 0:
                skipped.append((symbol, "missing/invalid current price"))
                remaining_after_phase1.append((symbol, qty, avg_entry, avg_entry))
                continue

            gain_pct = (current_price - avg_entry) / avg_entry
            if gain_pct <= MOO_SWEEP_MIN_PROFIT_PCT:
                skipped.append((symbol, f"not profitable ({pnl_color(gain_pct)}{gain_pct*100:+.2f}%{RESET})"))
                remaining_after_phase1.append((symbol, qty, avg_entry, current_price))
                continue

            # Always cancel any pre-existing open sell orders on this symbol
            # first, so the sweep owns and can sell the FULL qty rather than
            # only whatever fraction wasn't already tied up in an older order.
            _cancel_existing_sell_orders(symbol, sweep_label, now_str)
            sweep_qty = round(qty, 6)
            if sweep_qty <= 0:
                skipped.append((symbol, "zero qty after cancelling pre-existing orders"))
                continue

            print(f"{symbol}: {GREEN}+{gain_pct*100:.2f}%{RESET} unrealized "
                  f"(avg ${avg_entry:.2f} -> last ${current_price:.2f}). "
                  f"Selling {sweep_qty:.4f} sh via escalation chain.")
            filled_qty, notional, steps = _sell_with_escalation(
                symbol, sweep_qty, current_price, step1_type, step1_tif, sweep_label, now_str)

            if filled_qty > 0:
                avg_fill_price = notional / filled_qty
                logging.info(f"{now_str} {sweep_label} sold {symbol}: {filled_qty:.4f} sh "
                            f"@ avg ${avg_fill_price:.2f} via [{', '.join(steps)}] "
                            f"(unrealized {gain_pct*100:+.2f}% at submit time).")
                submitted.append((symbol, filled_qty, gain_pct, steps))
                leftover_qty = round(sweep_qty - filled_qty, 6)
                if leftover_qty > 0:
                    # Escalation chain didn't fully clear this position (e.g. a
                    # halt) -- what's left is still an open position and is a
                    # candidate for phase 2's portfolio check.
                    remaining_after_phase1.append((symbol, leftover_qty, avg_entry, current_price))
            else:
                skipped.append((symbol, "escalation chain produced no fill (see warnings above)"))
                remaining_after_phase1.append((symbol, sweep_qty, avg_entry, current_price))
        except Exception as e:
            print(f"{symbol}: {sweep_label} failed: {e}")
            logging.error(f"{now_str} {sweep_label} failed for {symbol}: {e}")
            skipped.append((symbol, f"error: {e}"))

    try:
        session.commit()
    except Exception as e:
        logging.error(f"{now_str} {sweep_label}: DB commit failed: {e}")
        session.rollback()

    print(f"\n{sweep_label} phase 1 (per-position) summary: {len(submitted)} position(s) sold, "
          f"{len(skipped)} skipped.")
    for sym, fq, gp, steps in submitted:
        print(f"  SOLD  {sym}: {fq:.4f} sh, {pnl_color(gp)}{gp*100:+.2f}%{RESET} unrealized at submit time, "
              f"steps=[{', '.join(steps)}]")
    for sym, reason in skipped:
        print(f"  skipped    {sym}: {reason}")

    # ---------------- Phase 2: portfolio-level liquidation ----------------
    if USE_PORTFOLIO_LIQUIDATION_SWEEP and remaining_after_phase1:
        _run_portfolio_liquidation_phase(sweep_label, step1_type, step1_tif,
                                         remaining_after_phase1, now_str)

    print("=================================================================================\n")


def _run_portfolio_liquidation_phase(sweep_label, step1_type, step1_tif,
                                     remaining_positions, now_str):
    """
    Phase 2: given the positions phase 1 left untouched -- normally
    unprofitable/flat positions, but also any position phase 1 tried to sell
    and only partially filled (e.g. a halt cut the escalation chain short) --
    compute the COMBINED unrealized P/L versus COMBINED cost basis. If that ratio clears
    PORTFOLIO_LIQUIDATION_MIN_PROFIT_PCT, sell all of them via the same
    escalation chain used in phase 1. Otherwise, do nothing -- these stay
    open exactly as phase 1 left them.
    """
    total_cost_basis = 0.0
    total_market_value = 0.0
    for symbol, qty, avg_entry, current_price in remaining_positions:
        total_cost_basis += qty * avg_entry
        total_market_value += qty * current_price

    if total_cost_basis <= 0:
        return

    portfolio_gain_pct = (total_market_value - total_cost_basis) / total_cost_basis
    print(f"\n{sweep_label} phase 2 (portfolio-level) check: {len(remaining_positions)} "
          f"remaining position(s), combined cost basis ${total_cost_basis:,.2f}, "
          f"combined market value ${total_market_value:,.2f}, "
          f"combined unrealized {pnl_color(portfolio_gain_pct)}{portfolio_gain_pct*100:+.2f}%{RESET} "
          f"(threshold {PORTFOLIO_LIQUIDATION_MIN_PROFIT_PCT*100:.2f}%).")

    if portfolio_gain_pct < PORTFOLIO_LIQUIDATION_MIN_PROFIT_PCT:
        print(f"  Combined unrealized {pnl_color(portfolio_gain_pct)}{portfolio_gain_pct*100:+.2f}%{RESET} is below the "
              f"{PORTFOLIO_LIQUIDATION_MIN_PROFIT_PCT*100:.2f}% threshold. "
              f"Leaving remaining positions open (no forced sells).")
        return

    print(f"  {GREEN}Combined unrealized {portfolio_gain_pct*100:+.2f}% clears the "
          f"{PORTFOLIO_LIQUIDATION_MIN_PROFIT_PCT*100:.2f}% threshold.{RESET} "
          f"Liquidating all {len(remaining_positions)} remaining position(s), "
          f"winners and losers together, via the escalation chain.")

    liq_submitted, liq_skipped = [], []
    for symbol, qty, avg_entry, current_price in remaining_positions:
        try:
            liq_label = f"{sweep_label} portfolio-liq"
            _cancel_existing_sell_orders(symbol, liq_label, now_str)
            sell_qty = round(qty, 6)
            if sell_qty <= 0:
                liq_skipped.append((symbol, "zero qty after cancelling pre-existing orders"))
                continue

            leg_gain_pct = (current_price - avg_entry) / avg_entry if avg_entry else 0.0
            print(f"{symbol}: {pnl_color(leg_gain_pct)}{leg_gain_pct*100:+.2f}%{RESET} individually "
                  f"(avg ${avg_entry:.2f} -> last ${current_price:.2f}). "
                  f"Selling {sell_qty:.4f} sh as part of portfolio liquidation.")
            filled_qty, notional, steps = _sell_with_escalation(
                symbol, sell_qty, current_price, step1_type, step1_tif, liq_label, now_str)

            if filled_qty > 0:
                avg_fill_price = notional / filled_qty
                logging.info(f"{now_str} {sweep_label} portfolio-liq sold {symbol}: "
                            f"{filled_qty:.4f} sh @ avg ${avg_fill_price:.2f} via "
                            f"[{', '.join(steps)}] (individual leg {leg_gain_pct*100:+.2f}%, "
                            f"portfolio combined {portfolio_gain_pct*100:+.2f}% at trigger).")
                liq_submitted.append((symbol, filled_qty, leg_gain_pct, steps))
            else:
                liq_skipped.append((symbol, "escalation chain produced no fill (see warnings above)"))
        except Exception as e:
            print(f"{symbol}: portfolio liquidation failed: {e}")
            logging.error(f"{now_str} {sweep_label} portfolio-liq failed for {symbol}: {e}")
            liq_skipped.append((symbol, f"error: {e}"))

    try:
        session.commit()
    except Exception as e:
        logging.error(f"{now_str} {sweep_label}: portfolio-liq DB commit failed: {e}")
        session.rollback()

    print(f"\n{sweep_label} phase 2 (portfolio-level) summary: {len(liq_submitted)} position(s) "
          f"sold, {len(liq_skipped)} skipped.")
    for sym, fq, gp, steps in liq_submitted:
        print(f"  SOLD  {sym}: {fq:.4f} sh, {pnl_color(gp)}{gp*100:+.2f}%{RESET} individually, "
              f"steps=[{', '.join(steps)}]")
    for sym, reason in liq_skipped:
        print(f"  skipped    {sym}: {reason}")


def run_premarket_profit_sweep():
    """
    At 9:25am Eastern, sell every currently-profitable open position via a
    3-step escalation chain: MOO order first (fills at the opening auction),
    falling back to an aggressive limit at the bid, falling back to a plain
    market order that fills regardless of price. Runs once per trading day.

    Standalone from sell_stocks(): does not touch profit_monitor state,
    scale-out tracking, or the intraday exit logic.
    """
    global _moo_sweep_last_run_date
    if not USE_PREMARKET_PROFIT_SWEEP:
        return

    now = datetime.now(eastern)
    today_str = now.date().strftime("%Y-%m-%d")
    with _moo_sweep_lock:
        if _moo_sweep_last_run_date == today_str:
            return
        if not _in_moo_sweep_window(now):
            return
        _moo_sweep_last_run_date = today_str  # claim the day before any I/O

    _run_profit_sweep("Pre-Market Profit Sweep (9:25am ET)", step1_type='market', step1_tif='opg')


def run_close_profit_sweep():
    """
    At CLOSE_SWEEP_HOUR:CLOSE_SWEEP_MINUTE Eastern (default 3:45pm, 15 minutes
    before the close), sell every currently-profitable open position via the
    same 3-step escalation chain, starting with a plain market order (there is
    no market-on-close order type used here -- MOC has its own earlier
    submission cutoff and would arrive too close to CLOSE_SWEEP_MINUTE to be
    reliable) and falling back to limit, then market, exactly as the AM sweep
    does. Runs once per trading day, only during regular market hours (this
    is called from inside the main trading loop, not the closed-market wait
    loop, since 3:45pm is itself during the open session).
    """
    global _close_sweep_last_run_date
    if not USE_CLOSE_PROFIT_SWEEP:
        return

    now = datetime.now(eastern)
    today_str = now.date().strftime("%Y-%m-%d")
    with _close_sweep_lock:
        if _close_sweep_last_run_date == today_str:
            return
        if not _in_close_sweep_window(now):
            return
        _close_sweep_last_run_date = today_str  # claim the day before any I/O

    _run_profit_sweep("Pre-Close Profit Sweep (3:45pm ET)", step1_type='market', step1_tif='day')


def stop_if_stock_market_is_closed():
    nyse = mcal.get_calendar('NYSE')
    while True:
        current_datetime = datetime.now(eastern)
        current_time_str = current_datetime.strftime("%A, %B %d, %Y, %I:%M:%S %p")
        sched = nyse.schedule(start_date=current_datetime.date(), end_date=current_datetime.date())

        if not sched.empty:
            market_open = sched.iloc[0]['market_open'].astimezone(eastern)
            market_close = sched.iloc[0]['market_close'].astimezone(eastern)
            if market_open <= current_datetime <= market_close:
                print("Market is open. Proceeding with trading operations.")
                logging.info(f"{current_time_str}: Market is open.")
                return
            msg = f"Market is closed. Open hours: {market_open.strftime('%I:%M %p')} - {market_close.strftime('%I:%M %p')}"
            # Runs its own once-per-day/9:25am gate internally; safe to call
            # on every tick of this wait loop. Only fires on a real trading
            # day (sched is non-empty here), never on weekends/holidays.
            try:
                run_premarket_profit_sweep()
            except Exception as e:
                logging.error(f"{current_time_str}: pre-market profit sweep raised: {e}")
        else:
            msg = "Market is closed today (holiday or weekend)."

        print('''
        *********************************************************************************
        ************ Billionaire Buying Strategy Version ********************************
        *********************************************************************************
            2026 Edition of the Billionaire Strategy Stock Market Trading Robot, Version 10
                        https://github.com/CodeProSpecialist
               Margin Account Rules Engine - No PDT Round-Trip Limits
        ''')
        print(f'Current date & time (Eastern Time): {current_time_str}')
        print(msg)
        print("Waiting until Stock Market Hours to begin the Stockbot Trading Program.\n")
        # Prints every 60s cycle while closed too; fetch itself is throttled
        # to 1/hour internally so this doesn't hammer Alpaca overnight.
        try:
            print_portfolio_gain_summary()
        except Exception as e:
            logging.warning(f"[portfolio] summary error while market closed: {e}")
        logging.info(f"{current_time_str}: {msg}")
        time.sleep(60)


def print_database_tables():
    if not PRINT_DATABASE:
        return
    print("\nTrade History In This Robot's Database (most recent 25):\n")
    print("Stock | Buy or Sell | Quantity | Avg. Price | Date \n")
    # Pull the 25 most recent rows by insertion order (id is autoincrement),
    # then flip back to chronological (oldest -> newest) for display so the
    # newest trade prints last, closest to the prompt.
    recent = (
        session.query(TradeHistory)
        .order_by(TradeHistory.id.desc())
        .limit(25)
        .all()
    )
    for record in reversed(recent):
        print(f"{record.symbols} | {record.action} | {record.quantity:.4f} | {record.price:.2f} | {record.date}")

    print("----------------------------------------------------------------\n")
    print("Positions in the Database To Sell On or After the Date Shown:\n")
    print("Stock | Quantity | Avg. Price | Date \n")
    for record in session.query(Position).all():
        cp = get_current_price(record.symbols)
        # BUGFIX: guard against None current price / zero avg price
        if cp is not None and record.avg_price:
            pct = ((cp - record.avg_price) / record.avg_price) * 100
            color = GREEN if pct >= 0 else RED
            print(f"{record.symbols} | {record.quantity:.4f} | {record.avg_price:.2f} | "
                  f"{record.purchase_date} | Price Change: {color}{pct:.2f}%{RESET}")
        else:
            print(f"{record.symbols} | {record.quantity:.4f} | {record.avg_price:.2f} | {record.purchase_date}")
    print("\n")


# =====================================================================
# Backtest (Backtrader, inlined, reuses live constants for parity)
# =====================================================================
#
# Adapted from a Backtrader script. Every trading rule below reads the SAME
# module-level constant the live bot uses (HARD_STOP_ATR_MULTIPLIER,
# HARD_STOP_MIN_PCT, ARM_PROFIT_PCT, PEAK_GIVEBACK_PCT, PEAK_GIVEBACK_FRACTION,
# HARD_FLOOR_PCT, SCALE_OUT_STAGES, RISK_PER_TRADE_PCT,
# MAX_ALLOCATION_PER_SYMBOL, BUY_SCORE_THRESHOLD), so if a constant is tuned
# the backtest immediately reflects the change on the next run -- no drift
# between backtested behavior and live behavior.
#
# Runs weekly on Sunday nights (see maybe_run_scheduled_backtest), never
# during trading hours, and NEVER blocks the live bot -- backtrader is a
# lazy optional dependency, so bot.py starts fine without it installed.
#
# What this does NOT cover:
#   - Intraday exit ordering: uses daily bars, so a bar that both hits the
#     hard stop AND rallies to a scale-out target on the same day resolves
#     as "hard stop first" (worst-first) rather than tracking true intraday
#     sequence.
#   - Live-execution frictions: no MOO/LMT escalation chain, no cancel-first
#     sweep logic, no pre-existing-order handling.
#   - Slippage is modeled as a flat percent (BACKTEST_SLIPPAGE_PCT) rather
#     than modeled per-fill via the actual escalation ladder.
# These caveats mean backtest results will *approximately* match live P&L
# for the same parameters, not exactly. That's the intended trade-off for a
# fast Backtrader run vs. a full tick-by-tick simulator.

BACKTEST_ENABLED = True
BACKTEST_SYMBOLS_SOURCE = 'candidate_list'   # 'candidate_list' or 'hardcoded'
# Hardcoded universe if BACKTEST_SYMBOLS_SOURCE == 'hardcoded'; otherwise
# pulls the in-memory SYMBOLS_TO_BUY_LIST produced by the inlined scanner.
BACKTEST_HARDCODED_SYMBOLS = [
    "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "XEL", "PEG",
    "ED", "WEC", "ES", "DTE", "PPL", "EIX", "FE", "ETR",
    "CMS", "LNT", "AEE", "PCG",
]
BACKTEST_LOOKBACK_YEARS = 2
BACKTEST_INITIAL_CASH = 100_000.0
BACKTEST_COMMISSION = 0.001
BACKTEST_SLIPPAGE_PCT = 0.0005
BACKTEST_MAX_OPEN_POSITIONS = 8
BACKTEST_MIN_SYMBOL_BARS = 250     # skip symbols with less than this much daily history
BACKTEST_STATE_PATH = os.path.join(_ml_base_dir, 'backtest_schedule_state.json')
BACKTEST_RESULTS_DIR = os.path.join(_ml_base_dir, 'backtest_results')
# Weekly on Sunday nights at 22:00 ET (well after Friday close, well before
# Monday premarket, and non-conflicting with the ML brain's 17:00 daily slot
# which finishes long before 22:00).
BACKTEST_RUN_WEEKDAY = 6           # 0=Mon ... 6=Sun
BACKTEST_RUN_HOUR = 22
BACKTEST_RUN_MINUTE_WINDOW = 30    # tolerance so a slow loop tick can't miss the slot

_backtest_availability_cache = {'checked': False, 'available': False, 'module': None}


def _backtest_lazy_import():
    """Lazy import of backtrader. Returns the module or None. Same defensive
    pattern as TensorFlow: an install without backtrader still runs the live
    bot fine, just without the weekly backtest."""
    if _backtest_availability_cache['checked']:
        return _backtest_availability_cache['module']
    try:
        import backtrader as bt
        _backtest_availability_cache['module'] = bt
        _backtest_availability_cache['available'] = True
    except Exception as e:
        logging.warning(f"backtest: backtrader unavailable ({e}); weekly backtest disabled.")
        _backtest_availability_cache['available'] = False
    _backtest_availability_cache['checked'] = True
    return _backtest_availability_cache['module']


def _backtest_get_universe():
    if BACKTEST_SYMBOLS_SOURCE == 'candidate_list':
        try:
            syms = get_symbols_to_buy()
            if syms:
                return syms
        except Exception:
            pass
    return list(BACKTEST_HARDCODED_SYMBOLS)


def _backtest_download_data(symbols, start, end):
    """Download daily OHLCV for the symbol list using the SHARED yfinance
    gate, so a backtest run can't accidentally blow past the rate limit the
    live bot is also using. Returns {symbol: DataFrame}. Any symbol that
    fails or has less than BACKTEST_MIN_SYMBOL_BARS bars is silently skipped.
    """
    print(f"[backtest] downloading {len(symbols)} symbols from {start} to {end} ...")
    logging.info(f"backtest: downloading {len(symbols)} symbols {start} to {end}")

    try:
        data_by_symbol = yf_download_batch(symbols, start=start, end=end, interval='1d')
    except Exception as e:
        logging.error(f"backtest: batch download failed: {e}")
        return {}

    out = {}
    for sym in symbols:
        df = data_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        try:
            df = df.dropna(how='all')
            if len(df) < BACKTEST_MIN_SYMBOL_BARS:
                continue
            # Backtrader wants Title-case column names
            df = df.rename(columns=str.title)
            if 'Volume' not in df.columns and 'volume' in df.columns:
                df['Volume'] = df['volume']
            out[sym] = df
        except Exception as e:
            logging.warning(f"backtest: skipping {sym} due to error: {e}")

    print(f"[backtest] {len(out)}/{len(symbols)} symbols usable after filtering")
    return out


def _build_backtest_strategy_class(bt):
    """Factory that builds the Backtrader Strategy class. Wrapped in a
    factory so the `bt` module reference is captured after the lazy import,
    and so the class isn't defined at bot.py import time (which would fail
    if backtrader isn't installed)."""
    from collections import defaultdict as _bt_defaultdict
    import math as _bt_math

    class BillionaireApproxStrategy(bt.Strategy):
        """Approximates the live bot's core buy-and-exit logic against
        daily bars. Reads all trading constants from bot.py's module-level
        names -- so tuning the live bot instantly re-tunes the backtest.
        """
        params = dict(printlog=False)

        def __init__(self):
            self.inds = {}
            self.order = {}
            self.entry_price = {}
            self.entry_atr = {}
            self.peak_price = {}
            self.armed = {}
            self.floor_pct = {}
            self.original_size = {}
            self.stages_fired = _bt_defaultdict(set)
            self.trade_log = []
            for d in self.datas:
                self.inds[d] = {
                    'sma200': bt.indicators.SMA(d.close, period=200),
                    'rsi':    bt.indicators.RSI(d.close, period=14),
                    'macd':   bt.indicators.MACD(d.close, period_me1=12, period_me2=26, period_signal=9),
                    'atr':    bt.indicators.ATR(d, period=14),
                    'volsma': bt.indicators.SMA(d.volume, period=14),
                }
                self.order[d] = None
                self.entry_price[d] = None
                self.entry_atr[d] = None
                self.peak_price[d] = None
                self.armed[d] = False
                self.floor_pct[d] = HARD_FLOOR_PCT
                self.original_size[d] = 0.0

        def log(self, txt, dt=None):
            if not self.p.printlog:
                return
            dt = dt or self.datas[0].datetime.date(0)
            print(f"[backtest] {dt.isoformat()}  {txt}")

        def notify_order(self, order):
            if order.status in (order.Submitted, order.Accepted):
                return
            d = order.data
            if order.status == order.Completed:
                if order.isbuy():
                    self.entry_price[d] = order.executed.price
                    self.entry_atr[d] = float(self.inds[d]['atr'][0])
                    self.peak_price[d] = order.executed.price
                    self.armed[d] = False
                    self.floor_pct[d] = HARD_FLOOR_PCT
                    self.original_size[d] = order.executed.size
                    self.stages_fired[d].clear()
            self.order[d] = None

        def notify_trade(self, trade):
            if not trade.isclosed:
                return
            self.trade_log.append({
                'symbol': trade.data._name, 'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm, 'barlen': trade.barlen,
            })

        def _score(self, d):
            """Reduced compute_buy_score approximation for daily bars.
            Same signal *concepts* as the live compute_buy_score, but only
            the ones that make sense at daily resolution (no 5m tick, no
            intraday pullback -- those need intraday bars we don't have here)."""
            ind = self.inds[d]
            score = 0

            rsi = float(ind['rsi'][0])
            if _bt_math.isfinite(rsi) and rsi < 50:
                score += 1

            if len(d) > 15:
                recent = [float(ind['rsi'][-i]) for i in range(1, 6)]
                prior = [float(ind['rsi'][-i]) for i in range(6, 11)]
                if np.mean(recent) < np.mean(prior):
                    score += 1

            macd = float(ind['macd'].macd[0])
            sig = float(ind['macd'].signal[0])
            if _bt_math.isfinite(macd) and _bt_math.isfinite(sig) and macd > sig:
                score += 1

            vol = float(d.volume[0])
            vsma = float(ind['volsma'][0])
            if vsma > 0 and vol >= 0.9 * vsma:
                score += 1

            if len(d) > 1:
                prev, cur = float(d.close[-1]), float(d.close[0])
                if prev > 0 and (prev - cur) / prev >= 0.002:
                    score += 1

            # Cheap hammer proxy (long lower wick relative to body).
            o = float(d.open[0]); h = float(d.high[0])
            l = float(d.low[0]);  c = float(d.close[0])
            body = abs(c - o)
            lower_wick = min(o, c) - l
            if body > 0 and lower_wick > 1.5 * body and c > o:
                score += 2

            return score

        def _position_size(self, d, price, atr):
            """Uses the SAME risk-per-share formula the live buy_stocks uses
            (HARD_STOP_ATR_MULTIPLIER * atr, floored at HARD_STOP_MIN_PCT * price),
            so a shift in either constant flows straight through to sizing."""
            if atr is None or atr <= 0 or price <= 0:
                return 0
            risk_per_share = max(HARD_STOP_ATR_MULTIPLIER * atr,
                                 HARD_STOP_MIN_PCT * price)
            risk_amount = self.broker.getvalue() * RISK_PER_TRADE_PCT
            size = risk_amount / risk_per_share
            max_shares = MAX_ALLOCATION_PER_SYMBOL / price
            size = min(size, max_shares)
            return max(_bt_math.floor(size), 0)

        def next(self):
            open_count = sum(1 for d in self.datas if self.getposition(d).size > 0)
            for d in self.datas:
                if self.order[d]:
                    continue
                pos = self.getposition(d)
                ind = self.inds[d]
                price = float(d.close[0])
                atr = float(ind['atr'][0])
                sma200 = float(ind['sma200'][0])

                # ---- EXITS ----
                if pos.size > 0:
                    entry = self.entry_price[d] or pos.price
                    gain = (price - entry) / entry if entry else 0.0

                    # 1. Hard stop (frozen entry ATR, same "min entry vs current" as live)
                    stop_atr_dollars = min(self.entry_atr[d] or atr, atr)
                    stop_dist = max(HARD_STOP_ATR_MULTIPLIER * stop_atr_dollars / entry,
                                    HARD_STOP_MIN_PCT)
                    if gain <= -stop_dist:
                        self.order[d] = self.close(data=d)
                        self._reset_state(d)
                        continue

                    # 2. Scale-outs (fire ONE stage per bar, mirroring live)
                    for idx, (trigger, frac) in enumerate(SCALE_OUT_STAGES):
                        if idx in self.stages_fired[d]:
                            continue
                        if gain >= trigger:
                            sell_size = _bt_math.floor(self.original_size[d] * frac)
                            sell_size = min(sell_size, pos.size)
                            if sell_size > 0:
                                self.order[d] = self.sell(data=d, size=sell_size)
                                self.stages_fired[d].add(idx)
                                self.floor_pct[d] = max(self.floor_pct[d], 0.0005)
                            break

                    # 3. Peak-following profit monitor
                    atr_pct = (atr / entry) if entry else 0.0
                    arm_pct = max(ARM_PROFIT_PCT, ATR_ARM_MULTIPLIER * atr_pct)
                    if not self.armed[d] and gain >= arm_pct:
                        self.armed[d] = True
                        self.peak_price[d] = price
                    if self.armed[d]:
                        if price > (self.peak_price[d] or price):
                            self.peak_price[d] = price
                        peak = self.peak_price[d]
                        peak_gain = (peak - entry) / entry if entry else 0.0
                        giveback = (peak - price) / peak if peak else 0.0
                        giveback_target = max(
                            PEAK_GIVEBACK_PCT,
                            ATR_GIVEBACK_FRACTION * arm_pct,
                            PEAK_GIVEBACK_FRACTION * peak_gain,
                        )
                        if giveback >= giveback_target and gain >= self.floor_pct[d]:
                            self.order[d] = self.close(data=d)
                            self._reset_state(d)
                            continue
                        if gain < self.floor_pct[d]:
                            self.order[d] = self.close(data=d)
                            self._reset_state(d)
                            continue
                    continue

                # ---- ENTRY ----
                if open_count >= BACKTEST_MAX_OPEN_POSITIONS:
                    continue
                if not _bt_math.isfinite(sma200) or price <= sma200:
                    continue
                rsi = float(ind['rsi'][0])
                if not _bt_math.isfinite(rsi) or rsi >= 50:
                    continue

                score = self._score(d)
                if score < BUY_SCORE_THRESHOLD_DEFAULT:
                    continue
                size = self._position_size(d, price, atr)
                if size < 1:
                    continue
                if size * price > self.broker.getcash() * 0.95:
                    continue

                self.order[d] = self.buy(data=d, size=size)
                open_count += 1

        def _reset_state(self, d):
            self.entry_price[d] = None
            self.entry_atr[d] = None
            self.peak_price[d] = None
            self.armed[d] = False
            self.floor_pct[d] = HARD_FLOOR_PCT
            self.original_size[d] = 0.0
            self.stages_fired[d].clear()

    return BillionaireApproxStrategy


def run_backtest():
    """One backtest pass over BACKTEST_LOOKBACK_YEARS of daily bars for the
    candidate universe. Returns a dict of results or None on failure. Cheap
    to call -- worst case it just returns None with a logged warning. Safe
    to schedule."""
    bt = _backtest_lazy_import()
    if bt is None:
        return None

    symbols = _backtest_get_universe()
    if not symbols:
        logging.warning("backtest: no symbols in universe; skipping.")
        return None

    end_date = datetime.now(eastern).date()
    start_date = end_date - timedelta(days=int(365 * BACKTEST_LOOKBACK_YEARS))
    data_map = _backtest_download_data(symbols, start_date.isoformat(),
                                       end_date.isoformat())
    if not data_map:
        return None

    StrategyCls = _build_backtest_strategy_class(bt)

    cerebro = bt.Cerebro(stdstats=False)   # stdstats=False avoids matplotlib requirement
    cerebro.broker.setcash(BACKTEST_INITIAL_CASH)
    cerebro.broker.setcommission(commission=BACKTEST_COMMISSION)
    cerebro.broker.set_slippage_perc(perc=BACKTEST_SLIPPAGE_PCT)

    for sym, df in data_map.items():
        data = bt.feeds.PandasData(dataname=df, name=sym,
                                   timeframe=bt.TimeFrame.Days, compression=1)
        cerebro.adddata(data)
    cerebro.addstrategy(StrategyCls)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    start_val = cerebro.broker.getvalue()
    try:
        results = cerebro.run()
    except Exception as e:
        logging.error(f"backtest: cerebro.run() failed: {e}")
        return None
    if not results:
        return None
    strat = results[0]

    final_val = cerebro.broker.getvalue()
    net_pnl = final_val - start_val

    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    ta = strat.analyzers.ta.get_analysis()
    rets = strat.analyzers.returns.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    total_trades = ta.get('total', {}).get('total', 0) if ta else 0
    won = ta.get('won', {}).get('total', 0) if ta else 0
    lost = ta.get('lost', {}).get('total', 0) if ta else 0
    avg_win_pnl = ta.get('won', {}).get('pnl', {}).get('average', 0) if won else 0
    avg_loss_pnl = ta.get('lost', {}).get('pnl', {}).get('average', 0) if lost else 0

    report = {
        'ran_at': datetime.now(eastern).isoformat(),
        'symbols_universe': len(symbols),
        'symbols_used': len(data_map),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'initial_cash': BACKTEST_INITIAL_CASH,
        'final_value': final_val,
        'net_pnl': net_pnl,
        'net_pnl_pct': (net_pnl / BACKTEST_INITIAL_CASH * 100) if BACKTEST_INITIAL_CASH else 0,
        'total_trades': total_trades,
        'wins': won,
        'losses': lost,
        'win_rate_pct': (won / total_trades * 100) if total_trades else 0,
        'avg_win_pnl': avg_win_pnl,
        'avg_loss_pnl': avg_loss_pnl,
        'sharpe_ratio': sharpe.get('sharperatio'),
        'max_drawdown_pct': dd.max.drawdown if hasattr(dd, 'max') else None,
        'max_drawdown_money': dd.max.moneydown if hasattr(dd, 'max') else None,
        'total_return_pct': rets.get('rtot', 0) * 100 if 'rtot' in rets else None,
        'annualized_return_pct': rets.get('rnorm100'),
        'sqn': sqn.get('sqn'),
        # Snapshot the constants this backtest ran under, so an operator
        # reading old results can see what parameters produced them.
        'parameters': {
            'HARD_STOP_ATR_MULTIPLIER': HARD_STOP_ATR_MULTIPLIER,
            'HARD_STOP_MIN_PCT': HARD_STOP_MIN_PCT,
            'ARM_PROFIT_PCT': ARM_PROFIT_PCT,
            'PEAK_GIVEBACK_PCT': PEAK_GIVEBACK_PCT,
            'PEAK_GIVEBACK_FRACTION': PEAK_GIVEBACK_FRACTION,
            'HARD_FLOOR_PCT': HARD_FLOOR_PCT,
            'RISK_PER_TRADE_PCT': RISK_PER_TRADE_PCT,
            'MAX_ALLOCATION_PER_SYMBOL': MAX_ALLOCATION_PER_SYMBOL,
            'SCALE_OUT_STAGES': [list(s) for s in SCALE_OUT_STAGES],
            'BUY_SCORE_THRESHOLD_DEFAULT': BUY_SCORE_THRESHOLD_DEFAULT,
        },
    }

    _backtest_save_report(report)
    _backtest_print_summary(report)
    return report


def _backtest_save_report(report):
    """Persist each run's report to a timestamped JSON under
    BACKTEST_RESULTS_DIR so history accumulates and operator can diff
    week-over-week."""
    try:
        os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
        stamp = datetime.now(eastern).strftime('%Y%m%d_%H%M%S')
        path = os.path.join(BACKTEST_RESULTS_DIR, f'backtest_{stamp}.json')
        with open(path, 'w') as f:
            _ml_json.dump(report, f, indent=2, default=str)
    except Exception as e:
        logging.warning(f"backtest: failed to save report: {e}")


def _backtest_print_summary(report):
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Symbols used:       {report['symbols_used']}/{report['symbols_universe']}")
    print(f"Window:             {report['start_date']} to {report['end_date']}")
    print(f"Initial cash:       ${report['initial_cash']:,.2f}")
    print(f"Final value:        ${report['final_value']:,.2f}")
    print(f"Net P&L:            {pnl_color(report['net_pnl'])}${report['net_pnl']:,.2f}  ({report['net_pnl_pct']:+.2f}%){RESET}")
    print(f"Total trades:       {report['total_trades']}  "
          f"({report['wins']}W / {report['losses']}L, "
          f"win rate {report['win_rate_pct']:.1f}%)")
    print(f"Avg winner P&L:     {GREEN}${report['avg_win_pnl']:,.2f}{RESET}")
    print(f"Avg loser P&L:      {RED}${report['avg_loss_pnl']:,.2f}{RESET}")
    if report['sharpe_ratio'] is not None:
        print(f"Sharpe:             {report['sharpe_ratio']:.3f}")
    if report['max_drawdown_pct'] is not None:
        print(f"Max drawdown:       {report['max_drawdown_pct']:.2f}% "
              f"(${report.get('max_drawdown_money', 0):,.2f})")
    if report['annualized_return_pct'] is not None:
        print(f"Annualized return:  {report['annualized_return_pct']:.2f}%")
    if report['sqn'] is not None:
        print(f"SQN:                {report['sqn']:.3f}")
    print("=" * 70 + "\n")


def maybe_run_scheduled_backtest():
    """Called from the main loop. Runs one backtest pass if we're inside the
    weekly scheduled window (Sunday 22:00 ET +/- 30 minutes) AND we haven't
    already run this week. Cheap to call every cycle (state check only).
    Returns a status string if it actually ran, None otherwise."""
    if not BACKTEST_ENABLED:
        return None
    if _backtest_lazy_import() is None:
        return None

    now = datetime.now(eastern)
    if now.weekday() != BACKTEST_RUN_WEEKDAY:
        return None
    if now.hour != BACKTEST_RUN_HOUR:
        return None
    if now.minute >= BACKTEST_RUN_MINUTE_WINDOW:
        return None

    # Once-per-week gate keyed by ISO year+week so a slow tick can't fire twice.
    iso_year, iso_week, _ = now.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"

    state = {}
    try:
        if os.path.exists(BACKTEST_STATE_PATH):
            with open(BACKTEST_STATE_PATH) as f:
                state = _ml_json.load(f)
    except Exception:
        state = {}
    if state.get('last_run_week') == week_key:
        return None

    # Claim the slot BEFORE the slow download starts, same pattern as ML
    # brain scheduling.
    state['last_run_week'] = week_key
    state['last_run_started_at'] = now.isoformat()
    try:
        with open(BACKTEST_STATE_PATH, 'w') as f:
            _ml_json.dump(state, f, indent=2)
    except Exception as e:
        logging.warning(f"backtest: failed to save schedule state: {e}")

    print(f"[backtest] weekly scheduled run starting at {now.isoformat()}")
    logging.info(f"backtest: weekly scheduled run starting at {now.isoformat()}")

    try:
        report = run_backtest()
    except Exception as e:
        logging.error(f"backtest: run_backtest raised: {e}")
        return f"backtest run raised {type(e).__name__}: {e}"

    if report is None:
        return "backtest run returned no report (see warnings above)"

    state['last_run_completed_at'] = datetime.now(eastern).isoformat()
    state['last_run_final_value'] = report['final_value']
    state['last_run_total_trades'] = report['total_trades']
    try:
        with open(BACKTEST_STATE_PATH, 'w') as f:
            _ml_json.dump(state, f, indent=2)
    except Exception:
        pass

    return (f"backtest weekly run complete: final=${report['final_value']:,.2f}, "
           f"trades={report['total_trades']}, win_rate={report['win_rate_pct']:.1f}%, "
           f"sharpe={report['sharpe_ratio']}")



# ANALYZE_TRADE_HISTORY_EVERY_N_CYCLES controls how often the main loop runs
# this (it queries the whole trade_features table, so it's not free to run
# every 60s tick). Purely informational -- it prints findings but does not
# change live parameters. Feed the printed buckets into BUY_SCORE_THRESHOLD
# tuning or REGIME_SIGNAL_WEIGHTS manually, or extend this into an automatic
# adjustment once you trust the sample size.
ANALYZE_TRADE_HISTORY_EVERY_N_CYCLES = 30   # roughly every 30 minutes at 60s/cycle
MIN_TRADES_FOR_ANALYSIS = 10


def analyze_trade_history():
    """
    REVIEW ITEM #7: query closed trades from TradeFeatures and report which
    entry-feature combinations were actually associated with profitable
    outcomes -- buy score bucket, regime, RSI bucket, pattern, MACD state.
    """
    closed = session.query(TradeFeatures).filter(TradeFeatures.outcome_pct.isnot(None)).all()
    if len(closed) < MIN_TRADES_FOR_ANALYSIS:
        print(f"Trade-history analysis: only {len(closed)} closed trades on record "
              f"(need {MIN_TRADES_FOR_ANALYSIS}+). Skipping.")
        return

    def _bucket_stats(rows, keyfn, label):
        buckets = {}
        for r in rows:
            k = keyfn(r)
            if k is None:
                continue
            buckets.setdefault(k, []).append(r.outcome_pct)
        print(f"\n  By {label}:")
        for k, outcomes in sorted(buckets.items(), key=lambda kv: -np.mean(kv[1])):
            win_rate = sum(1 for o in outcomes if o > 0) / len(outcomes) * 100
            print(f"    {k}: n={len(outcomes)}, avg={np.mean(outcomes)*100:+.2f}%, "
                  f"win rate={win_rate:.0f}%")

    print(f"\n--- Trade History Analysis ({len(closed)} closed trades) ---")
    overall = [r.outcome_pct for r in closed]
    print(f"  Overall: avg={np.mean(overall)*100:+.2f}%, "
          f"win rate={sum(1 for o in overall if o > 0) / len(overall) * 100:.0f}%")

    _bucket_stats(closed, lambda r: r.regime, "regime")
    _bucket_stats(closed, lambda r: int(r.buy_score) if r.buy_score is not None else None, "buy score")
    _bucket_stats(closed, lambda r: r.candlestick_pattern, "candlestick pattern")
    _bucket_stats(closed, lambda r: 'macd_bullish' if r.macd_above_signal else 'macd_bearish', "MACD state")
    print("--- End analysis ---\n")


def _atr_pct_bucket(atr_pct):
    """Bucket ATR% into readable ranges for the per-ATR expectancy report."""
    if atr_pct is None: return None
    if atr_pct < 0.01: return "<1%"
    if atr_pct < 0.02: return "1-2%"
    if atr_pct < 0.03: return "2-3%"
    if atr_pct < 0.05: return "3-5%"
    return ">=5%"


def _tod_bucket(time_of_day):
    """Bucket time_of_day (HH:MM string) into common trading windows."""
    if not time_of_day or ':' not in time_of_day:
        return None
    try:
        h = int(time_of_day.split(':')[0])
    except ValueError:
        return None
    if 9 <= h < 10: return "09:00-10:00 open"
    if 10 <= h < 12: return "10:00-12:00 morning"
    if 12 <= h < 14: return "12:00-14:00 midday"
    if 14 <= h < 15: return "14:00-15:00 afternoon"
    if 15 <= h < 16: return "15:00-16:00 close"
    return "other"


def report_expectancy():
    """
    Comprehensive trade-history report covering the metrics requested in the
    review: win rate, avg/median winner/loser, expectancy, profit factor,
    max drawdown, Sharpe/Sortino, avg holding time, trades/day, and returns
    bucketed by regime, buy score, ATR%, time of day, and symbol.

    Runs off closed trades in TradeFeatures. Prints to stdout and writes to
    log. Called on the same MIN_TRADES_FOR_ANALYSIS gate as analyze_trade_history.
    """
    closed = (session.query(TradeFeatures)
              .filter(TradeFeatures.outcome_pct.isnot(None))
              .all())
    if len(closed) < MIN_TRADES_FOR_ANALYSIS:
        print(f"Expectancy report: only {len(closed)} closed trades on record "
              f"(need {MIN_TRADES_FOR_ANALYSIS}+). Skipping.")
        return

    outcomes = np.array([r.outcome_pct for r in closed])
    winners = outcomes[outcomes > 0]
    losers = outcomes[outcomes < 0]
    n = len(outcomes)

    win_rate = len(winners) / n if n else 0.0
    avg_win = float(np.mean(winners)) if len(winners) else 0.0
    avg_loss = float(np.mean(losers)) if len(losers) else 0.0
    med_win = float(np.median(winners)) if len(winners) else 0.0
    med_loss = float(np.median(losers)) if len(losers) else 0.0

    # Expectancy per trade = win_rate*avg_win + (1-win_rate)*avg_loss.
    # avg_loss is already negative so this subtracts correctly.
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Profit factor = gross wins / gross losses (in absolute terms). Undefined
    # if no losers -- show as inf in that case.
    gross_win = float(winners.sum()) if len(winners) else 0.0
    gross_loss = float(-losers.sum()) if len(losers) else 0.0   # positive
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    # Max drawdown (as a fraction) on the equity curve of stacked returns.
    # This isn't a true portfolio-level drawdown -- it's the drawdown of a
    # signed sum of per-trade outcomes in chronological order, which is a
    # reasonable proxy for "worst string of losses" the strategy produced.
    equity = np.cumsum(outcomes)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = running_peak - equity   # non-negative
    max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0

    # Sharpe/Sortino: annualized from per-trade returns. We don't know trade
    # frequency exactly, so use total elapsed days between first and last
    # trade to estimate trades-per-year. Risk-free rate assumed 0 for a
    # short-holding-period strategy.
    std_returns = float(outcomes.std(ddof=1)) if n > 1 else 0.0
    downside = outcomes[outcomes < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    mean_return = float(outcomes.mean()) if n else 0.0

    # Estimate trades/day and holding time from entry/exit dates
    def _parse(dt_str):
        if not dt_str: return None
        try:
            return datetime.fromisoformat(dt_str) if 'T' in dt_str \
                else datetime.strptime(dt_str, '%Y-%m-%d')
        except Exception:
            return None

    holding_times = []
    entry_dts = []
    for r in closed:
        e, x = _parse(r.entry_date), _parse(r.exit_date)
        if e and x:
            hrs = (x - e).total_seconds() / 3600.0
            if hrs >= 0:
                holding_times.append(hrs)
                entry_dts.append(e)
    avg_holding_hrs = float(np.mean(holding_times)) if holding_times else 0.0
    if len(entry_dts) >= 2:
        span_days = max((max(entry_dts) - min(entry_dts)).days, 1)
        trades_per_day = n / span_days
    else:
        trades_per_day = 0.0

    trades_per_year = max(trades_per_day * 252, 1)   # trading days
    sharpe = (mean_return / std_returns * np.sqrt(trades_per_year)) if std_returns > 0 else 0.0
    sortino = (mean_return / downside_std * np.sqrt(trades_per_year)) if downside_std > 0 else 0.0

    # ---- print ----
    hdr = f"--- Expectancy Report ({n} closed trades) ---"
    print(f"\n{hdr}")
    print(f"  Win rate:            {win_rate*100:.1f}%   ({len(winners)}W / {len(losers)}L)")
    print(f"  Avg winner:          {GREEN}{avg_win*100:+.2f}%{RESET}    Avg loser: {RED}{avg_loss*100:+.2f}%{RESET}")
    print(f"  Median winner:       {GREEN}{med_win*100:+.2f}%{RESET}    Median loser: {RED}{med_loss*100:+.2f}%{RESET}")
    print(f"  Expectancy / trade:  {pnl_color(expectancy)}{expectancy*100:+.3f}%{RESET}")
    print(f"  Profit factor:       {profit_factor:.2f}" if np.isfinite(profit_factor) else "  Profit factor:       inf (no losers)")
    print(f"  Max drawdown (cum):  {max_dd*100:.2f}%")
    print(f"  Sharpe (annualized): {sharpe:.2f}")
    print(f"  Sortino (annualized):{sortino:.2f}")
    print(f"  Avg holding time:    {avg_holding_hrs:.1f} hours")
    print(f"  Trades per day:      {trades_per_day:.2f}")
    logging.info(f"expectancy: win_rate={win_rate:.3f} exp={expectancy:.5f} "
                f"pf={profit_factor:.2f} sharpe={sharpe:.2f} dd={max_dd:.4f} n={n}")

    # Per-bucket returns
    def _report_bucket(rows, keyfn, label):
        buckets = {}
        for r in rows:
            k = keyfn(r)
            if k is None: continue
            buckets.setdefault(k, []).append(r.outcome_pct)
        if not buckets: return
        print(f"\n  Return by {label}:")
        for k in sorted(buckets.keys(), key=lambda x: -float(np.mean(buckets[x]))):
            arr = np.array(buckets[k])
            wr = float((arr > 0).sum()) / len(arr) * 100
            print(f"    {k}: n={len(arr)}, avg={arr.mean()*100:+.2f}%, "
                  f"win rate={wr:.0f}%, median={float(np.median(arr))*100:+.2f}%")

    _report_bucket(closed, lambda r: r.regime, "regime")
    _report_bucket(closed, lambda r: int(r.buy_score) if r.buy_score is not None else None, "buy score")
    _report_bucket(closed, lambda r: _atr_pct_bucket(r.atr_pct), "ATR%")
    _report_bucket(closed, lambda r: _tod_bucket(r.time_of_day), "time of day")
    _report_bucket(closed, lambda r: r.symbols, "symbol")
    print("--- End expectancy report ---\n")


def compare_parameter_expectancy():
    """
    For each adaptive parameter this bot self-adjusts (buy_score_threshold,
    signal weights, etc.), look up which value was active at each closed
    trade's entry time, then report expectancy per parameter value. This is
    the honest, no-simulator alternative to backtesting parameter grids:
    instead of guessing what would have happened with different values, it
    shows what actually DID happen at values the bot has already tried.

    Requires at least MIN_TRADES_PER_PARAM_VALUE closed trades per value to
    be reported (small samples get grouped as "insufficient data" instead of
    printing misleadingly small stats).
    """
    MIN_TRADES_PER_PARAM_VALUE = 5

    log_rows = (session.query(AdaptiveParamLog)
                .order_by(AdaptiveParamLog.timestamp.asc())
                .all())
    closed_trades = (session.query(TradeFeatures)
                     .filter(TradeFeatures.outcome_pct.isnot(None))
                     .order_by(TradeFeatures.entry_date.asc())
                     .all())
    if not log_rows or not closed_trades:
        print("Parameter-expectancy comparison: no adaptive-param log rows or "
              "closed trades yet; skipping.")
        return

    def _parse_dt(s):
        if not s: return None
        try:
            return datetime.fromisoformat(s.replace(' ', 'T')) if 'T' in s or ' ' in s \
                else datetime.strptime(s, '%Y-%m-%d')
        except Exception:
            return None

    # Build a per-parameter timeline: for each param, a sorted list of
    # (start_datetime, value) pairs. First timestamp is the earliest known
    # log row for that param; before that we don't know the value, so trades
    # from before the first log entry are excluded per-param.
    timelines = {}
    for row in log_rows:
        key = f"{row.param_name}[{row.regime}]"
        timelines.setdefault(key, []).append((_parse_dt(row.timestamp), row.new_value))

    print(f"\n--- Parameter-Value Expectancy (from {len(closed_trades)} closed trades, "
          f"{len(log_rows)} param-change events) ---")

    for key, transitions in timelines.items():
        transitions = [(t, v) for t, v in transitions if t is not None]
        if not transitions:
            continue

        buckets = {}  # value -> list of outcome_pcts
        for tf in closed_trades:
            entry_dt = _parse_dt(tf.entry_date)
            if entry_dt is None:
                continue
            active_value = None
            for t, v in transitions:
                if entry_dt >= t:
                    active_value = v
                else:
                    break
            if active_value is None:
                continue
            buckets.setdefault(round(active_value, 4), []).append(tf.outcome_pct)

        if not buckets:
            continue

        # Compact per-value stats table
        rows_report = []
        for val in sorted(buckets.keys()):
            arr = np.array(buckets[val])
            if len(arr) < MIN_TRADES_PER_PARAM_VALUE:
                rows_report.append((val, len(arr), None, None, None))
            else:
                winners = arr[arr > 0]
                losers = arr[arr < 0]
                wr = float(len(winners)) / len(arr)
                avg_win = float(winners.mean()) if len(winners) else 0.0
                avg_loss = float(losers.mean()) if len(losers) else 0.0
                expectancy = wr * avg_win + (1 - wr) * avg_loss
                rows_report.append((val, len(arr), wr, expectancy, float(arr.mean())))

        # Only bother printing if at least one bucket has usable data
        if not any(r[2] is not None for r in rows_report):
            continue

        print(f"\n  Parameter: {key}")
        for val, n, wr, exp, mean in rows_report:
            if wr is None:
                print(f"    value {val}: n={n} (below {MIN_TRADES_PER_PARAM_VALUE}, insufficient data)")
            else:
                print(f"    value {val}: n={n}, expectancy={exp*100:+.3f}%, "
                      f"win rate={wr*100:.0f}%, avg={mean*100:+.2f}%")

    print("--- End parameter-expectancy comparison ---\n")


# ---------------- Data-driven expected return by score bucket (review item #4) ----------------
# The review correctly points out that `rank_score = score / atr_pct` treats
# `score` as if it WERE an expected return, when it's really just an
# arbitrary point total -- "a score of 7" has no inherent mathematical
# meaning as "7 units of profit". There's no way to build a true statistical
# expected-return model without a real backtest, which isn't something this
# bot can run on its own live-only trade history. What it CAN do is use its
# own accumulating TradeFeatures history as a (small, continuously-improving)
# empirical substitute: once enough closed trades exist at a given score
# level, use their ACTUAL average outcome_pct as the reward term instead of
# the raw score. Below MIN_TRADES_FOR_SCORE_REWARD_ESTIMATE samples at a
# given score, or with no history at all yet, this silently falls back to the
# raw score -- so a fresh bot behaves exactly as before, and the ranking only
# gets more grounded in real outcomes as trade history accumulates.
MIN_TRADES_FOR_SCORE_REWARD_ESTIMATE = 15
_score_reward_cache = {'date': None, 'buckets': {}}  # per-day cache; rebuilt once per trading day


def get_expected_return_by_score(score):
    """
    Returns the empirical average outcome_pct for closed trades whose
    buy_score rounds to `score`, or None if there isn't enough history yet at
    that score level (caller should fall back to the raw score in that case).
    Rebuilt at most once per trading day -- this is a slow-moving statistic,
    not something that needs a fresh DB query for every candidate in every
    scan cycle.
    """
    today_str = datetime.now(eastern).date().strftime("%Y-%m-%d")
    if _score_reward_cache['date'] != today_str:
        buckets = {}
        try:
            closed = session.query(TradeFeatures).filter(TradeFeatures.outcome_pct.isnot(None)).all()
            for r in closed:
                if r.buy_score is None:
                    continue
                buckets.setdefault(int(round(r.buy_score)), []).append(r.outcome_pct)
        except Exception as e:
            logging.warning(f"get_expected_return_by_score: history query failed: {e}")
        _score_reward_cache['date'] = today_str
        _score_reward_cache['buckets'] = buckets

    outcomes = _score_reward_cache['buckets'].get(int(round(score)))
    if outcomes is None or len(outcomes) < MIN_TRADES_FOR_SCORE_REWARD_ESTIMATE:
        return None
    return float(np.mean(outcomes))




# Periodically runs the bounded auto-adjustment pass (see AdaptiveParams
# above). This DOES auto-apply changes to live trading parameters -- but only
# within the guardrails documented on AdaptiveParams: minimum sample size,
# capped step size, hard bounds, and a full audit log of every decision.
ADAPT_EVERY_N_CYCLES_MAIN_LOOP = ADAPT_EVERY_N_CYCLES  # alias for clarity in main()


def run_adaptive_parameter_pass():
    adaptive_params.run_adjustment_pass()


# =====================================================================
# INLINED STOCK SCANNER (was: stock-list-writer-for-list-of-stock-symbols-to-scan.py)
# --------------------------------------------------------------------
# The scanner is no longer a separate process and no longer writes any
# text files. It runs in-process, produces a Python list of top-scoring
# S&P 500 symbols, and stores it in the module-level SYMBOLS_TO_BUY_LIST
# cache. get_symbols_to_buy() reads that cache. Purchases NEVER mutate
# the list (remove_symbols_from_trade_list is now a no-op) so the same
# candidate universe stays available for the whole trading day.
# =====================================================================

import concurrent.futures as _scanner_futures
from collections import defaultdict as _scanner_defaultdict

_SCANNER_CONFIG = {
    'historical_years': 2,
    'lookback_years': [1, 2],
    'seasonal_years': 2,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'vwap_window': 14,  # now used only by the volume-SMA baseline; VWAP is anchored/cumulative

    'bollinger_window': 20,
    'stochastic_k': 14,
    'stochastic_d': 3,
    'adx_period': 14,
    'adx_threshold': 25,
    'min_volume_increase': 1.5,
    'min_price_increase': 0.05,
    'min_seasonal_return': 0.05,
    'batch_size_fallback': 100,
    'max_workers': 20,
    'chart_top_n': 100,
}

# Full S&P 500 candidate universe the scanner ranks against.
# S&P 500 constituents refreshed to the 2026 index (Wikipedia snapshot).
# 24 removed vs the older list: AMTM, BK, CTRA, CZR, DAY, EMN, ENPH, FI (retickered),
# HOLX, IPG, JNPR, K, KMX, LKQ, LW, MHK, MKTX, MOH, MTCH, PARA (now PSKY), PAYC,
# QRVO, WBA. 35 added: APP, ARES, BNY, CASY, CB, CHTR, CIEN, CMG, CNC, CNP, COHR,
# COR, CRH, CRL, CVNA, CVX, DDOG, EME, FIX, HOOD, IBKR, LII, LITE, PSKY, Q, SATS,
# SCHW, SNDK, TPL, VEEV, VRT, WDAY, XYZ (plus BF-B / BRK-B kept in dash form for
# yfinance, plus MMC kept as an alias for MRSH after the 2026-01-14 Marsh rebrand,
# and FISV for the 2025-11-11 Fiserv Nasdaq move).
_SCANNER_SP500_UNIVERSE = [
    'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE',
    'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG',
    'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APP', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM',
    'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'ARES', 'BKR', 'BALL', 'BAC', 'BAX',
    'BDX', 'BRK-B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO',
    'BF-B', 'BLDR', 'BG', 'BXP', 'BNY', 'CHRW', 'CDNS', 'CPT', 'CPB', 'COF', 'CAH', 'CASY', 'CCL', 'CARR', 'CAT',
    'CB', 'CBOE', 'CBRE', 'CDW', 'CEG', 'CF', 'CFG', 'CHTR', 'CHD', 'CIEN', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX',
    'CME', 'CMS', 'KO', 'CTSH', 'COHR', 'CL', 'CMCSA', 'CMG', 'CAG', 'CNC', 'CNP', 'COP', 'ED', 'STZ', 'COO',
    'CPRT', 'GLW', 'COR', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CRH', 'CRL',
    'CRWD', 'CCI', 'CSX', 'CMI', 'CVNA', 'CVS', 'CVX', 'DHR', 'DRI', 'DDOG', 'DVA', 'DECK', 'DE', 'DELL', 'DAL', 'DVN',
    'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EME', 'ETN',
    'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR',
    'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST',
    'FRT', 'FDX', 'FIS', 'FISV', 'FITB', 'FIX', 'FSLR', 'FE', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN',
    'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS',
    'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HD', 'HON', 'HOOD', 'HRL', 'HST', 'HWM',
    'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IBKR', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF',
    'IP', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM',
    'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH',
    'LRCX', 'LII', 'LITE', 'LVS', 'LDOS', 'LEN', 'LLY', 'LIN', 'LYV', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB',
    'MPC', 'MRSH', 'MMC', 'MAR', 'MLM', 'MAS', 'MA', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET',
    'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS',
    'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS',
    'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS',
    'PCAR', 'PKG', 'PLTR', 'PANW', 'PH', 'PAYX', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSKY', 'PSX',
    'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM',
    'PWR', 'Q', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL',
    'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SATS', 'SLB', 'SCHW', 'STX', 'SRE', 'NOW', 'SHW', 'SNDK',
    'SPG', 'SWKS', 'SJM', 'SW',
    'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS',
    'TROW', 'TTWO', 'TPL', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT',
    'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH',
    'UHS', 'VLO', 'VEEV', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRT', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC',
    'WRB', 'GWW',
    'WAB', 'WMT', 'WDAY', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW',
    'WYNN', 'XEL', 'XYL', 'XYZ', 'YUM', 'ZBRA', 'ZBH', 'ZTS', 'TTD', 'COIN', 'DASH', 'TKO', 'WSM', 'EXE', 'APO',
]

# Sector filter (kept identical to the original writer's excluded_sectors).
_SCANNER_EXCLUDED_SECTORS = [
    'Energy', 'Oil & Gas', 'Natural Gas', 'Utilities', 'Electricity',
    'Basic Materials', 'Financial Services', 'Financials', 'Banks', 'Insurance',
    'Consumer Cyclical', 'Consumer Discretionary', 'Healthcare', 'Medical Devices',
    'Biotechnology', 'Pharmaceuticals', 'Real Estate', 'Consumer Defensive',
    'Communication Services', 'Industrials',
]


@sleep_and_retry
@limits(calls=120, period=60)
def _scanner_fetch_sector(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info.get('sector', 'Unknown')
    except Exception as e:
        logging.warning(f"scanner: fetch_sector({symbol}) failed: {e}")
        return 'Unknown'


@sleep_and_retry
@limits(calls=20, period=60)
def _scanner_batch_download(symbols, start_date, end_date, retries=3):
    """
    Fetch 2 years of daily OHLCV for `symbols` from Alpaca IEX (free plan).

    Migration note (from yfinance): the previous implementation used
    yf.download(tickers=..., group_by='ticker') which returned a pandas
    DataFrame with a MultiIndex on columns ([ticker, field]) and was
    consumed via `all_data.get(sym, pd.DataFrame())` -> per-symbol DF
    with columns 'Open','High','Low','Close','Volume'. We preserve
    EXACTLY that consumer contract by returning a MultiIndex DataFrame
    with the same shape, so no downstream indicator code changes.

    Alpaca performance on this: ~7s for 520 tickers vs yfinance's ~9min
    for the same universe -- 78x faster in real testing. Coverage is
    97.7% complete + 2.3% partial (partial = ex-index tickers like
    FI, WBA, K -- we accept partial data for those since they're mostly
    historical anyway).

    Symbol conversion: yfinance uses '-' for share classes (BRK-B),
    Alpaca uses '.' (BRK.B). We convert on the way in and back on the
    way out so the caller keeps its yfinance-form symbol list.
    """
    if not symbols:
        return pd.DataFrame(), [], []

    # yfinance-form -> Alpaca-form; keep a reverse map to key results back.
    alpaca_syms = [to_alpaca(s) for s in symbols]
    yf_by_alpaca = dict(zip(alpaca_syms, symbols))

    # Normalize date args (yfinance accepted strings; alpaca-py wants datetime).
    def _to_dt(x):
        if hasattr(x, 'timestamp'):
            return x
        # Parse ISO date string.
        try:
            return datetime.strptime(str(x), "%Y-%m-%d")
        except Exception:
            return datetime.now() - timedelta(days=365 * 2)

    start_dt = _to_dt(start_date)
    end_dt = _to_dt(end_date)

    # Batch at 100 (matches proven test config). Alpaca handles up to a few
    # hundred per call but 100 keeps individual request payloads small and
    # gives useful progress granularity.
    BATCH_SIZE = 100
    per_symbol_bars = {}   # alpaca_sym -> list of Bar objects
    valid_yf, invalid_yf = [], []

    for attempt in range(retries):
        per_symbol_bars.clear()
        any_batch_failed = False
        for i in range(0, len(alpaca_syms), BATCH_SIZE):
            batch = alpaca_syms[i:i + BATCH_SIZE]
            try:
                req = _AlpacaBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=_AlpacaTimeFrame.Day,
                    start=start_dt,
                    end=end_dt,
                    feed=_AlpacaDataFeed.IEX,
                )
                resp = alpaca_data_client.get_stock_bars(req)
                data = resp.data if hasattr(resp, 'data') else resp
                for asym in batch:
                    bars = data.get(asym, []) if hasattr(data, 'get') else []
                    if bars:
                        per_symbol_bars[asym] = bars
            except Exception as e:
                logging.error(f"scanner: Alpaca batch {i//BATCH_SIZE + 1} "
                              f"attempt {attempt+1} failed: {e}")
                any_batch_failed = True
        if not any_batch_failed or per_symbol_bars:
            # Got at least some data; proceed. Retries are for total failure.
            break
        if attempt < retries - 1:
            time.sleep(2 * (2 ** attempt))

    # Convert Bar objects to a per-symbol pandas DataFrame with the same
    # columns yfinance produced. Then combine into a MultiIndex DataFrame
    # keyed by yfinance-form symbols so downstream code is unchanged.
    all_data_dict = {}
    for asym, bars in per_symbol_bars.items():
        yf_sym = yf_by_alpaca.get(asym, asym)
        try:
            df = pd.DataFrame([{
                'Open':   float(b.open),
                'High':   float(b.high),
                'Low':    float(b.low),
                'Close':  float(b.close),
                'Volume': float(b.volume),
            } for b in bars],
                index=pd.DatetimeIndex(
                    [b.timestamp for b in bars], name='Date'),
            )
            if not df.empty and df['Close'].dropna().size > 0:
                all_data_dict[yf_sym] = df
                valid_yf.append(yf_sym)
            else:
                invalid_yf.append(yf_sym)
        except Exception as e:
            logging.warning(f"scanner: DataFrame build failed for {yf_sym}: {e}")
            invalid_yf.append(yf_sym)

    # Symbols we requested but never got any bars for.
    for s in symbols:
        if s not in valid_yf and s not in invalid_yf:
            invalid_yf.append(s)

    if not all_data_dict:
        return pd.DataFrame(), valid_yf, invalid_yf

    # Build the MultiIndex-column DataFrame that consumers expect.
    combined = pd.concat(all_data_dict, axis=1, keys=all_data_dict.keys())
    return combined, valid_yf, invalid_yf


def _scanner_validate_clean(data):
    if data is None or data.empty:
        return None
    req = ['Close', 'High', 'Low', 'Volume']
    if not all(c in data.columns for c in req):
        return None
    data = data.copy()
    for c in req:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(subset=req)
    if len(data) < max(_SCANNER_CONFIG['rsi_period'], _SCANNER_CONFIG['adx_period']):
        return None
    return data


def _scanner_indicators(data):
    if data is None or len(data) < max(_SCANNER_CONFIG['rsi_period'], _SCANNER_CONFIG['adx_period']):
        return None
    try:
        close = np.array(data['Close'], dtype=np.float64)
        high = np.array(data['High'], dtype=np.float64)
        low = np.array(data['Low'], dtype=np.float64)
        volume = np.array(data['Volume'], dtype=np.float64)
        if np.any(np.isnan(close)) or np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(volume)):
            return None
        ind = {}
        ind['rsi'] = talib.RSI(close, timeperiod=_SCANNER_CONFIG['rsi_period'])
        ind['macd'], ind['macd_signal'], _ = talib.MACD(
            close, fastperiod=_SCANNER_CONFIG['macd_fast'],
            slowperiod=_SCANNER_CONFIG['macd_slow'],
            signalperiod=_SCANNER_CONFIG['macd_signal'])
        typical = (high + low + close) / 3
        # REAL VWAP (anchored, cumulative from the start of the lookback slice)
        # ---------------------------------------------------------------------
        # Textbook VWAP is  Σ(TypicalPrice * Volume) / Σ(Volume)  cumulated from
        # a defined anchor -- classically the session open on intraday charts,
        # or on daily charts an "anchored VWAP" from a chosen date. The old
        # implementation here computed  SMA(TP*V, n) / SMA(V, n),  which is a
        # rolling volume-weighted MOVING AVERAGE (VWMA), not VWAP: it decays
        # older bars out of the window instead of accumulating them, so it
        # never reaches the true session/period volume-weighted average price.
        #
        # We're on daily bars scored over a fixed lookback slice (see
        # _scanner_score which slices `recent = stock_data.loc[start:end]`),
        # so the natural anchor is the first bar of that slice -- i.e. an
        # anchored VWAP over the whole scoring window. The final value
        # ind['vwap'][-1] is then the true volume-weighted average price for
        # the lookback period, and comparing latest_close to it answers the
        # actual question "is the current price above the money-weighted
        # average price of every share traded in this period?"
        cum_tpv = np.cumsum(typical * volume)
        cum_v = np.cumsum(volume)
        with np.errstate(divide='ignore', invalid='ignore'):
            ind['vwap'] = np.where(cum_v > 0, cum_tpv / cum_v, np.nan)
        ind['upper_band'], ind['middle_band'], ind['lower_band'] = talib.BBANDS(
            close, timeperiod=_SCANNER_CONFIG['bollinger_window'])
        ind['slowk'], ind['slowd'] = talib.STOCH(
            high, low, close,
            fastk_period=_SCANNER_CONFIG['stochastic_k'],
            slowk_period=_SCANNER_CONFIG['stochastic_d'],
            slowd_period=_SCANNER_CONFIG['stochastic_d'])
        ind['volume_sma'] = talib.SMA(volume, timeperiod=_SCANNER_CONFIG['vwap_window'])
        ind['volume'] = volume
        ind['adx'] = talib.ADX(high, low, close, timeperiod=_SCANNER_CONFIG['adx_period'])
        ind['obv'] = talib.OBV(close, volume)
        return ind
    except Exception as e:
        logging.error(f"scanner: indicator calc failed: {e}")
        return None


def _scanner_seasonal_return(data, cur_month, cur_year):
    rets = []
    for y in range(1, _SCANNER_CONFIG['seasonal_years'] + 1):
        year = cur_year - y
        start = f"{year}-{cur_month:02d}-01"
        nxt = datetime(year, cur_month, 1) + timedelta(days=32)
        end = (nxt.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        md = data.loc[start:end]
        if not md.empty and len(md) > 1:
            rets.append((md['Close'].iloc[-1] - md['Close'].iloc[0]) / md['Close'].iloc[0])
    return float(np.mean(rets)) if rets else 0.0


def _scanner_best_month(data, cur_year):
    monthly = _scanner_defaultdict(list)
    for y in range(1, _SCANNER_CONFIG['seasonal_years'] + 1):
        year = cur_year - y
        for m in range(1, 13):
            start = f"{year}-{m:02d}-01"
            nxt = datetime(year, m, 1) + timedelta(days=32)
            end = (nxt.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
            md = data.loc[start:end]
            if not md.empty and len(md) > 1:
                monthly[m].append((md['Close'].iloc[-1] - md['Close'].iloc[0]) / md['Close'].iloc[0])
    avg = {m: np.mean(v) for m, v in monthly.items() if v}
    return max(avg, key=avg.get) if avg else None


def _scanner_score(symbol, stock_data, years_ago, cur_month, cur_year):
    if stock_data is None or stock_data.empty:
        return None
    end_date = stock_data.index[-1]
    start_date = end_date - timedelta(days=365 * years_ago)
    recent = _scanner_validate_clean(stock_data.loc[start_date:end_date])
    if recent is None:
        return None
    ind = _scanner_indicators(recent)
    if ind is None:
        return None

    score = 0.0
    latest_close = recent['Close'].iloc[-1]
    earliest_close = recent['Close'].iloc[0]
    price_inc = (latest_close - earliest_close) / earliest_close

    if price_inc >= _SCANNER_CONFIG['min_price_increase']:
        score += price_inc * 100
    latest_rsi = ind['rsi'][-1]
    if _SCANNER_CONFIG['rsi_oversold'] < latest_rsi < _SCANNER_CONFIG['rsi_overbought']:
        score += 20
    elif latest_rsi <= _SCANNER_CONFIG['rsi_oversold']:
        score += 10
    if ind['macd'][-1] > ind['macd_signal'][-1]:
        score += 15
    latest_vol = ind['volume'][-1]
    avg_vol = ind['volume_sma'][-1]
    if avg_vol > 0 and latest_vol >= avg_vol * _SCANNER_CONFIG['min_volume_increase']:
        score += 15
    if latest_close > ind['vwap'][-1]:
        score += 10
    if latest_close > ind['upper_band'][-1]:
        score += 10
    elif latest_close < ind['lower_band'][-1]:
        score += 5
    if ind['slowk'][-1] > ind['slowd'][-1] and 20 < ind['slowk'][-1] < 80:
        score += 10
    if ind['adx'][-1] > _SCANNER_CONFIG['adx_threshold'] and ind['macd'][-1] > 0:
        score += 20
    if ind['obv'][-1] > ind['obv'][0]:
        score += 10

    avg_seasonal = _scanner_seasonal_return(stock_data, cur_month, cur_year)
    if avg_seasonal > _SCANNER_CONFIG['min_seasonal_return']:
        score += avg_seasonal * 100
    best_m = _scanner_best_month(stock_data, cur_year)
    if best_m == cur_month:
        score += 50

    return {'symbol': symbol, 'score': score, 'sector': 'Unknown'}


def _scanner_process(args):
    symbol, sdata, years_ago, cm, cy = args
    try:
        return _scanner_score(symbol, sdata, years_ago, cm, cy)
    except Exception as e:
        logging.error(f"scanner: process {symbol} failed: {e}")
        return None


def _scanner_render_progress(pct, label, bar_width=40, done=False):
    """Render a single-line 1%->100% progress bar with a phase label.
    Uses \\r so the line updates in place. Clamps to [1, 100] while work is
    in progress (so the user always sees at least 1%) and prints a newline
    when done=True to finalize the line."""
    pct = max(1, min(100, int(pct)))
    filled = int(round(bar_width * pct / 100))
    bar = "#" * filled + "-" * (bar_width - filled)
    end = "\n" if done else ""
    # Pad label to a stable width so shorter labels don't leave stray chars
    # behind when the previous longer label is overwritten.
    line = f"\r[scanner] [{bar}] {pct:3d}%  {label:<38s}"
    print(line, end=end, flush=True)


def run_stock_scanner(show_progress=False):
    """Run the full scanner in-process and return the ranked list of top
    S&P 500 symbols as a Python list. NO files are written or read. Returns
    an empty list on catastrophic failure so the caller can fall back.

    When `show_progress=True`, prints a 1%->100% progress bar to stdout
    that advances across all three phases (download -> scoring -> sector
    lookup). The background daily-refresh path leaves this off so the bar
    doesn't clobber the live trading loop's output."""
    t0 = time.time()
    logging.info("scanner: starting in-process scan")

    now = datetime.now(eastern)
    cur_month, cur_year = now.month, now.year
    end_date = now.date()
    start_date = end_date - timedelta(days=365 * _SCANNER_CONFIG['historical_years'])

    # ---- Phase weights (must sum to 100) ------------------------------------
    # Chosen so the bar advances at roughly the right pace for each phase's
    # real wall-clock cost: the batch download is a single blocking call
    # (small slice), the scoring loop dominates (biggest slice), and the
    # sector fetch is a slower per-symbol pass (medium slice).
    W_DOWNLOAD, W_SCORE, W_SECTOR = 10, 70, 20

    if show_progress:
        _scanner_render_progress(1, "downloading S&P 500 daily bars...")

    stocks = list(_SCANNER_SP500_UNIVERSE)
    all_data, valid, invalid = _scanner_batch_download(stocks, start_date, end_date)
    if not valid:
        if show_progress:
            _scanner_render_progress(100, "no valid data returned.", done=True)
        logging.warning("scanner: no valid symbols after download")
        return []

    if show_progress:
        _scanner_render_progress(W_DOWNLOAD,
                                 f"downloaded {len(valid)}/{len(stocks)} symbols")

    # ---- Phase 2: scoring across all lookback windows -----------------------
    lookbacks = _SCANNER_CONFIG['lookback_years']
    total_score_tasks = max(1, len(valid) * len(lookbacks))
    completed_score_tasks = 0
    scores = []
    for years_ago in lookbacks:
        args_list = [(s, all_data.get(s, pd.DataFrame()) if hasattr(all_data, 'get') else pd.DataFrame(),
                      years_ago, cur_month, cur_year) for s in valid]
        with _scanner_futures.ThreadPoolExecutor(max_workers=_SCANNER_CONFIG['max_workers']) as ex:
            futs = {ex.submit(_scanner_process, a): a[0] for a in args_list}
            for f in _scanner_futures.as_completed(futs):
                r = f.result()
                if r and r['score'] > 0:
                    scores.append(r)
                completed_score_tasks += 1
                if show_progress:
                    frac = completed_score_tasks / total_score_tasks
                    pct = W_DOWNLOAD + int(W_SCORE * frac)
                    _scanner_render_progress(
                        pct,
                        f"scoring symbols ({completed_score_tasks}/{total_score_tasks})")

    if not scores:
        if show_progress:
            _scanner_render_progress(100, "no scored symbols.", done=True)
        logging.warning("scanner: no scored symbols")
        return []

    # ---- Phase 3: sector attach (rate-limited) ------------------------------
    total_sector_tasks = max(1, len(scores))
    completed_sector_tasks = 0
    if show_progress:
        _scanner_render_progress(W_DOWNLOAD + W_SCORE,
                                 f"fetching sectors (0/{total_sector_tasks})")

    with _scanner_futures.ThreadPoolExecutor(max_workers=10) as ex:
        sec_futs = {ex.submit(_scanner_fetch_sector, sc['symbol']): sc for sc in scores}
        for f in _scanner_futures.as_completed(sec_futs):
            sc = sec_futs[f]
            try:
                sc['sector'] = f.result()
            except Exception:
                sc['sector'] = 'Unknown'
            completed_sector_tasks += 1
            if show_progress:
                frac = completed_sector_tasks / total_sector_tasks
                pct = W_DOWNLOAD + W_SCORE + int(W_SECTOR * frac)
                _scanner_render_progress(
                    pct,
                    f"fetching sectors ({completed_sector_tasks}/{total_sector_tasks})")

    df = pd.DataFrame(scores)
    # top 25 per sector, then filter excluded sectors, then top-N overall
    top = df.groupby('sector', group_keys=False).apply(lambda x: x.nlargest(25, 'score'))
    top = top[~top['sector'].isin(_SCANNER_EXCLUDED_SECTORS)]
    top = top.nlargest(_SCANNER_CONFIG['chart_top_n'], 'score')
    result = [to_yf(str(s)) for s in top['symbol'].tolist()]

    if show_progress:
        _scanner_render_progress(100,
                                 f"done: {len(result)} symbols in {time.time()-t0:.1f}s",
                                 done=True)

    logging.info(f"scanner: produced {len(result)} symbols in {time.time()-t0:.1f}s "
                 f"(valid={len(valid)}, invalid={len(invalid)})")
    if not show_progress:
        print(f"[scanner] produced {len(result)} candidate symbols in {time.time()-t0:.1f}s")
    return result


# --- In-memory candidate list, refreshed daily on a background thread ------
SYMBOLS_TO_BUY_LIST = []                # module-level cache; purchases do NOT remove entries
_symbols_to_buy_lock = threading.Lock()
_symbols_to_buy_last_run_date = None    # date the scanner last completed
_symbols_to_buy_refresh_thread = None
_SCANNER_REFRESH_HOUR = 16              # 16:15 ET, same slot the old writer used
_SCANNER_REFRESH_MINUTE = 15


def _do_scanner_refresh(blocking=False):
    """Actually run the scanner and swap the cache. Meant to run on a
    background thread so the trading loop is never blocked. When
    `blocking=True` the caller is being made to wait for this scan
    (initial startup population, or a get_symbols_to_buy() fallback),
    so print a clearly visible please-wait banner first."""
    global SYMBOLS_TO_BUY_LIST, _symbols_to_buy_last_run_date
    if blocking:
        print("")
        print("====================================================================")
        print(" PLEASE WAIT")
        print(" The stock scanner is running to populate the symbols-to-buy list.")
        print(" This scans the full S&P 500, downloads 2 years of daily bars, and")
        print(" ranks every symbol on technical + seasonal factors. It normally")
        print(" takes a few minutes on first startup. The trading loop will begin")
        print(" as soon as the candidate list is ready -- do not close this window.")
        print("====================================================================")
        print("", flush=True)
    try:
        new_list = run_stock_scanner(show_progress=blocking)
    except Exception as e:
        logging.error(f"scanner: refresh raised: {e}")
        new_list = []
    if new_list:
        with _symbols_to_buy_lock:
            SYMBOLS_TO_BUY_LIST = new_list
            _symbols_to_buy_last_run_date = datetime.now(eastern).date()
        print(f"[scanner] SYMBOLS_TO_BUY_LIST refreshed: {len(new_list)} symbols -- ready to trade.")
    else:
        # keep the previous list intact rather than blanking the bot's universe
        logging.warning("scanner: refresh produced no symbols; keeping previous list")


def refresh_symbols_to_buy_list_if_due():
    """Called every main-loop tick. Kicks off a background refresh once
    per day after 16:15 ET, and does an immediate (synchronous) first run
    if the cache is still empty (e.g. bot just started)."""
    global _symbols_to_buy_refresh_thread, _symbols_to_buy_last_run_date

    with _symbols_to_buy_lock:
        cache_empty = len(SYMBOLS_TO_BUY_LIST) == 0
        last_date = _symbols_to_buy_last_run_date

    # First-run: populate synchronously so get_symbols_to_buy() has data.
    if cache_empty:
        _do_scanner_refresh(blocking=True)
        return

    # Scheduled daily refresh, once per day after the target time.
    now = datetime.now(eastern)
    target = now.replace(hour=_SCANNER_REFRESH_HOUR, minute=_SCANNER_REFRESH_MINUTE,
                        second=0, microsecond=0)
    due_today = now >= target and (last_date is None or last_date < now.date())
    if not due_today:
        return

    # Avoid stacking runs if a prior refresh thread is still alive.
    if _symbols_to_buy_refresh_thread is not None and _symbols_to_buy_refresh_thread.is_alive():
        return

    print("[scanner] daily refresh window reached; launching background scan")
    t = threading.Thread(target=_do_scanner_refresh,
                         name='scanner-refresh', daemon=True)
    _symbols_to_buy_refresh_thread = t
    t.start()


def get_symbols_to_buy():
    """Returns the in-memory candidate list. NO file I/O. If the cache is
    empty on first call, runs the scanner synchronously to populate it."""
    with _symbols_to_buy_lock:
        if SYMBOLS_TO_BUY_LIST:
            return list(SYMBOLS_TO_BUY_LIST)
    # empty -> populate now
    _do_scanner_refresh(blocking=True)
    with _symbols_to_buy_lock:
        if not SYMBOLS_TO_BUY_LIST:
            print("\n****  Warning: scanner produced no symbols; candidate list is empty.  ****\n")
        return list(SYMBOLS_TO_BUY_LIST)


def remove_symbols_from_trade_list(symbol):
    """DISABLED by design: purchases must NOT remove a symbol from the
    in-memory candidate list. The list is refreshed once per day by the
    scanner and is otherwise immutable during the trading day. Kept as a
    no-op so existing call sites continue to work."""
    return


# BUGFIX: this wrapper was also @limits decorated on top of get_cached_data and
# _fetch_current_price. Only _fetch_current_price actually touches the network,
# so it is the only layer that should consume the budget.
def get_current_price(symbols, retries=3):
    for attempt in range(retries):
        try:
            price = get_cached_data(symbols, 'current_price', _fetch_current_price, symbols)
            if price is not None:
                return price
        except Exception as e:
            logging.error(f"Retry {attempt + 1}/{retries} failed for {symbols}: {e}")
            time.sleep(2 ** attempt)
    return None


def _last_close(symbol):
    """BUGFIX: took a pre-built Ticker and called .history() on it directly,
    bypassing the rate gate. It is a fallback path, so on a bad day it fired for
    EVERY symbol -- doubling real yfinance traffic invisibly. Now gated."""
    try:
        h = yf_history(symbol, period='1d')
        if h.empty:
            return None
        return float(h['Close'].iloc[-1])
    except Exception:
        return None


def _fetch_current_price(symbols):
    """Primary: Alpaca latest IEX trade (100ms typical). Fallback: yfinance
    1-min bar last close (previous behavior, kept for resilience if Alpaca
    is temporarily unreachable or a symbol has no recent IEX print).

    Migration note: was a two-step yfinance-only fetch (1m bar + last-close
    fallback). Now Alpaca-first + yfinance fallback. The 2-min cache in
    get_cached_data still applies, so hit rate is unchanged."""
    yf_symbol = to_yf(symbols)
    # ---- Primary: Alpaca IEX latest trade ----
    p = _alpaca_latest_trade_single(symbols)
    if p is not None and p > 0:
        return round(p, 4)

    # ---- Fallback: yfinance (original path) ----
    now = datetime.now(eastern)
    t = now.time()
    current_price = None
    try:
        if time2(4, 0) <= t < time2(20, 0):
            prepost = not (time2(9, 30) <= t < time2(16, 0))
            data = yf_history(symbols, period='1d', interval='1m', prepost=prepost)
            if not data.empty:
                current_price = float(data['Close'].iloc[-1])
        if current_price is None:
            current_price = _last_close(symbols)
    except Exception as e:
        logging.error(f"Error fetching current price for {yf_symbol}: {e}")
        current_price = _last_close(symbols)

    if current_price is None:
        logging.error(f"Failed to retrieve current price for {yf_symbol}.")
        return None
    return round(current_price, 4)


def _fetch_atr(sym):
    """22-period ATR from 60 days of daily bars.
    Primary: Alpaca IEX daily bars. Fallback: yfinance yf_history."""
    yf_symbol = to_yf(sym)
    try:
        data = _alpaca_daily_bars_single(sym, days=60)
        if data.empty or len(data) < 23:
            # Fallback to yfinance if Alpaca didn't give us enough bars
            # (rare -- only happens on the 12 partial-coverage tickers).
            data = yf_history(sym, period='60d')
        if len(data) < 23:
            return None
        atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=22)
        val = atr[-1]
        # BUGFIX: reject NaN/zero ATR which produced div-by-zero position sizes
        if val is None or not np.isfinite(val) or val <= 0:
            return None
        return float(val)
    except Exception as e:
        logging.error(f"Error calculating ATR for {yf_symbol}: {e}")
        return None


# BUGFIX: same nesting problem -- this wrapper only delegates to get_cached_data,
# whose _fetch_atr does the real network call. Don't double-count the budget.
def get_average_true_range(symbols):
    return get_cached_data(symbols, 'atr', _fetch_atr, symbols)


def get_atr_high_price(sym):
    atr = get_average_true_range(sym)
    cp = get_current_price(sym)
    return round(cp + 0.40 * atr, 4) if cp and atr else None


def get_atr_low_price(sym):
    atr = get_average_true_range(sym)
    cp = get_current_price(sym)
    return round(cp - 0.10 * atr, 4) if cp and atr else None


def is_in_uptrend(symbols_to_buy):
    # BUGFIX: refetched a full 1y of daily bars EVERY cycle to compute a
    # 200-day SMA that moves once a day. Now cached for 30m (CACHE_TTLS).
    yf_symbol = to_yf(symbols_to_buy)

    def _fetch_sma(sym):
        h = yf_history(sym, period='1y')
        if h.empty or len(h) < 200:
            return None
        return float(talib.SMA(h['Close'].values, timeperiod=200)[-1])

    sma_200 = get_cached_data(symbols_to_buy, 'uptrend', _fetch_sma, symbols_to_buy)
    if sma_200 is None:
        return False
    cp = get_current_price(symbols_to_buy)
    if cp is None or not np.isfinite(sma_200):
        return False
    return cp > sma_200


def get_daily_rsi(symbols_to_buy):
    # BUGFIX: refetched 60d of daily bars every cycle for a daily RSI. Cached 30m.
    def _fetch_rsi(sym):
        h = yf_history(sym, period='60d', interval='1d')
        if h.empty or len(h) < 15:
            return None
        r = talib.RSI(h['Close'].values, timeperiod=14)[-1]
        return round(float(r), 2) if np.isfinite(r) else None

    return get_cached_data(symbols_to_buy, 'daily_rsi', _fetch_rsi, symbols_to_buy)


# ---------------- Multi-timeframe confirmation (review item #4) ----------------
# Require the daily trend (already gated by is_in_uptrend/get_daily_rsi), the
# 60-minute trend, and a 5-minute reversal signal to all agree bullish before
# buying. This is intended to cut false entries where the daily picture looks
# fine but the stock is actively falling on a shorter timeframe right now.
def get_60min_trend_bullish(symbol):
    """60m bars: bullish if price is above its own 20-bar (≈20h) SMA."""
    def _fetch(sym):
        h = yf_history(sym, period='5d', interval='60m')
        if h.empty or len(h) < 20:
            return None
        close = h['Close'].values
        sma20 = talib.SMA(close, timeperiod=20)[-1]
        if not np.isfinite(sma20):
            return None
        return bool(close[-1] > sma20)

    result = get_cached_data(symbol, 'mtf_60m', _fetch, symbol)
    # Missing/short history: don't block the trade on a data gap, but don't
    # count it as confirmation either -- treat as neutral-pass.
    return True if result is None else result


def get_5min_reversal_bullish(symbol):
    """5m bars: bullish if the latest 5m close ticked up from the prior bar
    (a simple reversal-in-progress check for the shortest timeframe)."""
    def _fetch(sym):
        h = yf_history(sym, period='1d', interval='5m')
        if h.empty or len(h) < 3:
            return None
        close = h['Close'].values
        return bool(close[-1] >= close[-2])

    result = get_cached_data(symbol, 'mtf_5m', _fetch, symbol)
    return True if result is None else result


def multi_timeframe_confirms_bullish(symbol):
    """Daily trend is already checked by the caller (is_in_uptrend + daily RSI
    gate) before this runs; here we require the 60m and 5m timeframes to agree."""
    return get_60min_trend_bullish(symbol) and get_5min_reversal_bullish(symbol)


# ---------------- Relative strength (review item #3) ----------------
def get_relative_strength(symbol, benchmark='SPY'):
    """
    20-day return of `symbol` minus 20-day return of `benchmark`. Positive
    means the stock has been outperforming the benchmark over that window --
    a supplementary scanner signal, not a hard gate.
    """
    def _fetch(sym):
        try:
            sym_h = yf_history(sym, period='30d', interval='1d')
            bench_h = yf_history(benchmark, period='30d', interval='1d')
            if len(sym_h) < 21 or len(bench_h) < 21:
                return None
            sym_ret = float(sym_h['Close'].iloc[-1] / sym_h['Close'].iloc[-21] - 1)
            bench_ret = float(bench_h['Close'].iloc[-1] / bench_h['Close'].iloc[-21] - 1)
            return round(sym_ret - bench_ret, 4)
        except Exception as e:
            logging.warning(f"Relative strength fetch failed for {sym}: {e}")
            return None

    return get_cached_data(symbol, 'relative_strength', _fetch, symbol)


# ---------------- Earnings-date filter (review item #3) ----------------
EARNINGS_BLACKOUT_DAYS = 2  # skip new buys within this many days of an earnings print


def days_until_next_earnings(symbol):
    """
    Returns days until the next known earnings date, or None if unknown.
    Uses yfinance's calendar endpoint, which is not rate-gated the same way as
    price history, but we still route it through the shared cache to avoid
    hammering it every cycle.
    """
    def _fetch(sym):
        try:
            yf_gate.acquire()
            with yf_lock:
                cal = yf.Ticker(to_yf(sym)).calendar
            if not cal:
                return None
            # yfinance returns either a dict with 'Earnings Date' (list of dates)
            # or a DataFrame depending on version; handle both defensively.
            edate = None
            if isinstance(cal, dict):
                dates = cal.get('Earnings Date')
                if dates:
                    edate = dates[0] if isinstance(dates, (list, tuple)) else dates
            else:
                try:
                    edate = cal.loc['Earnings Date'].iloc[0]
                except Exception:
                    edate = None
            if edate is None:
                return None
            if hasattr(edate, 'date'):
                edate = edate.date()
            days = (edate - datetime.now(eastern).date()).days
            return int(days)
        except Exception as e:
            logging.info(f"Earnings date lookup failed for {sym}: {e}")
            return None

    return get_cached_data(symbol, 'earnings_date', _fetch, symbol)


def is_within_earnings_blackout(symbol):
    days = days_until_next_earnings(symbol)
    if days is None:
        return False  # unknown: don't block the trade on missing data
    return 0 <= days <= EARNINGS_BLACKOUT_DAYS


def calculate_technical_indicators(symbols, lookback_days=90):
    """Primary: Alpaca IEX daily bars. Fallback: yfinance yf_history.
    Adds macd, signal, rsi, volume columns via talib."""
    yf_symbol = to_yf(symbols)
    hist = _alpaca_daily_bars_single(symbols, days=lookback_days)
    if hist.empty or len(hist) < 35:
        hist = yf_history(symbols, period=f'{lookback_days}d')
    if hist.empty or len(hist) < 35:
        return hist
    hist['macd'], hist['signal'], _ = talib.MACD(hist['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    hist['rsi'] = talib.RSI(hist['Close'].values, timeperiod=14)
    hist['volume'] = hist['Volume']
    return hist


def print_technical_indicators(symbols, historical_data):
    if historical_data is None or historical_data.empty:
        return
    cols = [c for c in ['Close', 'macd', 'signal', 'rsi', 'volume'] if c in historical_data.columns]
    print(f"\nTechnical Indicators for {symbols}:\n")
    print(historical_data[cols].tail())
    print("")


def get_previous_price(symbols):
    if symbols in previous_prices:
        return previous_prices[symbols]
    cp = get_current_price(symbols)
    if cp is not None:
        previous_prices[symbols] = cp
    return cp


def update_previous_price(symbols, current_price):
    if current_price is not None:
        previous_prices[symbols] = current_price


# ---------------- 2026 Margin rules engine ----------------
def get_margin_state():
    """Replaces PDT checks with margin-account health metrics."""
    acct = api.get_account()
    equity = float(acct.equity)
    last_equity = float(acct.last_equity or equity)
    cash = float(acct.cash)
    buying_power = float(acct.buying_power)
    dt_bp = float(getattr(acct, 'daytrading_buying_power', 0) or 0)
    rt_bp = float(getattr(acct, 'regt_buying_power', 0) or 0)
    long_mv = float(getattr(acct, 'long_market_value', 0) or 0)
    maint = float(getattr(acct, 'maintenance_margin', 0) or 0)

    # Effective purchasing power under our own leverage cap, not FINRA's PDT rule.
    if ACCOUNT_MODE == 'margin':
        effective_bp = min(buying_power, equity * MAX_LEVERAGE)
    else:
        effective_bp = min(cash, equity * MAX_LEVERAGE)

    margin_ratio = (equity / long_mv) if long_mv > 0 else 1.0
    healthy = margin_ratio >= MAINTENANCE_MARGIN_FLOOR_PCT

    return {
        'equity': equity, 'last_equity': last_equity, 'cash': cash,
        'buying_power': buying_power, 'daytrading_buying_power': dt_bp,
        'regt_buying_power': rt_bp, 'long_market_value': long_mv,
        'maintenance_margin': maint, 'effective_bp': effective_bp,
        'margin_ratio': margin_ratio, 'healthy': healthy,
        'trading_blocked': bool(acct.trading_blocked),
        'account_blocked': bool(acct.account_blocked),
    }


def day_trades_allowed():
    """2026 rules: no PDT round-trip counter. Only broker-level blocks matter."""
    if UNLIMITED_DAY_TRADES:
        st = get_margin_state()
        return not (st['trading_blocked'] or st['account_blocked'])
    return True


# ---------------- Open design questions (documented, not changed) ----------------
# A code review raised several points that are genuine judgment calls about
# how the strategy should work, not bugs -- changing them without a backtest
# would be guessing at new behavior rather than fixing broken behavior. Noted
# here so the reasoning isn't lost, without silently acting on it:
#
# Item 3 (scoring quantitativeness): the buy score combines RSI-below-50,
# rising MACD, a bullish 60m trend, and a bullish 5m tick into one additive
# point total. The reviewer called this a coherent CONCEPT (long-term
# bullish + short-term oversold + beginning to reverse) but said the
# implementation should be "more quantitative" than a collection of loosely
# related indicators accumulating points -- e.g. a proper multi-factor model
# with fitted/backtested weights instead of hand-picked integers. The
# regime-weighted scoring (REGIME_SIGNAL_WEIGHTS) is a step toward that, and
# the AdaptiveParams auto-tuner now nudges per-signal weights based on this
# bot's own trade outcomes -- but neither is a real quantitative model, and
# building one properly needs historical backtesting data this bot doesn't
# have access to on its own.
#
# Item 8 (profit-monitor tightness / parameter choices in general): the
# reviewer's specific, actionable suggestion here (giveback should scale with
# peak gain, not just the arm threshold) IS implemented -- see
# PEAK_GIVEBACK_FRACTION and ProfitMonitorEngine._giveback_for_peak(). The
# broader point -- that ARM/GIVEBACK/FLOOR defaults are "extremely tight" and
# sacrifice larger moves -- is a return-vs-frequency tradeoff with no
# objectively correct answer; the current constants (ARM_PROFIT_PCT,
# PEAK_GIVEBACK_PCT, HARD_FLOOR_PCT near the top of this file) are left at
# their existing values rather than guessed wider, since "wider" is not
# unambiguously better without data on how it performs live.
#
# Item 10 (walk-forward validation): AdaptiveParams already does a bounded,
# guardrailed form of this for the buy-score threshold and signal weights
# (min sample size, capped step size, hard bounds -- see AdaptiveParams
# class). A full walk-forward optimization across the wider parameter space
# the reviewer describes (ARM/GIVEBACK/FLOOR, ATR multipliers, position
# sizing) would need a proper historical backtesting harness, which is a
# meaningfully larger project than a live-only bot can bootstrap from its own
# trade history alone.


def compute_buy_score(df, current_price, previous_price, last_price, regime=None, weights=None,
                      intraday_pullback_pct=None, pullback_5m_pct=None,
                      pullback_30m_pct=None, swing_high_distance_pct=None,
                      vwap_distance_pct=None):
    """
    BUGFIX: score is computed once, in one place, from clean booleans.
    Previously `score` was accumulated in two disconnected blocks with
    contradictory thresholds (`< 3` then `>= 3` with a `< 4` message).

    REVIEW ITEM #1: signal contributions are now weighted by market regime
    instead of always adding a flat +1/+2. A caller can pass `regime`/`weights`
    explicitly; otherwise the current live regime is looked up.

    REVIEW ITEM #2 (dip measurement): `price_decline` below compares the
    current price to the latest DAILY close, which measures "below
    yesterday's close" rather than "pulled back from where it's been
    trading today" -- a stock that ran up then pulled back intraday (e.g.
    $100 -> $99.80 -> $99.95) can still count as a "decline" against
    yesterday's close even though it's no longer near its intraday low.
    `intraday_pullback_pct`, when the caller supplies it (see
    price_history-based calculation in buy_stocks), measures the pullback
    from the recent INTRADAY high instead and is blended in as an additional
    signal alongside the existing daily-close measurement, not a replacement
    for it -- both can fire independently and both contribute to the score.
    """
    close = df['Close'].values
    open_ = df['Open'].values
    high = df['High'].values
    low = df['Low'].values

    if weights is None:
        weights = get_regime_weights(regime)

    reasons = []
    score = 0

    # --- Candlestick bullish reversal detection (most recent bar only) ---
    pattern_funcs = {
        'Hammer': talib.CDLHAMMER,
        'Bullish Engulfing': talib.CDLENGULFING,
        'Morning Star': talib.CDLMORNINGSTAR,
        'Piercing Line': talib.CDLPIERCING,
        'Three White Soldiers': talib.CDL3WHITESOLDIERS,
        'Dragonfly Doji': talib.CDLDRAGONFLYDOJI,
        'Inverted Hammer': talib.CDLINVERTEDHAMMER,
        'Tweezer Bottom': talib.CDLMATCHINGLOW,
    }
    detected = []
    for name, fn in pattern_funcs.items():
        try:
            res = fn(open_, high, low, close)
            # BUGFIX: require a BULLISH (>0) signal. The original accepted
            # `!= 0`, which let bearish (-100) prints count as buy signals.
            if len(res) and res[-1] > 0:
                detected.append(name)
        except Exception:
            continue

    if detected:
        score += weights.get('pattern', 2)
        reasons.append(f"patterns={','.join(detected)}")

    # --- RSI ---
    rsi_series = talib.RSI(close, timeperiod=14)
    latest_rsi = float(rsi_series[-1]) if len(rsi_series) and np.isfinite(rsi_series[-1]) else None
    rsi_decrease = False
    recent_avg_rsi = prior_avg_rsi = 0.0
    if len(rsi_series) >= 10:
        recent = rsi_series[-5:][np.isfinite(rsi_series[-5:])]
        prior = rsi_series[-10:-5][np.isfinite(rsi_series[-10:-5])]
        if len(recent) and len(prior):
            recent_avg_rsi, prior_avg_rsi = float(np.mean(recent)), float(np.mean(prior))
            rsi_decrease = recent_avg_rsi < prior_avg_rsi
    if latest_rsi is not None and latest_rsi < 50:
        score += weights.get('rsi_below_50', 1)
        reasons.append(f"rsi={latest_rsi:.1f}<50")
    if rsi_decrease:
        score += weights.get('rsi_falling', 1)
        reasons.append("rsi_falling")

    # --- Volume ---
    recent_avg_volume = float(df['Volume'].iloc[-5:].mean()) if len(df) >= 5 else 0.0
    prior_avg_volume = float(df['Volume'].iloc[-10:-5].mean()) if len(df) >= 10 else recent_avg_volume
    volume_decrease = recent_avg_volume < prior_avg_volume if len(df) >= 10 else False
    if not volume_decrease:
        score += weights.get('volume_holding', 1)
        reasons.append("volume_holding")

    # --- MACD ---
    macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_above_signal = False
    if len(macd) and np.isfinite(macd[-1]) and np.isfinite(macd_signal[-1]):
        macd_above_signal = macd[-1] > macd_signal[-1]
    if macd_above_signal:
        score += weights.get('macd_above_signal', 1)
        reasons.append("macd>signal")

    # --- Price decline (BUGFIX: numeric magnitude, not a bool compared to a price) ---
    decline_pct = (last_price - current_price) / last_price if last_price else 0.0
    price_decline = decline_pct >= 0.002
    if price_decline:
        score += weights.get('price_decline', 1)
        reasons.append(f"dip={decline_pct*100:.2f}%")

    # --- Intraday pullback from recent peak (review item #2, additional signal) ---
    # Distinct from price_decline above: this measures how far current_price
    # has pulled back from the highest price seen so far TODAY in this
    # symbol's rolling intraday history, not from yesterday's daily close.
    # A stock sitting well below today's high reads as a real intraday
    # pullback even on a day where it's still net positive vs. yesterday's
    # close (which price_decline alone would miss entirely).
    intraday_pullback = intraday_pullback_pct is not None and intraday_pullback_pct >= 0.002
    if intraday_pullback:
        score += weights.get('intraday_pullback', 1)
        reasons.append(f"intraday_pullback={intraday_pullback_pct*100:.2f}%")

    # --- Multi-timeframe pullback signals ---
    # 5-minute pullback: recent short-term dip, most relevant for a fast
    # reversal entry. Distinct from 1-min intraday_pullback: covers a slightly
    # longer window (up to ~250 minutes of trailing 5-min samples).
    pullback_5m = pullback_5m_pct is not None and pullback_5m_pct >= 0.003
    if pullback_5m:
        score += weights.get('pullback_5m', 1)
        reasons.append(f"pullback_5m={pullback_5m_pct*100:.2f}%")

    # 30-minute pullback: intermediate-timeframe pullback, filters out
    # ultra-fast noise while still being "today's move."
    pullback_30m = pullback_30m_pct is not None and pullback_30m_pct >= 0.005
    if pullback_30m:
        score += weights.get('pullback_30m', 1)
        reasons.append(f"pullback_30m={pullback_30m_pct*100:.2f}%")

    # Distance from recent swing high (uses the 60-minute samples buffer as
    # the "session-scale" high proxy). Larger distances = deeper pullback.
    swing_high_distance = swing_high_distance_pct is not None and swing_high_distance_pct >= 0.008
    if swing_high_distance:
        score += weights.get('swing_high_distance', 1)
        reasons.append(f"below_swing_high={swing_high_distance_pct*100:.2f}%")

    # Distance from VWAP-proxy (intraday mean of 1-min samples). Trading BELOW
    # VWAP is the classic dip-buy setup: the stock is currently offered at a
    # discount to what the average buyer paid today.
    below_vwap = vwap_distance_pct is not None and vwap_distance_pct >= 0.002
    if below_vwap:
        score += weights.get('vwap_distance', 1)
        reasons.append(f"below_vwap={vwap_distance_pct*100:.2f}%")

    # --- Pattern-specific confirmations ---
    pattern_bonus_w = weights.get('pattern_bonus', 1)
    for p in detected:
        if p == 'Hammer' and latest_rsi is not None and latest_rsi < 35 and decline_pct >= 0.003:
            score += pattern_bonus_w
        elif p == 'Bullish Engulfing' and prior_avg_volume and recent_avg_volume > 1.5 * prior_avg_volume:
            score += pattern_bonus_w
        elif p == 'Morning Star' and latest_rsi is not None and latest_rsi < 40:
            score += pattern_bonus_w
        elif p == 'Piercing Line' and recent_avg_rsi and recent_avg_rsi < 40:
            score += pattern_bonus_w
        elif p == 'Three White Soldiers' and not volume_decrease:
            score += pattern_bonus_w
        elif p == 'Dragonfly Doji' and latest_rsi is not None and latest_rsi < 30:
            score += pattern_bonus_w
        elif p == 'Inverted Hammer' and rsi_decrease:
            score += pattern_bonus_w
        elif p == 'Tweezer Bottom' and latest_rsi is not None and latest_rsi < 40:
            score += pattern_bonus_w

    return {
        'score': score, 'detected': detected, 'reasons': reasons,
        'latest_rsi': latest_rsi, 'rsi_decrease': rsi_decrease,
        'volume_decrease': volume_decrease, 'macd_above_signal': macd_above_signal,
        'price_decline': price_decline, 'decline_pct': decline_pct,
        'intraday_pullback': intraday_pullback, 'intraday_pullback_pct': intraday_pullback_pct,
    }


# Fallback threshold used only if regime lookup fails entirely (see
# get_buy_score_threshold). The live threshold is now dynamic per-regime, held
# in AdaptiveParams and auto-adjusted within guardrails (see AdaptiveParams).
BUY_SCORE_THRESHOLD_DEFAULT = 4


def _bull_batch_sample_prices(symbols):
    """
    Fetch the latest live trade for ALL bull-monitored symbols in ONE
    Alpaca REST call, using the IEX feed (free plan).

    Migration note (from yfinance batched 1m bar scrape): the previous
    implementation called yf_download_batch(symbols, period="1d",
    interval="1m") and took the last bar's close. That was working but
    IEX latest-trade is:
      - 5x-10x lower per-call latency (~100ms vs ~1s+)
      - Actual last print, not a 1-minute-lagged bar close
      - No yfinance rate-limit risk
      - Same batch-in-one-call model (still N=1 HTTP request per sample
        round regardless of symbol count)

    Returns {symbol: last_trade_price_float} keyed by the caller's
    ORIGINAL yfinance-form symbol (BRK-B, not BRK.B). Symbols with no
    IEX print available are omitted -- caller treats those as a missed
    sample for that round only.

    Symbol conversion: bot uses yfinance form ('-'), Alpaca uses '.',
    so we translate on the way in and reverse-map on the way out.
    """
    if not symbols:
        return {}

    alpaca_syms = [to_alpaca(s) for s in symbols]
    yf_by_alpaca = dict(zip(alpaca_syms, symbols))

    try:
        req = _AlpacaLatestTradeRequest(
            symbol_or_symbols=alpaca_syms,
            feed=_AlpacaDataFeed.IEX,
        )
        resp = alpaca_data_client.get_stock_latest_trade(req)
    except Exception as e:
        logging.warning(f"[bull] Alpaca latest-trade batch failed: {e}")
        return {}

    out = {}
    for asym, trade in resp.items():
        try:
            p = float(trade.price)
            if p > 0:
                yf_sym = yf_by_alpaca.get(asym, asym)
                out[yf_sym] = round(p, 4)
        except Exception:
            continue
    return out


def _bull_apply_sample(sym, p, acc):
    """
    Fold one new price sample `p` for symbol `sym` into its accumulator dict
    `acc[sym]`. Creates the accumulator on first sample. Updates tick counts,
    first/last price, running peak, and max drawdown.
    """
    if sym not in acc:
        acc[sym] = {
            'increased': 0,
            'decreased': 0,
            'samples': 0,
            'first_price': None,
            'last_price': None,
            'peak': None,
            'max_drawdown': 0.0,
            '_prev_price': None,
        }
    a = acc[sym]
    a['samples'] += 1
    if a['first_price'] is None:
        a['first_price'] = p
    a['last_price'] = p
    prev = a['_prev_price']
    if prev is not None:
        if p > prev:
            a['increased'] += 1
        elif p < prev:
            a['decreased'] += 1
    a['_prev_price'] = p
    if a['peak'] is None or p > a['peak']:
        a['peak'] = p
    else:
        dd = (a['peak'] - p) / a['peak']
        if dd > a['max_drawdown']:
            a['max_drawdown'] = dd


def _bull_batch_monitor(symbols, duration, interval):
    """
    Batch-driven bull monitor: ONE thread loops every `interval` seconds and
    calls _bull_batch_sample_prices(symbols) to pull all N symbols in a
    single HTTP request. Distributes each round's results into per-symbol
    accumulators.

    Replaces the previous N-parallel-threads-each-fetching-one-symbol design.
    Total HTTP calls per run = duration / interval (regardless of N), which
    scales flat with symbol count instead of linearly.

    Returns {symbol: {'increased', 'decreased', 'samples', 'first_price',
    'last_price', 'peak', 'max_drawdown'}}.
    """
    acc = {}
    stop_at = time.time() + duration
    round_count = 0
    while time.time() < stop_at:
        round_start = time.time()
        prices = _bull_batch_sample_prices(symbols)
        round_count += 1
        for sym, p in prices.items():
            _bull_apply_sample(sym, p, acc)

        # Sleep the remainder of this interval so we get an even sample
        # cadence regardless of how long the batch call took.
        elapsed = time.time() - round_start
        remaining_run = stop_at - time.time()
        if remaining_run <= 0:
            break
        time.sleep(max(0.0, min(interval - elapsed, remaining_run)))

    # Strip the internal _prev_price key before returning.
    for a in acc.values():
        a.pop('_prev_price', None)
    print(f"[bull] batch monitor completed: {round_count} sample rounds, "
          f"{len(acc)} symbols with data.")
    return acc


def _in_bull_buy_window(now_et):
    """True if `now_et` is between BULL_BUY_WINDOW_START and BULL_BUY_WINDOW_END."""
    start_h, start_m = BULL_BUY_WINDOW_START
    end_h, end_m = BULL_BUY_WINDOW_END
    start = now_et.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now_et.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return start <= now_et <= end


def _bull_market_scan_candidates(symbols_to_buy_list, regime, total_symbols_for_alloc):
    """
    Bull Market Advanced Stock Market Trading Robot v5 buy path.

    Runs only when regime == REGIME_BULL, USE_BULL_MARKET_STRATEGY is on, and
    the current time falls inside the bull-buy window (10:02 AM - 3:56 PM ET
    by default). Every symbol in `symbols_to_buy_list` is monitored IN
    PARALLEL for BULL_MONITOR_SECONDS (default 180s) with samples taken every
    BULL_SAMPLE_INTERVAL seconds, counting price up-ticks vs down-ticks. A
    symbol becomes a candidate iff ALL of these fire:
      - increases >= BULL_MIN_PRICE_INCREASES (default 3)
      - increases > decreases
      - Daily MACD signal line > BULL_MACD_SIGNAL_MIN (default 0.15)
      - Daily RSI(14) > BULL_RSI_MIN (default 70)
      - Latest daily volume > BULL_VOLUME_MULT_OF_MEAN x 90d mean volume

    NOTE ON TIMING: the monitoring pass makes this scan block for
    BULL_MONITOR_SECONDS + a couple seconds of overhead. The buy_stocks
    thread is the sole caller and it's already a background worker, so
    sell_stocks (which runs in parallel on its own thread) is not blocked.

    Returns candidate dicts in the same shape buy_stocks Phase 2 expects,
    tagged 'bull_strategy': True. Symbols are NOT position_book.claim()'d
    here -- the main buy_stocks loop is the single owner of claims and will
    de-dupe by api_symbol before running Phase 2.
    """
    if not USE_BULL_MARKET_STRATEGY or regime != REGIME_BULL or not symbols_to_buy_list:
        return []

    now_et = datetime.now(eastern)
    if not _in_bull_buy_window(now_et):
        print(f"[bull] outside bull buy window "
              f"{BULL_BUY_WINDOW_START[0]:02d}:{BULL_BUY_WINDOW_START[1]:02d}-"
              f"{BULL_BUY_WINDOW_END[0]:02d}:{BULL_BUY_WINDOW_END[1]:02d} ET "
              f"(now {now_et.strftime('%H:%M')}); skipping bull scan.")
        return []

    # Equal-cash allocation across the symbols the bot is currently scanning,
    # capped at the per-symbol max. Uses the total scanner universe as the
    # denominator so a small candidate count doesn't concentrate the account.
    try:
        cash_available = float(api.get_account().cash)
    except Exception as e:
        logging.warning(f"[bull] could not fetch cash for allocation: {e}")
        return []

    denom = max(total_symbols_for_alloc, 1)
    bull_alloc = min(BULL_MAX_ALLOCATION_PER_SYMBOL, cash_available / denom)
    bull_alloc = round(bull_alloc, 2)
    if bull_alloc < MIN_ORDER_NOTIONAL:
        # Not enough cash to fund even one bull buy at the floor.
        return []

    print(f"[bull] regime=BULL, in-window ({now_et.strftime('%H:%M')} ET); "
          f"per-symbol allocation ${bull_alloc:.2f} "
          f"(cash ${cash_available:.2f} / {denom} symbols, cap ${BULL_MAX_ALLOCATION_PER_SYMBOL:.2f})")

    # ---------------- Phase A: parallel price-tick monitoring ----------------
    # Rate-limit sanity check. Every symbol wants BULL_MONITOR_SECONDS /
    # ---------------- Phase A: batched price-tick monitoring ----------------
    # One coordinator thread pulls ALL symbols in a single yf.download batch
    # every BULL_SAMPLE_INTERVAL seconds, so total HTTP call volume is
    # constant in symbol count: BULL_MONITOR_SECONDS/BULL_SAMPLE_INTERVAL
    # requests regardless of whether we're monitoring 1 symbol or 50.
    # At the defaults (180s / 10s) that's 18 total requests per bull scan.
    n_syms = len(symbols_to_buy_list)
    print(f"[bull] batch-monitoring {n_syms} symbol(s) for "
          f"{BULL_MONITOR_SECONDS}s (one batched sample every {BULL_SAMPLE_INTERVAL}s "
          f"= {BULL_MONITOR_SECONDS // BULL_SAMPLE_INTERVAL} HTTP calls total)...")
    tick_results = _bull_batch_monitor(
        list(symbols_to_buy_list), BULL_MONITOR_SECONDS, BULL_SAMPLE_INTERVAL)
    print(f"[bull] tick monitoring complete; evaluating gates for "
          f"{len(tick_results)} symbol(s).")

    # ---------------- Phase B: apply the full v5 gate stack ----------------
    bull_candidates = []
    for symbol in list(symbols_to_buy_list):
        yf_symbol = to_yf(symbol)
        api_symbol = to_alpaca(symbol)
        try:
            ticks = tick_results.get(symbol)
            if ticks is None:
                print(f"[bull] {yf_symbol}: no tick data. Skipping.")
                continue

            increased = ticks['increased']
            decreased = ticks['decreased']
            samples = ticks['samples']
            first_price = ticks['first_price']
            current_price = ticks['last_price'] or get_current_price(symbol)
            max_dd = ticks['max_drawdown']

            if current_price is None or current_price <= 0:
                continue

            # v5 gate 0 (new): enough real samples to trust the counts.
            # Without this a symbol with 2 successful fetches (both up) would
            # trivially pass the "increased >= 3" and "increased > decreased"
            # tests below since we require inc >= 3 anyway -- but we ALSO
            # need enough coverage that a quiet down-move wasn't just missed
            # by yfinance timeouts.
            if samples < BULL_MIN_SAMPLES:
                print(f"[bull] {yf_symbol}: only {samples} live samples "
                      f"(need >= {BULL_MIN_SAMPLES}). Skipping.")
                continue

            # v5 gate 1: tick counts (unchanged, kept per user request)
            if not (increased >= BULL_MIN_PRICE_INCREASES and increased > decreased):
                print(f"[bull] {yf_symbol}: tick gate failed "
                      f"(inc={increased}, dec={decreased}, need inc>={BULL_MIN_PRICE_INCREASES} & inc>dec). Skipping.")
                continue

            # NEW gate 1a: net return over the monitoring window must be
            # positive by at least BULL_MIN_NET_RETURN. This closes the
            # "3 penny up-ticks beat 2 big down-ticks" loophole in the
            # count-only rule -- a stock that netted DOWN over 3 minutes
            # is not what we want to buy, even if inc > dec by count.
            if first_price is None or first_price <= 0:
                print(f"[bull] {yf_symbol}: no first_price recorded. Skipping.")
                continue
            net_return = (current_price - first_price) / first_price
            if net_return < BULL_MIN_NET_RETURN:
                print(f"[bull] {yf_symbol}: net-return gate failed "
                      f"(net={net_return*100:.3f}%, need >= {BULL_MIN_NET_RETURN*100:.3f}%). Skipping.")
                continue

            # NEW gate 1b: intra-window max drawdown from any running peak
            # must stay under BULL_MAX_MONITOR_DRAWDOWN. Kills the classic
            # pump-and-fade shape where the stock spikes at second 30 and
            # bleeds back over the next 2.5 minutes -- the count rule
            # can still pass this (many small up-ticks on the way down)
            # but drawdown-from-peak catches it directly.
            if max_dd > BULL_MAX_MONITOR_DRAWDOWN:
                print(f"[bull] {yf_symbol}: drawdown gate failed "
                      f"(max_dd={max_dd*100:.3f}%, cap {BULL_MAX_MONITOR_DRAWDOWN*100:.3f}%). Skipping.")
                continue

            # 90-day candles for MACD / RSI / volume gates
            df = get_cached_data(symbol, 'history_90d',
                                 lambda s: yf_history(s, period="90d"), symbol)
            if df.empty or len(df) < 40:
                print(f"[bull] {yf_symbol}: insufficient history ({len(df)} bars). Skipping.")
                continue

            try:
                macd_line, macd_signal_line, _ = talib.MACD(
                    df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
                rsi_series = talib.RSI(df['Close'], timeperiod=14)
                latest_signal = float(macd_signal_line.iloc[-1])
                latest_rsi = float(rsi_series.iloc[-1])
                latest_volume = float(df['Volume'].iloc[-1])
                mean_volume = float(df['Volume'].mean())
            except Exception as e:
                logging.warning(f"[bull] {yf_symbol}: indicator calc failed: {e}")
                continue

            # v5 gate 2: MACD signal > 0.15
            if not (latest_signal > BULL_MACD_SIGNAL_MIN):
                print(f"[bull] {yf_symbol}: MACD signal gate failed "
                      f"(signal={latest_signal:.3f}, need >{BULL_MACD_SIGNAL_MIN}). Skipping.")
                continue

            # v5 gate 3: RSI > 70
            if not (latest_rsi > BULL_RSI_MIN):
                print(f"[bull] {yf_symbol}: RSI gate failed "
                      f"(rsi={latest_rsi:.2f}, need >{BULL_RSI_MIN}). Skipping.")
                continue

            # v5 gate 4: volume > 0.85 x mean
            if not (latest_volume > BULL_VOLUME_MULT_OF_MEAN * mean_volume):
                print(f"[bull] {yf_symbol}: volume gate failed "
                      f"(vol={latest_volume:.0f}, need >{BULL_VOLUME_MULT_OF_MEAN}x{mean_volume:.0f}"
                      f"={BULL_VOLUME_MULT_OF_MEAN*mean_volume:.0f}). Skipping.")
                continue

            reasons = [
                f"bull_ticks(inc={increased}>dec={decreased},>={BULL_MIN_PRICE_INCREASES})",
                f"bull_net={net_return*100:.2f}%",
                f"bull_maxdd={max_dd*100:.2f}%<={BULL_MAX_MONITOR_DRAWDOWN*100:.2f}%",
                f"bull_macd_signal={latest_signal:.3f}>{BULL_MACD_SIGNAL_MIN}",
                f"bull_rsi={latest_rsi:.1f}>{BULL_RSI_MIN}",
                f"bull_vol={latest_volume:.0f}>{BULL_VOLUME_MULT_OF_MEAN}x_mean",
            ]

            atr_for_rank = get_average_true_range(symbol)
            atr_pct = (atr_for_rank / current_price) if atr_for_rank and current_price else None

            # Synthetic sig block so Phase 2 / trade-feature persistence code
            # keeps working without a special case. Score is set to the
            # regime's dynamic threshold so bull candidates rank alongside
            # normal candidates fairly (rank_score = score / atr_pct).
            sig = {
                'score': get_buy_score_threshold(REGIME_BULL),
                'reasons': reasons,
                'detected': ['bull_strategy'],
                'latest_rsi': latest_rsi,
                'macd_above_signal': None,
                'volume_decrease': False,
            }
            reward_term = max(sig['score'], 0.0001)
            rank_score = reward_term / max(atr_pct, 0.001) if atr_pct else reward_term

            bull_candidates.append({
                'symbol': symbol, 'yf_symbol': yf_symbol, 'api_symbol': api_symbol,
                'current_price': current_price, 'sig': sig, 'atr': atr_for_rank,
                'atr_pct': atr_pct, 'rank_score': rank_score, 'regime': regime,
                'reward_source': 'bull_strategy', 'expected_return': None,
                # Bull-strategy-specific hints consumed by Phase 2:
                'bull_strategy': True,
                'bull_allocation': bull_alloc,
            })
            print(f"[bull] CANDIDATE {yf_symbol}: {', '.join(reasons)} "
                  f"| price ${current_price:.2f} | alloc ${bull_alloc:.2f}")
        except Exception as e:
            logging.warning(f"[bull] gate-eval error for {yf_symbol}: {e}")
            continue

    return bull_candidates


def buy_stocks(symbols_to_buy_list, lock):
    print("Starting buy_stocks function...")

    # Dashboard-driven pause / emergency-stop check (both block new entries).
    if is_dashboard_emergency_stopped():
        print("  [dashboard] EMERGENCY STOP active — no new entries this cycle.")
        return
    if is_dashboard_paused():
        print("  [dashboard] paused via dashboard — no new entries this cycle.")
        return

    if not symbols_to_buy_list:
        logging.info("No symbols to buy.")
        return

    # BUGFIX: carry qty per-symbol. Previously the DB write used a single
    # leaked `filled_qty` from the last loop iteration for EVERY position.
    filled_records = []  # (alpaca_symbol, yf_symbol, qty, price, date_str)

    st = get_margin_state()
    if st['trading_blocked'] or st['account_blocked']:
        print("Account is blocked by the broker. No buys.")
        return
    if not st['healthy']:
        print(f"Margin health low (equity/long_mv = {st['margin_ratio']:.2f} < "
              f"{MAINTENANCE_MARGIN_FLOOR_PCT:.2f}). No new buys.")
        logging.warning("Margin maintenance floor breached; buys suspended.")
        return

    total_equity = st['equity']
    current_exposure = st['long_market_value']

    # Trade governor: note today's equity baseline, then check whether we
    # are permitted to enter new positions this cycle (loss-streak cooldown
    # or daily profit lock may block).
    GOVERNOR.note_session_equity(total_equity)
    can_go, gov_reason = GOVERNOR.can_trade(current_equity=total_equity)
    if not can_go:
        print(f"  [governor] entries paused: {gov_reason}. Skipping buy cycle.")
        return

    # Brain B (risk brain) cycle-level check. Evaluates portfolio state
    # (margin, exposure, open-position P/L, VIX/regime, consec losses,
    # recent churn) as a single 12-feature vector. In shadow mode prints
    # P(safe) each cycle for observation; in armed mode blocks new entries
    # when P(safe) < BRAIN_B_MIN_SAFE.
    try:
        pos_snap_for_brain_b = [
            {'symbol': s, 'avg_price': avg,
             'current_price': get_current_price(s) or avg}
            for s, (avg, _) in position_book.snapshot().items()
        ]
        b_veto, b_reason, b_p_safe = brain_b_check_veto(
            margin_state=st, position_snapshots=pos_snap_for_brain_b)
        if b_p_safe is not None:
            print(f"  [brain_b RISK] {b_reason}")
            floor_post('BRAIN_B',
                       'block' if b_veto else 'observation',
                       f'P(safe)={b_p_safe:.2f} — ' +
                       ('BLOCKED' if b_veto else 'OK'))
        if b_veto:
            return   # armed veto: skip the cycle
    except Exception as _e:
        logging.debug(f"brain_b cycle check failed: {_e}")

    max_new_exposure = min(
        total_equity * MAX_PORTFOLIO_EXPOSURE_PCT - current_exposure,
        st['effective_bp'] - CASH_BUFFER,
    )
    if max_new_exposure <= MIN_ORDER_NOTIONAL:
        print("Exposure / buying-power limit reached. No new buys.")
        return
    print(f"Equity ${total_equity:,.2f} | Exposure ${current_exposure:,.2f} | "
          f"Effective BP ${st['effective_bp']:,.2f} | Headroom ${max_new_exposure:,.2f}")

    today_date_str = datetime.now(eastern).date().strftime("%Y-%m-%d")

    # ---------------- Regime + dynamic threshold (review items #6, #8) ----------------
    regime_info = get_market_regime()
    regime = regime_info['regime']
    dynamic_threshold = get_buy_score_threshold(regime)
    regime_weights = get_regime_weights(regime)
    vix_str = f"{regime_info['vix']:.1f}" if regime_info['vix'] is not None else "n/a"
    print(f"Market regime: {regime.upper()} (VIX {vix_str}) -> buy score threshold {dynamic_threshold}")

    # ---------------- Bull-market strategy candidates (regime-gated) ----------------
    # These come from the ported Bull Market Advanced Stock Market Trading
    # Robot v8 buy path. They only appear when regime == BULL, they are
    # merged into the same ranked shortlist as the normal ML-scored
    # candidates, and MAX_NEW_POSITIONS_PER_CYCLE applies to the combined
    # list. Bull candidates are marked 'bull_strategy': True so Phase 2
    # sizes with the per-symbol cash cap, always places the 1% trailing
    # stop, and persists the position with strategy_tag='bull'.
    bull_candidates_raw = _bull_market_scan_candidates(
        symbols_to_buy_list, regime, len(symbols_to_buy_list))
    # De-dupe: if a symbol already holds an open position, or is claimed by
    # another thread this cycle, or somehow appeared twice, drop it here.
    _bull_claimed_syms = set()
    bull_candidates = []
    for bc in bull_candidates_raw:
        if bc['api_symbol'] in _bull_claimed_syms:
            continue
        if not position_book.claim(bc['api_symbol']):
            print(f"[bull] {bc['yf_symbol']}: busy in another thread this cycle. Skipping bull candidate.")
            continue
        _bull_claimed_syms.add(bc['api_symbol'])
        bull_candidates.append(bc)

    # ---------------- Phase 1: scan and rank all candidates (review item #9) ----------------
    # Score every symbol first WITHOUT buying, then only submit orders for the
    # top-ranked candidates that fit within the available headroom. Ranking key
    # is expected-reward / expected-risk: buy score (reward proxy) divided by
    # ATR% (risk proxy), so a high-score low-volatility setup ranks above an
    # equally-scored but much choppier one.
    candidates = []  # list of dicts with everything buy execution needs

    for symbol in list(symbols_to_buy_list):
        yf_symbol = to_yf(symbol)
        api_symbol = to_alpaca(symbol)
        now_str = datetime.now(eastern).strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")

        # BUGFIX: per-symbol claim. Without it, buy_stocks could fill and add a
        # position for a symbol that sell_stocks was concurrently deciding to
        # exit, and both threads would race on the same broker position.
        # NOTE: this also naturally excludes symbols already claimed by the
        # bull-market scan above -- they'll be handled via the bull path.
        if not position_book.claim(api_symbol):
            print(f"{yf_symbol}: busy in another thread this cycle. Skipping.")
            continue
        release_now = True
        try:
            current_price = get_current_price(symbol)
            if current_price is None or current_price <= 0:
                continue

            # Track rolling price history
            ts = time.time()
            if symbol not in price_history:
                price_history[symbol] = {i: [] for i in interval_map}
                last_stored[symbol] = {i: 0 for i in interval_map}
            for interval, delta in interval_map.items():
                if ts - last_stored[symbol][interval] >= delta:
                    price_history[symbol][interval].append(current_price)
                    price_history[symbol][interval] = price_history[symbol][interval][-50:]
                    last_stored[symbol][interval] = ts

            # NOTE: the 90d candle fetch used to happen here, before the SMA/RSI
            # gates below. It now runs only for symbols that survive them.

            # --- Trend + multi-timeframe filters ---
            # PERF: these run BEFORE the 90d candle fetch. Both are cached for
            # 30m, so on a warm cache they cost zero yfinance requests and reject
            # most symbols for free. Fetching the 90d history first (as before)
            # meant paying a request for every symbol that was about to be cut.
            if not is_in_uptrend(symbol):
                print(f"{yf_symbol}: below 200-day SMA. Skipping.")
                update_previous_price(symbol, current_price)
                continue

            daily_rsi = get_daily_rsi(symbol)
            if daily_rsi is None or daily_rsi > 50:
                print(f"{yf_symbol}: daily RSI not oversold ({daily_rsi}). Skipping.")
                update_previous_price(symbol, current_price)
                continue

            # REVIEW ITEM #4: multi-timeframe confirmation. Daily trend/RSI just
            # passed above; also require the 60m and 5m timeframes to agree
            # bullish before spending a request on the 90d candle history.
            if not multi_timeframe_confirms_bullish(symbol):
                print(f"{yf_symbol}: multi-timeframe confirmation failed (60m/5m not bullish). Skipping.")
                update_previous_price(symbol, current_price)
                continue

            # REVIEW ITEM #3: earnings blackout. Avoid opening new positions
            # right before/after an earnings print, which can gap through any
            # stop or profit-monitor logic.
            if is_within_earnings_blackout(symbol):
                print(f"{yf_symbol}: within {EARNINGS_BLACKOUT_DAYS}-day earnings blackout. Skipping.")
                update_previous_price(symbol, current_price)
                continue

            # Only survivors pay for the 90d candle history.
            df = get_cached_data(symbol, 'history_90d',
                                 lambda s: yf_history(s, period="90d"), symbol)
            # BUGFIX: MACD(26,9) and RSI(14) need ~35 bars. Original required
            # only 3 rows, producing all-NaN indicators that silently scored 0.
            if df.empty or len(df) < 40:
                print(f"{yf_symbol}: insufficient history ({len(df)} bars). Skipping.")
                continue

            previous_price = get_previous_price(symbol) or current_price
            last_price = float(df['Close'].iloc[-1])

            # REVIEW ITEM #2: intraday pullback from the recent intraday HIGH,
            # using the rolling 1-minute price history already being tracked
            # above -- distinct from price_decline in compute_buy_score, which
            # only compares against yesterday's daily close. Falls back to
            # None (signal simply doesn't fire) if there isn't enough intraday
            # history yet for this symbol today.
            # Distinct from price_decline in compute_buy_score (which only
            # compares against yesterday's daily close), these are the four
            # multi-timeframe intraday pullback signals: peak of the 1-min
            # rolling history is the "immediate" pullback, 5-min and 30-min
            # capture progressively longer windows, and swing-high uses the
            # widest bucket (60-min) as the "session high" proxy. VWAP
            # distance now uses REAL session-anchored intraday VWAP computed
            # from yfinance 1-minute OHLCV (see get_intraday_vwap below) --
            # volume-weighted and anchored to the session open, matching how
            # brokers/charts draw the VWAP line.
            intraday_samples = price_history.get(symbol, {}).get('1min', [])
            intraday_pullback_pct = None
            if len(intraday_samples) >= 3:
                intraday_high = max(intraday_samples)
                if intraday_high > 0:
                    intraday_pullback_pct = (intraday_high - current_price) / intraday_high

            samples_5m = price_history.get(symbol, {}).get('5min', [])
            pullback_5m_pct = None
            if len(samples_5m) >= 3:
                high_5m = max(samples_5m)
                if high_5m > 0:
                    pullback_5m_pct = (high_5m - current_price) / high_5m

            samples_30m = price_history.get(symbol, {}).get('30min', [])
            pullback_30m_pct = None
            if len(samples_30m) >= 3:
                high_30m = max(samples_30m)
                if high_30m > 0:
                    pullback_30m_pct = (high_30m - current_price) / high_30m

            samples_60m = price_history.get(symbol, {}).get('60min', [])
            swing_high_distance_pct = None
            if len(samples_60m) >= 5:
                swing_high = max(samples_60m)
                if swing_high > 0:
                    swing_high_distance_pct = (swing_high - current_price) / swing_high

            # REAL intraday VWAP -- session-anchored, volume-weighted, computed
            # from today's 1-minute OHLCV bars (Σ(TP*V)/Σ(V) cumulated from the
            # session open). Replaces the earlier price-only arithmetic mean of
            # our 1-min price samples, which was neither volume-weighted nor
            # anchored to the session and could drift arbitrarily far from the
            # broker's / charting-package VWAP line. Cached for 60s per symbol
            # so a new 1-minute bar (which is the fastest this can change) is
            # picked up promptly without hammering yfinance every loop tick.
            vwap_distance_pct = None
            true_vwap = get_intraday_vwap(symbol)
            if true_vwap is not None and true_vwap > 0:
                vwap_distance_pct = (true_vwap - current_price) / true_vwap

            sig = compute_buy_score(df, current_price, previous_price, last_price,
                                    regime=regime, weights=regime_weights,
                                    intraday_pullback_pct=intraday_pullback_pct,
                                    pullback_5m_pct=pullback_5m_pct,
                                    pullback_30m_pct=pullback_30m_pct,
                                    swing_high_distance_pct=swing_high_distance_pct,
                                    vwap_distance_pct=vwap_distance_pct)

            # Price-stability bonus (BUGFIX: defined unconditionally, so the
            # `else` logging branch can no longer raise NameError)
            price_stable = True
            hist5 = price_history.get(symbol, {}).get('5min', [])
            if len(hist5) >= 2 and hist5[-2]:
                price_stable = abs(hist5[-1] - hist5[-2]) / hist5[-2] < 0.005
                if price_stable:
                    sig['score'] += regime_weights.get('price_stable', 1)

            # REVIEW ITEM #3: relative strength vs SPY is added as a supplementary
            # scanner signal (not a hard gate) -- outperformance nudges the score.
            rel_strength = get_relative_strength(symbol)
            if rel_strength is not None and rel_strength > 0:
                sig['score'] += 1
                sig['reasons'].append(f"rs_vs_spy=+{rel_strength*100:.2f}%")

            # ATR is needed both for the ML feature vector below and for
            # ranking further down -- fetched once here (get_average_true_range
            # is itself cached, so this isn't a double cost vs. the old layout).
            atr_for_rank = get_average_true_range(symbol)
            atr_pct = (atr_for_rank / current_price) if atr_for_rank and current_price else None

            # ML brain adjustment: a small +/- nudge from the inlined
            # TensorFlow model above, trained on real historical market data
            # (train_ml_brain_from_historical_data) and fine-tuned on this
            # bot's own closed-trade history (train_ml_brain_from_live_trades).
            # Returns None (no opinion, no adjustment applied) until enough
            # LIVE trade history exists to be minimally trustworthy for LIVE
            # decisions -- see ML_MIN_TRADES_FOR_LIVE_ADJUSTMENT. This is
            # additive, never a gate: it cannot by itself push a symbol above
            # or below the buy threshold in a way the rule-based score didn't
            # already get close to on its own (capped at +/-ML_MAX_SCORE_ADJUSTMENT).
            ml_adjustment = None
            if USE_ML_BRAIN_ADJUSTMENT and _ml_brain_is_available():
                try:
                    ml_adjustment = get_ml_adjustment(
                        session, TradeFeatures,
                        buy_score=sig['score'], rsi=sig.get('latest_rsi'),
                        atr_pct=atr_pct,
                        macd_above_signal=sig.get('macd_above_signal'),
                        volume_holding=not sig.get('volume_decrease'),
                        regime=regime,
                        # New sequence-model kwargs: the Conv1D->LSTM model
                        # needs a rolling window of daily bars, built from
                        # the same df compute_buy_score just used above.
                        symbol=symbol, df=df, current_price=current_price,
                    )
                except Exception as e:
                    logging.warning(f"{yf_symbol}: ml_brain adjustment failed: {e}")
                    ml_adjustment = None
                if ml_adjustment is not None:
                    sig['score'] += ml_adjustment
                    sig['reasons'].append(f"ml_brain={ml_adjustment:+.2f}")

            # Brain F (bullish-trend picker) -- small additive bump when a
            # dedicated foundation+head brain says this symbol's technicals
            # look like a favorable buy setup. Uses the SAME df compute_buy_score
            # just consumed above -- zero extra HTTP. Bounded at
            # +/-BRAIN_F_MAX_SCORE_BUMP, cached BRAIN_F_INFER_CACHE_TTL seconds.
            # Never a gate; if it's off / model missing / feature build fails,
            # bump is 0.0 and buy_score is unchanged.
            try:
                brain_f_bump, brain_f_p = brain_f_get_score_bump(
                    symbol, df, current_price=current_price)
            except Exception as e:
                logging.warning(f"{yf_symbol}: brain_f bump failed: {e}")
                brain_f_bump, brain_f_p = 0.0, None
            if brain_f_p is not None:
                if BRAIN_F_MODE == 'shadow':
                    sig['reasons'].append(
                        f"brain_f(SHADOW) p_bullish={brain_f_p:.2f} bump_would_be={brain_f_bump:+.2f}")
                elif brain_f_bump != 0.0:
                    sig['score'] += brain_f_bump
                    sig['reasons'].append(f"brain_f={brain_f_bump:+.2f} (p={brain_f_p:.2f})")

            # Analog analyzer: look up historical days where this symbol had
            # a similar feature pattern, simulate what the bot's actual exit
            # rules would have done from each analog day, and adjust the buy
            # score based on the aggregate analog outcome. Bounded like ML
            # brain -- capped at +/-USE_ANALOG_MAX_ADJUSTMENT, only fires
            # when analog evidence is genuinely favorable (win rate > 55%
            # AND positive expectancy), and never a hard gate. Uses the
            # disk cache so this is cheap after the first pass per symbol.
            if USE_ANALOG_ADJUSTMENT:
                try:
                    analog = analyze_trade_analog(symbol)
                except Exception as e:
                    logging.warning(f"{yf_symbol}: analog analyzer failed: {e}")
                    analog = None
                if analog is not None and analog['n_analogs'] >= ANALOG_MIN_ANALOGS:
                    # Map (win_rate, expectancy) -> bounded score adjustment.
                    # Neutral zone: win_rate 50%, expectancy 0 -> 0 adjustment.
                    # Strong analog evidence (win_rate >= 65%, expectancy >= 1%)
                    # -> full USE_ANALOG_MAX_ADJUSTMENT boost. Bad analog
                    # evidence (win_rate <= 40%, expectancy <= -1%) subtracts.
                    wr_component = (analog['win_rate'] - 0.5) * 4.0   # +/-2 range at 0%/100%
                    exp_component = analog['expectancy'] * 100.0      # 1% expectancy -> +1
                    raw = (wr_component + exp_component) / 2.0
                    analog_adj = max(-USE_ANALOG_MAX_ADJUSTMENT,
                                     min(USE_ANALOG_MAX_ADJUSTMENT, raw))
                    # Only actually apply if the signal has meaningful magnitude
                    # (>= 0.25) -- avoids noise-level adjustments cluttering the log.
                    if abs(analog_adj) >= 0.25:
                        sig['score'] += analog_adj
                        sig['reasons'].append(
                            f"analog={analog_adj:+.2f} "
                            f"(n={analog['n_analogs']}, wr={analog['win_rate']*100:.0f}%, "
                            f"exp={analog['expectancy']*100:+.2f}%)")

            if not sig['detected']:
                print(f"{yf_symbol}: no bullish reversal pattern. Score {sig['score']}. Skipping.")
                update_previous_price(symbol, current_price)
                continue

            if sig['score'] < dynamic_threshold:
                print(f"{yf_symbol}: score {sig['score']} < {dynamic_threshold} ({regime} regime). "
                      f"[{'; '.join(sig['reasons'])}] Skipping.")
                logging.info(f"{now_str} Skipped {yf_symbol}: score {sig['score']} (threshold {dynamic_threshold})")
                update_previous_price(symbol, current_price)
                continue

            # ---------------- Early-morning entry gate (belt-and-suspenders) ----------------
            # Also enforced independently by the RISK_TIME vote inside
            # chair_decide. We check here too so we skip the (expensive)
            # ranking / analog / chair phases entirely on symbols that can't
            # possibly buy right now. In shadow mode the helper always
            # returns allow=True but the reason string records the
            # would-deny; we log it and continue.
            em_allow, em_reason = check_early_morning_entry_gate(current_price, df)
            if not em_allow:
                print(f"{yf_symbol}: {DIM}early-morning gate:{RESET} {em_reason}. Skipping.")
                logging.info(f"{now_str} Skipped {yf_symbol}: early-morning gate — {em_reason}")
                update_previous_price(symbol, current_price)
                continue
            elif em_reason.startswith('[SHADOW would-deny]'):
                print(f"  [early-gate SHADOW] {yf_symbol}: {em_reason} (would-block, letting through)")

            # REVIEW ITEM #4: rank by an empirical expected-return estimate
            # when enough of this bot's own trade history exists at this
            # score level; otherwise fall back to the raw score, exactly as
            # before. Either way this is divided by ATR% as the risk proxy.
            expected_return = get_expected_return_by_score(sig['score'])
            reward_term = expected_return if expected_return is not None else sig['score']
            reward_source = 'history' if expected_return is not None else 'score'
            # A negative empirical expected return at this score level should
            # never rank ABOVE a positive one just because ATR% divides it
            # toward zero -- floor it at a small positive epsilon so a
            # historically-losing score bucket sinks to the bottom of the
            # ranking instead of being inflated by a low-volatility stock.
            reward_term = max(reward_term, 0.0001) if reward_source == 'history' else reward_term
            rank_score = reward_term / max(atr_pct, 0.001) if atr_pct else reward_term

            candidates.append({
                'symbol': symbol, 'yf_symbol': yf_symbol, 'api_symbol': api_symbol,
                'current_price': current_price, 'sig': sig, 'atr': atr_for_rank,
                'atr_pct': atr_pct, 'rank_score': rank_score, 'regime': regime,
                'reward_source': reward_source, 'expected_return': expected_return,
            })
            update_previous_price(symbol, current_price)
            # Keep this symbol's claim held through phase 2 -- released below
            # once phase 2 decides whether to buy it.
            release_now = False
        finally:
            if release_now:
                position_book.release(api_symbol)

    # Merge in bull-market strategy candidates (already claimed above). The
    # combined list is ranked and MAX_NEW_POSITIONS_PER_CYCLE is applied to
    # the whole thing, so bull candidates compete on merit with ML-scored ones.
    if bull_candidates:
        print(f"[bull] merging {len(bull_candidates)} bull-strategy candidate(s) "
              f"into ranked pool of {len(candidates)} ML-scored candidate(s).")
        candidates.extend(bull_candidates)

    if not candidates:
        print("No candidates passed the scan/rank filters this cycle.")
        return

    # Rank best-first. Only the top N are actually bought (review item #9): the
    # rest are released so they don't sit locked out of a future cycle.
    candidates.sort(key=lambda c: c['rank_score'], reverse=True)
    to_buy = candidates[:MAX_NEW_POSITIONS_PER_CYCLE]
    skipped = candidates[MAX_NEW_POSITIONS_PER_CYCLE:]
    for c in skipped:
        print(f"{c['yf_symbol']}: ranked #{candidates.index(c)+1} of {len(candidates)} "
              f"(score {c['sig']['score']}, rank {c['rank_score']:.1f}) — outside top "
              f"{MAX_NEW_POSITIONS_PER_CYCLE}. Not buying this cycle.")
        position_book.release(c['api_symbol'])

    if to_buy:
        print(f"Ranked buy candidates ({len(to_buy)} of {len(candidates)}):")
        for i, c in enumerate(to_buy, 1):
            atr_pct_str = f"{c['atr_pct']*100:.2f}%" if c['atr_pct'] else "n/a"
            if c['reward_source'] == 'history':
                reward_str = f"empirical avg {c['expected_return']*100:+.2f}%"
            else:
                reward_str = f"raw score {c['sig']['score']} (insufficient trade history yet)"
            print(f"  #{i} {c['yf_symbol']}: {reward_str}, ATR% {atr_pct_str}, "
                  f"rank {c['rank_score']:.1f}")

    # ---------------- Phase 2: execute buys for the ranked shortlist ----------------
    for c in to_buy:
        symbol, yf_symbol, api_symbol = c['symbol'], c['yf_symbol'], c['api_symbol']
        current_price, sig = c['current_price'], c['sig']
        now_str = datetime.now(eastern).strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")
        try:
            # ---------------- Position sizing ----------------
            is_bull_buy = bool(c.get('bull_strategy'))
            if is_bull_buy:
                # Bull-strategy path: equal-cash allocation capped at the
                # per-symbol max, computed in the scan. Bypasses the risk-based
                # sizer entirely -- the bull strategy sizes by cash, not ATR.
                notional = float(c.get('bull_allocation') or 0.0)
                # Still clamp to available headroom / buying-power buffer so a
                # stale allocation figure can't run us into negative cash.
                with lock:
                    cash_available = float(api.get_account().cash)
                headroom = min(
                    BULL_MAX_ALLOCATION_PER_SYMBOL,
                    max_new_exposure,
                    (cash_available - CASH_BUFFER) if ACCOUNT_MODE == 'cash' else max_new_exposure,
                )
                notional = min(notional, headroom)
                if notional < MIN_ORDER_NOTIONAL:
                    print(f"[bull] {yf_symbol}: allocation ${notional:.2f} below ${MIN_ORDER_NOTIONAL:.2f} floor after headroom clamp. Skipping.")
                    continue
            elif ALL_BUY_ORDERS_ARE_1_DOLLAR:
                notional = MIN_ORDER_NOTIONAL
            else:
                atr = get_average_true_range(symbol)
                if atr is None:
                    print(f"{yf_symbol}: no valid ATR. Skipping.")
                    continue
                # RISK ALIGNMENT FIX (review item #5): risk_per_share now uses the
                # SAME multiplier the hard stop-loss actually enforces
                # (HARD_STOP_ATR_MULTIPLIER), so the 1%-of-equity risk this sizing
                # targets is the position's REAL maximum loss, not a distance no
                # stop was ever placed at.
                risk_per_share = HARD_STOP_ATR_MULTIPLIER * atr
                risk_amount = RISK_PER_TRADE_PCT * total_equity
                notional = (risk_amount / risk_per_share) * current_price

                with lock:
                    cash_available = float(api.get_account().cash)
                headroom = min(
                    MAX_ALLOCATION_PER_SYMBOL,
                    max_new_exposure,
                    (cash_available - CASH_BUFFER) if ACCOUNT_MODE == 'cash' else max_new_exposure,
                )

                # BUGFIX: on small accounts, risk-based sizing always lands below the
                # broker's $1 notional floor, so every trade was silently discarded.
                # Round up to the floor when headroom allows; only skip if it doesn't.
                # NOTE: rounding up intentionally exceeds RISK_PER_TRADE_PCT. See
                # MIN_EQUITY_TO_TRADE if you'd rather halt than over-risk.
                if notional < MIN_ORDER_NOTIONAL:
                    if headroom >= MIN_ORDER_NOTIONAL:
                        actual_risk_pct = (MIN_ORDER_NOTIONAL / current_price * risk_per_share) / total_equity * 100
                        print(f"{yf_symbol}: risk-sized ${notional:.2f} < ${MIN_ORDER_NOTIONAL:.2f} floor; "
                              f"rounding up (risk becomes {actual_risk_pct:.2f}% of equity)")
                        notional = MIN_ORDER_NOTIONAL
                    else:
                        print(f"{yf_symbol}: headroom ${headroom:.2f} < ${MIN_ORDER_NOTIONAL:.2f} minimum. Skipping.")
                        continue
                else:
                    # Slippage haircut
                    notional = min(notional, headroom) * 0.999

            notional = round(notional, 2)
            if notional < MIN_ORDER_NOTIONAL:
                print(f"{yf_symbol}: notional ${notional:.2f} below ${MIN_ORDER_NOTIONAL:.2f} minimum. Skipping.")
                continue

            with lock:
                bp = float(api.get_account().buying_power)
            if bp < notional + CASH_BUFFER:
                print(f"{yf_symbol}: insufficient buying power (${bp:.2f} < ${notional + CASH_BUFFER:.2f}).")
                continue

            if not day_trades_allowed():
                print("Broker has blocked trading on this account.")
                break

            qty_est = round(notional / current_price, 4)
            reason = f"score={sig['score']} [{'; '.join(sig['reasons'])}]"

            # ---------------- Blacklist gate (hard block) ----------------
            # Two-tier operator-controlled blacklist. Distinct from the soft
            # PerSymbolPerformance mute above (which is statistical), and
            # from the auto 72h that fires on any losing close.
            try:
                is_bl, bl_reason = BLACKLIST.is_blocked(yf_symbol)
                if is_bl:
                    print(f"  [blacklist] {yf_symbol}: {bl_reason}. Skipping.")
                    continue
            except Exception as _e:
                logging.debug(f"BlacklistManager check failed for {yf_symbol}: {_e}")

            # ---------------- Per-symbol performance mute (Kalshi port) ----------------
            # Fast, in-memory check: if THIS bot's own closed trades on this
            # symbol are net-negative with a poor win rate, mute for 24h.
            try:
                if PERSYMBOL.is_muted(yf_symbol):
                    wr = PERSYMBOL.win_rate(yf_symbol) * 100
                    pnl = PERSYMBOL.net_pnl(yf_symbol)
                    print(f"  [persymbol] {yf_symbol}: muted (win rate {wr:.0f}%, "
                          f"net P&L ${pnl:+.2f}). Skipping.")
                    continue
            except Exception as _e:
                logging.debug(f"PerSymbolPerformance check failed for {yf_symbol}: {_e}")

            # ---------------- Backtest brain (Brain C) — vote gathering ----
            # Still computed here (it needs `regime`) but the veto is now
            # centralized in the Chair below. The Chair reads bt_verdict/
            # bt_result and posts BACKTEST_C's vote to the floor.
            bt_verdict = None
            bt_result = None
            try:
                bt_verdict, bt_result = get_backtest_verdict(
                    yf_symbol, current_regime=regime)
                print_backtest_verdict(yf_symbol, bt_verdict, bt_result)
            except Exception as _e:
                logging.debug(f"backtest brain error for {yf_symbol}: {_e} "
                              "(treating as ABSTAIN)")

            # ---------------- Chair (Brain E) — final trade decision ----
            # Gathers votes from Trading_A / Backtest_C / Risk_B / Portfolio_D,
            # posts each vote to the Brain Trading Floor, applies deterministic
            # aggregation rules, and returns approve/deny + adjusted notional.
            try:
                chair_result = chair_decide(
                    yf_symbol, notional_requested=notional,
                    current_regime=regime, ml_adjustment=ml_adjustment,
                    bt_verdict=bt_verdict, bt_result=bt_result,
                    current_price=current_price, df=df)
                if not chair_result['approved']:
                    print(f"  {yf_symbol}: CHAIR denied — {chair_result['reason']}")
                    continue
                # Chair may have resized the trade via Portfolio_D's multiplier
                if chair_result['notional'] != notional:
                    notional = chair_result['notional']
                    qty_est = round(notional / current_price, 4)
                    reason = f"score={sig['score']} [{'; '.join(sig['reasons'])}] " \
                             f"(Chair sized to ${notional:.2f})"
            except Exception as _e:
                logging.warning(f"chair_decide failed for {yf_symbol}: {_e} "
                                "— falling back to legacy per-brain vetoes")
                # Fallback: if the Chair itself errored, honor Backtest C's
                # armed-mode veto so we don't submit a trade the backtest
                # would have blocked.
                if (bt_verdict == 'VETO'
                        and BACKTEST_BRAIN_MODE == 'armed'):
                    print(f"  {yf_symbol}: backtest brain VETOED (Chair fallback)")
                    continue

            print(f"Submitting buy: {api_symbol} ~{qty_est:.4f} sh @ ${current_price:.2f} "
                  f"(notional ${notional:.2f}) | {reason}")

            # Record brain predictions BEFORE submitting the order so trust
            # tracking still fires on partial-fill / rejection paths where
            # the outcome eventually arrives via sell_stocks' close hook.
            try:
                if ml_adjustment is not None:
                    inferred_pwin = 0.5 + (ml_adjustment / (2 * ML_MAX_SCORE_ADJUSTMENT))
                    inferred_pwin = max(0.0, min(1.0, inferred_pwin))
                    brain_trust_record_prediction('trading_brain_A', yf_symbol, inferred_pwin)
                if bt_result and bt_result.get('decision_win_rate') is not None:
                    brain_trust_record_prediction('backtest_brain_C', yf_symbol,
                                                   float(bt_result['decision_win_rate']))
            except Exception as _e:
                logging.debug(f"brain trust record failed for {yf_symbol}: {_e}")

            try:
                buy_order = api.submit_order(
                    symbol=api_symbol,
                    notional=notional,
                    side='buy',
                    type='market',
                    time_in_force='day',
                )
                logging.info(f"{now_str} Submitted buy {api_symbol} notional ${notional:.2f}: {reason}")

                filled_qty = 0.0
                filled_price = current_price
                terminal = False
                for _ in range(30):
                    try:
                        o = api.get_order(buy_order.id)
                    except Exception as e:
                        # BUGFIX: a transient network error during polling used to
                        # escape the APIError handler and kill the whole buy loop,
                        # skipping the DB persist for every prior fill in this pass.
                        logging.warning(f"{api_symbol}: poll error ({e}); retrying.")
                        time.sleep(2)
                        continue

                    # BUGFIX: track partial fills. The old code only broke on exactly
                    # 'filled', so a partially_filled order polled out and was logged
                    # as "not filled" -- while the shares were actually owned, with no
                    # DB row, no stop, and no trade history. Silent orphan position.
                    filled_qty = float(o.filled_qty or 0)
                    if o.filled_avg_price:
                        filled_price = float(o.filled_avg_price)

                    if o.status == 'filled':
                        terminal = True
                        break
                    if o.status in ('canceled', 'expired', 'rejected'):
                        print(f"{api_symbol}: order {o.status} (filled {filled_qty:.4f} before stopping).")
                        logging.warning(f"{api_symbol}: order {o.status}, partial qty {filled_qty:.4f}.")
                        terminal = True
                        break
                    time.sleep(2)

                # BUGFIX: cancel a still-open order that never reached a terminal
                # state, so it can't fill later behind our back and leave the broker
                # holding shares this bot has no record of.
                if not terminal:
                    try:
                        api.cancel_order(buy_order.id)
                        logging.warning(f"{api_symbol}: buy order timed out after 60s; cancel requested.")
                        print(f"{api_symbol}: order timed out, cancel requested.")
                        time.sleep(2)
                        o = api.get_order(buy_order.id)
                        filled_qty = float(o.filled_qty or 0)
                        if o.filled_avg_price:
                            filled_price = float(o.filled_avg_price)
                    except Exception as e:
                        logging.error(f"{api_symbol}: cancel/re-check failed: {e}")

                # Any qty actually acquired is recorded, whether the order completed
                # fully, partially, or was cancelled mid-flight.
                if filled_qty > 0:
                    print(f"Filled {filled_qty:.4f} sh of {api_symbol} @ "
                          f"{GREEN}${filled_price:.2f}{RESET} (cost ${filled_qty * filled_price:.2f})")
                    with open(csv_filename, mode='a', newline='') as f:
                        csv.DictWriter(f, fieldnames=fieldnames).writerow({
                            'Date': now_str, 'Buy': 'Buy', 'Sell': '',
                            'Quantity': filled_qty, 'Symbol': api_symbol,
                            'Price Per Share': filled_price,
                        })
                    filled_records.append((api_symbol, yf_symbol, filled_qty, filled_price, today_date_str, sig, is_bull_buy))

                    # Bull-strategy fills always get their own 1% trailing stop
                    # (BULL_TRAIL_PERCENT), regardless of the global toggle --
                    # the ported strategy's exit design depends on it. Normal
                    # fills still respect USE_TRAILING_STOP.
                    if is_bull_buy:
                        sid = place_trailing_stop_sell_order(
                            api_symbol, filled_qty, filled_price,
                            trail_percent=BULL_TRAIL_PERCENT)
                        print(f"[bull] Trailing stop ({BULL_TRAIL_PERCENT}%) for {api_symbol}: {sid or 'not placed (see log)'}")
                    elif USE_TRAILING_STOP and not ALL_BUY_ORDERS_ARE_1_DOLLAR:
                        sid = place_trailing_stop_sell_order(api_symbol, filled_qty, filled_price)
                        print(f"Trailing stop for {api_symbol}: {sid or 'not placed (see log)'}")
                else:
                    print(f"Buy order not filled for {api_symbol}")
                    logging.info(f"{now_str} Buy order not filled for {api_symbol}")

            except tradeapi.rest.APIError as e:
                print(f"Error submitting buy order for {api_symbol}: {e}")
                logging.error(f"Error submitting buy order for {api_symbol}: {e}")
                continue
            except Exception as e:
                # BUGFIX: catch-all so one unexpected failure can't abort the loop
                # and discard already-filled records awaiting persist.
                print(f"Unexpected error handling buy for {api_symbol}: {e}")
                logging.error(f"Unexpected error handling buy for {api_symbol}: {e}")
                continue

            update_previous_price(symbol, current_price)
            time.sleep(0.8)
        finally:
            # BUGFIX: always release, including on every `continue` path and on
            # exception, or the symbol stays locked out of trading forever.
            position_book.release(api_symbol)

    # ---------------- Persist fills ----------------
    if not filled_records:
        return
    try:
        with lock:
            for api_symbol, yf_symbol, qty, price, dstr, sig, is_bull_buy in filled_records:
                # BUGFIX: mutate the shared PositionBook in place instead of a
                # by-reference dict that refresh_* would later rebind away.
                position_book.upsert(api_symbol, round(price, 4), dstr)
                # NOTE: purchases NO LONGER remove the symbol from the
                # candidate list. The list is an in-memory universe of
                # ranked candidates refreshed once per day by the inlined
                # scanner, not a to-do queue. A filled buy just means the
                # bot now holds a position -- the same symbol may remain
                # eligible for future evaluations. (remove_symbols_from_trade_list
                # is kept as a no-op for backward compatibility.)

                session.add(TradeHistory(symbols=api_symbol, action='buy',
                                         quantity=qty, price=price, date=dstr))

                # REVIEW ITEM #7: snapshot the features present at entry so
                # they can later be joined against the eventual outcome.
                atr_val = get_average_true_range(yf_symbol)
                session.add(TradeFeatures(
                    symbols=api_symbol, entry_date=dstr, entry_price=price,
                    rsi=sig.get('latest_rsi'),
                    macd_above_signal=int(bool(sig.get('macd_above_signal'))),
                    atr_pct=(atr_val / price) if atr_val and price else None,
                    volume_holding=int(bool(sig.get('volume_decrease')) is False),
                    candlestick_pattern=','.join(sig.get('detected', [])) or None,
                    buy_score=sig.get('score'),
                    regime=regime,
                    time_of_day=datetime.now(eastern).strftime('%H:%M'),
                    entry_atr=atr_val,   # item #1: freeze entry-time ATR for stop calc
                ))
                # BUGFIX: merge instead of add — a re-buy of an existing symbol
                # previously raised an IntegrityError on the primary key.
                existing = session.query(Position).filter_by(symbols=api_symbol).one_or_none()
                new_tag = 'bull' if is_bull_buy else None
                if existing:
                    total_qty = existing.quantity + qty
                    existing.avg_price = ((existing.avg_price * existing.quantity) + (price * qty)) / total_qty
                    existing.quantity = total_qty
                    existing.purchase_date = dstr
                    # Weighted-average entry ATR across the combined position,
                    # weighted by dollar cost basis (not share count) so a
                    # re-buy at a very different price doesn't skew the frozen
                    # ATR arbitrarily. On a first-time missing entry_atr just
                    # accept the new one.
                    if existing.entry_atr and atr_val:
                        prev_basis = existing.avg_price * (existing.quantity - qty) if existing.quantity > qty else 0.0
                        new_basis = price * qty
                        total_basis = max(prev_basis + new_basis, 1e-9)
                        existing.entry_atr = ((existing.entry_atr * prev_basis) + (atr_val * new_basis)) / total_basis
                    elif atr_val:
                        existing.entry_atr = atr_val
                    # Strategy tag: a bull-strategy add-on to a bull position
                    # stays bull. Mixing strategies on one broker-side position
                    # is unavoidable (Alpaca has one avg_entry_price per
                    # symbol) -- resolve toward whichever exit rule is safer
                    # (bull's +0.5% target exits sooner than the ML path's
                    # profit monitor, so bull "wins" on any mixed re-buy).
                    if is_bull_buy or existing.strategy_tag == 'bull':
                        existing.strategy_tag = 'bull'
                else:
                    session.add(Position(symbols=api_symbol, quantity=qty,
                                         avg_price=price, purchase_date=dstr,
                                         entry_atr=atr_val,
                                         strategy_tag=new_tag))
            session.commit()
        print("Database updated successfully.")
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Database error: {e}")
        logging.error(f"Database error: {e}")
        return

    # BUGFIX: refresh_after_buy() used to run INSIDE `with lock`. It sleeps 2s
    # and then makes blocking API calls (list_positions plus a paginated order
    # lookup per symbol), holding the mutex for tens of seconds and serializing
    # both worker threads. Now called after the lock is released.
    refresh_after_buy()

    # First-cycle brain-thinking trail is complete; subsequent cycles stay
    # quiet. Safe to call every cycle -- only the first call flips the flag.
    try:
        _ml_mark_first_cycle_thinking_done()
    except Exception:
        pass


def refresh_after_buy():
    # BUGFIX: no longer rebinds globals. symbols_to_buy is refreshed by main()
    # each cycle, and the position view is mutated in place so the other thread
    # keeps seeing the same object.
    time.sleep(2)
    position_book.replace_all(update_symbols_to_sell_from_api())


@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def place_trailing_stop_sell_order(symbol, qty, current_price, retries=3, trail_percent=None):
    """
    Places a trailing stop on the whole-share portion. Alpaca does not accept
    fractional qty for trailing_stop orders, so any fractional remainder is left
    to the sell_stocks take-profit logic.

    BUGFIX: previously a failed stop just printed and moved on, leaving an
    unprotected position. Now retries with backoff and escalates on give-up.
    """
    whole = int(qty)
    if whole < 1:
        logging.info(f"{symbol}: qty {qty:.4f} < 1 whole share; trailing stop not supported "
                     f"by broker for fractional qty. Managed by sell_stocks instead.")
        return None

    tp = TRAIL_PERCENT if trail_percent is None else trail_percent
    for attempt in range(retries):
        try:
            stop_order = api.submit_order(
                symbol=symbol,
                qty=whole,
                side='sell',
                type='trailing_stop',
                trail_percent=str(tp),
                time_in_force='gtc',
            )
            logging.info(f"Placed trailing stop ({tp}%) for {whole} sh of {symbol}: {stop_order.id}")
            return stop_order.id
        except Exception as e:
            logging.error(f"Trailing stop attempt {attempt + 1}/{retries} failed for {symbol}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    # Give-up path: the position is live and unprotected. Make that loud.
    msg = (f"CRITICAL: could not place trailing stop for {whole} sh of {symbol} after "
           f"{retries} attempts. POSITION IS UNPROTECTED - exit relies on take-profit only.")
    print(f"{RED}{msg}{RESET}")
    logging.critical(msg)
    return None


@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def cancel_open_sell_orders(symbol):
    """
    Cancel resting sell orders (e.g. the GTC trailing stop) so a take-profit can
    sell the full position.

    BUGFIX: no cancel logic existed at all. A GTC trailing stop reserves shares
    at the broker, so sell_stocks could only ever offload the unreserved
    fraction -- the whole-share portion could never exit on the profit target.

    Returns True if it is safe to proceed with a full-size sell.
    """
    try:
        open_sells = [o for o in api.list_orders(status='open') if o.symbol == symbol and o.side == 'sell']
    except Exception as e:
        logging.error(f"{symbol}: could not list open orders: {e}")
        return False

    if not open_sells:
        return True

    for o in open_sells:
        try:
            api.cancel_order(o.id)
            logging.info(f"{symbol}: cancelled resting sell order {o.id} ({o.type}) to free shares.")
        except Exception as e:
            logging.error(f"{symbol}: failed to cancel sell order {o.id}: {e}")
            return False

    # Cancellation is asynchronous; wait for the broker to release the shares.
    for _ in range(10):
        time.sleep(1)
        try:
            still_open = [o for o in api.list_orders(status='open')
                          if o.symbol == symbol and o.side == 'sell']
            if not still_open:
                return True
        except Exception as e:
            logging.error(f"{symbol}: error confirming cancellation: {e}")
            return False

    logging.warning(f"{symbol}: sell orders still open after cancel; skipping this cycle.")
    return False


@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def get_most_recent_purchase_date(symbol):
    try:
        order_list = []
        CHUNK_SIZE = 500
        until = datetime.now(pytz.UTC).isoformat()
        # BUGFIX: unbounded while-loop could paginate forever on a busy account.
        for _ in range(10):
            chunk = api.list_orders(status='all', nested=False, direction='desc',
                                    until=until, limit=CHUNK_SIZE, symbols=[symbol])
            if not chunk:
                break
            order_list.extend(chunk)
            until = (chunk[-1].submitted_at - timedelta(seconds=1)).isoformat()
            if len(chunk) < CHUNK_SIZE:
                break

        buys = [o for o in order_list if o.side == 'buy' and o.status == 'filled' and o.filled_at]
        if buys:
            d = max(buys, key=lambda o: o.filled_at).filled_at.date()
            return d.strftime("%Y-%m-%d")
    except Exception as e:
        logging.error(f"Error fetching buy orders for {symbol}: {e}")
    return datetime.now(eastern).date().strftime("%Y-%m-%d")


@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def update_symbols_to_sell_from_api():
    positions = api.list_positions()
    d = {}
    live = set()
    for p in positions:
        sym = p.symbol
        live.add(sym)
        avg = float(p.avg_entry_price)
        qty = float(p.qty)
        pdate = get_most_recent_purchase_date(sym)
        row = session.query(Position).filter_by(symbols=sym).one_or_none()
        if row:
            row.quantity, row.avg_price, row.purchase_date = qty, avg, pdate
        else:
            session.add(Position(symbols=sym, quantity=qty, avg_price=avg, purchase_date=pdate))
        d[sym] = (avg, pdate)

    # BUGFIX: prune DB rows for positions that no longer exist at the broker,
    # otherwise sell_stocks kept trying to sell phantom holdings forever.
    for row in session.query(Position).all():
        if row.symbols not in live:
            session.delete(row)

    session.commit()
    return d


class PositionBook:
    """
    Thread-safe owner of the shared position view.

    BUGFIX (rebind race): main() passed `symbols_to_sell_dict` BY REFERENCE to
    both worker threads, and refresh_after_buy()/refresh_after_sell() then did
    `global symbols_to_sell_dict; symbols_to_sell_dict = {...}` -- REBINDING the
    global to a brand-new dict. The threads kept references to the OLD object,
    so every update after the first refresh was silently discarded and the two
    threads mutated different dicts. This class is never rebound; it mutates one
    dict in place under its own lock, so all readers see the same state.

    BUGFIX (per-symbol races): buy_stocks and sell_stocks could both act on the
    same symbol concurrently (a fill landing while sell was deciding to exit).
    claim()/release() give a per-symbol mutex so only one side touches a symbol
    at a time.
    """

    def __init__(self):
        self._data = {}                       # symbol -> (avg_price, purchase_date)
        self._lock = threading.RLock()
        self._claims = set()                  # symbols currently owned by a thread
        self._claims_cv = threading.Condition(self._lock)

    # ---- snapshot / read ----
    def snapshot(self):
        """Stable copy for iteration. Never iterate the live dict."""
        with self._lock:
            return dict(self._data)

    def get(self, symbol):
        with self._lock:
            return self._data.get(symbol)

    def symbols(self):
        with self._lock:
            return set(self._data)

    def __len__(self):
        with self._lock:
            return len(self._data)

    # ---- mutate in place (never rebind) ----
    def upsert(self, symbol, avg_price, purchase_date):
        with self._lock:
            self._data[symbol] = (avg_price, purchase_date)

    def remove(self, symbol):
        with self._lock:
            self._data.pop(symbol, None)

    def replace_all(self, mapping):
        """Refresh contents WITHOUT rebinding the object other threads hold."""
        with self._lock:
            self._data.clear()
            self._data.update(mapping)

    # ---- per-symbol claim ----
    def claim(self, symbol, timeout=0):
        """Try to take exclusive ownership of a symbol. False if already claimed."""
        with self._claims_cv:
            if symbol in self._claims:
                return False
            self._claims.add(symbol)
            return True

    def release(self, symbol):
        with self._claims_cv:
            self._claims.discard(symbol)
            self._claims_cv.notify_all()


# Single shared instance. Referenced directly by both threads; never reassigned.
position_book = PositionBook()


# ============================================================================
# CHANDELIER ATR TRAILING STOP (adapted from Kalshi bot's ChandelierEmaEngine)
# ============================================================================
# Dynamic ATR-based trailing stop that widens/tightens with volatility. Fires
# BEFORE the deep hard stop and INDEPENDENT of the profit-monitor's arm
# threshold, so it can cut a position that's rolled over even before the
# profit monitor has armed.
#
# Definition (long side, adapted for daily bars):
#   chandelier_stop = highest_high_since_entry - (CE_ATR_MULT * entry_ATR)
# The stop ratchets UP as the position runs (never down), so peak-since-entry
# is tracked and the stop is always max(prior_stop, new_computation).
#
# The bot's existing ProfitMonitorEngine already tracks peak_price per
# symbol -- we piggyback on that state instead of adding a new DB column.
# Entry ATR is stored on the Position row (entry_atr column) and is the
# canonical volatility snapshot for that position.
#
# MODES:
#   'off'     : disabled; no chandelier check runs
#   'shadow'  : computes and PRINTS the stop level every cycle but never
#               triggers an exit -- for observing behavior before arming
#   'armed'   : fires exits when current_price <= chandelier_stop

CHANDELIER_MODE = 'armed'                  # 'armed' | 'shadow' | 'off'
CHANDELIER_ATR_MULT = 3.0                  # Pine default; classic LeBeau constant
CHANDELIER_MIN_HOLD_SECS = 300             # don't trigger in the first 5 min after entry
CHANDELIER_REQUIRE_POSITIVE_RUN_PCT = 0.0  # require peak to have exceeded entry by at least this
                                           # (0.0 = trigger anytime peak > entry; e.g. 0.003 = require +0.3% peak first)

# Bad-news stop-loss tightener: when Brain F's negative-only rolling news
# sentiment for an owned position crosses BAD_NEWS_STOP_THRESHOLD, the
# chandelier stop is tightened to just BAD_NEWS_TIGHTEN_STOP_PCT below the
# current price (much tighter than 3x ATR). The tightener can only make the
# stop TIGHTER, never looser -- if the normal chandelier stop is already
# above the tightened one, the normal stop wins. Bypasses the positive-run
# gate too: on genuinely bad news we don't wait for a peak to form before
# defending capital.
BAD_NEWS_STOP_MODE = 'armed'               # 'armed' | 'shadow' | 'off'
BAD_NEWS_STOP_THRESHOLD = 0.20             # neg-only rolling sentiment >= this counts as "bad news"
                                           # (0.0-1.0 scale; 0.20 ~= at least one clearly negative headline in window)
BAD_NEWS_TIGHTEN_STOP_PCT = 0.003          # tighten stop to current_price * (1 - 0.003) = 0.3% below


# First time we saw a position (for min-hold gate). Populated by
# evaluate_chandelier_exit; cleared by clear_chandelier_state when the
# position is closed (called from the sell settlement path).
_chandelier_first_seen = {}
_chandelier_first_seen_lock = threading.Lock()


def _chandelier_first_seen_epoch(symbol):
    with _chandelier_first_seen_lock:
        ts = _chandelier_first_seen.get(symbol)
        if ts is None:
            ts = time.time()
            _chandelier_first_seen[symbol] = ts
        return ts


def clear_chandelier_state(symbol):
    with _chandelier_first_seen_lock:
        _chandelier_first_seen.pop(symbol, None)


def _chandelier_stop_price(entry_price, entry_atr, peak_since_entry,
                            current_price=None, bad_news=False):
    """Chandelier long-side stop. Returns None if inputs are unusable.

    When bad_news=True AND current_price is available, tightens the stop to
    current_price * (1 - BAD_NEWS_TIGHTEN_STOP_PCT), but only if that price
    is HIGHER than the normal ATR-based stop (the tightener can only make
    the stop tighter, never looser)."""
    if not entry_price or entry_price <= 0:
        return None
    if not entry_atr or entry_atr <= 0:
        return None
    if peak_since_entry is None or peak_since_entry <= 0:
        peak_since_entry = entry_price
    # Peak can never be below entry for our purposes (the highest price the
    # position has printed since we bought it, floored at entry).
    peak = max(peak_since_entry, entry_price)
    normal_stop = peak - CHANDELIER_ATR_MULT * entry_atr

    if bad_news and current_price and current_price > 0 and BAD_NEWS_STOP_MODE != 'off':
        tight_stop = current_price * (1.0 - BAD_NEWS_TIGHTEN_STOP_PCT)
        # Take the tighter (higher) of the two -- we're only allowed to
        # RAISE the stop on bad news, never lower it.
        return max(normal_stop, tight_stop)
    return normal_stop


def evaluate_chandelier_exit(symbol, entry_price, entry_atr, current_price,
                              peak_since_entry, position_opened_epoch):
    """Public helper used by sell_stocks. Returns (should_exit, stop_price, info_dict).

    should_exit is True only when CHANDELIER_MODE=='armed' AND
    current_price <= computed stop AND the min-hold gate has passed AND
    the peak has run at least CHANDELIER_REQUIRE_POSITIVE_RUN_PCT above entry.

    Bad-news override: if the symbol has meaningfully negative Alpaca news
    in the rolling window (neg_score >= BAD_NEWS_STOP_THRESHOLD) and
    BAD_NEWS_STOP_MODE=='armed', the stop tightens to 0.3% below current
    price AND the positive-run gate is bypassed -- we defend capital
    immediately rather than waiting for a peak.

    In 'shadow' mode returns should_exit=False but populates info so the
    caller can print the observed stop level.
    """
    # Look up bad-news signal (cached 15 min; free on a symbol already scored
    # by Brain F this cycle). Failure -> 0.0 -> bad_news stays False.
    neg_score = 0.0
    try:
        neg_score = float(brain_f_rolling_negative_sentiment(symbol))
    except Exception:
        neg_score = 0.0
    bad_news = (BAD_NEWS_STOP_MODE != 'off'
                and neg_score >= BAD_NEWS_STOP_THRESHOLD)

    stop = _chandelier_stop_price(entry_price, entry_atr, peak_since_entry,
                                   current_price=current_price,
                                   bad_news=bad_news)
    info = {
        'stop_price': stop,
        'peak_since_entry': peak_since_entry,
        'mode': CHANDELIER_MODE,
        'atr_mult': CHANDELIER_ATR_MULT,
        'entry_atr': entry_atr,
        'bad_news_score': neg_score,
        'bad_news_tightened': bad_news and BAD_NEWS_STOP_MODE == 'armed',
        'bad_news_mode': BAD_NEWS_STOP_MODE,
    }
    if stop is None:
        return False, None, info
    if CHANDELIER_MODE == 'off':
        return False, stop, info

    # Min-hold gate: never trigger in the first N seconds after we first
    # saw this position, to avoid a stop-out from noise right at entry.
    # (Applies even with bad news -- protects against news arriving in the
    # same minute as entry.)
    opened_epoch = position_opened_epoch or _chandelier_first_seen_epoch(symbol)
    if opened_epoch:
        age_secs = time.time() - opened_epoch
        if age_secs < CHANDELIER_MIN_HOLD_SECS:
            info['reason_gated'] = f'min_hold ({int(age_secs)}s < {CHANDELIER_MIN_HOLD_SECS}s)'
            return False, stop, info

    # Positive-run gate: only allow the chandelier to trigger once the peak
    # has printed above entry by at least the configured pct. BYPASSED when
    # bad news has actually tightened the stop -- on genuinely bad news we
    # don't wait for a peak to defend capital.
    if not info['bad_news_tightened']:
        peak = max(peak_since_entry or entry_price, entry_price)
        peak_run_pct = (peak - entry_price) / entry_price
        if peak_run_pct < CHANDELIER_REQUIRE_POSITIVE_RUN_PCT:
            info['reason_gated'] = (f'peak_run {peak_run_pct*100:.2f}% < '
                                     f'{CHANDELIER_REQUIRE_POSITIVE_RUN_PCT*100:.2f}%')
            return False, stop, info

    if CHANDELIER_MODE != 'armed':
        return False, stop, info

    if current_price <= stop:
        tag = 'BAD-NEWS STOP' if info['bad_news_tightened'] else 'chandelier'
        info['reason_fired'] = (f'current ${current_price:.2f} <= '
                                 f'{tag} ${stop:.2f}'
                                 + (f' (neg_news={neg_score:.2f})'
                                    if info['bad_news_tightened'] else ''))
        return True, stop, info
    return False, stop, info


def _get_peak_since_entry(symbol, entry_price, current_price):
    """Read the profit-monitor's tracked peak for this symbol; fall back to
    max(entry, current) if the monitor doesn't have state yet. This keeps
    the chandelier working from cycle #1 without needing its own state
    machine."""
    try:
        st = profit_monitor._state.get(symbol) if hasattr(profit_monitor, '_state') else None
        if st and st.get('peak_price'):
            return float(st['peak_price'])
    except Exception:
        pass
    return max(entry_price, current_price)


class ProfitMonitorEngine:
    """
    Peak-following exit. Instead of selling at the first tick above +0.5%, this
    arms at that level and then follows price to its high-water mark, selling
    only once price gives back PEAK_GIVEBACK_PCT from the peak.

    States per symbol:
      watching -> below the arm threshold, do nothing
      armed    -> above arm threshold, tracking peak_price
      exit     -> pulled back from peak, sell now

    There is no holding-period gate: a position can arm and exit the same
    second it was bought.
    """

    def __init__(self):
        self._state = {}          # symbol -> dict(peak_price, armed_at, last_seen, floor_pct)
        self._lock = threading.Lock()

    @staticmethod
    def _arm_threshold_for(atr_pct):
        """ATR-scaled arm threshold, or the flat fallback if ATR% is unavailable."""
        if atr_pct is None or atr_pct <= 0:
            return ARM_PROFIT_PCT
        return max(ARM_PROFIT_PCT, ATR_ARM_MULTIPLIER * atr_pct)

    @staticmethod
    def _giveback_for_peak(peak_gain_pct, arm_pct):
        """
        REVIEW ITEM #9: giveback now scales with how far the position has
        ACTUALLY run (peak_gain_pct), not just with the fixed arm threshold
        from when it first armed. A position that ran to +6% gets a wider
        giveback allowance than one that just barely armed at +1.2%, letting
        a strong trend breathe instead of being cut at the same tight margin
        every time. The arm-based floor (ATR_GIVEBACK_FRACTION x arm_pct)
        still applies as a MINIMUM, so a position that hasn't moved much past
        arming doesn't get an unreasonably wide giveback either.
        """
        arm_based_floor = ATR_GIVEBACK_FRACTION * arm_pct
        peak_based = PEAK_GIVEBACK_FRACTION * peak_gain_pct
        return max(PEAK_GIVEBACK_PCT, arm_based_floor, peak_based)

    def evaluate(self, symbol, entry_price, current_price, atr_pct=None):
        """Returns (should_sell: bool, info: dict) for logging/telemetry."""
        now = time.time()
        if not entry_price or entry_price <= 0 or not current_price or current_price <= 0:
            return False, {'state': 'invalid'}

        arm_pct = self._arm_threshold_for(atr_pct)
        gain = (current_price - entry_price) / entry_price

        with self._lock:
            st = self._state.get(symbol)

            # Not yet armed: wait for +arm_pct (ATR-scaled, or the flat fallback).
            # BUGFIX (round-trip rescue): even before arming, TRACK the peak so we
            # can detect a position that was profitable and then reversed into a
            # loss. Without this, a run to +0.35% followed by a drop to -1% was
            # invisible to the monitor and had to wait for the hard stop.
            if st is None:
                # First time we've seen this symbol -- seed the pre-arm state.
                self._state[symbol] = {
                    'peak_price': max(entry_price, current_price),
                    'armed_at': None,          # None => not yet armed
                    'last_seen': now,
                    'floor_pct': HARD_FLOOR_PCT,
                    'arm_pct': arm_pct,
                    'touched_profit': gain >= PROFIT_TOUCHED_PCT,
                }
                st = self._state[symbol]
            else:
                st['last_seen'] = now
                # Ratchet peak upward regardless of armed state.
                if current_price > st['peak_price']:
                    st['peak_price'] = current_price
                # Latch the "was profitable" flag once we touch the rescue level.
                if not st.get('touched_profit') and gain >= PROFIT_TOUCHED_PCT:
                    st['touched_profit'] = True

            # ---- Pre-arm branch: position has never crossed the arm threshold ----
            if st.get('armed_at') is None:
                # Cross the arm threshold now? Promote to armed and fall through
                # to the armed logic below with the peak we've been tracking.
                if gain >= arm_pct:
                    st['armed_at'] = now
                    st['arm_pct'] = arm_pct   # freeze arm_pct at moment of arming
                    peak_gain_now = (st['peak_price'] - entry_price) / entry_price
                    return False, {'state': 'armed', 'gain_pct': gain * 100,
                                   'peak_price': st['peak_price'],
                                   'peak_gain_pct': peak_gain_now * 100}

                # ROUND-TRIP RESCUE: touched a small profit then round-tripped
                # to breakeven or below. Cut it here rather than let the hard
                # stop absorb the full reversal.
                if st.get('touched_profit') and gain <= ROUND_TRIP_EXIT_GAIN:
                    peak_gain_now = (st['peak_price'] - entry_price) / entry_price
                    return True, {'state': 'exit_roundtrip', 'gain_pct': gain * 100,
                                  'peak_price': st['peak_price'],
                                  'peak_gain_pct': peak_gain_now * 100}

                # Still watching, still sub-arm, no rescue trigger.
                return False, {'state': 'watching', 'gain_pct': gain * 100,
                               'arm_at_pct': arm_pct * 100,
                               'touched_profit': st.get('touched_profit', False),
                               'peak_gain_pct': ((st['peak_price'] - entry_price)
                                                 / entry_price) * 100}

            # ---- Armed branch ----
            # last_seen + peak ratchet already handled at the top of the lock.
            # If this tick set a new peak, report it and don't evaluate exits
            # (giveback is 0 by definition on a new peak).
            peak = st['peak_price']
            if current_price >= peak and gain > 0:
                # peak was just updated to current_price above
                return False, {'state': 'new_peak', 'gain_pct': gain * 100,
                               'peak_price': peak, 'peak_gain_pct': gain * 100}

            peak_gain = (peak - entry_price) / entry_price
            giveback = (peak - current_price) / peak
            floor_pct = st.get('floor_pct', HARD_FLOOR_PCT)
            # Use the arm_pct RECORDED at arm-time for this position (not a
            # possibly-different current ATR reading), so the floor component
            # of the giveback calc stays stable for the life of this armed run.
            arm_pct_for_giveback = st.get('arm_pct', arm_pct)
            giveback_pct = self._giveback_for_peak(peak_gain, arm_pct_for_giveback)

            info = {'state': 'following', 'gain_pct': gain * 100,
                    'peak_price': peak, 'peak_gain_pct': peak_gain * 100,
                    'giveback_pct': giveback * 100, 'giveback_target_pct': giveback_pct * 100}

            # BUGFIX: giveback trigger no longer gated by `gain >= floor_pct`.
            # Previously, if a position peaked at +2% and gapped down to just
            # below the +0.1% floor, the 'exit' branch was skipped (gain not
            # above floor) and only the separate 'exit_floor' branch caught it.
            # That worked for the loss case but produced misleading telemetry
            # (the log said "collapsed to floor" when it really was a giveback
            # trigger) and the two-branch structure was fragile. Fire 'exit'
            # whenever giveback is met, and let 'exit_floor' handle the case
            # where the peak was tiny and giveback never triggered but the
            # position collapsed below floor on its own.
            if giveback >= giveback_pct:
                info['state'] = 'exit'
                return True, info

            # Collapsed below this position's floor after arming without ever
            # tripping the giveback trigger (small peak, straight-line drop):
            # cut it here rather than round-trip a winner into a loser. The
            # floor rises to breakeven once a scale-out stage has fired
            # (raise_floor_to_breakeven).
            if gain < floor_pct:
                info['state'] = 'exit_floor'
                return True, info

            return False, info

    def clear(self, symbol):
        with self._lock:
            self._state.pop(symbol, None)

    def raise_floor_to_breakeven(self, symbol, min_floor_pct=0.0005):
        """
        REVIEW ITEM #5: after a scale-out tranche sells part of the position,
        move the remainder's exit floor up to (near) breakeven so a reversal
        can no longer turn the remaining shares into a loser. Called with the
        state already armed (a scale-out only fires above the arm threshold).
        """
        with self._lock:
            st = self._state.get(symbol)
            if st is not None:
                st['floor_pct'] = max(st.get('floor_pct', HARD_FLOOR_PCT), min_floor_pct)

    def prune(self, live_symbols):
        """Drop state for positions that no longer exist or went stale."""
        now = time.time()
        with self._lock:
            for sym in list(self._state):
                if sym not in live_symbols or (now - self._state[sym]['last_seen']) > MONITOR_STALE_SECS:
                    self._state.pop(sym, None)

    def snapshot(self):
        with self._lock:
            return {s: dict(v) for s, v in self._state.items()}


profit_monitor = ProfitMonitorEngine()

# ---------------- Scaled-exit tracking (review item #5) ----------------
# symbol -> {'original_qty': float, 'stages_fired': set(stage_index)}
# Tracks how much of a position's ORIGINAL size has already been scaled out,
# so stage triggers are evaluated against the position as it was at entry,
# not against whatever qty remains after prior partial sells.
_scale_out_lock = threading.Lock()
_scale_out_state = {}


def _scale_out_get_or_init(symbol, current_qty):
    with _scale_out_lock:
        st = _scale_out_state.get(symbol)
        if st is None:
            st = {'original_qty': current_qty, 'stages_fired': set()}
            _scale_out_state[symbol] = st
        return st


def _scale_out_mark_fired(symbol, stage_index):
    with _scale_out_lock:
        st = _scale_out_state.get(symbol)
        if st is not None:
            st['stages_fired'].add(stage_index)


def _scale_out_clear(symbol):
    with _scale_out_lock:
        _scale_out_state.pop(symbol, None)


def sell_stocks(lock):
    print("Starting sell_stocks function...")
    to_remove = []
    now_str = datetime.now(eastern).strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")
    today_date_str = datetime.now(eastern).date().strftime("%Y-%m-%d")

    # Dashboard-initiated sell-all: if the flag has been set, we treat every
    # currently-held position as if the chandelier/hard stop fired -- market
    # sell full quantity via the same escalation chain. Consumed exactly
    # once so a stuck flag can't fire twice.
    _dash_sell_all = consume_dashboard_sell_all()
    if _dash_sell_all:
        print(f"  [dashboard] SELL ALL requested — closing every position at market.")
        logging.warning(f"{now_str} [dashboard] SELL ALL requested via WebSocket.")

    # NO holding-period gate. PDT is retired under the 2026 margin rules, so a
    # position may be sold the same second it was bought. purchase_date is now
    # recorded for reporting only.
    profit_monitor.prune(position_book.symbols())

    # BUGFIX: iterate a SNAPSHOT. Previously this walked the live shared dict
    # while buy_stocks mutated it -> "dictionary changed size during iteration".
    for symbol, (bought_price, purchase_date) in position_book.snapshot().items():
        # BUGFIX: per-symbol claim stops buy_stocks and sell_stocks from acting
        # on the same symbol at once (a fill landing mid-exit-decision).
        if not position_book.claim(symbol):
            print(f"{symbol}: busy in another thread this cycle. Skipping.")
            continue
        try:
            current_price = get_current_price(symbol)
            if current_price is None:
                continue

            position = api.get_position(symbol)
            bought_price = float(position.avg_entry_price)
            qty = float(position.qty)

            atr = get_average_true_range(symbol)
            atr_pct = (atr / current_price) if atr and current_price else None
            gain_now = (current_price - bought_price) / bought_price if bought_price else 0.0

            # Dashboard SELL-ALL short-circuit: bypass all other exit logic,
            # market-sell full quantity via the standard escalation chain.
            if _dash_sell_all:
                print(f"{symbol}: {RED}[dashboard SELL ALL]{RESET} closing "
                      f"{qty:.4f} sh at market.")
                _cancel_existing_sell_orders(symbol, "DASHBOARD SELL ALL", now_str)
                filled_qty, notional, steps = _sell_with_escalation(
                    symbol, qty, current_price, 'market', 'day',
                    "DASHBOARD SELL ALL", now_str)
                if filled_qty > 0:
                    avg_fill_price = notional / filled_qty
                    logging.info(f"{now_str} DASHBOARD SELL ALL sold {symbol}: "
                                f"{filled_qty:.4f} sh @ ${avg_fill_price:.2f} "
                                f"via [{', '.join(steps)}].")
                    to_remove.append((symbol, filled_qty, avg_fill_price))
                continue

            # Brain thinking (SELL side): prints P(win) forward from HERE plus
            # top feature contributions so the operator can see what the model
            # thinks about the held position each cycle. Diagnostic only --
            # the actual sell decision remains rule-based (trailing stop,
            # hard stop, timeout, take-profit, scale-out).
            try:
                _sell_df = get_cached_data(symbol, 'history_90d',
                                           lambda s: yf_history(s, period="90d"),
                                           symbol)
                if _sell_df is not None and not _sell_df.empty:
                    explain_sell_decision(symbol, _sell_df, current_price)
            except Exception as _e:
                logging.debug(f"{symbol}: sell-side brain thinking skipped ({_e}).")

            # ---------------- Bull-market strategy exit (regime-gated at buy) ----------------
            # Positions opened by the ported Bull Market v8 buy path exit at a
            # simple +0.5% flat target (BULL_TAKE_PROFIT_PCT). The broker-side
            # 1% trailing stop placed at fill handles the downside; this branch
            # handles the upside. Runs BEFORE the hard-stop / profit-monitor /
            # scale-out chain so bull positions never enter that logic.
            _bull_pos = None
            try:
                _bull_pos = session.query(Position).filter_by(symbols=symbol).one_or_none()
            except Exception as e:
                logging.warning(f"{symbol}: could not check strategy_tag: {e}")
            if _bull_pos is not None and _bull_pos.strategy_tag == 'bull':
                bull_target = bought_price * BULL_TAKE_PROFIT_PCT
                print(f"[bull] {symbol}: bought ${bought_price:.2f}, current ${current_price:.2f}, "
                      f"target ${bull_target:.2f} (+{(BULL_TAKE_PROFIT_PCT-1)*100:.2f}%)")
                if current_price >= bull_target:
                    # Cancel the broker-side trailing stop first so the whole
                    # position (not just the unreserved fraction) can be sold.
                    if not cancel_open_sell_orders(symbol):
                        print(f"[bull] {symbol}: could not free reserved shares. Retrying next cycle.")
                        continue
                    try:
                        # Refresh qty after cancel in case the stop had partially
                        # filled while we were cancelling it.
                        position = api.get_position(symbol)
                        qty = float(position.qty)
                    except Exception as e:
                        logging.error(f"[bull] {symbol}: could not refresh qty after cancel: {e}")
                        continue
                    try:
                        api.submit_order(
                            symbol=symbol, qty=qty, side='sell',
                            type='market', time_in_force='day',
                        )
                        print(f"[bull] {now_str}, SOLD {qty:.4f} sh of {symbol} @ ${current_price:.2f} "
                              f"(target {BULL_TAKE_PROFIT_PCT}x hit)")
                        logging.info(f"{now_str} [bull] Sold {qty:.4f} sh of {symbol} @ ${current_price:.2f}")
                        with open(csv_filename, mode='a', newline='') as f:
                            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                                'Date': now_str, 'Buy': '', 'Sell': 'Sell',
                                'Quantity': qty, 'Symbol': symbol,
                                'Price Per Share': current_price,
                            })
                        to_remove.append((symbol, qty, current_price))
                    except Exception as e:
                        print(f"[bull] {symbol}: sell order error: {e}")
                        logging.error(f"[bull] {symbol}: sell order error: {e}")
                # Bull-tagged positions skip the ML exit chain entirely --
                # only the flat +0.5% target or the broker-side trailing stop
                # ever closes them. Continue to next symbol regardless.
                continue

            # ---------------- Hard stop-loss (review items #6/#7 + item #1) ----------------
            # Fires independently of the profit monitor's armed/unarmed state.
            #
            # Item #1 fix (risk-per-trade honesty): the stop distance is now
            # calculated from the FROZEN entry-time ATR captured at buy time
            # (Position.entry_atr), NOT the current-cycle ATR. This closes
            # a real loophole: previously, if ATR spiked after entry (say a
            # macro-news shock), the stop distance would silently widen from
            # -3% to -8% under the position sizer's feet -- meaning the "1%
            # of equity" risk model this bot's sizing assumes was actually
            # allowing much larger losses than the operator agreed to.
            #
            # Additionally, the stop can only ever TIGHTEN, never widen: if
            # current ATR happens to have shrunk below entry ATR, we use the
            # smaller value; but if current ATR grew, we stay locked at the
            # entry-time value. The MIN_PCT floor applies too.
            #
            # A position without a captured entry_atr (older row from before
            # this migration) falls back to the current ATR as a best effort.
            try:
                _db_pos = session.query(Position).filter_by(symbols=symbol).one_or_none()
                entry_atr = _db_pos.entry_atr if _db_pos else None
            except Exception:
                entry_atr = None
            stop_atr = min(entry_atr, atr) if (entry_atr and atr) else (entry_atr or atr)
            if USE_HARD_STOP_LOSS and stop_atr and stop_atr > 0 and bought_price:
                stop_distance_pct = max(HARD_STOP_ATR_MULTIPLIER * (stop_atr / bought_price),
                                        HARD_STOP_MIN_PCT)
                if gain_now <= -stop_distance_pct:
                    stop_source = "entry-time" if entry_atr and stop_atr == entry_atr else "current"
                    print(f"{symbol}: {RED}HARD STOP{RESET} {gain_now*100:.2f}% <= "
                          f"-{stop_distance_pct*100:.2f}% ({HARD_STOP_ATR_MULTIPLIER:.1f}x "
                          f"{stop_source} ATR, floored at -{HARD_STOP_MIN_PCT*100:.1f}%). "
                          f"Selling full position via escalation chain regardless of "
                          f"profit-monitor state.")
                    logging.warning(f"{now_str} HARD STOP triggered for {symbol}: "
                                    f"{gain_now*100:.2f}% <= -{stop_distance_pct*100:.2f}% "
                                    f"(entry ${bought_price:.2f}, current ${current_price:.2f}, "
                                    f"stop_ATR ${stop_atr:.2f} [{stop_source}]).")
                    _cancel_existing_sell_orders(symbol, "HARD STOP", now_str)
                    filled_qty, notional, steps = _sell_with_escalation(
                        symbol, qty, current_price, 'market', 'day', "HARD STOP", now_str)
                    if filled_qty > 0:
                        avg_fill_price = notional / filled_qty
                        logging.info(f"{now_str} HARD STOP sold {symbol}: {filled_qty:.4f} sh "
                                    f"@ avg ${avg_fill_price:.2f} via [{', '.join(steps)}].")
                        # profit_monitor/_scale_out state is cleared by the
                        # to_remove consumer loop below, but ONLY once the
                        # position is fully closed -- a partial hard-stop fill
                        # correctly leaves that state in place for the shares
                        # still open, same as any other partial sell.
                        to_remove.append((symbol, filled_qty, avg_fill_price))
                    else:
                        print(f"{symbol}: {RED}hard stop escalation produced no fill{RESET} "
                              f"(see warnings above) -- position remains open, will retry "
                              f"the stop check next cycle.")
                    continue  # done with this symbol for this cycle either way

            # ---------------- Chandelier ATR trailing stop ----------------
            # Dynamic volatility-adjusted trail. Fires between the deep hard
            # stop (which only trips at 2x ATR from entry) and the profit
            # monitor (which only arms after the position is profitable).
            # This catches positions that rallied and rolled over before the
            # profit monitor armed. Uses entry_atr (frozen at buy) so the
            # stop doesn't wander with intraday volatility.
            if entry_atr and entry_atr > 0 and bought_price:
                peak_seen = _get_peak_since_entry(symbol, bought_price, current_price)
                ch_fire, ch_stop, ch_info = evaluate_chandelier_exit(
                    symbol=symbol, entry_price=bought_price, entry_atr=entry_atr,
                    current_price=current_price, peak_since_entry=peak_seen,
                    position_opened_epoch=None)
                if ch_stop is not None:
                    peak_run = ((peak_seen - bought_price) / bought_price) * 100
                    bad_news_tight = ch_info.get('bad_news_tightened', False)
                    neg_score = ch_info.get('bad_news_score', 0.0)
                    stop_label = 'BAD-NEWS STOP' if bad_news_tight else 'CHANDELIER STOP'
                    if ch_fire:
                        detail = (f"(neg_news={neg_score:.2f}, tightened to "
                                  f"{BAD_NEWS_TIGHTEN_STOP_PCT*100:.2f}% below current)"
                                  if bad_news_tight else
                                  f"(peak ${peak_seen:.2f}, +{peak_run:.2f}% run, "
                                  f"{CHANDELIER_ATR_MULT:.1f}x entry ATR ${entry_atr:.2f})")
                        print(f"{symbol}: {RED}{stop_label}{RESET} @ ${ch_stop:.2f} "
                              f"{detail}. Selling full position.")
                        logging.warning(f"{now_str} {stop_label} for {symbol}: "
                                        f"current ${current_price:.2f} <= stop ${ch_stop:.2f} "
                                        f"(entry ${bought_price:.2f}, peak ${peak_seen:.2f}"
                                        + (f", neg_news={neg_score:.2f}" if bad_news_tight else "")
                                        + ").")
                        _cancel_existing_sell_orders(symbol, stop_label, now_str)
                        filled_qty, notional, steps = _sell_with_escalation(
                            symbol, qty, current_price, 'market', 'day',
                            stop_label, now_str)
                        if filled_qty > 0:
                            avg_fill_price = notional / filled_qty
                            logging.info(f"{now_str} {stop_label} sold {symbol}: "
                                        f"{filled_qty:.4f} sh @ avg ${avg_fill_price:.2f} "
                                        f"via [{', '.join(steps)}].")
                            to_remove.append((symbol, filled_qty, avg_fill_price))
                        else:
                            print(f"{symbol}: {RED}{stop_label.lower()} escalation produced no "
                                  f"fill{RESET} -- position remains open, will retry next cycle.")
                        continue
                    else:
                        # Shadow / gated / not-yet-triggered: print observation line
                        # so the operator can see where the stop is sitting each cycle.
                        gated = ch_info.get('reason_gated', '')
                        mode_tag = ('' if CHANDELIER_MODE == 'armed'
                                    else f' ({CHANDELIER_MODE.upper()})')
                        gated_tag = f' [{gated}]' if gated else ''
                        news_tag = (f' [BAD NEWS neg={neg_score:.2f} -> tightened '
                                    f'to {BAD_NEWS_TIGHTEN_STOP_PCT*100:.2f}%]'
                                    if bad_news_tight else '')
                        distance_pct = (ch_stop - current_price) / current_price * 100
                        print(f"  [chandelier{mode_tag}] {symbol}: stop @ ${ch_stop:.2f} "
                              f"(peak +{peak_run:.2f}%, current is {distance_pct:+.2f}% "
                              f"from stop){gated_tag}{news_tag}")

            # ---------------- Scaled exits (review item #5) ----------------
            # Checked before the peak-following exit: sell a fixed fraction of
            # the ORIGINAL position at each configured gain milestone, moving
            # the profit monitor's floor to breakeven once the first stage
            # fires. The remainder keeps running under the normal monitor.
            #
            # BUGFIX: the order used to be marked "fired" (and the floor
            # raised to breakeven) immediately after submit_order() returned,
            # with no check on what the broker actually did with it. A
            # rejected, delayed, canceled, or partially-filled order would
            # still be treated as a completed stage -- the bot could believe
            # it locked in profit on shares that were never actually sold.
            # Now the order is polled to a terminal status first; the stage is
            # only marked fired for the quantity that ACTUALLY filled, using
            # the real fill price for logging, and the floor is only raised
            # if at least some of the tranche confirmed filled.
            if USE_SCALED_EXITS and SCALE_OUT_STAGES:
                sc_state = _scale_out_get_or_init(symbol, qty)
                for idx, (trigger_pct, frac) in enumerate(SCALE_OUT_STAGES):
                    if idx in sc_state['stages_fired']:
                        continue
                    if gain_now < trigger_pct:
                        break  # stages are in ascending order; none further can fire yet
                    scale_qty = round(sc_state['original_qty'] * frac, 4)
                    scale_qty = min(scale_qty, qty)
                    if scale_qty <= 0:
                        _scale_out_mark_fired(symbol, idx)
                        continue
                    print(f"{symbol}: {GREEN}+{gain_now*100:.2f}%{RESET} hit scale-out stage "
                          f"{idx+1} (+{trigger_pct*100:.1f}%) — selling {scale_qty:.4f} sh "
                          f"({frac*100:.0f}% of original). Awaiting fill confirmation before "
                          f"marking complete.")
                    try:
                        so = api.submit_order(symbol=symbol, qty=str(scale_qty), side='sell',
                                              type='market', time_in_force='day')
                    except Exception as e:
                        print(f"{symbol}: scale-out stage {idx+1} submit failed: {e}")
                        logging.error(f"{symbol}: scale-out stage {idx+1} submit failed: {e}")
                        break

                    terminal, filled_qty, filled_price, status = _poll_order_terminal(
                        so.id, SCALE_OUT_FILL_TIMEOUT_SECS)

                    if not terminal:
                        # Still working after the poll budget -- don't guess.
                        # Leave the stage unmarked so the NEXT cycle re-checks
                        # this same order's outcome rather than assuming
                        # anything about it now.
                        print(f"{symbol}: scale-out stage {idx+1} order {so.id} still "
                              f"{status} after {SCALE_OUT_FILL_TIMEOUT_SECS}s; will "
                              f"re-check next cycle. Not marking the stage complete yet.")
                        logging.info(f"{now_str} Scale-out stage {idx+1} for {symbol}: "
                                    f"order {so.id} not yet terminal (status={status}); "
                                    f"deferring to next cycle.")
                        break

                    if filled_qty <= 0:
                        # Rejected/canceled/expired with nothing filled: the
                        # tranche did not execute at all. Do NOT mark fired and
                        # do NOT raise the floor -- retry this same stage on a
                        # later cycle instead of silently losing the attempt.
                        print(f"{symbol}: scale-out stage {idx+1} order {status} with "
                              f"no fill. Will retry this stage on a later cycle.")
                        logging.warning(f"{now_str} Scale-out stage {idx+1} for {symbol}: "
                                       f"order {so.id} ended {status} with 0 filled qty.")
                        break

                    actual_price = filled_price if filled_price else current_price
                    if filled_qty < scale_qty:
                        print(f"{symbol}: scale-out stage {idx+1} PARTIALLY filled — "
                              f"{filled_qty:.4f} of {scale_qty:.4f} sh @ ${actual_price:.2f} "
                              f"(status={status}). Marking this stage complete for the "
                              f"filled amount only; the unfilled remainder is not resubmitted "
                              f"automatically (rare edge case worth checking manually).")
                        logging.warning(f"{now_str} Scale-out stage {idx+1} for {symbol}: "
                                       f"partial fill {filled_qty:.4f}/{scale_qty:.4f} sh "
                                       f"@ ${actual_price:.2f} (status={status}).")
                    else:
                        print(f"{symbol}: scale-out stage {idx+1} CONFIRMED filled — "
                              f"{filled_qty:.4f} sh @ ${actual_price:.2f}. Moving floor to breakeven.")
                        logging.info(f"{now_str} Scale-out stage {idx+1} for {symbol}: "
                                    f"confirmed filled {filled_qty:.4f} sh @ ${actual_price:.2f}.")

                    # Only reached once we KNOW at least part of the tranche
                    # actually filled -- this is the fix: fired/breakeven are
                    # now consequences of a confirmed fill, not of submission.
                    _scale_out_mark_fired(symbol, idx)
                    profit_monitor.raise_floor_to_breakeven(symbol)
                    with open(csv_filename, mode='a', newline='') as f:
                        csv.DictWriter(f, fieldnames=fieldnames).writerow({
                            'Date': now_str, 'Buy': '', 'Sell': 'Sell (scale-out)',
                            'Quantity': filled_qty, 'Symbol': symbol,
                            'Price Per Share': actual_price,
                        })
                    # Only fire one stage per cycle; re-evaluate qty/gain next pass.
                    break

            # ---------------- Exit decision (remaining shares) ----------------
            if USE_PROFIT_MONITOR:
                should_sell, info = profit_monitor.evaluate(symbol, bought_price, current_price, atr_pct=atr_pct)
                state = info.get('state')
                if state in ('watching',):
                    touched = info.get('touched_profit', False)
                    tp_tag = (f" [was +{info.get('peak_gain_pct', 0):.2f}%]"
                              if touched else "")
                    print(f"{symbol}: {pnl_color(info['gain_pct'])}{info['gain_pct']:+.2f}%{RESET} (arms at "
                          f"+{info['arm_at_pct']:.2f}%){tp_tag}. Holding.")
                    continue
                if state in ('armed', 'new_peak'):
                    print(f"{symbol}: {GREEN}{info['gain_pct']:+.2f}%{RESET} "
                          f"peak ${info['peak_price']:.2f} — following.")
                    continue
                if state == 'following' and not should_sell:
                    gvt = info.get('giveback_target_pct', PEAK_GIVEBACK_PCT * 100)
                    print(f"{symbol}: {GREEN}{info['gain_pct']:+.2f}%{RESET} "
                          f"peak +{info['peak_gain_pct']:.2f}% "
                          f"(giveback {info['giveback_pct']:.2f}% of "
                          f"{gvt:.2f}%). Following.")
                    continue
                if not should_sell:
                    continue
                if state == 'exit_roundtrip':
                    # BUGFIX: sub-arm profit rescue -- position touched a small
                    # profit then round-tripped to breakeven-or-below without
                    # ever arming the peak follower.
                    reason = (f"round-trip rescue: touched +{info['peak_gain_pct']:.2f}% "
                              f"then fell back to {info['gain_pct']:+.2f}% — exiting "
                              f"before it drops further")
                elif state == 'exit_floor':
                    reason = (f"dropped to {info['gain_pct']:+.2f}% after peaking "
                              f"+{info['peak_gain_pct']:.2f}% — cutting at floor")
                else:
                    reason = (f"peaked +{info['peak_gain_pct']:.2f}%, gave back "
                              f"{info['giveback_pct']:.2f}% — taking {info['gain_pct']:+.2f}%")
            else:
                sell_threshold = bought_price * TAKE_PROFIT_PCT
                if current_price < sell_threshold:
                    print(f"{symbol}: {RED}${current_price:.2f}{RESET} < target ${sell_threshold:.2f}. Holding.")
                    continue
                reason = f"hit +{(TAKE_PROFIT_PCT-1)*100:.2f}% target"

            # BUGFIX: cancel the resting trailing stop BEFORE selling. It reserves
            # shares at the broker, so without this the take-profit could only ever
            # sell the unreserved fraction and the whole-share portion was stuck.
            if not cancel_open_sell_orders(symbol):
                print(f"{symbol}: could not clear resting sell orders. Skipping this cycle.")
                continue

            # Re-read the position after cancellation: qty_available now reflects
            # the freed shares, and the position may have changed size.
            try:
                position = api.get_position(symbol)
            except Exception as e:
                print(f"{symbol}: position gone after cancel ({e}). Skipping.")
                logging.info(f"{symbol}: position no longer exists after cancel: {e}")
                continue

            qty = float(position.qty)
            qty_available = float(getattr(position, 'qty_available', qty) or qty)
            # BUGFIX: sell exactly what the broker says is sellable. Rounding a
            # fractional qty to 4dp could exceed the real position and be rejected.
            sell_qty = min(qty, qty_available)
            if sell_qty <= 0:
                print(f"{symbol}: nothing available to sell. Skipping.")
                continue

            print(f"Selling {sell_qty} sh of {symbol} @ {GREEN}${current_price:.2f}{RESET} "
                  f"(entry ${bought_price:.2f}) — {reason}")
            sell_order = api.submit_order(symbol=symbol, qty=str(sell_qty), side='sell',
                                          type='market', time_in_force='day')
            logging.info(f"{now_str} Submitted sell {sell_qty} sh of {symbol} at ~{current_price:.2f}: {reason}")

            # BUGFIX: the original never confirmed the sell filled -- it deleted the
            # DB row immediately, so a rejected sell silently desynced the DB from
            # the broker and the bot believed it was flat while still holding shares.
            sold_qty = 0.0
            sold_price = current_price
            for _ in range(15):
                try:
                    so = api.get_order(sell_order.id)
                except Exception as e:
                    logging.warning(f"{symbol}: sell poll error ({e}); retrying.")
                    time.sleep(2)
                    continue
                sold_qty = float(so.filled_qty or 0)
                if so.filled_avg_price:
                    sold_price = float(so.filled_avg_price)
                if so.status == 'filled':
                    break
                if so.status in ('canceled', 'expired', 'rejected'):
                    logging.warning(f"{symbol}: sell order {so.status}, filled {sold_qty:.4f}.")
                    break
                time.sleep(2)

            if sold_qty <= 0:
                print(f"{symbol}: sell did not fill. Position retained.")
                logging.warning(f"{now_str} Sell not filled for {symbol}; DB row retained.")
                continue

            print(f"Sold {sold_qty:.4f} sh of {symbol} @ {GREEN}${sold_price:.2f}{RESET}")
            logging.info(f"{now_str} Sold {sold_qty:.4f} sh of {symbol} at {sold_price:.2f}")

            with open(csv_filename, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow({
                    'Date': now_str, 'Buy': '', 'Sell': 'Sell',
                    'Quantity': sold_qty, 'Symbol': symbol,
                    'Price Per Share': sold_price,
                })
            to_remove.append((symbol, sold_qty, sold_price))

        except Exception as e:
            print(f"Error processing sell for {symbol}: {e}")
            logging.error(f"Error processing sell for {symbol}: {e}")
        finally:
            # BUGFIX: always release the claim, even on the `continue` paths and
            # on exception, or the symbol is permanently locked out of trading.
            position_book.release(symbol)

    if not to_remove:
        return
    try:
        with lock:
            for symbol, qty, price in to_remove:
                session.add(TradeHistory(symbols=symbol, action='sell',
                                         quantity=qty, price=price, date=today_date_str))
                # BUGFIX: a partial sell used to delete the whole Position row,
                # making the bot forget shares it still owned. Decrement instead,
                # and only remove the row when the position is actually closed.
                row = session.query(Position).filter_by(symbols=symbol).one_or_none()
                if row and (row.quantity - qty) > 1e-6:
                    row.quantity -= qty
                    print(f"{symbol}: partial sell, {row.quantity:.4f} sh still held.")
                else:
                    session.query(Position).filter_by(symbols=symbol).delete()
                    position_book.remove(symbol)
                    # Reset peak tracking so a later re-buy starts a fresh run
                    # rather than inheriting the old position's high-water mark.
                    profit_monitor.clear(symbol)
                    _scale_out_clear(symbol)
                    clear_chandelier_state(symbol)

                    # REVIEW ITEM #7: fill in the outcome on the most recent
                    # still-open TradeFeatures row for this symbol, so the
                    # entry-feature snapshot can be joined against what
                    # actually happened. Only closes the row once the position
                    # is fully flat, matching how a "trade" is defined here.
                    feat_row = (session.query(TradeFeatures)
                               .filter_by(symbols=symbol, exit_date=None)
                               .order_by(TradeFeatures.id.desc())
                               .first())
                    if feat_row and feat_row.entry_price:
                        feat_row.exit_date = today_date_str
                        feat_row.exit_price = price
                        feat_row.outcome_pct = (price - feat_row.entry_price) / feat_row.entry_price

                        # Kalshi-ported settlement hooks: record per-symbol
                        # performance, feed the trade governor, and settle
                        # any pending brain-trust predictions for this symbol.
                        try:
                            pnl_dollar = (price - feat_row.entry_price) * qty
                            won = feat_row.outcome_pct > 0
                            PERSYMBOL.record(symbol, won, pnl_dollar)
                            GOVERNOR.on_close(won)
                            brain_trust_settle_symbol(symbol, won)
                            # Any losing trade auto-blacklists the symbol for 72h.
                            # Winner clears any prior temp block (fresh proof it works).
                            if not won:
                                BLACKLIST.add_temporary(symbol)
                                floor_post('SAFETY', 'lesson',
                                           f"{symbol} loss ${pnl_dollar:+.2f} — 72h auto-blacklisted")
                            else:
                                floor_post('BOT', 'observation',
                                           f"{symbol} closed +${pnl_dollar:.2f}")
                        except Exception as _e:
                            logging.debug(f"settlement hooks failed for {symbol}: {_e}")
            session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Database error: {e}")
        logging.error(f"Database error: {e}")
        return

    # BUGFIX: refresh_after_sell() used to run INSIDE `with lock`. It makes
    # blocking API calls (list_positions + a paginated order lookup per symbol),
    # holding the mutex for tens of seconds and serializing both threads. It is
    # now called after the lock is released.
    refresh_after_sell()


def refresh_after_sell():
    # BUGFIX: no longer rebinds a global. replace_all() mutates the single shared
    # PositionBook in place, so both threads keep seeing the same object.
    position_book.replace_all(update_symbols_to_sell_from_api())


def load_positions_from_database():
    return {p.symbols: (p.avg_price, p.purchase_date) for p in session.query(Position).all()}


def reconcile_positions_on_startup():
    """
    Alpaca is the single source of truth. The local .db is only a cache.

    BUGFIX: main() previously did `load_positions_from_database()` and then only
    called the API `if not symbols_to_sell_dict` -- i.e. it ONLY synced when the
    DB was empty. A non-empty stale DB was therefore NEVER reconciled, so after a
    restart the bot would:
      - try to sell phantom positions closed while it was down (endless
        "position does not exist" errors), and
      - be blind to positions opened by hand or by another process.

    On startup we now:
      1. Pull live positions from Alpaca.
      2. DELETE any DB row with no matching live position.
      3. Insert/update rows for every live position (correcting drifted qty and
         avg_price, since the broker's numbers are authoritative).
      4. Re-arm the profit monitor so an in-flight winner keeps following its
         peak across the restart instead of dumping at the first tick.

    Raises on API failure: starting up on an unverified DB is more dangerous
    than not starting at all.
    """
    print("\n--- Reconciling local database against Alpaca positions ---")

    try:
        live_positions = api.list_positions()
    except Exception as e:
        # Do NOT silently fall back to the stale DB.
        msg = f"FATAL: cannot reach Alpaca to reconcile positions on startup: {e}"
        print(f"{RED}{msg}{RESET}")
        logging.critical(msg)
        raise

    live = {}
    for p in live_positions:
        try:
            live[p.symbol] = {'qty': float(p.qty), 'avg_price': float(p.avg_entry_price)}
        except (TypeError, ValueError) as e:
            logging.error(f"Skipping malformed position {getattr(p, 'symbol', '?')}: {e}")

    db_rows = {r.symbols: r for r in session.query(Position).all()}

    # --- 1. Drop DB rows Alpaca does not know about ---
    orphans = [s for s in db_rows if s not in live]
    for sym in orphans:
        row = db_rows[sym]
        print(f"  {RED}REMOVED{RESET} {sym}: in local DB ({row.quantity:.4f} sh @ "
              f"${row.avg_price:.2f}) but NOT held at Alpaca — deleting stale row.")
        logging.warning(f"Startup reconcile: deleting stale DB position {sym} "
                        f"(qty={row.quantity}, avg={row.avg_price}); not present at broker.")
        session.delete(row)
        profit_monitor.clear(sym)
        _scale_out_clear(sym)
        clear_chandelier_state(sym)

    # --- 2. Insert/correct rows for live positions ---
    result = {}
    for sym, info in live.items():
        qty, avg = info['qty'], info['avg_price']
        row = db_rows.get(sym)

        if row is None:
            pdate = get_most_recent_purchase_date(sym)
            print(f"  {GREEN}ADDED{RESET}   {sym}: held at Alpaca ({qty:.4f} sh @ "
                  f"${avg:.2f}) but missing locally — inserting.")
            logging.warning(f"Startup reconcile: adding untracked broker position {sym}.")
            session.add(Position(symbols=sym, quantity=qty, avg_price=avg, purchase_date=pdate))
        else:
            pdate = row.purchase_date or get_most_recent_purchase_date(sym)
            drift_qty = abs(row.quantity - qty) > 1e-6
            drift_avg = abs(row.avg_price - avg) > 0.005
            if drift_qty or drift_avg:
                print(f"  {RED}CORRECTED{RESET} {sym}: DB had {row.quantity:.4f} sh @ "
                      f"${row.avg_price:.2f}, broker says {qty:.4f} sh @ ${avg:.2f}.")
                logging.warning(f"Startup reconcile: correcting {sym} to broker values.")
            else:
                print(f"  {GREEN}OK{RESET}      {sym}: {qty:.4f} sh @ ${avg:.2f}")
            row.quantity, row.avg_price, row.purchase_date = qty, avg, pdate

        result[sym] = (avg, pdate)

    try:
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.critical(f"Startup reconcile commit failed: {e}")
        raise

    # --- 3. Re-arm the profit monitor for positions already in profit ---
    # Without this, a position that had run to +3% before the restart would lose
    # its peak and exit at the next +0.5% tick, giving back the whole move.
    if USE_PROFIT_MONITOR:
        for sym, (avg, _pdate) in result.items():
            cp = get_current_price(sym)
            if cp is None:
                continue
            gain = (cp - avg) / avg if avg else 0
            if gain >= ARM_PROFIT_PCT:
                # Seed the peak at the current price. The true pre-restart peak is
                # unknowable, so this conservatively restarts the ratchet from here.
                profit_monitor.evaluate(sym, avg, cp)
                print(f"  Re-armed profit monitor for {sym} at {gain*100:+.2f}% "
                      f"(peak reset to current price).")

    kept, removed, added = len(result), len(orphans), len([s for s in live if s not in db_rows])
    summary = f"Reconcile complete: {kept} live position(s), {removed} stale row(s) deleted, {added} added."
    print(f"--- {summary} ---\n")
    logging.info(summary)
    return result


def _run_and_release(fn, *args):
    """
    Thread entry point. scoped_session gives each thread its own Session, which
    must be released when the thread finishes or its DB connection leaks.
    Also stops an unhandled exception in a worker from dying silently.
    """
    try:
        fn(*args)
    except Exception as e:
        print(f"Unhandled error in {fn.__name__}: {e}")
        logging.exception(f"Unhandled error in {fn.__name__}: {e}")
    finally:
        Session.remove()


_LAST_PORTFOLIO_FETCH_TS = 0.0
_CACHED_PORTFOLIO_SUMMARY_TEXT = None
_PORTFOLIO_FETCH_MIN_INTERVAL_SEC = 3600.0  # refetch portfolio history at most 1/hour


def print_portfolio_gain_summary(force=False):
    """
    Print portfolio gain/loss over the past 24 hours, 7 days, and 14 days,
    using ONLY days when NYSE was open (via pandas_market_calendars).

    Data source: api.get_portfolio_history(period='1M', timeframe='1D'),
    which returns Alpaca's end-of-day equity marks -- already trading-day
    aligned, so we then match each equity mark to a valid NYSE session
    date and drop anything that doesn't line up (e.g. Alpaca timestamps
    falling on non-trading days due to timezone rounding).

    "Past N days" means: current equity vs the equity as of the trading
    session N sessions ago. If we don't have enough history yet (bot just
    launched, account is new), that window prints "insufficient history".

    FETCH vs PRINT: this function PRINTS every time it's called (so the
    main loop's 60s cadence shows a fresh section each cycle), but only
    RE-FETCHES portfolio history from Alpaca at most once per
    _PORTFOLIO_FETCH_MIN_INTERVAL_SEC (1 hour) -- in between calls it
    reprints the last rendered summary text. This keeps the terminal
    output present on every cycle without hammering the broker API for
    data that only updates end-of-day anyway. Pass force=True to bypass
    the fetch throttle on demand.
    """
    global _LAST_PORTFOLIO_FETCH_TS, _CACHED_PORTFOLIO_SUMMARY_TEXT
    now_ts = time.time()
    should_refetch = force or (now_ts - _LAST_PORTFOLIO_FETCH_TS) >= _PORTFOLIO_FETCH_MIN_INTERVAL_SEC

    if not should_refetch and _CACHED_PORTFOLIO_SUMMARY_TEXT is not None:
        # Reprint the cached rendering -- no network call, no recomputation.
        print(_CACHED_PORTFOLIO_SUMMARY_TEXT)
        return

    # ---------------- Fetch fresh portfolio history from Alpaca ----------------
    try:
        # 2M / 1D gives ~42 trading-day closes -- enough for the 30-day
        # (~21 trading sessions back) window plus a comfortable margin.
        # Alpaca returns lists in parallel: timestamp (unix), equity (float),
        # profit_loss, etc.
        hist = api.get_portfolio_history(period='2M', timeframe='1D')
    except Exception as e:
        msg = f"[portfolio] could not fetch portfolio history: {e}"
        print(msg)
        logging.warning(f"[portfolio] get_portfolio_history failed: {e}")
        # Fall back to whatever we had cached, if anything.
        if _CACHED_PORTFOLIO_SUMMARY_TEXT is not None:
            print(_CACHED_PORTFOLIO_SUMMARY_TEXT)
        return

    try:
        timestamps = list(hist.timestamp or [])
        equities = list(hist.equity or [])
    except Exception as e:
        print(f"[portfolio] malformed portfolio history response: {e}")
        return

    if not timestamps or not equities or len(timestamps) != len(equities):
        text = "[portfolio] no portfolio history returned yet (new account?)."
        print(text)
        _CACHED_PORTFOLIO_SUMMARY_TEXT = text
        _LAST_PORTFOLIO_FETCH_TS = now_ts
        return

    # Build a (date, equity) list, keeping only entries whose date falls on
    # a valid NYSE session. Alpaca's timestamps are UTC epoch seconds; we
    # convert to US/Eastern first so the date matches the trading calendar.
    paired = []
    for ts, eq in zip(timestamps, equities):
        if eq is None:
            continue
        try:
            dt_et = datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(eastern)
            paired.append((dt_et.date(), float(eq)))
        except Exception:
            continue

    if not paired:
        text = "[portfolio] portfolio history had no usable equity marks."
        print(text)
        _CACHED_PORTFOLIO_SUMMARY_TEXT = text
        _LAST_PORTFOLIO_FETCH_TS = now_ts
        return

    # Cross-reference with the NYSE calendar: only keep dates that are real
    # trading sessions. This drops any weekend/holiday equity mark that
    # snuck in due to timezone rounding.
    try:
        nyse = mcal.get_calendar('NYSE')
        earliest = min(d for d, _ in paired)
        latest = max(d for d, _ in paired)
        valid_sessions = set(d.date() for d in nyse.valid_days(
            start_date=earliest, end_date=latest))
    except Exception as e:
        logging.warning(f"[portfolio] NYSE calendar lookup failed: {e}")
        valid_sessions = None

    if valid_sessions is not None:
        paired = [(d, eq) for (d, eq) in paired if d in valid_sessions]

    # Collapse duplicate dates by taking the LAST equity for each date
    # (get_portfolio_history at 1D should already be one-per-day, but
    # belt-and-suspenders after the timezone conversion).
    by_date = {}
    for d, eq in paired:
        by_date[d] = eq
    ordered = sorted(by_date.items())  # [(date, equity)] ascending

    if len(ordered) < 2:
        text = (f"[portfolio] only {len(ordered)} trading-day equity mark(s) so far. "
                f"Need at least 2 to compute a gain %.")
        print(text)
        _CACHED_PORTFOLIO_SUMMARY_TEXT = text
        _LAST_PORTFOLIO_FETCH_TS = now_ts
        return

    latest_date, latest_equity = ordered[-1]

    # Uses module-level GREEN / RED / DIM / RESET / pnl_color helpers.
    # Set env var _NO_COLOR=1 to disable coloring globally.

    def window_result(n_sessions, label):
        """Return formatted line for 'past N trading sessions' comparison.
        Positive gain rendered green, negative red."""
        if len(ordered) <= n_sessions:
            return f"  {label:<10}: {DIM}insufficient history ({len(ordered)-1} session(s) available){RESET}"
        prior_date, prior_equity = ordered[-1 - n_sessions]
        if prior_equity <= 0:
            return f"  {label:<10}: prior equity was zero, cannot compute %"
        pct = (latest_equity - prior_equity) / prior_equity * 100.0
        delta = latest_equity - prior_equity
        color = pnl_color(delta)
        arrow = "▲" if delta >= 0 else "▼"
        sign = "+" if delta >= 0 else ""
        return (f"  {label:<10}: {color}{arrow} {pct:+.2f}%{RESET}  "
                f"{DIM}(${prior_equity:,.2f} on {prior_date} -> "
                f"${latest_equity:,.2f} on {latest_date}, "
                f"{color}{sign}${delta:,.2f}{RESET}{DIM}){RESET}")

    # Render into a single string so future cached prints show the exact
    # same text. Includes a footer noting the freshness timestamp so it's
    # obvious the data is up to an hour old when reprinted from cache.
    fetched_at_et = datetime.now(eastern).strftime("%I:%M:%S %p ET on %m-%d-%Y")
    lines = [
        "",
        "=========================== Portfolio Gain / Loss ===========================",
        f"  Current equity: ${latest_equity:,.2f}  (as of trading session {latest_date})",
        # "Past 24 hours" = previous trading session (1 session back).
        # "Past 7 days"   = 5 trading sessions back (a full trading week).
        # "Past 14 days"  = 10 trading sessions back (two full trading weeks).
        # "Past 30 days"  = 21 trading sessions back (approximate calendar month).
        # Trading sessions (not calendar days) because holidays and
        # weekends have no equity marks -- "1 calendar day ago" on a
        # Monday would land on Sunday and have no data.
        window_result(1,  "24 hours"),
        window_result(5,  "7 days"),
        window_result(10, "14 days"),
        window_result(21, "30 days"),
        f"  {DIM}Fetched at: {fetched_at_et}  (refetches at most 1/hour){RESET}",
        "============================================================================",
        "",
    ]
    text = "\n".join(lines)
    print(text)

    _CACHED_PORTFOLIO_SUMMARY_TEXT = text
    _LAST_PORTFOLIO_FETCH_TS = now_ts


# ============================================================================
# DASHBOARD WEBSOCKET SERVER (adapted from Kalshi's DashboardWSEngine)
# ============================================================================
# Runs a WebSocket server on its own asyncio loop in a daemon thread.
# Broadcasts a state snapshot at 1 Hz to every connected browser client.
# The HTML client is a separate file (alpaca_dashboard.html) served by any
# static-file server or opened directly from disk with the WS_URL set.
#
# Snapshot payload shape (JSON):
#   {
#     "type": "state",
#     "ts": epoch,
#     "account": {"equity": float, "buying_power": float, "cash": float, "exposure": float, "margin_ratio": float},
#     "positions": [{symbol, qty, avg_price, current_price, gain_pct, gain_usd, entry_atr}],
#     "recent_trades": [{symbol, action, qty, price, date}],
#     "regime": {"regime": str, "vix": float, "weak_bull_downgraded": bool},
#     "governor": {"can_trade": bool, "reason": str, "consec_losses": int, "day_locked": bool},
#     "backtest_brain": {"mode": str, "threshold": float, "cached": int},
#     "brain_b": {mode, min_safe, p_safe, model_ready, features, feature_labels},
#     "brain_d": {mode, model_ready, size_multiplier, aggression, concentration_multiplier, features, feature_labels},
#     "brain_f": {mode, model_ready, min_prob_to_bump, max_score_bump, last_symbol, last_p_bullish, last_bump, last_ts, session_bump_count, positive_bumps_session, features, feature_labels},
#     "chandelier": {"mode": str, "atr_mult": float},
#     "muted_symbols": [{symbol, win_rate, pnl, n}],
#     "brain_trust": {brain_name: {trust, hot, cold, n_predictions}},
#     "thinking_log": [{ts, side, symbol, prob, verdict, adj}]   # ring buffer of recent brain-thinking lines
#   }
#
# Client -> server command messages:
#   {"type": "emergency_stop"}   -- flip a global flag that both worker loops check
#   {"type": "robot_pause"}      -- pause new-entry decisions (positions still exit normally)
#   {"type": "robot_resume"}     -- clear pause
#   {"type": "sell_all"}         -- signals sell_stocks to close all positions at market next cycle
#
# All destructive commands require the browser to send a confirmation flag
# (checked HTML-side); the server just executes what it receives, so the
# HTML is the safety layer for accidental clicks.

DASHBOARD_WS_ENABLED = True
DASHBOARD_WS_HOST = '127.0.0.1'      # bind to localhost only by default; change to '0.0.0.0' to expose
DASHBOARD_WS_PORT = 8765             # SAME PORT AS KALSHI dashboard (do not run both bots simultaneously)
DASHBOARD_BROADCAST_INTERVAL_SECS = 1.0
DASHBOARD_THINKING_LOG_MAX = 200     # ring-buffer size

# Global command flags the trading loops honor. These are set by dashboard
# clients; the loops poll them once per cycle.
_dashboard_emergency_stop = False
_dashboard_paused = False
_dashboard_sell_all_requested = False
_dashboard_thinking_log = deque(maxlen=DASHBOARD_THINKING_LOG_MAX)
_dashboard_thinking_lock = threading.Lock()


def dashboard_record_thinking(side, symbol, prob, verdict, adj=None):
    """Called from _ml_print_thinking (patch point) so the dashboard's
    thinking log mirrors what's printed to the terminal. Safe no-op if the
    dashboard isn't running."""
    with _dashboard_thinking_lock:
        _dashboard_thinking_log.append({
            'ts': time.time(),
            'side': side,
            'symbol': symbol,
            'prob': float(prob) if prob is not None else None,
            'verdict': verdict,
            'adj': float(adj) if adj is not None else None,
        })


def is_dashboard_paused():
    return _dashboard_paused


def is_dashboard_emergency_stopped():
    return _dashboard_emergency_stop


def consume_dashboard_sell_all():
    """Called by sell_stocks. Returns True exactly once when the flag is
    set, then clears it. Prevents the request from firing twice."""
    global _dashboard_sell_all_requested
    if _dashboard_sell_all_requested:
        _dashboard_sell_all_requested = False
        return True
    return False


class DashboardWSEngine:
    """WebSocket dashboard server. Runs on its own asyncio loop in a
    daemon thread; the trading loops never touch asyncio."""
    def __init__(self, host=DASHBOARD_WS_HOST, port=DASHBOARD_WS_PORT):
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self._loop = None
        self._thread = None
        self._clients = set()
        self._websockets_mod = None

    def start(self):
        try:
            import websockets  # noqa
            self._websockets_mod = websockets
        except ImportError:
            logging.warning("[dashboard] `websockets` package not installed — "
                            "dashboard disabled. `pip install websockets` to enable.")
            print("[dashboard] websockets package not installed — dashboard disabled.")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True,
                                        name='DashboardWSEngine')
        self._thread.start()
        print(f"[dashboard] starting WebSocket server on ws://{self.host}:{self.port}")
        logging.info(f"[dashboard] starting on ws://{self.host}:{self.port}")

    def stop(self):
        self._stop.set()

    def _thread_main(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception as e:
            logging.error(f"[dashboard] loop crashed: {e}", exc_info=True)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _serve(self):
        import asyncio
        websockets = self._websockets_mod

        async def handler(ws):
            await self._handle_client(ws)

        async with websockets.serve(handler, self.host, self.port,
                                     ping_interval=20, ping_timeout=10,
                                     max_size=256 * 1024):
            logging.info(f"[dashboard] listening on ws://{self.host}:{self.port}")
            while not self._stop.is_set():
                try:
                    await self._broadcast(self._build_state())
                except Exception as e:
                    logging.debug(f"[dashboard] broadcast: {e}")
                await asyncio.sleep(DASHBOARD_BROADCAST_INTERVAL_SECS)

    async def _handle_client(self, ws):
        import json
        self._clients.add(ws)
        addr = getattr(ws, 'remote_address', ('?', '?'))
        logging.info(f"[dashboard] client connected: {addr} "
                     f"({len(self._clients)} total)")
        try:
            # Send an immediate snapshot on connect so the UI populates fast.
            await ws.send(json.dumps(self._build_state(), default=str))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                self._handle_message(msg)
        except Exception as e:
            logging.debug(f"[dashboard] client closed: {e}")
        finally:
            self._clients.discard(ws)
            logging.info(f"[dashboard] client gone: {addr}")

    async def _broadcast(self, payload):
        if not self._clients:
            return
        import json
        data = json.dumps(payload, default=str)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def _handle_message(self, msg):
        """Client -> server command handler. Runs on the asyncio thread, so
        setting module-level globals is safe (Python GIL makes assignment
        to a single name atomic)."""
        global _dashboard_emergency_stop, _dashboard_paused, _dashboard_sell_all_requested
        cmd = (msg.get('type') or '').lower()
        if cmd == 'emergency_stop':
            _dashboard_emergency_stop = True
            _dashboard_paused = True
            _dashboard_sell_all_requested = True
            logging.warning("[dashboard] EMERGENCY STOP received — pausing + sell-all requested.")
            print("[dashboard] EMERGENCY STOP received.")
        elif cmd == 'robot_pause':
            _dashboard_paused = True
            logging.info("[dashboard] pause requested — no new entries.")
        elif cmd == 'robot_resume':
            _dashboard_paused = False
            _dashboard_emergency_stop = False
            logging.info("[dashboard] resume requested — entries re-enabled.")
        elif cmd == 'sell_all':
            _dashboard_sell_all_requested = True
            logging.warning("[dashboard] sell-all requested — will fire next sell cycle.")

        # ---- Blacklist commands ----
        elif cmd == 'blacklist_add_perm':
            sym = (msg.get('symbol') or '').upper()
            if sym:
                BLACKLIST.add_permanent(sym)
                floor_post('USER', 'instruction', f"permanent blacklist added: {sym}")
        elif cmd == 'blacklist_add_temp':
            sym = (msg.get('symbol') or '').upper()
            hours = float(msg.get('hours') or 72)
            if sym:
                BLACKLIST.add_temporary(sym, duration_secs=hours * 3600)
                floor_post('USER', 'instruction',
                           f"{hours:.0f}h blacklist added: {sym}")
        elif cmd == 'blacklist_remove':
            sym = (msg.get('symbol') or '').upper()
            if sym:
                BLACKLIST.remove(sym)
                floor_post('USER', 'instruction', f"blacklist removed: {sym}")
        elif cmd == 'blacklist_clear_temp':
            with BLACKLIST._lock:
                BLACKLIST.temporary.clear()
                BLACKLIST._save()
            floor_post('USER', 'instruction', "all 72h auto-blacklist entries cleared")

        # ---- Brain Trading Floor commands ----
        elif cmd == 'trading_floor_msg':
            text = (msg.get('text') or '').strip()
            if text:
                self._handle_floor_command(text)
        elif cmd == 'floor_clear':
            floor_clear()
            floor_post('USER', 'instruction', 'floor cleared')

        # ---- Settings updates ----
        elif cmd == 'set_setting':
            self._apply_setting(msg.get('key'), msg.get('value'))

        else:
            logging.debug(f"[dashboard] ignoring unknown command: {cmd}")

    def _handle_floor_command(self, text):
        """User-typed commands from the Brain Trading Floor input box.
        Very small command grammar; mostly a way to inject blacklist ops
        and sanity queries without leaving the dashboard.
        """
        floor_post('USER', 'instruction', f"$ {text}")
        parts = text.split()
        head = parts[0].lower() if parts else ''
        if head in ('help', '-h', '?'):
            floor_post('BOT', 'info',
                       "commands: status | pause | resume | sell_all | "
                       "blacklist <SYM> [hours|perm] | unblacklist <SYM> | clear")
            return
        if head == 'status':
            can_go, reason = GOVERNOR.can_trade()
            floor_post('BOT', 'info',
                       f"paused={_dashboard_paused} emergency={_dashboard_emergency_stop} "
                       f"can_trade={can_go} reason='{reason}' "
                       f"consec_losses={GOVERNOR.consec_losses}")
            return
        if head == 'pause':
            self._handle_message({'type': 'robot_pause'}); return
        if head == 'resume':
            self._handle_message({'type': 'robot_resume'}); return
        if head == 'sell_all':
            self._handle_message({'type': 'sell_all'}); return
        if head == 'clear':
            floor_clear(); return
        if head == 'blacklist' and len(parts) >= 2:
            sym = parts[1].upper()
            arg = parts[2].lower() if len(parts) >= 3 else '72h'
            if arg in ('perm', 'permanent'):
                self._handle_message({'type': 'blacklist_add_perm', 'symbol': sym})
            else:
                # Parse "72h", "48h", "24", etc.
                try:
                    hours = float(arg.rstrip('h'))
                except Exception:
                    hours = 72
                self._handle_message({'type': 'blacklist_add_temp',
                                      'symbol': sym, 'hours': hours})
            return
        if head == 'unblacklist' and len(parts) >= 2:
            self._handle_message({'type': 'blacklist_remove', 'symbol': parts[1].upper()})
            return
        floor_post('BOT', 'info', f"unknown command: {head} (try 'help')")

    def _apply_setting(self, key, value):
        """Update a runtime setting from the dashboard. Only mutates specific
        module-level constants explicitly listed here; unknown keys ignored.
        Type-coerces value before assignment; logs the change."""
        global CHANDELIER_MODE, CHANDELIER_ATR_MULT, CHANDELIER_MIN_HOLD_SECS
        global BACKTEST_BRAIN_MODE, BACKTEST_MIN_WIN_RATE, BACKTEST_MIN_SAMPLES
        global CONSEC_LOSS_COOLDOWN, COOLDOWN_SECONDS, DAILY_PROFIT_LOCK_PCT
        global PERSYMBOL_MUTE_WINRATE, PERSYMBOL_MIN_TRADES
        global BRAIN_B_MODE, BRAIN_B_MIN_SAFE
        global BRAIN_D_MODE
        global BRAIN_F_MODE, BRAIN_F_MIN_PROB_TO_BUMP, BRAIN_F_MAX_SCORE_BUMP
        try:
            if key == 'chandelier_mode' and value in ('armed', 'shadow', 'off'):
                CHANDELIER_MODE = value
            elif key == 'chandelier_atr_mult':
                CHANDELIER_ATR_MULT = float(value)
            elif key == 'chandelier_min_hold_secs':
                CHANDELIER_MIN_HOLD_SECS = int(value)
            elif key == 'backtest_brain_mode' and value in ('armed', 'advisory'):
                BACKTEST_BRAIN_MODE = value
            elif key == 'backtest_min_win_rate':
                BACKTEST_MIN_WIN_RATE = float(value)
            elif key == 'backtest_min_samples':
                BACKTEST_MIN_SAMPLES = int(value)
            elif key == 'brain_b_mode' and value in ('armed', 'shadow'):
                BRAIN_B_MODE = value
            elif key == 'brain_b_min_safe':
                BRAIN_B_MIN_SAFE = float(value)
            elif key == 'brain_d_mode' and value in ('armed', 'shadow'):
                BRAIN_D_MODE = value
            elif key == 'brain_f_mode' and value in ('armed', 'shadow', 'off'):
                BRAIN_F_MODE = value
            elif key == 'brain_f_min_prob_to_bump':
                BRAIN_F_MIN_PROB_TO_BUMP = float(value)
            elif key == 'brain_f_max_score_bump':
                BRAIN_F_MAX_SCORE_BUMP = float(value)
            elif key == 'consec_loss_cooldown':
                CONSEC_LOSS_COOLDOWN = int(value)
            elif key == 'cooldown_seconds':
                COOLDOWN_SECONDS = int(value)
            elif key == 'daily_profit_lock_pct':
                DAILY_PROFIT_LOCK_PCT = float(value)
            elif key == 'persymbol_mute_winrate':
                PERSYMBOL_MUTE_WINRATE = float(value)
            elif key == 'persymbol_min_trades':
                PERSYMBOL_MIN_TRADES = int(value)
            else:
                floor_post('BOT', 'info', f"unknown or unchangeable setting: {key}")
                return
            floor_post('USER', 'instruction', f"setting updated: {key} = {value}")
            logging.info(f"[dashboard] setting {key} -> {value}")
        except Exception as e:
            floor_post('BOT', 'alert', f"failed to set {key}: {e}")

    def _build_state(self):
        """Assemble the snapshot payload. Every accessor is wrapped in
        try/except because the dashboard must never crash the bot."""
        state = {'type': 'state', 'ts': time.time()}

        # Account
        try:
            st = get_margin_state()
            state['account'] = {
                'equity': float(st.get('equity') or 0),
                'buying_power': float(st.get('buying_power') or 0),
                'cash': float(st.get('cash') or 0),
                'exposure': float(st.get('long_market_value') or 0),
                'margin_ratio': float(st.get('margin_ratio') or 0),
                'effective_bp': float(st.get('effective_bp') or 0),
                'healthy': bool(st.get('healthy', True)),
            }
        except Exception as e:
            state['account'] = {'error': str(e)}

        # Positions with live P&L
        positions = []
        try:
            for sym, (avg_price, purchase_date) in position_book.snapshot().items():
                try:
                    px = get_current_price(sym)
                except Exception:
                    px = None
                entry_atr = None
                try:
                    row = Session().query(Position).filter_by(symbols=sym).one_or_none()
                    if row:
                        entry_atr = row.entry_atr
                except Exception:
                    pass
                gain_pct = ((px - avg_price) / avg_price) if px and avg_price else 0.0
                # Compute chandelier stop for display
                ch_stop = None
                if entry_atr and avg_price:
                    peak = _get_peak_since_entry(sym, avg_price, px or avg_price)
                    ch_stop = _chandelier_stop_price(avg_price, entry_atr, peak)
                positions.append({
                    'symbol': sym,
                    'avg_price': float(avg_price) if avg_price else 0.0,
                    'current_price': float(px) if px else 0.0,
                    'gain_pct': float(gain_pct),
                    'purchase_date': str(purchase_date) if purchase_date else '',
                    'entry_atr': float(entry_atr) if entry_atr else None,
                    'chandelier_stop': float(ch_stop) if ch_stop else None,
                })
        except Exception as e:
            positions = [{'error': str(e)}]
        state['positions'] = positions

        # Recent trades (last 25)
        try:
            sess = Session()
            rows = (sess.query(TradeHistory)
                    .order_by(TradeHistory.id.desc())
                    .limit(25).all())
            state['recent_trades'] = [{
                'symbol': r.symbols,
                'action': r.action,
                'qty': float(r.quantity or 0),
                'price': float(r.price or 0),
                'date': r.date or '',
            } for r in rows]
        except Exception as e:
            state['recent_trades'] = [{'error': str(e)}]

        # Regime
        try:
            reg = get_market_regime()
            state['regime'] = {
                'regime': reg.get('regime', 'unknown'),
                'vix': float(reg.get('vix')) if reg.get('vix') is not None else None,
                'weak_bull_downgraded': bool(reg.get('weak_bull_downgraded', False)),
            }
        except Exception as e:
            state['regime'] = {'error': str(e)}

        # Governor
        try:
            can_go, reason = GOVERNOR.can_trade(current_equity=state.get('account', {}).get('equity'))
            state['governor'] = {
                'can_trade': can_go,
                'reason': reason,
                'consec_losses': GOVERNOR.consec_losses,
                'day_locked': GOVERNOR.day_locked,
                'in_cooldown': GOVERNOR.in_cooldown(),
            }
        except Exception as e:
            state['governor'] = {'error': str(e)}

        # Backtest brain status
        try:
            with _backtest_cache_lock:
                bt_cached = len(_backtest_cache)
            state['backtest_brain'] = {
                'mode': BACKTEST_BRAIN_MODE,
                'threshold': BACKTEST_MIN_WIN_RATE,
                'cached_symbols': bt_cached,
            }
        except Exception as e:
            state['backtest_brain'] = {'error': str(e)}

        # Chandelier status
        state['chandelier'] = {
            'mode': CHANDELIER_MODE,
            'atr_mult': CHANDELIER_ATR_MULT,
            'min_hold_secs': CHANDELIER_MIN_HOLD_SECS,
        }

        # Brain B (risk brain) status
        try:
            b = brain_b_evaluate(margin_state=state.get('account'),
                                  position_snapshots=state.get('positions'))
            state['brain_b'] = {
                'mode': BRAIN_B_MODE,
                'min_safe': BRAIN_B_MIN_SAFE,
                'p_safe': b['p_safe'] if b else None,
                'model_ready': os.path.exists(BRAIN_B_MODEL_PATH),
                'features': b['features'] if b else None,
                'feature_labels': BRAIN_B_FEATURE_LABELS,
            }
        except Exception as e:
            state['brain_b'] = {'error': str(e)}

        # Brain D (portfolio manager) status
        try:
            d = brain_d_evaluate()
            state['brain_d'] = {
                'mode': BRAIN_D_MODE,
                'model_ready': os.path.exists(BRAIN_D_MODEL_PATH),
                'size_multiplier': d['size_multiplier'] if d else None,
                'aggression': d['aggression'] if d else None,
                'concentration_multiplier': d['concentration_multiplier'] if d else None,
                'features': d['features'] if d else None,
                'feature_labels': BRAIN_D_FEATURE_LABELS,
            }
        except Exception as e:
            state['brain_d'] = {'error': str(e)}

        # Brain F (bullish-trend picker) status. Unlike Brain B/D (portfolio-
        # level, computed per-cycle), Brain F fires per-candidate-symbol
        # inside buy_stocks -- so the tile shows the LAST scored symbol +
        # session counters, not a fresh eval here.
        try:
            tr = _brain_f_tracker
            state['brain_f'] = {
                'mode': BRAIN_F_MODE,
                'model_ready': os.path.exists(BRAIN_F_MODEL_PATH),
                'min_prob_to_bump': BRAIN_F_MIN_PROB_TO_BUMP,   # HTML key
                'max_score_bump': BRAIN_F_MAX_SCORE_BUMP,       # HTML key
                'last_symbol': tr.get('last_symbol'),
                'last_p_bullish': tr.get('last_p_bullish'),
                'last_bump': tr.get('last_bump'),
                'last_ts': tr.get('last_ts'),
                'session_bump_count': tr.get('bumps_session', 0),          # HTML key
                'positive_bumps_session': tr.get('positive_bumps_session', 0),
                'features': tr.get('last_features'),
                'feature_labels': BRAIN_F_FEATURE_LABELS,
            }
        except Exception as e:
            state['brain_f'] = {'error': str(e)}

        # Muted symbols
        try:
            snap = PERSYMBOL.snapshot()
            muted = []
            for sym, s in snap.items():
                if s.get('n', 0) >= PERSYMBOL_MIN_TRADES and s.get('pnl', 0) < 0:
                    wr = s['wins'] / s['n'] if s['n'] else 0
                    if wr < PERSYMBOL_MUTE_WINRATE:
                        muted.append({
                            'symbol': sym, 'win_rate': wr,
                            'pnl': float(s['pnl']), 'n': int(s['n']),
                        })
            state['muted_symbols'] = muted
        except Exception as e:
            state['muted_symbols'] = [{'error': str(e)}]

        # Brain trust
        try:
            state['brain_trust'] = BRAIN_TRUST.snapshot()
        except Exception as e:
            state['brain_trust'] = {'error': str(e)}

        # Thinking log (recent brain reasoning lines)
        try:
            with _dashboard_thinking_lock:
                state['thinking_log'] = list(_dashboard_thinking_log)[-50:]
        except Exception:
            state['thinking_log'] = []

        # Command flags (so the UI can reflect state)
        state['flags'] = {
            'paused': _dashboard_paused,
            'emergency_stop': _dashboard_emergency_stop,
            'sell_all_pending': _dashboard_sell_all_requested,
        }

        # Blacklist (permanent + 72h auto)
        try:
            state['blacklist'] = BLACKLIST.snapshot()
        except Exception as e:
            state['blacklist'] = {'error': str(e)}

        # Brain Trading Floor feed
        try:
            state['floor'] = floor_snapshot(max_msgs=100)
        except Exception as e:
            state['floor'] = {'error': str(e)}

        # Profit Monitor state (peaks + armed/last-seen per symbol)
        try:
            pm_state = {}
            if hasattr(profit_monitor, '_state'):
                for sym, s in profit_monitor._state.items():
                    pm_state[sym] = {
                        'peak_price': float(s.get('peak_price') or 0),
                        'armed': bool(s.get('armed', False)),
                        'armed_at': float(s.get('armed_at') or 0),
                        'last_seen': float(s.get('last_seen') or 0),
                    }
            state['profit_monitor'] = pm_state
        except Exception as e:
            state['profit_monitor'] = {'error': str(e)}

        # Settings (dashboard-editable knobs)
        state['settings'] = {
            'chandelier_mode': CHANDELIER_MODE,
            'chandelier_atr_mult': CHANDELIER_ATR_MULT,
            'chandelier_min_hold_secs': CHANDELIER_MIN_HOLD_SECS,
            'backtest_brain_mode': BACKTEST_BRAIN_MODE,
            'backtest_min_win_rate': BACKTEST_MIN_WIN_RATE,
            'backtest_min_samples': BACKTEST_MIN_SAMPLES,
            'brain_b_mode': BRAIN_B_MODE,
            'brain_b_min_safe': BRAIN_B_MIN_SAFE,
            'brain_d_mode': BRAIN_D_MODE,
            'brain_f_mode': BRAIN_F_MODE,
            'brain_f_min_prob_to_bump': BRAIN_F_MIN_PROB_TO_BUMP,
            'brain_f_max_score_bump': BRAIN_F_MAX_SCORE_BUMP,
            'consec_loss_cooldown': CONSEC_LOSS_COOLDOWN,
            'cooldown_seconds': COOLDOWN_SECONDS,
            'daily_profit_lock_pct': DAILY_PROFIT_LOCK_PCT,
            'persymbol_mute_winrate': PERSYMBOL_MUTE_WINRATE,
            'persymbol_min_trades': PERSYMBOL_MIN_TRADES,
        }

        return state


# Global instance; started from main() if DASHBOARD_WS_ENABLED
dashboard_ws = DashboardWSEngine()


def main():
    global symbols_to_buy
    print("Starting main trading program...")

    # Start the WebSocket dashboard server early so the browser can see boot.
    # No-op if `websockets` isn't installed.
    if DASHBOARD_WS_ENABLED:
        try:
            dashboard_ws.start()
        except Exception as e:
            logging.warning(f"[dashboard] failed to start: {e}")

    # Ensure the tensorflow package variant matches this machine's hardware
    # BEFORE anything expensive runs. If _ml_ensure_tf_variant has to swap
    # tensorflow <-> tensorflow-cpu, it will pip-install the new wheel and
    # then os.execv() this same Python process so the freshly installed
    # native binary is loaded from a clean interpreter. Doing this here --
    # before get_symbols_to_buy() (a ~9-minute S&P 500 scan) and before
    # reconcile_positions_on_startup() -- means a first-launch install only
    # pays the scanner cost ONCE. If we deferred the swap until the first
    # ML training call, the restart would throw away all that work.
    if USE_ML_BRAIN_ADJUSTMENT:
        try:
            _ml_ensure_tf_variant()
        except Exception as e:
            logging.warning(f"ml_brain: early tensorflow variant check raised: {e}")

    # Brain B (risk brain) first-run pretraining. Cheap (~seconds on any CPU)
    # since it's 20k synthetic snapshots on a tiny network. No-op if the
    # model already exists on disk from a prior run.
    try:
        brain_b_pretrain_if_needed()
    except Exception as e:
        logging.warning(f"brain_b: startup pretrain check raised: {e}")

    # Brain D (portfolio manager) first-run pretraining. Same shape as
    # Brain B — 20k synthetic trajectories, seconds on CPU. No-op if
    # model already exists on disk.
    try:
        brain_d_pretrain_if_needed()
    except Exception as e:
        logging.warning(f"brain_d: startup pretrain check raised: {e}")

    # Brain F (bullish-trend picker) first-run pretraining happens LATER,
    # AFTER the ML brain first-run pretraining completes below. See the
    # Brain F block after run_ml_first_training_if_needed() for rationale
    # and the actual call.

    symbols_to_buy = get_symbols_to_buy()

    # Resume auto-adjusted parameters from the last run instead of resetting
    # to coded defaults -- the guardrails (min sample, step cap, bounds) still
    # apply to any further adjustment from here.
    adaptive_params.load_from_db()

    # BUGFIX: was `load_positions_from_database()`, which trusted the stale .db
    # on restart. Alpaca is authoritative; reconcile before touching anything.
    position_book.replace_all(reconcile_positions_on_startup())

    # First-run ML brain pretraining, kicked off HERE (before the trading
    # loop's stop_if_stock_market_is_closed() call) so the training happens
    # immediately on startup regardless of time of day. Otherwise, starting
    # the bot at e.g. 2am/4am/6am would leave it stuck in the market-closed
    # wait loop for hours before ever reaching maybe_run_scheduled_ml_training().
    #
    # This is a no-op (returns None quickly) when:
    #   - a model already exists on disk from a prior run, OR
    #   - the first_run_completed flag is already set in schedule_state.json, OR
    #   - TensorFlow isn't installed (ML feature disabled overall), OR
    #   - the user turned USE_ML_BRAIN_ADJUSTMENT off.
    # Only actually runs the (slow, minutes-long) pretraining pass on a truly
    # fresh install with no cached model, which is exactly when we want it.
    if USE_ML_BRAIN_ADJUSTMENT:
        try:
            first_status = run_ml_first_training_if_needed()
            if first_status is not None:
                print(f"ML brain first-run pretraining: {first_status}")
                logging.info(f"ML brain first-run pretraining: {first_status}")
        except Exception as e:
            # Never let a training-code exception block the bot from starting
            # to trade -- the ML adjustment is optional, trading is not.
            logging.warning(f"ML brain first-run pretraining raised: {e}; continuing without it.")

    # Brain F (bullish-trend picker) first-run pretraining. Sequenced AFTER
    # the ML brain first-run pretraining so both TF-heavy jobs don't compete
    # for the same CPU/GPU at once and so the operator sees ML brain progress
    # first (its output is more familiar). Same shape as Brain B/D: 20k
    # synthetic examples, seconds on CPU. Feeds a small bounded bump into
    # buy_stocks scoring. No-op if a valid (correct feature count) model
    # already exists on disk; auto-detects and rebuilds a stale one.
    try:
        brain_f_pretrain_if_needed()
    except Exception as e:
        logging.warning(f"brain_f: startup pretrain check raised: {e}")

    # BUGFIX: main() created a fresh `lock = threading.Lock()` while the
    # module-level buy_sell_lock sat unused, which was confusing and made it easy
    # to reintroduce a second, non-shared mutex. Use the one module-level lock.
    lock = buy_sell_lock
    cycle_count = 0

    # ---- First-run ML brain pretraining (fires immediately at startup) ----
    # Kicked off here BEFORE stop_if_stock_market_is_closed() so it happens
    # regardless of the time of day the bot is started. The old wiring only
    # reached the first-run call from inside the main trading loop, which
    # meant a bot started at 2am/4am/6am/anytime the market was closed sat
    # in the market-closed wait loop for hours before pretraining could even
    # begin. Now the check runs unconditionally at startup:
    #   - No existing model on disk  -> pretrains 2,500 examples now, blocking
    #     for a few minutes on first launch only. The trader gets a working
    #     ML brain by the time the market opens, no matter what hour they
    #     started the program.
    #   - Model already exists       -> returns immediately (cheap file check).
    #   - TensorFlow not available   -> returns immediately with a printed note.
    # The scheduled 17:00 ET daily training call inside the loop below still
    # handles all subsequent runs on the normal schedule.
    if USE_ML_BRAIN_ADJUSTMENT and _ml_brain_is_available():
        try:
            startup_ml_status = run_ml_first_training_if_needed()
            if startup_ml_status is not None:
                print(f"ML brain first-run at startup: {startup_ml_status}")
                logging.info(f"ML brain first-run at startup: {startup_ml_status}")
        except Exception as e:
            logging.warning(f"ML brain first-run at startup failed: {e}")

    while True:
        try:
            stop_if_stock_market_is_closed()
            cycle_count += 1
            now_str = datetime.now(eastern).strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")
            st = get_margin_state()

            print("------------------------------------------------------------------------------------\n")
            print("*****************************************************")
            print("******** Billionaire Buying Strategy Version ********")
            print("*****************************************************")
            print("2026 Edition of the Billionaire Strategy Stock Market Trading Robot, Version 10")
            print("by https://github.com/CodeProSpecialist")
            print("------------------------------------------------------------------------------------")
            print(f" {now_str} Cash Balance: ${st['cash']:,.2f}")
            print(f" Equity: ${st['equity']:,.2f} | Buying Power: ${st['buying_power']:,.2f} | "
                  f"Effective BP (leverage cap {MAX_LEVERAGE:.1f}x): ${st['effective_bp']:,.2f}")
            print(f" Day-trading BP: ${st['daytrading_buying_power']:,.2f} | Reg-T BP: ${st['regt_buying_power']:,.2f}")
            print(f" Margin health (equity/long_mv): {st['margin_ratio']:.2f} "
                  f"(floor {MAINTENANCE_MARGIN_FLOOR_PCT:.2f}) -> "
                  f"{GREEN + 'OK' + RESET if st['healthy'] else RED + 'BREACHED' + RESET}")
            print(f" Account mode: {ACCOUNT_MODE} | Day trades: "
                  f"{'UNLIMITED (2026 margin rules - PDT retired)' if UNLIMITED_DAY_TRADES else 'limited'}")
            try:
                rinfo = get_market_regime()
                vix_s = f"{rinfo['vix']:.1f}" if rinfo['vix'] is not None else "n/a"
                print(f" Market regime: {rinfo['regime'].upper()} (VIX {vix_s}) | "
                      f"buy threshold: {get_buy_score_threshold(rinfo['regime'])}")
            except Exception as e:
                logging.warning(f"Regime banner failed: {e}")
            print("------------------------------------------------------------------------------------\n")

            # Kick off (or skip) the once-per-day background scanner refresh
            # that rebuilds SYMBOLS_TO_BUY_LIST in memory. No files involved.
            refresh_symbols_to_buy_list_if_due()

            symbols_to_buy = get_symbols_to_buy()

            # PERF: one batched yf.download() seeds the daily SMA/RSI/ATR cache
            # for every symbol. Without it each symbol costs 3 separate yfinance
            # requests (48 for 16 symbols); batched it is 1. Cheap no-op when the
            # 30m cache TTLs are still warm.
            prewarm_daily_cache(symbols_to_buy)

            # BUGFIX: was `if not symbols_to_sell_dict:` -- the API resync only ran
            # when the dict was EMPTY, so a populated-but-stale view was never
            # corrected. Resync every cycle, in place, before starting threads.
            position_book.replace_all(update_symbols_to_sell_from_api())

            buy_thread = threading.Thread(target=_run_and_release,
                                          args=(buy_stocks, symbols_to_buy, lock),
                                          name='buy')
            sell_thread = threading.Thread(target=_run_and_release,
                                           args=(sell_stocks, lock),
                                           name='sell')
            buy_thread.start()
            sell_thread.start()
            # BUGFIX: bound the join. Without a timeout, a worker wedged on a
            # hung API call would freeze the main loop forever with no output.
            buy_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            sell_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            for t in (buy_thread, sell_thread):
                if t.is_alive():
                    msg = (f"WARNING: {t.name}_stocks thread still running after "
                           f"{THREAD_JOIN_TIMEOUT}s; continuing without it. It holds no "
                           f"lock indefinitely, but check for a hung API call.")
                    print(f"{RED}{msg}{RESET}")
                    logging.error(msg)

            # Runs its own once-per-day/3:45pm gate internally; safe to call
            # every cycle. Only meaningful during regular market hours, which
            # is exactly when this loop body runs.
            try:
                run_close_profit_sweep()
            except Exception as e:
                logging.error(f"Pre-close profit sweep raised: {e}")

            if PRINT_SYMBOLS_TO_BUY:
                print("\nSymbols to Purchase:\n")
                # BUGFIX: original shadowed the `symbols_to_buy` list with the
                # loop variable, destroying the list after the first pass.
                for sym in symbols_to_buy:
                    cp = get_current_price(sym)
                    if cp is None:
                        continue
                    prev = get_previous_price(sym) or cp
                    print(f"Symbol: {sym} | Current Price: {GREEN if cp > prev else RED}${cp:.2f}{RESET}")
                print("")

            if PRINT_ROBOT_STORED_BUY_AND_SELL_LIST_DATABASE:
                print_database_tables()

            if DEBUG:
                print("\nSymbols to Purchase:\n")
                for sym in symbols_to_buy:
                    cp = get_current_price(sym)
                    lo = get_atr_low_price(sym)
                    if cp is None:
                        continue
                    prev = get_previous_price(sym) or cp
                    lo_s = f"${lo:.2f}" if lo else "n/a"
                    print(f"Symbol: {sym} | Current: {GREEN if cp > prev else RED}${cp:.2f}{RESET} | ATR low: {lo_s}")
                print("\nSymbols to Sell:\n")
                for sym in sorted(position_book.symbols()):
                    cp = get_current_price(sym)
                    hi = get_atr_high_price(sym)
                    if cp is None:
                        continue
                    prev = get_previous_price(sym) or cp
                    hi_s = f"${hi:.2f}" if hi else "n/a"
                    print(f"Symbol: {sym} | Current: {GREEN if cp > prev else RED}${cp:.2f}{RESET} | ATR high: {hi_s}")
                print("")

            # REVIEW ITEM #7: informational diagnostics -- prints findings,
            # does not touch live parameters. Separate from the auto-adjuster.
            if cycle_count % ANALYZE_TRADE_HISTORY_EVERY_N_CYCLES == 0:
                try:
                    analyze_trade_history()
                except Exception as e:
                    logging.warning(f"Trade history analysis failed: {e}")
                try:
                    report_expectancy()
                except Exception as e:
                    logging.warning(f"Expectancy report failed: {e}")
                try:
                    compare_parameter_expectancy()
                except Exception as e:
                    logging.warning(f"Parameter-expectancy comparison failed: {e}")

            # REVIEW ITEM #10 (auto-applying, per your instruction): bounded,
            # point-based parameter adjustment. See AdaptiveParams for the
            # guardrails (min sample size, max step size, hard bounds, full
            # audit log) that keep "auto-applies" from meaning "unbounded".
            if cycle_count % ADAPT_EVERY_N_CYCLES == 0:
                try:
                    run_adaptive_parameter_pass()
                except Exception as e:
                    logging.warning(f"Adaptive parameter pass failed: {e}")

            # ML brain: unified scheduled training entry point.
            # Handles internally:
            #   - First-run bootstrap: 2,500-example pretrain if no model exists yet
            #   - Daily 17:00 ET window: 15,000-example historical pretrain (until
            #     cumulative pretraining hits the 20,000 lifetime cap)
            #   - Post-cap daily maintenance: fine-tune on the last day's live
            #     win/loss outcomes only
            # Once-per-window gating + lifetime cap tracking are handled by the
            # scheduling code itself. Cheap to call every cycle -- most calls are
            # a small file check + a JSON read + return None.
            if USE_ML_BRAIN_ADJUSTMENT and _ml_brain_is_available():
                try:
                    ml_status = maybe_run_scheduled_ml_training()
                    if ml_status:
                        print(f"ML brain training: {ml_status}")
                        logging.info(f"ML brain training: {ml_status}")
                except Exception as e:
                    logging.warning(f"ML brain scheduled training check failed: {e}")

            # Weekly backtest -- fires Sunday 22:00-22:30 ET, once per week.
            # Uses the SAME live constants the bot trades with (HARD_STOP,
            # ARM, GIVEBACK, SCALE_OUT_STAGES, etc.) so results reflect the
            # actual live rules -- no drift possible between backtest and
            # live behavior. Cheap on non-eligible calls (weekday/hour check
            # then early return). See BACKTEST_ENABLED to disable entirely.
            try:
                bt_status = maybe_run_scheduled_backtest()
                if bt_status:
                    print(f"Backtest: {bt_status}")
                    logging.info(f"Backtest: {bt_status}")
            except Exception as e:
                logging.warning(f"Scheduled backtest check failed: {e}")

            print("Waiting 1 minute before checking price data again........")
            # Prints every 60s cycle. Portfolio history is refetched from
            # Alpaca at most once per hour; in between, the cached rendering
            # is reprinted so the section still appears every cycle.
            try:
                print_portfolio_gain_summary()
            except Exception as e:
                logging.warning(f"[portfolio] summary error in main loop: {e}")
            time.sleep(60)

        except Exception as e:
            logging.error(f"Error encountered: {e}")
            print(f"Error encountered in main loop: {e}")
            time.sleep(120)


if __name__ == '__main__':
    try:
        print("Initializing trading bot...")
        main()
    except KeyboardInterrupt:
        print("\nShutting down.")
    except Exception as e:
        logging.error(f"Error encountered: {e}")
        print(f"Critical error: {e}")
    finally:
        Session.remove()
