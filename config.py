"""
Global configuration for the Regime Volatility Arbitrage Engine.
"""

import os

# ── Data API ─────────────────────────────────────────────────────────
POLYGON_API_KEY       = os.getenv("MASSIVE_API_KEY", "o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf")

# ── Pricing Engine ──────────────────────────────────────────────────
HURST_EXPONENT        = 0.07      # Literature consensus for SPX: H ≈ 0.07
V0                    = 0.04      # Initial spot variance — overridden at runtime
                                  # with market_iv² so model is always calibrated
LAMBDA_VOL_OF_VOL     = 0.3       # Vol-of-vol scaling factor
MEAN_REVERSION_SPEED  = 0.5       # λ(v) mean-reversion speed
LONG_RUN_VARIANCE     = 0.04      # θ in λ(v) = ν * sqrt(max(v, 0))
MC_PATHS              = 10_000    # Monte Carlo paths (production)
MC_STEPS_PER_DAY      = 24        # Time steps per trading day
SPOT_PRICE            = 585.0     # Current SPY price (updated at runtime)
RISK_FREE_RATE        = 0.053     # Fed funds rate

# ── Regime Filter ────────────────────────────────────────────────────
HMM_N_STATES          = 2         # "Calm" and "Turbulent"
HMM_TICKER            = "SPY"
HMM_HISTORY_YEARS     = 5
TURBULENCE_THRESHOLD  = 0.6       # P(Turbulent) above this → halt / hedge

# ── Strategy Entry / Exit Rules ──────────────────────────────────────
# Entry: requires ALL three conditions simultaneously:
#   1. HMM regime = Calm
#   2. Market IV >= MIN_MKT_IV  (VIX gate — avoids low-vol theta bleed)
#   3. model_IV - market_IV >= IV_ENTRY_THRESHOLD  (roughness spread)
IV_ENTRY_THRESHOLD    = 0.06      # 6 vol pts — roughness premium must clear 6%
                                  # (was 2%; raised because rough-vol premium is
                                  # always ~8-9%, so 2% never filtered anything)
IV_EXIT_THRESHOLD     = 0.005     # 0.5 vol pts — close when spread collapses
MIN_MKT_IV            = 0.18      # 18% — only trade when VIX ≥ 18
                                  # (low VIX = low realized vol = theta bleed)
MAX_HOLD_DAYS         = 5.0       # Hold full 5-day straddle life (was 3)
                                  # Cutting at 3 days left 2 days of gamma on table
MAX_LOSS_PCT          = 0.50      # Stop-loss at 50% of premium paid

# ── IBKR Connection ──────────────────────────────────────────────────
TWS_HOST              = "127.0.0.1"
TWS_PAPER_PORT        = 7497
TWS_LIVE_PORT         = 7496
TWS_CLIENT_ID         = 1
RECONNECT_DELAY_SEC   = 5
RECONNECT_MAX_RETRIES = 10

# ── Storage ──────────────────────────────────────────────────────────
HDF5_TICK_STORE       = "tick_data.h5"
