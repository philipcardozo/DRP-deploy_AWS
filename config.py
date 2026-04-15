"""
Global configuration for the Regime Volatility Arbitrage Engine.
"""

# ── Pricing Engine ──────────────────────────────────────────────────
HURST_EXPONENT      = 0.07        # Literature consensus for SPX: H ≈ 0.07
V0                  = 0.04        # Initial spot variance (σ² = 0.04 → σ = 20%)
LAMBDA_VOL_OF_VOL   = 0.3         # Vol-of-vol scaling factor
MEAN_REVERSION_SPEED = 0.5        # λ(v) mean-reversion speed
LONG_RUN_VARIANCE    = 0.04       # θ in λ(v) = ν * sqrt(max(v, 0))
MC_PATHS             = 10_000     # Number of Monte Carlo paths
MC_STEPS_PER_DAY     = 24         # Time steps per trading day
SPOT_PRICE           = 585.0      # Current SPY price
RISK_FREE_RATE       = 0.053      # Fed funds rate

# ── Regime Filter ───────────────────────────────────────────────────
HMM_N_STATES         = 2          # "Calm" and "Turbulent"
HMM_TICKER           = "SPY"
HMM_HISTORY_YEARS    = 5
TURBULENCE_THRESHOLD  = 0.6       # P(Turbulent) above this → halt / hedge

# ── IBKR Connection ────────────────────────────────────────────────
TWS_HOST              = "127.0.0.1"
TWS_PAPER_PORT        = 7497
TWS_LIVE_PORT         = 7496
TWS_CLIENT_ID         = 1
RECONNECT_DELAY_SEC   = 5
RECONNECT_MAX_RETRIES = 10

# ── Storage ─────────────────────────────────────────────────────────
HDF5_TICK_STORE       = "tick_data.h5"
