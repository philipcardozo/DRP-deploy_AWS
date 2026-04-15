# Regime Volatility Arbitrage Engine — Deployment Guide

**Project:** Systematic volatility arbitrage in short-dated equity-index options (SPY)  
**Authors:** Felipe Cardozo & Mitchell Scott, Ph.D. (Emory University)  
**Last Updated:** 2026-04-15

---

## Quick Summary

This is a **production-ready volatility arbitrage strategy** that:
- Uses **rough volatility pricing** (Volterra process, Hurst exponent H≈0.07) to identify mispriced options
- Applies a **2-state Gaussian HMM** to detect market regimes (Calm/Turbulent)
- Executes **ATM straddle trades** on Interactive Brokers (IBKR) via TWS
- Logs all ticks to HDF5, trades to CSV, and health metrics to a heartbeat log
- Achieves ~7.3% annualized return with 1.38 Sharpe ratio in backtests (regime-adjusted)

**Status:** Ready for paper trading; live trading only after validation suite passes + manual review

---

## 🏗️ AWS Infrastructure (Pre-configured)

### Instance Details
- **Public IP:** `18.220.38.215`
- **User:** `ubuntu`
- **SSH Key:** `volterra-new.pem` (chmod 400 in Model-deploy/)
- **Region:** `us-east-2` (Ohio)
- **OS:** Ubuntu 24.04.4 LTS
- **Instance Type:** (infer from ssh — likely t3.medium or similar)

### SSH Access
```bash
ssh -i "volterra-new.pem" ubuntu@18.220.38.215
```

### Installed Packages
```
✓ Python 3.12 + pip
✓ pandas, numpy, scipy (data science)
✓ requests (HTTP)
✓ build-essential (C/C++ compilation)
```

### Environment Variables (already in ~/.bashrc)
```bash
export MASSIVE_API_KEY="o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"  # Polygon.io
```

---

## 📊 Data APIs

### 1. Polygon.io (Verified ✓)
- **Status:** Working
- **API Key:** `o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf`
- **Endpoints Available:**
  - `/v3/reference/tickers` — stock/options tickers
  - `/v3/reference/options/contracts` — options chain data
  - `/v1/marketstatus` — market hours/status
  - Full historical options pricing data
- **Use Case:** Live options data for validation, supplementary IV/Greeks

### 2. Alpaca API (Not Working ❌)
- **Status:** 401 Unauthorized
- **Reason:** Credentials invalid or account not activated
- **Action Needed:** Register real Alpaca account if needed, or use Polygon.io instead

### 3. Alternative: yfinance (Local)
- No API key needed
- Can fetch historical SPY options/tickers from Yahoo Finance
- Good for backtesting and research

---

## 💻 Core Project Structure

### Main Modules

```
/Users/felipecardozo/Desktop/coding/Regime Volatility Arbitrage Engine/
├── main.py                      # Entry point (CLI with modes)
├── config.py                    # Global configuration
├── pricing_engine.py            # Rough-vol Volterra MC (Hybrid Scheme + Numba)
├── regime_filter.py             # 2-state Gaussian HMM (Baum–Welch)
├── orchestrator.py              # Strategy decision loop (10ms tick, 5s signal)
├── connection_manager.py        # IBKR TWS connectivity + HDF5 tick storage
├── execution_handler.py         # Order execution + chase algorithm
├── validation_suite.py          # MC convergence + regime stability tests
├── tests/                       # pytest suite
├── Model-deploy/
│   ├── DEPLOYMENT_GUIDE.md      # This file
│   ├── volterra-new.pem         # AWS SSH key
│   └── test_alpaca_api.py       # API test script
└── paper/
    └── main.tex                 # Academic paper (PDF locally)
```

### Key Parameters (in `config.py`)

**Pricing Engine:**
- `HURST_EXPONENT = 0.07` — volatility roughness (Gatheral 2018)
- `V0 = 0.04` — initial variance (20% annualized vol)
- `LAMBDA_VOL_OF_VOL = 0.3` — vol-of-vol scaling
- `MC_PATHS = 10_000` — Monte Carlo paths per pricing
- `MC_STEPS_PER_DAY = 24` — time resolution
- `SPOT_PRICE = 585.0` — current SPY price
- `RISK_FREE_RATE = 0.053` — risk-free rate (Fed funds)

**Regime Filter:**
- `HMM_N_STATES = 2` — Calm (0) and Turbulent (1)
- `HMM_TICKER = "SPY"` — train on daily SPY returns
- `HMM_HISTORY_YEARS = 5` — 5-year training window
- `TURBULENCE_THRESHOLD = 0.6` — P(Turbulent) > 60% gates entries

**IBKR Connection:**
- `TWS_HOST = "127.0.0.1"` — localhost (TWS must run locally)
- `TWS_PAPER_PORT = 7497` — paper trading
- `TWS_LIVE_PORT = 7496` — live trading
- `TWS_CLIENT_ID = 1` — market data client; execution uses client_id + 1
- `RECONNECT_DELAY_SEC = 5` — backoff on disconnect
- `RECONNECT_MAX_RETRIES = 10` — max reconnect attempts

---

## 🚀 Modes of Operation

```bash
python main.py --mode research    # Fit HMM, MC pricing, plot (no IBKR)
python main.py --mode validate    # Run validation suite (2 tests)
python main.py --mode test        # Smoke test of all layers
python main.py --mode paper       # Paper trading on IBKR (port 7497)
python main.py --mode live        # LIVE trading (port 7496)
```

### Mode Details

| Mode | Purpose | IBKR | Time | Output |
|------|---------|------|------|--------|
| `research` | Calibrate HMM, price straddles, visualize | No | ~5 min | regime_map.png, plots |
| `validate` | MC convergence + regime stability | No | ~10 min | validation_report.txt |
| `test` | Smoke test (one call/put/straddle) | No | <1 min | console logs |
| `paper` | Full strategy on paper account | Yes | ∞ | trade_log.csv, fills.csv, heartbeat.log |
| `live` | **LIVE TRADING** (gated by validation) | Yes | ∞ | same logs + real $$ |

---

## ⚙️ Deployment Checklist

### Pre-Deployment (Local Machine)

- [ ] Clone/pull latest code
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run `pytest tests/ -v` — all tests pass
- [ ] Run `python main.py --mode validate` — validation suite passes
- [ ] Run `python main.py --mode paper` for 24–48 hours on IBKR paper account
- [ ] Review `trade_log.csv` and `heartbeat.log` for stability

### AWS Deployment

- [ ] SSH into instance: `ssh -i volterra-new.pem ubuntu@18.220.38.215`
- [ ] Clone repository or sync code:
  ```bash
  git clone https://github.com/FelipeCardozo0/Regime-Volatility-Arbitrage-Engine.git
  cd Regime-Volatility-Arbitrage-Engine
  pip install -r requirements.txt
  ```
- [ ] Set up environment:
  ```bash
  export MASSIVE_API_KEY="o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"
  # Add to ~/.bashrc to persist
  ```
- [ ] Validate on AWS:
  ```bash
  python main.py --mode validate
  ```
- [ ] For **paper trading on AWS**, you need either:
  - An IBKR account with API enabled
  - TWS or Gateway running (not typically on cloud — this is the blocker)

### Notes on IBKR Architecture

**Issue:** TWS (Trader Workstation) is a GUI app designed for desktop. It does **not** run headless on cloud servers.

**Solutions:**
1. **IB Gateway (Recommended):** Lightweight API-only server (no GUI)
   ```bash
   # On AWS, download IB Gateway instead of TWS
   # Run: java -jar ibgateway-10.* start
   # Connects to localhost:4002 (or 4001 for paper)
   ```
2. **Local TWS + Port Forwarding:** Run TWS locally, SSH tunnel to AWS
   ```bash
   ssh -L 7497:localhost:7497 -i volterra-new.pem ubuntu@18.220.38.215
   ```
3. **Hybrid:** Run strategy on AWS, IBKR account on local machine
   - This is the **easiest for testing** (paper trading on your local TWS)

---

## 📥 Deployment Workflow (Recommended)

### Step 1: Validate Locally
```bash
cd ~/Desktop/coding/Regime\ Volatility\ Arbitrage\ Engine
python main.py --mode validate
# Check: All tests pass? Convergence plot looks good?
```

### Step 2: Paper Trade Locally (24–48 hours)
```bash
# Ensure IBKR TWS is running on port 7497 (paper account)
python main.py --mode paper
# Monitor: trade_log.csv, heartbeat.log
# Expected: 5–20 trades, Sharpe > 0.5
```

### Step 3: Sync to AWS
```bash
scp -i volterra-new.pem -r . ubuntu@18.220.38.215:~/regime-engine/
```

### Step 4: Run Research Mode on AWS (lightweight test)
```bash
ssh -i volterra-new.pem ubuntu@18.220.38.215
cd ~/regime-engine
python main.py --mode research
# Output: regime_map.png, pricing results
```

### Step 5: Paper Trading on AWS (if IB Gateway installed)
```bash
# Same as step 2, but on AWS
python main.py --mode paper
```

### Step 6: LIVE TRADING (Only After Validation ✓)
```bash
python main.py --mode live
# Gated by: validation_suite.py checks must pass
# Locked behind: manual review + papertrail of 48hrs+ paper performance
```

---

## 📋 Critical Decision Logic

The strategy follows a **priority-ordered decision tree** (executed every 5 seconds):

1. **Hard stop-loss:** Unrealised loss > 50% → `CLOSE_ALL`
2. **Time exit:** Holding period > 3 days → `CLOSE_ALL`
3. **Regime flip to Turbulent (P > 80%):**
   - If position open → `CLOSE_ALL`
   - If flat → `HOLD` (no entry)
4. **Moderate turbulence (60% < P ≤ 80%):**
   - If position open → `DELTA_HEDGE` (reduce exposure)
   - If flat → `HOLD` (no entry)
5. **Spread compression:** IV(model) - IV(market) < exit threshold (0.5%) → `CLOSE_ALL`
6. **Entry:** Regime = Calm **AND** IV spread > entry threshold (2%) → `ENTER_LONG` (ATM straddle)
7. **Default:** `HOLD`

**Example Trade Flow:**
```
[Calm regime, SPY = $585, IV spread = 2.5%]
  → ENTER_LONG: Buy ATM straddle (put + call @ 585)
  → Log to trade_log.csv
  
[5s later: IV spread narrows to 0.3%]
  → CLOSE_ALL: Alpha vanished
  → Record fill to fills.csv
  
[Heartbeat logged every 60s with uptime %, tick rate, queue size]
```

---

## 📊 Output Files & Monitoring

### During Execution

- **`tick_data.h5`** — All market ticks (bid/ask/last) stored in HDF5 (batched writes)
- **`trade_log.csv`** — One row per signal (timestamp, regime, model_iv, market_iv, action, PnL)
- **`fills.csv`** — One row per order fill (timestamp, symbol, qty, price, reason)
- **`heartbeat.log`** — System health every 60s (uptime %, tick rate, reconnects, signals)

### Analysis Commands

```bash
# Tail trade log
tail -f trade_log.csv

# Check system health
tail -f heartbeat.log

# Quick Pandas analysis (local)
python << 'EOF'
import pandas as pd
df = pd.read_csv('trade_log.csv')
print(f"Total signals: {len(df)}")
print(f"Sharpe: {df['pnl'].mean() / df['pnl'].std() if df['pnl'].std() > 0 else 0:.2f}")
print(f"Win rate: {(df['pnl'] > 0).mean():.2%}")
EOF
```

---

## 🔌 Integration with Polygon.io

The strategy uses **Polygon.io** for:
1. **Real-time options chains** — fetch current bid/ask for ATM straddle
2. **Volatility surface** — intra-day IV calibration (optional enhancement)
3. **Historical backtesting** — options prices for sensitivity analysis

### Example API Call

```python
import requests

API_KEY = "o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"

# Get current options contracts
response = requests.get(
    "https://api.polygon.io/v3/reference/options/contracts",
    params={
        "underlying_ticker": "SPY",
        "limit": 100,
        "apikey": API_KEY,
    }
)
contracts = response.json()["results"]
print(f"Found {len(contracts)} contracts")

# Filter ATM calls/puts (e.g., K ≈ 585)
atm_calls = [c for c in contracts if c["contract_type"] == "call" and abs(float(c["strike_price"]) - 585.0) < 2]
print(f"ATM calls: {atm_calls[:3]}")
```

---

## 🧪 Testing & Validation

### Unit Tests
```bash
pytest tests/ -v
# Covers: pricing_engine, regime_filter, orchestrator, execution_handler
```

### Smoke Test (all layers)
```bash
python main.py --mode test
# Quick price check: MC vs Black–Scholes, HMM fit, connection stubs
```

### Full Validation Suite
```bash
python main.py --mode validate
# Runs 2 tests:
#   1. MC convergence (O(N^-1/2) empirical verification)
#   2. Regime stability (train/test on real SPY data)
# Outputs: validation_report.txt + plots
```

### Integration Test (recommended before live)
```bash
# 1. Run on paper account for 48+ hours
python main.py --mode paper

# 2. Inspect logs
tail -100 trade_log.csv | cut -d, -f1-8

# 3. Check health
grep "SLA BREACH" heartbeat.log | wc -l  # Should be 0

# 4. Compute simple metrics
python << 'EOF'
import pandas as pd
df = pd.read_csv('trade_log.csv')
pnls = df['pnl'].dropna()
print(f"Mean PnL per trade: ${pnls.mean():.2f}")
print(f"Total trades: {len(df)}")
print(f"Max loss: ${pnls.min():.2f}")
EOF
```

---

## 🔐 Security & Best Practices

### Credentials & Keys

- **API Keys:** Store in environment variables or `.env`, never commit to git
  ```bash
  export MASSIVE_API_KEY="o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"  # Polygon.io
  export TWS_PASSWORD="your_ibkr_password"  # If auto-login needed
  ```

- **SSH Key:** `volterra-new.pem` should have `chmod 400` (AWS requirement)
  ```bash
  chmod 400 volterra-new.pem
  ```

- **IBKR Account:** Use a dedicated sub-account for algo trading
  - Never use your main account
  - Set tight permission limits (max position size, daily loss limit)
  - Enable 2FA for web access

### Monitoring & Alerts

- **Uptime SLA:** System logs a warning if uptime < 99% (see `SystemHealthMonitor`)
- **Tick Queue:** If queue grows unbounded, connection is broken — auto-reconnect triggers
- **Reconnect Counts:** Review `heartbeat.log` — >3 reconnects/hour suggests network issues
- **Error Logging:** All exceptions logged; review before escalating to live

### Kill Switch

```bash
# Graceful shutdown (finish current cycle, close positions)
Ctrl+C (SIGINT)

# Forced shutdown (if hung)
pkill -f "python main.py"
```

---

## 📈 Performance Expectations

From **5-year backtest** (regime-adjusted strategy):

| Metric | Value | Notes |
|--------|-------|-------|
| **Annualized Return** | 7.3% | Volatility arbitrage typically yields 5–10% |
| **Annualized Sharpe** | 1.38 | Excellent risk-adjusted return |
| **Max Drawdown** | −12.7% | Well-controlled via HMM gates |
| **Calmar Ratio** | 0.57 | Return per unit of drawdown |
| **Win Rate** | ~55% | Slightly better than coin-flip |
| **Avg Trade Duration** | 2–3 days | Straddle theta decay accelerates exits |

### Realistic Live Expectations

- **Lower return** (6–8%) due to:
  - Bid-ask friction
  - Commission (IBKR: ~$2–5 per contract)
  - Slippage (orders fill worse than model)
  - Transaction costs (gamma, vega rebalancing)

- **Regime detection lag:** Real data is messier than backtests
  - HMM may take 5–10 days to flip regimes vs. instant in backtest

---

## 🛠️ Troubleshooting

### Issue: "IBKR connection refused (port 7497)"
**Solution:**
- Ensure TWS is running and configured to allow API connections
- Settings → API → Enable ActiveX and Socket Clients
- Check port: `lsof -i :7497` (local machine)

### Issue: "HMM fit failed / singular matrix"
**Solution:**
- Increase `HMM_HISTORY_YEARS` in config (need more data)
- Or provide `rf.log_returns` directly instead of auto-fetching

### Issue: "MC convergence test failed"
**Solution:**
- Increase `MC_PATHS` and `MC_STEPS_PER_DAY` in config
- Or lower tolerance in `validation_suite.py` (accept O(N^-0.45) instead of O(N^-0.5))

### Issue: "Queue overflow / ticks dropping"
**Solution:**
- Increase tick_queue size in `ConnectionManager.__init__`
- Or reduce `reprice_interval` (signal loop only runs every 5s, not on every tick)

### Issue: "Polygon.io API 429 (rate limit)"
**Solution:**
- Free tier: 5 API calls/min
- Upgrade to paid tier for higher limits
- Or cache options chain (refresh every 1 min instead of 10s)

---

## 📞 Support & References

### Code References
- **Main Entry:** `main.py` (4 trading modes + health monitor)
- **Strategy Logic:** `orchestrator.py` (decision tree, position tracking)
- **Pricing:** `pricing_engine.py` (Volterra + Hybrid Scheme + Numba)
- **Regime:** `regime_filter.py` (Gaussian HMM + Baum–Welch)

### Academic References
1. **Rough volatility:** Gatheral, Jaisson, Rosenbaum (2018) — "Volatility is rough"
2. **Hybrid scheme:** Bennedsen, Lunde, Pakkanen (2017) — "Hybrid scheme for Brownian semistationary processes"
3. **HMM tutorial:** Rabiner (1989) — "A tutorial on hidden Markov models"
4. **Kelly sizing:** Thorp (2011) — "The Kelly criterion in blackjack, sports betting, and the stock market"

### Paper
Full mathematical derivations and empirical results in:
> **"Rough Volatility Arbitrage under Markov Regime: Volterra Process Approach with Double Exponential"**  
> Mitchell Scott, Ph.D. & Felipe Cardozo — Emory University

---

## ✅ Final Checklist Before Live Trading

- [ ] Validation suite passes (no convergence failures)
- [ ] Paper trading 48+ hours with stable PnL
- [ ] No SLA breaches in heartbeat.log (uptime ≥ 99%)
- [ ] All fills recorded and match trade log
- [ ] IBKR account sub-account created (not main account)
- [ ] Position size limits set in IBKR (max $10k notional per trade)
- [ ] Daily loss limit enabled ($5k max loss/day)
- [ ] Kill switch procedure tested (Ctrl+C closes positions)
- [ ] Monitoring set up (email alerts on SLA breach)
- [ ] Code review completed (strategy logic + risk checks)

---

**Ready to trade? Good luck! 🚀**

For questions or issues, refer to the code comments or the paper in `paper/main.tex`.
