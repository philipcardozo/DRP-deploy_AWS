# AWS Quick Reference — Regime Volatility Arbitrage Engine

## 🚀 One-Liner Deployment

```bash
bash aws_deploy.sh
```

This script will:
1. ✓ Test SSH connection
2. ✓ Clone/pull repository to ~/regime-engine
3. ✓ Install Python dependencies
4. ✓ Set environment variables (Polygon.io API key)
5. ✓ Run validation suite (optional)
6. ✓ Run lightweight research mode test

---

## 🔌 Manual AWS Access

**SSH into instance:**
```bash
ssh -i volterra-new.pem ubuntu@18.220.38.215
```

**Basic setup:**
```bash
cd ~/regime-engine
pip install -r requirements.txt
export MASSIVE_API_KEY="o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"
```

---

## ▶️ Running the Strategy

**Research mode (no IBKR needed, ~5 min):**
```bash
python main.py --mode research
# Output: regime_map.png, model prices
```

**Validation suite (10+ min):**
```bash
python main.py --mode validate
# Output: validation_report.txt + convergence plots
```

**Paper trading (requires IBKR TWS on port 7497):**
```bash
python main.py --mode paper
# Output: trade_log.csv, fills.csv, heartbeat.log
# Monitor with: tail -f trade_log.csv
```

**LIVE TRADING (port 7496):**
```bash
python main.py --mode live
# ⚠️  Only after validation passes + 48hrs paper trading
```

---

## 📊 Instance Details

| Field | Value |
|-------|-------|
| **IP Address** | 18.220.38.215 |
| **Region** | us-east-2 (Ohio) |
| **OS** | Ubuntu 24.04.4 LTS |
| **User** | ubuntu |
| **SSH Key** | volterra-new.pem (chmod 400) |

---

## 🔑 Credentials & APIs

| Service | Key/Endpoint | Status |
|---------|--------------|--------|
| **Polygon.io** | `o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf` | ✓ Working |
| **IBKR TWS (Paper)** | localhost:7497 | Requires local TWS |
| **IBKR TWS (Live)** | localhost:7496 | Requires local TWS |
| **Alpaca** | (Credentials failed 401) | ✗ Not working |

---

## 📊 Output Files

| File | Purpose | Format |
|------|---------|--------|
| `trade_log.csv` | All trading signals | CSV (timestamp, regime, IV spread, action, PnL) |
| `fills.csv` | Order fills | CSV (symbol, qty, price, reason) |
| `heartbeat.log` | System health | CSV (uptime %, ticks/min, queue size, errors) |
| `tick_data.h5` | Market ticks (bid/ask/last) | HDF5 binary |

---

## 🧪 Quick Tests

**Smoke test (all layers, <1 min):**
```bash
python main.py --mode test
```

**Check Polygon.io API:**
```bash
curl -s "https://api.polygon.io/v3/reference/tickers?limit=5&apikey=o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf" | jq .
```

**Download logs locally:**
```bash
scp -i volterra-new.pem ubuntu@18.220.38.215:~/regime-engine/trade_log.csv .
scp -i volterra-new.pem ubuntu@18.220.38.215:~/regime-engine/heartbeat.log .
```

---

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| `bash: python: command not found` | `python3 main.py --mode research` |
| `ModuleNotFoundError: No module named 'ibapi'` | `pip install ibapi` |
| `Connection refused (port 7497)` | Ensure TWS is running locally |
| `401 Unauthorized (Polygon.io)` | API key may be revoked; get new key from polygon.io |
| `Queue overflow` | Increase `tick_queue` size in `config.py` |

---

## 🛑 Stopping the Strategy

```bash
# Graceful shutdown (finish cycle, close positions)
Ctrl+C

# Forced kill (if hung)
pkill -f "python main.py"
```

---

## 📈 Expected Performance

From **backtests** (regime-adjusted):
- **Sharpe Ratio:** 1.38
- **Annual Return:** 7.3%
- **Max Drawdown:** −12.7%
- **Win Rate:** ~55%

**Live expectations:** 6–8% (lower due to friction, slippage, commissions)

---

## 🔐 Security Checklist

- [ ] SSH key: `chmod 400 volterra-new.pem`
- [ ] API key stored in environment, not hardcoded
- [ ] IBKR account: Use dedicated sub-account (not main)
- [ ] Daily loss limit: Set in IBKR (e.g., $5k max/day)
- [ ] Position size limit: Set in IBKR (e.g., $10k per trade)

---

## 📞 Full Documentation

See `DEPLOYMENT_GUIDE.md` for:
- Architecture overview
- Configuration parameters
- Testing & validation
- Troubleshooting guide
- Performance expectations
- Decision logic flowchart

---

**Ready to deploy? Run: `bash aws_deploy.sh`**

For questions: Check `DEPLOYMENT_GUIDE.md` or review code comments in `main.py`, `orchestrator.py`, `pricing_engine.py`.
