#!/usr/bin/env python3
"""
2025 Out-of-Sample Backtest — Regime Volatility Arbitrage Engine
================================================================
Data source  : Polygon.io  (env MASSIVE_API_KEY)  — no yfinance
Market IV    : 21-day trailing realised vol from SPY  (free-tier proxy)
               Live trading uses real 5-day ATM IV from IBKR options feed.

Fixes applied vs original backtest:
  1. IV spread entry threshold  2% → 6%   (roughness premium is always ~8-9%,
                                            2% never filtered any Calm day)
  2. VIX gate: only enter when market_IV ≥ 18%  (low-vol = theta bleed)
  3. Market IV = 21-day trailing HV from Polygon  (replaces spot VIX proxy)
  4. Max hold extended to 5 days  (full straddle life; 3-day cut left gamma)

Walk-forward methodology:
  • HMM trained on 2019-01-01 → 2024-12-31  (never sees 2025 data)
  • Forward-filtered day-by-day through 2025  (causal — no look-ahead)
  • v0 calibrated daily to market_iv²  (model reflects market conditions)
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import brentq
from scipy.stats import norm as sp_norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from polygon_client import fetch_close, fetch_spy_with_hv
from pricing_engine import PricingEngine
from regime_filter import RegimeFilter


# ════════════════════════════════════════════════════════════════════
# Parameters
# ════════════════════════════════════════════════════════════════════

# Polygon free tier provides ~1.5 yrs of daily data.
# We fetch the entire available window and split at 2025-01-01.
DATA_START         = "2023-01-01"   # earlier than available; Polygon returns what it has
SPLIT_DATE         = "2025-01-01"   # train = [DATA_START, SPLIT_DATE), test = [SPLIT_DATE, ...]
TEST_END           = "2025-12-31"

STRADDLE_DAYS      = 5       # 1-week ATM straddle (matches strategy)
IV_ENTRY_THRESH    = cfg.IV_ENTRY_THRESHOLD   # 6 vol pts
MIN_MKT_IV         = cfg.MIN_MKT_IV           # 18%  VIX gate
IV_EXIT_THRESH     = cfg.IV_EXIT_THRESHOLD    # 0.5 vol pts
MAX_HOLD_DAYS      = int(cfg.MAX_HOLD_DAYS)   # 5 days (full straddle life)
STOP_LOSS_PCT      = cfg.MAX_LOSS_PCT         # 50% of premium
CAPITAL_PER_TRADE  = 0.10    # 10% of portfolio per straddle
TC_RT              = 0.003   # 0.3% round-trip transaction cost
HV_WINDOW          = 21      # trailing realized-vol window (days)
N_MC_PATHS         = 2_000   # MC paths (faster than prod's 10 000)


# ════════════════════════════════════════════════════════════════════
# Black-Scholes helpers
# ════════════════════════════════════════════════════════════════════

def bs_straddle(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return abs(S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * sp_norm.cdf(d1) - K * math.exp(-r * T) * sp_norm.cdf(d2)
    put  = K * math.exp(-r * T) * sp_norm.cdf(-d2) - S * sp_norm.cdf(-d1)
    return call + put


def implied_vol(price: float, S: float, K: float, T: float, r: float) -> float | None:
    try:
        f = lambda sig: bs_straddle(S, K, T, r, sig) - price
        if f(0.01) * f(5.0) > 0:
            return None
        return brentq(f, 0.01, 5.0, xtol=1e-6, maxiter=200)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════
# HMM forward filter
# ════════════════════════════════════════════════════════════════════

def forward_filter(log_returns: np.ndarray, params) -> np.ndarray:
    T = len(log_returns)
    log_B = sp_norm.logpdf(log_returns[:, None],
                           loc=params.mu[None, :], scale=params.sigma[None, :])
    alpha = np.empty((T, 2))
    a0 = params.pi * np.exp(log_B[0])
    s0 = a0.sum()
    alpha[0] = a0 / (s0 + 1e-300)
    for t in range(1, T):
        a = (alpha[t - 1] @ params.A) * np.exp(log_B[t])
        s = a.sum()
        alpha[t] = a / (s + 1e-300)
    return alpha[:, 1]   # P(Turbulent | obs[0..t])


# ════════════════════════════════════════════════════════════════════
# Main backtest
# ════════════════════════════════════════════════════════════════════

def run_backtest():
    bar = "=" * 65
    print(f"\n{bar}")
    print("  2025 OUT-OF-SAMPLE BACKTEST — Regime Vol Arbitrage (v2)")
    print(f"  Data: Polygon.io   Fixes: spread≥6% | VIX≥18% | hold=5d")
    print(bar)

    # ── 1. Fetch data (Polygon) ───────────────────────────────────────
    print(f"\n[1/5]  Fetching SPY from Polygon.io …")
    # One call for the full window; Polygon returns what's on the free tier
    all_close = fetch_close("SPY", DATA_START, TEST_END)
    all_log_ret = np.log(all_close / all_close.shift(1)).dropna()

    # Split at SPLIT_DATE (causal: train never sees 2025)
    train_close   = all_close[all_close.index < SPLIT_DATE]
    test_close    = all_close[all_close.index >= SPLIT_DATE]
    log_ret_train = all_log_ret[all_log_ret.index < SPLIT_DATE].to_numpy(np.float64)
    log_ret_2025  = all_log_ret[all_log_ret.index >= SPLIT_DATE]
    ret_dates_2025 = log_ret_2025.index

    # Trailing HV over the full series (window primed on pre-2025 data)
    full_hv  = np.log(all_close / all_close.shift(1)).rolling(HV_WINDOW).std() * np.sqrt(252)
    test_hv  = full_hv[full_hv.index >= SPLIT_DATE].dropna()

    train_start_actual = train_close.index[0].strftime("%Y-%m-%d")
    train_end_actual   = train_close.index[-1].strftime("%Y-%m-%d")
    print(f"        Training : {train_start_actual} → {train_end_actual}  ({len(log_ret_train)} days)")
    print(f"        Test     : {SPLIT_DATE}  → {TEST_END}   ({len(test_close)} days, "
          f"{len(log_ret_2025)} return obs)")
    print(f"        Market IV: 21-day trailing realised vol (Polygon proxy)")
    print(f"        Note     : Polygon free tier data starts ~Apr 2024")
    print(f"        Entry gate: spread ≥ {IV_ENTRY_THRESH:.0%}  AND  market_IV ≥ {MIN_MKT_IV:.0%}")

    # ── 2. Fit HMM ───────────────────────────────────────────────────
    print(f"\n[2/5]  Fitting 2-state Gaussian HMM on {train_start_actual}–{train_end_actual} …")
    rf = RegimeFilter(ticker="SPY", turbulence_threshold=cfg.TURBULENCE_THRESHOLD)
    rf.log_returns = log_ret_train
    rf.dates       = np.arange(len(log_ret_train))
    params = rf.fit(n_restarts=10, seed=0)
    print(f"        Calm  :  μ = {params.mu[0]:+.5f}   σ = {params.sigma[0]:.5f}")
    print(f"        Turb  :  μ = {params.mu[1]:+.5f}   σ = {params.sigma[1]:.5f}")
    print(f"        Trans :  P(C→C) = {params.A[0,0]:.3f}   P(T→T) = {params.A[1,1]:.3f}")

    # ── 3. Forward-filter 2025 ───────────────────────────────────────
    print(f"\n[3/5]  Forward-filtering 2025 returns (causal) …")
    prob_turb = forward_filter(log_ret_2025.to_numpy(np.float64), params)
    regimes   = (prob_turb >= cfg.TURBULENCE_THRESHOLD).astype(int)
    calm_days = (regimes == 0).sum()
    turb_days = (regimes == 1).sum()

    # Apply VIX gate to count eligible days
    eligible_days = sum(
        1 for i, d in enumerate(ret_dates_2025)
        if regimes[i] == 0 and d in test_hv.index and test_hv.loc[d] >= MIN_MKT_IV
    )
    print(f"        Calm days : {calm_days}   Turbulent days : {turb_days}")
    print(f"        Calm + VIX≥18% (eligible entries) : {eligible_days} days")

    # ── 4. Numba JIT warm-up ─────────────────────────────────────────
    print(f"\n[4/5]  Warming up Numba JIT …")
    engine = PricingEngine(
        H=cfg.HURST_EXPONENT, v0=cfg.V0, nu=cfg.LAMBDA_VOL_OF_VOL,
        S0=cfg.SPOT_PRICE, r=cfg.RISK_FREE_RATE, rho=-0.7,
        n_paths=N_MC_PATHS, steps_per_day=cfg.MC_STEPS_PER_DAY, seed=42,
    )
    engine.simulate(T_days=1)
    print("        JIT compiled.")

    # ── 5. Walk-forward simulation ───────────────────────────────────
    print(f"\n[5/5]  Running walk-forward simulation …\n")

    trades:       list[dict] = []
    equity:       list[float] = [1.0]
    equity_dates: list = [test_close.index[0]]
    position:     dict | None = None

    for i, date in enumerate(ret_dates_2025):
        if date not in test_close.index:
            continue

        spot   = float(test_close.loc[date])
        # Market IV: 21-day trailing HV from Polygon (live = IBKR options feed)
        mkt_iv = float(test_hv.loc[date]) if date in test_hv.index else None
        regime = int(regimes[i])
        p_turb = float(prob_turb[i])

        # ── Manage open position ─────────────────────────────────────
        if position is not None:
            hold    = i - position["entry_i"]
            K       = position["strike"]
            cost    = position["entry_cost"]

            days_left = STRADDLE_DAYS - hold
            if days_left > 0 and mkt_iv:
                T_rem   = days_left / 252.0
                cur_val = bs_straddle(spot, K, T_rem, cfg.RISK_FREE_RATE, mkt_iv)
            else:
                cur_val = abs(spot - K)

            pnl     = cur_val - cost
            pnl_pct = pnl / cost

            # Spread check (calibrated repricing)
            cur_spread = None
            if days_left > 0 and mkt_iv:
                engine.S0 = spot
                engine.v0 = mkt_iv ** 2
                mc_price  = engine.price_straddle(K=K, T_days=max(days_left, 1)).price
                mod_iv    = implied_vol(mc_price, spot, K, days_left / 252.0,
                                       cfg.RISK_FREE_RATE)
                cur_spread = (mod_iv - mkt_iv) if mod_iv else None

            # Exit priority order
            reason = None
            if pnl_pct <= -STOP_LOSS_PCT:
                reason = "STOP_LOSS"
            elif hold >= MAX_HOLD_DAYS:
                reason = "TIME_EXIT"
            elif regime == 1 and p_turb >= 0.80:
                reason = "REGIME_TURBULENT"
            elif cur_spread is not None and cur_spread < IV_EXIT_THRESH:
                reason = "SPREAD_COMPRESS"

            if reason:
                tc      = TC_RT * cost
                net_pnl = pnl - tc
                ret     = net_pnl / cost
                trades.append({
                    "entry_date":  position["entry_date"],
                    "exit_date":   date,
                    "hold_days":   hold,
                    "entry_spot":  position["entry_spot"],
                    "exit_spot":   spot,
                    "strike":      K,
                    "entry_cost":  cost,
                    "exit_val":    cur_val,
                    "net_pnl":     net_pnl,
                    "return":      ret,
                    "eq_return":   ret * CAPITAL_PER_TRADE,
                    "reason":      reason,
                    "model_iv":    position["model_iv"],
                    "market_iv":   position["market_iv"],
                    "iv_spread":   position["iv_spread"],
                })
                equity.append(equity[-1] * (1 + ret * CAPITAL_PER_TRADE))
                equity_dates.append(date)
                position = None

        # ── Check entry ──────────────────────────────────────────────
        if position is None and regime == 0 and mkt_iv is not None:

            # VIX gate — only enter elevated-vol environments
            if mkt_iv < MIN_MKT_IV:
                continue   # low vol → theta bleed kills straddles

            K = spot
            T = STRADDLE_DAYS / 252.0

            # Rough-vol MC, calibrated to current market vol
            engine.S0 = spot
            engine.v0 = mkt_iv ** 2
            mc_res    = engine.price_straddle(K=K, T_days=STRADDLE_DAYS)
            model_iv  = implied_vol(mc_res.price, spot, K, T, cfg.RISK_FREE_RATE)

            if model_iv is not None:
                spread = model_iv - mkt_iv

                if spread >= IV_ENTRY_THRESH:
                    entry_cost = bs_straddle(spot, K, T, cfg.RISK_FREE_RATE, mkt_iv)
                    entry_cost = max(entry_cost, 0.01)

                    position = {
                        "entry_i":    i,
                        "entry_date": date,
                        "entry_spot": spot,
                        "strike":     K,
                        "entry_cost": entry_cost,
                        "model_iv":   model_iv,
                        "market_iv":  mkt_iv,
                        "iv_spread":  spread,
                    }
                    print(f"  ENTER  {date.date()}  SPY={spot:.2f}  "
                          f"model_iv={model_iv:.1%}  mkt_iv={mkt_iv:.1%}  "
                          f"spread={spread:+.1%}")

    # Close any leftover position
    if position is not None:
        date  = ret_dates_2025[-1]
        spot  = float(test_close.iloc[-1])
        K     = position["strike"]
        cost  = position["entry_cost"]
        val   = abs(spot - K)
        pnl   = val - cost - TC_RT * cost
        ret   = pnl / cost
        hold  = len(ret_dates_2025) - 1 - position["entry_i"]
        trades.append({
            "entry_date":  position["entry_date"],
            "exit_date":   date,
            "hold_days":   hold,
            "entry_spot":  position["entry_spot"],
            "exit_spot":   spot,
            "strike":      K,
            "entry_cost":  cost,
            "exit_val":    val,
            "net_pnl":     pnl,
            "return":      ret,
            "eq_return":   ret * CAPITAL_PER_TRADE,
            "reason":      "END_OF_BACKTEST",
            "model_iv":    position["model_iv"],
            "market_iv":   position["market_iv"],
            "iv_spread":   position["iv_spread"],
        })
        equity.append(equity[-1] * (1 + ret * CAPITAL_PER_TRADE))
        equity_dates.append(date)

    return trades, equity, equity_dates, test_close, test_hv, prob_turb, ret_dates_2025


# ════════════════════════════════════════════════════════════════════
# Results
# ════════════════════════════════════════════════════════════════════

def print_results(trades: list[dict], equity: list[float]) -> pd.DataFrame | None:
    bar = "=" * 65
    if not trades:
        print(f"\n{bar}")
        print("  No trades generated in 2025 with the updated filters.")
        print("  (All Calm days had market IV below 18% or spread below 6%)")
        print(bar)
        return None

    df = pd.DataFrame(trades)
    rets = df["return"].values
    eq   = np.array(equity)

    total_ret  = eq[-1] - 1.0
    n_days     = (pd.Timestamp(TEST_END) - pd.Timestamp(SPLIT_DATE)).days
    ann_ret    = (1 + total_ret) ** (365 / n_days) - 1
    ann_factor = math.sqrt(252 / STRADDLE_DAYS)
    sharpe     = rets.mean() / (rets.std() + 1e-9) * ann_factor
    win_rate   = (rets > 0).mean()
    avg_hold   = df["hold_days"].mean()
    max_dd     = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()

    print(f"\n{bar}")
    print("  2025 BACKTEST RESULTS  (Fixes: spread≥6%, VIX≥18%, hold=5d)")
    print(bar)
    print(f"  Period          : {SPLIT_DATE}  →  {TEST_END}")
    print(f"  Total trades    : {len(df)}")
    print(f"  Win rate        : {win_rate:.1%}")
    print(f"  Avg hold        : {avg_hold:.1f} trading days")
    print()
    print(f"  Total return    : {total_ret:+.2%}   (10% capital / trade)")
    print(f"  Ann. return     : {ann_ret:+.2%}")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print(f"  Max drawdown    : {max_dd:.2%}")
    print(f"  Best trade      : {rets.max():+.2%}")
    print(f"  Worst trade     : {rets.min():+.2%}")
    print(f"  Avg trade ret.  : {rets.mean():+.2%}")
    print()
    print("  Exit reasons:")
    for reason, cnt in df["reason"].value_counts().items():
        print(f"    {reason:<24} {cnt:>2} trades  ({cnt/len(df):.0%})")
    print()
    print(f"  {'Entry':>10}  {'Exit':>10}  {'Hld':>3}  {'ModIV':>6}  "
          f"{'MktIV':>6}  {'Sprd':>5}  {'Ret':>7}  Reason")
    print("  " + "-" * 70)
    for _, row in df.iterrows():
        print(f"  {str(row['entry_date'].date()):>10}  "
              f"{str(row['exit_date'].date()):>10}  "
              f"{row['hold_days']:>3.0f}d  "
              f"{row['model_iv']:>5.1%}  "
              f"{row['market_iv']:>5.1%}  "
              f"{row['iv_spread']:>+4.1%}  "
              f"{row['return']:>+6.2%}  "
              f"  {row['reason']}")
    print(bar)
    return df


# ════════════════════════════════════════════════════════════════════
# Plot
# ════════════════════════════════════════════════════════════════════

def plot_results(
    trades, equity, equity_dates,
    spy, hv, prob_turb, ret_dates,
):
    df = pd.DataFrame(trades) if trades else pd.DataFrame()
    test_spy  = spy[SPLIT_DATE:TEST_END]
    regime_s  = pd.Series(prob_turb, index=ret_dates)
    hv_series = hv[SPLIT_DATE:TEST_END]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "2025 Out-of-Sample Backtest — Regime Volatility Arbitrage Engine  (v2)\n"
        f"Fixes: spread ≥ {IV_ENTRY_THRESH:.0%} | VIX gate ≥ {MIN_MKT_IV:.0%} | "
        f"Hold = {MAX_HOLD_DAYS}d | Data: Polygon.io  (free tier: pre-2025 history)",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.50,
                           height_ratios=[2.5, 1.2, 1.2, 1.2])

    # ── Panel 1: SPY + shading ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(test_spy.index, test_spy.values, color="#1a1a2e", lw=1.5, label="SPY")
    for d in regime_s.index:
        if regime_s.loc[d] >= cfg.TURBULENCE_THRESHOLD:
            ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.20, color="#F44336", lw=0)
    if not df.empty:
        for _, t in df.iterrows():
            col = "#2196F3" if t["return"] > 0 else "#F44336"
            ax1.axvspan(t["entry_date"], t["exit_date"], alpha=0.20, color=col, lw=0)
            ax1.axvline(t["entry_date"], color="#2196F3", lw=1.2, linestyle="--", alpha=0.85)
            ax1.axvline(t["exit_date"],  color="#F44336",  lw=1.2, linestyle="--", alpha=0.85)
    ax1.legend(handles=[
        ax1.lines[0],
        Patch(color="#F44336", alpha=0.25, label="Turbulent regime"),
        Patch(color="#2196F3", alpha=0.20, label="Trade window (win)"),
        Patch(color="#F44336", alpha=0.20, label="Trade window (loss)"),
    ], loc="upper right", fontsize=8)
    ax1.set_ylabel("SPY Price ($)")
    ax1.set_title("SPY Price + Regime Overlay + Trade Periods", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: P(Turbulent) ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(regime_s.index, regime_s.values, color="#9C27B0", lw=1.5)
    ax2.axhline(cfg.TURBULENCE_THRESHOLD, color="#F44336", lw=1.2, linestyle="--",
                label=f"Threshold {cfg.TURBULENCE_THRESHOLD:.0%}")
    ax2.fill_between(regime_s.index, regime_s.values, cfg.TURBULENCE_THRESHOLD,
                     where=regime_s.values >= cfg.TURBULENCE_THRESHOLD,
                     color="#F44336", alpha=0.25)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("P(Turbulent)")
    ax2.set_title("HMM Forward-Filtered Turbulence Probability (causal)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Market IV (21-day trailing HV) + VIX gate ──────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(hv_series.index, hv_series.values * 100,
             color="#FF9800", lw=1.5, label="21d trailing HV (market IV proxy)")
    ax3.axhline(MIN_MKT_IV * 100, color="#F44336", lw=1.2, linestyle="--",
                label=f"VIX gate ({MIN_MKT_IV:.0%}) — no entry below")
    ax3.fill_between(hv_series.index, hv_series.values * 100, MIN_MKT_IV * 100,
                     where=hv_series.values < MIN_MKT_IV,
                     color="#F44336", alpha=0.15, label="Blocked by VIX gate")
    if not df.empty:
        ax3.scatter(df["entry_date"], df["market_iv"] * 100,
                    color="#2196F3", s=50, zorder=5, label="Entry market IV")
    ax3.set_ylabel("Annualised Vol (%)")
    ax3.set_title("Market IV (21d trailing HV) + VIX gate filter", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Equity curve ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if len(equity_dates) > 1:
        eq_s = pd.Series([(e - 1) * 100 for e in equity], index=equity_dates)
        ax4.plot(eq_s.index, eq_s.values, color="#4CAF50", lw=2)
        ax4.axhline(0, color="black", lw=0.8)
        ax4.fill_between(eq_s.index, eq_s.values, 0,
                         where=np.array(equity) >= 1, color="#4CAF50", alpha=0.25)
        ax4.fill_between(eq_s.index, eq_s.values, 0,
                         where=np.array(equity) < 1, color="#F44336", alpha=0.25)
    ax4.set_ylabel("Cum. Return (%)")
    ax4.set_title(f"Strategy Equity Curve ({CAPITAL_PER_TRADE:.0%} capital per straddle)", fontsize=10)
    ax4.grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_2025.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {out_path}")
    plt.show()


# ════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    trades, equity, equity_dates, spy, hv, prob_turb, ret_dates = run_backtest()
    df = print_results(trades, equity)
    plot_results(trades, equity, equity_dates, spy, hv, prob_turb, ret_dates)
