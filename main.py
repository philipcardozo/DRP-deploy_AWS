#!/usr/bin/env python3
"""
Regime Volatility Arbitrage Engine — Entry Point.

Modes
─────
  python main.py --mode paper      Live paper-trading on IBKR TWS port 7497
  python main.py --mode live       Live trading on IBKR TWS port 7496
  python main.py --mode research   HMM calibration + rough-vol pricing (no IBKR)
  python main.py --mode validate   Scientific validation suite (2 tests)
  python main.py --mode test       Quick smoke test of every layer
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config as cfg
from pricing_engine import PricingEngine
from regime_filter import RegimeFilter
from connection_manager import ConnectionManager, ConnStatus
from orchestrator import StrategyOrchestrator
from execution_handler import ExecutionHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ════════════════════════════════════════════════════════════════════
# System Health Monitor — heartbeat + uptime tracking
# ════════════════════════════════════════════════════════════════════

class SystemHealthMonitor:
    """
    Logs a structured heartbeat every `interval` seconds to both the
    console and `heartbeat.log`.  Tracks cumulative downtime and alerts
    when session uptime drops below the 99% SLA.

    Heartbeat record columns
    ────────────────────────
    timestamp, uptime_pct, status, ticks_rx, ticks_per_min,
    queue_size, reconnects, signals_total, last_error
    """

    SLA_UPTIME = 0.99          # 99 % uptime SLA
    LOG_COLUMNS = [
        "timestamp", "uptime_pct", "status",
        "ticks_rx", "ticks_per_min", "queue_size",
        "reconnects", "signals_total", "last_error",
    ]

    def __init__(
        self,
        conn:         ConnectionManager,
        orch:         StrategyOrchestrator,
        interval:     float = 60.0,
        log_path:     str   = "heartbeat.log",
    ):
        self.conn      = conn
        self.orch      = orch
        self.interval  = interval
        self._log_path = Path(log_path)

        self._session_start   = time.monotonic()
        self._downtime_secs   = 0.0
        self._last_check_ts   = time.monotonic()
        self._last_ticks_rx   = 0
        self._total_beats     = 0
        self._sla_breaches    = 0

        self._stop    = threading.Event()
        self._thread: threading.Thread | None = None
        self._file    = None
        self._writer  = None

    def start(self) -> None:
        exists = self._log_path.exists()
        self._file   = open(self._log_path, "a", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self.LOG_COLUMNS,
                                       extrasaction="ignore")
        if not exists or os.path.getsize(self._log_path) == 0:
            self._writer.writeheader()

        self._stop.clear()
        self._session_start = time.monotonic()

        self._thread = threading.Thread(
            target=self._loop,
            name="health-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("SystemHealthMonitor started  interval=%.0fs  sla=%.0f%%",
                    self.interval, self.SLA_UPTIME * 100)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._emit_heartbeat()   # final beat
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
        self._print_session_summary()

    # ── Internal ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self.interval)
            if not self._stop.is_set():
                self._emit_heartbeat()

    def _emit_heartbeat(self) -> None:
        now        = time.monotonic()
        elapsed    = now - self._session_start
        interval   = now - self._last_check_ts

        # Accumulate downtime when not connected
        if self.conn.state.status != ConnStatus.CONNECTED:
            self._downtime_secs += interval

        uptime_pct  = 1.0 - self._downtime_secs / max(elapsed, 1.0)
        ticks_rx    = self.conn.state.ticks_received
        ticks_delta = ticks_rx - self._last_ticks_rx
        ticks_per_min = ticks_delta / max(interval / 60.0, 1e-6)

        row = {
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "uptime_pct":    f"{uptime_pct:.4f}",
            "status":        self.conn.state.status.name,
            "ticks_rx":      ticks_rx,
            "ticks_per_min": f"{ticks_per_min:.1f}",
            "queue_size":    self.conn.tick_queue.qsize(),
            "reconnects":    self.conn.state.reconnect_count,
            "signals_total": len(self.orch.signals),
            "last_error":    self.conn.state.last_error[:60],
        }
        if self._writer:
            self._writer.writerow(row)

        level = logging.INFO
        msg   = (
            f"HEARTBEAT #{self._total_beats + 1:04d}  "
            f"uptime={uptime_pct:.2%}  "
            f"status={self.conn.state.status.name}  "
            f"ticks/min={ticks_per_min:.0f}  "
            f"queue={self.conn.tick_queue.qsize()}  "
            f"signals={len(self.orch.signals)}"
        )

        # SLA breach alert
        if uptime_pct < self.SLA_UPTIME:
            level = logging.WARNING
            msg  += f"  *** SLA BREACH — uptime {uptime_pct:.2%} < {self.SLA_UPTIME:.0%} ***"
            self._sla_breaches += 1

        logger.log(level, msg)

        self._last_check_ts = now
        self._last_ticks_rx = ticks_rx
        self._total_beats  += 1

    def _print_session_summary(self) -> None:
        elapsed   = time.monotonic() - self._session_start
        uptime    = 1.0 - self._downtime_secs / max(elapsed, 1.0)
        print("\n" + "═" * 62)
        print("  SYSTEM HEALTH SUMMARY")
        print("═" * 62)
        print(f"  Session duration  : {elapsed/60:.1f} min")
        print(f"  Session uptime    : {uptime:.3%}  "
              f"{'[OK]' if uptime >= self.SLA_UPTIME else '[SLA BREACH]'}")
        print(f"  Downtime          : {self._downtime_secs:.1f}s")
        print(f"  SLA breaches      : {self._sla_breaches}")
        print(f"  Heartbeats logged : {self._total_beats}")
        print(f"  Ticks received    : {self.conn.state.ticks_received:,}")
        print(f"  Reconnects        : {self.conn.state.reconnect_count}")
        print("═" * 62 + "\n")


# ════════════════════════════════════════════════════════════════════
# Shared component factory
# ════════════════════════════════════════════════════════════════════

def _build_engine() -> PricingEngine:
    return PricingEngine(
        H=cfg.HURST_EXPONENT,
        v0=cfg.V0,
        nu=cfg.LAMBDA_VOL_OF_VOL,
        S0=cfg.SPOT_PRICE,
        r=cfg.RISK_FREE_RATE,
        rho=-0.7,
        n_paths=cfg.MC_PATHS,
        steps_per_day=cfg.MC_STEPS_PER_DAY,
    )


def _build_regime(fit: bool = True) -> RegimeFilter:
    rf = RegimeFilter(
        ticker=cfg.HMM_TICKER,
        n_states=cfg.HMM_N_STATES,
        turbulence_threshold=cfg.TURBULENCE_THRESHOLD,
    )
    if fit:
        logger.info("Fetching %d years of %s data …",
                    cfg.HMM_HISTORY_YEARS, cfg.HMM_TICKER)
        rf.fetch_data(years=cfg.HMM_HISTORY_YEARS)
        logger.info("Fitting G-HMM (Baum-Welch) …")
        params = rf.fit()
        logger.info("  Calm  : μ=%.5f σ=%.5f", params.mu[0], params.sigma[0])
        logger.info("  Turb  : μ=%.5f σ=%.5f", params.mu[1], params.sigma[1])
    return rf


def _numba_warmup() -> None:
    logger.info("Numba JIT warm-up …")
    t0 = time.perf_counter()
    _ = PricingEngine(n_paths=10, steps_per_day=4).simulate(T_days=1)
    logger.info("  JIT compiled in %.2fs", time.perf_counter() - t0)


# ════════════════════════════════════════════════════════════════════
# Mode: Paper / Live trading
# ════════════════════════════════════════════════════════════════════

def run_trading(paper: bool = True) -> None:
    port      = cfg.TWS_PAPER_PORT if paper else cfg.TWS_LIVE_PORT
    mode_name = "PAPER" if paper else "LIVE"

    logger.info("═" * 62)
    logger.info("  %s TRADING MODE  (port %d)", mode_name, port)
    logger.info("═" * 62)

    _numba_warmup()

    engine = _build_engine()
    rf     = _build_regime(fit=True)

    # ── ConnectionManager (market data, clientId = 1) ──
    conn = ConnectionManager(
        host=cfg.TWS_HOST,
        port=port,
        client_id=cfg.TWS_CLIENT_ID,
        tick_store_path=cfg.HDF5_TICK_STORE,
        reconnect_delay=cfg.RECONNECT_DELAY_SEC,
        max_retries=cfg.RECONNECT_MAX_RETRIES,
    )

    # ── ExecutionHandler (order management, clientId = 2) ──
    exec_handler = ExecutionHandler(
        host=cfg.TWS_HOST,
        port=port,
        client_id=cfg.TWS_CLIENT_ID + 1,
        chase_interval=10.0,
        max_chase_steps=5,
        market_on_timeout=False,
        fills_log_path="fills.csv",
    )

    # ── StrategyOrchestrator ──
    orch = StrategyOrchestrator(
        engine=engine,
        regime=rf,
        conn=conn,
        t_days=5,
        iv_entry_threshold=0.02,
        iv_exit_threshold=0.005,
        max_hold_days=3.0,
        max_loss_pct=0.50,
        reprice_interval=5.0,
        log_path="trade_log.csv",
        fast_paths=2_000,
        on_signal=exec_handler.execute,
    )

    # ── Health monitor ──
    monitor = SystemHealthMonitor(
        conn=conn,
        orch=orch,
        interval=60.0,
        log_path="heartbeat.log",
    )

    # Subscribe to SPY market data
    conn.start()
    exec_handler.start()
    monitor.start()

    conn.subscribe_equity_ticks("SPY")
    # Uncomment to subscribe option ticks once you have expiry / strike:
    # conn.subscribe_option_ticks("SPY", expiry="20250117", strike=585.0, right="C")
    # conn.subscribe_option_ticks("SPY", expiry="20250117", strike=585.0, right="P")

    try:
        orch.start()         # blocking until SIGINT
    finally:
        monitor.stop()
        exec_handler.stop()
        conn.stop()
        logger.info("Shutdown complete.")


# ════════════════════════════════════════════════════════════════════
# Mode: Research
# ════════════════════════════════════════════════════════════════════

def run_research() -> None:
    logger.info("═" * 62)
    logger.info("  RESEARCH MODE")
    logger.info("═" * 62)

    rf = _build_regime(fit=True)
    signal = rf.current_signal()
    logger.info("Current regime: state=%d  P(Turb)=%.2f%%  → %s",
                signal.state, signal.prob_turbulent * 100, signal.action)

    rf.plot_regime_map(save_path="regime_map.png")
    logger.info("Regime map → regime_map.png")
    logger.info("Model summary:\n%s", rf.summary())

    _numba_warmup()
    engine = _build_engine()
    K = cfg.SPOT_PRICE

    t0 = time.perf_counter()
    straddle = engine.price_straddle(K=K, T_days=5)
    mc_time  = time.perf_counter() - t0

    bs_price = PricingEngine.black_scholes_straddle(
        S=cfg.SPOT_PRICE, K=K, T=5/252.0,
        r=cfg.RISK_FREE_RATE, sigma=np.sqrt(cfg.V0),
    )

    logger.info("1-Week ATM Straddle (K=%.0f):", K)
    logger.info("  Rough Vol MC : $%.4f  ±$%.4f", straddle.price, straddle.std_error)
    logger.info("  Black-Scholes: $%.4f", bs_price)
    logger.info("  MC time      : %.3fs  (%.2fms/path)",
                mc_time, mc_time / cfg.MC_PATHS * 1000)


# ════════════════════════════════════════════════════════════════════
# Mode: Validation suite
# ════════════════════════════════════════════════════════════════════

def run_validation() -> None:
    logger.info("═" * 62)
    logger.info("  VALIDATION SUITE MODE")
    logger.info("═" * 62)

    from validation_suite import ValidationSuite

    _numba_warmup()
    engine = PricingEngine(
        H=cfg.HURST_EXPONENT, v0=cfg.V0,
        nu=cfg.LAMBDA_VOL_OF_VOL, S0=100.0,
        r=cfg.RISK_FREE_RATE, rho=-0.7,
        n_paths=256, steps_per_day=8,
    )

    suite  = ValidationSuite(
        engine=engine,
        stability_ticker=cfg.HMM_TICKER,
        train_end="2019-12-31",
        test_end="2023-12-31",
        mc_n_values=[64, 128, 256, 512, 1024, 2048],
        mc_n_trials=15,
        output_dir=".",
    )
    report = suite.run(plot=True)

    print("\n" + report.stability.summary   if report.stability   else "")
    print("\n" + report.convergence.summary if report.convergence else "")

    if not report.all_passed:
        logger.error("Validation suite FAILED — do NOT deploy to live trading.")
        sys.exit(1)
    logger.info("All validation tests passed — system is deployment-ready.")


# ════════════════════════════════════════════════════════════════════
# Mode: Smoke test
# ════════════════════════════════════════════════════════════════════

def run_smoke_test() -> None:
    logger.info("── Smoke Test ──")

    logger.info("[1/4] PricingEngine …")
    pe  = PricingEngine(H=0.5, v0=0.04, nu=0.3, S0=100.0, r=0.05,
                        n_paths=500, steps_per_day=8, seed=0)
    res = pe.price_european_call(K=100, T_days=5)
    bs  = PricingEngine.black_scholes_call(100, 100, 5/252, 0.05, 0.2)
    logger.info("  MC=%.4f  BS=%.4f  diff=%.4f", res.price, bs, abs(res.price-bs))

    logger.info("[2/4] RegimeFilter …")
    rf  = RegimeFilter(n_states=2)
    rng = np.random.default_rng(42)
    obs = np.concatenate([rng.normal(5e-4, 8e-3, 500),
                          rng.normal(-1e-3, 2.5e-2, 200),
                          rng.normal(3e-4, 9e-3, 300)])
    rf.log_returns = obs
    rf.dates       = np.arange(len(obs))
    rf.fit(obs)
    states = rf.viterbi(obs)
    logger.info("  Viterbi: %d obs → %d states", len(states), len(np.unique(states)))

    logger.info("[3/4] ConnectionManager …")
    conn = ConnectionManager(port=7497)
    logger.info("  OK (stub)")

    logger.info("[4/4] ExecutionHandler …")
    eh = ExecutionHandler(port=7497, client_id=2, fills_log_path="/tmp/smoke_fills.csv")
    eh.start()
    eh.update_quote(bid=4.80, ask=5.20, spot=100.0)
    logger.info("  Quote updated, fill rate=%.0f%%", eh._fill_rate() * 100)
    eh.stop()

    logger.info("── All smoke tests passed ──")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regime Volatility Arbitrage Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "research", "validate", "test"],
        default="research",
        help=(
            "paper    → IBKR paper trading (port 7497)\n"
            "live     → IBKR live trading  (port 7496)\n"
            "research → HMM calibration + MC pricing  (no IBKR)\n"
            "validate → run scientific validation suite\n"
            "test     → quick smoke test"
        ),
    )
    args = parser.parse_args()

    dispatch = {
        "paper":    lambda: run_trading(paper=True),
        "live":     lambda: run_trading(paper=False),
        "research": run_research,
        "validate": run_validation,
        "test":     run_smoke_test,
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
