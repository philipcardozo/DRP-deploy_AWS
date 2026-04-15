"""
StrategyOrchestrator — The "Brain" of the Regime Volatility Arbitrage Engine.

Integrates all three layers into a unified automated trading loop:

    ┌─────────────────────────────────────────────────────────────────┐
    │                      StrategyOrchestrator                       │
    │                                                                 │
    │  ConnectionManager ──ticks──► MarketState                      │
    │                                    │                            │
    │                                    ▼                            │
    │                            _pricing_pass()                      │
    │                          ┌──────────┴──────────┐               │
    │                   PricingEngine           RegimeFilter          │
    │                  (MC Model IV)       (HMM Regime Signal)        │
    │                          └──────────┬──────────┘               │
    │                                     ▼                           │
    │                          _evaluate_signal()                     │
    │                         ┌──────────────────┐                   │
    │                         │   Logic Tree      │                   │
    │                         │  Calm + spread>θ  │──► ENTER_LONG     │
    │                         │  Turbulent        │──► CLOSE_ALL      │
    │                         │  otherwise        │──► HOLD/HEDGE     │
    │                         └──────────┬───────┘                   │
    │                                    │                            │
    │                         Position + PnL tracker                  │
    │                         CSV trade log                           │
    └─────────────────────────────────────────────────────────────────┘

Alpha logic
───────────
The model produces an implied volatility from the rough-vol Monte Carlo
price via Black-Scholes inversion.  The *spread* is:

    spread (vol points) = σ_model − σ_market

A positive spread means the market is *underpricing* volatility relative
to the rough-vol model.  Long-vol entry is triggered when:

    1. Regime = "Calm"  (HMM filter, forward-filtered probability)
    2. spread > iv_entry_threshold  (default 2.0 vol points = 0.02)

Exit is triggered by:
    • Regime flipping to "Turbulent" → CLOSE_ALL
    • Stop-loss: unrealised PnL < -max_loss_pct × entry cost
    • Time exit: holding period exceeds max_hold_days

All events are persisted to a structured CSV file for post-trade
analysis and strategy simulation.
"""

from __future__ import annotations

import csv
import math
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from pricing_engine import PricingEngine, OptionResult
from regime_filter import RegimeFilter, RegimeSignal, CALM, TURBULENT
from connection_manager import ConnectionManager, TickEvent


# ════════════════════════════════════════════════════════════════════
# Enumerations & data containers
# ════════════════════════════════════════════════════════════════════

class SignalAction(str, Enum):
    HOLD         = "HOLD"
    ENTER_LONG   = "ENTER_LONG"     # Buy straddle — long volatility
    ADD_LONG     = "ADD_LONG"       # Scale into existing long position
    DELTA_HEDGE  = "DELTA_HEDGE"    # Regime warming; hedge delta exposure
    CLOSE_ALL    = "CLOSE_ALL"      # Flatten all positions immediately
    STOP_LOSS    = "STOP_LOSS"      # Hard stop triggered


class PositionState(Enum):
    FLAT         = auto()
    LONG_VOL     = auto()    # Long straddle (long gamma/vega)
    DELTA_HEDGED = auto()    # Position hedged but not closed


@dataclass
class MarketState:
    """
    Thread-local snapshot of the latest tick data.
    Updated on every incoming tick; read by the pricing pass.
    """
    timestamp:     datetime = field(default_factory=datetime.now)
    spot_price:    float | None = None
    market_iv:     float | None = None   # annualised (0.20 = 20%)
    opt_bid:       float | None = None
    opt_ask:       float | None = None
    opt_mid:       float | None = None
    delta:         float | None = None
    gamma:         float | None = None
    vega:          float | None = None
    theta:         float | None = None

    def mid_price(self) -> float | None:
        if self.opt_mid is not None:
            return self.opt_mid
        if self.opt_bid and self.opt_ask:
            return (self.opt_bid + self.opt_ask) / 2
        return None


@dataclass
class Position:
    """Tracks the current open position and its economics."""
    state:        PositionState = PositionState.FLAT
    qty:          int           = 0         # straddle contracts
    entry_price:  float         = 0.0       # cost per contract
    entry_iv:     float         = 0.0       # market IV at entry
    entry_time:   datetime | None = None
    last_model_iv: float        = 0.0
    cumulative_pnl: float       = 0.0       # realised PnL

    def is_open(self) -> bool:
        return self.state != PositionState.FLAT

    def unrealised_pnl(self, current_price: float) -> float:
        if not self.is_open() or self.qty == 0:
            return 0.0
        return (current_price - self.entry_price) * self.qty * 100  # × multiplier

    def holding_days(self) -> float:
        if self.entry_time is None:
            return 0.0
        return (datetime.now() - self.entry_time).total_seconds() / 86_400


@dataclass
class PricingResult:
    """Output of a single pricing pass."""
    timestamp:       datetime
    spot_price:      float
    strike:          float
    t_days:          int
    model_price:     float
    model_iv:        float         # annualised (BS-inverted from MC price)
    market_iv:       float | None
    iv_spread:       float | None  # model_iv - market_iv (vol points)
    mc_std_error:    float
    regime_state:    int
    prob_turbulent:  float
    regime_action:   str
    hurst_exponent:  float


@dataclass
class TradeSignal:
    """Fully annotated signal emitted by the logic tree."""
    timestamp:      datetime
    action:         SignalAction
    pricing:        PricingResult
    position:       Position
    reason:         str           = ""
    unrealised_pnl: float         = 0.0


# ════════════════════════════════════════════════════════════════════
# CSV trade logger
# ════════════════════════════════════════════════════════════════════

_CSV_COLUMNS = [
    # Timing
    "timestamp", "session_uptime_s",
    # Market
    "spot_price", "opt_mid_price",
    "market_iv", "market_iv_pct",
    # Model
    "model_price", "model_iv", "model_iv_pct",
    "iv_spread_vols", "mc_std_error",
    "hurst_exponent",
    # Regime
    "regime_state", "regime_label",
    "prob_turbulent", "regime_action",
    # Signal
    "signal_action", "signal_reason",
    # Position
    "position_state", "position_qty",
    "entry_price", "entry_iv",
    "entry_time", "holding_days",
    "unrealised_pnl",
]


class CSVTradeLogger:
    """
    Thread-safe, append-only CSV writer.

    A background thread drains a queue so the hot-path signal loop
    never blocks on disk I/O.  On shutdown, `close()` flushes all
    remaining rows and syncs the file descriptor.
    """

    def __init__(self, path: str = "trade_log.csv"):
        self._path   = Path(path)
        self._q: queue.Queue[dict] = queue.Queue()
        self._lock   = threading.Lock()
        self._file   = None
        self._writer = None
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        exists = self._path.exists()
        self._file   = open(self._path, "a", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_COLUMNS,
                                      extrasaction="ignore")
        if not exists or os.path.getsize(self._path) == 0:
            self._writer.writeheader()

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="csv-logger",
            daemon=True,
        )
        self._thread.start()

    def log(self, row: dict) -> None:
        """Non-blocking enqueue. Drops silently if queue is full (>10k rows)."""
        try:
            self._q.put_nowait(row)
        except queue.Full:
            pass

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        # Drain residual rows
        while not self._q.empty():
            try:
                self._writer.writerow(self._q.get_nowait())
            except Exception:
                pass
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                row = self._q.get(timeout=0.2)
                self._writer.writerow(row)
                # Batch-drain remaining rows without extra sleeps
                while True:
                    try:
                        self._writer.writerow(self._q.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue


# ════════════════════════════════════════════════════════════════════
# Black-Scholes IV inversion helper
# ════════════════════════════════════════════════════════════════════

def _bs_straddle_price(S: float, K: float, T: float,
                       r: float, sigma: float) -> float:
    """Analytical ATM straddle price under Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(abs(S - K), 0.0)
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put  = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def implied_vol_from_price(price: float, S: float, K: float,
                            T: float, r: float,
                            lo: float = 0.001,
                            hi: float = 5.0) -> float | None:
    """
    Invert the Black-Scholes straddle formula to find the implied
    volatility consistent with a given straddle dollar price.

    Uses Brent's method (guaranteed convergence, no gradient needed).

    Returns None if the price lies outside the no-arbitrage bounds.
    """
    if T <= 0:
        return None
    try:
        intrinsic = max(abs(S - K), 0.0)
        if price < intrinsic:
            return None
        f = lambda sigma: _bs_straddle_price(S, K, T, r, sigma) - price
        if f(lo) * f(hi) > 0:
            # Extend search range
            hi = 10.0
            if f(lo) * f(hi) > 0:
                return None
        return brentq(f, lo, hi, xtol=1e-6, maxiter=100)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════
# StrategyOrchestrator
# ════════════════════════════════════════════════════════════════════

class StrategyOrchestrator:
    """
    Automated trading brain that integrates rough-vol pricing,
    HMM regime detection, and IBKR order flow.

    Decision tree
    ─────────────
    Every `reprice_interval` seconds:

      IF regime == Calm AND spread > iv_entry_threshold:
          → ENTER_LONG   (buy straddle)

      IF regime == Turbulent (P > turbulence_threshold):
          → DELTA_HEDGE  (if 0.6 < P < 0.8)
          → CLOSE_ALL    (if P > 0.8)

      IF position is open AND unrealised_pnl < -max_loss_pct × cost:
          → STOP_LOSS

      IF position held > max_hold_days:
          → CLOSE_ALL   (time exit)

    Parameters
    ──────────
    engine : PricingEngine
    regime : RegimeFilter (must be pre-fitted)
    conn   : ConnectionManager
    t_days : int
        Option expiry horizon in trading days (default 5 = 1 week).
    iv_entry_threshold : float
        Minimum Model IV − Market IV spread in vol points to enter.
        Default 0.02 = 2 volatility points (e.g. 22% model vs 20% market).
    iv_exit_threshold : float
        Close the position when the spread compresses below this value.
        Default 0.005 = 0.5 vol points.
    max_hold_days : float
        Force-close after this many days regardless of spread.
    max_loss_pct : float
        Hard stop: close if loss exceeds this fraction of entry cost.
        Default 0.50 = 50% of the premium paid.
    max_contracts : int
        Maximum straddle contracts (position size limit).
    reprice_interval : float
        Seconds between MC pricing passes.
    poll_interval : float
        Seconds between tick queue drains (controls CPU usage).
    log_path : str
        CSV file path for the trade log.
    fast_paths : int
        MC paths used during live inference (reduced for speed).
        Full `engine.n_paths` are used only for batch reporting.
    on_signal : Callable | None
        Optional callback invoked for every emitted TradeSignal.
    """

    def __init__(
        self,
        engine:              PricingEngine,
        regime:              RegimeFilter,
        conn:                ConnectionManager,
        t_days:              int   = 5,
        iv_entry_threshold:  float = 0.06,   # 6 vol pts — roughness premium gate
        iv_exit_threshold:   float = 0.005,
        min_mkt_iv:          float = 0.18,   # VIX gate: only enter when mkt IV ≥ 18%
        max_hold_days:       float = 5.0,    # full 5-day straddle life
        max_loss_pct:        float = 0.50,
        max_contracts:       int   = 10,
        reprice_interval:    float = 5.0,
        poll_interval:       float = 0.01,
        log_path:            str   = "trade_log.csv",
        fast_paths:          int   = 2_000,
        on_signal:           Callable[[TradeSignal], None] | None = None,
    ):
        self.engine             = engine
        self.regime             = regime
        self.conn               = conn
        self.t_days             = t_days
        self.iv_entry_threshold = iv_entry_threshold
        self.iv_exit_threshold  = iv_exit_threshold
        self.min_mkt_iv         = min_mkt_iv
        self.max_hold_days      = max_hold_days
        self.max_loss_pct       = max_loss_pct
        self.max_contracts      = max_contracts
        self.reprice_interval   = reprice_interval
        self.poll_interval      = poll_interval
        self.on_signal          = on_signal

        # State
        self._running    = False
        self._market     = MarketState()
        self._position   = Position()
        self._signals:   list[TradeSignal] = []
        self._last_reprice = 0.0
        self._session_start = 0.0
        self._prev_regime: int | None = None   # detect regime transitions
        self._reprice_lock = threading.Lock()  # prevent concurrent pricing

        # Fast engine for real-time inference
        self._fast_engine = PricingEngine(
            H=engine.H, v0=engine.v0, nu=engine.nu,
            S0=engine.S0, r=engine.r, rho=engine.rho,
            n_paths=fast_paths,
            steps_per_day=engine.steps_per_day,
            kappa=engine.kappa, J=engine.J,
            seed=engine.seed + 9999,
        )

        # Logging
        self._logger = CSVTradeLogger(log_path)

    # ════════════════════════════════════════════════════════════════
    # Lifecycle
    # ════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """
        Start the orchestrator.  Blocks the calling thread until
        stopped via SIGINT or `stop()`.
        """
        self._running = True
        self._session_start = time.monotonic()
        self._logger.open()

        signal.signal(signal.SIGINT,  self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

        self._banner()
        self.conn.start()

        try:
            while self._running:
                loop_start = time.monotonic()
                self._drain_ticks()
                self._maybe_reprice()
                # Sleep only the time remaining in the poll interval
                elapsed = time.monotonic() - loop_start
                sleep_for = max(0.0, self.poll_interval - elapsed)
                time.sleep(sleep_for)
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shut down all components."""
        self._running = False
        # Force-close any open position before shutdown
        if self._position.is_open():
            self._emit(SignalAction.CLOSE_ALL, reason="Orchestrator shutdown")
        self.conn.stop()
        self._logger.close()
        self._print_session_report()

    def __enter__(self) -> "StrategyOrchestrator":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ════════════════════════════════════════════════════════════════
    # Simulation / back-test mode
    # ════════════════════════════════════════════════════════════════

    def run_simulation(
        self,
        spot_prices:  np.ndarray,
        market_ivs:   np.ndarray,
        opt_prices:   np.ndarray,
        dates:        list[datetime] | None = None,
        log_path:     str = "simulation_log.csv",
    ) -> list[TradeSignal]:
        """
        Replay a historical data series through the logic tree without
        a live IBKR connection.

        Parameters
        ──────────
        spot_prices  : (T,) array of underlying spot prices.
        market_ivs   : (T,) array of market implied vols (annualised).
        opt_prices   : (T,) array of straddle market prices.
        dates        : Optional list of datetime labels.

        Returns
        ──────
        List of TradeSignal objects — one per simulation step.
        """
        T = len(spot_prices)
        if dates is None:
            today = datetime.now()
            dates = [today + timedelta(days=i) for i in range(T)]

        sim_logger = CSVTradeLogger(log_path)
        sim_logger.open()
        self._logger = sim_logger
        self._session_start = time.monotonic()
        self._position = Position()
        self._signals  = []
        self._prev_regime = None

        for i in range(T):
            # Inject synthetic tick into market state
            self._market = MarketState(
                timestamp=dates[i],
                spot_price=float(spot_prices[i]),
                market_iv=float(market_ivs[i]),
                opt_mid=float(opt_prices[i]),
            )

            # Regime signal (causal — uses data up to index i)
            if self.regime.log_returns is not None:
                obs_window = self.regime.log_returns[:max(i + 1, 1)]
            else:
                obs_window = np.array([0.0])

            try:
                regime_sig = self.regime.current_signal(obs_window)
            except Exception:
                continue

            # Pricing pass
            pr = self._run_pricing(regime_sig)
            if pr is None:
                continue

            # Logic tree
            self._evaluate_and_emit(pr)

        sim_logger.close()
        return list(self._signals)

    # ════════════════════════════════════════════════════════════════
    # Tick ingestion
    # ════════════════════════════════════════════════════════════════

    def _drain_ticks(self) -> None:
        """
        Non-blocking drain of the ConnectionManager queue.
        Updates MarketState with the latest snapshot.
        Caps at 1 000 ticks per call to maintain loop responsiveness.
        """
        for _ in range(1_000):
            try:
                tick: TickEvent = self.conn.tick_queue.get_nowait()
                self._on_tick(tick)
            except queue.Empty:
                break

    def _on_tick(self, tick: TickEvent) -> None:
        """Merge a single TickEvent into the current MarketState."""
        m = self._market
        m.timestamp = tick.timestamp

        if tick.und_price and tick.und_price > 0:
            m.spot_price = tick.und_price
        if tick.implied_vol and tick.implied_vol > 0:
            m.market_iv = tick.implied_vol
        if tick.delta is not None:
            m.delta = tick.delta
        if tick.gamma is not None:
            m.gamma = tick.gamma
        if tick.vega is not None:
            m.vega = tick.vega
        if tick.theta is not None:
            m.theta = tick.theta

        fn = tick.field_name.upper()
        if "BID" in fn and tick.value > 0:
            m.opt_bid = tick.value
        elif "ASK" in fn and tick.value > 0:
            m.opt_ask = tick.value
        elif fn in ("LAST", "CLOSE") and tick.value > 0:
            m.opt_mid = tick.value

        # Update engine spot price
        if m.spot_price:
            self._fast_engine.S0 = m.spot_price
            self.engine.S0 = m.spot_price

    # ════════════════════════════════════════════════════════════════
    # Pricing pass
    # ════════════════════════════════════════════════════════════════

    def _maybe_reprice(self) -> None:
        """Trigger a pricing + signal pass at the configured interval."""
        now = time.monotonic()
        if now - self._last_reprice < self.reprice_interval:
            return
        if not self._reprice_lock.acquire(blocking=False):
            return   # previous pricing pass still running
        try:
            self._last_reprice = now
            regime_sig = self.regime.current_signal()
            pr = self._run_pricing(regime_sig)
            if pr is not None:
                self._evaluate_and_emit(pr)
        finally:
            self._reprice_lock.release()

    def _run_pricing(self, regime_sig: RegimeSignal) -> PricingResult | None:
        """
        Run the Monte Carlo pricer and invert to an implied volatility.

        Uses the *fast* engine (reduced paths) during live inference to
        keep latency under `reprice_interval`.  Returns None when
        market data is insufficient.
        """
        spot = self._market.spot_price or self._fast_engine.S0
        if spot <= 0:
            return None

        strike = spot          # ATM straddle
        T_ann  = self.t_days / 252.0
        r      = self._fast_engine.r

        # ── Calibrate v0 to current market IV (IBKR options feed) ──
        # Matches how backtests are run: v0 = market_iv²  so the
        # rough-vol model reflects current market conditions and the
        # spread measures the pure roughness premium (H≈0.07 vs H=0.5).
        if self._market.market_iv and self._market.market_iv > 0:
            self._fast_engine.v0 = self._market.market_iv ** 2

        # ── Monte Carlo model price ──
        try:
            result: OptionResult = self._fast_engine.price_straddle(
                K=strike, T_days=self.t_days)
        except Exception as exc:
            return None

        # ── Invert to implied vol ──
        model_iv = implied_vol_from_price(
            result.price, spot, strike, T_ann, r)
        if model_iv is None:
            return None

        return PricingResult(
            timestamp=datetime.now(),
            spot_price=spot,
            strike=strike,
            t_days=self.t_days,
            model_price=result.price,
            model_iv=model_iv,
            market_iv=self._market.market_iv,
            iv_spread=(
                (model_iv - self._market.market_iv)
                if self._market.market_iv else None
            ),
            mc_std_error=result.std_error,
            regime_state=regime_sig.state,
            prob_turbulent=regime_sig.prob_turbulent,
            regime_action=regime_sig.action,
            hurst_exponent=self._fast_engine.H,
        )

    # ════════════════════════════════════════════════════════════════
    # Logic tree
    # ════════════════════════════════════════════════════════════════

    def _evaluate_and_emit(self, pr: PricingResult) -> None:
        """
        Apply the decision tree to a PricingResult and emit the
        resulting TradeSignal.

        Priority order (highest to lowest):
          1. Hard stop-loss on open positions
          2. Time exit on stale positions
          3. Turbulence exit (regime risk-off)
          4. Spread compression exit (alpha gone)
          5. Entry on Calm + positive spread
          6. Hold
        """
        pos  = self._position
        mkt  = self._market

        # ── 1. Stop-loss ──────────────────────────────────────────
        if pos.is_open():
            mid = mkt.mid_price()
            if mid:
                upnl = pos.unrealised_pnl(mid)
                max_loss = -self.max_loss_pct * pos.entry_price * pos.qty * 100
                if upnl < max_loss:
                    self._emit(SignalAction.STOP_LOSS,
                               pr=pr,
                               reason=f"PnL {upnl:.2f} < limit {max_loss:.2f}")
                    return

        # ── 2. Time exit ──────────────────────────────────────────
        if pos.is_open() and pos.holding_days() > self.max_hold_days:
            self._emit(SignalAction.CLOSE_ALL, pr=pr,
                       reason=f"Time exit {pos.holding_days():.1f}d > {self.max_hold_days}d")
            return

        # ── 3. Turbulence exit ────────────────────────────────────
        if pr.prob_turbulent > 0.8:
            if pos.is_open():
                self._emit(SignalAction.CLOSE_ALL, pr=pr,
                           reason=f"P(Turb)={pr.prob_turbulent:.2%} > 80%")
            else:
                self._emit(SignalAction.HOLD, pr=pr,
                           reason=f"Turbulent — staying flat P(Turb)={pr.prob_turbulent:.2%}")
            return

        if pr.prob_turbulent > 0.6:
            if pos.is_open():
                self._emit(SignalAction.DELTA_HEDGE, pr=pr,
                           reason=f"P(Turb)={pr.prob_turbulent:.2%} — hedging delta")
                self._position.state = PositionState.DELTA_HEDGED
            else:
                self._emit(SignalAction.HOLD, pr=pr,
                           reason=f"Elevated P(Turb)={pr.prob_turbulent:.2%} — no entry")
            return

        # ── 4. Spread compression exit ────────────────────────────
        if (pos.is_open()
                and pr.iv_spread is not None
                and pr.iv_spread < self.iv_exit_threshold):
            self._emit(
                SignalAction.CLOSE_ALL, pr=pr,
                reason=f"Spread compressed: {pr.iv_spread*100:.2f} < "
                       f"{self.iv_exit_threshold*100:.2f} vol pts")
            return

        # ── 5. Entry — Calm + VIX gate + positive spread ─────────
        mkt_iv_ok = (
            pr.market_iv is not None
            and pr.market_iv >= self.min_mkt_iv
        )
        if (pr.regime_state == CALM
                and mkt_iv_ok
                and pr.iv_spread is not None
                and pr.iv_spread > self.iv_entry_threshold):

            if pos.state == PositionState.FLAT:
                self._emit(
                    SignalAction.ENTER_LONG, pr=pr,
                    reason=f"Spread={pr.iv_spread*100:.2f}vp > "
                           f"{self.iv_entry_threshold*100:.2f}vp  "
                           f"model={pr.model_iv*100:.2f}%  "
                           f"mkt={pr.market_iv*100:.2f}%  "
                           f"VIX={pr.market_iv*100:.1f}%≥{self.min_mkt_iv*100:.0f}%")
                return

            elif pos.state == PositionState.LONG_VOL:
                # Already long — only scale if spread widening
                if (pr.iv_spread > pr.iv_spread * 1.2
                        and pos.qty < self.max_contracts):
                    self._emit(SignalAction.ADD_LONG, pr=pr,
                               reason="Spread widening — adding to long")
                    return

        # ── 6. Default — hold ─────────────────────────────────────
        self._emit(SignalAction.HOLD, pr=pr,
                   reason=f"No trigger  spread={_fmt_spread(pr.iv_spread)}")

    # ════════════════════════════════════════════════════════════════
    # Signal emission & position management
    # ════════════════════════════════════════════════════════════════

    def _emit(
        self,
        action: SignalAction,
        pr: PricingResult | None = None,
        reason: str = "",
    ) -> TradeSignal:
        """
        Emit a signal: update position state, log to CSV, call
        the optional on_signal callback, print to console.
        """
        pos = self._position
        mkt = self._market
        mid = mkt.mid_price()

        # ── Position state machine ──
        if action == SignalAction.ENTER_LONG:
            pos.state       = PositionState.LONG_VOL
            pos.qty         = 1
            pos.entry_price = mid or (pr.model_price if pr else 0.0)
            pos.entry_iv    = pr.market_iv if pr else 0.0
            pos.entry_time  = datetime.now()
            pos.last_model_iv = pr.model_iv if pr else 0.0

        elif action == SignalAction.ADD_LONG:
            pos.qty = min(pos.qty + 1, self.max_contracts)

        elif action in (SignalAction.CLOSE_ALL, SignalAction.STOP_LOSS):
            if mid and pos.is_open():
                realised = pos.unrealised_pnl(mid)
                pos.cumulative_pnl += realised
            pos.state       = PositionState.FLAT
            pos.qty         = 0
            pos.entry_price = 0.0
            pos.entry_time  = None

        elif action == SignalAction.DELTA_HEDGE:
            pos.state = PositionState.DELTA_HEDGED

        # ── Detect regime transitions ──
        regime_now = pr.regime_state if pr else self._prev_regime
        if (regime_now != self._prev_regime
                and self._prev_regime is not None):
            prev_lbl = "Calm" if self._prev_regime == CALM else "Turbulent"
            now_lbl  = "Calm" if regime_now == CALM else "Turbulent"
        self._prev_regime = regime_now

        # ── Build signal ──
        upnl = pos.unrealised_pnl(mid) if mid else 0.0
        sig  = TradeSignal(
            timestamp=datetime.now(),
            action=action,
            pricing=pr,
            position=pos,
            reason=reason,
            unrealised_pnl=upnl,
        )
        self._signals.append(sig)

        # ── Log to CSV ──
        row = self._build_csv_row(sig)
        self._logger.log(row)

        # ── Console ──
        self._print_signal(sig)

        # ── Callback ──
        if self.on_signal:
            try:
                self.on_signal(sig)
            except Exception:
                pass

        return sig

    # ════════════════════════════════════════════════════════════════
    # Public inspection API
    # ════════════════════════════════════════════════════════════════

    @property
    def signals(self) -> list[TradeSignal]:
        """All emitted signals (read-only copy)."""
        return list(self._signals)

    @property
    def position(self) -> Position:
        """Current position snapshot."""
        return self._position

    def session_stats(self) -> dict:
        """Summary statistics for the current session."""
        enter_sigs = [s for s in self._signals
                      if s.action == SignalAction.ENTER_LONG]
        close_sigs = [s for s in self._signals
                      if s.action in (SignalAction.CLOSE_ALL,
                                      SignalAction.STOP_LOSS)]
        spreads = [
            s.pricing.iv_spread
            for s in self._signals
            if s.pricing and s.pricing.iv_spread is not None
        ]
        return {
            "total_signals":    len(self._signals),
            "entries":          len(enter_sigs),
            "exits":            len(close_sigs),
            "stop_losses":      sum(1 for s in self._signals
                                    if s.action == SignalAction.STOP_LOSS),
            "cumulative_pnl":   self._position.cumulative_pnl,
            "unrealised_pnl":   (self._position.unrealised_pnl(
                                    self._market.mid_price() or 0.0)
                                  if self._position.is_open() else 0.0),
            "avg_iv_spread":    float(np.mean(spreads)) if spreads else None,
            "max_iv_spread":    float(np.max(spreads))  if spreads else None,
            "session_uptime_s": time.monotonic() - self._session_start,
        }

    # ════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════

    def _build_csv_row(self, sig: TradeSignal) -> dict:
        pr  = sig.pricing
        pos = sig.position
        mkt = self._market
        return {
            "timestamp":        sig.timestamp.isoformat(),
            "session_uptime_s": f"{time.monotonic() - self._session_start:.2f}",
            "spot_price":       f"{pr.spot_price:.4f}"    if pr else "",
            "opt_mid_price":    f"{mkt.mid_price():.4f}"  if mkt.mid_price() else "",
            "market_iv":        f"{pr.market_iv:.6f}"     if pr and pr.market_iv else "",
            "market_iv_pct":    f"{pr.market_iv*100:.2f}" if pr and pr.market_iv else "",
            "model_price":      f"{pr.model_price:.4f}"   if pr else "",
            "model_iv":         f"{pr.model_iv:.6f}"      if pr else "",
            "model_iv_pct":     f"{pr.model_iv*100:.2f}"  if pr else "",
            "iv_spread_vols":   f"{pr.iv_spread*100:.4f}" if pr and pr.iv_spread else "",
            "mc_std_error":     f"{pr.mc_std_error:.6f}"  if pr else "",
            "hurst_exponent":   f"{pr.hurst_exponent:.4f}" if pr else "",
            "regime_state":     pr.regime_state            if pr else "",
            "regime_label":     ("Calm" if pr.regime_state == CALM
                                 else "Turbulent")         if pr else "",
            "prob_turbulent":   f"{pr.prob_turbulent:.4f}" if pr else "",
            "regime_action":    pr.regime_action           if pr else "",
            "signal_action":    sig.action.value,
            "signal_reason":    sig.reason,
            "position_state":   pos.state.name,
            "position_qty":     pos.qty,
            "entry_price":      f"{pos.entry_price:.4f}",
            "entry_iv":         f"{pos.entry_iv:.4f}",
            "entry_time":       pos.entry_time.isoformat() if pos.entry_time else "",
            "holding_days":     f"{pos.holding_days():.4f}",
            "unrealised_pnl":   f"{sig.unrealised_pnl:.2f}",
        }

    def _print_signal(self, sig: TradeSignal) -> None:
        _ICONS = {
            SignalAction.HOLD:        " ·",
            SignalAction.ENTER_LONG:  "▲▲",
            SignalAction.ADD_LONG:    " ▲",
            SignalAction.DELTA_HEDGE: "◇·",
            SignalAction.CLOSE_ALL:   "▼▼",
            SignalAction.STOP_LOSS:   "✕✕",
        }
        pr  = sig.pricing
        pos = sig.position
        icon = _ICONS.get(sig.action, "  ")
        spread_str = (f"{pr.iv_spread*100:+.2f}vp"
                      if pr and pr.iv_spread is not None else "  N/A  ")
        regime_str = ("Calm" if (pr and pr.regime_state == CALM)
                      else "Turb")
        upnl_str   = f"uPnL={sig.unrealised_pnl:+.2f}" if pos.is_open() else ""

        print(
            f"{icon}  {sig.timestamp:%H:%M:%S}  [{regime_str}]  "
            f"{sig.action.value:<14}  "
            f"spread={spread_str}  "
            f"modelIV={pr.model_iv*100:.2f}%  "
            f"mktIV={(pr.market_iv*100 if pr and pr.market_iv else 0):.2f}%  "
            f"qty={pos.qty}  {upnl_str}  {sig.reason}"
            if pr else
            f"{icon}  {sig.timestamp:%H:%M:%S}  {sig.action.value}  {sig.reason}"
        )

    def _print_session_report(self) -> None:
        stats = self.session_stats()
        print("\n" + "═" * 62)
        print("  SESSION REPORT")
        print("═" * 62)
        for k, v in stats.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:<25} {val}")
        print("═" * 62 + "\n")

    def _banner(self) -> None:
        print("═" * 62)
        print("  Regime Volatility Arbitrage Engine — ONLINE")
        print(f"  H={self.engine.H:.3f}  |  "
              f"entry_threshold={self.iv_entry_threshold*100:.1f}vp  |  "
              f"T={self.t_days}d")
        print("═" * 62)

    def _handle_sigint(self, signum, frame) -> None:
        print("\nSIGINT — shutting down …")
        self._running = False


# ════════════════════════════════════════════════════════════════════
# Utility
# ════════════════════════════════════════════════════════════════════

def _fmt_spread(spread: float | None) -> str:
    if spread is None:
        return "N/A"
    return f"{spread*100:+.2f}vp"
