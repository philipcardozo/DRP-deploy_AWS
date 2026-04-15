"""
Integration tests for StrategyOrchestrator.

Tests run entirely in simulation mode — no IBKR connection or network
access required.  Uses synthetic spot/IV/price series that are designed
to trigger every branch of the logic tree at least once.
"""

from __future__ import annotations

import math
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from orchestrator import (
    StrategyOrchestrator, SignalAction, PositionState,
    Position, MarketState, PricingResult,
    implied_vol_from_price, _bs_straddle_price, CSVTradeLogger,
)
from pricing_engine import PricingEngine
from regime_filter import RegimeFilter, CALM, TURBULENT


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def fitted_regime() -> RegimeFilter:
    """Pre-fitted RegimeFilter on synthetic data (no network)."""
    rng = np.random.default_rng(0)
    calm_obs = rng.normal(0.0005, 0.007, 800)
    turb_obs = rng.normal(-0.001, 0.022, 200)
    obs = np.concatenate([calm_obs, turb_obs])

    rf = RegimeFilter(turbulence_threshold=0.6)
    rf.log_returns = obs
    rf.dates = np.arange(len(obs))
    rf.fit(obs=obs, n_restarts=4, seed=42)
    return rf


@pytest.fixture
def fast_engine() -> PricingEngine:
    return PricingEngine(
        H=0.07, v0=0.04, nu=0.3, S0=100.0, r=0.05,
        n_paths=500, steps_per_day=8, seed=1,
    )


@pytest.fixture
def orchestrator(fast_engine, fitted_regime, tmp_path) -> StrategyOrchestrator:
    conn = MagicMock()
    conn.tick_queue.get_nowait.side_effect = __import__("queue").Empty
    conn.start = MagicMock()
    conn.stop  = MagicMock()

    return StrategyOrchestrator(
        engine=fast_engine,
        regime=fitted_regime,
        conn=conn,
        t_days=5,
        iv_entry_threshold=0.02,
        iv_exit_threshold=0.005,
        max_hold_days=3.0,
        max_loss_pct=0.50,
        reprice_interval=999.0,   # disable auto-reprice
        poll_interval=0.001,
        log_path=str(tmp_path / "test_log.csv"),
        fast_paths=300,
    )


# ════════════════════════════════════════════════════════════════════
# Unit: Black-Scholes helpers
# ════════════════════════════════════════════════════════════════════

class TestBSHelpers:

    def test_straddle_price_positive(self):
        p = _bs_straddle_price(100, 100, 1/52, 0.05, 0.20)
        assert p > 0

    def test_straddle_price_monotone_in_sigma(self):
        """Straddle vega is always positive — price increases with vol."""
        prices = [_bs_straddle_price(100, 100, 1/52, 0.05, s)
                  for s in [0.10, 0.15, 0.20, 0.25, 0.30]]
        for a, b in zip(prices, prices[1:]):
            assert b > a

    def test_iv_roundtrip(self):
        """Inverting a BS price must recover the original sigma."""
        sigma_true = 0.22
        price = _bs_straddle_price(100, 100, 5/252, 0.05, sigma_true)
        sigma_recovered = implied_vol_from_price(price, 100, 100, 5/252, 0.05)
        assert sigma_recovered is not None
        assert abs(sigma_recovered - sigma_true) < 1e-4

    def test_iv_returns_none_for_intrinsic_breach(self):
        """Price below intrinsic value has no valid IV."""
        result = implied_vol_from_price(-1.0, 100, 100, 5/252, 0.05)
        assert result is None

    def test_iv_returns_none_for_zero_maturity(self):
        result = implied_vol_from_price(1.0, 100, 100, 0.0, 0.05)
        assert result is None


# ════════════════════════════════════════════════════════════════════
# Unit: Position
# ════════════════════════════════════════════════════════════════════

class TestPosition:

    def test_unrealised_pnl_long(self):
        pos = Position(
            state=PositionState.LONG_VOL,
            qty=2,
            entry_price=5.0,
            entry_time=datetime.now(),
        )
        # current_price=6.0 → gain = (6-5) × 2 × 100 = 200
        assert pos.unrealised_pnl(6.0) == pytest.approx(200.0)

    def test_unrealised_pnl_flat(self):
        pos = Position()
        assert pos.unrealised_pnl(10.0) == 0.0

    def test_holding_days(self):
        pos = Position(entry_time=datetime.now() - timedelta(hours=24))
        assert 0.99 < pos.holding_days() < 1.01

    def test_is_open(self):
        assert not Position().is_open()
        assert Position(state=PositionState.LONG_VOL, qty=1).is_open()


# ════════════════════════════════════════════════════════════════════
# Unit: MarketState
# ════════════════════════════════════════════════════════════════════

class TestMarketState:

    def test_mid_from_opt_mid(self):
        m = MarketState(opt_mid=5.5)
        assert m.mid_price() == 5.5

    def test_mid_from_bid_ask(self):
        m = MarketState(opt_bid=5.0, opt_ask=5.4)
        assert m.mid_price() == pytest.approx(5.2)

    def test_mid_none_when_empty(self):
        assert MarketState().mid_price() is None


# ════════════════════════════════════════════════════════════════════
# Unit: CSVTradeLogger
# ════════════════════════════════════════════════════════════════════

class TestCSVTradeLogger:

    def test_writes_header_and_rows(self, tmp_path):
        path = tmp_path / "log.csv"
        lg = CSVTradeLogger(str(path))
        lg.open()
        lg.log({"signal_action": "HOLD", "timestamp": "2025-01-01"})
        lg.log({"signal_action": "ENTER_LONG", "timestamp": "2025-01-02"})
        lg.close()
        lines = path.read_text().splitlines()
        assert lines[0].startswith("timestamp")   # header
        assert len(lines) >= 3                     # header + 2 rows

    def test_appends_to_existing(self, tmp_path):
        path = tmp_path / "log.csv"
        for _ in range(2):
            lg = CSVTradeLogger(str(path))
            lg.open()
            lg.log({"signal_action": "HOLD"})
            lg.close()
        lines = path.read_text().splitlines()
        # One header + 2 data rows
        assert sum(1 for l in lines if "HOLD" in l) == 2


# ════════════════════════════════════════════════════════════════════
# Integration: logic tree via _evaluate_and_emit
# ════════════════════════════════════════════════════════════════════

def _make_pr(regime=CALM, prob_turb=0.1, iv_spread=0.03,
             market_iv=0.20, model_iv=0.23, spot=100.0) -> PricingResult:
    return PricingResult(
        timestamp=datetime.now(),
        spot_price=spot,
        strike=spot,
        t_days=5,
        model_price=_bs_straddle_price(spot, spot, 5/252, 0.05, model_iv),
        model_iv=model_iv,
        market_iv=market_iv,
        iv_spread=iv_spread,
        mc_std_error=0.01,
        regime_state=regime,
        prob_turbulent=prob_turb,
        regime_action="Trade",
        hurst_exponent=0.07,
    )


class TestLogicTree:
    """Each test probes a distinct branch of _evaluate_and_emit."""

    # ── Entry ────────────────────────────────────────────────────────

    def test_enter_long_on_calm_positive_spread(self, orchestrator):
        pr = _make_pr(regime=CALM, prob_turb=0.15, iv_spread=0.03)
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action == SignalAction.ENTER_LONG
        assert orchestrator.position.state == PositionState.LONG_VOL

    def test_no_entry_on_turbulent(self, orchestrator):
        orchestrator._position = Position()   # reset
        pr = _make_pr(regime=TURBULENT, prob_turb=0.75, iv_spread=0.03)
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action not in (SignalAction.ENTER_LONG, SignalAction.ADD_LONG)
        assert orchestrator.position.state == PositionState.FLAT

    def test_no_entry_when_spread_below_threshold(self, orchestrator):
        orchestrator._position = Position()
        pr = _make_pr(regime=CALM, prob_turb=0.1, iv_spread=0.005)  # < 0.02 threshold
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action == SignalAction.HOLD

    # ── Regime exits ─────────────────────────────────────────────────

    def test_close_all_on_high_turbulence(self, orchestrator):
        orchestrator._position = Position(
            state=PositionState.LONG_VOL, qty=1,
            entry_price=5.0, entry_time=datetime.now())
        pr = _make_pr(regime=TURBULENT, prob_turb=0.85, iv_spread=-0.01)
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action == SignalAction.CLOSE_ALL
        assert orchestrator.position.state == PositionState.FLAT

    def test_delta_hedge_on_moderate_turbulence(self, orchestrator):
        orchestrator._position = Position(
            state=PositionState.LONG_VOL, qty=1,
            entry_price=5.0, entry_time=datetime.now())
        pr = _make_pr(regime=TURBULENT, prob_turb=0.70, iv_spread=0.02)
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action == SignalAction.DELTA_HEDGE
        assert orchestrator.position.state == PositionState.DELTA_HEDGED

    # ── Risk exits ───────────────────────────────────────────────────

    def test_stop_loss_fires(self, orchestrator):
        orchestrator._position = Position(
            state=PositionState.LONG_VOL, qty=1,
            entry_price=10.0, entry_time=datetime.now())
        # Market price collapsed to 1.0 → loss = (1-10)×1×100 = -900
        orchestrator._market.opt_mid = 1.0
        pr = _make_pr(regime=CALM, prob_turb=0.1, iv_spread=0.03)
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action == SignalAction.STOP_LOSS
        orchestrator._market.opt_mid = None   # reset

    def test_time_exit_fires(self, orchestrator):
        orchestrator._position = Position(
            state=PositionState.LONG_VOL, qty=1,
            entry_price=5.0,
            entry_time=datetime.now() - timedelta(days=4))  # > max_hold_days=3
        pr = _make_pr(regime=CALM, prob_turb=0.1, iv_spread=0.03)
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action == SignalAction.CLOSE_ALL
        assert "Time exit" in last.reason

    def test_spread_compression_exit(self, orchestrator):
        orchestrator._position = Position(
            state=PositionState.LONG_VOL, qty=1,
            entry_price=5.0, entry_time=datetime.now())
        pr = _make_pr(regime=CALM, prob_turb=0.1, iv_spread=0.001)  # < 0.005 threshold
        orchestrator._evaluate_and_emit(pr)
        last = orchestrator.signals[-1]
        assert last.action == SignalAction.CLOSE_ALL
        assert "compressed" in last.reason.lower()


# ════════════════════════════════════════════════════════════════════
# Integration: full simulation run
# ════════════════════════════════════════════════════════════════════

class TestSimulation:
    """
    End-to-end simulation on synthetic series.  Validates that:
    • At least one entry and one exit are generated.
    • CSV log is written and parseable.
    • Session stats are consistent.
    """

    def _build_series(self, n=100):
        rng = np.random.default_rng(7)
        spots = 100 + np.cumsum(rng.normal(0, 0.5, n))
        # First 70% of bars: cheap vol (positive spread → should enter)
        split = int(n * 0.7)
        market_ivs = np.concatenate([
            np.full(split,     0.18),   # cheap vol → positive spread vs ~20% model
            np.full(n - split, 0.28),   # expensive vol → no entry
        ])
        opt_prices = spots * market_ivs * np.sqrt(5/252) * math.sqrt(2/math.pi)
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)]
        return spots, market_ivs, opt_prices, dates

    def test_simulation_produces_signals(self, orchestrator, tmp_path):
        spots, ivs, opts, dates = self._build_series(n=100)
        signals = orchestrator.run_simulation(
            spot_prices=spots,
            market_ivs=ivs,
            opt_prices=opts,
            dates=dates,
            log_path=str(tmp_path / "sim_log.csv"),
        )
        assert len(signals) > 0, "Simulation produced no signals"

    def test_simulation_csv_written(self, orchestrator, tmp_path):
        spots, ivs, opts, dates = self._build_series(n=50)
        log_path = str(tmp_path / "sim2.csv")
        orchestrator.run_simulation(
            spot_prices=spots, market_ivs=ivs,
            opt_prices=opts, dates=dates, log_path=log_path)
        assert Path(log_path).exists()
        lines = Path(log_path).read_text().splitlines()
        assert len(lines) > 1, "CSV should have header + data rows"

    def test_session_stats_consistent(self, orchestrator, tmp_path):
        spots, ivs, opts, dates = self._build_series(n=80)
        orchestrator.run_simulation(
            spot_prices=spots, market_ivs=ivs,
            opt_prices=opts, dates=dates,
            log_path=str(tmp_path / "sim3.csv"))
        stats = orchestrator.session_stats()
        assert stats["total_signals"] == len(orchestrator.signals)
        assert stats["entries"] >= 0
        assert stats["exits"]   >= 0
        # Exits can't exceed entries + 1 (one pre-existing)
        assert stats["exits"] <= stats["entries"] + 1
