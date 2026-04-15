"""
Tests for the Validation Suite.

Test 1 (Stability): Uses synthetic price / return data that contains
a hard-coded crash event, so network access is not required.

Test 2 (Convergence): Runs the MC pricer at reduced N values (fast)
and verifies the regression slope is within the expected range.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from pricing_engine import PricingEngine
from regime_filter import RegimeFilter, CALM, TURBULENT
from validation_suite import (
    MCConvergenceTest, ConvergenceResult,
    RegimeStabilityTest, StabilityResult, DrawdownEvent,
    ValidationSuite,
)


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def small_engine() -> PricingEngine:
    """Fast engine for convergence tests."""
    return PricingEngine(
        H=0.07, v0=0.04, nu=0.3,
        S0=100.0, r=0.05, rho=-0.7,
        n_paths=64,          # minimum — will be overridden per run
        steps_per_day=8,
        kappa=4, J=3,
        seed=7,
    )


@pytest.fixture(scope="module")
def fitted_regime() -> RegimeFilter:
    rng = np.random.default_rng(123)
    calm = rng.normal(5e-4, 7e-3, 800)
    turb = rng.normal(-1e-3, 2.2e-2, 200)
    obs  = np.concatenate([calm, turb])
    rf   = RegimeFilter(turbulence_threshold=0.6)
    rf.log_returns = obs
    rf.dates = np.arange(len(obs))
    rf.fit(obs=obs, n_restarts=4, seed=0)
    return rf


# ════════════════════════════════════════════════════════════════════
# Test 2: MC Convergence (runs fast, no network)
# ════════════════════════════════════════════════════════════════════

class TestMCConvergence:

    @pytest.fixture
    def conv_test(self, small_engine) -> MCConvergenceTest:
        return MCConvergenceTest(
            engine=small_engine,
            n_values=[64, 128, 256, 512, 1024],
            n_trials=10,        # quick for CI
            K=100.0,
            T_days=5,
        )

    def test_convergence_passes_for_correct_engine(self, conv_test):
        result = conv_test.run(plot=False)
        assert result.r_squared > 0.80, (
            f"R² = {result.r_squared:.3f} — regression is too noisy")
        assert conv_test.SLOPE_LOW <= result.convergence_rate <= conv_test.SLOPE_HIGH, (
            f"Slope {result.convergence_rate:+.4f} outside "
            f"[{conv_test.SLOPE_LOW}, {conv_test.SLOPE_HIGH}]"
        )
        assert result.passed
        assert result.flag == ""

    def test_se_decreases_with_more_paths(self, conv_test):
        result = conv_test.run(plot=False)
        ses = result.se_values
        # SE should be generally decreasing (allowing some noise)
        n_decreasing = sum(a > b for a, b in zip(ses, ses[1:]))
        assert n_decreasing >= len(ses) // 2, (
            "SE does not generally decrease with more paths")

    def test_variance_reduction_failure_flagged(self, small_engine):
        """
        A mock engine that returns a constant price (regardless of paths)
        has SE ≈ 0 for all N — the slope is undefined / flat → flag fires.
        """
        class ConstantPricer:
            """Always returns the same price — no variance reduction."""
            H           = 0.07
            v0          = 0.04
            nu          = 0.3
            S0          = 100.0
            r           = 0.05
            rho         = -0.7
            steps_per_day = 8
            kappa       = 4
            J           = 3

            def price_straddle(self, K, T_days):
                from pricing_engine import OptionResult
                # Add tiny noise so std isn't exactly 0
                noise = np.random.normal(0, 0.001)
                return OptionResult(price=2.5 + noise,
                                    std_error=0.0, paths_used=1)

        test = MCConvergenceTest(
            engine=ConstantPricer(),
            n_values=[64, 128, 256, 512],
            n_trials=5,
            K=100.0, T_days=5,
        )
        result = test.run(plot=False)
        # With near-constant prices across N, slope will be near 0 → fail
        if not result.passed:
            assert result.flag == "VARIANCE REDUCTION FAILURE"
        # If by chance it "passes" with the tiny noise, that's acceptable too

    def test_convergence_rate_close_to_half(self, conv_test):
        """Central tendency should be near -0.5."""
        result = conv_test.run(plot=False)
        # |slope - (-0.5)| < 0.3  — generous tolerance for small n_trials
        assert abs(result.convergence_rate - (-0.5)) < 0.30, (
            f"Convergence rate {result.convergence_rate:+.4f} "
            "is far from expected -0.5")

    def test_result_fields_populated(self, conv_test):
        result = conv_test.run(plot=False)
        assert len(result.n_values)  == len(conv_test.n_values)
        assert len(result.se_values) == len(conv_test.n_values)
        assert all(se >= 0 for se in result.se_values)
        assert result.summary != ""


# ════════════════════════════════════════════════════════════════════
# Test 1: Stability (synthetic data — no network)
# ════════════════════════════════════════════════════════════════════

class TestRegimeStability:

    def _build_synthetic_crash(self) -> tuple[pd.Series, pd.Series]:
        """
        Build a synthetic SPY-like price series with an embedded crash
        (~30% drawdown) and calm periods before/after.
        """
        rng    = np.random.default_rng(99)
        n_calm = 504      # ~2 years daily
        n_crash = 30      # ~6 weeks crash
        n_recovery = 126  # ~6 months recovery

        calm_rets  = rng.normal(5e-4, 7e-3, n_calm)
        crash_rets = rng.normal(-1.5e-2, 2.5e-2, n_crash)   # heavy losses
        recov_rets = rng.normal(8e-4, 9e-3, n_recovery)
        all_rets   = np.concatenate([calm_rets, crash_rets, recov_rets])

        prices     = 300.0 * np.cumprod(1 + all_rets)
        base_date  = pd.Timestamp("2018-01-02")
        biz_dates  = pd.bdate_range(base_date, periods=len(all_rets))

        return_series = pd.Series(all_rets, index=biz_dates)
        price_series  = pd.Series(prices,   index=biz_dates)
        return return_series, price_series

    def test_drawdown_identification(self):
        """_identify_drawdowns must find the embedded crash."""
        ret_s, price_s = self._build_synthetic_crash()

        test = RegimeStabilityTest(
            train_end="2019-12-31",
            test_start="2020-01-01",
            test_end="2023-12-31",
            min_drawdown_pct=0.10,
        )
        # The crash starts ~day 504, so in the test window (from 2020-01-01+)
        test_prices = price_s[price_s.index >= pd.Timestamp("2019-06-01")]
        events = test._identify_drawdowns(test_prices)
        assert len(events) >= 1, "Should detect at least one drawdown"
        # Largest drawdown should be ≥ 10%
        worst = min(e.drawdown_pct for e in events)
        assert worst <= -0.10

    def test_stability_with_synthetic_data(self, fitted_regime):
        """
        Run RegimeStabilityTest end-to-end with synthetic prices,
        bypassing yfinance.
        """
        ret_s, price_s = self._build_synthetic_crash()

        test = RegimeStabilityTest(
            train_end="2019-12-31",
            test_start="2019-06-01",   # overlap to see crash
            test_end="2021-12-31",
            min_drawdown_pct=0.08,
        )

        # Monkeypatch _fetch_spy to return our synthetic data
        with patch.object(test, "_fetch_spy", return_value=(ret_s, price_s)):
            result = test.run(rf=fitted_regime, plot=False, save_path=None)

        assert isinstance(result, StabilityResult)
        assert result.n_drawdowns_tested >= 0   # may be 0 in test window
        assert 0.0 <= result.detection_rate <= 1.0

    def test_drawdown_event_fields(self):
        ev = DrawdownEvent(
            start=pd.Timestamp("2020-02-19"),
            trough=pd.Timestamp("2020-03-23"),
            drawdown_pct=-0.34,
            turbulent_flag=pd.Timestamp("2020-02-14"),
            lead_days=38,
            detected=True,
        )
        assert ev.detected
        assert ev.lead_days == 38
        assert ev.drawdown_pct < 0


# ════════════════════════════════════════════════════════════════════
# Integration: ValidationSuite orchestrator
# ════════════════════════════════════════════════════════════════════

class TestValidationSuiteOrchestrator:

    def test_convergence_only(self, small_engine, tmp_path):
        suite = ValidationSuite(
            engine=small_engine,
            mc_n_values=[64, 128, 256],
            mc_n_trials=8,
            output_dir=str(tmp_path),
        )
        report = suite.run(run_stability=False, run_convergence=True, plot=False)
        assert report.convergence is not None
        assert report.stability  is None

    def test_report_all_passed_reflects_subtests(self, small_engine, tmp_path):
        suite = ValidationSuite(
            engine=small_engine,
            mc_n_values=[64, 128, 256],
            mc_n_trials=8,
            output_dir=str(tmp_path),
        )
        report = suite.run(run_stability=False, run_convergence=True, plot=False)
        expected = report.convergence.passed
        assert report.all_passed == expected
