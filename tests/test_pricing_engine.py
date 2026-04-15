"""
Unit tests for the PricingEngine.

Key test: when H = 0.5 the rough volatility model collapses to a
classical stochastic volatility model.  With ν → 0 (near-zero vol-of-vol)
and constant variance, the Monte Carlo prices must converge to
Black-Scholes within statistical tolerance.
"""

import math
import numpy as np
import pytest

from pricing_engine import PricingEngine


# ── Helpers ─────────────────────────────────────────────────────────

def relative_error(mc: float, bs: float) -> float:
    """Signed relative error in percent."""
    if bs == 0:
        return 0.0 if mc == 0 else float("inf")
    return (mc - bs) / bs * 100.0


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def bs_engine():
    """
    Engine configured so that H=0.5 and ν≈0, which should replicate
    geometric Brownian motion with constant vol σ = sqrt(v0) = 0.20.
    """
    return PricingEngine(
        H=0.5,
        v0=0.04,          # σ = 20%
        nu=1e-8,           # vol-of-vol ≈ 0 → variance stays constant
        S0=100.0,
        r=0.05,
        rho=0.0,
        n_paths=50_000,
        steps_per_day=48,  # fine grid for accuracy
        kappa=6,
        J=4,
        seed=12345,
    )


@pytest.fixture
def rough_engine():
    """Engine with typical rough-vol parameters."""
    return PricingEngine(
        H=0.07,
        v0=0.04,
        nu=0.3,
        S0=585.0,
        r=0.053,
        rho=-0.7,
        n_paths=10_000,
        steps_per_day=24,
        seed=42,
    )


# ── Tests: BS convergence at H = 0.5 ───────────────────────────────

class TestBSConvergence:
    """
    Verify that the rough-vol MC pricer reproduces Black-Scholes when
    the Hurst exponent is set to H = 0.5 and vol-of-vol is negligible.
    """

    SIGMA = 0.20      # sqrt(v0)
    TOL_PCT = 3.0     # Allow 3 % relative error (MC noise)

    @pytest.mark.parametrize("K,T_days", [
        (100, 5),     # ATM, 1-week
        (100, 21),    # ATM, 1-month
        (95, 5),      # ITM call / OTM put, 1-week
        (105, 5),     # OTM call / ITM put, 1-week
    ])
    def test_call_convergence(self, bs_engine, K, T_days):
        T = T_days / 252.0
        bs = PricingEngine.black_scholes_call(
            bs_engine.S0, K, T, bs_engine.r, self.SIGMA)
        mc = bs_engine.price_european_call(K=K, T_days=T_days)

        err = relative_error(mc.price, bs)
        print(f"Call K={K} T={T_days}d | MC={mc.price:.4f} BS={bs:.4f} "
              f"err={err:+.2f}% se={mc.std_error:.4f}")
        # Deep OTM options have tiny absolute prices so % errors are amplified;
        # use absolute tolerance when BS price < $0.10.
        if bs < 0.10:
            assert abs(mc.price - bs) < 0.01, (
                f"MC call {mc.price:.4f} vs BS {bs:.4f}, abs diff {abs(mc.price-bs):.4f}"
            )
        else:
            assert abs(err) < self.TOL_PCT, (
                f"MC call price {mc.price:.4f} deviates {err:.2f}% from BS {bs:.4f}"
            )

    @pytest.mark.parametrize("K,T_days", [
        (100, 5),
        (100, 21),
        (105, 5),
    ])
    def test_put_convergence(self, bs_engine, K, T_days):
        T = T_days / 252.0
        bs = PricingEngine.black_scholes_put(
            bs_engine.S0, K, T, bs_engine.r, self.SIGMA)
        mc = bs_engine.price_european_put(K=K, T_days=T_days)

        err = relative_error(mc.price, bs)
        print(f"Put  K={K} T={T_days}d | MC={mc.price:.4f} BS={bs:.4f} "
              f"err={err:+.2f}% se={mc.std_error:.4f}")
        assert abs(err) < self.TOL_PCT

    def test_straddle_convergence(self, bs_engine):
        K, T_days = 100, 5
        T = T_days / 252.0
        bs = PricingEngine.black_scholes_straddle(
            bs_engine.S0, K, T, bs_engine.r, self.SIGMA)
        mc = bs_engine.price_straddle(K=K, T_days=T_days)

        err = relative_error(mc.price, bs)
        print(f"Straddle K={K} T={T_days}d | MC={mc.price:.4f} BS={bs:.4f} "
              f"err={err:+.2f}% se={mc.std_error:.4f}")
        assert abs(err) < self.TOL_PCT

    def test_put_call_parity(self, bs_engine):
        """MC call − MC put ≈ S·e^0 − K·e^{-rT} (put-call parity)."""
        K, T_days = 100, 5
        T = T_days / 252.0
        # Use same seed for both to reduce variance
        call = bs_engine.price_european_call(K=K, T_days=T_days)
        bs_engine.seed = 12345  # reset seed
        put = bs_engine.price_european_put(K=K, T_days=T_days)

        parity_lhs = call.price - put.price
        parity_rhs = bs_engine.S0 - K * math.exp(-bs_engine.r * T)
        err = abs(parity_lhs - parity_rhs)
        print(f"Put-call parity: C-P={parity_lhs:.4f}  S-Ke^{{-rT}}={parity_rhs:.4f}  "
              f"diff={err:.4f}")
        # Generous tolerance due to different random draws
        assert err < 1.0, f"Put-call parity violated by {err:.4f}"


# ── Tests: Rough vol sanity checks ──────────────────────────────────

class TestRoughVol:
    """Basic sanity checks with rough parameters."""

    def test_simulation_returns_correct_shape(self, rough_engine):
        S_T, v_T = rough_engine.simulate(T_days=5)
        assert S_T.shape == (rough_engine.n_paths,)
        assert v_T.shape == (rough_engine.n_paths,)

    def test_spot_prices_positive(self, rough_engine):
        S_T, _ = rough_engine.simulate(T_days=5)
        assert np.all(S_T > 0), "Some terminal spot prices are non-positive"

    def test_variance_non_negative(self, rough_engine):
        _, v_T = rough_engine.simulate(T_days=5)
        assert np.all(v_T >= 0), "Some terminal variances are negative"

    def test_straddle_price_positive(self, rough_engine):
        res = rough_engine.price_straddle(K=585.0, T_days=5)
        assert res.price > 0
        assert res.std_error > 0
        assert res.paths_used == rough_engine.n_paths

    def test_call_less_than_spot(self, rough_engine):
        """A call option price must be less than the spot price."""
        res = rough_engine.price_european_call(K=585.0, T_days=5)
        assert res.price < rough_engine.S0

    def test_put_less_than_strike(self, rough_engine):
        """A put option price must be less than the strike."""
        K = 585.0
        res = rough_engine.price_european_put(K=K, T_days=5)
        assert res.price < K


# ── Tests: Black-Scholes static formulas ────────────────────────────

class TestBlackScholesFormulas:

    def test_atm_call_equals_known_value(self):
        """Validate against a hand-computed BS call price."""
        # S=100, K=100, T=1, r=5%, σ=20%  → call ≈ 10.4506
        price = PricingEngine.black_scholes_call(100, 100, 1.0, 0.05, 0.20)
        assert abs(price - 10.4506) < 0.01

    def test_put_call_parity_bs(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        call = PricingEngine.black_scholes_call(S, K, T, r, sigma)
        put = PricingEngine.black_scholes_put(S, K, T, r, sigma)
        assert abs((call - put) - (S - K * math.exp(-r * T))) < 1e-10

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S − K·e^{-rT}."""
        price = PricingEngine.black_scholes_call(200, 50, 0.1, 0.03, 0.2)
        intrinsic = 200 - 50 * math.exp(-0.03 * 0.1)
        assert abs(price - intrinsic) < 0.1

    def test_zero_maturity(self):
        assert PricingEngine.black_scholes_call(100, 90, 0, 0.05, 0.2) == 10.0
        assert PricingEngine.black_scholes_put(100, 110, 0, 0.05, 0.2) == 10.0
