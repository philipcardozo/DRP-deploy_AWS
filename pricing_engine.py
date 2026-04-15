"""
Layer 1 — Mathematical Core: Rough Volatility Pricing Engine.

Implements the discrete Volterra process for spot variance:

    v_t = v_0 + (1/Γ(H+1/2)) ∫₀ᵗ (t−s)^{H−1/2} λ(v_s) dW_s

Using the Hybrid Scheme of Bennedsen, Lunde & Pakkanen (2017) which splits
the fractional kernel into:
    • A *singular near-field* component (first κ steps) handled by exact
      power-law weights.
    • A *smooth far-field* tail approximated with an exponential sum so
      the full convolution drops from O(N²) to O(N·κ + N·J).

All hot-path numerics are JIT-compiled via Numba for sub-millisecond
per-path simulation.

References
----------
[1] Bennedsen, M., Lunde, A. & Pakkanen, M. (2017).
    "Hybrid scheme for Brownian semistationary processes."
    Finance & Stochastics, 21(4), 931–965.
[2] Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018).
    "Volatility is rough." Quantitative Finance, 18(6), 933–949.
"""

from __future__ import annotations

import math
import numpy as np
import numba as nb
from scipy.stats import norm
from scipy.special import gamma as gamma_fn
from dataclasses import dataclass


# ════════════════════════════════════════════════════════════════════
# Numba-accelerated kernel helpers
# ════════════════════════════════════════════════════════════════════

@nb.njit(cache=True, fastmath=True)
def _kernel_weights(H: float, kappa: int, dt: float) -> np.ndarray:
    """
    Precompute the exact convolution weights for the near-field part of
    the fractional kernel K(t) = t^{H - 1/2} / Γ(H + 1/2).

    For lag j = 0, …, κ−1 the weight is the integral of the kernel over
    [(j)·dt, (j+1)·dt] so that no singularity is evaluated point-wise.

    w_j = ∫_{j·dt}^{(j+1)·dt} s^{H-1/2} ds  /  Γ(H + 1/2)
        = [ ((j+1)·dt)^{H+1/2} − (j·dt)^{H+1/2} ] / [(H+1/2)·Γ(H+1/2)]
    """
    alpha = H - 0.5  # exponent in kernel
    beta = H + 0.5   # integration exponent
    # Γ(H + 1/2) — Numba doesn't expose scipy.special, so use Stirling
    # for the denominator.  We'll pass it in from Python instead.
    weights = np.empty(kappa, dtype=np.float64)
    for j in range(kappa):
        upper = ((j + 1) * dt) ** beta
        lower = (j * dt) ** beta if j > 0 else 0.0
        weights[j] = (upper - lower) / beta
    return weights


@nb.njit(cache=True, fastmath=True)
def _exp_sum_coefficients(H: float, kappa: int, N: int, dt: float,
                          J: int) -> tuple:
    """
    Fit an exponential sum  Σ_{j=1}^{J} c_j · exp(−γ_j · t)  to the
    far-field tail of the fractional kernel for t > κ·dt.

    We use a geometric grid of decay rates and least-squares projection
    (simplified Beylkin–Monzón style).

    Returns (c, gamma) each of shape (J,).
    """
    # Sample points in the far-field region
    n_pts = min(500, N - kappa)
    if n_pts <= 0:
        return np.zeros(J, dtype=np.float64), np.ones(J, dtype=np.float64)

    t_pts = np.empty(n_pts, dtype=np.float64)
    k_pts = np.empty(n_pts, dtype=np.float64)
    alpha = H - 0.5
    for i in range(n_pts):
        t_pts[i] = (kappa + 1 + i) * dt
        k_pts[i] = t_pts[i] ** alpha

    # Geometric grid of decay rates spanning the far-field timescale
    gamma_min = 1.0 / ((N - kappa) * dt + 1e-12)
    gamma_max = 1.0 / (kappa * dt + 1e-12)
    gammas = np.empty(J, dtype=np.float64)
    ratio = (gamma_max / (gamma_min + 1e-30)) ** (1.0 / max(J - 1, 1))
    for j in range(J):
        gammas[j] = gamma_min * (ratio ** j)

    # Build design matrix A[i, j] = exp(−γ_j · t_i) and solve via
    # normal equations  AᵀA c = Aᵀ k   (Numba-friendly)
    A = np.empty((n_pts, J), dtype=np.float64)
    for i in range(n_pts):
        for j in range(J):
            A[i, j] = math.exp(-gammas[j] * t_pts[i])

    AtA = np.zeros((J, J), dtype=np.float64)
    Atk = np.zeros(J, dtype=np.float64)
    for i in range(n_pts):
        for ja in range(J):
            Atk[ja] += A[i, ja] * k_pts[i]
            for jb in range(J):
                AtA[ja, jb] += A[i, ja] * A[i, jb]

    # Tikhonov regularisation
    for j in range(J):
        AtA[j, j] += 1e-8

    # Solve via Cholesky-ish forward/back substitution (simple Gaussian elim)
    # Since J is small (4–8), this is fine.
    coeffs = np.zeros(J, dtype=np.float64)
    # Gaussian elimination with partial pivoting
    Ab = np.empty((J, J + 1), dtype=np.float64)
    for i in range(J):
        for j in range(J):
            Ab[i, j] = AtA[i, j]
        Ab[i, J] = Atk[i]

    for col in range(J):
        # Pivot
        max_row = col
        max_val = abs(Ab[col, col])
        for row in range(col + 1, J):
            if abs(Ab[row, col]) > max_val:
                max_val = abs(Ab[row, col])
                max_row = row
        if max_row != col:
            for k in range(J + 1):
                tmp = Ab[col, k]
                Ab[col, k] = Ab[max_row, k]
                Ab[max_row, k] = tmp
        # Eliminate
        for row in range(col + 1, J):
            factor = Ab[row, col] / (Ab[col, col] + 1e-30)
            for k in range(col, J + 1):
                Ab[row, k] -= factor * Ab[col, k]

    # Back substitution
    for i in range(J - 1, -1, -1):
        s = Ab[i, J]
        for j in range(i + 1, J):
            s -= Ab[i, j] * coeffs[j]
        coeffs[i] = s / (Ab[i, i] + 1e-30)

    return coeffs, gammas


@nb.njit(cache=True, parallel=True, fastmath=True)
def _simulate_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    v0: float,
    nu: float,           # vol-of-vol
    H: float,
    S0: float,
    r: float,
    rho: float,           # spot-vol correlation
    near_weights: np.ndarray,   # (kappa,)
    exp_coeffs: np.ndarray,     # (J,)
    exp_gammas: np.ndarray,     # (J,)
    kappa: int,
    inv_gamma_Hphalf: float,
    seed: int,
) -> tuple:
    """
    Simulate `n_paths` joint (S, v) paths under the rough Heston /
    rough Bergomi-style dynamics using the Hybrid Scheme.

    Returns
    -------
    S_T : np.ndarray, shape (n_paths,)   — terminal spot prices
    v_paths : np.ndarray, shape (n_paths, n_steps+1) — variance paths
    """
    sqrt_dt = math.sqrt(dt)
    J = exp_coeffs.shape[0]

    S_T = np.empty(n_paths, dtype=np.float64)
    v_terminal = np.empty(n_paths, dtype=np.float64)

    for p in nb.prange(n_paths):
        # Per-path RNG (deterministic per path for reproducibility)
        np.random.seed(seed + p)

        # State variables
        v = v0
        log_S = math.log(S0)

        # Near-field circular buffer for dW_v history
        dW_buf = np.zeros(kappa, dtype=np.float64)
        buf_idx = 0

        # Far-field exponential state variables  x_j
        x = np.zeros(J, dtype=np.float64)

        for i in range(n_steps):
            # Correlated Brownian increments
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()
            dW_S = sqrt_dt * z1
            dW_v = sqrt_dt * (rho * z1 + math.sqrt(1.0 - rho * rho) * z2)

            # ── Variance diffusion coefficient ──
            v_pos = max(v, 1e-10)
            sigma_v = nu * math.sqrt(v_pos)   # λ(v) = ν·√v

            # ── Near-field convolution ──
            near_sum = near_weights[0] * sigma_v * dW_v  # lag-0 (singular)
            # Add contributions from past increments
            for lag in range(1, min(i + 1, kappa)):
                hist_idx = (buf_idx - lag) % kappa
                near_sum += near_weights[lag] * dW_buf[hist_idx]

            # ── Far-field exponential update ──
            far_sum = 0.0
            for j in range(J):
                x[j] = math.exp(-exp_gammas[j] * dt) * x[j] + \
                        exp_coeffs[j] * sigma_v * dW_v
                far_sum += x[j]

            # ── Update variance ──
            v_new = v0 + inv_gamma_Hphalf * (near_sum + far_sum)
            v = max(v_new, 0.0)  # Absorbing boundary at 0

            # ── Update spot (log-Euler) ──
            vol = math.sqrt(max(v, 0.0))
            log_S += (r - 0.5 * v) * dt + vol * dW_S

            # Store dW_v (already includes σ_v scaling) in circular buffer
            dW_buf[buf_idx] = sigma_v * dW_v
            buf_idx = (buf_idx + 1) % kappa

        S_T[p] = math.exp(log_S)
        v_terminal[p] = v

    return S_T, v_terminal


# ════════════════════════════════════════════════════════════════════
# Public interface
# ════════════════════════════════════════════════════════════════════

@dataclass
class OptionResult:
    """Container for Monte Carlo pricing output."""
    price: float
    std_error: float
    paths_used: int
    iv_approx: float | None = None


class PricingEngine:
    """
    Rough-volatility Monte Carlo pricing engine.

    Parameters
    ----------
    H : float
        Hurst exponent. H < 0.5 → rough, H = 0.5 → classical Brownian.
    v0 : float
        Initial spot variance.
    nu : float
        Vol-of-vol parameter (scaling in the diffusion coefficient).
    S0 : float
        Current spot price.
    r : float
        Risk-free rate (continuous compounding).
    rho : float
        Instantaneous correlation between spot and variance Brownians.
    n_paths : int
        Number of Monte Carlo simulation paths.
    steps_per_day : int
        Temporal resolution (higher → better accuracy, slower).
    kappa : int
        Near-field window (number of lags for exact kernel weights).
    J : int
        Number of exponentials in the far-field approximation.
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(
        self,
        H: float = 0.07,
        v0: float = 0.04,
        nu: float = 0.3,
        S0: float = 585.0,
        r: float = 0.053,
        rho: float = -0.7,
        n_paths: int = 10_000,
        steps_per_day: int = 24,
        kappa: int = 12,
        J: int = 6,
        seed: int = 42,
    ):
        self.H = H
        self.v0 = v0
        self.nu = nu
        self.S0 = S0
        self.r = r
        self.rho = rho
        self.n_paths = n_paths
        self.steps_per_day = steps_per_day
        self.kappa = kappa
        self.J = J
        self.seed = seed

        # Precompute Γ(H + 1/2)
        self._gamma_Hphalf = gamma_fn(H + 0.5)
        self._inv_gamma_Hphalf = 1.0 / self._gamma_Hphalf

    # ── Kernel pre-computation ──────────────────────────────────────

    def _build_kernel(self, T_days: int) -> tuple:
        """Build near-field weights and far-field exponential coefficients."""
        n_steps = T_days * self.steps_per_day
        dt = (T_days / 252.0) / n_steps  # annualised dt

        near_w = _kernel_weights(self.H, self.kappa, dt)
        # Scale by 1/Γ(H+1/2) baked into the weights
        near_w = near_w * self._inv_gamma_Hphalf

        exp_c, exp_g = _exp_sum_coefficients(
            self.H, self.kappa, n_steps, dt, self.J
        )

        return n_steps, dt, near_w, exp_c, exp_g

    # ── Simulation ──────────────────────────────────────────────────

    def simulate(self, T_days: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the full Monte Carlo simulation.

        Parameters
        ----------
        T_days : int
            Option maturity in trading days (5 ≈ 1 week).

        Returns
        -------
        S_T : ndarray (n_paths,) — terminal spot prices.
        v_T : ndarray (n_paths,) — terminal variance values.
        """
        n_steps, dt, near_w, exp_c, exp_g = self._build_kernel(T_days)

        S_T, v_T = _simulate_paths(
            n_paths=self.n_paths,
            n_steps=n_steps,
            dt=dt,
            v0=self.v0,
            nu=self.nu,
            H=self.H,
            S0=self.S0,
            r=self.r,
            rho=self.rho,
            near_weights=near_w,
            exp_coeffs=exp_c,
            exp_gammas=exp_g,
            kappa=self.kappa,
            inv_gamma_Hphalf=self._inv_gamma_Hphalf,
            seed=self.seed,
        )
        return S_T, v_T

    # ── Pricing ─────────────────────────────────────────────────────

    def price_european_call(self, K: float, T_days: int = 5) -> OptionResult:
        """Price a European call via Monte Carlo."""
        S_T, _ = self.simulate(T_days)
        T_years = T_days / 252.0
        discount = math.exp(-self.r * T_years)
        payoffs = np.maximum(S_T - K, 0.0)
        price = discount * np.mean(payoffs)
        std_err = discount * np.std(payoffs) / math.sqrt(self.n_paths)
        return OptionResult(price=price, std_error=std_err,
                            paths_used=self.n_paths)

    def price_european_put(self, K: float, T_days: int = 5) -> OptionResult:
        """Price a European put via Monte Carlo."""
        S_T, _ = self.simulate(T_days)
        T_years = T_days / 252.0
        discount = math.exp(-self.r * T_years)
        payoffs = np.maximum(K - S_T, 0.0)
        price = discount * np.mean(payoffs)
        std_err = discount * np.std(payoffs) / math.sqrt(self.n_paths)
        return OptionResult(price=price, std_error=std_err,
                            paths_used=self.n_paths)

    def price_straddle(self, K: float, T_days: int = 5) -> OptionResult:
        """
        Price an ATM straddle (long call + long put at the same strike).
        Single simulation pass — both payoffs from the same paths.
        """
        S_T, _ = self.simulate(T_days)
        T_years = T_days / 252.0
        discount = math.exp(-self.r * T_years)
        payoffs = np.abs(S_T - K)  # |S_T − K| = call + put payoff
        price = discount * np.mean(payoffs)
        std_err = discount * np.std(payoffs) / math.sqrt(self.n_paths)
        return OptionResult(price=price, std_error=std_err,
                            paths_used=self.n_paths)

    # ── Black-Scholes reference (for validation) ────────────────────

    @staticmethod
    def black_scholes_call(S: float, K: float, T: float,
                           r: float, sigma: float) -> float:
        """Analytical Black-Scholes European call price."""
        if T <= 0:
            return max(S - K, 0.0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def black_scholes_put(S: float, K: float, T: float,
                          r: float, sigma: float) -> float:
        """Analytical Black-Scholes European put price."""
        if T <= 0:
            return max(K - S, 0.0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def black_scholes_straddle(S: float, K: float, T: float,
                               r: float, sigma: float) -> float:
        """Analytical BS straddle price (call + put)."""
        return (PricingEngine.black_scholes_call(S, K, T, r, sigma) +
                PricingEngine.black_scholes_put(S, K, T, r, sigma))
