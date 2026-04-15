"""
Layer 2 — Risk Manager: Gaussian Hidden Markov Model Regime Filter.

Implements a two-state G-HMM fitted via Expectation-Maximization
(Baum-Welch) on 5 years of daily SPY log-returns.  Real-time Viterbi
decoding and forward-filtered probabilities act as a "traffic light"
that gates the strategy engine.

State semantics
---------------
State 0 ("Calm")      : Low variance, positive drift  → momentum.
State 1 ("Turbulent") : High variance, negative drift → mean-reversion.

Design goals beyond the baseline
---------------------------------
1. Fully vectorised NumPy forward / backward passes — no Python loops
   over time (critical for calibrating on 1 200+ observations fast).
2. Multiple random EM restarts — picks the global maximum-likelihood
   solution instead of the first local optimum found.
3. Online `update_signal()` — propagates a single new observation
   through the filter without re-running the full history.
4. Model diagnostics: AIC, BIC, expected regime durations, transition
   matrix summary.
5. Five-panel Regime Map that visualises everything the model learns.

References
----------
Rabiner, L. (1989). "A tutorial on hidden Markov models and selected
    applications in speech recognition." Proc. IEEE, 77(2), 257–286.
Hamilton, J.D. (1989). "A new approach to the economic analysis of
    nonstationary time series." Econometrica, 57(2), 357–384.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import norm as sp_norm


# ════════════════════════════════════════════════════════════════════
# Constants & data containers
# ════════════════════════════════════════════════════════════════════

CALM, TURBULENT = 0, 1
_STATE_COLOR  = {CALM: "#2196F3", TURBULENT: "#F44336"}   # blue / red
_STATE_LABEL  = {CALM: "Calm",    TURBULENT: "Turbulent"}


@dataclass
class RegimeSignal:
    """Actionable output for a single time step."""
    date: dt.date
    log_return: float
    state: int              # 0 = Calm, 1 = Turbulent
    prob_calm: float
    prob_turbulent: float
    action: str             # "Trade" | "Delta Hedge" | "Halt Trading"


@dataclass
class HMMParams:
    """Fitted HMM parameter set."""
    pi: np.ndarray          # (K,)  initial-state distribution
    A: np.ndarray           # (K,K) row-stochastic transition matrix
    mu: np.ndarray          # (K,)  emission means
    sigma: np.ndarray       # (K,)  emission standard deviations

    def __repr__(self) -> str:
        lines = [
            "HMMParams:",
            f"  π  = {self.pi}",
            f"  μ  = {self.mu}",
            f"  σ  = {self.sigma}",
            f"  A  =\n{self.A}",
        ]
        return "\n".join(lines)


class ModelDiagnostics(NamedTuple):
    log_likelihood: float
    aic: float
    bic: float
    expected_duration_calm: float       # trading days
    expected_duration_turbulent: float  # trading days
    regime_fraction_calm: float
    regime_fraction_turbulent: float


# ════════════════════════════════════════════════════════════════════
# Vectorised HMM numerics
# ════════════════════════════════════════════════════════════════════

def _log_emission(obs: np.ndarray, mu: np.ndarray,
                  sigma: np.ndarray) -> np.ndarray:
    """
    Log-emission matrix  B[t, k] = log N(obs_t | μ_k, σ_k²).
    Shape (T, K) — fully vectorised.
    """
    # obs[:, None] broadcasts over K states
    return sp_norm.logpdf(obs[:, None], loc=mu[None, :], scale=sigma[None, :])


def _forward_scaled(log_B: np.ndarray, pi: np.ndarray,
                    A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled forward algorithm.

    Returns
    -------
    alpha : (T, K) — scaled forward variables
    log_scale : (T,) — log of the per-step scale factors
                 ⟹  log p(obs) = log_scale.sum()
    """
    T, K = log_B.shape
    alpha = np.empty((T, K))
    log_scale = np.empty(T)

    # t = 0
    a0 = pi * np.exp(log_B[0])
    s0 = a0.sum()
    log_scale[0] = np.log(s0 + 1e-300)
    alpha[0] = a0 / (s0 + 1e-300)

    # t = 1 … T-1  (single matrix-vector product per step)
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * np.exp(log_B[t])
        s = a.sum()
        log_scale[t] = np.log(s + 1e-300)
        alpha[t] = a / (s + 1e-300)

    return alpha, log_scale


def _backward_scaled(log_B: np.ndarray, A: np.ndarray,
                     log_scale: np.ndarray) -> np.ndarray:
    """
    Scaled backward algorithm.

    Returns
    -------
    beta : (T, K) — scaled backward variables, normalised by the
           same per-step scales used in the forward pass.
    """
    T, K = log_B.shape
    beta = np.empty((T, K))
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        b = A @ (np.exp(log_B[t + 1]) * beta[t + 1])
        beta[t] = b / (np.exp(log_scale[t + 1]) + 1e-300)

    return beta


def _e_step(obs: np.ndarray, pi: np.ndarray, A: np.ndarray,
            mu: np.ndarray, sigma: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Full E-step: compute γ (smoothed state probs) and ξ (transition
    probs) using the scaled forward-backward recursion.

    Returns
    -------
    gamma  : (T, K)
    xi_sum : (K, K) — sum over t of ξ(t, j, k), avoids storing (T, K, K)
    alpha  : (T, K) — kept for incremental online inference
    ll     : float  — log p(obs | θ)
    """
    log_B = _log_emission(obs, mu, sigma)
    alpha, log_scale = _forward_scaled(log_B, pi, A)
    beta = _backward_scaled(log_B, A, log_scale)

    # γ
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

    # ξ summed over t — shape (K, K)  — vectorised
    # ξ(t, j, k) ∝ α_t(j) · A_jk · B_k(x_{t+1}) · β_{t+1}(k)
    T = len(obs)
    B_next = np.exp(log_B[1:])           # (T-1, K)
    # For two states this is small; a batched outer product is fine.
    xi_sum = np.einsum(
        "tj,jk,tk,tk->jk",
        alpha[:-1],          # (T-1, K)
        A,                   # (K, K)
        B_next,              # (T-1, K)
        beta[1:],            # (T-1, K)
    )
    xi_sum /= xi_sum.sum() + 1e-300

    ll = log_scale.sum()
    return gamma, xi_sum, alpha, ll


def _m_step(obs: np.ndarray, gamma: np.ndarray,
            xi_sum: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    """M-step: closed-form re-estimation of (π, A, μ, σ)."""
    K = gamma.shape[1]
    pi = gamma[0] / (gamma[0].sum() + 1e-300)

    # Transition matrix (row-normalised)
    A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)

    # Emission parameters — weighted mean & std
    w = gamma.sum(axis=0) + 1e-300          # (K,)
    mu = gamma.T @ obs / w                  # (K,)
    diff = obs[:, None] - mu[None, :]       # (T, K)
    sigma = np.sqrt((gamma * diff**2).sum(axis=0) / w)
    sigma = np.maximum(sigma, 1e-6)

    return pi, A, mu, sigma


def _relabel(pi, A, mu, sigma, gamma):
    """Ensure State 0 = Calm (lower σ) and State 1 = Turbulent (higher σ)."""
    if sigma[0] > sigma[1]:
        return (pi[::-1].copy(),
                A[::-1, ::-1].copy(),
                mu[::-1].copy(),
                sigma[::-1].copy(),
                gamma[:, ::-1].copy())
    return pi, A, mu, sigma, gamma


# ════════════════════════════════════════════════════════════════════
# RegimeFilter — public class
# ════════════════════════════════════════════════════════════════════

class RegimeFilter:
    """
    Two-state Gaussian Hidden Markov Model for market regime detection.

    Typical usage
    -------------
    >>> rf = RegimeFilter(ticker="SPY", turbulence_threshold=0.6)
    >>> rf.fetch_data(years=5)
    >>> rf.fit(n_restarts=10)
    >>> signal = rf.current_signal()
    >>> print(signal.action)
    >>> rf.plot_regime_map(save_path="regime_map.png")

    Parameters
    ----------
    ticker : str
    turbulence_threshold : float
        P(Turbulent) above which the filter emits risk-off signals.
    n_iter : int
        Maximum Baum-Welch iterations per restart.
    tol : float
        Log-likelihood convergence tolerance.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        n_states: int = 2,
        turbulence_threshold: float = 0.6,
        n_iter: int = 300,
        tol: float = 1e-8,
    ):
        self.ticker = ticker
        self.K = n_states
        self.threshold = turbulence_threshold
        self.n_iter = n_iter
        self.tol = tol

        self.params: HMMParams | None = None
        self.log_returns: np.ndarray | None = None
        self.dates: np.ndarray | None = None
        self._gamma: np.ndarray | None = None
        self._alpha: np.ndarray | None = None    # kept for online update
        self._ll_history: list[float] = []
        self._fitted = False

    # ── Data ingestion ──────────────────────────────────────────────

    def fetch_data(self, years: int = 5) -> pd.Series:
        """
        Download `years` of daily SPY closes from Yahoo Finance and
        compute log-returns.  Stores results in self.log_returns /
        self.dates.
        """
        end   = dt.datetime.today()
        start = end - dt.timedelta(days=int(years * 365.25))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(self.ticker, start=start, end=end,
                             auto_adjust=True, progress=False)

        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {self.ticker!r}")

        close = df["Close"].squeeze()
        log_ret = np.log(close / close.shift(1)).dropna()

        self.log_returns = log_ret.to_numpy(dtype=np.float64)
        self.dates = log_ret.index.to_numpy()
        return log_ret

    # ── Fitting (Baum-Welch with restarts) ─────────────────────────

    def fit(self, obs: np.ndarray | None = None,
            n_restarts: int = 8, seed: int = 0) -> HMMParams:
        """
        Calibrate the G-HMM via Expectation-Maximization (Baum-Welch).

        Multiple random restarts are run; the parameter set achieving
        the highest final log-likelihood is kept.

        Parameters
        ----------
        obs : array-like | None
            Observation sequence.  Uses self.log_returns if None.
        n_restarts : int
            Number of independent EM runs.
        seed : int
            Base random seed for reproducibility.
        """
        if obs is None:
            if self.log_returns is None:
                raise ValueError("No observations — call fetch_data() first.")
            obs = self.log_returns
        obs = np.asarray(obs, dtype=np.float64)

        best_ll   = -np.inf
        best_pack = None

        rng = np.random.default_rng(seed)

        for restart in range(n_restarts):
            pi, A, mu, sigma = self._initialise(obs, rng)
            ll_prev = -np.inf
            ll_hist = []

            for _ in range(self.n_iter):
                gamma, xi_sum, alpha, ll = _e_step(obs, pi, A, mu, sigma)
                ll_hist.append(ll)
                if abs(ll - ll_prev) < self.tol:
                    break
                ll_prev = ll
                pi, A, mu, sigma = _m_step(obs, gamma, xi_sum)

            if ll > best_ll:
                best_ll   = ll
                best_pack = (pi, A, mu, sigma, gamma, alpha, ll_hist)

        pi, A, mu, sigma, gamma, alpha, ll_hist = best_pack
        pi, A, mu, sigma, gamma = _relabel(pi, A, mu, sigma, gamma)

        self.params       = HMMParams(pi=pi, A=A, mu=mu, sigma=sigma)
        self._gamma       = gamma
        self._alpha       = alpha
        self._ll_history  = ll_hist
        self._fitted      = True
        return self.params

    # ── Viterbi decoding ────────────────────────────────────────────

    def viterbi(self, obs: np.ndarray | None = None) -> np.ndarray:
        """
        Decode the globally most-likely state sequence using the
        Viterbi algorithm (log-space, underflow-safe).

        Returns
        -------
        states : ndarray (T,)  —  0 = Calm, 1 = Turbulent
        """
        self._require_fitted()
        if obs is None:
            obs = self.log_returns
        obs = np.asarray(obs, dtype=np.float64)

        p    = self.params
        T    = len(obs)
        K    = self.K
        logA = np.log(p.A + 1e-300)

        log_delta = np.empty((T, K))
        psi       = np.zeros((T, K), dtype=np.int32)

        log_delta[0] = np.log(p.pi + 1e-300) + sp_norm.logpdf(
            obs[0], p.mu, p.sigma)

        for t in range(1, T):
            # candidates[j, k] = log_delta[t-1, j] + log A[j, k]
            cand = log_delta[t - 1, :, None] + logA   # (K, K)
            psi[t] = cand.argmax(axis=0)
            log_delta[t] = (cand[psi[t], np.arange(K)] +
                            sp_norm.logpdf(obs[t], p.mu, p.sigma))

        # Backtrack
        states = np.empty(T, dtype=np.int32)
        states[T - 1] = log_delta[T - 1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ── Real-time signal (forward-filtered, causal) ─────────────────

    def current_signal(self, obs: np.ndarray | None = None) -> RegimeSignal:
        """
        Return the regime signal for the *most recent* observation.
        Uses only forward-filtered probabilities — no look-ahead bias.
        """
        self._require_fitted()
        if obs is None:
            obs = self.log_returns
        obs = np.asarray(obs, dtype=np.float64)

        p = self.params
        log_B = _log_emission(obs, p.mu, p.sigma)
        alpha, _ = _forward_scaled(log_B, p.pi, p.A)

        prob = alpha[-1]          # P(S_T = k | x_{1:T})
        prob_turb = float(prob[TURBULENT])
        prob_calm = float(prob[CALM])

        action = self._action(prob_turb)
        state  = TURBULENT if prob_turb > 0.5 else CALM

        date = (pd.Timestamp(self.dates[-1]).date()
                if self.dates is not None else dt.date.today())

        return RegimeSignal(
            date=date,
            log_return=float(obs[-1]),
            state=state,
            prob_calm=prob_calm,
            prob_turbulent=prob_turb,
            action=action,
        )

    def update_signal(self, new_return: float) -> RegimeSignal:
        """
        Incrementally update the filter with a single new observation
        *without* re-running the full history.

        Propagates the last forward vector α_{T-1} one step:
            α_T(k) ∝ [Σ_j α_{T-1}(j) A_{jk}] · B_k(x_T)

        Parameters
        ----------
        new_return : float
            The most recent log-return observation.

        Returns
        -------
        RegimeSignal for the new observation.
        """
        self._require_fitted()
        if self._alpha is None:
            return self.current_signal()

        p     = self.params
        alpha_prev = self._alpha[-1]                          # (K,)
        log_b = sp_norm.logpdf(new_return, p.mu, p.sigma)    # (K,)

        a_new = (alpha_prev @ p.A) * np.exp(log_b)
        s     = a_new.sum()
        a_new /= s + 1e-300

        # Append to alpha for future calls
        self._alpha = np.vstack([self._alpha, a_new[None, :]])

        prob_turb = float(a_new[TURBULENT])
        prob_calm = float(a_new[CALM])
        state     = TURBULENT if prob_turb > 0.5 else CALM

        return RegimeSignal(
            date=dt.date.today(),
            log_return=new_return,
            state=state,
            prob_calm=prob_calm,
            prob_turbulent=prob_turb,
            action=self._action(prob_turb),
        )

    # ── Smoothed state probabilities ────────────────────────────────

    def smoothed_probabilities(self) -> np.ndarray:
        """
        Return γ(t, k) = P(S_t = k | x_{1:T}), shape (T, K).
        Smoothed (uses future data) — suitable for analysis, not live trading.
        """
        self._require_fitted()
        return self._gamma

    # ── Model diagnostics ───────────────────────────────────────────

    def diagnostics(self) -> ModelDiagnostics:
        """
        Compute model fit statistics:
        - Log-likelihood
        - AIC = −2·ℓ + 2·n_params
        - BIC = −2·ℓ + n_params·ln(T)
        - Expected regime durations = 1 / (1 − A_{kk}) trading days
        - Fraction of time in each regime
        """
        self._require_fitted()
        p  = self.params
        T  = len(self.log_returns)

        # Number of free parameters: K-1 pi, K(K-1) A, K mu, K sigma
        n_params = (self.K - 1) + self.K * (self.K - 1) + 2 * self.K
        ll = float(self._ll_history[-1]) if self._ll_history else np.nan

        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(T)

        # Expected duration (geometric distribution)
        dur_calm  = 1.0 / (1.0 - p.A[CALM, CALM]      + 1e-12)
        dur_turb  = 1.0 / (1.0 - p.A[TURBULENT, TURBULENT] + 1e-12)

        viterbi_states = self.viterbi()
        frac_calm = float((viterbi_states == CALM).mean())
        frac_turb = float((viterbi_states == TURBULENT).mean())

        return ModelDiagnostics(
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            expected_duration_calm=dur_calm,
            expected_duration_turbulent=dur_turb,
            regime_fraction_calm=frac_calm,
            regime_fraction_turbulent=frac_turb,
        )

    def summary(self) -> str:
        """Human-readable model summary string."""
        self._require_fitted()
        p  = self.params
        d  = self.diagnostics()
        T  = len(self.log_returns)
        lines = [
            "═" * 56,
            f"  G-HMM Regime Filter  |  {self.ticker}  |  T = {T}",
            "═" * 56,
            f"  Log-likelihood : {d.log_likelihood:>12.2f}",
            f"  AIC            : {d.aic:>12.2f}",
            f"  BIC            : {d.bic:>12.2f}",
            "─" * 56,
            "  State parameters:",
            f"  {'State':<14} {'μ':>10} {'σ':>10} {'π₀':>8}",
            f"  {'Calm':<14} {p.mu[CALM]:>10.5f} {p.sigma[CALM]:>10.5f} "
            f"{p.pi[CALM]:>8.3f}",
            f"  {'Turbulent':<14} {p.mu[TURBULENT]:>10.5f} "
            f"{p.sigma[TURBULENT]:>10.5f} {p.pi[TURBULENT]:>8.3f}",
            "─" * 56,
            "  Transition matrix  A[from, to]:",
            f"            Calm     Turbulent",
            f"  Calm    {p.A[0,0]:>7.4f}    {p.A[0,1]:>7.4f}",
            f"  Turbul. {p.A[1,0]:>7.4f}    {p.A[1,1]:>7.4f}",
            "─" * 56,
            f"  Expected duration (days):",
            f"    Calm      : {d.expected_duration_calm:>6.1f}",
            f"    Turbulent : {d.expected_duration_turbulent:>6.1f}",
            f"  Regime fraction:",
            f"    Calm      : {d.regime_fraction_calm:>6.1%}",
            f"    Turbulent : {d.regime_fraction_turbulent:>6.1%}",
            "═" * 56,
        ]
        return "\n".join(lines)

    # ── Visualisation: five-panel Regime Map ────────────────────────

    def plot_regime_map(self, save_path: str | None = None) -> plt.Figure:
        """
        Generate a five-panel Regime Map figure:

        Panel 1 (tall) — Returns scatter coloured by Viterbi state,
                          regime background shading.
        Panel 2        — Rolling 21-day realised volatility with
                          regime colour bands.
        Panel 3        — P(Turbulent | data) fill, threshold line,
                          signal zone annotations.
        Panel 4        — Side-by-side return distribution histograms
                          with fitted Gaussian overlays.
        Panel 5        — EM log-likelihood convergence curve.
        """
        self._require_fitted()

        states  = self.viterbi()
        dates   = pd.to_datetime(self.dates)
        returns = self.log_returns
        gamma   = self.smoothed_probabilities()
        p       = self.params
        T       = len(returns)

        # ── Rolling realised vol (21-day window) ──
        roll_vol = (pd.Series(returns)
                      .rolling(21, min_periods=5)
                      .std()
                      .to_numpy() * np.sqrt(252))

        fig = plt.figure(figsize=(18, 14), facecolor="#0f0f0f")
        fig.suptitle(
            f"{self.ticker}  |  G-HMM Regime Map  "
            f"(T = {T} obs,  threshold = {self.threshold:.0%})",
            color="white", fontsize=15, fontweight="bold", y=0.98,
        )

        gs = gridspec.GridSpec(
            5, 2,
            figure=fig,
            height_ratios=[3, 1.5, 1.5, 2, 1.5],
            width_ratios=[3, 1],
            hspace=0.06, wspace=0.25,
        )

        # shared colour theme
        def _style(ax, title=""):
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.spines[:].set_color("#444444")
            if title:
                ax.set_title(title, color="#dddddd", fontsize=9, loc="left")

        # ── Panel 1: Returns scatter ──────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        _style(ax1, "Daily log-returns  (colour = Viterbi state)")

        # Shade background by regime
        _shade_background(ax1, dates, states)

        for s in (CALM, TURBULENT):
            mask = states == s
            ax1.scatter(
                dates[mask], returns[mask],
                c=_STATE_COLOR[s], s=5, alpha=0.75,
                label=_STATE_LABEL[s], edgecolors="none",
            )
        ax1.axhline(0, color="#888888", lw=0.6, ls="--")
        ax1.set_ylabel("Log return", color="#cccccc", fontsize=9)
        ax1.legend(
            handles=[
                Patch(color=_STATE_COLOR[CALM], label="Calm"),
                Patch(color=_STATE_COLOR[TURBULENT], label="Turbulent"),
            ],
            loc="upper left", fontsize=8,
            facecolor="#222222", labelcolor="white",
        )
        ax1.tick_params(labelbottom=False)

        # ── Panel 2: Rolling realised vol ─────────────────────────
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        _style(ax2, "Annualised 21-day realised volatility")
        _shade_background(ax2, dates, states)
        ax2.plot(dates, roll_vol, color="#FFD700", lw=1.0, alpha=0.85)
        ax2.set_ylabel("Ann. vol", color="#cccccc", fontsize=9)
        ax2.tick_params(labelbottom=False)

        # ── Panel 3: P(Turbulent) ─────────────────────────────────
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        _style(ax3, "P(Turbulent | data)  — forward-smoothed")
        ax3.fill_between(dates, gamma[:, TURBULENT],
                         alpha=0.5, color=_STATE_COLOR[TURBULENT])
        ax3.axhline(self.threshold, color="white", ls="--", lw=1.0,
                    label=f"Threshold {self.threshold:.0%}")
        # Annotate action zones
        ax3.fill_between(dates, self.threshold, gamma[:, TURBULENT],
                         where=gamma[:, TURBULENT] >= self.threshold,
                         color="#FF6B6B", alpha=0.35, label="Risk-off zone")
        ax3.set_ylabel("P(Turbulent)", color="#cccccc", fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                   loc="upper left")
        ax3.set_xlabel("Date", color="#cccccc", fontsize=9)
        ax3.tick_params(axis="x", colors="#cccccc", labelsize=7)

        # ── Panel 4: Return distributions (right column, rows 0-2) ──
        ax4 = fig.add_subplot(gs[0:3, 1])
        _style(ax4, "Return distributions by regime")

        x_grid = np.linspace(returns.min() * 1.3, returns.max() * 1.3, 300)
        bins   = np.linspace(returns.min(), returns.max(), 50)

        for s in (CALM, TURBULENT):
            mask = states == s
            r_s  = returns[mask]
            ax4.hist(r_s, bins=bins, orientation="horizontal",
                     alpha=0.45, color=_STATE_COLOR[s],
                     density=True, label=_STATE_LABEL[s])
            pdf = sp_norm.pdf(x_grid, p.mu[s], p.sigma[s])
            ax4.plot(pdf, x_grid, color=_STATE_COLOR[s], lw=1.5)

        ax4.axhline(0, color="#888888", lw=0.5, ls="--")
        ax4.set_xlabel("Density", color="#cccccc", fontsize=8)
        ax4.set_ylabel("Log return", color="#cccccc", fontsize=8)
        ax4.legend(fontsize=7, facecolor="#222222", labelcolor="white")

        # ── Panel 5: EM convergence ───────────────────────────────
        ax5 = fig.add_subplot(gs[3, 0])
        _style(ax5, "Baum-Welch EM convergence  (best restart)")
        ax5.plot(self._ll_history, color="#00E5FF", lw=1.2)
        ax5.set_ylabel("Log-likelihood", color="#cccccc", fontsize=9)
        ax5.set_xlabel("EM iteration", color="#cccccc", fontsize=9)
        ax5.tick_params(axis="x", colors="#cccccc", labelsize=7)

        # ── Panel 6: Model summary text ───────────────────────────
        ax6 = fig.add_subplot(gs[3:, 1])
        ax6.axis("off")
        ax6.set_facecolor("#1a1a2e")
        d = self.diagnostics()
        txt = (
            f"AIC  : {d.aic:>10.1f}\n"
            f"BIC  : {d.bic:>10.1f}\n"
            f"LL   : {d.log_likelihood:>10.1f}\n\n"
            f"Calm  μ : {p.mu[CALM]:>+8.5f}\n"
            f"Calm  σ : {p.sigma[CALM]:>8.5f}\n"
            f"Calm  E[dur] : {d.expected_duration_calm:>5.1f}d\n\n"
            f"Turb  μ : {p.mu[TURBULENT]:>+8.5f}\n"
            f"Turb  σ : {p.sigma[TURBULENT]:>8.5f}\n"
            f"Turb  E[dur] : {d.expected_duration_turbulent:>5.1f}d\n\n"
            f"Calm  : {d.regime_fraction_calm:>6.1%}\n"
            f"Turb  : {d.regime_fraction_turbulent:>6.1%}"
        )
        ax6.text(
            0.05, 0.97, txt,
            transform=ax6.transAxes,
            va="top", fontsize=8,
            color="#dddddd",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#111122", edgecolor="#444466"),
        )

        # ── Panel 7: Transition heatmap ───────────────────────────
        ax7 = fig.add_subplot(gs[4, 0])
        _style(ax7, "Transition probability matrix")
        im = ax7.imshow(p.A, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        for (i, j), val in np.ndenumerate(p.A):
            ax7.text(j, i, f"{val:.3f}", ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")
        ax7.set_xticks([0, 1]); ax7.set_yticks([0, 1])
        ax7.set_xticklabels(["→ Calm", "→ Turbulent"],
                            color="#cccccc", fontsize=8)
        ax7.set_yticklabels(["Calm →", "Turbulent →"],
                            color="#cccccc", fontsize=8)
        plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())

        return fig

    # ── Convenience ─────────────────────────────────────────────────

    def fit_predict(self, years: int = 5, n_restarts: int = 8,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        One-shot: fetch data, fit model, return (dates, states).
        """
        self.fetch_data(years=years)
        self.fit(n_restarts=n_restarts, seed=seed)
        return self.dates, self.viterbi()

    # ── Private helpers ─────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first.")

    def _action(self, prob_turb: float) -> str:
        if prob_turb > 0.8:
            return "Halt Trading"
        elif prob_turb > self.threshold:
            return "Delta Hedge"
        return "Trade"

    @staticmethod
    def _initialise(obs: np.ndarray,
                    rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
        """
        Random initialisation with K-means++ seeding for the means.
        Ensures every restart begins from a different point in
        parameter space, improving coverage of the likelihood surface.
        """
        K = 2
        # K-means++ seed: pick first centre uniformly, second by distance²
        idx0 = rng.integers(0, len(obs))
        mu0  = obs[idx0]
        d2   = (obs - mu0) ** 2
        idx1 = rng.choice(len(obs), p=d2 / d2.sum())
        mu   = np.array([mu0, obs[idx1]])

        sigma = np.full(K, np.std(obs) * rng.uniform(0.5, 1.5, K))
        sigma = np.maximum(sigma, 1e-4)

        # Slightly sticky initialisation
        diag = rng.uniform(0.80, 0.97)
        A    = np.array([[diag, 1 - diag],
                         [1 - diag, diag]])

        pi = rng.dirichlet(np.ones(K))

        return pi, A, mu, sigma


# ════════════════════════════════════════════════════════════════════
# Plot helper
# ════════════════════════════════════════════════════════════════════

def _shade_background(ax: plt.Axes,
                      dates: pd.DatetimeIndex,
                      states: np.ndarray) -> None:
    """
    Draw alternating background bands for each turbulent episode on `ax`.
    Uses axvspan with a low alpha so scatter points remain visible.
    """
    in_turb = False
    start   = None
    for i, (d, s) in enumerate(zip(dates, states)):
        if s == TURBULENT and not in_turb:
            start   = d
            in_turb = True
        elif s == CALM and in_turb:
            ax.axvspan(start, d, color=_STATE_COLOR[TURBULENT],
                       alpha=0.12, linewidth=0)
            in_turb = False
    if in_turb:
        ax.axvspan(start, dates[-1], color=_STATE_COLOR[TURBULENT],
                   alpha=0.12, linewidth=0)
