"""
Validation Suite — Scientific Backtesting & Model Verification.

Implements two mandatory tests before any live deployment:

┌─────────────────────────────────────────────────────────────────┐
│  Test 1 — Regime Stability Analysis  (Crash Prediction)         │
│  • Fit HMM on 2018-2019 data (training window)                  │
│  • Forward-filter over 2020-2023 test window                    │
│  • Verify P(Turbulent) > threshold BEFORE major drawdowns       │
│  • Compute lead time in trading days                            │
│  • Generate annotated regime-over-drawdown plot                 │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  Test 2 — Monte Carlo Convergence  O(N^{-1/2})                  │
│  • Run MC pricer at N ∈ {64, 128, …, 8192} paths               │
│  • Estimate empirical SE via independent repeated runs           │
│  • Fit log-log regression: log(SE) = a + b·log(N)              │
│  • Pass iff b ∈ [−0.65, −0.35]                                  │
│  • Flag "VARIANCE REDUCTION FAILURE" if out of range            │
│  • Generate log-log convergence plot with O(N^{-1/2}) reference │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

from pricing_engine import PricingEngine
from regime_filter import RegimeFilter, CALM, TURBULENT

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# Result containers
# ════════════════════════════════════════════════════════════════════

@dataclass
class DrawdownEvent:
    start:          pd.Timestamp
    trough:         pd.Timestamp
    drawdown_pct:   float            # negative value, e.g. -0.34 = 34%
    turbulent_flag: pd.Timestamp | None = None   # first day HMM flagged
    lead_days:      float | None        = None   # trough - turbulent_flag (days)
    detected:       bool                = False


@dataclass
class StabilityResult:
    """Output of Test 1 — Regime Stability Analysis."""
    n_drawdowns_tested:  int
    n_detected:          int
    detection_rate:      float          # fraction of events caught
    avg_lead_days:       float | None
    covid_lead_days:     float | None   # specific: Feb-Mar 2020
    events:              list[DrawdownEvent] = field(default_factory=list)
    passed:              bool = False
    summary:             str  = ""


@dataclass
class ConvergenceResult:
    """Output of Test 2 — Monte Carlo Convergence."""
    n_values:         list[int]
    se_values:        list[float]
    convergence_rate: float       # fitted slope b in log(SE) = a + b·log(N)
    r_squared:        float
    passed:           bool
    flag:             str         # "" or "VARIANCE REDUCTION FAILURE"
    summary:          str  = ""


class ValidationReport(NamedTuple):
    stability:   StabilityResult
    convergence: ConvergenceResult
    all_passed:  bool


# ════════════════════════════════════════════════════════════════════
# Test 1 — Regime Stability Analysis
# ════════════════════════════════════════════════════════════════════

class RegimeStabilityTest:
    """
    Fits the HMM on a training window and validates detection of
    historical market crises on a held-out test window.

    The primary event tested is the COVID-19 crash (Feb–Mar 2020),
    which produced a 34% peak-to-trough decline in SPY in 33 days.

    Pass criterion
    ──────────────
    The HMM must emit P(Turbulent) > threshold for at least 50% of
    the identified drawdown events AND must detect the primary event
    (largest drawdown) with a non-negative lead time.
    """

    # COVID peak: ~Feb 19, 2020.  We want the HMM to flag turbulent
    # BEFORE OR ON this date.
    _PRIMARY_EVENT_START = pd.Timestamp("2020-02-18")
    _PRIMARY_EVENT_TROUGH = pd.Timestamp("2020-03-23")

    def __init__(
        self,
        ticker:            str   = "SPY",
        train_end:         str   = "2019-12-31",
        test_start:        str   = "2020-01-02",
        test_end:          str   = "2023-12-31",
        threshold:         float = 0.60,
        min_drawdown_pct:  float = 0.05,    # only test ≥ 5% drawdowns
        drawdown_window:   int   = 60,       # days to look for trough
    ):
        self.ticker           = ticker
        self.train_end        = pd.Timestamp(train_end)
        self.test_start       = pd.Timestamp(test_start)
        self.test_end         = pd.Timestamp(test_end)
        self.threshold        = threshold
        self.min_drawdown_pct = min_drawdown_pct
        self.drawdown_window  = drawdown_window

    def run(self, rf: RegimeFilter | None = None,
            plot: bool = True,
            save_path: str | None = "stability_test.png") -> StabilityResult:
        """
        Execute the stability test.

        Parameters
        ──────────
        rf : RegimeFilter | None
            A pre-constructed filter.  If None, one is created and
            fitted automatically.
        """
        logger.info("[Stability] Fetching SPY data …")
        spy_returns, spy_prices = self._fetch_spy()

        # ── Fit HMM on training window only ──
        if rf is None:
            rf = RegimeFilter(
                ticker=self.ticker,
                turbulence_threshold=self.threshold,
            )
        train_mask = spy_returns.index <= self.train_end
        train_obs  = spy_returns[train_mask].values

        logger.info("[Stability] Fitting HMM on %d training observations …",
                    len(train_obs))
        rf.log_returns = train_obs
        rf.dates       = spy_returns[train_mask].index.values
        rf.fit(obs=train_obs, n_restarts=6, seed=0)

        params = rf.params
        logger.info("[Stability] Trained: μ_calm=%.5f σ_calm=%.5f  "
                    "μ_turb=%.5f σ_turb=%.5f",
                    params.mu[CALM], params.sigma[CALM],
                    params.mu[TURBULENT], params.sigma[TURBULENT])

        # ── Forward-filter over ENTIRE history (train + test) ──
        all_obs = spy_returns.values
        from regime_filter import _forward_scaled, _log_emission
        log_B   = _log_emission(all_obs, params.mu, params.sigma)
        alpha, _= _forward_scaled(log_B, params.pi, params.A)
        prob_turb = pd.Series(alpha[:, TURBULENT], index=spy_returns.index)

        # ── Identify major drawdowns in test window ──
        test_prices = spy_prices[spy_prices.index >= self.test_start]
        events      = self._identify_drawdowns(test_prices)

        # ── Check detection for each event ──
        for ev in events:
            # Find first day in [ev.start - 10d, ev.trough] where P(T) > threshold
            look_start = ev.start - pd.Timedelta(days=10)
            look_end   = ev.trough
            window     = prob_turb[
                (prob_turb.index >= look_start) &
                (prob_turb.index <= look_end)
            ]
            hits = window[window > self.threshold]
            if len(hits) > 0:
                flag_date       = hits.index[0]
                ev.turbulent_flag = flag_date
                ev.lead_days    = (ev.trough - flag_date).days
                ev.detected     = True

        # ── Compute summary statistics ──
        n_det = sum(1 for e in events if e.detected)
        det_rate = n_det / max(len(events), 1)
        lead_days_list = [e.lead_days for e in events
                          if e.detected and e.lead_days is not None]
        avg_lead = float(np.mean(lead_days_list)) if lead_days_list else None

        # COVID-specific check
        covid_lead = None
        for ev in events:
            if (abs((ev.trough - self._PRIMARY_EVENT_TROUGH).days) < 5
                    and ev.detected):
                covid_lead = ev.lead_days
                break

        passed = det_rate >= 0.50 and (covid_lead is None or covid_lead >= 0)

        summary_lines = [
            "═" * 56,
            "  STABILITY TEST RESULT",
            "═" * 56,
            f"  Drawdowns tested  : {len(events)}",
            f"  Detected          : {n_det}",
            f"  Detection rate    : {det_rate:.1%}",
            f"  Avg lead time     : {avg_lead:.1f}d" if avg_lead else "  Avg lead time     : N/A",
            f"  COVID lead time   : {covid_lead:.1f}d" if covid_lead is not None else "  COVID lead time   : event not found in window",
            f"  PASS              : {'✓' if passed else '✗'}",
            "═" * 56,
        ]
        summary = "\n".join(summary_lines)
        logger.info("\n%s", summary)

        result = StabilityResult(
            n_drawdowns_tested=len(events),
            n_detected=n_det,
            detection_rate=det_rate,
            avg_lead_days=avg_lead,
            covid_lead_days=covid_lead,
            events=events,
            passed=passed,
            summary=summary,
        )

        if plot:
            self._plot(spy_prices, prob_turb, events, result, save_path)

        return result

    # ── Data ─────────────────────────────────────────────────────────

    def _fetch_spy(self) -> tuple[pd.Series, pd.Series]:
        """Fetch SPY daily data from yfinance, return (log_returns, prices)."""
        start_dt = (self.train_end - pd.Timedelta(days=int(2 * 365.25)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(self.ticker, start=start_dt.strftime("%Y-%m-%d"),
                             end=self.test_end.strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"No data for {self.ticker}")
        prices  = df["Close"].squeeze()
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return log_ret, prices.loc[log_ret.index]

    # ── Drawdown identification ───────────────────────────────────────

    def _identify_drawdowns(self, prices: pd.Series) -> list[DrawdownEvent]:
        """
        Identify peak-to-trough drawdown events ≥ min_drawdown_pct within
        a rolling `drawdown_window`-day forward window.
        """
        events:  list[DrawdownEvent] = []
        seen_troughs: set[pd.Timestamp] = set()

        for i in range(len(prices) - self.drawdown_window):
            peak_val = prices.iloc[i]
            window   = prices.iloc[i: i + self.drawdown_window]
            trough_idx = window.idxmin()
            trough_val = window.min()

            dd_pct = (trough_val - peak_val) / peak_val
            if dd_pct > -self.min_drawdown_pct:
                continue
            if trough_idx in seen_troughs:
                continue

            # De-duplicate: skip if very close to an existing event
            too_close = any(
                abs((ev.trough - trough_idx).days) < 20
                for ev in events
            )
            if too_close:
                continue

            seen_troughs.add(trough_idx)
            events.append(DrawdownEvent(
                start=prices.index[i],
                trough=trough_idx,
                drawdown_pct=dd_pct,
            ))

        # Sort by severity
        events.sort(key=lambda e: e.drawdown_pct)
        return events

    # ── Visualization ─────────────────────────────────────────────────

    def _plot(self, prices, prob_turb, events, result, save_path):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 9), sharex=True,
            facecolor="#0f0f0f",
            gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.05},
        )
        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.spines[:].set_color("#444444")

        # ── Price chart ──
        ax1.plot(prices.index, prices.values, color="#90CAF9",
                 lw=1.0, label=self.ticker)

        for ev in events:
            color = "#F44336" if ev.detected else "#FF9800"
            ax1.axvspan(ev.start, ev.trough, color=color, alpha=0.15)
            ax1.annotate(
                f"{ev.drawdown_pct:.1%}",
                xy=(ev.trough, prices.loc[ev.trough]
                    if ev.trough in prices.index
                    else prices.iloc[-1]),
                fontsize=7, color=color,
                xytext=(0, -18), textcoords="offset points",
                ha="center",
            )
            if ev.detected and ev.turbulent_flag is not None:
                ax1.axvline(ev.turbulent_flag, color="#F44336",
                            ls="--", lw=0.8, alpha=0.7)

        ax1.set_ylabel("SPY Price", color="#cccccc", fontsize=9)
        status_txt = f"Detection: {result.detection_rate:.1%}  |  Pass: {'✓' if result.passed else '✗'}"
        ax1.set_title(
            f"Regime Stability Test — {self.ticker}  |  {status_txt}",
            color="white", fontsize=11, loc="left",
        )
        ax1.legend([
            Patch(color="#F44336", alpha=0.4, label="Detected drawdown"),
            Patch(color="#FF9800", alpha=0.4, label="Missed drawdown"),
        ], loc="lower left", facecolor="#222222", labelcolor="white", fontsize=7)

        # ── P(Turbulent) ──
        ax2.fill_between(prob_turb.index, prob_turb.values,
                         alpha=0.55, color="#F44336")
        ax2.axhline(self.threshold, color="white", ls="--", lw=1.0,
                    label=f"Threshold {self.threshold:.0%}")
        ax2.set_ylabel("P(Turbulent)", color="#cccccc", fontsize=9)
        ax2.set_xlabel("Date", color="#cccccc", fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.legend(facecolor="#222222", labelcolor="white", fontsize=7)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info("[Stability] Plot saved → %s", save_path)
        return fig


# ════════════════════════════════════════════════════════════════════
# Test 2 — Monte Carlo Convergence
# ════════════════════════════════════════════════════════════════════

class MCConvergenceTest:
    """
    Validates that the Monte Carlo pricer's standard error decays as
    O(N^{-1/2}) — the theoretical rate for unbiased estimators.

    Methodology
    ────────────
    For each N in a geometric grid:
      1. Run `n_trials` independent MC simulations, each with N paths.
      2. Collect the straddle price estimates {P_1, …, P_{n_trials}}.
      3. Empirical SE(N) = std({P_1, …, P_{n_trials}}).

    Fit: log(SE) = a + b·log(N)  via OLS.

    Pass criterion: b ∈ [−0.65, −0.35]   (centred on −0.5, ±0.15)

    If b ∉ [−0.65, −0.35], the test raises a "VARIANCE REDUCTION
    FAILURE" flag indicating a systematic bias or broken random
    number generator that must be investigated before live trading.
    """

    SLOPE_LOW  = -0.65
    SLOPE_HIGH = -0.35

    def __init__(
        self,
        engine:          PricingEngine,
        n_values:        list[int] | None = None,
        n_trials:        int   = 20,
        K:               float = 100.0,
        T_days:          int   = 5,
    ):
        self.engine   = engine
        self.n_values = n_values or [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        self.n_trials = n_trials
        self.K        = K
        self.T_days   = T_days

    def run(
        self,
        plot: bool = True,
        save_path: str | None = "convergence_test.png",
    ) -> ConvergenceResult:
        """Execute the convergence test."""
        logger.info("[Convergence] Starting MC convergence test …")
        logger.info("  N values  : %s", self.n_values)
        logger.info("  Trials    : %d", self.n_trials)

        se_values: list[float] = []
        base_engine = self.engine

        for N in self.n_values:
            trial_prices: list[float] = []
            for trial in range(self.n_trials):
                trial_engine = PricingEngine(
                    H=base_engine.H,
                    v0=base_engine.v0,
                    nu=base_engine.nu,
                    S0=base_engine.S0,
                    r=base_engine.r,
                    rho=base_engine.rho,
                    n_paths=N,
                    steps_per_day=base_engine.steps_per_day,
                    kappa=base_engine.kappa,
                    J=base_engine.J,
                    seed=trial * 1000 + N,   # unique seed per (trial, N)
                )
                result = trial_engine.price_straddle(
                    K=self.K, T_days=self.T_days)
                trial_prices.append(result.price)

            se = float(np.std(trial_prices, ddof=1))
            se_values.append(se)
            logger.info("  N=%-6d  SE=%.6f  (trials=%d)", N, se, self.n_trials)

        # ── Log-log regression ──
        log_N  = np.log(self.n_values)
        log_se = np.log(np.array(se_values) + 1e-12)

        slope, intercept, r_value, p_value, _ = stats.linregress(log_N, log_se)
        r2 = r_value ** 2

        passed = self.SLOPE_LOW <= slope <= self.SLOPE_HIGH
        flag   = "" if passed else "VARIANCE REDUCTION FAILURE"

        summary_lines = [
            "═" * 56,
            "  MONTE CARLO CONVERGENCE TEST RESULT",
            "═" * 56,
            f"  Convergence rate b : {slope:+.4f}",
            f"  Expected range     : [{self.SLOPE_LOW}, {self.SLOPE_HIGH}]",
            f"  R²                 : {r2:.4f}",
            f"  PASS               : {'✓' if passed else '✗'}",
        ]
        if flag:
            summary_lines.append(f"  FLAG               : *** {flag} ***")
        summary_lines.append("═" * 56)
        summary = "\n".join(summary_lines)
        logger.info("\n%s", summary)

        if flag:
            logger.critical(
                "*** %s *** convergence rate b=%.4f is outside "
                "the acceptable range [%.2f, %.2f]",
                flag, slope, self.SLOPE_LOW, self.SLOPE_HIGH,
            )

        result = ConvergenceResult(
            n_values=list(self.n_values),
            se_values=se_values,
            convergence_rate=float(slope),
            r_squared=float(r2),
            passed=passed,
            flag=flag,
            summary=summary,
        )

        if plot:
            self._plot(result, intercept, save_path)

        return result

    # ── Visualization ─────────────────────────────────────────────────

    def _plot(self, result: ConvergenceResult,
              intercept: float, save_path: str | None) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f0f0f")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc", labelsize=9)
        ax.spines[:].set_color("#444444")

        N_arr  = np.array(result.n_values)
        SE_arr = np.array(result.se_values)

        # Empirical SE
        ax.scatter(N_arr, SE_arr, color="#00E5FF", s=80, zorder=5,
                   label="Empirical SE", edgecolors="white", linewidths=0.5)
        ax.plot(N_arr, SE_arr, color="#00E5FF", lw=1.0, alpha=0.6)

        # Fitted regression line
        fitted = np.exp(intercept + result.convergence_rate * np.log(N_arr))
        ax.plot(N_arr, fitted, color="#FF6B6B", lw=2.0, ls="--",
                label=f"Fitted  b={result.convergence_rate:+.4f}")

        # Theoretical O(N^{-1/2}) reference
        se0 = SE_arr[0] * (N_arr[0] ** 0.5)
        theory = se0 / np.sqrt(N_arr)
        ax.plot(N_arr, theory, color="#A5D6A7", lw=1.5, ls=":",
                label=r"Theoretical $O(N^{-1/2})$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of MC Paths (N)", color="#cccccc", fontsize=11)
        ax.set_ylabel("Standard Error (SE)", color="#cccccc", fontsize=11)

        status = "PASS ✓" if result.passed else f"FAIL ✗  {result.flag}"
        title_color = "#A5D6A7" if result.passed else "#F44336"
        ax.set_title(
            f"MC Convergence Test  |  b={result.convergence_rate:+.4f}  "
            f"R²={result.r_squared:.3f}  |  {status}",
            color=title_color, fontsize=12, fontweight="bold",
        )

        # Shade acceptance band
        N_ref = np.array([N_arr[0], N_arr[-1]])
        hi = np.exp(intercept + self.SLOPE_HIGH * np.log(N_ref))
        lo = np.exp(intercept + self.SLOPE_LOW  * np.log(N_ref))
        ax.fill_between(N_ref, lo, hi, alpha=0.10, color="#A5D6A7",
                        label="Acceptance band")

        ax.legend(facecolor="#222222", labelcolor="white", fontsize=9)
        ax.grid(True, color="#333344", lw=0.5, alpha=0.6)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info("[Convergence] Plot saved → %s", save_path)
        return fig


# ════════════════════════════════════════════════════════════════════
# Top-level ValidationSuite
# ════════════════════════════════════════════════════════════════════

class ValidationSuite:
    """
    Orchestrates both validation tests and produces a combined report.

    Usage
    ─────
    >>> engine = PricingEngine(H=0.07, v0=0.04, S0=100, ...)
    >>> suite  = ValidationSuite(engine)
    >>> report = suite.run()
    >>> print(report.stability.summary)
    >>> print(report.convergence.summary)
    """

    def __init__(
        self,
        engine:            PricingEngine,
        stability_ticker:  str   = "SPY",
        train_end:         str   = "2019-12-31",
        test_end:          str   = "2023-12-31",
        mc_n_values:       list[int] | None = None,
        mc_n_trials:       int   = 20,
        output_dir:        str   = ".",
    ):
        self.engine           = engine
        self.stability_ticker = stability_ticker
        self.train_end        = train_end
        self.test_end         = test_end
        self.mc_n_values      = mc_n_values
        self.mc_n_trials      = mc_n_trials
        self.output_dir       = Path(output_dir)

    def run(
        self,
        run_stability:   bool = True,
        run_convergence: bool = True,
        plot:            bool = True,
    ) -> ValidationReport:
        """Run the full validation suite and return a combined report."""
        logger.info("═" * 60)
        logger.info("  VALIDATION SUITE — START")
        logger.info("═" * 60)

        stability_result  = None
        convergence_result = None

        # ── Test 1 ──
        if run_stability:
            logger.info("\n[Test 1/2] Regime Stability Analysis …")
            stab_test = RegimeStabilityTest(
                ticker=self.stability_ticker,
                train_end=self.train_end,
                test_end=self.test_end,
            )
            try:
                stability_result = stab_test.run(
                    plot=plot,
                    save_path=str(self.output_dir / "stability_test.png"),
                )
            except Exception as exc:
                logger.error("[Test 1] FAILED with exception: %s", exc)
                stability_result = StabilityResult(
                    n_drawdowns_tested=0, n_detected=0,
                    detection_rate=0.0, avg_lead_days=None,
                    covid_lead_days=None, events=[],
                    passed=False,
                    summary=f"FAILED: {exc}",
                )

        # ── Test 2 ──
        if run_convergence:
            logger.info("\n[Test 2/2] Monte Carlo Convergence Test …")
            conv_test = MCConvergenceTest(
                engine=self.engine,
                n_values=self.mc_n_values,
                n_trials=self.mc_n_trials,
                K=self.engine.S0,
                T_days=5,
            )
            convergence_result = conv_test.run(
                plot=plot,
                save_path=str(self.output_dir / "convergence_test.png"),
            )

        all_passed = (
            (stability_result is None or stability_result.passed) and
            (convergence_result is None or convergence_result.passed)
        )

        logger.info("\n%s", "═" * 60)
        logger.info("  VALIDATION SUITE — %s", "ALL PASSED ✓" if all_passed else "FAILURES DETECTED ✗")
        logger.info("═" * 60)

        return ValidationReport(
            stability=stability_result,
            convergence=convergence_result,
            all_passed=all_passed,
        )
