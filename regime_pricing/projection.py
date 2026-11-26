"""A regime mixture and its continuous Dupire projection on the same vanilla surface."""

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import ndtr, logsumexp

SPOT, REVEAL, MATURITY, INITIAL_VOL = 100.0, 0.5, 1.0, 0.20
VOLS = np.array([0.10, 0.40])
# Signal becomes public exactly at REVEAL; the final volatility component is not
# known from that announcement. The spot has the same law in both signal states.
SIGNALS = {"calm": np.array([0.85, 0.15]), "stress": np.array([0.20, 0.80])}
WEIGHTS = (SIGNALS["calm"] + SIGNALS["stress"]) / 2


def put_moments(spot, strike, variances, weights):
    v = np.asarray(variances)
    truncated = [
        spot**n
        * np.dot(
            weights,
            np.exp((n * n - n) * v / 2)
            * ndtr((np.log(strike / spot) + (0.5 - n) * v) / np.sqrt(v)),
        )
        for n in range(3)
    ]
    m0, m1, m2 = truncated
    premium = strike * m0 - m1
    variance = max(0.0, strike**2 * m0 - 2 * strike * m1 + m2 - premium**2)
    cov = (strike + spot) * m1 - m2 - spot * strike * m0
    underlier_var = spot**2 * np.dot(weights, np.expm1(v))
    return dict(
        premium=premium,
        delta=-m1 / spot,
        payoff_variance=variance,
        variance_gain=underlier_var,
        covariance=cov,
        optimal_hedge=cov / underlier_var,
    )


def dupire_variance(t, x):
    """x is log(S/S0); variance is conditioned on spot using the marginal densities."""
    if t < REVEAL:
        return np.full_like(x, INITIAL_VOL**2)
    v = INITIAL_VOL**2 * REVEAL + VOLS**2 * (t - REVEAL)
    log_weights = np.log(WEIGHTS) - 0.5 * np.log(v) - 0.5 * (x[:, None] + v / 2) ** 2 / v
    return np.exp(log_weights - logsumexp(log_weights, axis=1)[:, None]) @ (VOLS**2)


def local_put(strike, space, steps):
    """Backward Crank-Nicolson on log spot with initial implicit damping."""
    xmax = 2.4
    grid = np.linspace(-xmax, xmax, space + 1)
    dx = grid[1] - grid[0]
    dt = (MATURITY - REVEAL) / steps
    spots = SPOT * np.exp(grid)
    value = np.maximum(strike - spots, 0.0)
    left_boundary = strike - spots[0]
    for n in range(steps):
        midpoint = MATURITY - (n + 0.5) * dt
        var = dupire_variance(midpoint, grid[1:-1])
        lower = 0.5 * var * (dx**-2 + 0.5 / dx)
        upper = 0.5 * var * (dx**-2 - 0.5 / dx)
        diag = -var / dx**2
        theta = 1.0 if n < 2 else 0.5
        rhs = value[1:-1] + (1 - theta) * dt * (
            lower * value[:-2] + diag * value[1:-1] + upper * value[2:]
        )
        rhs[0] += theta * dt * lower[0] * left_boundary
        band = np.zeros((3, space - 1))
        band[0, 1:] = -theta * dt * upper[:-1]
        band[1] = 1 - theta * dt * diag
        band[2, :-1] = -theta * dt * lower[1:]
        value[1:-1] = solve_banded((1, 1), band, rhs, check_finite=False)
        value[0], value[-1] = left_boundary, 0.0
    j = space // 2
    return dict(
        premium=float(value[j]), delta=float((value[j + 1] - value[j - 1]) / (2 * dx * SPOT))
    )


def implied(spot, strike, time, premium):
    def residual(vol):
        return put_moments(spot, strike, [vol**2 * time], [1.0])["premium"] - premium

    return float(brentq(residual, 0.001, 3.0))


def projection_demo():
    curves = {}
    maturity_left = MATURITY - REVEAL
    variances = VOLS**2 * maturity_left
    for strike in (90, 100, 110):
        models = {}
        for signal, weights in SIGNALS.items():
            m = put_moments(SPOT, strike, variances, weights)
            models[signal] = {
                **m,
                "implied_vol": implied(SPOT, strike, maturity_left, m["premium"]),
            }
        coarse, fine = local_put(strike, 600, 160), local_put(strike, 1200, 320)
        models["dupire"] = {
            **fine,
            "implied_vol": implied(SPOT, strike, maturity_left, fine["premium"]),
        }
        models["grid_change"] = {k: abs(fine[k] - coarse[k]) for k in fine}
        curves[str(strike)] = models
    target = curves["90"]
    regimes = [target[k] for k in SIGNALS]
    pooled = sum(m["covariance"] for m in regimes) / sum(m["variance_gain"] for m in regimes)

    def rmse(hedges):
        errors = [
            m["payoff_variance"] - 2 * h * m["covariance"] + h * h * m["variance_gain"]
            for m, h in zip(regimes, hedges, strict=True)
        ]
        return float(np.sqrt(np.mean(errors)))

    hedge = dict(
        dupire_delta=rmse([target["dupire"]["delta"]] * 2),
        regime_delta=rmse([m["delta"] for m in regimes]),
        best_spot_only=rmse([pooled] * 2),
        best_with_signal=rmse([m["optimal_hedge"] for m in regimes]),
    )
    return dict(
        spot=SPOT,
        reveal=REVEAL,
        maturity=MATURITY,
        quotes=curves,
        single_period_hedge_strike=90,
        pooled_optimal_units=pooled,
        hedge_rmse=hedge,
        signal_probabilities=[0.5, 0.5],
        vols=VOLS.tolist(),
        conditional_weights={k: v.tolist() for k, v in SIGNALS.items()},
    )
