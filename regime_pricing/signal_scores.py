"""Analytic conditional put prices and post-signal pricing RMSE."""

from math import fsum, log, sqrt
from statistics import NormalDist


def signal_price_scores(smile):
    """Score the conditional put against analytic prices under the specified mixture."""
    signals = ("calm", "stress")
    probabilities = smile["signal_probabilities"]  # calm, then stress
    spot = smile["spot"]
    strike = smile["single_period_hedge_strike"]
    remaining = smile["maturity"] - smile["reveal"]
    cdf = NormalDist().cdf
    component_puts = []
    for volatility in smile["vols"]:
        std = volatility * sqrt(remaining)
        d1 = log(spot / strike) / std + std / 2
        component_puts.append(strike * cdf(std - d1) - spot * cdf(-d1))

    targets = []
    for state in signals:
        weights = smile["conditional_weights"][state]
        targets.append(fsum(w * put for w, put in zip(weights, component_puts, strict=True)))

    def average(values):
        return fsum(p * v for p, v in zip(probabilities, values, strict=True))

    def rmse(prices):
        errors = ((p - t) ** 2 for p, t in zip(prices, targets, strict=True))
        return format(sqrt(average(errors)), ".3f")

    quotes = smile["quotes"][str(strike)]
    regime = [quotes[state]["premium"] for state in signals]
    center = average(regime)
    half_response = [(center + price) / 2 for price in regime]
    return {
        "reference_mean": format(average(targets), ".3f"),
        "half_response_prices": dict(zip(signals, (format(x, ".3f") for x in half_response), strict=True)),
        "pricing_rmse": {
            "regime": rmse(regime),
            "half_response": rmse(half_response),
            "dupire": rmse([quotes["dupire"]["premium"]] * len(signals)),
        },
    }
