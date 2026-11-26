"""Recompute the prices and errors in the public volatility signal experiment."""

import json

from regime_pricing.projection import projection_demo
from regime_pricing.signal_scores import signal_price_scores


def main():
    smile = projection_demo()
    strike = smile["single_period_hedge_strike"]
    quotes = smile["quotes"][str(strike)]
    scores = signal_price_scores(smile)
    report = {
        "spot": smile["spot"],
        "strike": strike,
        "signal_time_years": smile["reveal"],
        "maturity_years": smile["maturity"],
        "conditional_put_prices": {
            state: format(quotes[state]["premium"], ".3f")
            for state in ("calm", "stress", "dupire")
        },
        "dupire_price_change_on_refinement": format(quotes["grid_change"]["premium"], ".6f"),
        **scores,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
