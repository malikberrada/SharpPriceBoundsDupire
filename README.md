# Pricing an index put after a public volatility signal

**Abdelmalik Berrada** · Independent Quantitative Researcher, Casablanca, Morocco

This repository reproduces the post-signal pricing experiment in Section 7 of *Sharp price bounds for index puts from one additional wing quote*. A regime model and its spot-only Dupire local-volatility projection match the same initial European vanilla surface. At the same spot after a public signal, the regime model assigns different values to a put in the calm and stress states. The projection gives a single price.

## Run the experiment

Use Python 3.12. From the repository root, run

```console
python -m pip install -r requirements.txt
python compare_signal.py
python -m unittest discover -s tests -v
```

The calculation runs offline once NumPy and SciPy are installed. It constructs the volatility mixture, solves the backward pricing equation for the projected diffusion on two grids, and checks pricing errors against independently calculated conditional Black–Scholes mixture prices.

## Prices and errors

At year 0.5, spot is 100 and a European put with strike 90 has six months left to expiry. Volatility up to that date is 20%. The calm and stress signals are equally likely. The conditional probability of 40% volatility over the remaining six months is 15% after a calm signal and 80% after a stress signal. The other possible volatility is 10%.

| Model response | Calm price | Stress price | Pricing RMSE |
|---|---:|---:|---:|
| Regime model, full response | 1.133 | 5.169 | 0.000 |
| Regime model, half response | 2.142 | 4.160 | 1.009 |
| Spot-only Dupire projection | 3.141 | 3.141 | 2.018 |

All entries are in index points. The full-response model uses the specified conditional mixture, so its zero pricing error follows by construction. Halving the response moves each conditional price halfway toward its probability-weighted mean. This reduces the pricing RMSE by 50% relative to the spot-only projection in this experiment. Refining the Dupire grid from 600 to 1,200 spatial intervals and from 160 to 320 time steps changes the projected put price by about 0.000532 points.

## Implementation

The projection weights each volatility component by its density at the current time and spot. The backward solver uses Crank–Nicolson with two initial implicit steps. The independent scoring routine prices each lognormal component analytically and combines the errors using the two signal probabilities.

[Mixture and projection](regime_pricing/projection.py) · [Independent pricing check](regime_pricing/signal_scores.py) · [Regression tests](tests/test_signal.py)
