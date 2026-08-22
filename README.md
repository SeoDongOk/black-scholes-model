# black-scholes-model

Proof of the Black-Scholes equation, and a Black-Scholes model built in Python.

- `proof/` — derivation of the Black-Scholes PDE
- `black_scholes/` — a vectorized Black-Scholes-Merton pricer, greeks, and implied volatility
- `examples/demo.py` — prices an option chain, inverts it back to a smile, prints the greeks
- `GME.ipynb` — geometric Brownian motion paths

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Use

```python
import black_scholes as bs

bs.call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)   # 10.450584
bs.greeks(100, 100, 1.0, 0.05, 0.20, kind=bs.PUT)        # every greek in one pass
bs.implied_volatility(10.450584, 100, 100, 1.0, 0.05)    # 0.20
```

Every function broadcasts over its arguments, so a whole chain prices in one call:

```python
strikes = np.array([90.0, 100.0, 110.0])
vols    = np.array([0.26, 0.22, 0.23])
bs.call_price(100.0, strikes, 0.5, 0.03, vols, q=0.01)
```

## Parameters

| | |
|---|---|
| `S` | spot price |
| `K` | strike |
| `T` | time to expiry, in years |
| `r` | continuously compounded risk-free rate |
| `sigma` | annualized volatility, as a decimal (`0.20` is 20%) |
| `q` | continuous dividend yield, default `0` |

`q` doubles as the cost-of-carry knob, so one implementation covers the usual variants:

| `q` | model |
|---|---|
| `0` | non-dividend-paying stock — Black-Scholes (1973) |
| dividend yield | dividend-paying stock or index — Merton (1973) |
| `r` | option on a future — Black (1976) |
| foreign rate | currency option — Garman-Kohlhagen (1983) |

## Greeks

`delta`, `gamma`, `vega`, `theta`, `rho`, plus the second-order `vanna`, `vomma`, and `charm`.

They are returned unscaled, in the natural units of the formula. Rescale at the call site:

| greek | returned per | common quote |
|---|---|---|
| `vega`, `vanna` | `1.00` of vol | `/ 100` for a vol point |
| `vomma` | `1.00` of vol squared | `/ 100**2` |
| `theta`, `charm` | year | `/ 365` for a calendar day |
| `rho` | `1.00` of rate | `/ 10_000` for a basis point |

## Degenerate inputs

`T <= 0` or `sigma <= 0` are not errors. The value collapses to the discounted intrinsic
value on the forward, and the second-order greeks to zero.

`implied_volatility` returns `nan` — rather than raising — for a quote that violates the
no-arbitrage bounds, so one bad row does not abort a whole chain. `price_bounds` reports
those bounds directly.

## Tests

```bash
pytest
```

141 tests. Prices are checked against textbook values and put-call parity; every greek is
checked against a central finite difference of the function below it. The pricing formula
is separately cross-checked against a 4M-path Monte Carlo (all cases within 1.6 standard
errors).
