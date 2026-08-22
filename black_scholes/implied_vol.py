"""Implied volatility by numerical inversion of the pricing formula.

The price is strictly increasing in sigma, so a bracketing solver is
both safe and fast. Brent's method is used rather than Newton because
vega collapses toward zero for deep in- and out-of-the-money options,
where Newton steps become unstable.
"""

import numpy as np
from scipy.optimize import brentq

from .core import CALL, _broadcast, _normalize_kind, price

__all__ = ["implied_volatility", "price_bounds"]

SIGMA_LO = 1e-9
SIGMA_HI = 10.0


def price_bounds(S, K, T, r, q=0.0, kind=CALL):
    """No-arbitrage (lower, upper) bounds on the option price.

    A quote outside these bounds cannot be produced by any volatility,
    so implied_volatility returns nan for it.
    """
    S, K, T, r, _, q = _broadcast(S, K, T, r, 0.0, q)
    w = _normalize_kind(kind)
    fwd_pv = S * np.exp(-q * T)
    strike_pv = K * np.exp(-r * T)
    lower = np.maximum(w * (fwd_pv - strike_pv), 0.0)
    upper = np.where(w > 0, fwd_pv, strike_pv)
    return lower, upper


def _solve_one(target, S, K, T, r, q, w, tol, maxiter):
    if not np.isfinite(target) or T <= 0:
        return np.nan

    lo = max(0.0, w * (S * np.exp(-q * T) - K * np.exp(-r * T)))
    hi = S * np.exp(-q * T) if w > 0 else K * np.exp(-r * T)
    # Bounds are open; a quote sitting on them implies 0 or infinite vol.
    if target <= lo + tol or target >= hi - tol:
        return np.nan

    def objective(s):
        return float(price(S, K, T, r, s, q, kind=w)) - target

    if objective(SIGMA_HI) < 0:
        # Priced above anything a 1000% vol can produce.
        return np.nan
    return brentq(objective, SIGMA_LO, SIGMA_HI, xtol=tol, maxiter=maxiter)


def implied_volatility(target_price, S, K, T, r, q=0.0, kind=CALL,
                       tol=1e-8, maxiter=100):
    """Volatility that reproduces target_price, or nan if none exists.

    Returns nan rather than raising when the quote violates the
    no-arbitrage bounds, so a whole option chain can be inverted in one
    call without the bad rows aborting the good ones.
    """
    target = np.asarray(target_price, dtype=float)
    S, K, T, r, _, q = _broadcast(S, K, T, r, 0.0, q)
    w = _normalize_kind(kind)
    target, S, K, T, r, q, w = np.broadcast_arrays(target, S, K, T, r, q, w)

    out = np.empty(target.shape, dtype=float)
    flat = [x.ravel() for x in (out, target, S, K, T, r, q, w)]
    o, *cols = flat
    for i in range(o.size):
        o[i] = _solve_one(*(c[i] for c in cols), tol=tol, maxiter=maxiter)
    result = o.reshape(target.shape)
    return result if result.ndim else result[()]
