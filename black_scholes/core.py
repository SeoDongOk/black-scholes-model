"""Black-Scholes-Merton pricing and greeks.

All functions are vectorized over their numeric arguments via numpy
broadcasting, and share the same parameterization:

    S      spot price of the underlying
    K      strike price
    T      time to expiry, in years
    r      continuously compounded risk-free rate
    sigma  volatility, annualized, as a decimal (0.20 == 20%)
    q      continuous dividend yield (default 0)

The dividend yield doubles as a cost-of-carry knob, so a single
implementation covers the usual variants:

    q = 0            non-dividend-paying stock (Black-Scholes 1973)
    q = dividend     dividend-paying stock or index (Merton 1973)
    q = r            option on a future (Black 1976)
    q = r_foreign    currency option (Garman-Kohlhagen 1983)

Degenerate inputs (T <= 0 or sigma <= 0) are not errors: the option
value collapses to the discounted intrinsic value on the forward, and
the greeks to their limiting values.
"""

import numpy as np
from scipy.stats import norm

__all__ = [
    "forward", "d1", "d2", "price", "call_price", "put_price",
    "delta", "gamma", "vega", "theta", "rho",
    "vanna", "vomma", "charm", "greeks",
]

CALL = "call"
PUT = "put"


def _normalize_kind(kind):
    """Map a call/put flag onto +1 / -1."""
    k = np.asarray(kind)
    if k.dtype.kind in "US":
        lowered = np.char.lower(k.astype(str))
        if not np.all(np.isin(lowered, [CALL, PUT])):
            bad = np.unique(lowered[~np.isin(lowered, [CALL, PUT])])
            raise ValueError(f"kind must be 'call' or 'put', got {bad.tolist()}")
        return np.where(lowered == CALL, 1.0, -1.0)
    if not np.all(np.isin(k, [1, -1])):
        raise ValueError("numeric kind must be +1 (call) or -1 (put)")
    return k.astype(float)


def _broadcast(S, K, T, r, sigma, q):
    return np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                 for x in (S, K, T, r, sigma, q)))


def _degenerate(T, sigma):
    """Mask of inputs with no diffusion left: expired or zero vol."""
    return (T <= 0) | (sigma <= 0)


def forward(S, K=None, T=0.0, r=0.0, q=0.0):
    """Forward price S * exp((r - q) * T). K is accepted but unused."""
    S, T, r, q = (np.asarray(x, dtype=float) for x in (S, T, r, q))
    return S * np.exp((r - q) * T)


def _d1_d2(S, K, T, r, sigma, q):
    """d1, d2 with infinities substituted where the diffusion vanishes.

    Setting d1 = d2 = +/-inf according to moneyness of the forward makes
    N(d1), N(d2) collapse to the correct 0/1 limits, so the pricing
    formula stays valid without a separate branch.
    """
    vol_t = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        _d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_t
        _d2 = _d1 - vol_t

    deg = _degenerate(T, sigma)
    if np.any(deg):
        fwd = S * np.exp((r - q) * T)
        limit = np.where(fwd > K, np.inf, np.where(fwd < K, -np.inf, 0.0))
        _d1 = np.where(deg, limit, _d1)
        _d2 = np.where(deg, limit, _d2)
    return _d1, _d2


def d1(S, K, T, r, sigma, q=0.0):
    return _d1_d2(*_broadcast(S, K, T, r, sigma, q))[0]


def d2(S, K, T, r, sigma, q=0.0):
    return _d1_d2(*_broadcast(S, K, T, r, sigma, q))[1]


def price(S, K, T, r, sigma, q=0.0, kind=CALL):
    """Black-Scholes-Merton value of a European option."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    w = _normalize_kind(kind)
    _dd1, _dd2 = _d1_d2(S, K, T, r, sigma, q)
    return w * (S * np.exp(-q * T) * norm.cdf(w * _dd1)
                - K * np.exp(-r * T) * norm.cdf(w * _dd2))


def call_price(S, K, T, r, sigma, q=0.0):
    return price(S, K, T, r, sigma, q, CALL)


def put_price(S, K, T, r, sigma, q=0.0):
    return price(S, K, T, r, sigma, q, PUT)


def delta(S, K, T, r, sigma, q=0.0, kind=CALL):
    """dV/dS."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    w = _normalize_kind(kind)
    _dd1, _ = _d1_d2(S, K, T, r, sigma, q)
    return w * np.exp(-q * T) * norm.cdf(w * _dd1)


def gamma(S, K, T, r, sigma, q=0.0):
    """d2V/dS2. Identical for calls and puts."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    _dd1, _ = _d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-q * T) * norm.pdf(_dd1) / (S * sigma * np.sqrt(T))
    return np.where(_degenerate(T, sigma), 0.0, out)


def vega(S, K, T, r, sigma, q=0.0):
    """dV/dsigma, per 1.00 of vol. Divide by 100 for a 1-vol-point move."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    _dd1, _ = _d1_d2(S, K, T, r, sigma, q)
    out = S * np.exp(-q * T) * norm.pdf(_dd1) * np.sqrt(T)
    return np.where(_degenerate(T, sigma), 0.0, out)


def theta(S, K, T, r, sigma, q=0.0, kind=CALL):
    """dV/dt, per year. Divide by 365 for a calendar-day decay."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    w = _normalize_kind(kind)
    _dd1, _dd2 = _d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = -S * np.exp(-q * T) * norm.pdf(_dd1) * sigma / (2 * np.sqrt(T))
    decay = np.where(_degenerate(T, sigma), 0.0, decay)
    carry = w * q * S * np.exp(-q * T) * norm.cdf(w * _dd1)
    disc = -w * r * K * np.exp(-r * T) * norm.cdf(w * _dd2)
    return decay + carry + disc


def rho(S, K, T, r, sigma, q=0.0, kind=CALL):
    """dV/dr at fixed q, per 1.00 of rate. Divide by 10000 for a basis point."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    w = _normalize_kind(kind)
    _, _dd2 = _d1_d2(S, K, T, r, sigma, q)
    return w * K * T * np.exp(-r * T) * norm.cdf(w * _dd2)


def vanna(S, K, T, r, sigma, q=0.0):
    """d2V/dS dsigma. Identical for calls and puts."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    _dd1, _dd2 = _d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.exp(-q * T) * norm.pdf(_dd1) * _dd2 / sigma
    return np.where(_degenerate(T, sigma), 0.0, out)


def vomma(S, K, T, r, sigma, q=0.0):
    """d2V/dsigma2, also known as volga. Identical for calls and puts."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    _dd1, _dd2 = _d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vega(S, K, T, r, sigma, q) * _dd1 * _dd2 / sigma
    return np.where(_degenerate(T, sigma), 0.0, out)


def charm(S, K, T, r, sigma, q=0.0, kind=CALL):
    """d(delta)/dt, per year: how the hedge ratio drifts as expiry nears."""
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    w = _normalize_kind(kind)
    _dd1, _dd2 = _d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        drift = (np.exp(-q * T) * norm.pdf(_dd1)
                 * (2 * (r - q) * T - _dd2 * sigma * np.sqrt(T))
                 / (2 * T * sigma * np.sqrt(T)))
    drift = np.where(_degenerate(T, sigma), 0.0, drift)
    return w * q * np.exp(-q * T) * norm.cdf(w * _dd1) - drift


def greeks(S, K, T, r, sigma, q=0.0, kind=CALL):
    """Every quantity above in one pass, as a dict."""
    args = (S, K, T, r, sigma, q)
    return {
        "price": price(*args, kind=kind),
        "delta": delta(*args, kind=kind),
        "gamma": gamma(*args),
        "vega": vega(*args),
        "theta": theta(*args, kind=kind),
        "rho": rho(*args, kind=kind),
        "vanna": vanna(*args),
        "vomma": vomma(*args),
        "charm": charm(*args, kind=kind),
    }
