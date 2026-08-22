"""KRX ELW conventions layered on the Black-Scholes core.

A Korean ELW is a warrant quoted in won per warrant, where one warrant
settles for `conversion_ratio` units of the underlying:

    payoff = max(w * (S_T - K), 0) * conversion_ratio

Price and every greek therefore scale by that one constant, which is the
single most common source of error when pricing an ELW off an index
level. Nothing here re-derives the model; it only applies the ratio and
the market conventions the KRX quotes in.

Black-Scholes only values a plain European warrant. Two attributes in
the exchange feed decide whether it applies at all:

    exercise style   must be European; an American warrant needs a tree
    KO barrier       must be absent; a knock-out warrant is a barrier
                     option and is worth strictly less than this model says

`assert_priceable` turns both into a hard precondition rather than a
comment, so a warrant this model cannot value fails loudly instead of
being quietly mispriced.
"""

import numpy as np

from . import core
from .implied_vol import implied_volatility as _implied_volatility

__all__ = [
    "ELWNotPriceable", "assert_priceable", "is_priceable",
    "year_fraction", "price", "greeks", "implied_volatility",
    "parity", "gearing", "disparity", "breakeven_underlying",
    "iv_resolution", "TICK_SIZE_WON",
]

EUROPEAN = "유럽형"
AMERICAN = "미국형"


class ELWNotPriceable(ValueError):
    """Raised for a warrant outside what Black-Scholes can value."""


def is_priceable(exercise_style, ko_barrier=0):
    """Whether Black-Scholes applies to this warrant.

    exercise_style is the exchange's own string, e.g. ka30012's
    `elwrght_exec_way`. ko_barrier is its `kobarr`, zero when absent.
    """
    style_ok = str(exercise_style).strip() in (EUROPEAN, "European", "EUROPEAN")
    try:
        barrier_ok = float(ko_barrier or 0) == 0.0
    except (TypeError, ValueError):
        barrier_ok = False
    return style_ok and barrier_ok


def assert_priceable(exercise_style, ko_barrier=0, code=""):
    """Reject a warrant this model cannot value, naming the reason."""
    style = str(exercise_style).strip()
    if style not in (EUROPEAN, "European", "EUROPEAN"):
        raise ELWNotPriceable(
            f"{code or 'warrant'}: exercise style {style!r} is not European; "
            "an American warrant carries early-exercise value this model omits"
        )
    try:
        barrier = float(ko_barrier or 0)
    except (TypeError, ValueError):
        raise ELWNotPriceable(f"{code or 'warrant'}: unreadable KO barrier {ko_barrier!r}")
    if barrier != 0.0:
        raise ELWNotPriceable(
            f"{code or 'warrant'}: knock-out barrier at {barrier} makes this a "
            "barrier option, worth strictly less than the plain European value"
        )


def year_fraction(survive_days, basis=365.0):
    """Convert the exchange's remaining-days count into years.

    ELW settles on a calendar-day schedule, so 365 is the natural basis.
    Pass 252 to quote time in trading days instead; the choice matters
    most for short-dated warrants, where theta dominates.
    """
    return np.asarray(survive_days, dtype=float) / basis


def price(S, K, T, r, sigma, q=0.0, conversion_ratio=1.0, kind=core.CALL):
    """Theoretical warrant value in won."""
    return np.asarray(conversion_ratio, dtype=float) * core.price(S, K, T, r, sigma, q, kind)


def greeks(S, K, T, r, sigma, q=0.0, conversion_ratio=1.0, kind=core.CALL):
    """Every greek, scaled to one warrant."""
    ratio = np.asarray(conversion_ratio, dtype=float)
    return {k: ratio * v
            for k, v in core.greeks(S, K, T, r, sigma, q, kind).items()}


def implied_volatility(warrant_price, S, K, T, r, q=0.0, conversion_ratio=1.0,
                       kind=core.CALL, **kwargs):
    """Volatility implied by a warrant quote, or nan if none exists.

    Dividing the quote by the conversion ratio is what puts it back on
    the same scale as the underlying, which is where the inversion runs.
    """
    ratio = np.asarray(conversion_ratio, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_unit = np.asarray(warrant_price, dtype=float) / ratio
    per_unit = np.where(ratio > 0, per_unit, np.nan)
    return _implied_volatility(per_unit, S, K, T, r, q, kind, **kwargs)


# --- conventions the exchange quotes alongside the price ---------------------

def parity(S, K, kind=core.CALL):
    """Moneyness in percent: 100 is at the money, above 100 is in the money."""
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    w = core._normalize_kind(kind)
    return np.where(w > 0, S / K, K / S) * 100.0


def gearing(S, warrant_price, conversion_ratio=1.0):
    """Underlying value controlled per won of warrant."""
    S = np.asarray(S, dtype=float)
    ratio = np.asarray(conversion_ratio, dtype=float)
    warrant_price = np.asarray(warrant_price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return S * ratio / warrant_price


def disparity(market_price, theoretical_price):
    """Percent by which the market quote exceeds the model value.

    Positive means the warrant is being sold above what the model says
    it is worth, which for an ELW is the normal state: the liquidity
    provider quotes a spread and is the only meaningful counterparty.
    """
    market_price = np.asarray(market_price, dtype=float)
    theoretical_price = np.asarray(theoretical_price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (market_price / theoretical_price - 1.0) * 100.0


TICK_SIZE_WON = 5.0


def iv_resolution(vega, tick=TICK_SIZE_WON):
    """Vol points spanned by one tick of the quote.

    Implied vol is only as precise as the price grid it is inverted from.
    A deep in-the-money warrant can carry a vega of a fraction of a won
    per vol point while still quoting on a five won tick, so its implied
    vol moves by tens of points between adjacent quotes and means
    nothing. Screening on implied vol without checking this ranks
    rounding noise.

    vega is per one vol point, i.e. the greek divided by 100.
    """
    vega = np.abs(np.asarray(vega, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vega > 0, np.asarray(tick, dtype=float) / vega, np.inf)


def breakeven_underlying(K, warrant_price, conversion_ratio=1.0, kind=core.CALL):
    """Underlying level at which the warrant repays its purchase price at expiry."""
    K = np.asarray(K, dtype=float)
    ratio = np.asarray(conversion_ratio, dtype=float)
    warrant_price = np.asarray(warrant_price, dtype=float)
    w = core._normalize_kind(kind)
    with np.errstate(divide="ignore", invalid="ignore"):
        return K + w * warrant_price / ratio
