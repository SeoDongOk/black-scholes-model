import numpy as np
import pytest

import black_scholes as bs

# (S, K, T, r, sigma, q) spanning ITM/ATM/OTM, short/long dated, with and
# without carry.
CASES = [
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),
    (100.0, 120.0, 0.50, 0.03, 0.35, 0.02),
    (100.0,  80.0, 2.00, 0.01, 0.15, 0.04),
    ( 42.0,  40.0, 0.25, 0.07, 0.45, 0.00),
    (250.0, 260.0, 0.08, 0.04, 0.60, 0.01),
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.05),  # q == r, i.e. Black-76
]
KINDS = [bs.CALL, bs.PUT]


def test_textbook_value():
    # Hull, Options Futures and Other Derivatives: S=K=100, T=1, r=5%, vol=20%.
    assert bs.call_price(100, 100, 1, 0.05, 0.20) == pytest.approx(10.450584, abs=1e-6)
    assert bs.put_price(100, 100, 1, 0.05, 0.20) == pytest.approx(5.573526, abs=1e-6)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_put_call_parity(S, K, T, r, sigma, q):
    c = bs.call_price(S, K, T, r, sigma, q)
    p = bs.put_price(S, K, T, r, sigma, q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), rel=1e-12)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_parity_holds_for_delta_and_theta(S, K, T, r, sigma, q):
    args = (S, K, T, r, sigma, q)
    assert (bs.delta(*args, kind=bs.CALL) - bs.delta(*args, kind=bs.PUT)
            == pytest.approx(np.exp(-q * T), rel=1e-12))
    assert (bs.gamma(*args) == pytest.approx(bs.gamma(*args), rel=1e-12))


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_price_within_no_arbitrage_bounds(S, K, T, r, sigma, q, kind):
    lo, hi = bs.price_bounds(S, K, T, r, q, kind)
    v = bs.price(S, K, T, r, sigma, q, kind)
    assert lo <= v <= hi


# --- greeks against central finite differences -------------------------------

def _central(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2 * h)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_delta_matches_finite_difference(S, K, T, r, sigma, q, kind):
    fd = _central(lambda s: bs.price(s, K, T, r, sigma, q, kind), S, S * 1e-5)
    assert bs.delta(S, K, T, r, sigma, q, kind) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_gamma_matches_finite_difference(S, K, T, r, sigma, q, kind):
    fd = _central(lambda s: bs.delta(s, K, T, r, sigma, q, kind), S, S * 1e-5)
    assert bs.gamma(S, K, T, r, sigma, q) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_vega_matches_finite_difference(S, K, T, r, sigma, q, kind):
    fd = _central(lambda v: bs.price(S, K, T, r, v, q, kind), sigma, 1e-6)
    assert bs.vega(S, K, T, r, sigma, q) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_rho_matches_finite_difference(S, K, T, r, sigma, q, kind):
    fd = _central(lambda rr: bs.price(S, K, T, rr, sigma, q, kind), r, 1e-7)
    assert bs.rho(S, K, T, r, sigma, q, kind) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_theta_matches_finite_difference(S, K, T, r, sigma, q, kind):
    # theta is dV/dt while the formula differentiates in T, hence the sign.
    fd = -_central(lambda t: bs.price(S, K, t, r, sigma, q, kind), T, T * 1e-6)
    assert bs.theta(S, K, T, r, sigma, q, kind) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_vanna_matches_finite_difference(S, K, T, r, sigma, q, kind):
    fd = _central(lambda v: bs.delta(S, K, T, r, v, q, kind), sigma, 1e-6)
    assert bs.vanna(S, K, T, r, sigma, q) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_vomma_matches_finite_difference(S, K, T, r, sigma, q):
    fd = _central(lambda v: bs.vega(S, K, T, r, v, q), sigma, 1e-6)
    assert bs.vomma(S, K, T, r, sigma, q) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_charm_matches_finite_difference(S, K, T, r, sigma, q, kind):
    fd = -_central(lambda t: bs.delta(S, K, t, r, sigma, q, kind), T, T * 1e-6)
    assert bs.charm(S, K, T, r, sigma, q, kind) == pytest.approx(fd, rel=1e-4)


# --- implied volatility -------------------------------------------------------

@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_implied_volatility_round_trips(S, K, T, r, sigma, q, kind):
    v = bs.price(S, K, T, r, sigma, q, kind)
    assert bs.implied_volatility(v, S, K, T, r, q, kind) == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize("kind", KINDS)
def test_implied_volatility_is_nan_outside_bounds(kind):
    lo, hi = bs.price_bounds(100, 100, 1, 0.05, 0.0, kind)
    assert np.isnan(bs.implied_volatility(float(lo) - 1.0, 100, 100, 1, 0.05, 0.0, kind))
    assert np.isnan(bs.implied_volatility(float(hi) + 1.0, 100, 100, 1, 0.05, 0.0, kind))


def test_implied_volatility_vectorizes_over_a_chain():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.array([0.30, 0.25, 0.20, 0.22, 0.28])
    quotes = bs.call_price(100.0, strikes, 0.5, 0.03, vols)
    recovered = bs.implied_volatility(quotes, 100.0, strikes, 0.5, 0.03)
    assert recovered == pytest.approx(vols, abs=1e-8)


# --- degenerate inputs --------------------------------------------------------

@pytest.mark.parametrize("kind,expected", [(bs.CALL, 5.0), (bs.PUT, 0.0)])
def test_expired_option_is_worth_intrinsic(kind, expected):
    assert bs.price(105, 100, 0.0, 0.05, 0.2, 0.0, kind) == pytest.approx(expected)


@pytest.mark.parametrize("kind", KINDS)
def test_zero_vol_is_discounted_intrinsic_on_the_forward(kind):
    S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.01
    fwd = S * np.exp((r - q) * T)
    w = 1.0 if kind == bs.CALL else -1.0
    expected = np.exp(-r * T) * max(w * (fwd - K), 0.0)
    assert bs.price(S, K, T, r, 0.0, q, kind) == pytest.approx(expected)


@pytest.mark.parametrize("T,sigma", [(0.0, 0.2), (1.0, 0.0)])
def test_second_order_greeks_vanish_when_diffusion_does(T, sigma):
    for fn in (bs.gamma, bs.vega, bs.vanna, bs.vomma):
        assert fn(100, 100, T, 0.05, sigma) == 0.0


def test_deep_out_of_the_money_stays_finite_and_nonnegative():
    v = bs.call_price(100.0, 10_000.0, 0.01, 0.05, 0.10)
    assert np.isfinite(v) and v >= 0.0


# --- interface ----------------------------------------------------------------

def test_broadcasting_shapes():
    S = np.linspace(80, 120, 5).reshape(5, 1)
    T = np.array([0.25, 0.5, 1.0]).reshape(1, 3)
    assert bs.call_price(S, 100.0, T, 0.05, 0.2).shape == (5, 3)


def test_greeks_dict_agrees_with_individual_functions():
    args = (100.0, 105.0, 0.75, 0.04, 0.25, 0.01)
    g = bs.greeks(*args, kind=bs.PUT)
    assert g["price"] == pytest.approx(bs.price(*args, kind=bs.PUT))
    assert g["vomma"] == pytest.approx(bs.vomma(*args))
    assert set(g) == {"price", "delta", "gamma", "vega", "theta", "rho",
                      "vanna", "vomma", "charm"}


def test_rejects_bad_kind():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.05, 0.2, 0.0, "straddle")


def test_black76_equals_carry_free_pricing():
    # q == r removes the drift, so the option prices off the forward alone.
    F, K, T, r, sigma = 100.0, 95.0, 1.0, 0.05, 0.3
    from scipy.stats import norm
    dd1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    dd2 = dd1 - sigma * np.sqrt(T)
    expected = np.exp(-r * T) * (F * norm.cdf(dd1) - K * norm.cdf(dd2))
    assert bs.call_price(F, K, T, r, sigma, q=r) == pytest.approx(expected, rel=1e-12)
