import numpy as np
import pytest

import black_scholes as bs
from black_scholes import elw

# A real ka30012 response, KOSPI200 call, 2025-12.
SNAPSHOT = {
    "elwexec_pric": "535.00", "elwcnvt_rt": "100.0000", "cur_prc": "+4985",
    "elwsrvive_dys": "17", "elwrght_type": "CALL", "elwprty": "109.27",
    "elwgear": "11.72", "elwrght_exec_way": "유럽형", "kobarr": "0",
}
K = 535.00
RATIO = 100.0
MARKET = 4985.0
# The exchange does not publish the underlying level, but parity pins it.
S = K * 109.27 / 100.0


def test_parity_reproduces_the_exchange_figure():
    assert float(elw.parity(S, K, bs.CALL)) == pytest.approx(109.27, abs=0.01)


def test_gearing_reproduces_the_exchange_figure():
    # Independent confirmation that the conversion ratio multiplies rather
    # than divides: any other reading misses elwgear by two orders of magnitude.
    assert float(elw.gearing(S, MARKET, RATIO)) == pytest.approx(11.72, abs=0.01)


def test_put_parity_inverts_moneyness():
    # A put is in the money below the strike, so parity flips the ratio.
    assert float(elw.parity(S, K, bs.PUT)) == pytest.approx(K / S * 100.0)
    assert float(elw.parity(S, K, bs.PUT)) < 100 < float(elw.parity(S, K, bs.CALL))


# --- the guards ---------------------------------------------------------------

def test_plain_european_warrant_is_priceable():
    assert elw.is_priceable(SNAPSHOT["elwrght_exec_way"], SNAPSHOT["kobarr"])
    elw.assert_priceable(SNAPSHOT["elwrght_exec_way"], SNAPSHOT["kobarr"])


@pytest.mark.parametrize("style,barrier", [
    ("미국형", "0"),      # early exercise this model does not value
    ("유럽형", "250"),    # knock-out, i.e. a barrier option
    ("미국형", "250"),
])
def test_unpriceable_warrants_are_rejected(style, barrier):
    assert not elw.is_priceable(style, barrier)
    with pytest.raises(elw.ELWNotPriceable):
        elw.assert_priceable(style, barrier, code="TEST01")


def test_rejection_names_the_reason():
    with pytest.raises(elw.ELWNotPriceable, match="barrier"):
        elw.assert_priceable("유럽형", "250")
    with pytest.raises(elw.ELWNotPriceable, match="European"):
        elw.assert_priceable("미국형", "0")


def test_unreadable_barrier_is_not_silently_accepted():
    assert not elw.is_priceable("유럽형", "n/a")
    with pytest.raises(elw.ELWNotPriceable):
        elw.assert_priceable("유럽형", "n/a")


# --- the conversion ratio -----------------------------------------------------

@pytest.mark.parametrize("ratio", [1.0, 0.01, 100.0])
@pytest.mark.parametrize("kind", [bs.CALL, bs.PUT])
def test_price_scales_linearly_with_the_conversion_ratio(ratio, kind):
    args = (S, K, 0.25, 0.03, 0.30, 0.018)
    assert (float(elw.price(*args, conversion_ratio=ratio, kind=kind))
            == pytest.approx(ratio * float(bs.price(*args, kind=kind))))


@pytest.mark.parametrize("ratio", [1.0, 0.01, 100.0])
def test_every_greek_scales_with_the_conversion_ratio(ratio):
    args = (S, K, 0.25, 0.03, 0.30, 0.018)
    scaled = elw.greeks(*args, conversion_ratio=ratio, kind=bs.CALL)
    plain = bs.greeks(*args, kind=bs.CALL)
    for name in plain:
        assert float(scaled[name]) == pytest.approx(ratio * float(plain[name]))


@pytest.mark.parametrize("ratio", [1.0, 0.01, 100.0])
@pytest.mark.parametrize("kind", [bs.CALL, bs.PUT])
def test_implied_volatility_round_trips_through_the_ratio(ratio, kind):
    S_, K_, T, r, sigma, q = 300.0, 310.0, 0.4, 0.03, 0.28, 0.015
    quote = elw.price(S_, K_, T, r, sigma, q, ratio, kind)
    recovered = elw.implied_volatility(quote, S_, K_, T, r, q, ratio, kind)
    assert float(recovered) == pytest.approx(sigma, abs=1e-8)


def test_zero_conversion_ratio_yields_nan_rather_than_dividing_by_zero():
    assert np.isnan(float(elw.implied_volatility(100.0, S, K, 0.25, 0.03, 0.018,
                                                 conversion_ratio=0.0)))


# --- deep in the money: implied vol is not identifiable -----------------------

def test_deep_itm_quote_at_the_floor_returns_nan_not_a_fabricated_vol():
    # The snapshot's time value is ~26 won on a ~4959 won intrinsic. Under
    # r=3%, q=1.8% the quote sits below the no-arbitrage floor, and a solver
    # that reported a number here would be inventing one.
    iv = elw.implied_volatility(MARKET, S, K, 17 / 365, 0.03, 0.018, RATIO, bs.CALL)
    assert np.isnan(float(iv))


def test_deep_itm_implied_vol_is_hostage_to_the_carry_assumption():
    # Same quote, same day, three plausible dividend yields: the answer moves
    # by several vol points or vanishes entirely. This is why an IV screen has
    # to be restricted to warrants near the money.
    T = 17 / 365
    results = [elw.implied_volatility(MARKET, S, K, T, 0.03, q, RATIO, bs.CALL)
               for q in (0.010, 0.018, 0.025)]
    assert np.isnan(float(results[0]))
    assert np.isnan(float(results[1]))
    assert float(results[2]) == pytest.approx(0.2024, abs=0.005)


def test_near_the_money_implied_vol_survives_the_same_carry_spread():
    # The contrast with the case above. At the money the same 1.5 point
    # spread of dividend yields still answers, and moves the vol by about
    # one point rather than flipping between 20% and unidentifiable.
    S_atm, T = 535.0, 0.25
    quote = elw.price(S_atm, K, T, 0.03, 0.25, 0.018, RATIO, bs.CALL)
    ivs = [float(elw.implied_volatility(quote, S_atm, K, T, 0.03, q, RATIO, bs.CALL))
           for q in (0.010, 0.018, 0.025)]
    assert not any(np.isnan(v) for v in ivs)
    assert max(ivs) - min(ivs) < 0.015


# --- the remaining conventions ------------------------------------------------

def test_year_fraction_uses_calendar_days_by_default():
    assert float(elw.year_fraction(365)) == pytest.approx(1.0)
    assert float(elw.year_fraction(17)) == pytest.approx(17 / 365)
    assert float(elw.year_fraction(252, basis=252)) == pytest.approx(1.0)


def test_disparity_is_the_premium_over_model_value():
    assert float(elw.disparity(4985.0, 4347.95)) == pytest.approx(14.65, abs=0.01)
    assert float(elw.disparity(100.0, 100.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("kind,expected", [(bs.CALL, 535 + 49.85), (bs.PUT, 535 - 49.85)])
def test_breakeven_moves_the_strike_by_the_premium_per_unit(kind, expected):
    assert float(elw.breakeven_underlying(K, MARKET, RATIO, kind)) == pytest.approx(expected)


def test_screening_a_chain_vectorizes():
    strikes = np.array([520.0, 535.0, 550.0, 565.0])
    quotes = elw.price(S, strikes, 0.25, 0.03, 0.28, 0.018, RATIO, bs.CALL)
    ivs = elw.implied_volatility(quotes, S, strikes, 0.25, 0.03, 0.018, RATIO, bs.CALL)
    assert ivs.shape == strikes.shape
    assert ivs == pytest.approx(np.full(4, 0.28), abs=1e-8)


# --- implied vol is only as precise as the price grid -------------------------

def test_iv_resolution_flags_the_deep_itm_snapshot_as_unmeasurable():
    # The snapshot's vega is about 0.3 won per vol point against a five won
    # tick, so adjacent quotes are more than ten vol points apart.
    g = elw.greeks(S, K, 17 / 365, 0.03, 0.13, 0.018, RATIO, bs.CALL)
    assert float(elw.iv_resolution(float(g["vega"]) / 100)) > 10.0


def test_iv_resolution_passes_a_near_the_money_warrant():
    g = elw.greeks(S, 585.0, 45 / 365, 0.03, 0.13, 0.018, RATIO, bs.CALL)
    assert float(elw.iv_resolution(float(g["vega"]) / 100)) < 0.5


def test_zero_vega_has_no_resolution_at_all():
    assert float(elw.iv_resolution(0.0)) == np.inf


def test_iv_resolution_scales_with_the_tick():
    assert float(elw.iv_resolution(10.0, tick=5.0)) == pytest.approx(0.5)
    assert float(elw.iv_resolution(10.0, tick=10.0)) == pytest.approx(1.0)
