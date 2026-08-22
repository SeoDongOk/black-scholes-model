"""Screen an ELW candidate list the way a selector would.

The input rows are shaped like Kiwoom's ka30012 response, so the parsing
here is what an ELW trading bot would do with the payload it already
receives. The point is the ordering: reject what the model cannot value,
then reject where the model cannot speak, and only then rank.
"""

import numpy as np

import black_scholes as bs
from black_scholes import elw

# Underlying level and carry are inputs the warrant feed does not carry.
KOSPI200 = 584.59
RISK_FREE = 0.030
DIV_YIELD = 0.018

# Reject a warrant whose implied vol moves by more than this between two
# adjacent quotes. Deep in the money the vega collapses toward the tick
# size, and the implied vol stops carrying information.
MAX_IV_POINTS_PER_TICK = 2.0

CANDIDATES = [
    # code, name, strike, ratio, quote, days, type, style, ko barrier
    ("58J123", "KOSPI200 콜 A", 535.00, 100.0, 4985, 17, "CALL", "유럽형", "0"),
    ("58J456", "KOSPI200 콜 B", 580.00, 100.0,  1210, 45, "CALL", "유럽형", "0"),
    ("58J789", "KOSPI200 콜 C", 585.00, 100.0,  1005, 45, "CALL", "유럽형", "0"),
    ("58J012", "KOSPI200 콜 D", 590.00, 100.0,   930, 45, "CALL", "유럽형", "0"),
    ("58J345", "KOSPI200 콜 E", 600.00, 100.0,   520, 45, "CALL", "유럽형", "0"),
    ("58J678", "조기종료 콜 F", 585.00, 100.0,   640, 45, "CALL", "유럽형", "560"),
    ("58J901", "미국형 콜 G",   585.00, 100.0,  1080, 45, "CALL", "미국형", "0"),
]


def screen(rows, S=KOSPI200, r=RISK_FREE, q=DIV_YIELD):
    ranked, rejected = [], []

    for code, name, K, ratio, quote, days, right, style, barrier in rows:
        kind = bs.CALL if right == "CALL" else bs.PUT

        if not elw.is_priceable(style, barrier):
            try:
                elw.assert_priceable(style, barrier, code)
            except elw.ELWNotPriceable as e:
                rejected.append((code, name, str(e).split(": ", 1)[1]))
            continue

        T = float(elw.year_fraction(days))
        iv = float(elw.implied_volatility(quote, S, K, T, r, q, ratio, kind))
        if np.isnan(iv):
            rejected.append((code, name, "quote outside the no-arbitrage bounds"))
            continue

        g = elw.greeks(S, K, T, r, iv, q, ratio, kind)
        vega_pt = float(g["vega"]) / 100
        resolution = float(elw.iv_resolution(vega_pt))
        if resolution > MAX_IV_POINTS_PER_TICK:
            rejected.append((code, name,
                             f"one tick moves implied vol by {resolution:.1f} points; "
                             f"vega is only {vega_pt:.2f} won per point"))
            continue

        par = float(elw.parity(S, K, kind))
        ranked.append({
            "code": code, "name": name, "iv": iv, "parity": par, "quote": quote,
            "delta": float(g["delta"]), "theta_day": float(g["theta"]) / 365,
            "vega_pt": vega_pt, "resolution": resolution,
            "breakeven": float(elw.breakeven_underlying(K, quote, ratio, kind)),
        })

    ranked.sort(key=lambda x: x["iv"])
    return ranked, rejected


ranked, rejected = screen(CANDIDATES)

print(f"KOSPI200 {KOSPI200}   r={RISK_FREE:.1%}   q={DIV_YIELD:.1%}\n")
print("ranked cheapest first, by implied volatility")
print(f"{'code':<8}{'name':<16}{'IV':>8}{'parity':>8}{'quote':>7}"
      f"{'delta':>8}{'theta/d':>9}{'vega/pt':>9}{'+-IV':>7}{'breakeven':>11}")
for x in ranked:
    print(f"{x['code']:<8}{x['name']:<16}{x['iv']:7.2%}{x['parity']:8.1f}"
          f"{x['quote']:7,.0f}{x['delta']:8.1f}{x['theta_day']:9.1f}"
          f"{x['vega_pt']:9.1f}{x['resolution']:7.2f}{x['breakeven']:11.2f}")

print("\nrejected")
for code, name, why in rejected:
    print(f"  {code} {name:<16} {why}")

if ranked:
    best = ranked[0]
    print(f"\npick {best['code']} at {best['iv']:.2%} implied, against "
          f"{ranked[-1]['iv']:.2%} for the most expensive candidate.")
    print(f"the current selector would instead take whichever of these traded most.")
