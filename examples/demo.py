"""Price an option chain, invert it back to a smile, and show the greeks."""

import numpy as np

import black_scholes as bs

S, T, r, q = 100.0, 0.5, 0.03, 0.01

# A chain quoted with a skew: downside strikes carry the higher vol.
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
vols = np.array([0.32, 0.26, 0.22, 0.23, 0.27])
quotes = bs.call_price(S, strikes, T, r, vols, q)

print(f"spot={S}  T={T}y  r={r:.1%}  q={q:.1%}  forward={float(bs.forward(S, T=T, r=r, q=q)):.4f}\n")

print("strike     vol      call     recovered vol")
recovered = bs.implied_volatility(quotes, S, strikes, T, r, q)
for k, v, c, iv in zip(strikes, vols, quotes, recovered):
    print(f"{k:6.1f}  {v:6.2%}  {c:8.4f}      {iv:6.2%}")

print("\ngreeks, at-the-money call")
g = bs.greeks(S, 100.0, T, r, 0.22, q, kind=bs.CALL)
units = {
    "price": ("", 1.0), "delta": ("per 1.00 of spot", 1.0),
    "gamma": ("per 1.00 of spot^2", 1.0), "vega": ("per vol point", 1 / 100),
    "theta": ("per calendar day", 1 / 365), "rho": ("per basis point", 1 / 10_000),
    "vanna": ("per vol point", 1 / 100), "vomma": ("per vol point^2", 1 / 100 ** 2),
    "charm": ("per calendar day", 1 / 365),
}
for name, value in g.items():
    label, scale = units[name]
    print(f"  {name:<6} {float(value) * scale:12.6f}  {label}")
