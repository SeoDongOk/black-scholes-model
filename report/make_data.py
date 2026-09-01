"""Generate the resolution-map data the report renders.

Writes report/data.js. Re-run after changing any assumption below.
"""

import json
import pathlib

import numpy as np

import black_scholes as bs
from black_scholes import elw

# The snapshot the report opens with: a KOSPI200 call warrant.
SPOT = 584.59        # implied by the exchange's own parity of 109.27 on a 535 strike
TICK = 5.0           # won, the ELW quote increment
RATIO = 100.0        # conversion ratio, confirmed against the exchange's gearing
RATE = 0.030
DIVIDEND = 0.018
SIGMA = 0.15

STRIKES = np.arange(380, 821, 10.0)
DAYS = np.array([3, 5, 7, 10, 14, 20, 30, 45, 60, 90, 120, 180], float)

MEASURABLE = 2.0     # vol points per tick; past this the implied vol is rounding


def build():
    K, D = np.meshgrid(STRIKES, DAYS)
    greeks = elw.greeks(SPOT, K, D / 365, RATE, SIGMA, DIVIDEND, RATIO, bs.CALL)
    res = elw.iv_resolution(greeks["vega"] / 100, TICK)
    res = np.where(np.isfinite(res), np.minimum(res, 100.0), 100.0)

    band = []
    for row in res:
        ok = STRIKES[row < MEASURABLE]
        band.append([round(float(ok.min() / SPOT * 100), 1),
                     round(float(ok.max() / SPOT * 100), 1)] if len(ok) else [100.0, 100.0])

    return {
        "spot": SPOT, "tick": TICK, "sigma": SIGMA,
        "moneyness": [round(float(k / SPOT * 100), 2) for k in STRIKES],
        "days": DAYS.tolist(),
        "res": np.round(res, 2).tolist(),
        "band": band,
    }


if __name__ == "__main__":
    out = pathlib.Path(__file__).with_name("data.js")
    out.write_text("const DATA = " + json.dumps(build(), separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
