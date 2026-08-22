"""Does delta hedging actually replicate the option?

The Black-Scholes derivation claims a short option, hedged continuously
in the underlying, is riskless. This simulates that hedge on discrete
GBM paths and measures what is left over.

Two experiments:

  1. Hedging error shrinks as the rehedge interval shrinks. The residual
     is discretization, not model error.
  2. When realized vol differs from the vol the option was sold at, the
     leftover is not noise. Hedging at the implied vol locks in

         BS(sigma_implied) - BS(sigma_realized)

     regardless of the path taken, which is the edge a vol trader is
     actually pricing.
"""

import numpy as np

import black_scholes as bs

S0, K, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
N_PATHS = 20_000
SEED = 12345


def simulate(sigma_real, sigma_imp, n_steps, n_paths=N_PATHS, seed=SEED):
    """Sell one call at sigma_imp, delta hedge n_steps times under sigma_real.

    Returns the P&L per path, in premium units.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    z = rng.standard_normal((n_paths, n_steps))
    log_steps = (r - q - 0.5 * sigma_real ** 2) * dt + sigma_real * np.sqrt(dt) * z
    S = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)),
                                    np.cumsum(log_steps, axis=1)], axis=1))

    premium = float(bs.call_price(S0, K, T, r, sigma_imp, q))

    # Short the call, hold delta shares, keep the rest in cash at r.
    shares = bs.delta(S[:, 0], K, T, r, sigma_imp, q, bs.CALL)
    cash = premium - shares * S0

    for i in range(1, n_steps):
        cash *= np.exp(r * dt)
        tau = T - t[i]
        new_shares = bs.delta(S[:, i], K, tau, r, sigma_imp, q, bs.CALL)
        cash -= (new_shares - shares) * S[:, i]
        shares = new_shares

    cash *= np.exp(r * dt)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    return cash + shares * S[:, -1] - payoff, premium


print("=" * 74)
print("1. Hedging error vs rehedge frequency  (realized vol == implied vol == 20%)")
print("=" * 74)
print(f"{'rehedges':>10} {'mean P&L':>12} {'std P&L':>12} {'std / premium':>16}")
for n in (12, 52, 252, 1008):
    pnl, prem = simulate(0.20, 0.20, n)
    print(f"{n:>10} {pnl.mean():12.4f} {pnl.std():12.4f} {pnl.std() / prem:15.1%}")
print(f"\npremium sold = {prem:.4f}")
print("std falls roughly as 1/sqrt(n): the residual is discretization, not model error.")

print()
print("=" * 74)
print("2. Selling at the wrong vol  (252 rehedges, sold at implied 20%)")
print("=" * 74)
print(f"{'realized vol':>13} {'mean P&L':>12} {'predicted':>12} {'std P&L':>10}")
sigma_imp = 0.20
for sigma_real in (0.10, 0.15, 0.20, 0.25, 0.30):
    pnl, prem = simulate(sigma_real, sigma_imp, 252)
    predicted = float(bs.call_price(S0, K, T, r, sigma_imp, q)
                      - bs.call_price(S0, K, T, r, sigma_real, q))
    print(f"{sigma_real:12.0%} {pnl.mean():12.4f} {predicted:12.4f} {pnl.std():10.4f}")
print("\nShort an option and you are short gamma: you profit when the underlying")
print("moves less than the vol you sold, and lose when it moves more. The edge is")
print("set the moment you trade, and the hedge collects it whatever the path does.")
