# report

`resolution.html` — a note on where implied volatility stops being measurable,
built from the pricer in this repository and a real KOSPI200 ELW quote.

`make_data.py` regenerates `data.js`, which the page inlines:

```bash
python report/make_data.py
```

Every assumption the map depends on — spot, tick size, conversion ratio, rate,
dividend yield, and the volatility the greeks are evaluated at — is a named
constant at the top of `make_data.py`.
