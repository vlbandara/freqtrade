"""Round 6 audit: how much of round-3 stream A (dilution short) and the locked A+B+C portfolio
came from truncated histories mistaken for new listings?

Same locked parameters as final_eval_modern.py; only the listing-age input changes.
"""

import edge_lab as L
import modern_lab as M
import pandas as pd
import r6_common as C6


pd.set_option("display.width", 250)
p = L.load_panel()
close = p["close"]
ok = C6.genuine_listing(close)
started = close.apply(lambda s: s.first_valid_index()) > close.index[0] + pd.Timedelta("7D")
fake = ok.index[started & ~ok]
print(f"pairs starting inside the data: {int(started.sum())}; flagged as truncated: {len(fake)}")
print(sorted(fake))

wA_orig = M.dilution_short(p, 30, 1)
orig_age = M.listing_age_hours
M.listing_age_hours = C6.listing_age_hours_clean
wA_clean = M.dilution_short(p, 30, 1)
M.listing_age_hours = orig_age

# attribution of the original stream to the truncated pairs (short legs only)
pp = C6.per_pair_pnl(wA_orig, p)
contrib = pp[[c for c in fake if c in pp.columns]]
print("\n#### Stream A (original) P&L from truncated 'new listings', by year (sum of hourly returns)")
print(contrib.sum(axis=1).groupby(contrib.index.year).sum().round(4).to_string())
top = contrib.loc["2024"].sum().sort_values()
print("2024 contribution per truncated pair:\n", top[top.abs() > 1e-4].round(4).to_string())

wB = M.btc_vs_alts(p, 20)
wC = M.majors_trend(p, "donchian", 20, 55, 0.15)
rows = []
for tag, wA in (("original", wA_orig), ("clean listings", wA_clean)):
    S = {k: C6.daily(L.run(k, w, p).pnl).loc["2022":] for k, w in (("A", wA), ("B", wB), ("C", wC))}
    port, _ = M.inv_vol_mix(S)
    for name, d in (("A", S["A"]), ("PORTFOLIO", port)):
        for split, (a, b) in M.SPLITS_MODERN.items():
            rows.append({"data": tag, "stream": name, "split": split, **C6.stats(d.loc[a:b])})
        yr = (1 + d).groupby(d.index.year).prod() - 1
        rows[-1].update({f"y{y}": yr.get(y) for y in range(2022, 2026)})
print("\n#### Locked round-3 config, original vs clean listing dates")
print(pd.DataFrame(rows).set_index(["data", "stream", "split"]).round(3).to_string())
