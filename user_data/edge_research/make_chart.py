"""Equity-curve figure for REPORT.md (light surface, reference palette, direct labels)."""

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

carry = pd.read_parquet("carry_pnl.parquet")["carry"]
fc = pd.read_parquet("candidate_pnl.parquet")["pnl"]
trend = pd.read_parquet("ensemble_pnl.parquet")["trend"]
series = [
    ("Funding carry harvest (1x) - EDGE", carry, S1),
    ("Funding contrarian L/S - rejected", fc, S2),
    ("Trend sleeve - rejected", trend, S3),
]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.axvspan(pd.Timestamp("2024-01-01", tz="UTC"), carry.index[-1], color="#efeee9", lw=0)
for x in ("2023-01-01", "2024-01-01"):
    ax.axvline(pd.Timestamp(x, tz="UTC"), color=GRID, lw=1)
for lbl, x in (
    ("in-sample", "2021-12-01"),
    ("validation", "2023-07-01"),
    ("blind out-of-sample", "2024-09-15"),
):
    ax.text(
        pd.Timestamp(x, tz="UTC"),
        0.985,
        lbl,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color=INK2,
        fontsize=9,
    )
nudge = {S1: -10, S2: 0, S3: 12}
for name, s, col in series:
    s = s.loc["2021-01-01":]
    eq = (1 + s).cumprod()
    eq = eq.resample("1D").last()
    ax.plot(eq.index, eq.values, color=col, lw=2, solid_capstyle="round")
    ax.annotate(
        f"{name}\n{eq.iloc[-1]:.2f}x",
        (eq.index[-1], eq.iloc[-1]),
        xytext=(6, nudge[col]),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=INK,
    )
ax.set_yscale("log")
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([0.75, 1, 1.5, 2, 3, 4])
ax.set_yticklabels(["0.75x", "1x", "1.5x", "2x", "3x", "4x"])
ax.set_ylim(0.85, 4.6)
ax.grid(axis="y", color=GRID, lw=0.8)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.set_title(
    "Growth of 1 USDT, after fees, slippage and funding (log scale)",
    loc="left",
    color=INK,
    fontsize=11,
)
ax.set_xlim(carry.loc["2021":].index[0], carry.index[-1] + pd.Timedelta(days=330))
fig.tight_layout()
fig.savefig("equity_curves.png", facecolor=SURF)
print("saved")
