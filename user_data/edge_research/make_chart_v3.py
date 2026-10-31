"""Round-3 chart: modern-regime portfolio and its two strongest streams, 2022-01 .. 2025-05."""

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
d = pd.read_parquet("modern_pnl.parquet").loc["2022-01-01":]
series = [
    ("Short new listings (A)", d["A"], S2),
    ("Portfolio A+B+C", d["portfolio"], S1),
    ("Majors trend (C)", d["C"], S3),
]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
end = d.index[-1]
ax.axvspan(pd.Timestamp("2024-01-01", tz="UTC"), end, color="#efeee9", lw=0)
for lbl, x in (("design window (2022-23)", "2023-01-01"), ("holdout, opened once", "2024-09-15")):
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
nudge = {S1: 0, S2: 0, S3: -6}
for name, s, col in series:
    eq = (1 + s).cumprod()
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
ax.set_yticks([0.8, 1, 1.5, 2, 3, 4])
ax.set_yticklabels(["0.8x", "1x", "1.5x", "2x", "3x", "4x"])
ax.grid(axis="y", color=GRID, lw=0.8)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.set_title(
    "Modern-regime strategies (no 2021): growth of 1 USDT after fees and funding (log scale)",
    loc="left",
    color=INK,
    fontsize=11,
)
ax.set_xlim(pd.Timestamp("2022-01-01", tz="UTC"), end + pd.Timedelta(days=260))
fig.tight_layout()
fig.savefig("equity_curves_v3.png", facecolor=SURF)
print("saved", {n: round(float((1 + s).prod()), 3) for n, s, _ in series})
