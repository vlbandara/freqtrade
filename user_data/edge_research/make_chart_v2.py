"""Round-2 equity curves (light surface, validated reference palette, direct labels)."""

import carry2_lab as C2
import carry_lab as C
import edge_lab as L
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

p = L.load_panel()
spot = C.load_spot(p["close"])
w = C2.target_weights(p, spot, L.universe(p, 50), top_k=10)
v2_1x = C2.simulate(p, spot, w, lev=1, borrow_apr=0.08)[0]
v2_2x = C2.simulate(p, spot, w, lev=2, borrow_apr=0.08)[0]
v1 = pd.read_parquet("carry_pnl.parquet")["carry"]
series = [
    ("Carry v2, 2x (8% borrow)", v2_2x, S2),
    ("Carry v2, 1x", v2_1x, S1),
    ("Round-1 carry (old engine)", v1, S3),
]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
end = v1.index[-1]
ax.axvspan(pd.Timestamp("2024-01-01", tz="UTC"), end, color="#efeee9", lw=0)
for x in ("2023-01-01", "2024-01-01"):
    ax.axvline(pd.Timestamp(x, tz="UTC"), color=GRID, lw=1)
for lbl, x in (
    ("in-sample", "2021-12-01"),
    ("validation", "2023-07-01"),
    ("out-of-sample", "2024-09-15"),
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
nudge = {S1: 6, S2: 0, S3: -8}
for name, s, col in series:
    eq = (1 + s.loc["2021-01-01":]).cumprod().resample("1D").last()
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
ax.set_yticks([1, 1.5, 2, 3])
ax.set_yticklabels(["1x", "1.5x", "2x", "3x"])
ax.grid(axis="y", color=GRID, lw=0.8)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.set_title(
    "Funding carry harvest: growth of 1 USDT after fees, funding and borrow cost (log scale)",
    loc="left",
    color=INK,
    fontsize=11,
)
ax.set_xlim(pd.Timestamp("2021-01-01", tz="UTC"), end + pd.Timedelta(days=330))
fig.tight_layout()
fig.savefig("equity_curves_v2.png", facecolor=SURF)
pd.DataFrame({"v2_1x": v2_1x, "v2_2x": v2_2x}).to_parquet("carry_v2_pnl.parquet")
print("saved")
