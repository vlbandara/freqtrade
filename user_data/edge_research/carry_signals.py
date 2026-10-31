"""Print the funding-carry target book as of the last candle in the local dataset.

Each line is a hedged pair: BUY spot + SELL the same notional of the USDT-M perp.
usage: python3 carry_signals.py [capital_usdt] [leverage]
"""

import sys

import carry_lab as C
import edge_lab as L


CK = dict(k_days=3, thr_in=1e-4, thr_out=5e-5, top_k=10, perp_margin=0.2)


def main(capital: float = 10_000.0, lev: float = 1.0) -> None:
    p = L.load_panel()
    spot = C.load_spot(p["close"])
    _, w, _ = C.carry(p, spot, L.universe(p, 50), **CK)
    last = w.index[-1]
    book = w.iloc[-1]
    book = book[book > 0].sort_values(ascending=False)
    favg = p["funding_known"].rolling(72, min_periods=8).mean().iloc[-1]
    notional = capital * lev / (1 + CK["perp_margin"])
    print(f"as of {last} UTC | capital {capital:,.0f} USDT | leverage {lev}x")
    if book.empty:
        print("no coin above the funding threshold -> stay in cash (stablecoins)")
    for pair, wt in book.items():
        print(
            f"{pair.replace('_USDT_USDT', ''):>10}: long spot / short perp {wt * notional:>10,.0f} USDT"
            f"  (3d avg funding {favg[pair] * 100:.4f}%/8h = {favg[pair] * 3 * 365:.1%} APR)"
        )


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:3]]
    main(*args)
