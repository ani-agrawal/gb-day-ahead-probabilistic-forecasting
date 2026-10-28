"""Development-period overlap of the QRA estimation pool (Section 4.6, final paragraph).

1. Share of the expanding QRA estimation pool (matched forecasts from 1 July 2024 through
   D-2) that falls in the July-December 2024 development period, at the first, a mid-year
   and the last weekly block of 2025.
2. Monthly mean CRPS of three-model QRA against its two principal constituents (the
   Encoder-Decoder LSTM and QRF), and the lowest of the three in each month.
"""
import pandas as pd

from rev_common import crps11, load, quantiles

POOL_START, DEV_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")
for start in [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-02"), pd.Timestamp("2025-12-31")]:
    end = start - pd.Timedelta(days=2)
    total = (end - POOL_START).days + 1
    dev = (min(end, DEV_END) - POOL_START).days + 1
    print(f"block {start.date()}: pool {total} days, development share {dev / total:.3f}")

f = load()
y = f.actual_price.to_numpy()
models = ["Three-model QRA", "Encoder-Decoder LSTM", "QRF"]
c = pd.DataFrame({m: crps11(y, quantiles(f, m)) for m in models})
c["month"] = f.settlement_date.dt.month.to_numpy()
g = c.groupby("month").mean()
g["lowest"] = g[models].idxmin(axis=1)
print(g.round(2).to_string())
print("months led by three-model QRA:", list(g.index[g.lowest == "Three-model QRA"]))
