"""Threshold-weighted CRPS (Section 4.2; Table 5).

twCRPS(F, y) = integral of (F(z) - 1{y <= z})^2 w(z) dz with w(z) = 1{z > t},
t = 150 and 200 GBP/MWh, the two thresholds fixed before the test year. With the
chaining function v(z) = max(z, t) (Gneiting and Ranjan, 2011) this equals the CRPS of
the forecast and outcome after both are transformed by v. Quantiles are equivariant
under the monotone v, so the score is obtained by applying v to the saved forecast
quantiles and to the outcome and evaluating Equation (2) unchanged.
"""
import numpy as np
import pandas as pd

from rev_common import FULL_GRID, OUT, crps11, load, quantiles

THRESHOLDS = (150.0, 200.0)

f = load()
y = f.actual_price.to_numpy()
rows = []
for m in FULL_GRID:
    Q = quantiles(f, m)
    row = {"model": m, "CRPS": crps11(y, Q).mean()}
    for t in THRESHOLDS:
        tw = crps11(np.maximum(y, t), np.maximum(Q, t))
        row[f"twCRPS_{t:.0f}"] = tw.mean()
        row[f"twCRPS_{t:.0f}_share_from_hours_above"] = tw[y > t].sum() / tw.sum()
    rows.append(row)
out = pd.DataFrame(rows)
# sanity: with t below every price and quantile the score must equal CRPS
lo = min(y.min(), min(quantiles(f, m).min() for m in FULL_GRID)) - 1
for m in FULL_GRID:
    Q = quantiles(f, m)
    assert np.allclose(crps11(np.maximum(y, lo), np.maximum(Q, lo)), crps11(y, Q))
out.to_csv(OUT / "twcrps.csv", index=False)
print(out.assign(rank150=out.twCRPS_150.rank().astype(int), rank200=out.twCRPS_200.rank().astype(int))
      .round(4).to_string(index=False))
