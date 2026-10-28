"""Hours of each forecast-time regime inside the 60-day rolling conformal calibration window
(Section 6.5; reason a regime-dependent conformal variant needs its own design).

For every 2025 forecast day D, counts the hours of each regime among the 60 delivery days
ending D-2 (the window of the rolling conformal correction, W = 60), and reports how many
forecast days have no such hours or fewer than 30. Also finds the largest calibration size
n at which the correction of each 90% bound, computed exactly as in
modelling/conformal/run_conformal.py, equals the largest observed error.
"""
import numpy as np
import pandas as pd

from rev_common import model_table

W = 60
t = model_table()
t = t[t.settlement_date.between("2024-07-01", "2025-12-31")]
rl = t.net_demand_forecast / 1000
regimes = {"RL >= 30 GW": rl >= 30, "OPMR surplus <= 0": t.opmr_surplus_norm <= 0, "RL < 5 GW": rl < 5}
days = pd.date_range("2025-01-01", "2025-12-31")


def corr(e, alpha_tail):
    """The correction of modelling/conformal/run_conformal.py."""
    n = len(e)
    level = min(1.0, (1 - alpha_tail) * (1 + 1 / n))
    return float(np.quantile(e, level, method="higher"))


# largest calibration size at which the 90%-bound correction equals the largest observed error
n_max = max(n for n in range(1, 500) if corr(np.arange(n, dtype=float), 0.05) == n - 1)
assert all(corr(np.arange(n, dtype=float), 0.05) == n - 1 for n in range(1, n_max + 1))
print(f"each 90% bound's correction equals the largest observed error for every n <= {n_max}")

for name, mask in regimes.items():
    daily = mask.groupby(t.settlement_date).sum()
    counts = np.array([int(daily[(daily.index >= d - pd.Timedelta(days=2 + W - 1)) &
                                 (daily.index <= d - pd.Timedelta(days=2))].sum()) for d in days])
    print(f"{name:18} hours in window (of {24 * W}): min {counts.min()}, median {int(np.median(counts))}, "
          f"max {counts.max()}; days with none {int((counts == 0).sum())}; days with fewer than 30 "
          f"{int((counts < 30).sum())}; days with fewer than {n_max + 1} {int((counts <= n_max).sum())} "
          f"(of {len(days)})")
