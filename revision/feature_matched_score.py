"""Scoring of the feature-matched QRF alongside the full QRF and the LSTM (Tables 8 and A.19).

Reports CRPS and conditional 90% coverage with the Table 5 companions (MAE, coverage,
MPIL, Winkler) and the threshold-weighted CRPS. Coverage differences use the method of
coverage_bootstrap.py (10,000 resamples of delivery days, episode resampling as the
robustness check); CRPS differences use the Section 4.2 Diebold-Mariano statistic on
daily totals, two-sided.
"""
import numpy as np
import pandas as pd
from scipy import stats

from rev_common import KEY, LEVELS, OUT, QCOLS, REGIMES, THRESHOLDS, covered90, crps11, load, quantiles, winkler90

B, SEED, GAP, CI = 10_000, 20261025, 24, (2.5, 97.5)
MATCHED = "Feature-matched QRF"
f = load()
m = pd.read_csv(OUT / "feature_matched_qrf" / "predictions.csv", parse_dates=["settlement_date"])
m = f[KEY].merge(m, on=KEY, how="left")
assert len(m) == len(f) and np.allclose(m.actual_price, f.actual_price) and m[QCOLS].notna().all().all()
for q in QCOLS:
    f[f"{MATCHED}|{q}"] = m[q].to_numpy()

y = f.actual_price.to_numpy()
f["ts"] = f.settlement_date + pd.to_timedelta(f.hour, unit="h")
day_codes, days = pd.factorize(f.settlement_date, sort=True)
COMP = ["QRF", MATCHED, "Encoder-Decoder LSTM"]

# ---------------- point scores
rows = []
for name in COMP:
    Q = quantiles(f, name)
    cov = covered90(f, name)
    r = {"model": name, "CRPS": crps11(y, Q).mean(), "MAE": np.abs(y - Q[:, 5]).mean(), "cov90": cov.mean(),
         "MPIL90": (Q[:, -1] - Q[:, 0]).mean(), "Winkler90": winkler90(y, Q[:, 0], Q[:, -1]).mean()}
    for thr, rule in REGIMES.items():
        r[f"cov {thr}"] = cov[rule(y)].mean()
    for t in THRESHOLDS:                            # threshold-weighted CRPS, as in twcrps.py
        r[f"twCRPS_{t}"] = crps11(np.maximum(y, t), np.maximum(Q, t)).mean()
    r["MPIL90 > 200"] = (Q[y > 200, -1] - Q[y > 200, 0]).mean()
    rows.append(r)
pts = pd.DataFrame(rows)
pts.to_csv(OUT / "feature_matched_scores.csv", index=False)


# ---------------- DM on daily CRPS (two-sided), Section 4.2 HAC form
def dm(a, b):
    la = pd.Series(crps11(y, quantiles(f, a))).groupby(day_codes).sum().to_numpy()
    lb = pd.Series(crps11(y, quantiles(f, b))).groupby(day_codes).sum().to_numpy()
    d = la - lb
    T = len(d)
    lag = int(np.floor(1.5 * T ** (1 / 3)))
    u = d - d.mean()
    s = (u @ u) / T
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / (T - k)
    stat = d.mean() / np.sqrt(s / T)
    return d.mean() / 24, stat, 2 * stats.norm.sf(abs(stat))


dms = pd.DataFrame([{"model_a": a, "model_b": b, "mean_hourly_crps_diff": dd, "dm_stat": st, "p_two_sided": p}
                    for a, b in [(MATCHED, "QRF"), (MATCHED, "Encoder-Decoder LSTM")]
                    for dd, st, p in [dm(a, b)]])
dms.to_csv(OUT / "feature_matched_dm.csv", index=False)

# ---------------- coverage differences, method of coverage_bootstrap.py
COV = np.column_stack([covered90(f, n) for n in COMP]).astype(float)
PAIRS = [(MATCHED, "QRF"), (MATCHED, "Encoder-Decoder LSTM"), ("QRF", "Encoder-Decoder LSTM")]


def episodes(mask):
    ts = f.ts.to_numpy()[mask]
    return np.concatenate([[0], np.cumsum(np.diff(ts) / np.timedelta64(1, "h") >= GAP)])


def boot(ids, n, vals, rng):
    hours = np.bincount(ids, minlength=n).astype(float)
    tot = np.stack([np.bincount(ids, weights=vals[:, j], minlength=n) for j in range(vals.shape[1])], 1)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ tot) / (counts @ hours)[:, None]


rng = np.random.default_rng(SEED)
diff_rows, ci_rows = [], []
for thr, rule in REGIMES.items():
    mask = rule(y)
    ce = COV[mask]
    point = ce.mean(axis=0)
    ep = episodes(mask)
    for unit, (ids, n) in {"day": (day_codes[mask], len(days)), "episode": (ep, ep.max() + 1)}.items():
        reps = boot(ids, n, ce, rng)
        for j, name in enumerate(COMP):
            lo, hi = np.nanpercentile(reps[:, j], CI)
            ci_rows.append({"regime": thr, "unit": unit, "model": name, "coverage": point[j], "lo": lo, "hi": hi})
        for a, b in PAIRS:
            ia, ib = COMP.index(a), COMP.index(b)
            lo, hi = np.nanpercentile(reps[:, ia] - reps[:, ib], CI)
            diff_rows.append({"regime": thr, "unit": unit, "model_a": a, "model_b": b,
                              "diff": point[ia] - point[ib], "lo": lo, "hi": hi,
                              "excludes_zero": bool(lo > 0 or hi < 0)})
pd.DataFrame(ci_rows).to_csv(OUT / "feature_matched_coverage_ci.csv", index=False)
dif = pd.DataFrame(diff_rows)
dif.to_csv(OUT / "feature_matched_coverage_differences.csv", index=False)

pd.set_option("display.width", 220)
print(pts.round(3).to_string(index=False))
print()
print(dms.round(4).to_string(index=False))
print()
print(dif.round(3).to_string(index=False))
