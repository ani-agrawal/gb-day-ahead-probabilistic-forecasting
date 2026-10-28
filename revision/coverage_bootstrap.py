"""Bootstrap intervals for conditional 90% coverage (Section 4.2; Tables 6, A.9 and A.13).

Regimes: negative prices, above 150 and above 200 GBP/MWh. Fixed before any result
was inspected:
  * primary: resample the 365 delivery days with replacement, 10,000 replications,
    95% percentile intervals;
  * episode: hours in the regime belong to the same episode if separated by fewer
    than 24 hours (sensitivity: 12 and 48 hours); robustness: resample episodes;
  * paired differences: both models' coverage computed on the same drawn days (or
    episodes) in each replication, interval taken on the difference series.
Point estimates are the observed coverage values; the bootstrap supplies intervals only.
"""
import itertools

import numpy as np
import pandas as pd

from rev_common import MODELS, OUT, REGIMES, covered90, load

B = 10_000
SEED = 20261025
GAP_PRIMARY = 24
GAP_SENSITIVITY = (12, 48)
CI = (2.5, 97.5)

f = load()
f["ts"] = f.settlement_date + pd.to_timedelta(f.hour, unit="h")
y = f.actual_price.to_numpy()
day_codes, days = pd.factorize(f.settlement_date, sort=True)
assert len(days) == 365
COV = np.column_stack([covered90(f, m) for m in MODELS]).astype(float)   # hours x models


def episodes(mask, gap_h):
    """Episode id per extreme hour: a new episode starts when the gap to the
    previous extreme hour is not fewer than gap_h hours."""
    ts = f.ts.to_numpy()[mask]
    gaps = np.diff(ts) / np.timedelta64(1, "h")
    return np.concatenate([[0], np.cumsum(gaps >= gap_h)])


def boot_coverage(unit_ids, n_units, cov_ext, rng):
    """Resample units with replacement; return replications x models coverage.
    unit_ids maps each extreme hour to its resampling unit (day or episode)."""
    hours = np.bincount(unit_ids, minlength=n_units).astype(float)          # extreme hours per unit
    covd = np.stack([np.bincount(unit_ids, weights=cov_ext[:, j], minlength=n_units)
                     for j in range(cov_ext.shape[1])], axis=1)              # covered hours per unit
    counts = rng.multinomial(n_units, np.full(n_units, 1 / n_units), size=B).astype(float)
    den = counts @ hours
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ covd) / den[:, None], int((den == 0).sum())


rng = np.random.default_rng(SEED)
marg, pair, epi_tables, sens = [], [], [], []
for thr, rule in REGIMES.items():
    mask = rule(y)
    cov_ext = COV[mask]
    point = cov_ext.mean(axis=0)
    ep = episodes(mask, GAP_PRIMARY)
    n_ep = ep.max() + 1

    # ---- episode table (dates, duration, hours, max price, covered hours per model)
    sub = f.loc[mask, ["ts", "actual_price"]].assign(episode=ep)
    for e, g in sub.groupby("episode"):
        row = {"threshold": thr, "episode": f"E{e + 1}", "start": g.ts.min(), "end": g.ts.max(),
               "duration_h": int((g.ts.max() - g.ts.min()) / pd.Timedelta(hours=1)) + 1,
               "hours_above": len(g), "max_price": round(g.actual_price.max(), 1)}
        for j, m in enumerate(MODELS):
            row[m] = int(COV[g.index, j].sum())
        epi_tables.append(row)

    # ---- marginal and paired intervals, two resampling units
    units = {"day": (day_codes[mask], len(days)), "episode": (ep, n_ep)}
    for method, (ids, n_units) in units.items():
        reps, empty = boot_coverage(ids, n_units, cov_ext, rng)
        for j, m in enumerate(MODELS):
            lo, hi = np.nanpercentile(reps[:, j], CI)
            marg.append({"threshold": thr, "method": method, "model": m, "n_hours": int(mask.sum()),
                         "n_days": int(len(np.unique(day_codes[mask]))), "n_episodes": int(n_ep),
                         "coverage": point[j], "ci_lo": lo, "ci_hi": hi, "empty_replications": empty})
        for a, b in itertools.permutations(range(len(MODELS)), 2):
            if a > b:
                continue
            d = reps[:, a] - reps[:, b]
            lo, hi = np.nanpercentile(d, CI)
            pair.append({"threshold": thr, "method": method, "model_a": MODELS[a], "model_b": MODELS[b],
                         "diff": point[a] - point[b], "ci_lo": lo, "ci_hi": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})

    # ---- footnote: sensitivity of the episode grouping (12 / 24 / 48 h)
    for gap in (GAP_SENSITIVITY[0], GAP_PRIMARY, GAP_SENSITIVITY[1]):
        epg = episodes(mask, gap)
        reps, _ = boot_coverage(epg, epg.max() + 1, cov_ext, np.random.default_rng(SEED + gap))
        q, l = MODELS.index("QRF"), MODELS.index("Encoder-Decoder LSTM")
        dl, dh = np.nanpercentile(reps[:, q] - reps[:, l], CI)
        sens.append({"threshold": thr, "gap_h": gap, "n_episodes": int(epg.max() + 1),
                     "QRF_minus_LSTM": point[q] - point[l], "ci_lo": dl, "ci_hi": dh,
                     "excludes_zero": bool(dl > 0 or dh < 0)})

marg, pair, epi, sens = map(pd.DataFrame, (marg, pair, epi_tables, sens))

# Reconciliation: episode totals must equal the Table 5 covered-hour counts
for thr, rule in REGIMES.items():
    e = epi[epi.threshold == thr]
    tot = e[MODELS].sum()
    n = int(rule(y).sum())
    for j, m in enumerate(MODELS):
        assert tot[m] == int(COV[rule(y), j].sum())
    assert e.hours_above.sum() == n
    print(f"{thr}: {n} hours in {len(e)} episodes; episode totals reconcile with Table 5 coverage")

marg.to_csv(OUT / "coverage_ci.csv", index=False)
pair.to_csv(OUT / "coverage_paired_differences.csv", index=False)
epi.to_csv(OUT / "coverage_episodes.csv", index=False)
sens.to_csv(OUT / "coverage_episode_gap_sensitivity.csv", index=False)
print("wrote coverage_ci.csv, coverage_paired_differences.csv, coverage_episodes.csv, coverage_episode_gap_sensitivity.csv")
