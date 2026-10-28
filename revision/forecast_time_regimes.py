"""Conditional coverage over forecast-time regimes (Section 5.2.1; Tables A.16 and A.17).

Regimes are known before the 09:20 auction gate and their cut points were fixed
before the test year:
  * forecast residual load (net_demand_forecast = demand - transmission wind -
    embedded wind - embedded solar): below 5 GW, 5-30 GW, at least 30 GW, the
    thresholds that Section 3.4 / Figure 3 derive from the 2023-2024 data;
  * NESO OPMR normalised surplus at or below zero (planned surplus not positive).
Method as coverage_bootstrap.py: 10,000 resamples of delivery days, percentile
intervals, paired differences on the same draws; episode resampling (runs separated
by fewer than 24 hours) as the robustness check for the scarce regimes. Because the
regimes are defined ex ante, ordinary CRPS remains proper within them.
"""
import numpy as np
import pandas as pd

from rev_common import FULL_GRID, KEY, MODELS, OUT, covered90, crps11, load, model_table, quantiles

B, SEED, CI, GAP = 10_000, 20261026, (2.5, 97.5), 24

f = load()
tab = model_table()
f = f.merge(tab[KEY + ["net_demand_forecast", "opmr_surplus_norm"]], on=KEY, how="left")
assert len(f) == 8759 and f[["net_demand_forecast", "opmr_surplus_norm"]].notna().all().all()
f["ts"] = f.settlement_date + pd.to_timedelta(f.hour, unit="h")
rl = f.net_demand_forecast.to_numpy() / 1000.0
s = f.opmr_surplus_norm.to_numpy()
y = f.actual_price.to_numpy()

REGIMES = {
    "RL < 5 GW": (rl < 5, True),
    "RL 5-30 GW": ((rl >= 5) & (rl < 30), False),     # spans nearly every day: no episode check
    "RL >= 30 GW": (rl >= 30, True),
    "OPMR surplus <= 0": (s <= 0, True),
}
day_codes, days = pd.factorize(f.settlement_date, sort=True)
COV = np.column_stack([covered90(f, m) for m in MODELS]).astype(float)
CRPS = {m: crps11(y, quantiles(f, m)) for m in FULL_GRID}


def episodes(mask):
    ts = f.ts.to_numpy()[mask]
    return np.concatenate([[0], np.cumsum(np.diff(ts) / np.timedelta64(1, "h") >= GAP)])


def boot(ids, n_units, vals, rng):
    hours = np.bincount(ids, minlength=n_units).astype(float)
    tot = np.stack([np.bincount(ids, weights=vals[:, j], minlength=n_units) for j in range(vals.shape[1])], 1)
    counts = rng.multinomial(n_units, np.full(n_units, 1 / n_units), size=B).astype(float)
    den = counts @ hours
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ tot) / den[:, None]


CONTRASTS = [("QRF", "Encoder-Decoder LSTM"), ("Three-model QRA", "Encoder-Decoder LSTM"),
             ("QRF", "Three-model QRA"), ("Rolling CQR", "Encoder-Decoder LSTM")]
rng = np.random.default_rng(SEED)
cov_rows, diff_rows, crps_rows = [], [], []
for name, (mask, do_ep) in REGIMES.items():
    ce = COV[mask]
    point = ce.mean(axis=0)
    units = {"day": (day_codes[mask], len(days))}
    if do_ep:
        ep = episodes(mask)
        units["episode"] = (ep, ep.max() + 1)
    meta = {"regime": name, "n_hours": int(mask.sum()), "n_days": int(len(np.unique(day_codes[mask]))),
            "n_episodes": int(units["episode"][1]) if do_ep else None,
            "hours_below_0": int((y[mask] < 0).sum()), "hours_above_150": int((y[mask] > 150).sum()),
            "hours_above_200": int((y[mask] > 200).sum()), "median_price": float(np.median(y[mask]))}
    reps = {k: boot(ids, n, ce, rng) for k, (ids, n) in units.items()}
    for j, m in enumerate(MODELS):
        row = {**meta, "model": m, "coverage": point[j]}
        for k, r in reps.items():
            row[f"{k}_lo"], row[f"{k}_hi"] = np.nanpercentile(r[:, j], CI)
        cov_rows.append(row)
    for a, b in CONTRASTS:
        ia, ib = MODELS.index(a), MODELS.index(b)
        row = {"regime": name, "model_a": a, "model_b": b, "diff": point[ia] - point[ib]}
        for k, r in reps.items():
            lo, hi = np.nanpercentile(r[:, ia] - r[:, ib], CI)
            row[f"{k}_lo"], row[f"{k}_hi"], row[f"{k}_excludes_zero"] = lo, hi, bool(lo > 0 or hi < 0)
        diff_rows.append(row)
    for m in FULL_GRID:
        crps_rows.append({"regime": name, "model": m, "mean_crps": CRPS[m][mask].mean()})

cov, dif, crp = map(pd.DataFrame, (cov_rows, diff_rows, crps_rows))
cov.to_csv(OUT / "forecast_time_regime_coverage.csv", index=False)
dif.to_csv(OUT / "forecast_time_regime_differences.csv", index=False)
crp.to_csv(OUT / "forecast_time_regime_crps.csv", index=False)

pd.set_option("display.width", 220)
for name in REGIMES:
    c = cov[cov.regime == name]
    r0 = c.iloc[0]
    print(f"\n== {name}: {r0.n_hours} h, {r0.n_days} days, episodes {r0.n_episodes}; "
          f"<0 {r0.hours_below_0}, >150 {r0.hours_above_150}, >200 {r0.hours_above_200}, median {r0.median_price:.1f}")
    for _, r in c.iterrows():
        ep = f"  ep [{r.episode_lo:.3f}, {r.episode_hi:.3f}]" if "episode_lo" in r and pd.notna(r.get("episode_lo")) else ""
        cr = crp[(crp.regime == name) & (crp.model == r.model)]
        cs = f"  CRPS {cr.mean_crps.iloc[0]:7.2f}" if len(cr) else ""
        print(f"   {r.model:22} {r.coverage:.3f}  day [{r.day_lo:.3f}, {r.day_hi:.3f}]{ep}{cs}")
    for _, r in dif[dif.regime == name].iterrows():
        ep = f"  ep [{r.episode_lo:+.3f}, {r.episode_hi:+.3f}]" if pd.notna(r.get("episode_lo")) else ""
        print(f"   diff {r.model_a[:15]:15} - {r.model_b[:15]:15} {r['diff']:+.3f}  day [{r.day_lo:+.3f}, {r.day_hi:+.3f}]{ep}")
