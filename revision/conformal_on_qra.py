"""Conformal recalibration applied to the three-model QRA forecasts (Sections 4.5 and 5.4;
Table A.24).

Design fixed before any QRA result was computed:
  * Conformal procedure = modelling/conformal/run_conformal.py, reimplemented below and
    checked: applied to the LSTM's own inputs it must reproduce all eight reported LSTM
    conformal columns (static and rolling, 80% and 90% bounds).
  * QRA calibration forecasts: qra_backtest of modelling/qra/run_three_way_qra_qw.py,
    unchanged except that its block grid starts on 2 October 2024 (the same 7-day grid
    extended back 13 weeks), so every 2024 forecast has at least three months of pooled
    constituent forecasts. Each block is fitted only on matched forecasts and prices
    ending D-2, so the 2025 forecasts cannot change; they are checked against the reported
    three-model QRA forecasts.
  * Static CQR on QRA: corrections from its 2 October - 31 December 2024 errors (91 days),
    held fixed through 2025 (the LSTM used 1 July - 31 December 2024; QRA has no earlier
    forecasts).
  * Rolling CQR on QRA: W = 60 delivery days ending D-2, carried over from the LSTM's
    development selection, not re-selected; warm-up from the 2024 calibration forecasts.
  * Same asymmetric scores, (1+1/n) finite-sample level, higher order statistic, nesting
    enforcement and Table 10 metrics. Differences against raw QRA carry day- and
    episode-level bootstrap intervals (10,000 replications).
"""
import json
import sys

import numpy as np
import pandas as pd

from rev_common import KEY, LSTM_DEV, LSTM_RUN, MODELLING, OUT, QCOLS, QRA_OUT, TEST_END, TEST_START

RES = OUT / "conformal_on_qra"
RES.mkdir(exist_ok=True)
INTERVALS = [("q05", "q95", 0.10), ("q10", "q90", 0.20)]
W = 60
CAL_START = pd.Timestamp("2024-10-02")


# ------------------------------------------------------------- conformal (as run_conformal.py)
def corr(e, alpha_tail):
    n = len(e)
    level = min(1.0, (1 - alpha_tail) * (1 + 1 / n))
    return float(np.quantile(e, level, method="higher"))


def cqr(cal, sub, w):
    """cal: calibration frame (KEY, actual_price, bounds); sub: 2025 frame. Returns sub with
    *_static and *_roll bounds after nesting enforcement, exactly as run_conformal.py."""
    sub = sub.copy()
    for lo, hi, alpha in INTERVALS:
        e_lo, e_hi = cal[lo] - cal["actual_price"], cal["actual_price"] - cal[hi]
        sub[f"{lo}_static"] = sub[lo] - corr(e_lo, alpha / 2)
        sub[f"{hi}_static"] = sub[hi] + corr(e_hi, alpha / 2)
        hist = pd.concat([cal[KEY + ["actual_price", lo, hi]], sub[KEY + ["actual_price", lo, hi]]],
                         ignore_index=True)
        hist["settlement_date"] = pd.to_datetime(hist["settlement_date"])
        hist["e_lo"] = hist[lo] - hist["actual_price"]
        hist["e_hi"] = hist["actual_price"] - hist[hi]
        by_day = {d: g for d, g in hist.groupby("settlement_date")}
        days_all = sorted(by_day)
        lo_adj, hi_adj = [], []
        for day, g in sub.groupby("settlement_date"):
            cut = day - pd.Timedelta(days=2)
            wd = [d for d in days_all if d <= cut][-w:]
            wdf = pd.concat([by_day[d] for d in wd])
            lo_adj.append(g[lo] - corr(wdf["e_lo"], alpha / 2))
            hi_adj.append(g[hi] + corr(wdf["e_hi"], alpha / 2))
        sub[f"{lo}_roll"] = pd.concat(lo_adj).sort_index()
        sub[f"{hi}_roll"] = pd.concat(hi_adj).sort_index()
    for s in ("static", "roll"):
        sub[f"q05_{s}"] = np.minimum(sub[f"q05_{s}"], sub[f"q10_{s}"])
        sub[f"q95_{s}"] = np.maximum(sub[f"q95_{s}"], sub[f"q90_{s}"])
    return sub


def winkler(y, lo, hi, alpha):
    pen = np.where(y < lo, (2 / alpha) * (lo - y), np.where(y > hi, (2 / alpha) * (y - hi), 0.0))
    return float(np.mean(hi - lo + pen))


# ------------------------------------------------------------- gate 1: LSTM reproduction
C5 = ["q05", "q10", "q50", "q90", "q95"]
seeds = []
for s in (42, 7, 101):
    p = pd.read_csv(LSTM_DEV / f"encdec_u64_b16es14_seed{s}" / "predictions.csv")
    p[C5] = np.sort(p[C5].to_numpy(), axis=1)
    seeds.append(p.sort_values(KEY).reset_index(drop=True))
lcal = seeds[0][KEY + ["actual_price"]].copy()
lcal[C5] = np.sort(np.mean([c[C5].to_numpy() for c in seeds], axis=0), axis=1)
lcal["settlement_date"] = pd.to_datetime(lcal["settlement_date"])
ltest = pd.read_csv(LSTM_RUN / "ENSEMBLE_11q" / "predictions.csv")
ltest["settlement_date"] = pd.to_datetime(ltest["settlement_date"])
ltest = ltest.sort_values(KEY).reset_index(drop=True)
lsub = ltest[(ltest.settlement_date >= TEST_START) & (ltest.settlement_date <= TEST_END)].copy()
mine = cqr(lcal, lsub, W).reset_index(drop=True)
pub = pd.read_csv(LSTM_RUN / "conformal_ENSEMBLE" / "predictions_2025_primary.csv",
                  parse_dates=["settlement_date"]).sort_values(KEY).reset_index(drop=True)
assert (mine[KEY].values == pub[KEY].values).all()
cols8 = [f"{b}_{s}" for b in ("q05", "q95", "q10", "q90") for s in ("static", "roll")]
dev = max(float(np.abs(mine[c] - pub[c]).max()) for c in cols8)
assert dev < 1e-9, f"conformal reimplementation does not reproduce the LSTM columns ({dev})"
print(f"GATE 1 PASS: LSTM conformal columns reproduced (max |diff| {dev:.1e})")

# ------------------------------------------------------------- QRA forecasts from 2 Oct 2024
sys.path.insert(0, str(MODELLING / "qra"))
import run_three_way_qra_qw as mod  # noqa: E402
b = mod.b
lstm = b.lstm_frame().rename(columns={c: f"lstm_{c}" for c in QCOLS})
qrf = b.qrf_frame().rename(columns={c: f"qrf_{c}" for c in QCOLS})
m = lstm.merge(qrf.drop(columns="actual_price"), on=KEY).merge(mod.lear_qra_frame(), on=KEY)
all_oos = m.dropna().sort_values(KEY).reset_index(drop=True)
assert all_oos.settlement_date.min() == pd.Timestamp("2024-07-01")
span = all_oos[(all_oos.settlement_date >= CAL_START) & (all_oos.settlement_date <= TEST_END)].copy()
assert (pd.Timestamp("2025-01-01") - CAL_START).days % mod.BLOCK_DAYS == 0      # same block grid
mod.TEST_START = CAL_START                                                     # only change
qsorted, _, _ = mod.qra_backtest(all_oos, span, "three_way_qw", 0.0)
qra = span[KEY + ["actual_price"]].reset_index(drop=True)
qra[QCOLS] = qsorted
qpub = pd.read_csv(QRA_OUT / "three_way_qw_predictions.csv",
                   parse_dates=["settlement_date"]).sort_values(KEY).reset_index(drop=True)
q2025 = qra[qra.settlement_date >= TEST_START].reset_index(drop=True)
assert (q2025[KEY].values == qpub[KEY].values).all()
dev2 = float(np.abs(q2025[QCOLS].to_numpy() - qpub[[f"three_way_qw_{c}" for c in QCOLS]].to_numpy()).max())
assert dev2 < 1e-9, f"QRA rebuild does not reproduce the published 2025 forecasts ({dev2})"
print(f"GATE 2 PASS: rebuilt 2025 QRA forecasts equal the published ones (max |diff| {dev2:.1e})")
qcal = qra[qra.settlement_date < TEST_START].reset_index(drop=True)
assert qcal.settlement_date.min() == CAL_START and qcal.settlement_date.max() == pd.Timestamp("2024-12-31")
qra.to_csv(RES / "qra_forecasts_2024q4_2025.csv", index=False)

# ------------------------------------------------------------- conformal on QRA
qsub = q2025.copy()
out = cqr(qcal, qsub, W).reset_index(drop=True)
out.to_csv(RES / "predictions_2025.csv", index=False)
y = out.actual_price.to_numpy()
regimes = {"< 0": y < 0, "> 150": y > 150, "> 200": y > 200}
LAYERS = {"Raw three-model QRA": "", "Static split CQR": "_static", "Rolling asymmetric CQR": "_roll"}
rows = []
for name, suf in LAYERS.items():
    r = {"layer": name, "cal_days": int(qcal.settlement_date.nunique()), "cal_hours": len(qcal), "W": W}
    for lo, hi, alpha in INTERVALS:
        lab = f"{int(round((1 - alpha) * 100))}"
        L, H = out[lo + suf].to_numpy(), out[hi + suf].to_numpy()
        ins = (y >= L) & (y <= H)
        r[f"cov{lab}"], r[f"mpil{lab}"], r[f"winkler{lab}"] = ins.mean(), (H - L).mean(), winkler(y, L, H, alpha)
        if lab == "90":
            r["miss_below90"], r["miss_above90"] = (y < L).mean(), (y > H).mean()
            for k, msk in regimes.items():
                r[f"cov90 {k}"] = ins[msk].mean()
    rows.append(r)
met = pd.DataFrame(rows)
met.to_csv(RES / "metrics.csv", index=False)

# ------------------------------------------------------------- intervals for the changes (coverage_bootstrap.py method)
B, SEED, GAP, CI = 10_000, 20261025, 24, (2.5, 97.5)
ts = out.settlement_date + pd.to_timedelta(out.hour, unit="h")
day_codes, days = pd.factorize(out.settlement_date, sort=True)
COV = np.column_stack([((y >= out["q05" + s]) & (y <= out["q95" + s])).to_numpy() for s in LAYERS.values()]).astype(float)
rng = np.random.default_rng(SEED)
diff_rows = []
for k, msk in {"all hours": np.ones_like(y, bool), **regimes}.items():
    ce = COV[msk]
    point = ce.mean(axis=0)
    t = ts.to_numpy()[msk]
    ep = np.concatenate([[0], np.cumsum(np.diff(t) / np.timedelta64(1, "h") >= GAP)])
    units = {"day": (day_codes[msk], len(days))}
    if k != "all hours":
        units["episode"] = (ep, ep.max() + 1)
    for unit, (ids, n) in units.items():
        hours = np.bincount(ids, minlength=n).astype(float)
        tot = np.stack([np.bincount(ids, weights=ce[:, j], minlength=n) for j in range(3)], 1)
        cnt = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = (cnt @ tot) / (cnt @ hours)[:, None]
        for j, name in [(1, "Static split CQR"), (2, "Rolling asymmetric CQR")]:
            lo_, hi_ = np.nanpercentile(reps[:, j] - reps[:, 0], CI)
            diff_rows.append({"regime": k, "unit": unit, "layer": name, "diff_vs_raw": point[j] - point[0],
                              "lo": lo_, "hi": hi_, "excludes_zero": bool(lo_ > 0 or hi_ < 0)})
dif = pd.DataFrame(diff_rows)
dif.to_csv(RES / "differences_vs_raw.csv", index=False)
(RES / "manifest.json").write_text(json.dumps({
    "gate1_lstm_max_abs_diff": dev, "gate2_qra_max_abs_diff": dev2, "W": W,
    "cal_start": str(CAL_START.date()), "cal_end": "2024-12-31", "cal_hours": len(qcal), "B": B, "seed": SEED}, indent=1))

pd.set_option("display.width", 250)
print(met.round(3).to_string(index=False))
print(dif.round(3).to_string(index=False))
