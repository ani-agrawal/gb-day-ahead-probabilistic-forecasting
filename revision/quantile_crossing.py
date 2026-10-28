"""Quantile crossing before rearrangement, by model (Section 4.3; Table A.23): share of 2025
forecast rows with crossing, mean and median size of the inversions, and the effect of
rearrangement on CRPS.

Definitions (eleven reported levels 0.05..0.95, 8,759 hours of 2025):
  crossing row   any adjacent pair q_tau > q_tau' with tau < tau'
  magnitude      mean of (q_tau - q_tau') over the inverted adjacent pairs, GBP/MWh
  CRPS effect    Equation (2) on the unsorted outputs minus Equation (2) on the
                 reported (rearranged) quantiles

Raw outputs are captured by running each model's own code under modelling/ unchanged and
recording the array passed to its np.sort call; every rerun must reproduce the reported
quantiles. Deterministic models only. The LSTM is not rerun: each seed's eleven quantiles
come from two separately sorted groups of networks (five and six levels), and the crossing
reported for it is that found when the two groups are combined. QRF quantiles are read
from one weighted empirical distribution; the 'qrf' mode checks this on two refit weeks.

Usage: python quantile_crossing.py lstm | qrf | qra | lear_qra | linear | xgboost
(the xgboost mode requires xgboost_comparator.py test 2 to have been run first)
"""
import json
import sys
import types

import numpy
import numpy as np
import pandas as pd

from rev_common import (KEY, LEAR_QRA_OUT, LEVELS, LSTM_RUN, MODELLING, OUT, QCOLS, QRA_OUT, SNAP, TEST_END,
                        TEST_START, crps11, lean_features, model_table)

RES = OUT / "quantile_crossing"
RES.mkdir(exist_ok=True)


def stats(name, y, raw, final):
    d = np.diff(raw, axis=1)
    inv = d < 0
    out = {"model": name, "rows": int(len(raw)), "rows_with_crossing": float(inv.any(axis=1).mean()),
           "inverted_pair_share": float(inv.mean()),
           "mean_magnitude": float(-d[inv].mean()) if inv.any() else 0.0,
           "median_magnitude": float(np.median(-d[inv])) if inv.any() else 0.0,
           "crps_unsorted": float(crps11(y, raw).mean()), "crps_published": float(crps11(y, final).mean())}
    out["crps_effect"] = out["crps_unsorted"] - out["crps_published"]
    out["crps_effect_pct"] = 100 * out["crps_effect"] / out["crps_published"]
    (RES / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


class SortSpy(types.ModuleType):
    """Stands in for numpy inside an original module; records what np.sort is given."""
    def __init__(self):
        super().__init__("numpy_spy")
        self.calls = []

    def __getattr__(self, n):
        return getattr(numpy, n)

    def sort(self, a, axis=-1, **kw):
        self.calls.append(np.array(a, copy=True))
        return numpy.sort(a, axis=axis, **kw)


def published(path, prefix=""):
    d = pd.read_csv(path, parse_dates=["settlement_date"])
    d = d[(d.settlement_date >= TEST_START) & (d.settlement_date <= TEST_END)].sort_values(KEY).reset_index(drop=True)
    return d, d[[prefix + c for c in QCOLS]].to_numpy()


mode = sys.argv[1]

if mode == "lstm":
    run = LSTM_RUN
    seeds, raws, sorts = (42, 7, 101), [], []
    for s in seeds:
        a = pd.read_csv(run / f"5q_seed{s}/predictions.csv", parse_dates=["settlement_date"])
        b = pd.read_csv(run / f"fill6_seed{s}/predictions.csv", parse_dates=["settlement_date"])
        m = a.merge(b.drop(columns="actual_price"), on=KEY).sort_values(KEY)
        m = m[(m.settlement_date >= TEST_START) & (m.settlement_date <= TEST_END)].reset_index(drop=True)
        raws.append(m[QCOLS].to_numpy())
        sorts.append(np.sort(m[QCOLS].to_numpy(), axis=1))
        y = m.actual_price.to_numpy()
    pub, final = published(run / "conformal_ENSEMBLE/predictions_2025_primary.csv")
    assert np.allclose(np.mean(sorts, axis=0), final, atol=1e-6), "per-seed sort then average must give Table 5 LSTM"
    per_seed = [stats(f"lstm_seed{s}", y, r, so) for s, r, so in zip(seeds, raws, sorts)]
    # pooled over seeds, and the ensemble-level CRPS effect of the per-seed rearrangement
    allraw = np.vstack(raws)
    d = np.diff(allraw, axis=1)
    inv = d < 0
    ens = {"model": "Encoder-Decoder LSTM", "rows_with_crossing": float(np.mean([p["rows_with_crossing"] for p in per_seed])),
           "inverted_pair_share": float(inv.mean()), "mean_magnitude": float(-d[inv].mean()),
           "median_magnitude": float(np.median(-d[inv])),
           "crps_unsorted": float(crps11(y, np.mean(raws, axis=0)).mean()), "crps_published": float(crps11(y, final).mean()),
           "pairs_with_crossing": [f"{QCOLS[j]}-{QCOLS[j + 1]}" for j in range(10) if inv[:, j].any()]}
    ens["crps_effect"] = ens["crps_unsorted"] - ens["crps_published"]
    ens["crps_effect_pct"] = 100 * ens["crps_effect"] / ens["crps_published"]
    (RES / "lstm.json").write_text(json.dumps(ens, indent=1))
    print("ENSEMBLE", json.dumps(ens, indent=1))

elif mode == "qrf":
    # QRF quantiles are read from one weighted empirical distribution; check two refits unsorted
    from quantile_forest import RandomForestQuantileRegressor
    from sklearn.impute import SimpleImputer
    table = model_table()
    feats = lean_features()
    worst = 0.0
    for day in [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-02")]:
        end = day - pd.Timedelta(days=2)
        win = table[table.settlement_date.between(end - pd.Timedelta(days=363), end)]
        imp = SimpleImputer(strategy="median").fit(win[feats])
        f = RandomForestQuantileRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5, random_state=42,
                                          n_jobs=-1).fit(imp.transform(win[feats]), win.price.to_numpy())
        rows = table[table.settlement_date.between(day, day + pd.Timedelta(days=6))]
        q = f.predict(imp.transform(rows[feats]), quantiles=list(LEVELS))
        worst = min(worst, float(np.diff(q, axis=1).min()))
    print("QRF unsorted: smallest adjacent difference over two refit weeks =", worst)
    (RES / "qrf.json").write_text(json.dumps({"model": "QRF", "min_adjacent_difference_unsorted": worst}))

elif mode == "qra":
    sys.path.insert(0, str(MODELLING / "qra"))
    import run_three_way_qra_qw as mod
    b = mod.b
    lstm = b.lstm_frame().rename(columns={c: f"lstm_{c}" for c in QCOLS})
    qrf = b.qrf_frame().rename(columns={c: f"qrf_{c}" for c in QCOLS})
    m = lstm.merge(qrf.drop(columns="actual_price"), on=KEY).merge(mod.lear_qra_frame(), on=KEY)
    all_oos = m.dropna().sort_values(KEY).reset_index(drop=True)
    test = all_oos[(all_oos.settlement_date >= TEST_START) & (all_oos.settlement_date <= TEST_END)].copy()
    y = test.actual_price.to_numpy()
    saved = pd.read_csv(QRA_OUT / "three_way_qw_metrics.csv").set_index("model")
    for name, label in [("two_way", "Two-model QRA"), ("three_way_qw", "Three-model QRA")]:
        spy = SortSpy()
        mod.np = spy
        srt, rate, _ = mod.qra_backtest(all_oos, test, name, 0.0)
        mod.np = numpy
        raw = spy.calls[-1]
        pub, final = published(QRA_OUT / "three_way_qw_predictions.csv", f"{name}_")
        assert (pub[KEY].values == test[KEY].values).all()
        assert np.allclose(srt, final, atol=1e-6), f"{name}: rerun does not reproduce published quantiles"
        assert abs(rate - saved.loc[name, "raw_crossing_rate"]) < 1e-12
        stats(label, y, raw, final)

elif mode == "lear_qra":
    sys.path.insert(0, str(MODELLING / "lear_qra"))
    import run_lear_qra_combine as mod
    merged = mod.load_merged()
    spy = SortSpy()
    mod.np = spy
    res, rate, _ = mod.combine(merged, mod.WINDOWS["principal"])
    mod.np = numpy
    raw = spy.calls[-1]
    keep = ((res.settlement_date >= TEST_START) & (res.settlement_date <= TEST_END)).to_numpy()
    pub, final = published(LEAR_QRA_OUT / "lear_qra_quantiles_principal.csv")
    r2025 = res[keep].reset_index(drop=True)
    assert (r2025[KEY].values == pub[KEY].values).all()
    assert np.allclose(r2025[QCOLS].to_numpy(), final, atol=1e-6), "LEAR-QRA rerun does not reproduce published"
    saved = pd.read_csv(LEAR_QRA_OUT / "lear_qra_metrics.csv").set_index("model")
    assert abs(rate - saved.loc["LEAR-QRA-principal", "raw_crossing_rate"]) < 1e-12
    stats("LEAR-QRA", pub.actual_price.to_numpy(), raw[keep], final)

elif mode == "linear":
    sys.path.insert(0, str(MODELLING / "tabular"))
    import fit_tabular_models as mod
    table = model_table()
    feats = lean_features()
    lin = [c for c in feats if c not in ("hour", "hour_sin", "hour_cos")]
    days = sorted(table.loc[table.settlement_date.between(TEST_START, TEST_END), "settlement_date"].unique())
    idx = [list(mod.FIT_LEVELS).index(t) for t in LEVELS]
    fitted, raws, ys, keys = None, [], [], []
    for i, day in enumerate(days):
        if fitted is None or i % mod.REFIT["linear_qr"] == 0:
            fitted = mod.fit_linear(mod.calibration(table, day), lin)
            print(f"refit {i // 7:2d} {pd.Timestamp(day):%Y-%m-%d}", flush=True)
        rows = table[table.settlement_date == day]
        raws.append(mod.predict_linear(fitted, rows, lin))
        ys.append(rows.price.to_numpy())
        keys.append(rows[KEY])
    raw15 = np.vstack(raws)
    final15 = np.sort(raw15, axis=1)
    pub, final = published(SNAP / "linear_qr_test_predictions.csv")
    assert (pd.concat(keys).values == pub[KEY].values).all()
    assert np.allclose(final15[:, idx], final, atol=1e-6), "linear QR rerun does not reproduce published quantiles"
    np.save(RES / "linear_raw15.npy", raw15)
    stats("Linear QR", np.concatenate(ys), raw15[:, idx], final)

elif mode == "xgboost":
    argv, sys.argv = sys.argv, ["xgboost_comparator.py", "import-only"]     # module runs nothing for this mode
    import xgboost_comparator as core
    sys.argv = argv
    spy = SortSpy()
    core.np = spy
    pred, log = core.backtest(TEST_START, TEST_END, 2)
    core.np = numpy
    raw = np.vstack(spy.calls)
    pub, final = published(OUT / "xgboost" / "predictions.csv")
    assert np.allclose(pred[QCOLS].to_numpy(), final, atol=1e-6), "XGBoost rerun does not reproduce saved quantiles"
    stats("XGBoost", pub.actual_price.to_numpy(), raw, final)
