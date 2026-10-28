"""Feature-matched QRF (Section 5.2.2; Tables 8 and A.19).

One re-run of the QRF pipeline of modelling/tabular/fit_tabular_models.py (same
estimator, hyperparameters, median imputation, pooled fit across hours, 15 fitted
levels, weekly refit on the test-day index, window ending D-2) with two changes that
match the Encoder-Decoder LSTM:

  * inputs: the LSTM's information set only, i.e. its 13 known-future decoder
    channels for the delivery hour plus the 24 cleared prices of D-1 that its encoder
    reads (built exactly as in modelling/lstm/test_backtest.py);
  * window: 728 delivery days ending D-2 instead of 364. The LSTM holds out the final
    14 of these days for early stopping; QRF has no early stopping, so it uses all 728.

Features and window are therefore matched jointly. The 'gate' mode reruns the headline
QRF configuration and must reproduce its saved forecasts before the matched run is used.

Usage: python feature_matched_qrf.py gate [days]   # headline config, must reproduce saved QRF
       python feature_matched_qrf.py matched      # the feature-matched run
"""
import sys
import time

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer

from rev_common import KEY, OUT, RQ1, TEST_END, TEST_START, lean_features, model_table

FIT_LEVELS = np.array([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])
FIT_QCOLS = [f"q{int(q * 100):02d}" for q in FIT_LEVELS]
INFORMATION_LAG, REFIT = 2, 7
KNOWN = ["demand_forecast", "transmission_wind_forecast", "embedded_wind_forecast", "embedded_solar_forecast",
         "gas_price", "carbon_price", "opmr_surplus_norm", "hour_sin", "hour_cos", "month_sin", "month_cos",
         "dayofweek", "is_bank_holiday"]
PROFILE = [f"price_dm1_h{h:02d}" for h in range(24)]

mode = sys.argv[1]
table = model_table()

if mode == "gate":
    FEATURES = lean_features()
    WINDOW = 364
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
else:
    # D-1 profile exactly as the LSTM encoder sees it (one-step forward interpolation across hours)
    wide = (table.pivot_table(index="settlement_date", columns="hour", values="price", aggfunc="first")
            .sort_index().interpolate(axis=1, limit=1, limit_direction="forward"))
    wide = wide.reindex(columns=range(24))
    wide.columns = PROFILE
    wide.index = wide.index + pd.Timedelta(days=1)          # prices of D-1 attached to delivery day D
    table = table.merge(wide, left_on="settlement_date", right_index=True, how="left")
    FEATURES = KNOWN + PROFILE
    WINDOW = 728
    n_days = 365
assert len(FEATURES) == (37 if mode == "matched" else 34), len(FEATURES)


def calibration(day):
    end = day - pd.Timedelta(days=INFORMATION_LAG)
    start = end - pd.Timedelta(days=WINDOW - 1)
    return table[table["settlement_date"].between(start, end)]


def fit(data):
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(data[FEATURES])
    model = RandomForestQuantileRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5,
                                          random_state=42, n_jobs=-1)
    model.fit(x, data["price"].to_numpy())
    return imputer, model


days = sorted(table.loc[table.settlement_date.between(TEST_START, TEST_END), "settlement_date"].unique())[:n_days]
fitted, frames, refits, t0 = None, [], [], time.time()
for i, day in enumerate(days):
    if fitted is None or i % REFIT == 0:
        train = calibration(day)
        t1 = time.time()
        fitted = fit(train)
        refits.append({"refit_index": len(refits), "test_day": day, "cal_start": train.settlement_date.min(),
                       "cal_end": train.settlement_date.max(), "cal_rows": len(train),
                       "fit_seconds": round(time.time() - t1, 1)})
        print(f"refit {len(refits) - 1:2d} {day:%Y-%m-%d} rows {len(train):5d} "
              f"({refits[-1]['fit_seconds']}s, {time.time() - t0:.0f}s total)", flush=True)
    rows = table[table.settlement_date == day]
    imputer, model = fitted
    q = np.sort(model.predict(imputer.transform(rows[FEATURES]), quantiles=FIT_LEVELS.tolist()), axis=1)
    frame = pd.DataFrame({"settlement_date": rows.settlement_date.to_numpy(), "hour": rows.hour.to_numpy(),
                          "actual_price": rows.price.to_numpy()})
    for j, c in enumerate(FIT_QCOLS):
        frame[c] = q[:, j]
    frames.append(frame)
pred = pd.concat(frames, ignore_index=True)

if mode == "gate":
    saved = pd.read_csv(RQ1 / "quantile_forest/predictions.csv", parse_dates=["settlement_date"])
    saved = saved[saved.settlement_date.isin(days)].sort_values(KEY).reset_index(drop=True)
    assert len(saved) == len(pred) and (saved[KEY].values == pred[KEY].values).all()
    dev = np.abs(saved[FIT_QCOLS].to_numpy() - pred[FIT_QCOLS].to_numpy()).max()
    log = pd.read_csv(RQ1 / "quantile_forest/refit_log.csv", parse_dates=["cal_start", "cal_end"])
    same_windows = all((log.loc[r["refit_index"], ["cal_start", "cal_end", "cal_rows"]].tolist()
                        == [r["cal_start"], r["cal_end"], r["cal_rows"]]) for r in refits)
    print(f"gate over {len(days)} days: max |quantile difference| = {dev:.2e}; refit windows identical: {same_windows}")
    print("GATE", "PASS" if dev < 1e-6 and same_windows else "FAIL")
else:
    d = OUT / "feature_matched_qrf"
    d.mkdir(exist_ok=True)
    pred.to_csv(d / "predictions.csv", index=False)
    pd.DataFrame(refits).to_csv(d / "refit_log.csv", index=False)
    pd.Series(FEATURES, name="feature").to_csv(d / "features.csv", index=False)
    print(f"saved {len(pred)} forecasts, {len(refits)} refits, {time.time() - t0:.0f}s")
