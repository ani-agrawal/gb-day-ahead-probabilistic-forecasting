"""Post-hoc grouped-input ablation of the headline QRF and linear QR over the
the 2025 held-out evaluation year.

The analysis was conducted after the headline models were frozen; it is for
interpretation only and does not inform model or feature selection. The held-out
test year is used post-hoc for this
interpretation; they were never used to select or modify the headline models.

Arms mirror the RNN source-group ablation. Removing a forecast source also
removes every derived feature containing it (net demand, renewable share,
low-residual-load), so the demand/wind/solar arms overlap in their removals;
effects are contribution sensitivities, not additive decompositions, and the
overlap is disclosed. Generation-mix outturn lags stay in their own family
(covered by the common-core run), so gen_wind_mw_lag2d is NOT removed by the
no_wind arm. The 'additional tabular sources removed' row is the common-core
run (hourly_model_recal_epex_commoncore), not rerun here.

Design mirrors 03_model_training.py exactly: 364-day rolling windows ending
D-2, weekly refits for both models, per-hour linear QR (alpha 0.01, highs
solver) dropping the hour terms, QRF(500, leaf 5, max_features 0.5, seed 42),
median imputation (and scaling, linear) per window per hour, 15-level fitting
grid (the paper evaluates the shared 11 levels),
row-sort. Resumable per arm/model (skips completed outputs).

Outputs -> 06_outputs/RQ1/model_runs/ablation_testperiod/<arm>/{qrf,linear}/
"""
from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from quantile_forest import RandomForestQuantileRegressor

QUANTILES = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]
Q_COLS = [f"q{int(q * 100):02d}" for q in QUANTILES]
RANDOM_STATE = 42
CAL_WINDOW_DAYS = 364
TARGET_LAG_DAYS = 2
REFIT_EVERY = 7
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")

DERIVED_DWS = ["net_demand_forecast", "renewable_share_forecast",
               "low_residual_load_mw", "low_residual_load_flag"]
ARMS = {
    "no_price_history": ["price_lag1d", "price_lag2d", "price_lag7d"],
    "no_demand": ["demand_forecast"] + DERIVED_DWS,
    "no_wind": ["transmission_wind_forecast", "embedded_wind_forecast"] + DERIVED_DWS,
    "no_solar": ["embedded_solar_forecast"] + DERIVED_DWS,
    "no_gas": ["gas_price"],
    "no_carbon": ["carbon_price"],
    "no_opmr": ["opmr_national_surplus", "opmr_surplus_norm", "opmr_surplus_rev_7d"],
    "no_calendar": ["hour", "hour_sin", "hour_cos", "dayofweek",
                    "month_sin", "month_cos", "is_bank_holiday"],
}

ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "06_outputs" / "RQ1" / "model_runs" / "ablation_testperiod"
model = pd.read_parquet(ROOT / "04_data" / "processed" / "model_table_hourly_epex.parquet")
model["settlement_date"] = pd.to_datetime(model["settlement_date"])
model = model.sort_values(["settlement_date", "hour"]).reset_index(drop=True)
lean = pd.read_csv(ROOT / "04_data" / "metadata" / "lean_feature_list_hourly.csv")["feature"].tolist()
test_days = sorted(model.loc[(model["settlement_date"] >= TEST_START)
                             & (model["settlement_date"] <= TEST_END), "settlement_date"].unique())
print(f"tabular ablation: {len(ARMS)} arms, {len(test_days)} test days", flush=True)


def atomic_csv(frame, path):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def frame_from(rows, qm):
    f = pd.DataFrame({"settlement_date": rows["settlement_date"].values,
                      "hour": rows["hour"].values,
                      "actual_price": rows["price"].values})
    for j, c in enumerate(Q_COLS):
        f[c] = qm[:, j]
    return f


def run_qrf(feats, out_dir):
    fitted, frames = None, []
    for i, day in enumerate(test_days):
        if fitted is None or i % REFIT_EVERY == 0:
            end = day - pd.Timedelta(days=TARGET_LAG_DAYS)
            start = end - pd.Timedelta(days=CAL_WINDOW_DAYS - 1)
            cal = model[(model.settlement_date >= start) & (model.settlement_date <= end)]
            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(cal[feats])
            qrf = RandomForestQuantileRegressor(n_estimators=500, min_samples_leaf=5,
                                                max_features=0.5, random_state=RANDOM_STATE,
                                                n_jobs=-1)
            qrf.fit(X, cal["price"].values)
            fitted = (imputer, qrf)
        rows = model[model.settlement_date == day]
        qm = np.sort(fitted[1].predict(fitted[0].transform(rows[feats]),
                                       quantiles=QUANTILES), axis=1)
        frames.append(frame_from(rows, qm))
    preds = pd.concat(frames, ignore_index=True)
    atomic_csv(preds, out_dir / "predictions.csv")


def run_linear(feats, out_dir):
    lin_feats = [c for c in feats if c not in ("hour", "hour_sin", "hour_cos")]
    fitted, frames = None, []
    for i, day in enumerate(test_days):
        if fitted is None or i % REFIT_EVERY == 0:
            end = day - pd.Timedelta(days=TARGET_LAG_DAYS)
            start = end - pd.Timedelta(days=CAL_WINDOW_DAYS - 1)
            cal = model[(model.settlement_date >= start) & (model.settlement_date <= end)]
            fitted = {}
            for h, cal_h in cal.groupby("hour"):
                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                X = scaler.fit_transform(imputer.fit_transform(cal_h[lin_feats]))
                y = cal_h["price"].values
                models = {}
                for qi, q in enumerate(QUANTILES):
                    qr = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
                    qr.fit(X, y)
                    models[Q_COLS[qi]] = qr
                fitted[h] = (imputer, scaler, models)
        rows = model[model.settlement_date == day]
        hours = rows["hour"].values
        out = np.zeros((len(rows), len(QUANTILES)))
        for h in np.unique(hours):
            fh = fitted.get(h) or fitted[min(fitted, key=lambda k: abs(k - h))]
            sel = hours == h
            X = fh[1].transform(fh[0].transform(rows.loc[sel, lin_feats]))
            for qi, qc in enumerate(Q_COLS):
                out[sel, qi] = fh[2][qc].predict(X)
        frames.append(frame_from(rows, np.sort(out, axis=1)))
    preds = pd.concat(frames, ignore_index=True)
    atomic_csv(preds, out_dir / "predictions.csv")


for arm, dropped in ARMS.items():
    feats = [c for c in lean if c not in dropped]
    n_expected = len(lean) - len([d for d in dropped if d in lean])
    assert len(feats) == n_expected, arm
    for mdl, runner in [("qrf", run_qrf), ("linear", run_linear)]:
        out_dir = OUT_BASE / arm / mdl
        if (out_dir / "predictions.csv").exists():
            print(f"SKIP {arm}/{mdl}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"RUN {arm}/{mdl}: {len(feats)} features "
              f"(dropped {len(lean) - len(feats)})", flush=True)
        runner(feats, out_dir)
        print(f"done {arm}/{mdl}", flush=True)
print("TABULAR_ABLATION_DONE", flush=True)
