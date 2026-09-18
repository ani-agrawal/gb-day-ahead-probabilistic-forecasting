"""Package equivalence check: our LEAR recipe vs the actual epftoolbox LEAR.

Runs in the isolated .venv_epf so both engines share one sklearn/numpy build
(package versions cancel). Uses a canonical two-exogenous configuration that the
packaged LEAR supports natively, and isolates exactly the parts we
re-implemented (invariant scaling + two-step estimator) by feeding BOTH engines
epftoolbox's own Xtrain/Ytrain/Xtest matrices. Reports, to tight tolerance:
  (a) invariant scaler:   our InvariantScaler vs epftoolbox scaling('Invariant')
  (b) feature construction: our pandas builder vs epftoolbox _build_and_split_XYs
  (c) selected penalties:  all 24 AIC-selected lambdas
  (d) coefficient vectors: all 24 hourly LASSO coef vectors
  (e) forecasts:           all 24 hourly day-ahead point forecasts
Any nonzero difference is printed and must be explained.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))              # import our InvariantScaler
from run_lear import InvariantScaler              # noqa: E402

SNAP = HERE.parent.parent / "input_snapshot"

# --- load epftoolbox LEAR directly, bypassing the TensorFlow DNN import path ---
import epftoolbox  # noqa: E402
from epftoolbox.data import scaling as epf_scaling  # noqa: E402
_base = Path(epftoolbox.__file__).parent
_spec = importlib.util.spec_from_file_location("epf_lear_direct", _base / "models" / "_lear.py")
_lear = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lear)
LEAR = _lear.LEAR

W = 360                                            # calibration window (days) for the check
FORECAST_DAYS = pd.date_range("2024-12-18", "2024-12-22", freq="D")  # 5 clean days


def epf_predict(model, X):
    """epftoolbox LEAR.predict, made numpy-2 safe (its Yp[h]=predict(X) assigns a
    size-1 array, which numpy 2 rejects). Uses the package's OWN fitted scalerX,
    per-hour Lasso models and scalerY, so it stays faithful to the package."""
    X = X.copy()
    X[:, :-7] = model.scalerX.transform(X[:, :-7])
    yp = np.array([float(model.models[h].predict(X)[0]) for h in range(24)])
    return model.scalerY.inverse_transform(yp.reshape(1, -1)).ravel()


def canonical_df() -> pd.DataFrame:
    """epftoolbox format: hourly index, ['Price','Exogenous 1','Exogenous 2'], no gaps."""
    df = pd.read_parquet(SNAP / "model_table_hourly_epex.parquet")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df[(df.settlement_date >= "2023-06-01") & (df.settlement_date <= "2024-12-31")]
    idx = df["settlement_date"] + pd.to_timedelta(df["hour"], unit="h")
    out = pd.DataFrame({"Price": df["price"].values,
                        "Exogenous 1": df["demand_forecast"].values,
                        "Exogenous 2": df["transmission_wind_forecast"].values}, index=idx.values)
    out.index = pd.DatetimeIndex(out.index)
    full = pd.date_range(out.index.min().normalize(), out.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    out = out.reindex(full).interpolate("linear").bfill().ffill()   # clean the 23h day + isolated gaps
    return out


def our_scale_recipe(Xtrain, Ytrain, Xtest):
    """Replicate epftoolbox.recalibrate with OUR InvariantScaler + two-step LASSO."""
    from sklearn.linear_model import Lasso, LassoLarsIC
    sy = InvariantScaler().fit(Ytrain)
    Ys = sy.transform(Ytrain)
    sx = InvariantScaler().fit(Xtrain[:, :-7])
    Xtr = Xtrain.copy(); Xtr[:, :-7] = sx.transform(Xtrain[:, :-7])
    Xte = Xtest.copy(); Xte[:, :-7] = sx.transform(Xtest[:, :-7])
    alphas, coefs, yp = [], [], np.zeros(24)
    for h in range(24):
        a = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xtr, Ys[:, h]).alpha_
        m = Lasso(alpha=a, max_iter=2500).fit(Xtr, Ys[:, h])
        alphas.append(a); coefs.append(m.coef_.copy())
        yp[h] = float(m.predict(Xte)[0])
    yp = np.sinh(yp) * sy.mad_ + sy.median_       # inverse invariant (per-column)
    return np.array(alphas), np.array(coefs), yp, (sx, sy)


def our_features_for_date(df, date):
    """Build the 240 continuous feature values our way, to compare against epftoolbox as a multiset."""
    vals = []
    P = df["Price"]
    for hour in range(24):
        t = date + pd.Timedelta(hours=hour)
        for pd_ in (1, 2, 3, 7):
            vals.append(P.loc[t - pd.Timedelta(hours=24 * pd_)])
    for hour in range(24):
        t = date + pd.Timedelta(hours=hour)
        for off in (0, 1, 7):
            for ex in ("Exogenous 1", "Exogenous 2"):
                vals.append(df[ex].loc[t - pd.Timedelta(hours=24 * off)])
    return np.sort(np.array(vals, dtype=float))


def main():
    df = canonical_df()
    print(f"canonical df: {df.shape[0]} hourly rows, {df.index[0]} .. {df.index[-1]}, no NaN={df.notna().all().all()}\n")
    model = LEAR(calibration_window=W)

    sc_max, feat_max, pen_max, coef_max, fc_max = 0.0, 0.0, 0.0, 0.0, 0.0
    for date in FORECAST_DAYS:
        df_train = df.loc[:date - pd.Timedelta(hours=1)].iloc[-W * 24:]
        df_test = df.loc[date - pd.Timedelta(weeks=2):]
        Xtrain, Ytrain, Xtest = model._build_and_split_XYs(df_train=df_train, df_test=df_test, date_test=date)

        # (a) scaler equivalence on the continuous block
        [epf_scaled], _ = epf_scaling([Xtrain[:, :-7].copy()], "Invariant")
        our_scaled = InvariantScaler().fit(Xtrain[:, :-7]).transform(Xtrain[:, :-7])
        sc_max = max(sc_max, np.nanmax(np.abs(epf_scaled - our_scaled)))

        # (b) feature construction: our multiset vs epftoolbox's Xtest continuous values
        feat_max = max(feat_max, np.max(np.abs(our_features_for_date(df, date) - np.sort(Xtest[0, :-7]))))

        # epftoolbox forecast + its fitted penalties/coefs
        model.recalibrate(Xtrain=Xtrain.copy(), Ytrain=Ytrain.copy())
        epf_alpha = np.array([model.models[h].alpha for h in range(24)])
        epf_coef = np.array([model.models[h].coef_ for h in range(24)])
        epf_yp = epf_predict(model, Xtest.copy())

        # our recipe on the identical matrices
        our_alpha, our_coef, our_yp, _ = our_scale_recipe(Xtrain.copy(), Ytrain.copy(), Xtest.copy())

        pen_max = max(pen_max, np.max(np.abs(epf_alpha - our_alpha)))
        coef_max = max(coef_max, np.max(np.abs(epf_coef - our_coef)))
        fc_max = max(fc_max, np.max(np.abs(epf_yp - our_yp)))
        print(f"{date.date()}: forecast max|Δ|={np.max(np.abs(epf_yp-our_yp)):.2e} GBP/MWh, "
              f"penalty max|Δ|={np.max(np.abs(epf_alpha-our_alpha)):.2e}, "
              f"epf h0-h3={np.round(epf_yp[:4],2)}")

    print("\n=== equivalence summary (max abs diff across 5 days x 24 hours) ===")
    rows = [("(a) invariant scaler", sc_max, 1e-9),
            ("(b) feature construction (multiset)", feat_max, 1e-6),
            ("(c) AIC penalties (24)", pen_max, 1e-9),
            ("(d) LASSO coefficient vectors", coef_max, 1e-8),
            ("(e) day-ahead forecasts (GBP/MWh)", fc_max, 1e-6)]
    for name, val, tol in rows:
        print(f"{name:<42}max|Δ|={val:.2e}  tol={tol:.0e}  {'PASS' if val <= tol else 'REVIEW'}")
    ok = all(v <= t for _, v, t in rows)
    print("\nEQUIVALENCE CONFIRMED" if ok else "\nDIFFERENCES EXCEED TOLERANCE — see per-line REVIEW")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
