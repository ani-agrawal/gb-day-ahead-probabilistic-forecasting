"""Fit the historical, linear-QR and QRF benchmarks over held-out 2025."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "04_data/processed/model_table_hourly_epex.parquet"
FEATURES = ROOT / "04_data/metadata/lean_feature_list_hourly.csv"
OUT = ROOT / "06_outputs/RQ1/model_runs/hourly_model_recal_epex"

FIT_LEVELS = np.array([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])
FIT_QCOLS = [f"q{int(q * 100):02d}" for q in FIT_LEVELS]
PAPER_LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                         0.60, 0.70, 0.80, 0.90, 0.95])
PAPER_QCOLS = [f"q{int(q * 100):02d}" for q in PAPER_LEVELS]
START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
WINDOW_DAYS, INFORMATION_LAG = 364, 2
REFIT = {"gaussian_baseline": 1, "linear_qr": 7, "quantile_forest": 7}


def assemble_day(rows: pd.DataFrame, quantiles: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({
        "settlement_date": rows["settlement_date"].to_numpy(),
        "hour": rows["hour"].to_numpy(),
        "actual_price": rows["price"].to_numpy(),
    })
    if "delivery_start_utc" in rows:
        frame.insert(2, "delivery_start_utc", rows["delivery_start_utc"].to_numpy())
    for j, column in enumerate(FIT_QCOLS):
        frame[column] = np.sort(quantiles, axis=1)[:, j]
    return frame


def calibration(table: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    end = day - pd.Timedelta(days=INFORMATION_LAG)
    start = end - pd.Timedelta(days=WINDOW_DAYS - 1)
    return table[table["settlement_date"].between(start, end)]


def fit_historical(data: pd.DataFrame):
    by_hour = data.groupby("hour")["price"].quantile(FIT_LEVELS).unstack()
    by_hour.columns = FIT_QCOLS
    return by_hour, data["price"].quantile(FIT_LEVELS).to_numpy()


def predict_historical(fitted, rows: pd.DataFrame) -> np.ndarray:
    by_hour, fallback = fitted
    return np.vstack([
        by_hour.loc[hour].to_numpy() if hour in by_hour.index else fallback
        for hour in rows["hour"]
    ])


def fit_linear(data: pd.DataFrame, features: list[str]):
    fitted = {}
    for hour, hourly in data.groupby("hour"):
        imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
        x = scaler.fit_transform(imputer.fit_transform(hourly[features]))
        models = []
        for level in FIT_LEVELS:
            model = QuantileRegressor(quantile=level, alpha=0.01, solver="highs")
            model.fit(x, hourly["price"].to_numpy())
            models.append(model)
        fitted[hour] = (imputer, scaler, models)
    return fitted


def predict_linear(fitted, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    output = np.zeros((len(rows), len(FIT_LEVELS)))
    hours = rows["hour"].to_numpy()
    for hour in np.unique(hours):
        selected = hours == hour
        key = hour if hour in fitted else min(fitted, key=lambda h: abs(h - hour))
        imputer, scaler, models = fitted[key]
        x = scaler.transform(imputer.transform(rows.loc[selected, features]))
        output[selected] = np.column_stack([model.predict(x) for model in models])
    return output


def fit_qrf(data: pd.DataFrame, features: list[str]):
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(data[features])
    model = RandomForestQuantileRegressor(
        n_estimators=500, min_samples_leaf=5, max_features=0.5,
        random_state=42, n_jobs=-1,
    )
    model.fit(x, data["price"].to_numpy())
    return imputer, model


def predict_qrf(fitted, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    imputer, model = fitted
    return model.predict(imputer.transform(rows[features]), quantiles=FIT_LEVELS.tolist())


def backtest(name: str, table: pd.DataFrame, fit, predict) -> pd.DataFrame:
    directory = OUT / name
    prediction_path = directory / "predictions.csv"
    if prediction_path.exists():
        print(f"[skip] {name}: {prediction_path.relative_to(ROOT)} exists")
        return pd.read_csv(prediction_path, parse_dates=["settlement_date"])

    days = sorted(table.loc[table["settlement_date"].between(START, END),
                            "settlement_date"].unique())
    fitted = None
    frames, refits = [], []
    for i, day in enumerate(days):
        if fitted is None or i % REFIT[name] == 0:
            train = calibration(table, day)
            fitted = fit(train)
            refits.append({
                "refit_index": len(refits), "test_day": day,
                "window_mode": "rolling", "cal_start": train["settlement_date"].min(),
                "cal_end": train["settlement_date"].max(), "cal_rows": len(train),
            })
        rows = table[table["settlement_date"] == day]
        frames.append(assemble_day(rows, predict(fitted, rows)))

    predictions = pd.concat(frames, ignore_index=True)
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(refits).to_csv(directory / "refit_log.csv", index=False)
    print(f"[done] {name}: {len(refits)} refits, {len(predictions):,} forecasts")
    return predictions


def score(name: str, frame: pd.DataFrame) -> dict[str, float | str | int]:
    y, q = frame["actual_price"].to_numpy(), frame[PAPER_QCOLS].to_numpy()
    error = y[:, None] - q
    pinball = np.where(error >= 0, PAPER_LEVELS * error,
                       (PAPER_LEVELS - 1) * error)
    inside = (y >= q[:, 0]) & (y <= q[:, -1])
    width = q[:, -1] - q[:, 0]
    winkler = width + 20 * (q[:, 0] - y) * (y < q[:, 0]) + 20 * (y - q[:, -1]) * (y > q[:, -1])
    return {
        "model": name, "n_hours": len(frame),
        "CRPS_11": float(np.mean(2 * np.trapezoid(pinball, x=PAPER_LEVELS, axis=1))),
        "MAE_q50": float(np.mean(np.abs(y - q[:, 5]))),
        "cov_90": float(inside.mean()), "width_90": float(width.mean()),
        "winkler_90": float(winkler.mean()),
    }


def main() -> None:
    table = pd.read_parquet(TABLE)
    table["settlement_date"] = pd.to_datetime(table["settlement_date"])
    table = table.sort_values(["settlement_date", "hour"]).reset_index(drop=True)
    features = pd.read_csv(FEATURES)["feature"].tolist()
    linear_features = [c for c in features if c not in ("hour", "hour_sin", "hour_cos")]

    forecasts = {
        "historical_baseline": backtest(
            "gaussian_baseline", table, fit_historical, predict_historical
        ),
        "linear_qr": backtest(
            "linear_qr", table,
            lambda data: fit_linear(data, linear_features),
            lambda fitted, rows: predict_linear(fitted, rows, linear_features),
        ),
        "qrf": backtest(
            "quantile_forest", table,
            lambda data: fit_qrf(data, features),
            lambda fitted, rows: predict_qrf(fitted, rows, features),
        ),
    }
    metrics = pd.DataFrame([score(name, frame) for name, frame in forecasts.items()])
    comparison = OUT / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(comparison / "model_comparison.csv", index=False)
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
