"""Generate the QRF's leakage-safe 2024-H2 development forecasts for QRA.

The specification matches the reported QRF: 364-day rolling windows ending
D-2, weekly refits, the frozen lean feature list, 500 trees, leaf size 5,
max_features 0.5 and random_state 42. Quantiles are monotonically rearranged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer


QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95]
Q_COLS = [f"q{int(q * 100):02d}" for q in QUANTILES]
CAL_WINDOW_DAYS = 364
TARGET_LAG_DAYS = 2
REFIT_EVERY = 7
TEST_START = pd.Timestamp("2024-07-01")
TEST_END = pd.Timestamp("2024-12-31")

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "04_data" / "processed" / "model_table_hourly_epex.parquet"
FEATURES = ROOT / "04_data" / "metadata" / "lean_feature_list_hourly.csv"
OUT = (ROOT / "06_outputs" / "RQ1" / "model_runs" /
       "hourly_model_recal_epex" / "tuning" / "qrf_dev2024H2_predictions.csv")


def main() -> None:
    model = pd.read_parquet(TABLE)
    model["settlement_date"] = pd.to_datetime(model["settlement_date"])
    model = model.sort_values(["settlement_date", "hour"]).reset_index(drop=True)
    features = pd.read_csv(FEATURES)["feature"].tolist()
    test_days = sorted(model.loc[
        model["settlement_date"].between(TEST_START, TEST_END), "settlement_date"
    ].unique())

    fitted = None
    frames: list[pd.DataFrame] = []
    for i, day in enumerate(test_days):
        if fitted is None or i % REFIT_EVERY == 0:
            end = day - pd.Timedelta(days=TARGET_LAG_DAYS)
            start = end - pd.Timedelta(days=CAL_WINDOW_DAYS - 1)
            calibration = model[model["settlement_date"].between(start, end)]
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(calibration[features])
            forest = RandomForestQuantileRegressor(
                n_estimators=500,
                min_samples_leaf=5,
                max_features=0.5,
                random_state=42,
                n_jobs=-1,
            )
            forest.fit(x_train, calibration["price"].to_numpy())
            fitted = (imputer, forest)
            print(f"refit {day:%Y-%m-%d}: {start:%Y-%m-%d} to {end:%Y-%m-%d}",
                  flush=True)

        rows = model[model["settlement_date"] == day]
        quantiles = fitted[1].predict(
            fitted[0].transform(rows[features]), quantiles=QUANTILES
        )
        quantiles = np.sort(quantiles, axis=1)
        frame = pd.DataFrame({
            "settlement_date": rows["settlement_date"].to_numpy(),
            "hour": rows["hour"].to_numpy(),
            "actual_price": rows["price"].to_numpy(),
        })
        for j, column in enumerate(Q_COLS):
            frame[column] = quantiles[:, j]
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    if len(result) != 4416 or result.duplicated(["settlement_date", "hour"]).any():
        raise AssertionError("unexpected 2024-H2 QRF delivery grid")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"written {len(result):,} forecasts -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
