"""Shared inputs and scoring helpers for the reported QRA models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "input_snapshot"

LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
          0.60, 0.70, 0.80, 0.90, 0.95]
QCOLS = [f"q{int(level * 100):02d}" for level in LEVELS]
COLS_5Q = ["q05", "q10", "q50", "q90", "q95"]
COLS_F6 = ["q20", "q30", "q40", "q60", "q70", "q80"]
KEY = ["settlement_date", "hour"]
LEV = np.asarray(LEVELS)
BLOCK_DAYS, LAG_DAYS = 7, 2
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
SEEDS = (42, 7, 101)


def read(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["settlement_date"])
    return frame[KEY + ["actual_price", *columns]].sort_values(KEY).reset_index(drop=True)


def dev_lstm_ensemble() -> pd.DataFrame:
    per_seed, reference = [], None
    for seed in SEEDS:
        principal = read(SNAP / f"lstm_dev_5q_seed{seed}.csv", COLS_5Q)
        interior = read(SNAP / f"lstm_dev_fill6_seed{seed}.csv", COLS_F6)
        merged = principal.merge(interior, on=KEY, suffixes=("", "_fill6"))
        if reference is None:
            reference = merged[KEY + ["actual_price"]].copy()
        per_seed.append(merged[QCOLS].to_numpy())
    assert reference is not None
    ensemble = reference.copy()
    ensemble[QCOLS] = np.sort(np.mean(per_seed, axis=0), axis=1)
    return ensemble


def lstm_frame() -> pd.DataFrame:
    development = dev_lstm_ensemble()
    test = read(SNAP / "lstm_test_ensemble11q_predictions.csv", QCOLS)
    test = test[test["settlement_date"].between(TEST_START, TEST_END)]
    return pd.concat([development, test], ignore_index=True)


def qrf_frame() -> pd.DataFrame:
    development = read(SNAP / "qrf_dev2024H2_predictions.csv", QCOLS)
    test = read(SNAP / "qrf_test_predictions.csv", QCOLS)
    test = test[test["settlement_date"].between(TEST_START, TEST_END)]
    return pd.concat([development, test], ignore_index=True)


def crps11(actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    error = actual[:, None] - quantiles
    pinball = np.where(error >= 0, LEV * error, (LEV - 1) * error)
    return 2 * np.trapezoid(pinball, x=LEV, axis=1)


def score(actual: np.ndarray, quantiles: np.ndarray, name: str) -> dict[str, float | str]:
    error = actual[:, None] - quantiles
    pinball = np.where(error >= 0, LEV * error, (LEV - 1) * error)
    lower, upper = quantiles[:, 0], quantiles[:, -1]
    inside = (actual >= lower) & (actual <= upper)
    winkler = ((upper - lower)
               + 20.0 * (lower - actual) * (actual < lower)
               + 20.0 * (actual - upper) * (actual > upper))
    negative, high150, high200 = actual < 0, actual > 150, actual > 200
    return {
        "model": name,
        "crps_11": float(np.mean(crps11(actual, quantiles))),
        "mean_pinball_11q": float(pinball.mean()),
        "mae_q50": float(np.mean(np.abs(actual - quantiles[:, 5]))),
        "cov90": float(inside.mean()),
        "width90": float(np.mean(upper - lower)),
        "winkler90": float(winkler.mean()),
        "cov_neg": float(inside[negative].mean()) if negative.any() else np.nan,
        "cov_gt150": float(inside[high150].mean()) if high150.any() else np.nan,
        "cov_gt200": float(inside[high200].mean()) if high200.any() else np.nan,
    }


def dm_pvalue(loss_a: pd.Series, loss_b: pd.Series) -> float:
    difference = (loss_a - loss_b).to_numpy()
    sample_size = len(difference)
    mean = float(difference.mean())
    lag = int(np.floor(1.5 * sample_size ** (1 / 3)))
    centred = difference - mean
    variance = np.mean(centred ** 2) + sum(
        2 * (1 - k / (lag + 1)) * np.mean(centred[k:] * centred[:-k])
        for k in range(1, lag + 1)
    )
    return float(stats.norm.cdf(mean / np.sqrt(variance / sample_size)))
