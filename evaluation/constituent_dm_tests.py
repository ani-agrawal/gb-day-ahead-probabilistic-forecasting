"""One-sided Diebold--Mariano tests for the paper's constituent models.

Loss is daily total CRPS from the common eleven-quantile grid. Newey--West HAC
variance uses lag floor(1.5*T**(1/3)). The QRA scripts save their own DM tables;
this file supplies QRF versus the historical/linear benchmarks and the
Encoder--Decoder LSTM versus QRF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.95])
COLS = [f"q{int(t * 100):02d}" for t in LEVELS]
KEY = ["settlement_date", "hour"]
ROOT = Path(__file__).resolve().parents[1]
RQ1 = ROOT / "06_outputs/RQ1/model_runs/hourly_model_recal_epex"
RQ2 = ROOT / "06_outputs/RQ2/model_runs/seq2seq_final_b16es14/ENSEMBLE_11q"
SOURCES = {
    "historical": RQ1 / "gaussian_baseline/predictions.csv",
    "linear_qr": RQ1 / "linear_qr/predictions.csv",
    "qrf": RQ1 / "quantile_forest/predictions.csv",
    "lstm": RQ2 / "predictions.csv",
}
COMPARISONS = (("qrf", "historical"), ("qrf", "linear_qr"), ("lstm", "qrf"))
START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")


def crps_rows(frame: pd.DataFrame) -> np.ndarray:
    y = frame["actual_price"].to_numpy()
    q = frame[COLS].to_numpy()
    error = y[:, None] - q
    pinball = np.where(error >= 0, LEVELS * error, (LEVELS - 1) * error)
    return 2 * np.trapezoid(pinball, x=LEVELS, axis=1)


def dm_daily(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int]:
    difference = (a - b).to_numpy()
    n = len(difference)
    mean = float(difference.mean())
    lag = int(np.floor(1.5 * n ** (1 / 3)))
    centred = difference - mean
    variance = np.mean(centred ** 2)
    for k in range(1, lag + 1):
        covariance = np.mean(centred[k:] * centred[:-k])
        variance += 2 * (1 - k / (lag + 1)) * covariance
    statistic = mean / np.sqrt(variance / n)
    return mean, statistic, float(stats.norm.cdf(statistic)), n


def main() -> None:
    frames = {}
    for name, path in SOURCES.items():
        frame = pd.read_csv(path, parse_dates=["settlement_date"])
        frame = frame[frame["settlement_date"].between(START, END)].sort_values(KEY).reset_index(drop=True)
        frame["crps"] = crps_rows(frame)
        frames[name] = frame

    reference = frames["qrf"]
    for name, frame in frames.items():
        if not frame[KEY].equals(reference[KEY]):
            raise AssertionError(f"delivery grid differs for {name}")
        if not np.allclose(frame["actual_price"], reference["actual_price"]):
            raise AssertionError(f"actual prices differ for {name}")

    daily = {
        name: frame.groupby("settlement_date")["crps"].sum()
        for name, frame in frames.items()
    }
    rows = []
    for model_a, model_b in COMPARISONS:
        difference, statistic, p_value, n_days = dm_daily(daily[model_a], daily[model_b])
        rows.append({
            "period": "2025", "A": model_a, "B": model_b, "n_days": n_days,
            "mean_daily_loss_diff": difference, "DM_stat": statistic,
            "p_one_sided_A_better": p_value,
        })
        print(f"{model_a} vs {model_b}: DM={statistic:.3f}, p={p_value:.5f}")

    output = ROOT / "06_outputs/final_results_tables/constituent_dm_tests.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"written -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
