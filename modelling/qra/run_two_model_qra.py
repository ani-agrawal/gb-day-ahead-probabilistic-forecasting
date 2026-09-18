"""Causal 11-quantile weekly-updated QRA combination of the final LSTM and QRF, 2025.

Quantile-wise QRA (method selected on the 2024-H2 development comparison;
classical median-based QRA was the rejected arm). The weekly update cadence was
declared ex ante on design-coherence grounds (the constituent models refit
weekly) BEFORE any 2025 cadence comparison was run; no alternative cadence is
evaluated here.

For each of the 11 quantile levels tau, the meta-model is

    Q_tau(y) = intercept + beta_LSTM * LSTM_tau + beta_QRF * QRF_tau,

fitted by pinball loss, pooled across delivery hours. Coefficients are
re-estimated before each 7-day block of 2025 (blocks aligned to 1 Jan, matching
the constituents' refit grid) using every matched out-of-sample forecast-actual
pair from 1 Jul 2024 through D-2 relative to the block start. The constituent
forecasts are frozen; nothing is retrained here.

The development-period 11-quantile LSTM ensemble is assembled exactly as the
production ENSEMBLE_11q: per seed, the five principal quantiles (5q run) and six
interior quantiles (fill6 run) are merged; seeds are averaged per level; each
row is then monotonically rearranged. Raw crossing rates of the QRA output are
audited before the same rearrangement.

Outputs -> 06_outputs/RQ2/model_runs/qra_lstm_qrf/2025_11q_weekly/
           predictions.csv, aggregate_metrics.csv, monthly_metrics.csv,
           coefficients.csv, dm_tests.csv, dev_lstm_ensemble_11q.csv,
           run_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.linear_model import QuantileRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_outputs" / "RQ2" / "model_runs" / "qra_lstm_qrf" / "2025_11q_weekly"

DEV_QRF = (ROOT / "06_outputs" / "RQ1" / "model_runs" / "hourly_model_recal_epex"
           / "tuning" / "qrf_dev2024H2_predictions.csv")
DEV_LSTM_BASE = ROOT / "06_outputs" / "RQ2" / "model_runs" / "rnn_dev" / "seq2seq_gate_v2"
SEEDS = (42, 7, 101)
DEV_5Q = {s: DEV_LSTM_BASE / f"encdec_u64_b16es14_seed{s}" / "predictions.csv" for s in SEEDS}
DEV_FILL6 = {s: DEV_LSTM_BASE / f"encdec_u64_b16es14_fill6_seed{s}" / "predictions.csv"
             for s in SEEDS}
TEST_QRF = (ROOT / "06_outputs" / "RQ1" / "model_runs" / "hourly_model_recal_epex"
            / "quantile_forest" / "predictions.csv")
TEST_LSTM = (ROOT / "06_outputs" / "RQ2" / "model_runs" / "seq2seq_final_b16es14"
             / "ENSEMBLE_11q" / "predictions.csv")

KEY = ["settlement_date", "hour"]
LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
QCOLS = [f"q{int(t * 100):02d}" for t in LEVELS]
COLS_5Q = ["q05", "q10", "q50", "q90", "q95"]
COLS_F6 = ["q20", "q30", "q40", "q60", "q70", "q80"]
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")
BLOCK_DAYS = 7
LAG_DAYS = 2  # D-2 information cutoff, as everywhere in the pipeline


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction(path: Path, cols: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
    required = KEY + ["actual_price", *cols]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frame = frame[required].sort_values(KEY).reset_index(drop=True)
    if frame.duplicated(KEY).any() or frame.isna().any().any():
        raise ValueError(f"{path}: duplicate keys or missing values")
    return frame


def build_dev_lstm_ensemble() -> pd.DataFrame:
    """Per seed merge 5q + fill6 into 11 levels; average seeds; rearrange rows."""
    per_seed = []
    reference = None
    for seed in SEEDS:
        principal = read_prediction(DEV_5Q[seed], COLS_5Q)
        interior = read_prediction(DEV_FILL6[seed], COLS_F6)
        merged = principal.merge(interior, on=KEY, suffixes=("", "_f6"),
                                 validate="one_to_one")
        if not np.array_equal(merged["actual_price"], merged["actual_price_f6"]):
            raise ValueError(f"seed {seed}: actual prices differ between 5q and fill6")
        if len(merged) != len(principal):
            raise ValueError(f"seed {seed}: 5q/fill6 grids differ")
        if reference is None:
            reference = merged[KEY + ["actual_price"]].copy()
        else:
            if not (np.array_equal(reference["settlement_date"].to_numpy(),
                                   merged["settlement_date"].to_numpy())
                    and np.array_equal(reference["hour"].to_numpy(),
                                       merged["hour"].to_numpy())):
                raise ValueError(f"seed {seed}: date-hour grid differs across seeds")
            if not np.array_equal(reference["actual_price"], merged["actual_price"]):
                raise ValueError(f"seed {seed}: actuals differ across seeds")
        per_seed.append(merged[QCOLS].to_numpy())
    ensemble = reference.copy()
    ensemble[QCOLS] = np.sort(np.mean(per_seed, axis=0), axis=1)
    if len(ensemble) != 4416:
        raise ValueError(f"dev ensemble: expected 4,416 hours, found {len(ensemble):,}")
    return ensemble


def align_pair(lstm: pd.DataFrame, qrf: pd.DataFrame, label: str) -> pd.DataFrame:
    left = lstm.rename(columns={"actual_price": "actual_lstm",
                                **{c: f"lstm_{c}" for c in QCOLS}})
    right = qrf.rename(columns={"actual_price": "actual_qrf",
                                **{c: f"qrf_{c}" for c in QCOLS}})
    merged = left.merge(right, on=KEY, validate="one_to_one")
    if not np.array_equal(merged["actual_lstm"], merged["actual_qrf"]):
        raise ValueError(f"{label}: actual prices differ between LSTM and QRF")
    merged["actual_price"] = merged["actual_lstm"]
    keep = KEY + ["actual_price"] + [f"lstm_{c}" for c in QCOLS] + [f"qrf_{c}" for c in QCOLS]
    merged = merged[keep].sort_values(KEY).reset_index(drop=True)
    if merged.isna().any().any():
        raise ValueError(f"{label}: missing aligned values")
    return merged


def crps11(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    lev = np.array(LEVELS)
    pin = np.where(y[:, None] - q >= 0, lev * (y[:, None] - q), (lev - 1) * (y[:, None] - q))
    return 2 * np.trapezoid(pin, x=lev, axis=1)


def score(frame: pd.DataFrame, pred: np.ndarray, arm: str, period: str,
          raw_crossing: float) -> dict[str, object]:
    y = frame["actual_price"].to_numpy()
    lev = np.array(LEVELS)
    pin = np.where(y[:, None] - pred >= 0, lev * (y[:, None] - pred),
                   (lev - 1) * (y[:, None] - pred))
    lower, upper = pred[:, 0], pred[:, -1]
    inside = (y >= lower) & (y <= upper)
    negative, spike = y < 0, y > 200
    return {
        "arm": arm, "period": period, "n_hours": len(y),
        "crps_11": float(np.mean(2 * np.trapezoid(pin, x=lev, axis=1))),
        "mean_pinball_11q": float(pin.mean()),
        "pinball_q05": float(pin[:, 0].mean()), "pinball_q50": float(pin[:, 5].mean()),
        "pinball_q95": float(pin[:, -1].mean()),
        "mae_q50": float(np.mean(np.abs(y - pred[:, 5]))),
        "cov90": float(inside.mean()), "width90": float(np.mean(upper - lower)),
        "winkler90": float(np.mean((upper - lower)
                                   + 20.0 * (lower - y) * (y < lower)
                                   + 20.0 * (y - upper) * (y > upper))),
        "cov90_negative": float(inside[negative].mean()) if negative.any() else np.nan,
        "n_negative": int(negative.sum()),
        "cov90_spike200": float(inside[spike].mean()) if spike.any() else np.nan,
        "n_spike200": int(spike.sum()),
        "raw_crossing_rate": raw_crossing,
    }


def dm_pvalue(daily_a: pd.Series, daily_b: pd.Series) -> float:
    """One-sided DM on daily CRPS sums, HAC variance; p small => A better."""
    d = (daily_a - daily_b).to_numpy()
    t_len, mean = len(d), float(np.mean(daily_a - daily_b))
    lag = int(np.floor(1.5 * t_len ** (1 / 3)))
    centred = d - mean
    var = np.mean(centred ** 2) + sum(
        2 * (1 - k / (lag + 1)) * np.mean(centred[k:] * centred[:-k])
        for k in range(1, lag + 1))
    return float(stats.norm.cdf(mean / np.sqrt(var / t_len)))


def main() -> None:
    dev = align_pair(build_dev_lstm_ensemble(), read_prediction(DEV_QRF, QCOLS),
                     "development")
    test_lstm = read_prediction(TEST_LSTM, QCOLS)
    test_lstm = test_lstm[test_lstm["settlement_date"].between(TEST_START, TEST_END)]
    test_qrf = read_prediction(TEST_QRF, QCOLS)
    test_qrf = test_qrf[test_qrf["settlement_date"].between(TEST_START, TEST_END)]
    test = align_pair(test_lstm, test_qrf, "2025")
    if len(test) != 8759:
        raise ValueError(f"2025: expected 8,759 hours, found {len(test):,}")
    all_oos = pd.concat([dev, test], ignore_index=True).sort_values(KEY).reset_index(drop=True)

    qra_raw = np.full((len(test), len(QCOLS)), np.nan)
    coefficient_rows = []
    block_starts = pd.date_range(TEST_START, TEST_END, freq=f"{BLOCK_DAYS}D")
    for start in block_starts:
        end = min(start + pd.Timedelta(days=BLOCK_DAYS - 1), TEST_END)
        cutoff = start - pd.Timedelta(days=LAG_DAYS)
        training = all_oos[all_oos["settlement_date"] <= cutoff]
        mask = test["settlement_date"].between(start, end).to_numpy()
        block = test.loc[mask]
        for j, (tau, col) in enumerate(zip(LEVELS, QCOLS)):
            features = [f"lstm_{col}", f"qrf_{col}"]
            model = QuantileRegressor(quantile=tau, alpha=0.0, fit_intercept=True,
                                      solver="highs")
            model.fit(training[features].to_numpy(), training["actual_price"].to_numpy())
            qra_raw[mask, j] = model.predict(block[features].to_numpy())
            coefficient_rows.append({
                "block_start": start, "block_end": end,
                "train_end": training["settlement_date"].max(),
                "n_train_hours": len(training), "quantile": tau,
                "intercept": float(model.intercept_),
                "beta_lstm": float(model.coef_[0]),
                "beta_qrf": float(model.coef_[1]),
            })
        print(f"block {start.date()} fitted (train through "
              f"{training['settlement_date'].max().date()})", flush=True)
    if np.isnan(qra_raw).any():
        raise ValueError("Unfitted 2025 rows remain")

    raw_crossing = float(np.any(np.diff(qra_raw, axis=1) < 0, axis=1).mean())
    predictions = {
        "lstm_ensemble": (test[[f"lstm_{c}" for c in QCOLS]].to_numpy(), 0.0),
        "qrf": (test[[f"qrf_{c}" for c in QCOLS]].to_numpy(), 0.0),
        "qra_weekly": (np.sort(qra_raw, axis=1), raw_crossing),
    }
    raw_sorted_delta = 100 * (np.mean(crps11(test["actual_price"].to_numpy(), qra_raw))
                              - np.mean(crps11(test["actual_price"].to_numpy(),
                                               np.sort(qra_raw, axis=1)))) \
        / np.mean(crps11(test["actual_price"].to_numpy(), np.sort(qra_raw, axis=1)))

    metrics_rows, monthly_rows = [], []
    output = test[KEY + ["actual_price"]].copy()
    daily_crps = {}
    for arm, (pred, crossing) in predictions.items():
        metrics_rows.append(score(test, pred, arm, "2025", crossing))
        for col_index, col in enumerate(QCOLS):
            output[f"{arm}_{col}"] = pred[:, col_index]
    # Raw pre-rearrangement QRA quantiles, so the crossing audit is reproducible.
    for col_index, col in enumerate(QCOLS):
        output[f"qra_weekly_raw_{col}"] = qra_raw[:, col_index]
    for arm, (pred, crossing) in predictions.items():
        holder = test[KEY + ["actual_price"]].copy()
        holder["crps"] = crps11(test["actual_price"].to_numpy(), pred)
        daily_crps[arm] = holder.groupby("settlement_date")["crps"].sum()
        for month in range(1, 13):
            mask = (test["settlement_date"].dt.month == month).to_numpy()
            monthly_rows.append(score(test.loc[mask], pred[mask], arm,
                                      f"2025-{month:02d}", crossing))

    dm_rows = [{"model_a": "qra_weekly", "model_b": rival, "period": "2025",
                "p_value": dm_pvalue(daily_crps["qra_weekly"], daily_crps[rival])}
               for rival in ("lstm_ensemble", "qrf")]

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows).sort_values("crps_11")
    metrics.to_csv(OUT / "aggregate_metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(OUT / "monthly_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(OUT / "dm_tests.csv", index=False)
    output.to_csv(OUT / "predictions.csv", index=False)
    build_dev_lstm_ensemble().to_csv(OUT / "dev_lstm_ensemble_11q.csv", index=False)

    manifest = {
        "purpose": "Causal 2025 11-quantile quantile-wise QRA, weekly-updated "
                   "with an expanding training pool (recursive scheme)",
        "created_by": Path(__file__).name,
        "sklearn_version": sklearn.__version__,
        "levels": LEVELS,
        "cadence": {"block_days": BLOCK_DAYS, "information_cutoff": f"D-{LAG_DAYS}",
                    "training_pool": "expanding from 2024-07-01 (recursive), not a "
                                     "fixed-length trailing window",
                    "decision": "weekly declared ex ante on design-coherence grounds "
                                "(constituents refit weekly); no 2025 cadence "
                                "comparison was run"},
        "qra": {"form": "quantile-wise linear QRA, pooled across delivery hours",
                "estimator": "sklearn.linear_model.QuantileRegressor",
                "alpha": 0.0, "fit_intercept": True, "solver": "highs",
                "crossing_treatment": "row-wise monotone rearrangement after raw "
                                      "crossing audit"},
        "raw_vs_sorted_crps_delta_pct": round(float(raw_sorted_delta), 4),
        "inputs": {
            "dev_qrf": {"path": str(DEV_QRF.relative_to(ROOT)), "sha256": sha256(DEV_QRF)},
            "dev_lstm_5q": [{"seed": s, "path": str(p.relative_to(ROOT)),
                             "sha256": sha256(p)} for s, p in DEV_5Q.items()],
            "dev_lstm_fill6": [{"seed": s, "path": str(p.relative_to(ROOT)),
                                "sha256": sha256(p)} for s, p in DEV_FILL6.items()],
            "test_qrf": {"path": str(TEST_QRF.relative_to(ROOT)), "sha256": sha256(TEST_QRF)},
            "test_lstm": {"path": str(TEST_LSTM.relative_to(ROOT)), "sha256": sha256(TEST_LSTM)},
        },
        "test_boundary": "2025 rows enter QRA training only once their delivery day "
                         "is <= D-2 of the block being fitted",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    shown = ["arm", "crps_11", "mean_pinball_11q", "pinball_q50", "pinball_q95",
             "cov90", "width90", "winkler90", "cov90_spike200", "raw_crossing_rate"]
    print("\n=== Causal 2025 11-quantile weekly QRA ===")
    print(metrics[shown].round(4).to_string(index=False))
    print("\n=== DM (daily CRPS sums, one-sided, p small => QRA better) ===")
    print(pd.DataFrame(dm_rows).round(5).to_string(index=False))
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
