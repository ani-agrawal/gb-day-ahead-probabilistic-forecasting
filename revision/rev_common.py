"""Shared paths, loader and scoring functions for the revision analyses.

Every location is relative to the repository root and follows the layout written by
the constituent pipeline (run_pipeline.py) and pipeline/prepare_publication_snapshot.py.
The loader assembles the saved 2025 forecasts of every model in Table 5, plus the two
conformal variants of Table 10, into one frame aligned on delivery date and hour; the
scoring functions implement Equations (1), (2) and (4) of the paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "modelling" / "input_snapshot"
FEATURES = ROOT / "04_data" / "metadata" / "lean_feature_list_hourly.csv"
RUNS = ROOT / "06_outputs"
RQ1 = RUNS / "RQ1" / "model_runs" / "hourly_model_recal_epex"
RQ1_ABLATION = RUNS / "RQ1" / "model_runs" / "ablation_testperiod"
LSTM_RUN = RUNS / "RQ2" / "model_runs" / "seq2seq_final_b16es14"
LSTM_DEV = RUNS / "RQ2" / "model_runs" / "rnn_dev" / "seq2seq_gate_v2"
LSTM_ABLATION = RUNS / "RQ2" / "model_runs" / "ablation_testperiod_twostage"
MODELLING = ROOT / "modelling"
LEAR_QRA_OUT = MODELLING / "lear_qra" / "outputs"
QRA_OUT = MODELLING / "qra" / "outputs"
OUT = ROOT / "revision" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

KEY = ["settlement_date", "hour"]
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
QCOLS = [f"q{int(round(t * 100)):02d}" for t in LEVELS]
THRESHOLDS = (150, 200)            # GBP/MWh, fixed before the test year (Section 4.2)
# Price regimes of the conditional coverage columns, including negative prices
REGIMES = {"< 0": lambda y: y < 0, "> 150": lambda y: y > 150, "> 200": lambda y: y > 200}

# Table 5 order, then the Table 10 conformal variants
MODELS = ["Historical benchmark", "Linear QR", "LEAR-QRA", "QRF", "Encoder-Decoder LSTM",
          "Two-model QRA", "Three-model QRA", "Static CQR", "Rolling CQR"]
FULL_GRID = MODELS[:7]             # models with all eleven quantiles


def _slice(d):
    d = d.copy()
    d["settlement_date"] = pd.to_datetime(d["settlement_date"])
    d = d[(d.settlement_date >= TEST_START) & (d.settlement_date <= TEST_END)]
    return d.sort_values(KEY).reset_index(drop=True)


def load():
    """Frame with actual_price and, per model, '<model>|qXX' columns.
    The conformal variants carry only their 90% bounds, as q05 and q95."""
    lstm = _slice(pd.read_csv(LSTM_RUN / "conformal_ENSEMBLE" / "predictions_2025_primary.csv"))
    out = lstm[KEY + ["actual_price"]].copy()

    def add(name, df, cols):
        d = _slice(df)
        m = out[KEY].merge(d, on=KEY, how="left")
        assert len(m) == len(out), name
        for q, src in cols.items():
            out[f"{name}|{q}"] = m[src].to_numpy()

    same = {q: q for q in QCOLS}
    add("Historical benchmark", pd.read_csv(RQ1 / "gaussian_baseline" / "predictions.csv"), same)
    add("Linear QR", pd.read_csv(SNAP / "linear_qr_test_predictions.csv"), same)
    add("LEAR-QRA", pd.read_csv(LEAR_QRA_OUT / "lear_qra_quantiles_principal.csv"), same)
    add("QRF", pd.read_csv(SNAP / "qrf_test_predictions.csv"), same)
    add("Encoder-Decoder LSTM", lstm, same)
    qra = pd.read_csv(QRA_OUT / "three_way_qw_predictions.csv")
    add("Two-model QRA", qra, {q: f"two_way_{q}" for q in QCOLS})
    add("Three-model QRA", qra, {q: f"three_way_qw_{q}" for q in QCOLS})
    add("Static CQR", lstm, {"q05": "q05_static", "q95": "q95_static"})
    add("Rolling CQR", lstm, {"q05": "q05_roll", "q95": "q95_roll"})

    assert len(out) == 8759, len(out)
    assert out.notna().all().all(), out.columns[out.isna().any()].tolist()
    return out


def model_table():
    t = pd.read_parquet(SNAP / "model_table_hourly_epex.parquet")
    t["settlement_date"] = pd.to_datetime(t["settlement_date"])
    return t.sort_values(KEY).reset_index(drop=True)


def lean_features():
    return pd.read_csv(FEATURES)["feature"].tolist()


def quantiles(f, name):
    return f[[f"{name}|{q}" for q in QCOLS]].to_numpy()


def pinball(y, q, tau):
    """Equation (1), elementwise."""
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


def crps11(y, Q):
    """Equation (2): trapezoidal CRPS over the eleven-level grid, per hour."""
    L = pinball(y[:, None], Q, LEVELS[None, :])
    return 2 * (((L[:, :-1] + L[:, 1:]) / 2) * np.diff(LEVELS)[None, :]).sum(axis=1)


def covered90(f, name):
    y = f.actual_price.to_numpy()
    return (y >= f[f"{name}|q05"].to_numpy()) & (y <= f[f"{name}|q95"].to_numpy())


def winkler90(y, lo, hi, alpha=0.10):
    """Equation (4)."""
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
