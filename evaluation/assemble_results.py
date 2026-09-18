"""Assemble the three-seed Encoder--Decoder LSTM and evaluate paper models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "06_outputs" / "RQ2" / "model_runs"
RQ1_ROOT = ROOT / "06_outputs" / "RQ1" / "model_runs" / "hourly_model_recal_epex"
LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
QCOLS = [f"q{int(q * 100):02d}" for q in LEVELS]
KEY = ["settlement_date", "hour"]
SEEDS = [42, 7, 101]
PERIODS = {
    "2025_primary": ("2025-01-01", "2025-12-31"),
}


def merge_seed(run_dir: Path, seed: int) -> pd.DataFrame:
    p5 = pd.read_csv(run_dir / f"5q_seed{seed}" / "predictions.csv")
    p6 = pd.read_csv(run_dir / f"fill6_seed{seed}" / "predictions.csv")
    frame = p5.merge(p6[KEY + ["q20", "q30", "q40", "q60", "q70", "q80"]],
                     on=KEY, validate="one_to_one")
    if len(frame) != len(p5) or len(frame) != len(p6):
        raise AssertionError(f"seed {seed}: 5q/fill6 grids differ")
    frame[QCOLS] = np.sort(frame[QCOLS].to_numpy(), axis=1)
    return frame.sort_values(KEY).reset_index(drop=True)


def build_ensemble(run_dir: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    frames = {seed: merge_seed(run_dir, seed) for seed in SEEDS}
    base = frames[SEEDS[0]]
    for seed, frame in frames.items():
        if not frame[KEY].equals(base[KEY]):
            raise AssertionError(f"seed {seed}: delivery grid differs")
        if not np.allclose(frame["actual_price"], base["actual_price"]):
            raise AssertionError(f"seed {seed}: actual prices differ")
    ensemble = base[KEY + ["actual_price"]].copy()
    ensemble[QCOLS] = np.sort(
        np.mean([frame[QCOLS].to_numpy() for frame in frames.values()], axis=0), axis=1
    )
    out = run_dir / "ENSEMBLE_11q"
    out.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(out / "predictions.csv", index=False)
    return ensemble, frames


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    penalty = np.where(y < lo, (2 / alpha) * (lo - y),
                       np.where(y > hi, (2 / alpha) * (y - hi), 0.0))
    return float(np.mean(width + penalty))


def evaluate(name: str, period: str, frame: pd.DataFrame) -> dict[str, object]:
    y = frame["actual_price"].to_numpy()
    q = frame[QCOLS].to_numpy()
    err = y[:, None] - q
    pinball = np.where(err >= 0, LEVELS * err, (LEVELS - 1) * err)
    in90 = (y >= frame["q05"]) & (y <= frame["q95"])
    in80 = (y >= frame["q10"]) & (y <= frame["q90"])
    neg, spike = y < 0, y > 150
    return {
        "model": name,
        "period": period,
        "n_hours": len(frame),
        "CRPS_11": round(float((2 * np.trapezoid(pinball, x=LEVELS, axis=1)).mean()), 3),
        "MAE_q50": round(float(np.abs(y - frame["q50"]).mean()), 3),
        "cov_90": round(float(in90.mean()), 3),
        "width_90": round(float((frame["q95"] - frame["q05"]).mean()), 1),
        "winkler_90": round(winkler(y, frame["q05"].to_numpy(), frame["q95"].to_numpy(), .10), 1),
        "cov_80": round(float(in80.mean()), 3),
        "winkler_80": round(winkler(y, frame["q10"].to_numpy(), frame["q90"].to_numpy(), .20), 1),
        "cov90_neg": round(float(in90[neg].mean()) if neg.any() else np.nan, 3),
        "n_neg": int(neg.sum()),
        "cov90_gt150": round(float(in90[spike].mean()) if spike.any() else np.nan, 3),
        "n_gt150": int(spike.sum()),
    }


def load_rq1(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in QCOLS if column not in frame]
    if missing:
        raise ValueError(f"{path}: missing shared quantiles {missing}")
    return frame[KEY + ["actual_price"] + QCOLS].copy()


def make_table(run_dir: Path, system_name: str, ensemble: pd.DataFrame,
               seeds: dict[int, pd.DataFrame], other: tuple[str, pd.DataFrame] | None) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {system_name: ensemble}
    prefix = "lstm_seed"
    frames.update({f"{prefix}{seed}": frame for seed, frame in seeds.items()})
    if other is not None:
        frames[other[0]] = other[1]
    frames.update({
        "historical_baseline": load_rq1(RQ1_ROOT / "gaussian_baseline" / "predictions.csv"),
        "linear_qr_perhour": load_rq1(RQ1_ROOT / "linear_qr" / "predictions.csv"),
        "qrf": load_rq1(RQ1_ROOT / "quantile_forest" / "predictions.csv"),
    })
    reference = ensemble.sort_values(KEY).reset_index(drop=True)
    rows = []
    for name, frame in frames.items():
        frame = frame.copy()
        frame["settlement_date"] = pd.to_datetime(frame["settlement_date"])
        frame = frame.sort_values(KEY).reset_index(drop=True)
        if len(frame) != len(reference):
            raise AssertionError(f"{name}: row count differs from the common evaluation grid")
        if not np.allclose(frame["actual_price"], reference["actual_price"]):
            raise AssertionError(f"{name}: actual prices differ from the common evaluation grid")
        for period, (start, end) in PERIODS.items():
            selected = frame[(frame["settlement_date"] >= start) & (frame["settlement_date"] <= end)]
            rows.append(evaluate(name, period, selected))
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "ENSEMBLE_11q" / "final_framework_tables.csv", index=False)
    return table


def main() -> None:
    two_dir = MODEL_ROOT / "seq2seq_final_b16es14"
    two, two_seeds = build_ensemble(two_dir)
    two_table = make_table(two_dir, "encoder_decoder_lstm", two, two_seeds, None)
    print("\nHeadline results")
    print(two_table[two_table.model.isin(["encoder_decoder_lstm", "qrf", "linear_qr_perhour",
                                         "historical_baseline"])].to_string(index=False))


if __name__ == "__main__":
    main()
