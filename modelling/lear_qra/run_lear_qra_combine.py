"""LEAR-QRA: combine the multi-window LEAR point forecasts into LEAR quantiles.

This is the literature-standard probabilistic LEAR benchmark (Marcjasz et al. 2023;
Lipiecki et al. 2024): at each quantile level tau, a quantile regression uses the
window-specific LEAR point forecasts as predictors,

    q_tau = b0 + sum_w b_{w,tau} * P^LEAR_w .

Two specifications:
  * principal   : windows 364, 546, 728 (numerically stable; 273 excluded for
                  demonstrated non-convergence instability, not forecast score)
  * sensitivity : all four windows including 273

Protocol mirrors the study's other QRA stages: weekly refits, D-2 information
cutoff, expanding calibration pool (from the Jan 2024 point-forecast backcast).
Quantiles are produced from 1 Jul 2024 (so the downstream three-way QRA has its
development pool) through 31 Dec 2025, then monotonically rearranged. The window
point forecasts are collinear through 2024 (the longer windows coincide until
enough history accrues) and become distinct in 2025; QRA fitted values are well
defined throughout, only the individual weights are non-identifiable early on.

Reads outputs/lear_qra_point_w{273,364,546,728}.csv. Writes under lear_qra/outputs/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import QuantileRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PT = ROOT / "outputs"
SNAP = ROOT / "input_snapshot"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
QCOLS = [f"q{int(t*100):02d}" for t in LEVELS]
LEV = np.array(LEVELS)
KEY = ["settlement_date", "hour"]
WINDOWS = {"principal": [364, 546, 728], "sensitivity": [273, 364, 546, 728]}
BLOCK_DAYS, LAG_DAYS = 7, 2
OUTPUT_START = pd.Timestamp("2024-07-01")     # LEAR-QRA quantiles from here (dev pool for three-way)
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")


def crps11(y, q):
    pin = np.where(y[:, None] - q >= 0, LEV * (y[:, None] - q), (LEV - 1) * (y[:, None] - q))
    return 2 * np.trapezoid(pin, x=LEV, axis=1)


def score(y, q, name):
    pin = np.where(y[:, None] - q >= 0, LEV * (y[:, None] - q), (LEV - 1) * (y[:, None] - q))
    lo, hi = q[:, 0], q[:, -1]
    inside = (y >= lo) & (y <= hi)
    neg, h150, h200 = y < 0, y > 150, y > 200
    return {"model": name, "crps_11": float(np.mean(crps11(y, q))),
            "mean_pinball_11q": float(pin.mean()), "mae_q50": float(np.mean(np.abs(y - q[:, 5]))),
            "cov90": float(inside.mean()), "width90": float(np.mean(hi - lo)),
            "winkler90": float(np.mean((hi - lo) + 20.0 * (lo - y) * (y < lo) + 20.0 * (y - hi) * (y > hi))),
            "cov_neg": float(inside[neg].mean()) if neg.any() else np.nan,
            "cov_gt150": float(inside[h150].mean()) if h150.any() else np.nan,
            "cov_gt200": float(inside[h200].mean()) if h200.any() else np.nan}


def load_merged():
    frames = []
    for w in [273, 364, 546, 728]:
        d = pd.read_csv(PT / f"lear_qra_point_w{w}.csv", parse_dates=["settlement_date"])
        frames.append(d.set_index(KEY)[f"lear_w{w}"])
    actual = pd.read_csv(PT / "lear_qra_point_w728.csv", parse_dates=["settlement_date"]).set_index(KEY)["actual_price"]
    m = pd.concat(frames + [actual], axis=1).reset_index().sort_values(KEY).reset_index(drop=True)
    assert m.notna().all().all(), "missing values in merged window forecasts"
    return m


def refit_origins(merged):
    """Weekly dev origins Jul-Dec 2024, then the exact QRF 2025 origins (Jan 1/8/15...),
    matching the study's rolling protocol; the continuous-7-day sequence would offset 2025."""
    days = pd.DatetimeIndex(sorted(merged.settlement_date.unique()))
    dev = list(days[(days >= OUTPUT_START) & (days <= pd.Timestamp("2024-12-31"))][::BLOCK_DAYS])
    qrf = pd.read_csv(SNAP / "qrf_refit_log.csv")
    qrf["test_day"] = pd.to_datetime(qrf["test_day"])
    test = sorted(qrf.loc[(qrf.test_day >= TEST_START) & (qrf.test_day <= TEST_END), "test_day"].tolist())
    return dev + test


def combine(merged, windows):
    """Weekly-refit causal QRA across the given window forecasts; returns quantiles for OUTPUT_START..TEST_END."""
    feats = [f"lear_w{w}" for w in windows]
    target = merged[(merged.settlement_date >= OUTPUT_START) & (merged.settlement_date <= TEST_END)].copy()
    origins = refit_origins(merged)
    oa = pd.DatetimeIndex(origins)
    raw = np.full((len(target), len(LEVELS)), np.nan)
    coef_rows = []
    for i, start in enumerate(origins):
        nxt = oa[i + 1] if i + 1 < len(oa) else TEST_END + pd.Timedelta(days=1)
        cutoff = start - pd.Timedelta(days=LAG_DAYS)
        tr = merged[merged.settlement_date <= cutoff]
        mask = ((target.settlement_date >= start) & (target.settlement_date < nxt)).to_numpy()
        if not mask.any():
            continue
        Xtr, ytr = tr[feats].to_numpy(), tr["actual_price"].to_numpy()
        Xte = target.loc[mask, feats].to_numpy()
        for j, tau in enumerate(LEVELS):
            mdl = QuantileRegressor(quantile=tau, alpha=0.0, fit_intercept=True, solver="highs").fit(Xtr, ytr)
            raw[mask, j] = mdl.predict(Xte)
            coef_rows.append({"block_start": str(start.date()), "quantile": tau,
                              "intercept": float(mdl.intercept_),
                              **{f: float(c) for f, c in zip(feats, mdl.coef_)}, "n_train": int(len(tr))})
    assert not np.isnan(raw).any()
    crossing = float(np.any(np.diff(raw, axis=1) < 0, axis=1).mean())
    q = np.sort(raw, axis=1)
    res = target[KEY + ["actual_price"]].copy()
    for j, c in enumerate(QCOLS):
        res[c] = q[:, j]
    return res, crossing, pd.DataFrame(coef_rows)


def main():
    merged = load_merged()
    rows, all_metrics = {}, []
    for name, windows in WINDOWS.items():
        res, crossing, coefs = combine(merged, windows)
        res.to_csv(OUT / f"lear_qra_quantiles_{name}.csv", index=False)
        coefs.to_csv(OUT / f"lear_qra_coefficients_{name}.csv", index=False)
        test = res[(res.settlement_date >= TEST_START) & (res.settlement_date <= TEST_END)]
        assert len(test) == 8759, (name, len(test))
        row = score(test.actual_price.to_numpy(), test[QCOLS].to_numpy(), f"LEAR-QRA-{name}")
        row["windows"] = "+".join(map(str, windows)); row["raw_crossing_rate"] = crossing
        all_metrics.append(row); rows[name] = res
        print(f"{name} ({'+'.join(map(str,windows))}): CRPS {row['crps_11']:.3f}, MAE {row['mae_q50']:.3f}, "
              f"cov90 {row['cov90']:.3f}, width {row['width90']:.1f}, winkler {row['winkler90']:.1f}, "
              f"cov>200 {row['cov_gt200']:.3f}, crossing {crossing:.4f}")
    # reference: single-window LEAR-QR
    try:
        prev = pd.read_csv(OUT / "lear_qr_metrics.csv").iloc[0]
        print(f"\n(reference: single-window LEAR-QR CRPS {prev['crps_11']:.3f}, cov>200 {prev['cov_gt200']:.3f})")
    except Exception:
        pass
    pd.DataFrame(all_metrics).to_csv(OUT / "lear_qra_metrics.csv", index=False)
    (OUT / "lear_qra_combine_manifest.json").write_text(json.dumps({
        "specs": WINDOWS, "excluded_273_reason": "numerical instability (forecast-sensitivity test), not score",
        "protocol": {"block_days": BLOCK_DAYS, "cutoff": f"D-{LAG_DAYS}",
                     "output_span": "2024-07-01..2025-12-31", "pool": "expanding from Jan-2024 backcast"},
        "sklearn": sklearn.__version__,
        "point_inputs_sha256": {f"w{w}": hashlib.sha256((PT / f"lear_qra_point_w{w}.csv").read_bytes()).hexdigest()
                                for w in [273, 364, 546, 728]}}, indent=2) + "\n")
    # audit: principal quantiles cover the three-way's needed span with no gaps
    p = rows["principal"]
    span_ok = (p.settlement_date.min() <= OUTPUT_START and p.settlement_date.max() == TEST_END
               and not p.duplicated(KEY).any() and np.isfinite(p[QCOLS].to_numpy()).all())
    print(f"\nAUDIT principal: span {p.settlement_date.min().date()}..{p.settlement_date.max().date()}, "
          f"rows {len(p)}, no-dups+finite {'ok' if span_ok else 'FAIL'}")
    print("LEAR_QRA_COMBINE_DONE")


if __name__ == "__main__":
    main()
