"""LEAR point-forecast backtest for the Applied Energy paper (isolated under Publication/).

A LEAR model following Lago et al. (2021): 24 separate per-hour LASSO regressions
sharing one day-indexed feature matrix. The estimator, scaling and the price-lag
and weekday-dummy construction replicate the epftoolbox implementation exactly
(verified against models/_lear.py):
  * LassoLarsIC(criterion="aic", max_iter=2500) selects lambda, then a
    coordinate-descent Lasso(alpha=lambda, max_iter=2500) is refit and used for
    prediction;
  * the invariant scaler applies per-column median/MAD first, then arcsinh;
  * weekday dummies (the last 7 features) are left unscaled.
The four hourly forecast channels and the six gate-aligned daily commodity
features (gas/carbon latest/lag1/lag7, each <= D-2) are study-specific
adaptations to our information set and boundary; epftoolbox itself supports an
arbitrary number of exogenous inputs but has no notion of the daily-commodity
gate alignment or the D-2 (rather than D-1) training endpoint used here.

Feature set (397, day-indexed):
  96  price lags   : the 24-hour profiles of delivery days D-1, D-2, D-3, D-7
  288 forecasts    : demand, transmission wind, embedded wind, embedded solar,
                     each as the 24-hour profiles at D, D-1, D-7 (all gate-legal)
  6   commodities  : gas and carbon, latest/lag1/lag7 eligible closes (<= D-2)
  7   weekday dummies

Protocol: 728-day trailing window ending D-2 (expanding until 728 present),
weekly refits reusing the QRF origins for 2025, D-2 information cutoff. The
coefficients are frozen within each weekly block but every day is forecast from
its own feature row, rebuilt from that day's gate-legal information.

Outputs (all under Publication/modelling/): raw LEAR point forecasts, a per-refit
log with per-hour penalties and non-zero counts, per-forecast-day commodity
source dates, a post-run audit and a run manifest (config/script/environment
fingerprint). Resumable via idempotent per-refit files. Deterministic given a
pinned environment (no seeds).

Usage:
  python run_lear.py --smoke     # one refit, prints the checklist, writes nothing final
  python run_lear.py             # full dev+2025 backtest, resumable
  python run_lear.py --audit     # re-run the audit over existing outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso, LassoLarsIC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # Publication/modelling
SNAP = ROOT / "input_snapshot"
OUT = ROOT / "outputs"
LOGS = ROOT / "logs"
BY_REFIT = OUT / "by_refit"
META_DIR = LOGS / "refit_meta"
for d in (OUT, LOGS, BY_REFIT, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "window_days": 728,
    "target_lag_days": 2,          # information cutoff D-2
    "refit_every": 7,
    "dev_start": "2024-07-01",
    "dev_end": "2024-12-31",
    "test_start": "2025-01-01",
    "test_end": "2025-12-31",
    "price_lag_days": [1, 2, 3, 7],
    "forecast_offsets": [0, 1, 7],   # exogenous at D, D-1, D-7
    "hourly_forecasts": ["demand_forecast", "transmission_wind_forecast",
                         "embedded_wind_forecast", "embedded_solar_forecast"],
    "expected_dev_hours": 4416,
    "expected_test_hours": 8759,
}
HOURS = list(range(24))
COMMODITY_DATE_COLS = ["gas_latest_date", "gas_lag1_date", "gas_lag7_date",
                       "carbon_latest_date", "carbon_lag1_date", "carbon_lag7_date"]


# ----------------------------------------------------------------------------- scaling
class InvariantScaler:
    """epftoolbox 'Invariant' transform: per-column median/MAD scaling, then arcsinh.

    Matches epftoolbox.data.scaling InvariantScaler exactly: mad is statsmodels'
    (raw MAD / 0.6745), the median/MAD step is applied first, arcsinh second.
    """

    def fit(self, X: np.ndarray) -> "InvariantScaler":
        self.median_ = np.nanmedian(X, axis=0)
        raw_mad = np.nanmedian(np.abs(X - self.median_), axis=0)
        self.mad_ = np.where(raw_mad > 0, raw_mad / 0.6744897501960817, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.arcsinh((X - self.median_) / self.mad_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_1d(self, z: np.ndarray, j: int) -> np.ndarray:
        return np.sinh(z) * self.mad_[j] + self.median_[j]


# ----------------------------------------------------------------------------- inputs
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_snapshot() -> dict:
    """Confirm every snapshot input matches SHA256SUMS.txt (self-contained integrity)."""
    sums = {}
    for line in (SNAP / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.strip().split(None, 1)
        sums[name.strip()] = digest
    verified = {}
    for name, want in sums.items():
        got = sha256(SNAP / name)
        if got != want:
            raise AssertionError(f"snapshot integrity FAIL: {name} {got} != {want}")
        verified[name] = got
    return verified


def load_table() -> pd.DataFrame:
    df = pd.read_parquet(SNAP / "model_table_hourly_epex.parquet")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df[df.settlement_date <= CFG["test_end"]].copy()
    return df.sort_values(["settlement_date", "hour"]).reset_index(drop=True)


def read_commodity(path: Path) -> pd.Series:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw["d"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    price = pd.to_numeric(raw["Price"].astype(str).str.replace(",", ""), errors="coerce")
    s = pd.Series(price.values, index=raw["d"].values).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def commodity_features(days: pd.DatetimeIndex, raw: pd.Series, name: str) -> pd.DataFrame:
    """latest (<= D-2), lag1 (preceding close), lag7 (<= latest_date - 7d); with source dates."""
    cd = raw.index.to_numpy()
    cv = raw.to_numpy()
    lookup = (days - pd.Timedelta(days=CFG["target_lag_days"])).to_numpy()
    pos = np.searchsorted(cd, lookup, side="right") - 1
    rows = []
    for p in pos:
        if p < 0:
            rows.append((np.nan, np.nan, np.nan, None, None, None))
            continue
        latest_v, latest_d = cv[p], cd[p]
        lag1_v, lag1_d = (cv[p - 1], cd[p - 1]) if p - 1 >= 0 else (np.nan, None)
        p7 = np.searchsorted(cd, latest_d - np.timedelta64(7, "D"), side="right") - 1
        lag7_v, lag7_d = (cv[p7], cd[p7]) if p7 >= 0 else (np.nan, None)
        rows.append((latest_v, lag1_v, lag7_v, latest_d, lag1_d, lag7_d))
    return pd.DataFrame(rows, index=days,
                        columns=[f"{name}_latest", f"{name}_lag1", f"{name}_lag7",
                                 f"{name}_latest_date", f"{name}_lag1_date", f"{name}_lag7_date"])


def pad_dst_hour(wide: pd.DataFrame, dst_days: list) -> pd.DataFrame:
    """Interpolate the single spring clock-change input hour (missing hour 1) within
    each affected day's 24-hour profile, between the adjacent hours. Input padding
    only; the realised target is left missing and unscored (targets_wide is separate).
    """
    for d in dst_days:
        if d in wide.index and pd.isna(wide.at[d, 1]) and pd.notna(wide.at[d, 0]) and pd.notna(wide.at[d, 2]):
            wide.at[d, 1] = 0.5 * (wide.at[d, 0] + wide.at[d, 2])
    return wide


def build_day_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.DatetimeIndex(sorted(df.settlement_date.unique()))
    dst_days = [d for d, g in df.groupby("settlement_date") if 1 not in set(g.hour.tolist())]
    price_wide = pad_dst_hour(
        df.pivot(index="settlement_date", columns="hour", values="price").reindex(index=days, columns=HOURS),
        dst_days)
    blocks = {}
    for L in CFG["price_lag_days"]:
        shifted = price_wide.shift(L)
        for h in HOURS:
            blocks[f"price_lag{L}_h{h:02d}"] = shifted[h]
    for series in CFG["hourly_forecasts"]:
        wide = pad_dst_hour(
            df.pivot(index="settlement_date", columns="hour", values=series).reindex(index=days, columns=HOURS),
            dst_days)
        for off in CFG["forecast_offsets"]:
            shifted = wide.shift(off)
            tag = {0: "D", 1: "Dm1", 7: "Dm7"}[off]
            for h in HOURS:
                blocks[f"{series}_{tag}_h{h:02d}"] = shifted[h]
    X = pd.DataFrame(blocks, index=days)
    gas = commodity_features(days, read_commodity(SNAP / "uk_natural_gas_futures_investing.csv"), "gas")
    carbon = commodity_features(days, read_commodity(SNAP / "uk_emissions_allowances_futures_investing.csv"), "carbon")
    for c in ["gas_latest", "gas_lag1", "gas_lag7"]:
        X[c] = gas[c]
    for c in ["carbon_latest", "carbon_lag1", "carbon_lag7"]:
        X[c] = carbon[c]
    dow = days.dayofweek
    for k in range(7):
        X[f"dow_{k}"] = (dow == k).astype(float)
    comm_dates = pd.concat([gas.filter(like="_date"), carbon.filter(like="_date")], axis=1)[COMMODITY_DATE_COLS]
    return X, comm_dates


DUMMY_COLS = [f"dow_{k}" for k in range(7)]


def refit_origins(df: pd.DataFrame) -> list[pd.Timestamp]:
    days = pd.DatetimeIndex(sorted(df.settlement_date.unique()))
    dev = days[(days >= CFG["dev_start"]) & (days <= CFG["dev_end"])]
    dev_ref = list(dev[::CFG["refit_every"]])
    qrf = pd.read_csv(SNAP / "qrf_refit_log.csv")
    qrf["test_day"] = pd.to_datetime(qrf["test_day"])
    test_ref = sorted(qrf.loc[(qrf.test_day >= CFG["test_start"]) &
                              (qrf.test_day <= CFG["test_end"]), "test_day"].tolist())
    return dev_ref + test_ref


# ----------------------------------------------------------------------------- backtest
def fit_block(origin, X_day, targets_wide, cont_cols):
    """Fit 24 per-hour LEAR models on the 728-day window ending origin-D-2."""
    end = origin - pd.Timedelta(days=CFG["target_lag_days"])
    start = end - pd.Timedelta(days=CFG["window_days"] - 1)
    win = X_day.index[(X_day.index >= start) & (X_day.index <= end)]
    Xtr_full = X_day.loc[win]
    Xtr_full = Xtr_full[Xtr_full.notna().all(axis=1)]      # drop incomplete rows
    train_days = Xtr_full.index
    scaler = InvariantScaler().fit(Xtr_full[cont_cols].to_numpy(dtype=float))
    Xtr = np.hstack([scaler.transform(Xtr_full[cont_cols].to_numpy(dtype=float)),
                     Xtr_full[DUMMY_COLS].to_numpy(dtype=float)])
    models, yscalers = {}, {}
    penalties = {h: np.nan for h in HOURS}
    nnz = {h: -1 for h in HOURS}
    for h in HOURS:
        y = targets_wide.loc[train_days, h].to_numpy(dtype=float)
        ok = np.isfinite(y)
        if ok.sum() <= Xtr.shape[1]:
            models[h] = None
            continue
        ys = InvariantScaler().fit(y[ok].reshape(-1, 1))
        ysc = ys.transform(y[ok].reshape(-1, 1)).ravel()
        alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xtr[ok], ysc).alpha_
        m = Lasso(alpha=alpha, max_iter=2500).fit(Xtr[ok], ysc)
        models[h], yscalers[h] = m, ys
        penalties[h] = float(alpha)
        nnz[h] = int(np.sum(m.coef_ != 0))
    valid_pen = [p for p in penalties.values() if np.isfinite(p)]
    valid_nnz = [n for n in nnz.values() if n >= 0]
    meta = {"origin": str(origin.date()), "n_train_days": int(len(train_days)),
            "n_features": int(Xtr.shape[1]),
            "cal_start": str(train_days.min().date()), "cal_end": str(train_days.max().date()),
            "median_penalty": float(np.median(valid_pen)) if valid_pen else None,
            "median_nonzero": float(np.median(valid_nnz)) if valid_nnz else None,
            "penalty_by_hour": [penalties[h] for h in HOURS],
            "nonzero_by_hour": [nnz[h] for h in HOURS]}
    return (scaler, models, yscalers, cont_cols), meta


def predict_day(day, X_day, fitted):
    scaler, models, yscalers, cont_cols = fitted
    row = X_day.loc[[day]]
    xc = row[cont_cols].to_numpy(dtype=float)
    if not np.isfinite(xc).all():             # impute forecast-row gaps with train medians
        xc = np.where(np.isfinite(xc), xc, scaler.median_)   # median_ is on the original scale
    xrow = np.hstack([scaler.transform(xc), row[DUMMY_COLS].to_numpy(dtype=float)])
    return {h: (np.nan if models[h] is None
                else float(yscalers[h].inverse_1d(models[h].predict(xrow), 0)[0]))
            for h in HOURS}


def run_audit(pred: pd.DataFrame, df: pd.DataFrame) -> bool:
    pred = pred.copy()
    pred["settlement_date"] = pd.to_datetime(pred["settlement_date"])
    checks = []
    dev = pred[(pred.settlement_date >= CFG["dev_start"]) & (pred.settlement_date <= CFG["dev_end"])]
    test = pred[(pred.settlement_date >= CFG["test_start"]) & (pred.settlement_date <= CFG["test_end"])]
    checks.append(("dev hours == 4416", len(dev) == CFG["expected_dev_hours"], len(dev)))
    checks.append(("test hours == 8759", len(test) == CFG["expected_test_hours"], len(test)))
    checks.append(("no duplicate (date,hour) keys",
                   not pred.duplicated(["settlement_date", "hour"]).any(),
                   int(pred.duplicated(["settlement_date", "hour"]).sum())))
    checks.append(("no missing forecasts", pred["lear_point"].notna().all(),
                   int(pred["lear_point"].isna().sum())))
    m = df[["settlement_date", "hour", "price"]].rename(columns={"price": "ref"})
    j = pred.merge(m, on=["settlement_date", "hour"], how="left")
    checks.append(("actuals match model table",
                   np.allclose(j["actual_price"], j["ref"], atol=1e-9, equal_nan=True),
                   float(np.nanmax(np.abs(j["actual_price"] - j["ref"])))))
    # exact-grid: LEAR keys == the authoritative model-table grid over the forecast span
    span = df[(df.settlement_date >= CFG["dev_start"]) & (df.settlement_date <= CFG["test_end"])]
    grid = set(map(tuple, span[["settlement_date", "hour"]].to_numpy()))
    got = set(map(tuple, pred[["settlement_date", "hour"]].to_numpy()))
    checks.append(("exact (date,hour) grid matches model table",
                   grid == got, f"missing {len(grid - got)}, extra {len(got - grid)}"))
    # commodity gate-legality across EVERY forecast origin: all six source dates <= D-2
    cc = pred if all(c in pred.columns for c in COMMODITY_DATE_COLS) else pred.merge(
        pd.read_csv(OUT / "commodity_source_dates.csv", parse_dates=["settlement_date", *COMMODITY_DATE_COLS]),
        on=["settlement_date", "hour"], how="left")
    d2 = pd.to_datetime(cc["settlement_date"]) - pd.Timedelta(days=CFG["target_lag_days"])
    viol = int(sum((pd.to_datetime(cc[c]) > d2).sum() for c in COMMODITY_DATE_COLS))
    checks.append(("all commodity source dates <= D-2 (every forecast day)", viol == 0, f"{viol} violations"))
    full = pd.concat([dev, test])
    checks.append(("block coverage contiguous (dev+test == total)",
                   len(full) == len(pred), len(pred) - len(full)))
    print(f"\n{'AUDIT CHECK':<44}{'PASS':<6}DETAIL")
    for name, ok, detail in checks:
        print(f"{name:<44}{'ok' if ok else 'FAIL':<6}{detail}")
    ok_all = all(ok for _, ok, _ in checks)
    print("AUDIT PASS" if ok_all else "AUDIT FAILED")
    return ok_all


def write_manifest(verified: dict, origins) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "config": CFG,
        "n_refit_origins": len(origins),
        "snapshot_sha256": verified,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__},
    }
    (LOGS / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    verified = verify_snapshot()
    df = load_table()
    X_day, comm_dates = build_day_features(df)
    targets_wide = (df.pivot(index="settlement_date", columns="hour", values="price")
                      .reindex(index=X_day.index, columns=HOURS))
    cont_cols = [c for c in X_day.columns if c not in DUMMY_COLS]
    origins = refit_origins(df)
    assert len(cont_cols) == 390 and len(DUMMY_COLS) == 7, (len(cont_cols), len(DUMMY_COLS))

    if args.smoke:
        sys.exit(0 if run_smoke(df, X_day, targets_wide, comm_dates, cont_cols, origins, verified) else 1)
    if args.audit:
        pred = pd.read_csv(OUT / "lear_point_forecasts.csv")
        sys.exit(0 if run_audit(pred, df) else 1)

    origin_arr = pd.DatetimeIndex(origins)
    for i, origin in enumerate(origins):
        rf = BY_REFIT / f"lear_{origin.date()}.csv"
        mf = META_DIR / f"{origin.date()}.json"
        if rf.exists() and mf.exists() and rf.stat().st_size > 0 and mf.stat().st_size > 0:
            continue                          # idempotent resume: skip only fully completed refits
        fitted, meta = fit_block(origin, X_day, targets_wide, cont_cols)
        nxt = origin_arr[i + 1] if i + 1 < len(origin_arr) else pd.Timestamp(CFG["test_end"]) + pd.Timedelta(days=1)
        block_days = X_day.index[(X_day.index >= origin) & (X_day.index < nxt)]
        rows = []
        for day in block_days:
            fc = predict_day(day, X_day, fitted)
            cd = comm_dates.loc[day]
            present = df.loc[df.settlement_date == day, ["hour", "price"]]
            for _, r in present.iterrows():
                rows.append({"origin": str(origin.date()), "settlement_date": day,
                             "hour": int(r.hour), "lear_point": fc[int(r.hour)],
                             "actual_price": r.price,
                             **{c: cd[c] for c in COMMODITY_DATE_COLS}})
        # atomic writes: temp file then rename, so a crash cannot leave a "complete" partial
        tmp = rf.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, rf)
        tmpm = mf.with_suffix(".json.tmp")
        tmpm.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(tmpm, mf)
        print(f"[{i+1}/{len(origins)}] {origin.date()}: {meta['n_train_days']}d train, "
              f"median nnz {meta['median_nonzero']:.0f}, {len(rows)} rows", flush=True)

    # ---- consolidate, audit, manifest ----
    pred = pd.concat([pd.read_csv(p) for p in sorted(BY_REFIT.glob("lear_*.csv"))], ignore_index=True)
    pred["settlement_date"] = pd.to_datetime(pred["settlement_date"])
    pred = pred.sort_values(["settlement_date", "hour"]).reset_index(drop=True)
    pred[["origin", "settlement_date", "hour", "lear_point", "actual_price"]].to_csv(
        OUT / "lear_point_forecasts.csv", index=False)
    pred[["settlement_date", "hour", *COMMODITY_DATE_COLS]].to_csv(
        OUT / "commodity_source_dates.csv", index=False)
    metas = [json.loads((META_DIR / f"{o.date()}.json").read_text()) for o in origins]
    pd.DataFrame([{k: m[k] for k in ("origin", "n_train_days", "n_features", "cal_start",
                                     "cal_end", "median_penalty", "median_nonzero")}
                  for m in metas]).to_csv(LOGS / "lear_refit_log.csv", index=False)
    pen_rows = [{"origin": m["origin"], "hour": h, "penalty": m["penalty_by_hour"][h],
                 "nonzero": m["nonzero_by_hour"][h]} for m in metas for h in HOURS]
    pd.DataFrame(pen_rows).to_csv(LOGS / "lear_refit_penalties.csv", index=False)
    write_manifest(verified, origins)
    if not run_audit(pred, df):
        sys.exit(1)                           # fail loudly: non-zero exit on any audit failure
    print("LEAR_POINT_DONE")


def run_smoke(df, X_day, targets_wide, comm_dates, cont_cols, origins, verified):
    origin = [o for o in origins if o >= pd.Timestamp(CFG["test_start"])][0]
    print(f"=== SMOKE: single refit at origin {origin.date()} ===\n")
    fitted, meta = fit_block(origin, X_day, targets_wide, cont_cols)
    scaler, models, yscalers, _ = fitted
    n_feat = len(cont_cols) + len(DUMMY_COLS)
    d2 = origin - pd.Timedelta(days=CFG["target_lag_days"])
    cd = comm_dates.loc[origin]
    comm_ok = all(pd.notna(cd[c]) and pd.Timestamp(cd[c]) <= d2 for c in COMMODITY_DATE_COLS)
    fc = predict_day(origin, X_day, fitted)
    checks = [
        ("snapshot integrity (SHA256SUMS.txt)", bool(verified), f"{len(verified)} files verified"),
        ("X has 397 columns", n_feat == 397, f"{n_feat}"),
        ("weekday dummies unscaled (0/1)",
         set(np.unique(X_day.loc[[origin], DUMMY_COLS].to_numpy())).issubset({0.0, 1.0}), ""),
        ("continuous cols scaled (~0-centred on train)",
         np.nanmax(np.abs(np.median(scaler.transform(
             X_day.loc[X_day.index <= d2][cont_cols].dropna().to_numpy(dtype=float)), axis=0))) < 3, ""),
        ("n_train > 397", meta["n_train_days"] > 397, f"n={meta['n_train_days']}"),
        ("all 24 hourly models fitted", sum(models[h] is not None for h in HOURS) == 24, ""),
        ("all 24 penalties & nnz recorded per hour",
         all(np.isfinite(meta["penalty_by_hour"][h]) and meta["nonzero_by_hour"][h] >= 0 for h in HOURS), ""),
        ("all 24 forecasts finite", all(np.isfinite(v) for v in fc.values()), ""),
        ("all coefficients finite",
         all(np.isfinite(models[h].coef_).all() for h in HOURS if models[h] is not None), ""),
        ("training endpoint == origin - 2d", meta["cal_end"] == str(d2.date()), meta["cal_end"]),
        ("ALL 6 commodity source dates recorded & <= D-2", comm_ok,
         f"gas {pd.Timestamp(cd['gas_latest_date']).date()}/{pd.Timestamp(cd['gas_lag1_date']).date()}/"
         f"{pd.Timestamp(cd['gas_lag7_date']).date()}, carbon {pd.Timestamp(cd['carbon_latest_date']).date()}"),
    ]
    spring = pd.Timestamp("2025-03-30")
    sp = df.loc[df.settlement_date == spring, "hour"].tolist()
    checks.append(("clock-change day scored on observed hours only",
                   len(sp) == 23 and 1 not in sp, f"{len(sp)}h, missing hour 1"))
    affected = pd.to_datetime(["2025-03-30", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-06"])
    nan_days = [str(d.date()) for d in affected if X_day.loc[d, cont_cols].isna().any()]
    checks.append(("spring-DST feature rows padded (5 affected days complete)",
                   len(nan_days) == 0, f"incomplete: {nan_days}" if nan_days else "all 5 complete"))
    print(f"{'CHECK':<52}{'PASS':<6}DETAIL")
    for name, ok, detail in checks:
        print(f"{name:<52}{'ok' if ok else 'FAIL':<6}{detail}")
    print(f"\nrefit meta: n_train={meta['n_train_days']}, median penalty {meta['median_penalty']:.4g}, "
          f"median nnz {meta['median_nonzero']:.0f}")
    print(f"per-hour nnz (h0-h5): {meta['nonzero_by_hour'][:6]}")
    print(f"sample forecasts {origin.date()} h0-h5: {[round(fc[h],1) for h in range(6)]}")
    print(f"actual          {origin.date()} h0-h5: "
          f"{[round(float(df[(df.settlement_date==origin)&(df.hour==h)].price.iloc[0]),1) for h in range(6)]}")
    ok_all = all(ok for _, ok, _ in checks)
    print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
    return ok_all


if __name__ == "__main__":
    main()
