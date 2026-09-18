"""Multi-window canonical LEAR point forecasts for the literature-standard LEAR-QRA.

Produces LEAR point forecasts at one calibration-window length using the CANONICAL
two-exogenous epftoolbox LEAR (247 features), so QRA can later combine several
window forecasts into LEAR quantiles (the standard LEAR-QRA benchmark; Marcjasz
et al. 2023, Lipiecki et al. 2024). Penalty procedure is the epftoolbox hybrid
(LARS+AIC selection, then coordinate-descent Lasso refit).

Adapted to the study's common weekly rolling protocol rather than epftoolbox's
standalone daily recalibration: the 24 LASSO
coefficient sets refit weekly and stay fixed within each block, while every day
gets newly constructed gate-legal inputs and its own 24 forecasts. Refit origins
are weekly across the Jan-Dec 2024 backcast/development span, then the EXACT saved
QRF origins for 2025 (so the 2025 grid matches the other models). Every estimation
sample ends at D-2.

Feature set (247): 96 price lags (D-1,D-2,D-3,D-7) + 2 exogenous (demand and total
renewable [wind+solar] forecasts) at D/D-1/D-7 (144) + 7 weekday dummies. The
2-exogenous spec is required for AIC validity (n > p + 1) across the shorter
windows and the early backcast, which the 397-feature spec cannot satisfy.

Usage: python run_lear_qra_window.py --window 273 [--smoke|--audit]
Outputs (per window w): outputs/lear_qra_point_w{w}.csv, logs/lear_qra_refit_w{w}.csv,
logs/lear_qra_penalties_w{w}.csv, logs/lear_qra_manifest_w{w}.json
"""
from __future__ import annotations

import argparse
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

import run_lear as base  # reuse verified InvariantScaler, DST padding, snapshot check, loader

OUT = Path(__file__).resolve().parent.parent / "outputs"
LOGS = Path(__file__).resolve().parent.parent / "logs"
OUT.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
BACKCAST_START = pd.Timestamp("2024-01-01")
DEV_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")
HOURS = base.HOURS
LAG = base.CFG["target_lag_days"]
REFIT_EVERY = base.CFG["refit_every"]
EXOG = ["demand_forecast", "renewable_forecast_total"]   # canonical load + total RES (wind+solar)
PRICE_LAGS = [1, 2, 3, 7]
OFFSETS = [0, 1, 7]
DUMMY_COLS = [f"dow_{k}" for k in range(7)]
REFIT_LASSO_MAXITER = 2500    # epftoolbox value; near-p=n short-window non-convergence is tolerated (as in epftoolbox), not chased
WINDOWS = [273, 364, 546, 728]


def build_2exog_features(df):
    days = pd.DatetimeIndex(sorted(df.settlement_date.unique()))
    dst_days = [d for d, g in df.groupby("settlement_date") if 1 not in set(g.hour.tolist())]
    price_wide = base.pad_dst_hour(
        df.pivot(index="settlement_date", columns="hour", values="price").reindex(index=days, columns=HOURS),
        dst_days)
    blocks = {}
    for L in PRICE_LAGS:
        sh = price_wide.shift(L)
        for h in HOURS:
            blocks[f"price_lag{L}_h{h:02d}"] = sh[h]
    for s in EXOG:
        wide = base.pad_dst_hour(
            df.pivot(index="settlement_date", columns="hour", values=s).reindex(index=days, columns=HOURS), dst_days)
        for off in OFFSETS:
            sh = wide.shift(off)
            tag = {0: "D", 1: "Dm1", 7: "Dm7"}[off]
            for h in HOURS:
                blocks[f"{s}_{tag}_h{h:02d}"] = sh[h]
    X = pd.DataFrame(blocks, index=days)
    dow = days.dayofweek
    for k in range(7):
        X[f"dow_{k}"] = (dow == k).astype(float)
    return X


def refit_origins(df):
    """Weekly across the Jan-Dec 2024 backcast/dev span, then the exact QRF 2025 origins."""
    days = pd.DatetimeIndex(sorted(df.settlement_date.unique()))
    dev = list(days[(days >= BACKCAST_START) & (days <= DEV_END)][::REFIT_EVERY])
    qrf = pd.read_csv(base.SNAP / "qrf_refit_log.csv")
    qrf["test_day"] = pd.to_datetime(qrf["test_day"])
    test = sorted(qrf.loc[(qrf.test_day >= TEST_START) & (qrf.test_day <= TEST_END), "test_day"].tolist())
    return dev + test


def fit_block(origin, X_day, targets_wide, cont_cols, window):
    end = origin - pd.Timedelta(days=LAG)
    start = end - pd.Timedelta(days=window - 1)
    win = X_day.index[(X_day.index >= start) & (X_day.index <= end)]
    Xtr_full = X_day.loc[win]
    Xtr_full = Xtr_full[Xtr_full.notna().all(axis=1)]
    train_days = Xtr_full.index
    scaler = base.InvariantScaler().fit(Xtr_full[cont_cols].to_numpy(float))
    Xtr = np.hstack([scaler.transform(Xtr_full[cont_cols].to_numpy(float)),
                     Xtr_full[DUMMY_COLS].to_numpy(float)])
    p = Xtr.shape[1]
    models, yscalers = {}, {}
    penalties = {h: np.nan for h in HOURS}
    nnz = {h: -1 for h in HOURS}
    nfit_samp = {h: 0 for h in HOURS}
    nonconv = 0
    max_dual_gap = 0.0
    for h in HOURS:
        y = targets_wide.loc[train_days, h].to_numpy(float)
        ok = np.isfinite(y)
        nfit_samp[h] = int(ok.sum())
        if ok.sum() <= p + 1:                 # AIC needs n > p + intercept
            models[h] = None
            continue
        ys = base.InvariantScaler().fit(y[ok].reshape(-1, 1))
        ysc = ys.transform(y[ok].reshape(-1, 1)).ravel()
        alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xtr[ok], ysc).alpha_
        m = Lasso(alpha=alpha, max_iter=REFIT_LASSO_MAXITER).fit(Xtr[ok], ysc)
        dg = float(getattr(m, "dual_gap_", np.nan))
        if getattr(m, "n_iter_", 0) >= REFIT_LASSO_MAXITER:   # hit the cap => not fully converged
            nonconv += 1
        if np.isfinite(dg):
            max_dual_gap = max(max_dual_gap, dg)
        models[h], yscalers[h] = m, ys
        penalties[h] = float(alpha)
        nnz[h] = int(np.sum(m.coef_ != 0))
    meta = {"origin": str(origin.date()), "window": window, "n_train": int(len(train_days)),
            "n_features": int(p), "cal_start": str(train_days.min().date()),
            "cal_end": str(train_days.max().date()),
            "n_fitted": int(sum(models[h] is not None for h in HOURS)),
            "n_nonconverged": int(nonconv), "max_dual_gap": round(max_dual_gap, 6),
            "penalty_by_hour": [penalties[h] for h in HOURS],
            "nonzero_by_hour": [nnz[h] for h in HOURS],
            "n_sample_by_hour": [nfit_samp[h] for h in HOURS]}
    return (scaler, models, yscalers, cont_cols), meta


def predict_day(day, X_day, fitted):
    scaler, models, yscalers, cont_cols = fitted
    row = X_day.loc[[day]]
    xc = row[cont_cols].to_numpy(float)
    if not np.isfinite(xc).all():
        xc = np.where(np.isfinite(xc), xc, scaler.median_)
    xrow = np.hstack([scaler.transform(xc), row[DUMMY_COLS].to_numpy(float)])
    return {h: (np.nan if models[h] is None
                else float(yscalers[h].inverse_1d(models[h].predict(xrow), 0)[0])) for h in HOURS}


def audit(pred, df, window):
    pred = pred.copy(); pred["settlement_date"] = pd.to_datetime(pred["settlement_date"])
    col = f"lear_w{window}"
    span = df[(df.settlement_date >= BACKCAST_START) & (df.settlement_date <= TEST_END)]
    grid = set(map(tuple, span[["settlement_date", "hour"]].to_numpy()))
    got = set(map(tuple, pred[["settlement_date", "hour"]].to_numpy()))
    j = pred.merge(df[["settlement_date", "hour", "price"]], on=["settlement_date", "hour"], how="left")
    checks = [
        ("exact (date,hour) grid == model table", grid == got, f"missing {len(grid-got)}, extra {len(got-grid)}"),
        ("no duplicate keys", not pred.duplicated(["settlement_date", "hour"]).any(),
         int(pred.duplicated(["settlement_date", "hour"]).sum())),
        ("all forecasts finite", np.isfinite(pred[col]).all(), int((~np.isfinite(pred[col])).sum())),
        ("actuals match model table", np.allclose(j.actual_price, j.price, atol=1e-9),
         float(np.nanmax(np.abs(j.actual_price - j.price)))),
    ]
    print(f"\n=== AUDIT w{window} ===")
    for n, okc, d in checks:
        print(f"  {'ok' if okc else 'FAIL':<5}{n:<42}{d}")
    ok_all = all(okc for _, okc, _ in checks)
    print("AUDIT PASS" if ok_all else "AUDIT FAILED")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True, choices=WINDOWS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    verified = base.verify_snapshot()
    df = base.load_table()
    X_day = build_2exog_features(df)
    targets_wide = (df.pivot(index="settlement_date", columns="hour", values="price")
                      .reindex(index=X_day.index, columns=HOURS))
    cont_cols = [c for c in X_day.columns if c not in DUMMY_COLS]
    assert len(cont_cols) == 240 and X_day.shape[1] == 247, (len(cont_cols), X_day.shape[1])
    origins = refit_origins(df)

    if args.audit:
        pred = pd.read_csv(OUT / f"lear_qra_point_w{args.window}.csv")
        sys.exit(0 if audit(pred, df, args.window) else 1)

    if args.smoke:
        for origin in (origins[0], [o for o in origins if o >= TEST_START][0]):
            fitted, meta = fit_block(origin, X_day, targets_wide, cont_cols, args.window)
            fc = predict_day(origin, X_day, fitted)
            d2 = origin - pd.Timedelta(days=LAG)
            print(f"SMOKE w{args.window} origin {origin.date()}: n_train={meta['n_train']} p={meta['n_features']} "
                  f"({'n>p+1 OK' if meta['n_train'] > meta['n_features']+1 else 'AIC-INVALID'}), "
                  f"cal_end={meta['cal_end']} (D-2={d2.date()} {'OK' if meta['cal_end']==str(d2.date()) else 'BAD'}), "
                  f"fitted={meta['n_fitted']}/24, nonconv={meta['n_nonconverged']}, "
                  f"finite={all(np.isfinite(v) for v in fc.values())}, fc h0-h5={[round(fc[h],1) for h in range(6)]}")
        print(f"2025 first origin = {[o for o in origins if o >= TEST_START][0].date()} (must be 2025-01-01)")
        return

    origin_arr = pd.DatetimeIndex(origins)
    rows, metas = [], []
    for i, origin in enumerate(origins):
        fitted, meta = fit_block(origin, X_day, targets_wide, cont_cols, args.window)
        if meta["n_fitted"] != 24:          # structural failure only: abort early
            raise RuntimeError(f"w{args.window} origin {meta['origin']}: only {meta['n_fitted']}/24 hourly models fitted")
        # Non-convergence in the near-p=n short-window regime is tolerated and recorded, as in
        # epftoolbox (which ignores ConvergenceWarning); the finite-forecast audit is the real guard.
        metas.append(meta)
        nxt = origin_arr[i + 1] if i + 1 < len(origin_arr) else TEST_END + pd.Timedelta(days=1)
        for day in X_day.index[(X_day.index >= origin) & (X_day.index < nxt)]:
            fc = predict_day(day, X_day, fitted)
            for _, r in df.loc[df.settlement_date == day, ["hour", "price"]].iterrows():
                rows.append({"settlement_date": day, "hour": int(r.hour),
                             f"lear_w{args.window}": fc[int(r.hour)], "actual_price": r.price})
        print(f"[{i+1}/{len(origins)}] {origin.date()} w{args.window}: n_train={meta['n_train']}, "
              f"fitted={meta['n_fitted']}/24, nonconv={meta['n_nonconverged']}", flush=True)

    pred = pd.DataFrame(rows)
    total_nonconv = int(sum(m["n_nonconverged"] for m in metas))
    ok = audit(pred, df, args.window)          # audit BEFORE persisting the final file
    if total_nonconv:
        print(f"note: {total_nonconv} hourly fits did not fully converge at max_iter={REFIT_LASSO_MAXITER} "
              f"(tolerated as in epftoolbox; all forecasts finite per audit, count recorded in manifest)")
    if not ok:
        sys.exit(1)
    # audit passed -> persist atomically
    tmp = OUT / f"lear_qra_point_w{args.window}.csv.tmp"
    pred.to_csv(tmp, index=False); os.replace(tmp, OUT / f"lear_qra_point_w{args.window}.csv")
    pd.DataFrame([{k: m[k] for k in ("origin", "window", "n_train", "n_features", "cal_start",
                                     "cal_end", "n_fitted", "n_nonconverged", "max_dual_gap")}
                  for m in metas]).to_csv(LOGS / f"lear_qra_refit_w{args.window}.csv", index=False)
    pd.DataFrame([{"origin": m["origin"], "hour": h, "penalty": m["penalty_by_hour"][h],
                   "nonzero": m["nonzero_by_hour"][h], "n_sample": m["n_sample_by_hour"][h]}
                  for m in metas for h in HOURS]).to_csv(
        LOGS / f"lear_qra_penalties_w{args.window}.csv", index=False)
    (LOGS / f"lear_qra_manifest_w{args.window}.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": args.window, "n_origins": len(origins), "exogenous": EXOG,
        "estimator": "LassoLarsIC(aic) select -> Lasso refit (max_iter %d)" % REFIT_LASSO_MAXITER,
        "total_nonconverged_hours": total_nonconv,
        "nonconverged_origins_2024_backcast": int(sum(
            m["n_nonconverged"] > 0 and m["origin"] < "2025-01-01" for m in metas)),
        "nonconverged_origins_2025": int(sum(
            m["n_nonconverged"] > 0 and m["origin"] >= "2025-01-01" for m in metas)),
        "max_dual_gap_any_origin": round(max(m["max_dual_gap"] for m in metas), 6),
        "convergence_note": ("epftoolbox max_iter=2500; non-convergence tolerated and recorded, "
                             "not chased; reference-implementation outputs, concentrated in the "
                             "near-p=n 273-day early-backcast fits"),
        "snapshot_sha256": verified,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}, indent=2) + "\n")
    print(f"W{args.window}_DONE")


if __name__ == "__main__":
    main()
