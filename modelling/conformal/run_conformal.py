"""Conformal layers on the ensembled system: static and rolling asymmetric CQR.

Design used in the paper:
- Asymmetric scores per bound: E_lo = q_lo - y, E_hi = y - q_hi; each bound
  corrected at the (1 - alpha/2) level with the (1+1/n) finite-sample factor.
  Corrections may be negative (tightening). Scores pooled across hours.
- Intervals: (q05, q95, alpha=.10) and (q10, q90, alpha=.20). Median and
  interior quantiles pass through unchanged (conformal calibrates the named
  intervals).
- STATIC split CQR: the 2025 test intervals are corrected with 2024-H2
  calibration scores from the same final LSTM configuration. The correction
  is constant throughout the test year by design.
- ROLLING asymmetric CQR: for each delivery day D, corrections re-estimated
  from the trailing W delivery days (W re-selected on the 2024-H2 calibration set) with day <= D-2 (the newest complete
  day at the 09:20 D-1 gate); state updates between successive daily
  auctions, never intraday. Warm-up: 2024-H2 scores.
- LIMITATIONS (stated in methodology): 2024-H2 also served as the dev-
  validation period, so the static-2025 calibration set is not untouched;
  scores are pooled across dependent hours, so results are empirical
  marginal hourly calibration, not a finite-sample guarantee.

Outputs -> 06_outputs/RQ2/model_runs/rnn_2025_backtest/conformal_ENSEMBLE/
  conformal_results.csv (raw/static/rolling x period x band metrics)
  weekly_coverage.csv   (per-week 90% coverage per layer - drift exhibit)
  predictions_<period>.csv (raw + static + rolling adjusted bounds)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

W_CANDIDATES = [30, 60, 90]   # re-selected on the ensemble's own 2024-H2
                               # calibration forecasts ONLY (burn-in then evaluate;
                               # criterion: |cov90 - 0.90| first, Winkler second).
INTERVALS = [("q05", "q95", 0.10), ("q10", "q90", 0.20)]
LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
COLS = [f"q{int(t * 100):02d}" for t in LEVELS]
KEY = ["settlement_date", "hour"]

ROOT = Path(__file__).resolve().parents[2]
RQ2 = ROOT / "06_outputs" / "RQ2" / "model_runs" / "rnn_2025_backtest"
OUT = ROOT / "06_outputs" / "RQ2" / "model_runs" / "seq2seq_final_b16es14" / "conformal_ENSEMBLE"
OUT.mkdir(parents=True, exist_ok=True)

# ---- calibration ensemble (2024-H2): average the three seeds
# The encdec_u64_b16es14_seed* arms are the final b16/ES14 encoder-decoder
# system's own 2024-H2 dev forecasts (5-quantile runs), so the calibration
# scores come from the same configuration that produced the test forecasts.
# Conformal uses only the band bounds (q05/q95, q10/q90), so no interior fill.
C5 = ["q05", "q10", "q50", "q90", "q95"]
def load_cal_path(d):
    p = pd.read_csv(d / "predictions.csv")
    p[C5] = np.sort(p[C5].to_numpy(), axis=1)
    return p.sort_values(KEY).reset_index(drop=True)

CAL_BASE = ROOT / "06_outputs" / "RQ2" / "model_runs" / "rnn_dev" / "seq2seq_gate_v2"
cal_seeds = [load_cal_path(CAL_BASE / f"encdec_u64_b16es14_seed{s}") for s in (42, 7, 101)]
for c in cal_seeds[1:]:
    assert len(c) == len(cal_seeds[0]), "calibration seeds: row counts differ"
    assert (c["settlement_date"].values == cal_seeds[0]["settlement_date"].values).all()
    assert (c["hour"].values == cal_seeds[0]["hour"].values).all()
    assert np.allclose(c["actual_price"], cal_seeds[0]["actual_price"]), "cal seeds: actuals differ"
cal = cal_seeds[0][KEY + ["actual_price"]].copy()
cal[C5] = np.sort(np.mean([c[C5].to_numpy() for c in cal_seeds], axis=0), axis=1)
cal["settlement_date"] = pd.to_datetime(cal["settlement_date"])
print(f"calibration ensemble: {len(cal)} hours "
      f"({cal.settlement_date.min().date()} -> {cal.settlement_date.max().date()})")

test = pd.read_csv(ROOT / "06_outputs" / "RQ2" / "model_runs" / "seq2seq_final_b16es14" / "ENSEMBLE_11q" / "predictions.csv")
test["settlement_date"] = pd.to_datetime(test["settlement_date"])
test = test.sort_values(KEY).reset_index(drop=True)

def rolling_on_frame(frame, w, lo, hi, alpha):
    """Rolling corrections within one frame (used for W selection on cal)."""
    f = frame.copy()
    f["e_lo"], f["e_hi"] = f[lo] - f["actual_price"], f["actual_price"] - f[hi]
    by_day = {d: g for d, g in f.groupby("settlement_date")}
    days = sorted(by_day)
    lo_a, hi_a, mask = [], [], []
    for day, g in f.groupby("settlement_date"):
        cut = day - pd.Timedelta(days=2)
        wd = [d for d in days if d <= cut][-w:]
        if len(wd) < w:
            continue
        wdf = pd.concat([by_day[d] for d in wd])
        lo_a.append(g[lo] - corr(wdf["e_lo"], alpha / 2))
        hi_a.append(g[hi] + corr(wdf["e_hi"], alpha / 2))
        mask.append(g.index)
    idx = np.concatenate([m.values for m in mask])
    return f.loc[idx, "actual_price"].to_numpy(), pd.concat(lo_a).to_numpy(), pd.concat(hi_a).to_numpy()

PERIODS = {"2025_primary": ("2025-01-01", "2025-12-31")}
STATIC_CAL = {"2025_primary": cal}


def corr(e, alpha_tail):
    n = len(e)
    level = min(1.0, (1 - alpha_tail) * (1 + 1 / n))
    return float(np.quantile(e, level, method="higher"))  # exact upper order statistic


def winkler(y, lo, hi, alpha):
    w = hi - lo
    pen = np.where(y < lo, (2 / alpha) * (lo - y),
                   np.where(y > hi, (2 / alpha) * (y - hi), 0.0))
    return float(np.mean(w + pen))


def metrics(df, lo, hi, alpha, label, period, layer):
    y = df["actual_price"].to_numpy()
    l, h = df[f"{lo}_adj"].to_numpy(), df[f"{hi}_adj"].to_numpy()
    inside = (y >= l) & (y <= h)
    neg, s200, s150 = y < 0, y > 200, y > 150
    return {"layer": layer, "period": period, "band": label,
            "coverage": round(float(inside.mean()), 3),
            "miss_below": round(float((y < l).mean()), 4),
            "miss_above": round(float((y > h).mean()), 4),
            "width": round(float((h - l).mean()), 1),
            "winkler": round(winkler(y, l, h, alpha), 1),
            "cov_neg": round(float(inside[neg].mean()) if neg.any() else np.nan, 3),
            "cov_gt200": round(float(inside[s200].mean()) if s200.any() else np.nan, 3),
            "cov_gt150": round(float(inside[s150].mean()) if s150.any() else np.nan, 3)}


print("\n=== W selection on 2024-H2 calibration only (90% band) ===")
w_stats = []
COMMON_EVAL_START = cal["settlement_date"].min() + pd.Timedelta(days=92)  # common window: all candidates scored on the same final stretch
for w in W_CANDIDATES:
    y_w, lo_w, hi_w = rolling_on_frame(cal[cal.settlement_date >= COMMON_EVAL_START - pd.Timedelta(days=w + 2)], w, "q05", "q95", 0.10)
    cov = float(((y_w >= lo_w) & (y_w <= hi_w)).mean())
    wk = winkler(y_w, lo_w, hi_w, 0.10)
    w_stats.append((abs(cov - 0.90), wk, w, cov))
    print(f"W={w}: eval days n={len(y_w)//24}, cov90 {cov:.3f}, winkler {wk:.1f}")
w_stats.sort()
W = w_stats[0][2]
print(f"SELECTED W = {W} (criterion: |cov-0.90| then Winkler)")


rows, weekly = [], []
for period, (a, b) in PERIODS.items():
    sub = test[(test.settlement_date >= a) & (test.settlement_date <= b)].copy()
    calp = STATIC_CAL[period]
    for lo, hi, alpha in INTERVALS:
        # STATIC corrections from the period's calibration set
        e_lo, e_hi = calp[lo] - calp["actual_price"], calp["actual_price"] - calp[hi]
        sub[f"{lo}_static"] = sub[lo] - corr(e_lo, alpha / 2)
        sub[f"{hi}_static"] = sub[hi] + corr(e_hi, alpha / 2)
        # ROLLING: trailing W delivery days ending D-2 (warm-up: cal + earlier test)
        hist = pd.concat([cal[KEY + ["actual_price", lo, hi]],
                          test[test.settlement_date < a][KEY + ["actual_price", lo, hi]],
                          sub[KEY + ["actual_price", lo, hi]]], ignore_index=True)
        hist["settlement_date"] = pd.to_datetime(hist["settlement_date"])
        hist["e_lo"] = hist[lo] - hist["actual_price"]
        hist["e_hi"] = hist["actual_price"] - hist[hi]
        by_day = {d: g for d, g in hist.groupby("settlement_date")}
        days_all = sorted(by_day)
        lo_adj, hi_adj = [], []
        for day, g in sub.groupby("settlement_date"):
            cut = day - pd.Timedelta(days=2)
            window_days = [d for d in days_all if d <= cut][-W:]
            wdf = pd.concat([by_day[d] for d in window_days])
            lo_adj.append(g[lo] - corr(wdf["e_lo"], alpha / 2))
            hi_adj.append(g[hi] + corr(wdf["e_hi"], alpha / 2))
        sub[f"{lo}_roll"] = pd.concat(lo_adj).sort_index()
        sub[f"{hi}_roll"] = pd.concat(hi_adj).sort_index()
    # NESTING ENFORCEMENT (outer-widening rearrangement): where the calibrated
    # 90% bounds cross the calibrated 80% bounds, the OUTER bound is widened to
    # enclose the inner one. Conservative for the 90% band; 80% band untouched.
    for suffix in ("static", "roll"):
        v_lo = (sub[f"q05_{suffix}"] > sub[f"q10_{suffix}"]).sum()
        v_hi = (sub[f"q90_{suffix}"] > sub[f"q95_{suffix}"]).sum()
        print(f"{period} {suffix}: nesting violations lo={v_lo} hi={v_hi} "
              f"of {len(sub)} rows -> outer-widened")
        sub[f"q05_{suffix}"] = np.minimum(sub[f"q05_{suffix}"], sub[f"q10_{suffix}"])
        sub[f"q95_{suffix}"] = np.maximum(sub[f"q95_{suffix}"], sub[f"q90_{suffix}"])
    # metrics AFTER enforcement
    for lo, hi, alpha in INTERVALS:
        label = f"{int((1 - alpha) * 100)}%"
        for layer, (lc, hc) in {"raw": (lo, hi),
                                "static": (f"{lo}_static", f"{hi}_static"),
                                "rolling": (f"{lo}_roll", f"{hi}_roll")}.items():
            sub[f"{lo}_adj"], sub[f"{hi}_adj"] = sub[lc], sub[hc]
            rows.append(metrics(sub, lo, hi, alpha, label, period, layer))
            if alpha == 0.10:
                y = sub["actual_price"]
                wk = sub.assign(inside=((y >= sub[lc]) & (y <= sub[hc])).astype(float)) \
                        .groupby(sub.settlement_date.dt.to_period("W"))["inside"].mean()
                for p, v in wk.items():
                    weekly.append({"period": period, "layer": layer,
                                   "week": str(p.start_time.date()), "cov90": round(v, 3)})
    keep = KEY + ["actual_price"] + COLS + \
           [c for c in sub.columns if c.endswith(("_static", "_roll"))]
    sub[keep].to_csv(OUT / f"predictions_{period}.csv", index=False)

res = pd.DataFrame(rows)
res["W_selected"] = W
res.to_csv(OUT / "conformal_results.csv", index=False)
pd.DataFrame(weekly).to_csv(OUT / "weekly_coverage.csv", index=False)
print("\n=== CONFORMAL RESULTS (ensemble base) ===")
for period in PERIODS:
    print(f"\n--- {period} ---")
    print(res[res.period == period].to_string(index=False))
