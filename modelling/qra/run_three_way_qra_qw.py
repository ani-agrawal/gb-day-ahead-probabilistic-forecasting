"""Three-model QRA: LSTM + QRF + multi-window LEAR-QRA.

At each quantile level tau, the combination is

    q_tau = b0 + bL*q_tau^LSTM + bF*q_tau^QRF + bLEAR*q_tau^{LEAR-QRA} ,

using the principal LEAR-QRA forecasts (364/546/728-day windows) as the third
predictor.

Protocol identical to the existing QRA stages (7-day blocks from 1 Jan 2025 -> the
Jan 1/8/15 grid, D-2 cutoff, expanding pool from 1 Jul 2024). Arms: two_way
(self-check vs 5.234), three_way_qw (plain), three_way_qw_reg (dev-selected L1).
Shared input and scoring routines are defined in ``common.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import QuantileRegressor

import common as b

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
LEVELS, QCOLS, KEY, LEV = b.LEVELS, b.QCOLS, b.KEY, b.LEV
BLOCK_DAYS, LAG_DAYS = b.BLOCK_DAYS, b.LAG_DAYS
TEST_START, TEST_END = b.TEST_START, b.TEST_END


def lear_qra_frame():
    d = pd.read_csv(b.ROOT / "lear_qra/outputs/lear_qra_quantiles_principal.csv", parse_dates=["settlement_date"])
    d = d.rename(columns={c: f"lqra_{c}" for c in QCOLS})
    return d[KEY + [f"lqra_{c}" for c in QCOLS]]


def feats_for(arm, qc):
    return [f"lstm_{qc}", f"qrf_{qc}"] if arm == "two" else [f"lstm_{qc}", f"qrf_{qc}", f"lqra_{qc}"]


def qra_backtest(all_oos, test, name, alpha):
    flag = "two" if name == "two_way" else "three"   # feature-set flag; label rows with the display name
    raw = np.full((len(test), len(LEVELS)), np.nan)
    coef_rows = []
    for start in pd.date_range(TEST_START, TEST_END, freq=f"{BLOCK_DAYS}D"):
        end = min(start + pd.Timedelta(days=BLOCK_DAYS - 1), TEST_END)
        tr = all_oos[all_oos.settlement_date <= start - pd.Timedelta(days=LAG_DAYS)]
        mask = test.settlement_date.between(start, end).to_numpy()
        for j, (tau, qc) in enumerate(zip(LEVELS, QCOLS)):
            f = feats_for(flag, qc)
            mdl = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True, solver="highs").fit(
                tr[f].to_numpy(), tr["actual_price"].to_numpy())
            raw[mask, j] = mdl.predict(test.loc[mask, f].to_numpy())
            rec = {"arm": name, "block_start": str(start.date()), "quantile": tau, "intercept": float(mdl.intercept_)}
            rec.update({f"beta_{c}": float(v) for c, v in zip(f, mdl.coef_)})
            coef_rows.append(rec)
    assert not np.isnan(raw).any()
    return np.sort(raw, axis=1), float(np.any(np.diff(raw, axis=1) < 0, axis=1).mean()), coef_rows


def select_reg_alpha(all_oos):
    dev = all_oos[(all_oos.settlement_date >= "2024-11-01") & (all_oos.settlement_date <= "2024-12-31")].copy()
    best, ba = np.inf, 1e-3
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1):
        raw = np.full((len(dev), len(LEVELS)), np.nan)
        for start in pd.date_range("2024-11-01", "2024-12-31", freq="7D"):
            tr = all_oos[all_oos.settlement_date <= start - pd.Timedelta(days=LAG_DAYS)]
            mask = dev.settlement_date.between(start, start + pd.Timedelta(days=6)).to_numpy()
            if not mask.any():
                continue
            for j, (tau, qc) in enumerate(zip(LEVELS, QCOLS)):
                f = feats_for("three", qc)
                mdl = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True, solver="highs").fit(
                    tr[f].to_numpy(), tr["actual_price"].to_numpy())
                raw[mask, j] = mdl.predict(dev.loc[mask, f].to_numpy())
        ok = ~np.isnan(raw).any(axis=1)
        y = dev.actual_price.to_numpy()[ok]; q = np.sort(raw[ok], axis=1)
        pin = np.where(y[:, None] - q >= 0, LEV * (y[:, None] - q), (LEV - 1) * (y[:, None] - q)).mean()
        if pin < best:
            best, ba = pin, alpha
    return ba, best


def main():
    lstm = b.lstm_frame().rename(columns={c: f"lstm_{c}" for c in QCOLS})
    qrf = b.qrf_frame().rename(columns={c: f"qrf_{c}" for c in QCOLS})
    lqra = lear_qra_frame()
    m = lstm.merge(qrf.drop(columns="actual_price"), on=KEY).merge(lqra, on=KEY)
    all_oos = m.dropna().sort_values(KEY).reset_index(drop=True)
    test = all_oos[(all_oos.settlement_date >= TEST_START) & (all_oos.settlement_date <= TEST_END)].copy()
    assert len(test) == 8759, len(test)
    y = test.actual_price.to_numpy()

    reg_alpha, reg_pin = select_reg_alpha(all_oos)
    preds, crossings, coefs = {}, {}, []
    for name, alpha in [("two_way", 0.0), ("three_way_qw", 0.0), ("three_way_qw_reg", reg_alpha)]:
        s, cr, co = qra_backtest(all_oos, test, name, alpha)
        preds[name], crossings[name] = s, cr
        coefs += co
    const = {"LSTM": test[[f"lstm_{c}" for c in QCOLS]].to_numpy(),
             "QRF": test[[f"qrf_{c}" for c in QCOLS]].to_numpy(),
             "LEAR-QRA": test[[f"lqra_{c}" for c in QCOLS]].to_numpy()}

    rows, daily = [], {}
    for name, q in {**const, **preds}.items():
        r = b.score(y, q, name); r["raw_crossing_rate"] = crossings.get(name, 0.0); rows.append(r)
        h = test[KEY].copy(); h["c"] = b.crps11(y, q); daily[name] = h.groupby("settlement_date")["c"].sum()
    metrics = pd.DataFrame(rows).sort_values("crps_11")
    metrics.to_csv(OUT / "three_way_qw_metrics.csv", index=False)
    dm = [{"a": a, "b": bb, "p_value": b.dm_pvalue(daily[a], daily[bb])}
          for a, bb in [("three_way_qw", "two_way"), ("three_way_qw", "LSTM"), ("three_way_qw", "QRF"),
                        ("three_way_qw", "LEAR-QRA"), ("three_way_qw_reg", "three_way_qw")]]
    pd.DataFrame(dm).to_csv(OUT / "three_way_qw_dm.csv", index=False)
    pd.DataFrame(coefs).to_csv(OUT / "three_way_qw_coefficients.csv", index=False)
    out = test[KEY + ["actual_price"]].copy()
    for name, q in preds.items():
        for j, c in enumerate(QCOLS):
            out[f"{name}_{c}"] = q[:, j]
    out.to_csv(OUT / "three_way_qw_predictions.csv", index=False)
    (OUT / "three_way_qw_manifest.json").write_text(json.dumps({
        "reg_alpha_dev": reg_alpha, "sklearn": sklearn.__version__,
        "note": "three-model QRA using LSTM, QRF and principal 364/546/728-day LEAR-QRA forecasts"
    }, indent=2) + "\n")

    print("=== three-model QRA (LSTM + QRF + LEAR-QRA), 2025 ===")
    print(metrics[["model", "crps_11", "mae_q50", "cov90", "width90", "winkler90", "cov_gt200",
                   "raw_crossing_rate"]].round(3).to_string(index=False))
    print("\n=== DM (one-sided, p small => a better) ===")
    print(pd.DataFrame(dm).round(5).to_string(index=False))
    tw = pd.DataFrame(coefs); tw = tw[tw.arm == "three_way_qw"]
    lq = tw.filter(like="beta_lqra")
    print(f"\nLEAR-QRA weight range across blocks x quantiles: {lq.min().min():.3f} to {lq.max().max():.3f}")
    print(f"two_way self-check CRPS = {metrics.set_index('model').loc['two_way','crps_11']:.3f} "
          f"(two-model result = 5.234); reg alpha(dev) = {reg_alpha}")
    print("THREE_WAY_QW_DONE")


if __name__ == "__main__":
    main()
