"""XGBoost quantile gradient-boosted trees on the QRF feature matrix (Sections 4.3 and 5.2.3;
Tables 4, 9, A.20 and A.21).

The QRF reads its quantiles off realised training prices retained in its leaves; boosted
trees are also a tree ensemble, but their quantiles are additive leaf values fitted by
pinball minimisation, as the LSTM's are. Protocol identical to the paper's QRF where the
learner allows: the 34-feature lean list, median imputation fitted inside each window,
364-day rolling window ending D-2, weekly refits on the same test-day index, random
state 42, column sampling 0.5 (QRF max_features) and minimum leaf weight 5 (QRF leaf
size). One multi-quantile booster returns the eleven levels of the paper's grid, rows
sorted. Boosting rounds are set by early stopping on validation pinball loss over the
final 14 days of each window (the LSTM's holdout), learning rate 0.05, at most 2,000
rounds, patience 50.

Development comparison (no automated search): tree depth chosen on mean pinball loss
over July-December 2024 with the same rolling protocol, before any 2025 forecast. The
loss fell monotonically towards depth 3, the edge of the initial grid {3, 6, 9}, so the
grid was extended in the same direction to depths 2 and 1 (stumps) before any 2025
forecast was made; selection is over {1, 2, 3, 6, 9}.

Usage: python xgboost_comparator.py dev          # depths 3, 6, 9
       python xgboost_comparator.py dev 2 1      # extension; appends to the same table
       python xgboost_comparator.py capacity 2   # training vs holdout check at the selected depth
       python xgboost_comparator.py test 2       # 2025 run at the selected depth
"""
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer

from rev_common import KEY, LEVELS, OUT, QCOLS, TEST_END, TEST_START, lean_features, model_table

WINDOW, LAG, REFIT, ES_DAYS = 364, 2, 7, 14
DEV_START, DEV_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")
DEPTHS = (3, 6, 9)
FEATURES = lean_features()
assert len(FEATURES) == 34

table = model_table()


def params(depth):
    return {"objective": "reg:quantileerror", "quantile_alpha": LEVELS, "tree_method": "hist",
            "learning_rate": 0.05, "max_depth": depth, "min_child_weight": 5, "colsample_bynode": 0.5,
            "subsample": 0.8, "seed": 42, "nthread": -1}


def pinball_mean(y, Q):
    u = y[:, None] - Q
    return float(np.mean(np.maximum(LEVELS * u, (LEVELS - 1) * u)))


def fit(day, depth):
    end = day - pd.Timedelta(days=LAG)
    start = end - pd.Timedelta(days=WINDOW - 1)
    es_start = end - pd.Timedelta(days=ES_DAYS - 1)
    win = table[table.settlement_date.between(start, end)]
    tr, va = win[win.settlement_date < es_start], win[win.settlement_date >= es_start]
    imputer = SimpleImputer(strategy="median").fit(win[FEATURES])     # as the QRF: whole window
    dtr = xgb.DMatrix(imputer.transform(tr[FEATURES]), label=tr.price.to_numpy())
    dva = xgb.DMatrix(imputer.transform(va[FEATURES]), label=va.price.to_numpy())
    booster = xgb.train(params(depth), dtr, num_boost_round=2000, evals=[(dva, "val")],
                        early_stopping_rounds=50, verbose_eval=False)
    info = {"cal_start": start, "cal_end": end, "train_rows": len(tr), "val_rows": len(va),
            "best_iteration": booster.best_iteration}
    return (imputer, booster), info, (tr, va)


def predict(fitted, rows):
    imputer, booster = fitted
    d = xgb.DMatrix(imputer.transform(rows[FEATURES]))
    return np.sort(booster.predict(d, iteration_range=(0, booster.best_iteration + 1)), axis=1)


def backtest(start, end, depth):
    days = sorted(table.loc[table.settlement_date.between(start, end), "settlement_date"].unique())
    fitted, frames, log, t0 = None, [], [], time.time()
    for i, day in enumerate(days):
        if fitted is None or i % REFIT == 0:
            fitted, info, _ = fit(day, depth)
            log.append({"refit_index": len(log), "test_day": day, **info})
            print(f"  depth {depth} refit {len(log) - 1:2d} {day:%Y-%m-%d} rounds {info['best_iteration'] + 1:4d} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rows = table[table.settlement_date == day]
        Q = predict(fitted, rows)
        fr = pd.DataFrame({"settlement_date": rows.settlement_date.to_numpy(), "hour": rows.hour.to_numpy(),
                           "actual_price": rows.price.to_numpy()})
        for j, c in enumerate(QCOLS):
            fr[c] = Q[:, j]
        frames.append(fr)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(log)


mode = sys.argv[1]
if mode == "dev":
    rows = []
    depths = [int(x) for x in sys.argv[2:]] or list(DEPTHS)
    for depth in depths:
        pred, log = backtest(DEV_START, DEV_END, depth)
        y, Q = pred.actual_price.to_numpy(), pred[QCOLS].to_numpy()
        rows.append({"max_depth": depth, "dev_mean_pinball": pinball_mean(y, Q),
                     "dev_cov90": float(((y >= Q[:, 0]) & (y <= Q[:, -1])).mean()),
                     "dev_mpil90": float((Q[:, -1] - Q[:, 0]).mean()),
                     "median_rounds": float(log.best_iteration.median() + 1), "n_hours": len(pred)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    path = OUT / "xgboost_development.csv"
    if path.exists():                                   # extension stage: keep the earlier depths
        old = pd.read_csv(path)
        res = pd.concat([old[~old.max_depth.isin(res.max_depth)], res]).sort_values("max_depth").reset_index(drop=True)
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    print("selected max_depth:", int(res.loc[res.dev_mean_pinball.idxmin(), "max_depth"]))
elif mode == "capacity":
    # training vs holdout at a mid-year refit and at the first refit, selected depth
    depth, out = int(sys.argv[2]), []
    for day in [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-02")]:
        fitted, info, (tr, va) = fit(day, depth)
        for name, part in [("train", tr), ("holdout", va)]:
            Q = predict(fitted, part)
            y = part.price.to_numpy()
            out.append({"refit_day": day, "part": name, "rows": len(part), "rounds": info["best_iteration"] + 1,
                        "mean_pinball": pinball_mean(y, Q),
                        "cov90": float(((y >= Q[:, 0]) & (y <= Q[:, -1])).mean()),
                        "mpil90": float((Q[:, -1] - Q[:, 0]).mean())})
    out = pd.DataFrame(out)
    out.to_csv(OUT / "xgboost_capacity.csv", index=False)
    print(out.round(3).to_string(index=False))
elif mode == "test":
    depth = int(sys.argv[2])
    pred, log = backtest(TEST_START, TEST_END, depth)
    d = OUT / "xgboost"
    d.mkdir(exist_ok=True)
    pred.to_csv(d / "predictions.csv", index=False)
    log.assign(max_depth=depth).to_csv(d / "refit_log.csv", index=False)
    print(f"saved {len(pred)} forecasts, {len(log)} refits")
