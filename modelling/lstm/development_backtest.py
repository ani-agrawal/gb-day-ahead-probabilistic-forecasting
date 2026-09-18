"""Encoder--Decoder LSTM development backtest over 2024-H2.

An encoder reads the previous day's 24 cleared prices. Its final hidden and
cell states initialise a decoder that processes the 24 delivery hours' known
inputs. A time-distributed Dense(1) projection supplies one quantile per hour.
Each quantile has its own network and pinball loss. The reported configuration
uses 64 units, batch size 16, a 14-day early-stopping holdout, weekly refits,
728-day windows ending D-2, and seeds 42, 7 and 101.

Usage: python rq2_seq2seq_encdec.py <units> <seed>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

UNITS, SEED = int(sys.argv[1]), int(sys.argv[2])
BATCH = int(os.environ.get("BATCH", 32))
ES_DAYS = int(os.environ.get("ES_DAYS", 28))
RUN_TAG = os.environ.get("RUN_TAG", "")
ASINH = os.environ.get("TARGET", "") == "asinh"  # variance-stabilising transform; quantiles invert exactly (monotone)
SYSTEM = os.environ.get("SYSTEM", "5q")

KNOWN = [
    "demand_forecast", "transmission_wind_forecast",
    "embedded_wind_forecast", "embedded_solar_forecast",
    "gas_price", "carbon_price", "opmr_surplus_norm",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek",
    "is_bank_holiday",
]
if SYSTEM == "5q":
    TAUS_5 = [0.05, 0.10, 0.50, 0.90, 0.95]
    Q_COLS = ["q05", "q10", "q50", "q90", "q95"]
elif SYSTEM == "fill6":
    TAUS_5 = [0.20, 0.30, 0.40, 0.60, 0.70, 0.80]
    Q_COLS = ["q20", "q30", "q40", "q60", "q70", "q80"]
else:
    raise ValueError(f"Unknown SYSTEM={SYSTEM!r}; expected '5q' or 'fill6'")
DROPOUT = 0.1
CAL_WINDOW_DAYS = 728
TARGET_LAG_DAYS = 2
REFIT_EVERY = 7
ES_HOLDOUT_DAYS = ES_DAYS
TEST_START = pd.Timestamp("2024-07-01")
TEST_END = pd.Timestamp("2024-12-31")

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (ROOT / "06_outputs" / "RQ2" / "model_runs" / "rnn_dev" / "seq2seq_gate_v2"
           / f"encdec_u{UNITS}{RUN_TAG}_seed{SEED}")
if (RUN_DIR / "predictions.csv").exists():
    print(f"SKIP encdec u{UNITS} seed {SEED}")
    sys.exit(0)
RUN_DIR.mkdir(parents=True, exist_ok=True)

tbl = pd.read_parquet(ROOT / "04_data" / "processed" / "model_table_hourly_epex.parquet")
tbl["settlement_date"] = pd.to_datetime(tbl["settlement_date"])
tbl = tbl.sort_values(["settlement_date", "hour"]).reset_index(drop=True)
price_wide = (tbl.pivot_table(index="settlement_date", columns="hour",
                              values="price", aggfunc="first").sort_index()
              .interpolate(axis=1, limit=1, limit_direction="forward"))
known_by_day = {d: (g.set_index("hour")[KNOWN].reindex(range(24))
                    .interpolate(limit_direction="both").to_numpy(dtype="float32"))
                for d, g in tbl.groupby("settlement_date")}
test_days = sorted(tbl.loc[(tbl["settlement_date"] >= TEST_START)
                           & (tbl["settlement_date"] <= TEST_END), "settlement_date"].unique())
print(f"TF {tf.__version__}; encdec u{UNITS}; {SYSTEM}; seed {SEED}; (24,1)+(24,{len(KNOWN)}) inputs; "
      f"{len(test_days)} dev days", flush=True)


def day_sample(day: pd.Timestamp):
    hist = price_wide.reindex([day - pd.Timedelta(days=1)]).to_numpy(dtype="float32").reshape(24, 1)
    fut = known_by_day.get(pd.Timestamp(day))
    if fut is None:
        fut = np.full((24, len(KNOWN)), np.nan, dtype="float32")
    rows = tbl[tbl["settlement_date"] == day].set_index("hour")
    y = rows["price"].reindex(range(24)).to_numpy(dtype="float32")
    if ASINH:
        hist = np.arcsinh(hist)
        y = np.arcsinh(y)
    return hist, fut, y, rows


def training_arrays(day_start, day_end):
    Hs, Fs, Ys = [], [], []
    for day in price_wide.index[(price_wide.index >= day_start) & (price_wide.index <= day_end)]:
        h, f, y, _ = day_sample(day)
        if np.isnan(h).any() or np.isnan(f).any() or np.isnan(y).any():
            continue
        Hs.append(h); Fs.append(f); Ys.append(y)
    return np.stack(Hs), np.stack(Fs), np.stack(Ys)


def make_loss(tau: float):
    t = tf.constant(tau, dtype=tf.float32)

    def pinball(y_true, y_pred):
        err = y_true - y_pred if y_pred.shape.rank == y_true.shape.rank else y_true - y_pred[..., 0]
        return tf.reduce_mean(tf.maximum(t * err, (t - 1.0) * err))

    return pinball


def build_model(tau: float):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    hist_in = keras.Input(shape=(24, 1))
    fut_in = keras.Input(shape=(24, len(KNOWN)))
    _, h, c = layers.LSTM(UNITS, return_state=True)(hist_in)   # encoder
    dec = layers.LSTM(UNITS, return_sequences=True)(fut_in, initial_state=[h, c])
    dec = layers.Dropout(DROPOUT)(dec)
    out = layers.Dense(1)(dec)
    model = keras.Model([hist_in, fut_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=make_loss(tau))
    return model


fitted, scalers = None, None
refit_log, day_frames = [], []
loss_rows = []
for i, day in enumerate(test_days):
    if fitted is None or i % REFIT_EVERY == 0:
        cal_end = day - pd.Timedelta(days=TARGET_LAG_DAYS)
        cal_start = cal_end - pd.Timedelta(days=CAL_WINDOW_DAYS - 1)
        es_start = cal_end - pd.Timedelta(days=ES_HOLDOUT_DAYS - 1)
        Xh_tr, Xf_tr, y_tr = training_arrays(cal_start, es_start - pd.Timedelta(days=1))
        Xh_es, Xf_es, y_es = training_arrays(es_start, cal_end)
        h_mu, h_sd = Xh_tr.mean(), Xh_tr.std() + 1e-9
        f_mu = Xf_tr.mean(axis=(0, 1))
        f_sd = Xf_tr.std(axis=(0, 1)) + 1e-9
        y_mu, y_sd = y_tr.mean(), y_tr.std()
        scalers = (h_mu, h_sd, f_mu, f_sd, y_mu, y_sd)
        tr_x = [(Xh_tr - h_mu) / h_sd, (Xf_tr - f_mu) / f_sd]
        es_x = [(Xh_es - h_mu) / h_sd, (Xf_es - f_mu) / f_sd]
        fitted, entry = {}, {"refit_index": len(refit_log), "test_day": day,
                             "cal_start": cal_start, "cal_end": cal_end}
        for tau, col in zip(TAUS_5, Q_COLS):
            m = build_model(tau)
            es_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                  restore_best_weights=True)
            hst = m.fit(tr_x, (y_tr - y_mu) / y_sd,
                  validation_data=(es_x, (y_es - y_mu) / y_sd),
                  epochs=200, batch_size=BATCH, callbacks=[es_cb], verbose=0)
            fitted[col] = m
            entry[f"epochs_{col}"] = len(hst.history["loss"])
            entry[f"clean_train_{col}"] = round(float(m.evaluate(tr_x, (y_tr - y_mu) / y_sd, verbose=0)), 4)
            entry[f"clean_val_{col}"] = round(float(m.evaluate(es_x, (y_es - y_mu) / y_sd, verbose=0)), 4)
            for ep, (tl, vl) in enumerate(zip(hst.history["loss"], hst.history["val_loss"])):
                loss_rows.append({"refit_index": len(refit_log), "tau": tau,
                                  "epoch": ep, "loss": tl, "val_loss": vl})
        refit_log.append(entry)
        print(f"refit {len(refit_log) - 1} at {day:%Y-%m-%d}", flush=True)

    h_mu, h_sd, f_mu, f_sd, y_mu, y_sd = scalers
    hist, fut, _, rows = day_sample(day)
    x = [((hist - h_mu) / h_sd)[None], ((fut - f_mu) / f_sd)[None]]
    q = np.column_stack([fitted[c].predict(x, verbose=0)[0, :, 0] for c in Q_COLS])
    q = q * y_sd + y_mu
    if ASINH:
        q = np.sinh(q)  # exact quantile inversion back to GBP/MWh
    q = np.sort(q, axis=-1)
    real_hours = rows.index.to_numpy()
    frame = pd.DataFrame({"settlement_date": day, "hour": real_hours,
                          "actual_price": rows["price"].values})
    for j, c in enumerate(Q_COLS):
        frame[c] = q[real_hours, j]
    day_frames.append(frame)

preds = pd.concat(day_frames, ignore_index=True)
preds.to_csv(RUN_DIR / "predictions.csv", index=False)
pd.DataFrame(refit_log).to_csv(RUN_DIR / "refit_log.csv", index=False)
pd.DataFrame(loss_rows).to_csv(RUN_DIR / "loss_history.csv", index=False)

a = preds["actual_price"].values
metrics = {"arm": f"encdec_u{UNITS}{RUN_TAG}", "seed": SEED}
for c, t in zip(Q_COLS, TAUS_5):
    err = a - preds[c]
    metrics[f"pin{c[1:]}"] = round(float(np.mean(np.where(err >= 0, t * err, (t - 1) * err))), 3)
metrics["mean_pinball_5q"] = round(float(np.mean([metrics[f"pin{c[1:]}"] for c in Q_COLS])), 3)
if "q05" in preds.columns and "q95" in preds.columns:  # outer band absent on fill6 runs
    in90 = (a >= preds["q05"]) & (a <= preds["q95"])
    metrics["cov_90"] = round(float(in90.mean()), 3)
    metrics["width_90"] = round(float((preds["q95"] - preds["q05"]).mean()), 1)
pd.DataFrame([metrics]).to_csv(RUN_DIR / "metrics.csv", index=False)
print(f"encdec u{UNITS} seed {SEED} dev result:", metrics, flush=True)
