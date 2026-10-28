"""Chronology of the price episodes above 200 GBP/MWh relative to QRF's training support
(Section 5.2, paragraph following Table 6).

For each episode of coverage_episodes.csv: its hours above 200 GBP/MWh, the maximum
price in the training window of the QRF fit in force on each delivery day (QRF is refitted
weekly, so this is the 364-day window ending two days before the most recent refit), the
hours that exceeded that maximum and how many of them each model covered, and the
largest QRF 0.95 quantile in the episode. Counts only; no subgroup coverage rates.
Run coverage_bootstrap.py first.
"""
import numpy as np
import pandas as pd

from rev_common import MODELS, OUT, RQ1, covered90, load, model_table

table = model_table()
refits = pd.read_csv(RQ1 / "quantile_forest" / "refit_log.csv",
                     parse_dates=["test_day", "cal_start", "cal_end"]).sort_values("test_day")


def window_max(day):
    r = refits[refits.test_day <= day].iloc[-1]            # the refit in force on this day
    return table.loc[table.settlement_date.between(r.cal_start, r.cal_end), "price"].max()


f = load()
f["ts"] = f.settlement_date + pd.to_timedelta(f.hour, unit="h")
ep = pd.read_csv(OUT / "coverage_episodes.csv", parse_dates=["start", "end"])
ep = ep[ep.threshold == "> 200"]

rows = []
for _, r in ep.iterrows():
    sub = f[(f.ts >= r.start) & (f.ts <= r.end) & (f.actual_price > 200)].copy()
    sub["window_max"] = [window_max(d) for d in sub.settlement_date]
    beyond = (sub.actual_price > sub.window_max).to_numpy()
    row = {"episode": r.episode, "start": r.start, "hours_above_200": len(sub), "peak": sub.actual_price.max(),
           "window_max_min": sub.window_max.min(), "window_max_max": sub.window_max.max(),
           "hours_above_window_max": int(beyond.sum()), "qrf_q95_max": sub["QRF|q95"].max()}
    for m in MODELS:
        row[f"covered | {m}"] = int(covered90(sub, m).sum())
        row[f"covered beyond support | {m}"] = int(covered90(sub, m)[beyond].sum())
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(OUT / "episode_support.csv", index=False)

pd.set_option("display.width", 220)
print(out[["episode", "start", "hours_above_200", "peak", "window_max_min", "window_max_max",
           "hours_above_window_max", "qrf_q95_max", "covered | QRF", "covered | Encoder-Decoder LSTM"]]
      .to_string(index=False))
print("hours above the window maximum covered by any model:",
      int(np.sum([out[f"covered beyond support | {m}"].sum() for m in MODELS])))
