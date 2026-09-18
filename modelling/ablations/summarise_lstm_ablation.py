"""Summary table for the two-stage LSTM test-period ablation.

Computes, from raw predictions, each arm's mean 5-quantile pinball per
period as a percentage delta
against that seed's own headline arm (paired within seed), for all three
project seeds (42/7/101), then reports the seed-average delta with the
seed range. Post-hoc, interpretation-only. The no_price_history arm is a separate
class: it zeroes the encoder input, measuring the total value of the
information the first stage encodes rather than a like-for-like column drop.

Outputs -> 06_outputs/RQ2/model_runs/ablation_testperiod_twostage/ablation_summary_twostage.csv
           + copy in 06_outputs/final_results_tables/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TAUS = [0.05, 0.10, 0.50, 0.90, 0.95]
QC = ["q05", "q10", "q50", "q90", "q95"]
PERIODS = {"2025_primary": ("2025-01-01", "2025-12-31")}
ARMS = ["no_demand", "no_wind", "no_solar", "no_gas", "no_carbon",
        "no_opmr", "no_calendar", "no_price_history"]

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / "06_outputs/RQ2/model_runs/ablation_testperiod_twostage"
BASE = ROOT / "06_outputs/RQ2/model_runs/seq2seq_final_b16es14/5q_seed42/predictions.csv"


def mean_pinball(df):
    y = df["actual_price"].to_numpy()
    return float(np.mean([np.mean(np.where(y - df[q] >= 0, t * (y - df[q]),
                                           (t - 1) * (y - df[q])))
                          for q, t in zip(QC, TAUS)]))


SEEDS = [42, 7, 101]
HEAD = ROOT / "06_outputs/RQ2/model_runs/seq2seq_final_b16es14"
base_mp = {}
for s_ in SEEDS:
    b = pd.read_csv(HEAD / f"5q_seed{s_}/predictions.csv", parse_dates=["settlement_date"])
    base_mp[s_] = {per: mean_pinball(b[(b.settlement_date >= a) & (b.settlement_date <= b_)])
                   for per, (a, b_) in PERIODS.items()}

rows = []
for arm in ARMS:
    for s_ in SEEDS:
        d = ABL / (arm if s_ == 42 else f"{arm}_seed{s_}")
        df = pd.read_csv(d / "predictions.csv", parse_dates=["settlement_date"])
        for per, (a, b_) in PERIODS.items():
            sub = df[(df.settlement_date >= a) & (df.settlement_date <= b_)]
            m = mean_pinball(sub)
            rows.append({"arm": arm, "seed": s_, "period": per,
                         "mean_pinball_5q": round(m, 3),
                         "delta_pct": round(100 * (m - base_mp[s_][per]) / base_mp[s_][per], 1)})

per_seed = pd.DataFrame(rows)
agg = (per_seed.groupby(["arm", "period"])["delta_pct"]
       .agg(delta_mean="mean", delta_min="min", delta_max="max").round(1).reset_index())
out = per_seed.merge(agg, on=["arm", "period"])
out.to_csv(ABL / "ablation_summary_twostage.csv", index=False)
out.to_csv(ROOT / "06_outputs/final_results_tables/ablation_summary_twostage.csv", index=False)
piv = agg.pivot(index="arm", columns="period", values="delta_mean").loc[ARMS][["2025_primary"]]
print("Encoder-Decoder LSTM ablation, THREE-SEED AVERAGE delta % (2025):")
print(piv.to_string())
rng = agg[agg.period == "2025_primary"].set_index("arm").loc[ARMS]
print("\n2025 seed ranges:")
for a_ in ARMS:
    r = rng.loc[a_]
    print(f"  {a_:<18} {r.delta_mean:+.1f}%  [{r.delta_min:+.1f}, {r.delta_max:+.1f}]")
