"""Tail-restricted ablation (Section 5.5; Table A.18).

Recomputes the grouped ablation of Figure 10 on upper-tail pinball loss instead of
the five-quantile mean. Upper-tail pinball loss is the sum of the pinball losses at
tau = 0.90 and 0.95, the same definition as the tail Model Confidence Set. No model is
refitted: every ablation arm saved its 2025 forecasts at per-quantile resolution.

Paired as in Figure 10: each arm against its own complete model (linear QR and QRF
against the headline tabular runs; each LSTM seed against that seed's headline run);
LSTM values are the mean of the within-seed changes over seeds 42, 7 and 101, with the
seed range reported. The five-quantile column is recomputed first and must reproduce
the published Figure 10 values before the tail column is used.
"""
import numpy as np
import pandas as pd

from rev_common import KEY, LSTM_ABLATION, LSTM_RUN, OUT, RQ1, RQ1_ABLATION, TEST_END, TEST_START

ARMS = ["no_wind", "no_demand", "no_solar", "no_gas", "no_carbon", "no_price_history", "no_calendar", "no_opmr"]
SEEDS = [42, 7, 101]
FIVE = {"q05": 0.05, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q95": 0.95}
TAIL = {"q90": 0.90, "q95": 0.95}

# Figure 10 / Section 5.5 as published (%); None where the text gives no number
# ("approximately zero", "no measurable effect", "close to zero")
PUBLISHED = {
    ("Linear QR", "no_wind"): 26.9, ("QRF", "no_wind"): 38.1, ("LSTM", "no_wind"): 18.7,
    ("Linear QR", "no_demand"): 15.0, ("QRF", "no_demand"): 19.4, ("LSTM", "no_demand"): 5.2,
    ("Linear QR", "no_gas"): 10.6, ("QRF", "no_gas"): 1.9, ("LSTM", "no_gas"): 20.0,
    ("LSTM", "no_carbon"): 6.8, ("QRF", "no_price_history"): 6.2, ("LSTM", "no_price_history"): 6.3,
    ("Linear QR", "no_solar"): 9.3, ("QRF", "no_solar"): 4.0,
}


def read(path):
    d = pd.read_csv(path, parse_dates=["settlement_date"])
    d = d[(d.settlement_date >= TEST_START) & (d.settlement_date <= TEST_END)]
    return d.sort_values(KEY).reset_index(drop=True)


def loss(d, levels):
    """Mean over hours of the summed pinball losses at the given levels."""
    y = d.actual_price.to_numpy()
    tot = 0.0
    for c, t in levels.items():
        e = y - d[c].to_numpy()
        tot = tot + np.where(e >= 0, t * e, (t - 1) * e)
    return float(np.mean(tot))


def scores(d):
    return {"five_q": loss(d, FIVE) / len(FIVE), "upper_tail": loss(d, TAIL),
            "pin90": loss(d, {"q90": 0.90}), "pin95": loss(d, {"q95": 0.95})}


def same_grid(a, b):
    assert len(a) == len(b) == 8759
    assert (a[KEY].values == b[KEY].values).all() and np.allclose(a.actual_price, b.actual_price)


rows = []
# ---- tabular models: one run per arm against the headline run
for model, sub in [("Linear QR", "linear"), ("QRF", "qrf")]:
    head = read(RQ1 / {"linear": "linear_qr", "qrf": "quantile_forest"}[sub]
                / "predictions.csv")
    base = scores(head)
    for arm in ARMS:
        a = read(RQ1_ABLATION / arm / sub / "predictions.csv")
        same_grid(a, head)
        s = scores(a)
        rows.append({"model": model, "arm": arm, "seed": None,
                     **{f"base_{k}": v for k, v in base.items()}, **{f"arm_{k}": v for k, v in s.items()},
                     **{f"pct_{k}": 100 * (s[k] - base[k]) / base[k] for k in s}})
# ---- LSTM: within-seed changes for seeds 42, 7, 101
for seed in SEEDS:
    head = read(LSTM_RUN / f"5q_seed{seed}" / "predictions.csv")
    base = scores(head)
    for arm in ARMS:
        a = read(LSTM_ABLATION / (arm if seed == 42 else f"{arm}_seed{seed}")
                 / "predictions.csv")
        same_grid(a, head)
        s = scores(a)
        rows.append({"model": "LSTM", "arm": arm, "seed": seed,
                     **{f"base_{k}": v for k, v in base.items()}, **{f"arm_{k}": v for k, v in s.items()},
                     **{f"pct_{k}": 100 * (s[k] - base[k]) / base[k] for k in s}})

per_run = pd.DataFrame(rows)
per_run.to_csv(OUT / "tail_ablation_per_run.csv", index=False)

pct = [c for c in per_run if c.startswith("pct_")]
# Figure 10 averaged the per-seed changes after rounding each to 0.1
# (twostage_ablation_summary.py); the same order is kept so the columns match it
summ = per_run.assign(**{k: per_run[k].round(1) for k in pct}).groupby(["model", "arm"], sort=False)[pct] \
    .mean().reset_index()
rng = per_run[per_run.model == "LSTM"].groupby("arm")[pct].agg(["min", "max"])
for k in pct:
    summ[f"{k}_seed_min"] = summ.arm.map(rng[(k, "min")]).where(summ.model == "LSTM")
    summ[f"{k}_seed_max"] = summ.arm.map(rng[(k, "max")]).where(summ.model == "LSTM")
# rank of each arm within model, on each measure (1 = largest deterioration)
for k in ["pct_five_q", "pct_upper_tail"]:
    summ[f"rank_{k[4:]}"] = summ.groupby("model")[k].rank(ascending=False).astype(int)
summ.to_csv(OUT / "tail_ablation.csv", index=False)

# ---- gate: the five-quantile column must reproduce Figure 10
ok = True
for (m, arm), pub in PUBLISHED.items():
    got = summ[(summ.model == m) & (summ.arm == arm)].pct_five_q.iloc[0]
    if round(got, 1) != pub:
        ok, _ = False, print(f"MISMATCH {m} {arm}: {got:.2f} vs published {pub}")
print("Figure 10 reproduction:", "PASS" if ok else "FAIL")

pd.set_option("display.width", 220)
for m in ["Linear QR", "QRF", "LSTM"]:
    s = summ[summ.model == m]
    print(f"\n== {m}   (complete-model upper-tail loss "
          f"{per_run[per_run.model == m].base_upper_tail.mean():.3f}, five-q {per_run[per_run.model == m].base_five_q.mean():.3f})")
    for _, r in s.iterrows():
        sr = (f"  seeds 5q [{r.pct_five_q_seed_min:+.1f}, {r.pct_five_q_seed_max:+.1f}]"
              f" tail [{r.pct_upper_tail_seed_min:+.1f}, {r.pct_upper_tail_seed_max:+.1f}]") if m == "LSTM" else ""
        print(f"   {r.arm:17} five-q {r.pct_five_q:+6.1f}% (#{r.rank_five_q})   upper-tail {r.pct_upper_tail:+6.1f}% "
              f"(#{r.rank_upper_tail})   pin90 {r.pct_pin90:+6.1f}%  pin95 {r.pct_pin95:+6.1f}%{sr}")
