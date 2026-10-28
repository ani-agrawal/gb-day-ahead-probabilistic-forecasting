"""Tail-specific significance testing (Section 4.2; Tables A.14 and A.15).

1. Diebold-Mariano tests on daily total pinball loss at tau = 0.90 and 0.95,
   separately, for the nine pairs the paper tests on daily CRPS; two-sided.
2. Model Confidence Set (Hansen, Lunde and Nason, 2011) over the seven Table 5
   models, on daily CRPS and on daily upper-tail pinball loss (the daily sum of
   pinball loss at tau = 0.90 and 0.95).
Check: the DM implementation must reproduce the nine published one-sided CRPS
p-values before anything new is reported.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from rev_common import FULL_GRID, LEVELS, OUT, crps11, load, pinball, quantiles

f = load()
y = f.actual_price.to_numpy()
day = f.settlement_date.to_numpy()
T_DAYS = 365
HAC_LAG = int(np.floor(1.5 * T_DAYS ** (1 / 3)))          # Section 4.2
assert HAC_LAG == 10


def daily(x):
    return pd.Series(x).groupby(day).sum().to_numpy()


losses = {"CRPS": {}, "pinball 0.90": {}, "pinball 0.95": {}, "upper-tail pinball": {}}
i90, i95 = list(LEVELS).index(0.90), list(LEVELS).index(0.95)
for m in FULL_GRID:
    Q = quantiles(f, m)
    losses["CRPS"][m] = daily(crps11(y, Q))
    p90, p95 = pinball(y, Q[:, i90], 0.90), pinball(y, Q[:, i95], 0.95)
    losses["pinball 0.90"][m] = daily(p90)
    losses["pinball 0.95"][m] = daily(p95)
    losses["upper-tail pinball"][m] = daily(p90 + p95)


def dm(la, lb, lag=HAC_LAG):
    """DM statistic on d = la - lb with Newey-West (Bartlett) long-run variance.
    Autocovariances are averaged over their T - k overlapping terms, as in the
    code that produced the published p-values."""
    d = la - lb
    T = len(d)
    u = d - d.mean()
    lrv = np.mean(u * u)
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.mean(u[k:] * u[:-k])
    return d.mean() / np.sqrt(lrv / T)


# Pairs and directions as published: model A has lower loss than model B
PAIRS = [("QRF", "Historical benchmark", "< 0.0001"), ("QRF", "Linear QR", "0.0079"),
         ("Encoder-Decoder LSTM", "QRF", "0.1949"), ("Two-model QRA", "Encoder-Decoder LSTM", "0.0408"),
         ("Two-model QRA", "QRF", "< 0.0001"), ("Three-model QRA", "Two-model QRA", "0.0050"),
         ("Three-model QRA", "Encoder-Decoder LSTM", "0.0111"), ("Three-model QRA", "QRF", "< 0.0001"),
         ("Three-model QRA", "LEAR-QRA", "< 0.0001")]


def fmt(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


ok = True
for a, b, pub in PAIRS:
    got = fmt(norm.cdf(dm(losses["CRPS"][a], losses["CRPS"][b])))
    ok &= got == pub
    print(f"gate  {a:22} < {b:22} one-sided p {got:>9}  published {pub:>9}  {'OK' if got == pub else 'MISMATCH'}")
assert ok, "DM implementation does not reproduce the published p-values"

rows = []
for tau in ("pinball 0.90", "pinball 0.95"):
    for a, b, _ in PAIRS:
        la, lb = losses[tau][a], losses[tau][b]
        s = dm(la, lb)
        rows.append({"loss": tau, "model_a": a, "model_b": b, "mean_daily_diff": (la - lb).mean(),
                     "mean_hourly_a": la.sum() / len(y), "mean_hourly_b": lb.sum() / len(y),
                     "dm_stat": s, "p_two_sided": 2 * norm.sf(abs(s)),
                     "lower_loss": a if s < 0 else b})
tail = pd.DataFrame(rows)
tail.to_csv(OUT / "tail_dm_tests.csv", index=False)

# ---------------- Model Confidence Set, T_max statistic, moving-block bootstrap
B, BLOCK, SEED = 10_000, HAC_LAG, 20261025


def block_indices(T, block, B, rng):
    n = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(B, n))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T     # circular blocks
    return idx.reshape(B, -1)[:, :T]


def mcs(L, names, rng):
    """Hansen, Lunde and Nason (2011) MCS with the T_max statistic.
    L: T x m daily losses. Returns {model: MCS p-value}."""
    T = L.shape[0]
    idx = block_indices(T, BLOCK, B, rng)
    live = list(range(L.shape[1]))
    pvals, running = {}, 0.0
    while len(live) > 1:
        Lm = L[:, live]
        d = Lm - Lm.mean(axis=1, keepdims=True)                 # loss relative to the set average
        dbar = d.mean(axis=0)
        dbar_b = d[idx].mean(axis=1)                            # B x m bootstrap means
        var = ((dbar_b - dbar) ** 2).mean(axis=0)
        t = dbar / np.sqrt(var)
        t_b = ((dbar_b - dbar) / np.sqrt(var)).max(axis=1)
        p = (t_b >= t.max()).mean()
        running = max(running, p)
        worst = live[int(np.argmax(t))]
        pvals[names[worst]] = running
        live.remove(worst)
    pvals[names[live[0]]] = 1.0
    return pvals


rng = np.random.default_rng(SEED)
mrows = []
for loss in ("CRPS", "upper-tail pinball"):
    L = np.column_stack([losses[loss][m] for m in FULL_GRID])
    pv = mcs(L, FULL_GRID, rng)
    for m in FULL_GRID:
        mrows.append({"loss": loss, "model": m, "mean_hourly_loss": losses[loss][m].sum() / len(y),
                      "mcs_p": pv[m], "in_90pct_mcs": pv[m] >= 0.10, "in_75pct_mcs": pv[m] >= 0.25})
mcs_df = pd.DataFrame(mrows)
mcs_df.to_csv(OUT / "model_confidence_set.csv", index=False)

pd.set_option("display.width", 200)
print("\nTail DM (two-sided):")
print(tail[["loss", "model_a", "model_b", "mean_hourly_a", "mean_hourly_b", "dm_stat", "p_two_sided", "lower_loss"]]
      .round(4).to_string(index=False))
print("\nMCS (T_max, block 10, B = 10,000):")
print(mcs_df.round(4).to_string(index=False))
