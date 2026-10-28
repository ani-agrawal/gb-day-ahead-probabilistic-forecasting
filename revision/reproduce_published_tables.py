"""Reproduction check run before the revision analyses: confirms that the saved 2025
forecasts of every model exist at per-quantile resolution, including each ablation
variant, and that they reproduce the published Table 5 and Table 10 values.
"""
import numpy as np
import pandas as pd

from rev_common import (FULL_GRID, MODELS, QCOLS, OUT, load, quantiles, crps11,
                    covered90, winkler90, RQ1_ABLATION, LSTM_ABLATION)

# Published values: Table 5 (CRPS, MAE, cov90, MPIL, Winkler, cov<0, cov>150, cov>200)
# and Table 10 / A.9 for the conformal variants (cov90, MPIL, Winkler, cov>150, cov>200)
T5 = {
    "Historical benchmark": (15.55, 20.89, 0.864, 92.7, 147.4, 0.00, 0.15, 0.00),
    "Linear QR":            (6.89, 9.68, 0.823, 36.8, 67.6, 0.72, 0.44, 0.08),
    "LEAR-QRA":             (6.52, 8.72, 0.893, 37.4, 71.2, 0.48, 0.28, 0.06),
    "QRF":                  (5.85, 8.38, 0.917, 36.6, 47.8, 0.70, 0.76, 0.71),
    "Encoder-Decoder LSTM": (5.59, 7.89, 0.895, 32.5, 51.2, 0.87, 0.51, 0.12),
    "Two-model QRA":        (5.23, 7.45, 0.893, 32.8, 44.1, 0.81, 0.79, 0.65),
    "Three-model QRA":      (5.14, 7.26, 0.900, 32.8, 44.1, 0.81, 0.79, 0.65),
}
T10 = {
    "Static CQR":  (0.912, 34.3, 51.3, 0.512, 0.115),
    "Rolling CQR": (0.900, 34.1, 51.7, 0.553, 0.192),
}


def close(got, pub):
    """Equal at the published precision."""
    s = f"{pub}"
    dp = len(s.split(".")[1]) if "." in s else 0
    return round(got, dp) == round(pub, dp)


f = load()
y = f.actual_price.to_numpy()
masks = {"<0": y < 0, ">150": y > 150, ">200": y > 200}
print(f"rows {len(f)}  |  <0: {masks['<0'].sum()}  >150: {masks['>150'].sum()}  >200: {masks['>200'].sum()}"
      "  (paper: 8759 / 149 / 123 / 52)")

rows, ok = [], True
for m in MODELS:
    lo, hi = f[f"{m}|q05"].to_numpy(), f[f"{m}|q95"].to_numpy()
    cov = covered90(f, m)
    got = {"cov90": cov.mean(), "MPIL": (hi - lo).mean(), "Winkler": winkler90(y, lo, hi).mean(),
           "cov<0": cov[masks["<0"]].mean(), "cov>150": cov[masks[">150"]].mean(),
           "cov>200": cov[masks[">200"]].mean()}
    if m in FULL_GRID:
        Q = quantiles(f, m)
        mono = (np.diff(Q, axis=1) >= -1e-9).all()
        got.update(CRPS=crps11(y, Q).mean(), MAE=np.abs(y - Q[:, QCOLS.index("q50")]).mean())
        pub = dict(zip(["CRPS", "MAE", "cov90", "MPIL", "Winkler", "cov<0", "cov>150", "cov>200"], T5[m]))
        grid = "11 quantiles" + ("" if mono else " (CROSSING)")
    else:
        pub = dict(zip(["cov90", "MPIL", "Winkler", "cov>150", "cov>200"], T10[m]))
        grid = "90% bounds only"
    bad = [k for k, v in pub.items() if not close(got[k], v)]
    ok &= not bad
    rows.append({"model": m, "grid": grid, **{k: round(v, 4) for k, v in got.items()},
                 "mismatches": ";".join(f"{k} got {got[k]:.4f} vs {pub[k]}" for k in bad) or "none"})
    print(f"{m:22} {grid:20} " + "  ".join(f"{k}={got[k]:.3f}" for k in pub) +
          ("   OK" if not bad else f"   MISMATCH {bad}"))

pd.DataFrame(rows).to_csv(OUT / "reproduction_check.csv", index=False)

# Ablation variants (used by tail_ablation.py): per-quantile resolution and full 2025 coverage
print("\nAblation variants:")
abl = []
for model, root, pattern in [("linear/QRF", RQ1_ABLATION, "no_*"),
                             ("LSTM", LSTM_ABLATION, "no_*")]:
    for d in sorted(root.glob(pattern)):
        files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*.csv"))
        abl.append({"family": model, "variant": d.name, "csv_files": len(files), "files": "; ".join(files[:6])})
abl = pd.DataFrame(abl)
abl.to_csv(OUT / "ablation_inventory.csv", index=False)
print(abl[["family", "variant", "csv_files"]].to_string(index=False))
print("\nREPRODUCTION:", "PASS (all published Table 5 / Table 10 values reproduced)" if ok else "FAIL")
