# Code map

All model-fitting code is under `modelling/`; `evaluation/` contains only
forecast assembly, metrics and statistical tests.

## Constituent forecast pipeline

Only executable sources needed by the paper are retained. Model-development
notebooks, rejected architectures and duplicated figure code are intentionally
outside this release.

| Stage | Entry point | Purpose |
|---|---|---|
| Point-in-time feature table | `modelling/data/prepare_features.py` | Builds the leakage-safe hourly modelling table |
| Historical, linear QR and QRF | `modelling/tabular/fit_tabular_models.py` | Fits the three RQ1 benchmarks and evaluates their common 11 paper quantiles |
| 2024-H2 QRF forecasts | `modelling/tabular/qrf_dev2024h2.py` | Supplies matched development forecasts for QRA |
| Encoder--Decoder LSTM | `modelling/lstm/development_backtest.py`, `modelling/lstm/test_backtest.py` | Fits development and held-out 2025 forecast arms |
| LSTM ensemble and metrics | `evaluation/assemble_results.py` | Combines seeds and quantile arms and evaluates the LSTM |
| Static and rolling CQR | `modelling/conformal/run_conformal.py` | Calibrates the reported prediction intervals |
| Two-model QRA | `modelling/qra/run_two_model_qra.py` | Fits the weekly LSTM--QRF combination |
| Constituent significance | `evaluation/constituent_dm_tests.py` | Computes daily-loss DM tests |
| Grouped ablations | `modelling/ablations/` | Runs the post-hoc QRF, linear-QR and LSTM ablations |

Each LSTM quantile is fitted by a separate network with pinball loss. The two
saved arms (five principal and six interior quantiles) jointly supply the eleven
levels. Three fixed seeds are averaged, then row-wise monotone rearrangement is
used to remove crossing.

The RQ1 forecast artefacts retain their original 15-level fitting grid,
including four additional extreme-tail levels. Cross-model tables and tests use
only the eleven levels common to every paper model.

## Manuscript extension

1. `modelling/lear/run_lear_qra_window.py` fits canonical 247-feature LEAR
   point forecasts at 273-, 364-, 546- and 728-day windows. The 273-day arm is a
   recorded numerical-stability sensitivity; the principal ensemble uses
   364/546/728 days.
2. `modelling/lear_qra/run_lear_qra_combine.py` estimates LEAR-QRA quantiles
   from those window forecasts under the common weekly, D-2 protocol.
3. `modelling/qra/run_three_way_qra_qw.py` combines the LSTM, QRF and
   LEAR-QRA forecasts. It also reproduces the two-model QRA as an internal check
   and fits the regularised comparison.
`modelling/lear/run_lear.py` contains the scaler, DST and input routines shared
by the multi-window implementation. The optional cross-check in
`modelling/lear/tests/` verifies those routines against epftoolbox.
