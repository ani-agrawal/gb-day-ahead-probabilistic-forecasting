# Revision analyses

Scripts for the analyses added to the paper in revision. They reuse the saved forecasts
of the constituent pipeline and refit models only where stated. Shared paths, the
forecast loader and the scoring functions of Equations (1), (2) and (4) are in
`rev_common.py`.

## Prerequisites

1. Run the constituent pipeline (`python run_pipeline.py full-paper --rebuild-data`) and
   create the restricted snapshot (`python pipeline/prepare_publication_snapshot.py`),
   so that `06_outputs/`, `modelling/input_snapshot/`, `modelling/lear_qra/outputs/` and
   `modelling/qra/outputs/` are populated.
2. Use the environment in `environment.yml` (it adds `xgboost==3.2.0` for the XGBoost
   comparator).
3. Run every script from this folder (`cd revision`). Outputs are written to
   `revision/outputs/`.

## Run order and paper items

| Step | Script | Paper item |
|---|---|---|
| 1 | `reproduce_published_tables.py` | Check: saved forecasts reproduce Tables 5 and 10 |
| 2 | `coverage_bootstrap.py` | Section 5.2; Tables 6, A.9 and A.13 |
| 3 | `tail_tests.py` | Sections 4.2 and 5.2; Tables A.14 and A.15 |
| 4 | `twcrps.py` | Threshold-weighted CRPS columns of Table 5 |
| 5 | `forecast_time_regimes.py` | Section 5.2.1; Tables A.16 and A.17 |
| 6 | `episode_support.py` | Section 5.2, episode chronology (after step 2) |
| 7 | `feature_matched_qrf.py gate`, then `feature_matched_qrf.py matched` | Section 5.2.2 |
| 8 | `feature_matched_score.py` | Tables 8 and A.19 (after step 7) |
| 9 | `xgboost_comparator.py dev`, `dev 2 1`, `capacity 2`, `test 2` | Section 4.3; Table 4 row; depth row of Table A.21 |
| 10 | `xgboost_score.py` | Section 5.2.3; Tables 9 and A.20 (after step 9) |
| 11 | `tail_ablation.py` | Section 5.5; Table A.18 |
| 12 | `quantile_crossing.py lstm`, `qrf`, `qra`, `lear_qra`, `linear`, `xgboost` | Section 4.3; Table A.23 (the `xgboost` mode after step 9) |
| 13 | `conformal_on_qra.py` | Sections 4.5 and 5.4; Table A.24 |
| 14 | `qra_development_overlap.py` | Section 4.6, development-period overlap |
| 15 | `regime_window_counts.py` | Section 6.5, regime sparsity in the rolling calibration window |

Several scripts check their inputs before reporting anything: step 1 reproduces the
published values, step 3 reproduces the published Diebold-Mariano p-values, step 7
reproduces the headline QRF forecasts before the matched run, step 11 reproduces Figure
10, and steps 12 and 13 reproduce the reported forecasts of every model they rerun.

The bootstrap scripts use fixed seeds (20261025 for the coverage intervals, the tail
tests, the feature-matched, XGBoost and conformal comparisons; 20261026 for the
forecast-time regimes), so their intervals are reproducible exactly.

## Items not produced by these scripts

* The candidate values in Table A.21 other than the XGBoost depth come from the
  development comparisons of the original pipeline (Section 4 and Table 4).
* Table A.22 reports approximate computational costs taken from the timestamps of the
  original runs; nothing was rerun to time it.
