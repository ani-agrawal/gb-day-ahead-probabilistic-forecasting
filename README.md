# Probabilistic day-ahead electricity-price forecasting in Great Britain

This repository contains the modelling and evaluation code accompanying the
paper on probabilistic forecasts of hourly EPEX GB day-ahead prices. It includes
the QRF, Encoder--Decoder LSTM, conformal, LEAR-QRA and forecast-combination
experiments.

The licensed EPEX SPOT target cannot be redistributed. Instructions for
obtaining and arranging the required inputs are provided in
[Data access](docs/DATA_ACCESS.md) and [Input schema](docs/INPUT_SCHEMA.md).

## Reproduction

Create the environment using `environment.yml` or `requirements.txt`, then run:

```bash
python run_pipeline.py preflight
python run_pipeline.py full-paper --rebuild-data
```

The model sequence is documented in [Code map](docs/CODE_MAP.md).

## Revision analyses

Version 1.1.0 adds the analyses introduced in revision: bootstrap intervals for
conditional coverage, tail-specific tests, threshold-weighted CRPS, forecast-time
regimes, a feature-matched QRF, an XGBoost comparator, quantile-crossing statistics and
conformal recalibration of the combination. They are in `revision/`, with run order and
the paper item each script produces in [revision/README.md](revision/README.md). This
version also adds `04_data/metadata/lean_feature_list_hourly.csv`, the 34-feature
specification read by the tabular models.

## Citation and licence

This software is released under the MIT licence. Citation metadata are provided
in `CITATION.cff`.
