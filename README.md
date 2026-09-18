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

## Citation and licence

This software is released under the MIT licence. Citation metadata are provided
in `CITATION.cff`. A versioned Zenodo archive will be linked here after the
`v1.0.0` release.
