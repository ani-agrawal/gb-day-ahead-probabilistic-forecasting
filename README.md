# Probabilistic day-ahead electricity-price forecasting in Great Britain

This repository contains the modelling and evaluation code accompanying the
paper on probabilistic forecasts of hourly EPEX GB day-ahead prices. It covers
the historical and linear benchmarks, quantile regression forest (QRF),
Encoder--Decoder LSTM, conformal calibration, multi-window LEAR-QRA, and the
two- and three-model Quantile Regression Averaging (QRA) combinations.

The release is code-only. The licensed EPEX SPOT target cannot be redistributed,
and hourly forecast artefacts are omitted because they repeat the realised target.
Public inputs must be retrieved from their original providers. See
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) and
[`docs/INPUT_SCHEMA.md`](docs/INPUT_SCHEMA.md).

## Repository map

| Path | Purpose |
|---|---|
| `modelling/data/` | Point-in-time feature-table construction |
| `modelling/tabular/` | Historical, linear-QR and QRF benchmarks |
| `modelling/lstm/` | Encoder--Decoder LSTM development and 2025 backtests |
| `modelling/conformal/` | Static and rolling conformal calibration |
| `modelling/lear/` | Verified epftoolbox-equivalent LEAR engine and multi-window forecasts |
| `modelling/lear_qra/` | LEAR-QRA probabilistic post-processing |
| `modelling/qra/` | Two- and three-model QRA and regularised comparison |
| `modelling/ablations/` | Grouped QRF, linear-QR and LSTM ablations |
| `evaluation/` | Forecast assembly, paper metrics and constituent DM tests |
| `pipeline/` | Orchestration, local staging and integrity checks |
| `reference_results/` | Aggregate manuscript results that do not disclose licensed prices |

The relationship between the modelling stages and their outputs is described in
[`docs/CODE_MAP.md`](docs/CODE_MAP.md).

## Environment

The results of record used Python 3.13 and the pinned packages in
`requirements.txt`. Either create the Conda environment:

```bash
conda env create -f environment.yml
conda activate gb-probabilistic-epf
```

or install into a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
```

TensorFlow results can vary slightly across operating systems and hardware even
with fixed seeds. LEAR, QRA and metric calculations are deterministic under the
pinned environment.

## Reproduction sequence

1. Obtain the licensed EPEX target and download the public inputs listed in
   `docs/DATA_ACCESS.md`. Place them under `04_data/raw/` using the documented
   names.
2. Inspect the expected schema and point-in-time boundary:

   ```bash
   python run_pipeline.py preflight
   ```

3. Reproduce the paper's constituent forecasts, conformal analysis and
   two-model QRA:

   ```bash
   python run_pipeline.py constituents --rebuild-data
   ```

4. Build the manuscript additions (multi-window LEAR-QRA and three-model QRA):

   ```bash
   python run_pipeline.py publication-models
   ```

5. Recreate the constituent-model significance tests:

   ```bash
   python run_pipeline.py evaluation
   ```

`full-paper` runs steps 3--5 in order. The post-hoc ablation study is deliberately
separate because it is substantially longer: `python run_pipeline.py ablations`.
The complete neural-network reproduction
is an overnight CPU job; reruns are resumable where the production scripts have
already written complete forecast arms.

The publication-model stage first creates a local, ignored snapshot under
`modelling/input_snapshot/`. This prevents the LEAR/QRA code from depending
on paths elsewhere on the author's machine while keeping restricted inputs out
of version control.

## Results that can be checked without licensed data

`reference_results/` contains aggregate LEAR-QRA and three-model QRA tables.
They allow the reported manuscript values and significance statements to be
checked without publishing hourly realised prices. Full numerical reproduction
requires lawful access to the target series.

## Information boundary

For delivery day D, all estimated models use a common forecast-time boundary at
the 09:20 D-1 auction gate. The newest complete realised-price and outturn day
available to training is D-2. Rolling-origin refits and QRA estimation use only
forecast--outcome pairs eligible by that cutoff.

## Citation, archive and licence

This software is released under the MIT licence; see [`LICENSE`](LICENSE).
Machine-readable citation metadata are provided in
[`CITATION.cff`](CITATION.cff). Aidan O'Sullivan contributed methodology and
supervision and should be recorded as a supervisory contributor in the archive
metadata rather than as a software creator.

The public repository URL and Zenodo DOI will be added after the `v1.0.0`
release has been archived. The exact deposit sequence is documented in
[`docs/ZENODO_DEPOSIT.md`](docs/ZENODO_DEPOSIT.md), and the remaining checks are
listed in [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).
