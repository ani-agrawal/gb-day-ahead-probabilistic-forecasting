# Data access and redistribution

The public repository does not contain the data. This is deliberate: the target
series is licensed EPEX SPOT GB day-ahead auction data, and several derived
forecast files contain the same realised prices. Possession of this code does
not grant a right to redistribute any third-party source.

## Sources used

| Provider | Inputs | URL | Final retrieval/access date recorded for the paper |
|---|---|---|---|
| Elexon BMRS | Balancing prices, generation outturns and REMIT records | <https://bmrs.elexon.co.uk/api-documentation/introduction> | 9 August 2026 |
| NESO Data Portal | Demand, transmission-wind and embedded-renewable forecasts; interconnector outturns | <https://www.neso.energy/data-portal> | 9 August 2026 |
| NESO Data Portal | Daily Operational Planning Margin Requirement vintages | <https://www.neso.energy/data-portal/daily-opmr> | 9 August 2026 |
| Investing.com | UK natural-gas futures proxy | <https://uk.investing.com/commodities/natural-gas-historical-data?cid=1057002> | 24 July 2026 |
| Investing.com | UK emissions-allowance futures proxy | <https://uk.investing.com/commodities/uk-emissions-allowances-energy-c1-futures-historical-data> | 24 July 2026 |
| EPEX SPOT | Hourly GB day-ahead clearing price target | Licensed; not redistributed | Not a public-source retrieval |

The dates above are access dates, not claims that providers never revised their
data afterwards. Exact numerical reproduction therefore uses a lawfully held
frozen snapshot. Fresh downloads may differ due to provider corrections.

## Expected local layout

Place source files under `04_data/raw/` in the names documented below and read
by `modelling/data/prepare_features.py`. The licensed target must be
supplied locally as:

```text
04_data/raw/licensed_epex/DAM_outturn_combined.parquet
```

The provider links above document the BMRS and NESO sources. Gas and carbon
histories are manual downloads and are expected as:

```text
04_data/raw/manual_commodities/uk_natural_gas_futures_investing.csv
04_data/raw/manual_commodities/uk_emissions_allowances_futures_investing.csv
```

Run `python run_pipeline.py preflight` after populating the local data tree.

## Restricted publication snapshot

The LEAR and three-model QRA scripts use a compact local snapshot assembled from
the constituent runs. Create it only after the core pipeline has finished:

```bash
python pipeline/prepare_publication_snapshot.py
```

The destination is ignored by Git. It contains the model table, the QRF and LSTM
development/test forecasts and their realised prices, so it must not be added to
a public commit. The SHA-256 checksums of the results-of-record snapshot are
retained in `docs/results_of_record_input_checksums.txt` for audit purposes.
