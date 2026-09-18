# Input schema

The modelling table is an hourly, local-delivery grid keyed by
`settlement_date` and `hour`. A normal day has 24 rows; spring clock-change days
have 23 scored hours. The autumn duplicate hour is represented by one local-hour
observation according to the rules in `modelling/data/prepare_features.py`.

Minimum columns needed by the manuscript LEAR-QRA extension:

| Column | Type | Meaning |
|---|---|---|
| `settlement_date` | date/datetime | Local GB delivery date |
| `hour` | integer | Local delivery hour, 0--23 |
| `price` | numeric | Licensed EPEX GB clearing price, GBP/MWh |
| `demand_forecast` | numeric | Gate-legal demand forecast |
| `transmission_wind_forecast` | numeric | Gate-legal transmission wind forecast |
| `embedded_wind_forecast` | numeric | Gate-legal embedded wind forecast |
| `embedded_solar_forecast` | numeric | Gate-legal embedded solar forecast |
| `renewable_forecast_total` | numeric | Wind-plus-solar aggregate used by canonical two-exogenous LEAR |

The full constituent pipeline additionally uses timestamped forecast vintages,
gas and carbon proxies, OPMR, calendar terms and leakage-safe lagged outturns.
`modelling/data/prepare_features.py` is the authoritative transformation from raw inputs to
`04_data/processed/model_table_hourly_epex.parquet`.

Prediction files use the same two keys, `actual_price`, and quantile columns:

```text
q05 q10 q20 q30 q40 q50 q60 q70 q80 q90 q95
```

The five-principal-quantile neural arms contain q05, q10, q50, q90 and q95;
the companion fill-six arms contain the remaining six columns. The assembly
stage averages matching quantiles across seeds and monotonically rearranges each
row before evaluation.
