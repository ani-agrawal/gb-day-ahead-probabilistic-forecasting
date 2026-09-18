# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Diss (.venv)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Feature engineering for the initial GB day-ahead price forecasting model.
#
# This notebook builds a point-in-time modelling table from raw public data.
# It deliberately does not train a model. The output is a clean parquet table
# used by 03_model_training.ipynb.
#
# Core leakage rule:
# For delivery day D, forecast-time features must have been available before
# the day-ahead auction cut-off on D-1 at 09:20 Europe/London time. Outturns
# are only used as D-2 or D-7 lagged features. D-1 outturns are NOT legal:
# the auction for day D runs at 09:20 on D-1, before most of D-1 has been
# delivered, so same-hour D-1 values do not exist at bid time.

# %%
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CACHE_DIR = Path.cwd() / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

# %%
# 1. Paths and settings

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2025-12-31")
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")

AUCTION_CUTOFF_HOUR = 9
AUCTION_CUTOFF_MINUTE = 20
LOCAL_TZ = ZoneInfo("Europe/London")


def find_project_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "04_data").exists() and (base / "modelling").exists():
            return base
    raise FileNotFoundError("Could not find repository root containing 04_data and modelling")


PROJECT_ROOT = find_project_root()
RAW_DIR = PROJECT_ROOT / "04_data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "04_data" / "processed"
METADATA_DIR = PROJECT_ROOT / "04_data" / "metadata"
AUDIT_DIR = METADATA_DIR / "feature_engineering_audits"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")


# %%
# 2. Helper functions

def normalise_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("K", "e3", regex=False)
        .str.replace("M", "e6", regex=False),
        errors="coerce",
    )


def settlement_key(df: pd.DataFrame, date_col: str, sp_col: str) -> pd.DataFrame:
    out = df.copy()
    out["settlement_date"] = normalise_date(out[date_col])
    out["settlement_period"] = pd.to_numeric(out[sp_col], errors="coerce").astype("Int64")
    return out


def delivery_cutoff_utc(settlement_dates: pd.Series) -> pd.Series:
    """D-1 09:20 Europe/London auction cut-off, returned as UTC timestamps.

    Built directly on D-1's calendar date rather than as "09:20 on D minus
    24 hours": the subtraction form shifts the wall-clock time by an hour
    across DST changes (found in the 14 Jul 2026 leakage re-audit, where the
    three autumn clock-change delivery days got a 10:20 cut-off).
    """
    dates = pd.to_datetime(settlement_dates, errors="coerce").dt.normalize()
    cutoff_map = {}
    for d in pd.Series(dates.dropna().unique()):
        cutoff_map[d] = (
            pd.Timestamp((d - pd.Timedelta(days=1)).date(), tz=LOCAL_TZ)
            .replace(hour=AUCTION_CUTOFF_HOUR, minute=AUCTION_CUTOFF_MINUTE)
        ).tz_convert("UTC")
    return dates.map(cutoff_map)


def latest_available_before_cutoff(
    df: pd.DataFrame,
    date_col: str,
    sp_col: str,
    publish_col: str,
    value_cols: list[str],
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """
    For each settlement date/period, keep only the latest row whose publish time
    is at or before the day-ahead auction cut-off. This is the central
    point-in-time safeguard.
    """
    work = settlement_key(df[[date_col, sp_col, publish_col, *value_cols]], date_col, sp_col)
    work = work[(work["settlement_date"] >= START_DATE) & (work["settlement_date"] <= END_DATE)].copy()
    work["_publish_time"] = pd.to_datetime(work[publish_col], errors="coerce", utc=True)
    work["_cutoff_time"] = delivery_cutoff_utc(work["settlement_date"])
    work["_safe_before_cutoff"] = work["_publish_time"] <= work["_cutoff_time"]

    audit_cols = ["settlement_date", "settlement_period", "_publish_time", "_cutoff_time", "_safe_before_cutoff"]
    work[audit_cols].head(5000).to_csv(AUDIT_DIR / f"{prefix}_timing_sample.csv", index=False)

    before = work[work["_safe_before_cutoff"]].copy()
    before = before.dropna(subset=["settlement_date", "settlement_period", "_publish_time"])
    before = before.sort_values("_publish_time")
    selected = before.groupby(["settlement_date", "settlement_period"], as_index=False).tail(1)

    keep = ["settlement_date", "settlement_period", "_publish_time", *value_cols]
    selected = selected[keep].rename(columns={"_publish_time": f"{prefix}_publish_time"})
    selected = selected.rename(columns={c: f"{prefix}_{c}" for c in value_cols})

    stats = {
        "raw_rows": len(work),
        "safe_rows_before_cutoff": len(before),
        "unsafe_rows_after_cutoff": int((~work["_safe_before_cutoff"]).sum()),
        "selected_settlement_periods": len(selected),
        "unique_settlement_periods": work[["settlement_date", "settlement_period"]].drop_duplicates().shape[0],
        "safe_row_pct": round(float(work["_safe_before_cutoff"].mean() * 100), 3) if len(work) else 0,
    }
    return selected, stats


def add_date_lag(
    base: pd.DataFrame,
    source: pd.DataFrame,
    cols: list[str],
    lag_days: int,
    suffix: str,
) -> pd.DataFrame:
    lagged = source[["settlement_date", "settlement_period", *cols]].copy()
    lagged["settlement_date"] = lagged["settlement_date"] + pd.Timedelta(days=lag_days)
    lagged = lagged.rename(columns={c: f"{c}_{suffix}" for c in cols})
    return base.merge(lagged, on=["settlement_date", "settlement_period"], how="left")


def uk_bank_holidays_2023_2025() -> set[pd.Timestamp]:
    dates = [
        "2023-01-02", "2023-04-07", "2023-04-10", "2023-05-01",
        "2023-05-08", "2023-05-29", "2023-08-28", "2023-12-25", "2023-12-26",
        "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06",
        "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26",
        "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05",
        "2025-05-26", "2025-08-25", "2025-12-25", "2025-12-26",
    ]
    return {pd.Timestamp(d) for d in dates}


BANK_HOLIDAYS = uk_bank_holidays_2023_2025()

# %%
# 3. Target variable: EPEX SPOT GB day-ahead hourly auction outturn
#
# Licensed EPEX SPOT data supplied locally under 04_data/raw/licensed_epex/,
# hourly in Europe/London time. The half-hourly feature spine below is a pure
# calendar grid, so no alternative price proxy enters the pipeline.

epex_raw = pd.read_parquet(RAW_DIR / "licensed_epex" / "DAM_outturn_combined.parquet")
epex = epex_raw.copy()
epex["delivery_start_local"] = pd.to_datetime(epex["Time"])  # tz-aware Europe/London
epex["delivery_start_utc"] = epex["delivery_start_local"].dt.tz_convert("UTC")
epex["settlement_date"] = epex["delivery_start_local"].dt.tz_localize(None).dt.normalize()
epex["hour"] = epex["delivery_start_local"].dt.hour
epex["price"] = pd.to_numeric(epex["outturn_price"], errors="coerce")
epex = epex[
    (epex["settlement_date"] >= START_DATE)
    & (epex["settlement_date"] <= END_DATE)
    & epex["price"].notna()
].copy()

# The autumn clock-change day repeats local hour 1 (25 auction products); the
# spring day skips one (23). Averaging the duplicated hour keeps one row per
# (date, local hour), matching the hourly modelling grid.
epex_hourly = (
    epex.groupby(["settlement_date", "hour"], as_index=False)
    .agg(delivery_start_utc=("delivery_start_utc", "first"), price=("price", "mean"))
)
print(f"EPEX hourly target: {epex_hourly.shape}, "
      f"{epex_hourly['settlement_date'].min():%Y-%m-%d} -> {epex_hourly['settlement_date'].max():%Y-%m-%d}")

# Timezone-handling assertions, re-verified on every rebuild: the file must
# arrive tz-aware, and local-day lengths must match the GB clock changes.
assert str(epex_raw["Time"].dtype).endswith("Europe/London]"), "EPEX Time must be tz-aware Europe/London"
day_len = epex.groupby("settlement_date").size()
short_days = set(day_len[day_len == 23].index.strftime("%Y-%m-%d"))
long_days = set(day_len[day_len == 25].index.strftime("%Y-%m-%d"))
assert short_days <= {"2023-03-26", "2024-03-31", "2025-03-30"}, f"unexpected 23-hour days: {sorted(short_days)}"
# the raw extract keeps only one of the two 01:00 products on 2023-10-29, so
# that autumn day arrives as 24 rows; the duplicated-hour mean handles both.
assert long_days <= {"2024-10-27", "2025-10-26"}, f"unexpected 25-hour days: {sorted(long_days)}"
assert day_len[~day_len.index.strftime("%Y-%m-%d").isin(sorted(short_days | long_days))].eq(24).all(), \
    "non-clock-change day without 24 hours"
assert not epex_hourly.duplicated(["settlement_date", "hour"]).any()

# Half-hourly calendar spine for the feature build: every local half-hour of
# 2023-2025 (46/48/50 periods on clock-change days), independent of any price
# source. Features are engineered on this grid and aggregated to hourly below.
spine_local = pd.date_range(
    START_DATE.tz_localize(LOCAL_TZ),
    (END_DATE + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ),
    freq="30min", inclusive="left",
)
target = pd.DataFrame({"delivery_start_local": spine_local})
target["delivery_start_utc"] = target["delivery_start_local"].dt.tz_convert("UTC")
target["settlement_date"] = target["delivery_start_local"].dt.tz_localize(None).dt.normalize()
target["settlement_period"] = target.groupby("settlement_date").cumcount() + 1
target = target.drop(columns=["delivery_start_local"])

print(target.shape)

# %%
# 4. Forecast-time fundamentals

demand_raw = pd.read_parquet(RAW_DIR / "day_ahead_demand_forecast.parquet")
demand, demand_stats = latest_available_before_cutoff(
    demand_raw,
    date_col="Date",
    sp_col="Settlement_Period",
    publish_col="Publish_Datetime",
    value_cols=["Demand_Forecast"],
    prefix="demand",
)
demand = demand.rename(columns={"demand_Demand_Forecast": "demand_forecast"})
demand["demand_forecast"] = pd.to_numeric(demand["demand_forecast"], errors="coerce")

wind_raw = pd.read_parquet(RAW_DIR / "day_ahead_wind_forecast.parquet")
wind, wind_stats = latest_available_before_cutoff(
    wind_raw,
    date_col="Date",
    sp_col="Settlement_period",
    publish_col="Forecast_Timestamp",
    value_cols=["Capacity", "Incentive_forecast"],
    prefix="transmission_wind",
)
wind = wind.rename(
    columns={
        "transmission_wind_Capacity": "transmission_wind_capacity",
        "transmission_wind_Incentive_forecast": "transmission_wind_forecast",
    }
)
for col in ["transmission_wind_capacity", "transmission_wind_forecast"]:
    if col in wind.columns:
        wind[col] = pd.to_numeric(wind[col], errors="coerce")

solar_raw = pd.read_parquet(
    RAW_DIR / "embedded_solar_forecast.parquet",
    columns=[
        "SETTLEMENT_DATE",
        "SETTLEMENT_PERIOD",
        "EMBEDDED_SOLAR_FORECAST",
        "EMBEDDED_WIND_FORECAST",
        "Forecast_Datetime",
    ],
)
solar, solar_stats = latest_available_before_cutoff(
    solar_raw,
    date_col="SETTLEMENT_DATE",
    sp_col="SETTLEMENT_PERIOD",
    publish_col="Forecast_Datetime",
    value_cols=["EMBEDDED_SOLAR_FORECAST", "EMBEDDED_WIND_FORECAST"],
    prefix="embedded",
)
solar = solar.rename(
    columns={
        "embedded_EMBEDDED_SOLAR_FORECAST": "embedded_solar_forecast",
        "embedded_EMBEDDED_WIND_FORECAST": "embedded_wind_forecast",
    }
)
solar["embedded_solar_forecast"] = pd.to_numeric(solar["embedded_solar_forecast"], errors="coerce")
solar["embedded_wind_forecast"] = pd.to_numeric(solar["embedded_wind_forecast"], errors="coerce")

print("Demand timing:", demand_stats)
print("Transmission wind timing:", wind_stats)
print("Embedded solar/wind timing:", solar_stats)

# --- BMRS vintage upgrade for demand and transmission wind forecasts ---
# The NESO portal archives keep one publish time per forecast, so roughly half
# of these forecasts fail the 09:20 D-1 cut-off and drop out. BMRS carries the
# same NESO forecasts with the full publication history (NDF ~48 vintages/day,
# WINDFOR ~8/day), so a pre-cut-off vintage exists for almost every period.
# Prefer the latest legal BMRS vintage; fall back to the portal value.
# Optional forecast-vintage files may be obtained from the provider APIs described in docs/DATA_ACCESS.md.
ndf_path = RAW_DIR / "bmrs_ndf_vintages.parquet"
if ndf_path.exists():
    ndf_raw = pd.read_parquet(ndf_path)
    demand_bmrs, demand_bmrs_stats = latest_available_before_cutoff(
        ndf_raw,
        date_col="settlementDate",
        sp_col="settlementPeriod",
        publish_col="publishTime",
        value_cols=["demand"],
        prefix="demand_bmrs",
    )
    demand_bmrs = demand_bmrs.rename(columns={"demand_bmrs_demand": "demand_forecast_bmrs"})
    demand_bmrs["demand_forecast_bmrs"] = pd.to_numeric(demand_bmrs["demand_forecast_bmrs"], errors="coerce")
    demand = demand.merge(
        demand_bmrs[["settlement_date", "settlement_period", "demand_forecast_bmrs"]],
        on=["settlement_date", "settlement_period"],
        how="outer",
    )
    demand["demand_forecast"] = demand["demand_forecast_bmrs"].combine_first(demand["demand_forecast"])
    demand = demand.drop(columns=["demand_forecast_bmrs"])
    print(f"demand forecast: BMRS NDF vintages merged, missing now "
          f"{demand['demand_forecast'].isna().mean() * 100:.1f}% of rows")

windfor_path = RAW_DIR / "bmrs_windfor_vintages.parquet"
if windfor_path.exists():
    windfor_raw = pd.read_parquet(windfor_path)
    wind_bmrs, wind_bmrs_stats = latest_available_before_cutoff(
        windfor_raw,
        date_col="settlementDate",
        sp_col="settlementPeriod",
        publish_col="publishTime",
        value_cols=["generation"],
        prefix="transmission_wind_bmrs",
    )
    wind_bmrs = wind_bmrs.rename(
        columns={"transmission_wind_bmrs_generation": "transmission_wind_forecast_bmrs"}
    )
    wind_bmrs["transmission_wind_forecast_bmrs"] = pd.to_numeric(
        wind_bmrs["transmission_wind_forecast_bmrs"], errors="coerce"
    )
    wind = wind.merge(
        wind_bmrs[["settlement_date", "settlement_period", "transmission_wind_forecast_bmrs"]],
        on=["settlement_date", "settlement_period"],
        how="outer",
    )
    wind["transmission_wind_forecast"] = wind["transmission_wind_forecast_bmrs"].combine_first(
        wind["transmission_wind_forecast"]
    )
    wind = wind.drop(columns=["transmission_wind_forecast_bmrs"])
    print(f"transmission wind forecast: BMRS WINDFOR vintages merged, missing "
          f"{wind['transmission_wind_forecast'].isna().mean() * 100:.1f}% of frame rows "
          f"(hourly-only horizons are filled on the full grid in the table build)")


# %%
# 5. Scarcity features: LoLP and de-rated margin

lolp_raw = pd.read_parquet(RAW_DIR / "scarcity_lolp_derated_margin.parquet")
lolp, lolp_stats = latest_available_before_cutoff(
    lolp_raw,
    date_col="settlementDate",
    sp_col="settlementPeriod",
    publish_col="publishTime",
    value_cols=["forecastHorizon", "lossOfLoadProbability", "deratedMargin"],
    prefix="lolp",
)
lolp = lolp.rename(
    columns={
        "lolp_forecastHorizon": "lolp_forecast_horizon",
        "lolp_lossOfLoadProbability": "lolp",
        "lolp_deratedMargin": "derated_margin",
    }
)
for col in ["lolp", "derated_margin"]:
    if col in lolp.columns:
        lolp[col] = pd.to_numeric(lolp[col], errors="coerce")
lolp["lolp_positive"] = (lolp["lolp"] > 0).astype(int) if "lolp" in lolp.columns else pd.Series(dtype=int)
lolp["log1p_lolp"] = np.log1p(lolp["lolp"]) if "lolp" in lolp.columns else pd.Series(dtype=float)

print("LoLP timing:", lolp_stats)


# %%
# 6. Commodity features: D-2 close to avoid the 09:20 D-1 leak

def read_investing_csv(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["commodity_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in ["Price", "Open", "High", "Low", "Vol.", "Change %"]:
        if col in df.columns:
            clean = col.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
            df[f"{prefix}_{clean}"] = as_numeric(df[col])
    return df.sort_values("commodity_date")


gas = read_investing_csv(
    RAW_DIR / "manual_commodities" / "uk_natural_gas_futures_investing.csv",
    "gas",
)
carbon = read_investing_csv(
    RAW_DIR / "manual_commodities" / "uk_emissions_allowances_futures_investing.csv",
    "carbon",
)

commodity_dates = target[["settlement_date"]].drop_duplicates().sort_values("settlement_date")

# At the D-1 09:20 auction, the D-1 daily close is not known yet. Use the last
# available close no later than D-2.
commodity_dates["commodity_lookup_date"] = commodity_dates["settlement_date"] - pd.Timedelta(days=2)

gas_features = pd.merge_asof(
    commodity_dates,
    gas[["commodity_date", "gas_price"]].dropna().sort_values("commodity_date"),
    left_on="commodity_lookup_date",
    right_on="commodity_date",
    direction="backward",
).drop(columns=["commodity_date", "commodity_lookup_date"])

carbon_features = pd.merge_asof(
    commodity_dates,
    carbon[["commodity_date", "carbon_price"]].dropna().sort_values("commodity_date"),
    left_on="commodity_lookup_date",
    right_on="commodity_date",
    direction="backward",
).drop(columns=["commodity_date", "commodity_lookup_date"])

commodities = gas_features.merge(carbon_features, on="settlement_date", how="outer")
print(commodities.head())

# %%
# 6b. OPMR D-2 vintage margin features (NESO Daily OPMR)

opmr_raw = pd.read_csv(RAW_DIR / "csv_opmr_daily.csv")
opmr_raw.columns = opmr_raw.columns.str.strip()
opmr_raw["publish_date"] = pd.to_datetime(opmr_raw["Publish Date"], errors="coerce")
opmr_raw["target_date"] = pd.to_datetime(opmr_raw["Date"], errors="coerce")
opmr_raw["lead_days"] = (opmr_raw["target_date"] - opmr_raw["publish_date"]).dt.days

OPMR_COLS = {
    "National Surplus": "opmr_national_surplus",
    "Generator Availability": "opmr_generator_availability",
    "Maximum I/C Import": "opmr_max_ic_import",
    "Constrained Plant": "opmr_constrained_plant",
    "OPMR total": "opmr_total_requirement",
    "Minimum Demand Forecast": "opmr_min_demand_forecast",
    "Negative Reserve": "opmr_negative_reserve",
    "Peak Demand Forecast": "opmr_peak_demand_forecast",
}
for col in OPMR_COLS:
    opmr_raw[col] = pd.to_numeric(opmr_raw[col], errors="coerce")

# Publications at lead < 2 days land after the D-1 09:20 auction cut-off and
# must not be used. A zero Peak or Minimum Demand Forecast marks a corrupt
# NESO publication (a zero peak also corrupts the derived margin/surplus on
# that row), so those rows are excluded and the previous vintage is used.
opmr_valid = opmr_raw[
    (opmr_raw["lead_days"] >= 2)
    & (opmr_raw["Peak Demand Forecast"] > 0)
    & (opmr_raw["Minimum Demand Forecast"] > 0)
].copy()


def latest_opmr_vintage(min_lead: int) -> pd.DataFrame:
    eligible = opmr_valid[opmr_valid["lead_days"] >= min_lead]
    return (
        eligible.sort_values(["target_date", "publish_date"])
        .groupby("target_date", as_index=False)
        .tail(1)
    )


opmr_d2 = latest_opmr_vintage(2).rename(columns=OPMR_COLS)
opmr_d7 = latest_opmr_vintage(7)[["target_date", "National Surplus"]].rename(
    columns={"National Surplus": "opmr_national_surplus_d7"}
)

opmr_features = opmr_d2[["target_date", *OPMR_COLS.values()]].merge(
    opmr_d7, on="target_date", how="left"
)
opmr_features["opmr_surplus_norm"] = (
    opmr_features["opmr_national_surplus"]
    / opmr_features["opmr_peak_demand_forecast"].replace(0, np.nan)
)
# Revision of the surplus outlook between the D-7 and D-2 vintages for the
# same delivery day: a deteriorating margin the level alone does not show.
opmr_features["opmr_surplus_rev_7d"] = (
    opmr_features["opmr_national_surplus"] - opmr_features["opmr_national_surplus_d7"]
)
opmr_features = opmr_features.drop(
    columns=["opmr_peak_demand_forecast", "opmr_national_surplus_d7"]
).rename(columns={"target_date": "settlement_date"})

opmr_stats = {
    "target_dates": len(opmr_features),
    "lead2_share": float((opmr_d2["lead_days"] == 2).mean()),
    "max_lead_used": int(opmr_d2["lead_days"].max()),
}
print("OPMR vintage coverage:", opmr_stats)

# %%
# 7. REMIT event-list metadata features

remit_raw = pd.read_parquet(RAW_DIR / "remit_event_list.parquet")
remit = remit_raw.copy()
remit["publish_time"] = pd.to_datetime(remit["publishTime"], errors="coerce", utc=True)
remit = remit.dropna(subset=["publish_time"]).copy()

date_cutoffs = target[["settlement_date"]].drop_duplicates().sort_values("settlement_date")
date_cutoffs["auction_cutoff_utc"] = delivery_cutoff_utc(date_cutoffs["settlement_date"])

remit_rows = []
for row in date_cutoffs.itertuples(index=False):
    cutoff = row.auction_cutoff_utc
    recent_1d = remit[(remit["publish_time"] > cutoff - pd.Timedelta(days=1)) & (remit["publish_time"] <= cutoff)]
    recent_7d = remit[(remit["publish_time"] > cutoff - pd.Timedelta(days=7)) & (remit["publish_time"] <= cutoff)]
    recent_30d = remit[(remit["publish_time"] > cutoff - pd.Timedelta(days=30)) & (remit["publish_time"] <= cutoff)]
    remit_rows.append(
        {
            "settlement_date": row.settlement_date,
            "remit_events_published_1d": len(recent_1d),
            "remit_events_published_7d": len(recent_7d),
            "remit_events_published_30d": len(recent_30d),
            "remit_unique_mrid_published_7d": recent_7d["mrid"].nunique() if "mrid" in recent_7d else np.nan,
        }
    )

remit_features = pd.DataFrame(remit_rows)
print(remit_features.head())

# %%
# 8. Lagged outturn features

system_price = pd.read_parquet(RAW_DIR / "system_prices_outturn.parquet")
system_price = settlement_key(system_price, "settlementDate", "settlementPeriod")
system_price = system_price[
    ["settlement_date", "settlement_period", "systemSellPrice", "systemBuyPrice", "netImbalanceVolume"]
].copy()
for col in ["systemSellPrice", "systemBuyPrice", "netImbalanceVolume"]:
    system_price[col] = pd.to_numeric(system_price[col], errors="coerce")

fuel = pd.read_parquet(RAW_DIR / "generation_by_fuel_outturn.parquet")
fuel = settlement_key(fuel, "settlementDate", "settlementPeriod")
fuel = fuel[(fuel["settlement_date"] >= START_DATE - pd.Timedelta(days=10)) & (fuel["settlement_date"] <= END_DATE)].copy()
fuel["generation"] = pd.to_numeric(fuel["generation"], errors="coerce")
fuel_wide = (
    fuel.pivot_table(
        index=["settlement_date", "settlement_period"],
        columns="fuelType",
        values="generation",
        aggfunc="mean",
    )
    .reset_index()
)
fuel_wide.columns = [
    "settlement_date" if c == "settlement_date"
    else "settlement_period" if c == "settlement_period"
    else f"gen_{str(c).lower()}_mw"
    for c in fuel_wide.columns
]

hist = pd.read_parquet(RAW_DIR / "historic_demand_wind_solar_interconnector_outturn.parquet")
hist = settlement_key(hist, "SETTLEMENT_DATE", "SETTLEMENT_PERIOD")
flow_cols = [c for c in hist.columns if c.endswith("_FLOW")]
for col in flow_cols + ["ND", "TSD", "EMBEDDED_WIND_GENERATION", "EMBEDDED_SOLAR_GENERATION"]:
    hist[col] = pd.to_numeric(hist[col], errors="coerce")
hist["net_interconnector_flow"] = hist[flow_cols].sum(axis=1)
hist_outturn = hist[
    [
        "settlement_date",
        "settlement_period",
        "ND",
        "TSD",
        "EMBEDDED_WIND_GENERATION",
        "EMBEDDED_SOLAR_GENERATION",
        "net_interconnector_flow",
        *flow_cols,
    ]
].copy()

# %%
# 9. Build modelling table

model = target.copy()

for frame in [demand, wind, solar, commodities, opmr_features, remit_features, lolp]:
    keys = ["settlement_date", "settlement_period"] if "settlement_period" in frame.columns else ["settlement_date"]
    model = model.merge(frame, on=keys, how="left")

# WINDFOR forecasts beyond ~day-ahead are hourly (odd settlement periods
# only). They are deliberately left unfilled here: the skipna hourly mean in
# section 11 then recovers each native hourly record exactly. (An earlier
# interpolate-to-half-hourly step, needed when the modelling grid was
# half-hourly, was removed 16 Jul 2026 - after hourly aggregation it blended
# adjacent hours roughly 75/25 instead of preserving the original values.)
if "transmission_wind_forecast" not in model.columns:
    model["transmission_wind_forecast"] = np.nan
if "embedded_wind_forecast" not in model.columns:
    model["embedded_wind_forecast"] = np.nan

# Derived fundamentals (net demand, renewable share, residual-load features,
# the wind-missing flag) are computed on the HOURLY grid in section 11, so
# every ratio and difference is consistent with the hourly means and the
# missing flag means "no record anywhere in the delivery hour".

model["delivery_date"] = model["settlement_date"]
model["month"] = model["settlement_date"].dt.month
model["dayofweek"] = model["settlement_date"].dt.dayofweek
model["is_weekend"] = model["dayofweek"].isin([5, 6]).astype(int)
model["is_bank_holiday"] = model["settlement_date"].isin(BANK_HOLIDAYS).astype(int)
model["hour"] = ((model["settlement_period"].astype(float) - 1) / 2).astype(float)
model["sp_sin"] = np.sin(2 * np.pi * model["settlement_period"].astype(float) / 48)
model["sp_cos"] = np.cos(2 * np.pi * model["settlement_period"].astype(float) / 48)
model["month_sin"] = np.sin(2 * np.pi * model["month"] / 12)
model["month_cos"] = np.cos(2 * np.pi * model["month"] / 12)

# Price lags are computed on the hourly grid from the EPEX target itself
# (section 11): the auction target makes lag1d legal, since prices for
# delivery day D-1 clear at the D-2 morning auction.
model = add_date_lag(model, system_price, ["systemSellPrice", "systemBuyPrice", "netImbalanceVolume"], 2, "lag2d")
model = add_date_lag(model, system_price, ["systemSellPrice", "systemBuyPrice", "netImbalanceVolume"], 7, "lag7d")
model = add_date_lag(model, fuel_wide, [c for c in fuel_wide.columns if c.startswith("gen_")], 2, "lag2d")
model = add_date_lag(model, fuel_wide, [c for c in fuel_wide.columns if c.startswith("gen_")], 7, "lag7d")
model = add_date_lag(model, hist_outturn, [c for c in hist_outturn.columns if c not in ["settlement_date", "settlement_period"]], 2, "lag2d")
model = add_date_lag(model, hist_outturn, [c for c in hist_outturn.columns if c not in ["settlement_date", "settlement_period"]], 7, "lag7d")

model = model.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

model_path = PROCESSED_DIR / "feature_table_halfhourly.parquet"
model.to_parquet(model_path, index=False)
print(f"Saved half-hourly feature table (no price columns): {model_path}")
print(model.shape)

# %%
# 10. Feature list and leakage audit

identifier_cols = {
    "settlement_date",
    "settlement_period",
    "delivery_start_utc",
    "delivery_date",
}
target_or_post_auction_cols = {
    "price",
    "mid_volume",
}
publish_cols = {c for c in model.columns if c.endswith("_publish_time")}

candidate_feature_cols = [
    c for c in model.columns
    if c not in identifier_cols
    and c not in target_or_post_auction_cols
    and c not in publish_cols
    and not str(model[c].dtype).startswith("datetime")
    and pd.api.types.is_numeric_dtype(model[c])
]

MAX_FEATURE_MISSING_PCT = 95.0
excluded_high_missing_features = [
    c for c in candidate_feature_cols
    if model[c].isna().mean() * 100 > MAX_FEATURE_MISSING_PCT
]
feature_cols = [c for c in candidate_feature_cols if c not in excluded_high_missing_features]

leakage_flags = []
for col in feature_cols:
    lower = col.lower()
    if col in {"ND", "TSD", "EMBEDDED_WIND_GENERATION", "EMBEDDED_SOLAR_GENERATION"}:
        leakage_flags.append((col, "same-day outturn column should not be a feature"))
    if any(token in lower for token in ["outturn", "generation", "flow", "systembuy", "systemsell", "imbalance"]):
        if not lower.endswith(("lag2d", "lag3d", "lag7d")):
            if not lower.startswith(("demand_forecast", "transmission_wind", "embedded_", "renewable_", "total_wind", "net_demand")):
                leakage_flags.append((col, "outturn-like variable without lag suffix"))
    if col == "mid_volume":
        leakage_flags.append((col, "same-day MID volume is only known after delivery and leaks"))

feature_metadata = pd.DataFrame(
    {
        "feature": candidate_feature_cols,
        "missing_pct": [model[c].isna().mean() * 100 for c in candidate_feature_cols],
        "dtype": [str(model[c].dtype) for c in candidate_feature_cols],
        "used_in_initial_model": [c in feature_cols for c in candidate_feature_cols],
    }
).sort_values("feature")
feature_metadata.to_csv(METADATA_DIR / "initial_model_feature_list.csv", index=False)

timing_audit = pd.DataFrame(
    [
        {"source": "demand_forecast", **demand_stats},
        {"source": "transmission_wind_forecast", **wind_stats},
        {"source": "embedded_solar_wind_forecast", **solar_stats},
        {"source": "lolp_derated_margin", **lolp_stats},
    ]
)
timing_audit.to_csv(AUDIT_DIR / "forecast_timing_summary.csv", index=False)

leakage_audit = pd.DataFrame(
    {
        "check": [
            "rows",
            "candidate_features",
            "used_features",
            "train_rows_2023_2024",
            "test_rows_2025",
            "leakage_flags",
            "excluded_high_missing_features",
            "commodity_lag_days",
            "remit_feature_type",
        ],
        "value": [
            len(model),
            len(candidate_feature_cols),
            len(feature_cols),
            int((model["settlement_date"] <= TRAIN_END).sum()),
            int((model["settlement_date"] >= TEST_START).sum()),
            len(leakage_flags),
            len(excluded_high_missing_features),
            2,
            "metadata event counts only; no unavailable MW in raw REMIT file",
        ],
    }
)
leakage_audit.to_csv(AUDIT_DIR / "leakage_audit.csv", index=False)

if excluded_high_missing_features:
    pd.DataFrame({"feature": excluded_high_missing_features}).to_csv(
        AUDIT_DIR / "excluded_high_missing_features.csv",
        index=False,
    )
if leakage_flags:
    pd.DataFrame(leakage_flags, columns=["feature", "issue"]).to_csv(
        AUDIT_DIR / "leakage_flags.csv",
        index=False,
    )

print(timing_audit)
print(leakage_audit)
print(feature_metadata[feature_metadata["used_in_initial_model"]].head(10))


# %%
# 11. Hourly modelling table for the day-ahead hourly auction
#
# The morning day-ahead auction clears 24 hourly products (23/25 on clock-change
# days), so the modelling grid is hourly. Features engineered on the half-hourly
# grid above are aggregated to hourly (mean; max for flags) and the EPEX GB
# hourly auction outturn is attached directly as the target - it is native to
# this grid, so no aggregation of the target is involved.

hourly = model.drop(columns=["hour", "sp_sin", "sp_cos"]).copy()
hourly["hour"] = (
    hourly["delivery_start_utc"].dt.tz_convert("Europe/London").dt.hour
)

HOURLY_KEYS = ["settlement_date", "hour"]
flag_cols = [
    c for c in ["low_residual_load_flag", "wind_forecast_missing", "lolp_positive", "is_weekend", "is_bank_holiday"]
    if c in hourly.columns
]

agg_rules = {}
for c in hourly.columns:
    if c in HOURLY_KEYS or c in ("price", "mid_volume", "settlement_period"):
        continue
    if c in flag_cols:
        agg_rules[c] = "max"
    elif str(hourly[c].dtype).startswith("datetime"):
        agg_rules[c] = "first"
    else:
        agg_rules[c] = "mean"

grouped = hourly.groupby(HOURLY_KEYS, as_index=False)
hourly_table = grouped.agg(agg_rules)

# attach the EPEX hourly auction outturn as the target (native hourly grid)
hourly_table = hourly_table.merge(
    epex_hourly[[*HOURLY_KEYS, "price"]], on=HOURLY_KEYS, how="left"
)
hourly_table = hourly_table.dropna(subset=["price"]).copy()

# Derived fundamentals on the hourly grid (moved from the half-hourly build,
# 16 Jul 2026): the skipna hourly mean recovers native-hourly WINDFOR records
# exactly, and deriving here keeps every ratio consistent with hourly means.
# A missing transmission wind forecast means "no pre-cut-off record for this
# delivery hour", never "zero wind"; it stays NaN for the training imputer
# and the flag lets tree models separate "missing" from "low wind".
hourly_table["wind_forecast_missing"] = hourly_table["transmission_wind_forecast"].isna().astype(int)
hourly_table["total_wind_forecast"] = (
    hourly_table["transmission_wind_forecast"] + hourly_table["embedded_wind_forecast"]
)
hourly_table["net_demand_forecast"] = (
    hourly_table["demand_forecast"]
    - hourly_table["total_wind_forecast"]
    - hourly_table["embedded_solar_forecast"]
)
hourly_table["renewable_forecast_total"] = (
    hourly_table["total_wind_forecast"] + hourly_table["embedded_solar_forecast"]
)
hourly_table["renewable_share_forecast"] = (
    hourly_table["renewable_forecast_total"] / hourly_table["demand_forecast"].replace(0, np.nan)
)
# negative prices are threshold events; expose the low-residual-load cliff edge
hourly_table["pure_surplus_mw"] = (
    hourly_table["renewable_forecast_total"] - hourly_table["demand_forecast"]
).clip(lower=0)
hourly_table["low_residual_load_mw"] = (
    hourly_table["demand_forecast"] - hourly_table["renewable_forecast_total"]
).clip(lower=0)
hourly_table["surplus_share_of_demand"] = (
    hourly_table["pure_surplus_mw"] / hourly_table["demand_forecast"].replace(0, np.nan)
)
hourly_table["low_residual_load_flag"] = (
    (hourly_table["low_residual_load_mw"] < 5000)
    .astype(float)
    .where(hourly_table["low_residual_load_mw"].notna())
)
if "gen_nuclear_mw_lag2d" in hourly_table.columns:
    hourly_table["nuclear_renewable_pressure_lag2d"] = (
        hourly_table["gen_nuclear_mw_lag2d"] * hourly_table["renewable_share_forecast"]
    )
if "gen_nuclear_mw_lag7d" in hourly_table.columns:
    hourly_table["nuclear_renewable_pressure_lag7d"] = (
        hourly_table["gen_nuclear_mw_lag7d"] * hourly_table["renewable_share_forecast"]
    )

# Price lags from the EPEX target itself. lag1d is legal for an auction
# target: prices for delivery day D-1 clear at the D-2 morning auction and
# are published within the hour, so they are known at the D-1 09:20 gate.
# lag2d and lag7d complete the classical day-ahead autoregressive set.
lag_source = hourly_table[[*HOURLY_KEYS, "price"]].copy()
for lag_days, suffix in [(1, "lag1d"), (2, "lag2d"), (7, "lag7d")]:
    lagged = lag_source.copy()
    lagged["settlement_date"] = lagged["settlement_date"] + pd.Timedelta(days=lag_days)
    lagged = lagged.rename(columns={"price": f"price_{suffix}"})
    hourly_table = hourly_table.merge(lagged, on=HOURLY_KEYS, how="left")

hourly_table["hour_sin"] = np.sin(2 * np.pi * hourly_table["hour"] / 24)
hourly_table["hour_cos"] = np.cos(2 * np.pi * hourly_table["hour"] / 24)
hourly_table = hourly_table.sort_values(HOURLY_KEYS).reset_index(drop=True)

hourly_path = PROCESSED_DIR / "model_table_hourly_epex.parquet"
hourly_table.to_parquet(hourly_path, index=False)
print(f"Saved hourly model table (EPEX target): {hourly_path}")
print(hourly_table.shape)

# hourly feature list, mirroring the half-hourly audit rules
hourly_identifiers = {"settlement_date", "delivery_start_utc", "delivery_date"}
hourly_publish_cols = {c for c in hourly_table.columns if c.endswith("_publish_time")}
hourly_candidates = [
    c for c in hourly_table.columns
    if c not in hourly_identifiers
    and c not in {"price", "mid_volume"}
    and c not in hourly_publish_cols
    and not str(hourly_table[c].dtype).startswith("datetime")
    and pd.api.types.is_numeric_dtype(hourly_table[c])
]
hourly_used = [
    c for c in hourly_candidates
    if hourly_table[c].isna().mean() * 100 <= MAX_FEATURE_MISSING_PCT
]
pd.DataFrame(
    {
        "feature": hourly_candidates,
        "missing_pct": [hourly_table[c].isna().mean() * 100 for c in hourly_candidates],
        "dtype": [str(hourly_table[c].dtype) for c in hourly_candidates],
        "used_in_initial_model": [c in hourly_used for c in hourly_candidates],
    }
).sort_values("feature").to_csv(METADATA_DIR / "initial_model_feature_list_hourly.csv", index=False)
print(f"Hourly features: {len(hourly_used)} used of {len(hourly_candidates)} candidates")
