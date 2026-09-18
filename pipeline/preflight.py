"""Validate the environment and locally supplied inputs for the code-only release."""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "04_data" / "raw"
REQUIRED_INPUTS = [
    RAW / "licensed_epex/DAM_outturn_combined.parquet",
    RAW / "day_ahead_demand_forecast.parquet",
    RAW / "day_ahead_wind_forecast.parquet",
    RAW / "embedded_solar_forecast.parquet",
    RAW / "scarcity_lolp_derated_margin.parquet",
    RAW / "manual_commodities/uk_natural_gas_futures_investing.csv",
    RAW / "manual_commodities/uk_emissions_allowances_futures_investing.csv",
    RAW / "csv_opmr_daily.csv",
    RAW / "remit_event_list.parquet",
    RAW / "system_prices_outturn.parquet",
    RAW / "generation_by_fuel_outturn.parquet",
    RAW / "historic_demand_wind_solar_interconnector_outturn.parquet",
]
CODE_FILES = [
    ROOT / "modelling/data/prepare_features.py",
    ROOT / "modelling/tabular/fit_tabular_models.py",
    ROOT / "modelling/lstm/development_backtest.py",
    ROOT / "modelling/lstm/test_backtest.py",
    ROOT / "modelling/lear/run_lear_qra_window.py",
    ROOT / "modelling/lear_qra/run_lear_qra_combine.py",
    ROOT / "modelling/qra/run_three_way_qra_qw.py",
]


def main() -> None:
    print(f"Repository root: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")

    missing_code = [str(path.relative_to(ROOT)) for path in CODE_FILES if not path.is_file()]
    if missing_code:
        raise FileNotFoundError("Release is incomplete:\n  " + "\n  ".join(missing_code))
    print(f"[ok] {len(CODE_FILES)} core code entry points are present")

    modules = {
        "numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
        "scipy": "scipy", "pyarrow": "pyarrow", "TensorFlow": "tensorflow",
        "Keras": "keras", "quantile-forest": "quantile_forest",
    }
    failures = []
    versions = {}
    for label, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            versions[label] = getattr(module, "__version__", "installed")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    if failures:
        raise RuntimeError("Environment is incomplete:\n  " + "\n  ".join(failures))
    print("[ok] environment: " + ", ".join(f"{k} {v}" for k, v in versions.items()))

    model_table = ROOT / "04_data/processed/model_table_hourly_epex.parquet"
    if model_table.exists():
        table = pd.read_parquet(model_table, columns=["settlement_date", "hour", "price"])
        if table[["settlement_date", "hour"]].duplicated().any() or table["price"].isna().any():
            raise AssertionError("processed table contains duplicate keys or missing target prices")
        print(f"[ok] processed table: {len(table):,} rows, unique keys, complete target")
    else:
        missing = [str(path.relative_to(ROOT)) for path in REQUIRED_INPUTS if not path.is_file()]
        if missing:
            message = (
                "The code release is intact, but local data are unavailable. This is expected in "
                "the public repository because the target is licensed. Follow docs/DATA_ACCESS.md."
            )
            print("[data unavailable] " + message)
            print("Missing local inputs:\n  " + "\n  ".join(missing))
            sys.exit(2)
        print("[ok] raw inputs are present; run with --rebuild-data to create the model table")

    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
