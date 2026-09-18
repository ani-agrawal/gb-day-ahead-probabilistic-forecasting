#!/usr/bin/env python3
"""Stage the compact, restricted input snapshot used by LEAR and three-model QRA.

The destination is gitignored because the model table and prediction files contain
licensed realised EPEX prices. This script only copies files; it never modifies the
constituent results. Use --check-only to validate paths without copying anything.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "modelling" / "input_snapshot"

SOURCES = {
    "model_table_hourly_epex.parquet":
        "04_data/processed/model_table_hourly_epex.parquet",
    "uk_natural_gas_futures_investing.csv":
        "04_data/raw/manual_commodities/uk_natural_gas_futures_investing.csv",
    "uk_emissions_allowances_futures_investing.csv":
        "04_data/raw/manual_commodities/uk_emissions_allowances_futures_investing.csv",
    "qrf_refit_log.csv":
        "06_outputs/RQ1/model_runs/hourly_model_recal_epex/quantile_forest/refit_log.csv",
    "qrf_dev2024H2_predictions.csv":
        "06_outputs/RQ1/model_runs/hourly_model_recal_epex/tuning/qrf_dev2024H2_predictions.csv",
    "qrf_test_predictions.csv":
        "06_outputs/RQ1/model_runs/hourly_model_recal_epex/quantile_forest/predictions.csv",
    "linear_qr_test_predictions.csv":
        "06_outputs/RQ1/model_runs/hourly_model_recal_epex/linear_qr/predictions.csv",
    "lstm_test_ensemble11q_predictions.csv":
        "06_outputs/RQ2/model_runs/seq2seq_final_b16es14/ENSEMBLE_11q/predictions.csv",
}

for seed in (42, 7, 101):
    SOURCES[f"lstm_dev_5q_seed{seed}.csv"] = (
        "06_outputs/RQ2/model_runs/rnn_dev/seq2seq_gate_v2/"
        f"encdec_u64_b16es14_seed{seed}/predictions.csv"
    )
    SOURCES[f"lstm_dev_fill6_seed{seed}.csv"] = (
        "06_outputs/RQ2/model_runs/rnn_dev/seq2seq_gate_v2/"
        f"encdec_u64_b16es14_fill6_seed{seed}/predictions.csv"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT,
        help="root containing 04_data and 06_outputs (default: repository root)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="check every source and print hashes without copying",
    )
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()

    missing = [relative for relative in SOURCES.values()
               if not (source_root / relative).is_file()]
    if missing:
        lines = "\n  ".join(missing)
        raise FileNotFoundError(
            "Publication snapshot cannot be assembled; missing:\n  " + lines
        )

    print(f"Source root: {source_root}")
    print(f"Destination: {DESTINATION}")
    print("CHECK ONLY: no files will be copied" if args.check_only else
          "RESTRICTED LOCAL STAGING: destination is excluded by .gitignore")

    records: list[tuple[str, str]] = []
    if not args.check_only:
        DESTINATION.mkdir(parents=True, exist_ok=True)

    for target_name, relative in SOURCES.items():
        source = source_root / relative
        digest = sha256(source)
        records.append((target_name, digest))
        print(f"[ok] {target_name:<48} {source.stat().st_size:>12,} bytes  {digest[:12]}")
        if not args.check_only:
            temporary = DESTINATION / f".{target_name}.tmp"
            shutil.copy2(source, temporary)
            temporary.replace(DESTINATION / target_name)

    if not args.check_only:
        sums = "".join(f"{digest}  {name}\n" for name, digest in records)
        (DESTINATION / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
        print(f"\nSNAPSHOT READY: {len(records)} files; SHA256SUMS.txt written")
        print("Do not add modelling/input_snapshot contents to a public commit.")
    else:
        print(f"\nSOURCE CHECK PASSED: {len(records)} files available")


if __name__ == "__main__":
    main()
