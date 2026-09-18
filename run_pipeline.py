#!/usr/bin/env python3
"""Paper-specific orchestration for the public code release."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
LOG_DIR = ROOT / "logs"
PIPELINE = ROOT / "pipeline"
MODELLING = ROOT / "modelling"
TABULAR = MODELLING / "tabular"
LSTM = MODELLING / "lstm"
CONFORMAL = MODELLING / "conformal"
QRA = MODELLING / "qra"
ABLATIONS = MODELLING / "ablations"
EVALUATION = ROOT / "evaluation"
SEEDS = (42, 7, 101)


def run(command: list[str], stage: str, env_extra: dict[str, str] | None = None,
        cwd: Path = ROOT) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{stamp}_{stage}.log"
    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(LOG_DIR / "matplotlib_cache"),
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONHASHSEED": "0",
    })
    if env_extra:
        env.update(env_extra)
    print(f"\n[{stage}] {' '.join(command)}")
    print(f"[{stage}] log: {log_path.relative_to(ROOT)}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def preflight() -> None:
    run([PYTHON, str(PIPELINE / "preflight.py")], "00_preflight")


def build_data(rebuild: bool) -> None:
    model = ROOT / "04_data/processed/model_table_hourly_epex.parquet"
    if model.exists() and not rebuild:
        print("[01_data] using the local processed table; pass --rebuild-data to recreate it")
        return
    run([PYTHON, str(MODELLING / "data" / "prepare_features.py")], "01_feature_engineering")


def train_tabular() -> None:
    run([PYTHON, "fit_tabular_models.py"], "02_tabular_models", cwd=TABULAR)
    run([PYTHON, "qrf_dev2024h2.py"], "02_qrf_development", cwd=TABULAR)


def train_lstm() -> None:
    # The development forecasts are inputs to conformal calibration and QRA.
    for seed in SEEDS:
        for system, suffix in (("5q", ""), ("fill6", "_fill6")):
            run(
                [PYTHON, "development_backtest.py", "64", str(seed)],
                f"03_lstm_development_{system}_seed{seed}",
                {"SYSTEM": system, "BATCH": "16", "ES_DAYS": "14",
                 "RUN_TAG": f"_b16es14{suffix}"},
                LSTM,
            )
    for seed in SEEDS:
        for system in ("5q", "fill6"):
            run(
                [PYTHON, "test_backtest.py", system, str(seed)],
                f"04_lstm_2025_{system}_seed{seed}",
                {"BATCH": "16", "ES_DAYS": "14", "RUN_TAG": "_b16es14"},
                LSTM,
            )
    run([PYTHON, str(EVALUATION / "assemble_results.py")], "04_assemble_lstm")


def conformal_and_two_model_qra() -> None:
    run([PYTHON, "run_conformal.py"], "05_conformal", cwd=CONFORMAL)
    run([PYTHON, "run_two_model_qra.py"], "05_two_model_qra", cwd=QRA)


def constituent_models(rebuild_data: bool) -> None:
    preflight()
    build_data(rebuild_data)
    train_tabular()
    train_lstm()
    conformal_and_two_model_qra()


def prepare_snapshot() -> None:
    run([PYTHON, str(PIPELINE / "prepare_publication_snapshot.py")],
        "06_publication_snapshot")


def publication_models() -> None:
    prepare_snapshot()
    lear_dir = MODELLING / "lear"
    for window in (273, 364, 546, 728):
        run(
            [PYTHON, "run_lear_qra_window.py", "--window", str(window)],
            f"07_lear_window_{window}", cwd=lear_dir,
        )
    run(
        [PYTHON, "run_lear_qra_combine.py"], "07_lear_qra",
        cwd=MODELLING / "lear_qra",
    )
    run(
        [PYTHON, "run_three_way_qra_qw.py"], "07_three_model_qra",
        cwd=QRA,
    )


def evaluation() -> None:
    run([PYTHON, "constituent_dm_tests.py"], "08_constituent_dm_tests", cwd=EVALUATION)


def ablations() -> None:
    run([PYTHON, "tabular_ablation.py"], "09_tabular_ablations", cwd=ABLATIONS)
    arms = ("no_demand", "no_wind", "no_solar", "no_gas", "no_carbon",
            "no_opmr", "no_calendar", "no_price_history")
    for seed in SEEDS:
        for arm in arms:
            run(
                [PYTHON, "lstm_ablation.py", arm, str(seed)],
                f"09_lstm_ablation_{arm}_seed{seed}", cwd=ABLATIONS,
            )
    run([PYTHON, "summarise_lstm_ablation.py"], "09_lstm_ablation_summary", cwd=ABLATIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the constituent forecasts and publication extensions."
    )
    parser.add_argument(
        "command",
        choices=("preflight", "constituents", "publication-models", "evaluation",
                 "ablations", "full-paper", "audit-release"),
    )
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="recreate the processed model table from locally supplied raw inputs",
    )
    args = parser.parse_args()

    if args.command == "preflight":
        preflight()
    elif args.command == "constituents":
        constituent_models(args.rebuild_data)
    elif args.command == "publication-models":
        publication_models()
    elif args.command == "evaluation":
        evaluation()
    elif args.command == "ablations":
        ablations()
    elif args.command == "audit-release":
        run([PYTHON, str(PIPELINE / "audit_public_release.py")], "10_release_audit")
    elif args.command == "full-paper":
        constituent_models(args.rebuild_data)
        publication_models()
        evaluation()
        print("\nCOMPLETE: all non-ablation paper models and evaluations reproduced.")
        print("Run `python run_pipeline.py ablations` separately for the long post-hoc ablation study.")


if __name__ == "__main__":
    main()
