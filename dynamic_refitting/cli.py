"""Command-line interface for dynamic_refitting.

Usage examples::

    dynamic_refitting train --config config.yaml
    dynamic_refitting evaluate --model model.joblib --data test.csv
    dynamic_refitting monitor --model model.joblib --data prod.csv
    dynamic_refitting refit --model model.joblib --data new_data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

logger = logging.getLogger("dynamic_refitting.cli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train a pipeline from a YAML/JSON config."""
    from dynamic_refitting.config import PipelineConfig
    from dynamic_refitting.autopipe import AutoPipeBoost, AutoPipeLogreg

    config_path = Path(args.config)
    if config_path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            with open(config_path) as f:
                cfg_dict = yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML is required for YAML config files.")
            sys.exit(1)
    else:
        with open(config_path) as f:
            cfg_dict = json.load(f)

    config = PipelineConfig.from_dict(cfg_dict.get("pipeline", cfg_dict))
    data = pd.read_csv(args.data)
    y = data[config.target_col]
    X = data.drop(columns=[config.target_col])

    model_type = cfg_dict.get("model_type", "boost")
    if model_type == "boost":
        pipe = AutoPipeBoost(
            config=config,
            optuna_n_trials=cfg_dict.get("optuna_n_trials", 20),
        )
    else:
        pipe = AutoPipeLogreg(config=config)

    pipe.fit(X, y, run_cv=True)

    output = args.output or "model.joblib"
    pipe.save(output)
    logger.info("Model saved to %s", output)

    if args.metrics_output:
        with open(args.metrics_output, "w") as f:
            json.dump(pipe.train_metrics_, f, indent=2, default=str)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a saved model on a dataset."""
    from dynamic_refitting.config import ScoringPipeline
    from dynamic_refitting.utils.metrics import calc_metrics

    pipeline = ScoringPipeline.load(args.model)
    data = pd.read_csv(args.data)
    target = args.target or "target"
    y = data[target]
    X = data.drop(columns=[target])

    proba = pipeline.predict_proba(X)
    scores = proba[:, 1] if proba.ndim == 2 else proba
    metrics = calc_metrics(y.values, scores)
    print(json.dumps(metrics, indent=2, default=str))


def cmd_monitor(args: argparse.Namespace) -> None:
    """Run drift monitoring on production data."""
    from dynamic_refitting.config import ScoringPipeline
    from dynamic_refitting.monitoring.drift import FeatureDriftDetector

    pipeline = ScoringPipeline.load(args.model)
    train_data = pd.read_csv(args.train_data)
    prod_data = pd.read_csv(args.data)
    target = args.target or "target"

    detector = FeatureDriftDetector(psi_threshold=args.psi_threshold or 0.2)
    train_X = train_data.drop(columns=[target], errors="ignore")
    detector.fit(train_X)
    detector.transform(prod_data.drop(columns=[target], errors="ignore"))

    report = {
        "drifted_features": detector.drifted_features,
        "n_drifted": len(detector.drifted_features),
        "details": detector.drift_report_,
    }
    print(json.dumps(report, indent=2, default=str))


def cmd_refit(args: argparse.Namespace) -> None:
    """Refit a model with new data."""
    from dynamic_refitting.config import ScoringPipeline

    pipeline = ScoringPipeline.load(args.model)
    data = pd.read_csv(args.data)
    target = args.target or "target"
    y = data[target]
    X = data.drop(columns=[target])

    pipeline.fit(X, y)
    output = args.output or args.model
    pipeline.save(output)
    logger.info("Refitted model saved to %s", output)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dynamic_refitting",
        description="CLI for the dynamic_refitting scoring library.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train a pipeline.")
    p_train.add_argument("--config", required=True, help="Config YAML/JSON.")
    p_train.add_argument("--data", required=True, help="Training CSV.")
    p_train.add_argument("--output", help="Output model path.")
    p_train.add_argument("--metrics-output", help="Output metrics JSON path.")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a model.")
    p_eval.add_argument("--model", required=True, help="Saved model path.")
    p_eval.add_argument("--data", required=True, help="Evaluation CSV.")
    p_eval.add_argument("--target", default="target", help="Target column.")

    # monitor
    p_mon = sub.add_parser("monitor", help="Monitor for drift.")
    p_mon.add_argument("--model", required=True, help="Saved model path.")
    p_mon.add_argument("--train-data", required=True, help="Training CSV.")
    p_mon.add_argument("--data", required=True, help="Production CSV.")
    p_mon.add_argument("--target", default="target")
    p_mon.add_argument("--psi-threshold", type=float, default=0.2)

    # refit
    p_refit = sub.add_parser("refit", help="Refit a model.")
    p_refit.add_argument("--model", required=True, help="Saved model path.")
    p_refit.add_argument("--data", required=True, help="New data CSV.")
    p_refit.add_argument("--target", default="target")
    p_refit.add_argument("--output", help="Output model path.")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "refit":
        cmd_refit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
