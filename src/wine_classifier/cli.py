"""Command line: train, evaluate and save a model; or predict from a CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .model import LABELS, evaluate, load_dataset

DEFAULT_DATA = Path("data/wine_dataset.csv")


def cmd_train(args: argparse.Namespace) -> int:
    x, y = load_dataset(args.data)
    model, result = evaluate(x, y, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(x.columns)}, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(result.report)
    print(f"Hold-out accuracy: {result.accuracy:.4f}")
    print(f"5-fold CV accuracy: {result.cv_accuracy_mean:.4f} +/- {result.cv_accuracy_std:.4f}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    bundle = joblib.load(args.model)
    df = pd.read_csv(args.csv)
    missing = [c for c in bundle["features"] if c not in df]
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {missing}")
    names = {v: k for k, v in LABELS.items()}
    df["predicted_style"] = [names[int(p)] for p in bundle["model"].predict(df[bundle["features"]])]
    print(df[["predicted_style"]].value_counts().to_string())
    if args.output:
        df.to_csv(args.output, index=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train, evaluate and save a model")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--out", type=Path, default=Path("models"))
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Classify wines from a CSV")
    predict.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    predict.add_argument("--csv", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    predict.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
