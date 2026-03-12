#!/usr/bin/env python3
"""Run inference with the uploaded Bayesian-logit app bundle (.pkl)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_bundle(bundle_path: Path) -> dict:
    with bundle_path.open("rb") as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
            raise RuntimeError(
                "Failed to load model bundle. This bundle was trained with scikit-learn 1.6.1; "
                "please install a compatible version (e.g. `pip install scikit-learn==1.6.1`)."
            ) from exc


def predict_default_probability(bundle: dict, records: list[dict]) -> list[float]:
    preprocess = bundle["preprocess_pipeline"]
    beta = np.asarray(bundle["beta_mean"], dtype=float)
    intercept = float(bundle["intercept_mean"])

    frame = pd.DataFrame.from_records(records)

    training_columns = bundle.get("training_columns")
    if training_columns:
        missing = [col for col in training_columns if col not in frame.columns]
        for col in missing:
            frame[col] = np.nan
        frame = frame[training_columns]

    transformed = preprocess.transform(frame)
    linear_term = transformed @ beta + intercept
    probabilities = _sigmoid(np.asarray(linear_term).reshape(-1))
    return probabilities.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict loan default probabilities using logit_app_bundle.pkl and JSON input rows."
        )
    )
    parser.add_argument(
        "--bundle",
        default="logit_app_bundle.pkl",
        help="Path to the serialized model bundle (.pkl).",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file containing either one object or a list of objects.",
    )
    args = parser.parse_args()

    bundle = load_bundle(Path(args.bundle))
    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload if isinstance(payload, list) else [payload]
    probs = predict_default_probability(bundle, records)

    print(json.dumps({"default_probability": probs}, indent=2))


if __name__ == "__main__":
    main()
