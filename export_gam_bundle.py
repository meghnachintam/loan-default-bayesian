import pickle
from pathlib import Path

import numpy as np


def build_feature_variable_map(feature_names, numeric_features, categorical_features):
    mapping = {}
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)

    for name in feature_names:
        matched = None

        for col in numeric_set:
            if name == col or name.startswith(f"{col}_sp"):
                matched = col
                break

        if matched is None:
            for col in categorical_set:
                if name == col or name.startswith(f"{col}_"):
                    matched = col
                    break

        mapping[name] = matched or name

    return mapping


def export_gam_bundle(
    preprocess_gam,
    trace_gam_same,
    feature_names_gam_same,
    numeric_features,
    categorical_features,
    train_proc,
    output_path="bayesian_credit_models/gam_app_bundle.pkl",
):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    beta = trace_gam_same.posterior["beta"].mean(dim=("chain", "draw")).values
    intercept = trace_gam_same.posterior["intercept"].mean(dim=("chain", "draw")).values

    categorical_options = {}
    for col in categorical_features:
        values = (
            train_proc[col]
            .astype("string")
            .dropna()
            .unique()
            .tolist()
        )
        categorical_options[col] = sorted(str(v) for v in values if str(v) != "MISSING")

    payload = {
        "preprocess_gam": preprocess_gam,
        "beta": np.asarray(beta, dtype=float),
        "intercept": float(intercept),
        "feature_names": list(feature_names_gam_same),
        "feature_variable_map": build_feature_variable_map(
            feature_names=feature_names_gam_same,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        ),
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "categorical_options": categorical_options,
    }

    with output.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved GAM app bundle to {output.resolve()}")
