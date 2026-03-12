#!/usr/bin/env python3
"""Streamlit UI for scoring loan default risk from logit_app_bundle.pkl."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import load_bundle, predict_default_probability


st.set_page_config(page_title="Loan Default Probability", page_icon="📉", layout="centered")
st.title("Bayesian Loan Default Risk Scorer")
st.caption("Score one or many records using the uploaded logit_app_bundle.pkl")

bundle_path = Path("logit_app_bundle.pkl")
if not bundle_path.exists():
    st.error("Could not find logit_app_bundle.pkl in the project root.")
    st.stop()

bundle = load_bundle(bundle_path)
training_columns = bundle.get("training_columns") or []

st.subheader("Input")
st.write(
    "Paste JSON for a single record (`{...}`) or batch records (`[{...}, {...}]`)."
)

example_payload = (
    {col: None for col in training_columns[: min(10, len(training_columns))]}
    if training_columns
    else {"loan_amount": 250000, "Credit_Score": 700}
)
raw_json = st.text_area(
    "Payload JSON",
    value=json.dumps(example_payload, indent=2),
    height=240,
)

if st.button("Predict default probability", type="primary"):
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()

    records = payload if isinstance(payload, list) else [payload]

    try:
        probs = predict_default_probability(bundle, records)
    except Exception as exc:  # surfaced to user in app
        st.error(f"Prediction failed: {exc}")
        st.stop()

    result_df = pd.DataFrame(records)
    result_df["default_probability"] = probs

    st.subheader("Predictions")
    st.dataframe(result_df, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        data=result_df.to_csv(index=False),
        file_name="loan_default_predictions.csv",
        mime="text/csv",
    )
