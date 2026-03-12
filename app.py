import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

st.set_page_config(page_title="Bayesian Loan Default Analyst", layout="wide")

NUM_COLS = [
    "loan_amount",
    "rate_of_interest",
    "Credit_Score",
    "LTV",
    "income",
    "dtir1",
    "Interest_rate_spread",
    "Upfront_charges",
    "property_value",
    "term",
]

CAT_COLS = [
    "loan_purpose",
    "Region",
    "Gender",
    "loan_type",
    "occupancy_type",
    "credit_type",
    "Security_Type",
    "approv_in_adv",
    "loan_limit",
    "business_or_commercial",
]


def risk_label(prob: float) -> str:
    if prob < 0.10:
        return "low"
    if prob < 0.25:
        return "moderate"
    if prob < 0.40:
        return "elevated"
    return "high"


def pretty_var_name(v: str) -> str:
    mapping = {
        "loan_amount": "loan amount",
        "rate_of_interest": "interest rate",
        "Credit_Score": "credit score",
        "LTV": "loan-to-value ratio",
        "income": "income",
        "dtir1": "debt-to-income ratio",
        "Interest_rate_spread": "interest rate spread",
        "Upfront_charges": "upfront charges",
        "property_value": "property value",
        "term": "loan term",
        "loan_type": "loan type",
        "occupancy_type": "occupancy type",
        "credit_type": "credit type",
        "Gender": "gender",
        "loan_purpose": "loan purpose",
        "Region": "region",
        "Security_Type": "security type",
        "approv_in_adv": "approval in advance",
        "loan_limit": "loan limit",
        "business_or_commercial": "business/commercial flag",
    }
    return mapping.get(v, v.replace("_", " "))


def feature_to_raw_var(feature_name: str) -> str:
    if feature_name in RAW_VARS_LOGIT:
        return feature_name
    for var in sorted(RAW_VARS_LOGIT, key=len, reverse=True):
        if feature_name.startswith(var + "_"):
            return var
    return feature_name


def load_bundle(path: str = "logit_app_bundle.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict):
        raise ValueError("Expected a dictionary bundle in logit_app_bundle.pkl")

    preprocess = bundle.get("preprocess_pipeline") or bundle.get("preprocess_logit")
    beta_mean = bundle.get("beta_mean")
    intercept_mean = bundle.get("intercept_mean")
    feature_names = bundle.get("feature_names")
    training_columns = bundle.get("training_columns")

    missing = [
        name
        for name, value in {
            "preprocess_pipeline": preprocess,
            "beta_mean": beta_mean,
            "intercept_mean": intercept_mean,
        }.items()
        if value is None
    ]
    if missing:
        raise KeyError(f"Bundle is missing required keys: {', '.join(missing)}")

    if feature_names is None and hasattr(preprocess, "get_feature_names_out"):
        feature_names = preprocess.get_feature_names_out()

    return preprocess, np.asarray(beta_mean), float(intercept_mean), feature_names, training_columns


def _to_1d_dense_array(values) -> np.ndarray:
    if sparse.issparse(values):
        return values.toarray().reshape(-1)
    return np.asarray(values).reshape(-1)


def score_row(raw_row: dict):
    row_df = pd.DataFrame([raw_row])

    for c in NUM_COLS + CAT_COLS:
        if c not in row_df.columns:
            row_df[c] = pd.NA

    row_df = row_df[NUM_COLS + CAT_COLS].copy()
    X_row = PREPROCESS.transform(row_df)

    eta = INTERCEPT_MEAN + X_row @ BETA_MEAN
    prob = 1 / (1 + np.exp(-_to_1d_dense_array(eta)))

    contrib = _to_1d_dense_array(X_row) * BETA_MEAN
    contrib_df = pd.DataFrame({"feature": FEATURE_NAMES, "contribution": contrib})
    contrib_df["variable"] = contrib_df["feature"].apply(feature_to_raw_var)
    contrib_df = (
        contrib_df.groupby("variable", as_index=False)["contribution"]
        .sum()
        .assign(abs_contribution=lambda d: d["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
    )

    return float(prob[0]), contrib_df


def build_statement(prob: float, contributions: pd.DataFrame, top_n: int = 3):
    up = contributions[contributions["contribution"] > 0].head(top_n)
    down = contributions[contributions["contribution"] < 0].head(top_n)

    statement = (
        f"Predicted default probability is {prob * 100:.1f}%, "
        f"which maps to **{risk_label(prob)} risk**. "
    )

    if len(up):
        statement += "Top risk-increasing factors: " + ", ".join(pretty_var_name(v) for v in up["variable"]) + ". "
    if len(down):
        statement += "Top risk-reducing factors: " + ", ".join(pretty_var_name(v) for v in down["variable"]) + "."
    return statement


st.title("Bayesian Loan Default Analyst Frontend")
st.caption("Input borrower/loan variables and get probability + analyst interpretation based on the notebook model bundle.")

bundle_path = Path("logit_app_bundle.pkl")
if not bundle_path.exists():
    st.error("`logit_app_bundle.pkl` not found in repository root.")
    st.stop()

try:
    PREPROCESS, BETA_MEAN, INTERCEPT_MEAN, FEATURE_NAMES, TRAINING_COLUMNS = load_bundle(str(bundle_path))
except Exception as exc:
    st.exception(exc)
    st.stop()

RAW_VARS_LOGIT = NUM_COLS + CAT_COLS
if FEATURE_NAMES is None:
    st.error("Could not resolve feature names from bundle/preprocessor.")
    st.stop()

if TRAINING_COLUMNS:
    st.info(f"Model was trained with {len(TRAINING_COLUMNS)} raw columns.")

with st.form("loan_input"):
    st.subheader("Numeric inputs")
    n1, n2 = st.columns(2)
    numeric_values = {}
    for idx, col in enumerate(NUM_COLS):
        target_col = n1 if idx % 2 == 0 else n2
        with target_col:
            numeric_values[col] = st.number_input(pretty_var_name(col).title(), value=0.0)

    st.subheader("Categorical inputs")
    c1, c2 = st.columns(2)
    categorical_values = {}
    for idx, col in enumerate(CAT_COLS):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            categorical_values[col] = st.text_input(pretty_var_name(col).title(), value="")

    top_n = st.slider("Top factors in interpretation", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("Score borrower")

if submitted:
    payload = {**numeric_values, **categorical_values}
    prob, contributions = score_row(payload)

    st.metric("Default Probability", f"{prob * 100:.2f}%")
    st.metric("Risk Band", risk_label(prob).title())

    st.markdown("### Analyst interpretation")
    st.write(build_statement(prob, contributions, top_n=top_n))

    st.markdown("### Variable contribution table")
    show_df = contributions[["variable", "contribution"]].copy()
    show_df["variable"] = show_df["variable"].map(pretty_var_name)
    st.dataframe(show_df, use_container_width=True)
