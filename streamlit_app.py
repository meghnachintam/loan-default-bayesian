import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE_PATH = APP_DIR / "bayesian_credit_models" / "gam_app_bundle.pkl"


@dataclass
class GamArtifacts:
    preprocess_gam: Any
    beta: np.ndarray
    intercept: float
    feature_names: list[str]
    feature_variable_map: dict[str, str]
    numeric_features: list[str]
    categorical_features: list[str]
    categorical_options: dict[str, list[str]]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@st.cache_resource
def load_artifacts(bundle_path: str) -> GamArtifacts:
    path = Path(bundle_path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    required = {
        "preprocess_gam",
        "beta",
        "intercept",
        "feature_names",
        "feature_variable_map",
        "numeric_features",
        "categorical_features",
        "categorical_options",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Bundle is missing keys: {sorted(missing)}")

    return GamArtifacts(
        preprocess_gam=payload["preprocess_gam"],
        beta=np.asarray(payload["beta"], dtype=float),
        intercept=float(payload["intercept"]),
        feature_names=list(payload["feature_names"]),
        feature_variable_map=dict(payload["feature_variable_map"]),
        numeric_features=list(payload["numeric_features"]),
        categorical_features=list(payload["categorical_features"]),
        categorical_options=dict(payload["categorical_options"]),
    )


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
        "loan_type": "loan type",
        "occupancy_type": "occupancy type",
        "credit_type": "credit type",
        "Gender": "gender",
        "loan_purpose": "loan purpose",
    }
    return mapping.get(v, v.replace("_", " "))


def format_value(var: str, value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if var == "rate_of_interest":
        return f"{float(value):.2f}%"
    if var == "Credit_Score":
        return f"{float(value):.0f}"
    if var == "LTV":
        return f"{float(value):.1f}%"
    if var == "dtir1":
        return f"{float(value):.1f}"
    if var in {"income", "loan_amount"}:
        return f"${float(value):,.0f}"
    return str(value)


def explain_numeric_variable(var: str, raw_value: Any, contribution: float) -> str:
    direction_up = contribution > 0

    if pd.isna(raw_value):
        return (
            f"{pretty_var_name(var).capitalize()} is missing, and the model treats missingness "
            f"as part of the borrower risk profile."
        )

    raw_value = float(raw_value)

    if var == "Credit_Score":
        if raw_value < 580:
            base = (
                f"The borrower's credit score is {raw_value:.0f}, which falls in a very weak credit range. "
                f"This level is typically associated with prior repayment issues or limited credit strength."
            )
        elif raw_value < 670:
            base = (
                f"The borrower's credit score is {raw_value:.0f}, which is below prime quality. "
                f"This range generally signals weaker historical credit performance and more limited borrowing quality."
            )
        elif raw_value < 740:
            base = (
                f"The credit score is {raw_value:.0f}, which is in a near-prime to solid range. "
                f"This usually reflects acceptable repayment history but not the strongest borrower profile in the portfolio."
            )
        else:
            base = (
                f"The credit score is {raw_value:.0f}, which is strong. "
                f"Borrowers in this range generally show better historical repayment behavior and stronger credit quality."
            )
        tail = (
            " In this case, the model interprets credit score as increasing risk."
            if direction_up
            else " In this case, the model interprets credit score as reducing risk."
        )
        return base + tail

    if var == "LTV":
        if raw_value < 60:
            base = (
                f"The loan-to-value ratio is {raw_value:.1f}%, indicating substantial borrower equity in the property. "
                f"Low leverage generally provides a stronger financial cushion and reduces default incentives."
            )
        elif raw_value < 80:
            base = (
                f"The loan-to-value ratio is {raw_value:.1f}%, which is moderate. "
                f"This suggests the borrower still retains meaningful equity, limiting credit risk under normal conditions."
            )
        elif raw_value < 90:
            base = (
                f"The loan-to-value ratio is {raw_value:.1f}%, which indicates relatively high leverage. "
                f"At this level, the borrower has a smaller equity buffer, making repayment stress more concerning."
            )
        else:
            base = (
                f"The loan-to-value ratio is {raw_value:.1f}%, which is very high. "
                f"This suggests limited borrower equity, and loans at this leverage level are generally more vulnerable to delinquency if financial stress occurs."
            )
        tail = (
            " The model treats this as a risk-increasing feature."
            if direction_up
            else " The model treats this as a risk-reducing feature for this borrower profile."
        )
        return base + tail

    if var == "dtir1":
        if raw_value < 20:
            base = (
                f"The debt-to-income ratio is {raw_value:.1f}, which is very low. "
                f"This implies debt obligations consume a relatively small share of income, supporting repayment capacity."
            )
        elif raw_value < 36:
            base = (
                f"The debt-to-income ratio is {raw_value:.1f}, which is generally manageable. "
                f"This indicates the borrower's payment burden is likely sustainable relative to income."
            )
        elif raw_value < 43:
            base = (
                f"The debt-to-income ratio is {raw_value:.1f}, which is somewhat elevated. "
                f"At this level, debt service begins to place more pressure on household cash flow."
            )
        else:
            base = (
                f"The debt-to-income ratio is {raw_value:.1f}, which is high. "
                f"This suggests a large share of income is already committed to debt payments, increasing affordability stress."
            )
        tail = (
            " The model associates this with higher delinquency risk."
            if direction_up
            else " The model associates this with lower delinquency risk."
        )
        return base + tail

    if var == "rate_of_interest":
        if raw_value < 3.5:
            base = (
                f"The interest rate is {raw_value:.2f}%, which is very low. "
                f"Lower rates reduce monthly payment burden and are often observed among stronger-quality borrowers."
            )
        elif raw_value < 5.0:
            base = (
                f"The interest rate is {raw_value:.2f}%, which is moderate. "
                f"This creates a manageable payment burden for most borrowers, although it still contributes to repayment cost."
            )
        else:
            base = (
                f"The interest rate is {raw_value:.2f}%, which is high relative to lower-rate loans. "
                f"Higher rates increase monthly payments and may also reflect somewhat riskier loan pricing."
            )
        tail = (
            " In this prediction, the rate contributes upward pressure on delinquency risk."
            if direction_up
            else " In this prediction, the rate contributes downward pressure on delinquency risk."
        )
        return base + tail

    if var == "income":
        if raw_value < 40000:
            base = (
                f"Reported income is {format_value(var, raw_value)}, which is relatively low. "
                f"Lower income can leave borrowers with less flexibility to absorb payment shocks or unexpected expenses."
            )
        elif raw_value < 80000:
            base = (
                f"Reported income is {format_value(var, raw_value)}, which is moderate. "
                f"This can support repayment, although financial resilience may still depend on leverage and debt burden."
            )
        elif raw_value < 150000:
            base = (
                f"Reported income is {format_value(var, raw_value)}, which is solid. "
                f"Higher income generally improves repayment capacity and reduces sensitivity to moderate financial stress."
            )
        else:
            base = (
                f"Reported income is {format_value(var, raw_value)}, which is high. "
                f"Borrowers at this income level typically have greater capacity to sustain payments and absorb shocks."
            )
        tail = (
            " In this case, the model still views income as contributing to higher risk."
            if direction_up
            else " In this case, the model views income as an offsetting strength that lowers predicted risk."
        )
        return base + tail

    if var == "loan_amount":
        if raw_value < 150000:
            base = (
                f"The loan amount is {format_value(var, raw_value)}, which is relatively small. "
                f"Smaller balances may reduce absolute payment burden, although risk still depends on borrower income and leverage."
            )
        elif raw_value < 400000:
            base = (
                f"The loan amount is {format_value(var, raw_value)}, which is moderate for the portfolio. "
                f"At this level, risk depends more on repayment capacity, leverage, and pricing than on size alone."
            )
        else:
            base = (
                f"The loan amount is {format_value(var, raw_value)}, which is large. "
                f"Larger loans increase exposure size, but they can also be associated with stronger underwriting or higher-income borrowers."
            )
        tail = (
            " Here, the model interprets loan size as increasing risk."
            if direction_up
            else " Here, the model interprets loan size as reducing risk relative to the broader borrower profile."
        )
        return base + tail

    return (
        f"{pretty_var_name(var).capitalize()} has value {format_value(var, raw_value)}. "
        + (
            "The model treats this as increasing predicted risk."
            if direction_up
            else "The model treats this as reducing predicted risk."
        )
    )


def explain_categorical_variable(var: str, raw_value: Any, contribution: float) -> str:
    direction_up = contribution > 0
    display_value = "missing" if pd.isna(raw_value) else str(raw_value)

    if var == "Gender":
        base = (
            f"The borrower is categorized as {display_value}. "
            f"This variable captures patterns present in the training data rather than a direct causal financial mechanism."
        )
    elif var == "loan_type":
        base = (
            f"The loan is categorized as {display_value}. "
            f"Different loan structures may have different repayment behavior, pricing, or underwriting characteristics in the historical data."
        )
    elif var == "occupancy_type":
        base = (
            f"The occupancy type is {display_value}. "
            f"Owner-occupied, investor, and secondary-property loans can perform differently because borrower incentives and financial priorities vary."
        )
    elif var == "credit_type":
        base = (
            f"The credit reporting category is {display_value}. "
            f"This may proxy differences in credit bureau reporting patterns or borrower segments observed in the training data."
        )
    elif var == "loan_purpose":
        base = (
            f"The loan purpose is {display_value}. "
            f"Borrowers taking loans for different purposes may exhibit different delinquency behavior depending on refinancing motives, purchase context, or financial stress."
        )
    else:
        base = (
            f"{pretty_var_name(var).capitalize()} is {display_value}. "
            f"This category contributes based on relationships learned from the training data."
        )

    tail = (
        " For this borrower, the category pushes predicted risk upward."
        if direction_up
        else " For this borrower, the category helps lower predicted risk."
    )
    return base + tail


def make_raw_row(artifacts: GamArtifacts, form_values: dict[str, Any]) -> pd.DataFrame:
    row = {}
    for col in artifacts.numeric_features:
        value = form_values.get(col)
        row[col] = np.nan if value in ("", None) else float(value)
    for col in artifacts.categorical_features:
        value = form_values.get(col)
        row[col] = pd.NA if value in ("", None, "__MISSING__") else str(value)
    return pd.DataFrame([row])


def score_row(artifacts: GamArtifacts, raw_row_df: pd.DataFrame) -> tuple[float, np.ndarray]:
    x_row = artifacts.preprocess_gam.transform(raw_row_df)
    eta = artifacts.intercept + x_row @ artifacts.beta
    prob = float(sigmoid(np.asarray(eta).reshape(-1))[0])
    return prob, np.asarray(x_row).reshape(-1)


def compute_variable_contributions(
    artifacts: GamArtifacts,
    transformed_row: np.ndarray,
) -> pd.DataFrame:
    contribution = transformed_row * artifacts.beta
    df = pd.DataFrame(
        {
            "feature": artifacts.feature_names,
            "variable": [artifacts.feature_variable_map.get(name, name) for name in artifacts.feature_names],
            "contribution": contribution,
        }
    )
    grouped = (
        df.groupby("variable", as_index=False)["contribution"]
        .sum()
        .assign(abs_contribution=lambda x: x["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
    )
    return grouped


def build_detailed_explanations(
    raw_row: pd.Series,
    contributions: pd.DataFrame,
    numeric_features: list[str],
    top_n: int = 3,
) -> tuple[list[str], list[str]]:
    up = contributions[contributions["contribution"] > 0].head(top_n)
    down = contributions[contributions["contribution"] < 0].head(top_n)

    up_text = []
    down_text = []

    for _, record in up.iterrows():
        var = record["variable"]
        raw_val = raw_row.get(var, np.nan)
        if var in numeric_features:
            up_text.append(explain_numeric_variable(var, raw_val, record["contribution"]))
        else:
            up_text.append(explain_categorical_variable(var, raw_val, record["contribution"]))

    for _, record in down.iterrows():
        var = record["variable"]
        raw_val = raw_row.get(var, np.nan)
        if var in numeric_features:
            down_text.append(explain_numeric_variable(var, raw_val, record["contribution"]))
        else:
            down_text.append(explain_categorical_variable(var, raw_val, record["contribution"]))

    return up_text, down_text


def generate_statement(
    prob: float,
    contributions: pd.DataFrame,
    raw_row: pd.Series,
    numeric_features: list[str],
    top_n: int = 3,
) -> str:
    risk = risk_label(prob)
    statement = (
        f"The model predicts an estimated delinquency probability of {prob * 100:.1f}% "
        f"for this borrower, which corresponds to a {risk} risk classification. "
    )

    up = contributions[contributions["contribution"] > 0].head(top_n)
    down = contributions[contributions["contribution"] < 0].head(top_n)

    if not up.empty:
        statement += (
            "The strongest factors increasing risk are "
            + ", ".join(pretty_var_name(v) for v in up["variable"].tolist())
            + ". "
        )
    if not down.empty:
        statement += (
            "The strongest offsetting factors are "
            + ", ".join(pretty_var_name(v) for v in down["variable"].tolist())
            + ". "
        )

    up_text, down_text = build_detailed_explanations(raw_row, contributions, numeric_features, top_n=top_n)
    if up_text:
        statement += "Risk-increasing interpretation: " + " ".join(up_text) + " "
    if down_text:
        statement += "Risk-reducing interpretation: " + " ".join(down_text)
    return statement


def render_numeric_inputs(artifacts: GamArtifacts) -> dict[str, Any]:
    values: dict[str, Any] = {}
    defaults = {
        "loan_amount": 250000.0,
        "rate_of_interest": 4.25,
        "Credit_Score": 700.0,
        "LTV": 80.0,
        "income": 75000.0,
        "dtir1": 35.0,
    }
    for col in artifacts.numeric_features:
        label = pretty_var_name(col).capitalize()
        values[col] = st.number_input(
            label,
            value=float(defaults.get(col, 0.0)),
            step=1.0 if col != "rate_of_interest" else 0.1,
            format="%.2f" if col == "rate_of_interest" else "%.1f",
        )
    return values


def render_categorical_inputs(artifacts: GamArtifacts) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for col in artifacts.categorical_features:
        options = ["__MISSING__"] + artifacts.categorical_options.get(col, [])
        values[col] = st.selectbox(
            pretty_var_name(col).capitalize(),
            options=options,
            format_func=lambda x: "Missing / unknown" if x == "__MISSING__" else str(x),
        )
    return values


def render_contribution_table(contributions: pd.DataFrame) -> pd.DataFrame:
    out = contributions.copy()
    out["variable"] = out["variable"].map(pretty_var_name)
    return out.rename(columns={"variable": "Variable", "contribution": "Contribution"})[
        ["Variable", "Contribution", "abs_contribution"]
    ]


def main() -> None:
    st.set_page_config(page_title="Bayesian GAM Loan Risk", page_icon=":bar_chart:", layout="wide")
    st.title("Bayesian GAM Loan Risk Explainer")
    st.caption("GAM-only scoring with raw-input explanations and grouped spline/categorical contributions.")

    bundle_path = st.sidebar.text_input("Artifact bundle path", value=str(DEFAULT_BUNDLE_PATH))

    try:
        artifacts = load_artifacts(bundle_path)
    except Exception as exc:
        st.error(f"Could not load GAM bundle: {exc}")
        st.stop()

    with st.form("loan_form"):
        left, right = st.columns(2)
        with left:
            numeric_inputs = render_numeric_inputs(artifacts)
        with right:
            categorical_inputs = render_categorical_inputs(artifacts)
        submitted = st.form_submit_button("Score Borrower", use_container_width=True)

    if not submitted:
        st.info("Enter the borrower attributes and click Score Borrower.")
        return

    form_values = {**numeric_inputs, **categorical_inputs}
    raw_row_df = make_raw_row(artifacts, form_values)
    prob, transformed_row = score_row(artifacts, raw_row_df)
    contributions = compute_variable_contributions(artifacts, transformed_row)
    statement = generate_statement(
        prob=prob,
        contributions=contributions,
        raw_row=raw_row_df.iloc[0],
        numeric_features=artifacts.numeric_features,
        top_n=3,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Delinquency Probability", f"{prob * 100:.2f}%")
    with c2:
        st.metric("Risk Band", risk_label(prob).title())

    st.subheader("Analyst Explanation")
    st.write(statement)

    st.subheader("Top Variable Contributions")
    st.dataframe(render_contribution_table(contributions.head(10)), use_container_width=True)

    with st.expander("Raw Input Row"):
        st.json(json.loads(raw_row_df.to_json(orient="records"))[0])


if __name__ == "__main__":
    main()
