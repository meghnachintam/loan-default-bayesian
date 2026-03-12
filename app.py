from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_CANDIDATES = [
    APP_DIR / "logit_app_bundle.pkl",
    Path("/Users/meghna/Downloads/logit_app_bundle.pkl"),
]

PRETTY_NAMES = {
    "loan_amount": "Loan Amount",
    "rate_of_interest": "Interest Rate",
    "Credit_Score": "Credit Score",
    "LTV": "Loan-to-Value Ratio",
    "income": "Income",
    "dtir1": "Debt-to-Income Ratio",
    "Interest_rate_spread": "Interest Rate Spread",
    "Upfront_charges": "Upfront Charges",
    "property_value": "Property Value",
    "term": "Loan Term",
    "loan_type": "Loan Type",
    "occupancy_type": "Occupancy Type",
    "credit_type": "Credit Type",
    "Gender": "Gender",
    "loan_purpose": "Loan Purpose",
    "Region": "Region",
    "Security_Type": "Security Type",
    "approv_in_adv": "Approval In Advance",
    "loan_limit": "Loan Limit",
    "business_or_commercial": "Business Or Commercial",
}

NUMERIC_HELP = {
    "loan_amount": "Example: 250000",
    "rate_of_interest": "Example: 4.25",
    "Credit_Score": "Example: 720",
    "LTV": "Example: 80",
    "income": "Example: 9000",
    "dtir1": "Example: 35",
    "Interest_rate_spread": "Example: 0.45",
    "Upfront_charges": "Example: 1200",
    "property_value": "Example: 350000",
    "term": "Example: 360",
}

DEFAULT_NUMERIC_VALUES = {
    "loan_amount": "250000",
    "rate_of_interest": "4.25",
    "Credit_Score": "720",
    "LTV": "80",
    "income": "9000",
    "dtir1": "35",
    "Interest_rate_spread": "0.45",
    "Upfront_charges": "1200",
    "property_value": "350000",
    "term": "360",
}


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-value))


def risk_label(probability: float) -> str:
    if probability < 0.10:
        return "Low"
    if probability < 0.25:
        return "Moderate"
    if probability < 0.40:
        return "Elevated"
    return "High"


def pretty_var_name(name: str) -> str:
    return PRETTY_NAMES.get(name, name.replace("_", " ").title())


def format_value(name: str, value: object) -> str:
    if pd.isna(value):
        return "missing"
    if name in {"rate_of_interest", "Interest_rate_spread"}:
        return f"{float(value):.2f}%"
    if name == "Credit_Score":
        return f"{float(value):.0f}"
    if name == "LTV":
        return f"{float(value):.1f}%"
    if name == "dtir1":
        return f"{float(value):.1f}"
    if name in {"income", "loan_amount", "property_value", "Upfront_charges"}:
        return f"${float(value):,.0f}"
    if name == "term":
        return f"{float(value):.0f} months"
    return str(value)


def feature_to_raw_var(feature_name: str, raw_vars: list[str]) -> str:
    if feature_name in raw_vars:
        return feature_name

    for raw_var in sorted(raw_vars, key=len, reverse=True):
        if feature_name.startswith(f"{raw_var}_"):
            return raw_var

    return feature_name


def explain_numeric_variable(var: str, raw_value: object, contribution: float) -> str:
    direction_up = contribution > 0

    if pd.isna(raw_value):
        return (
            f"{pretty_var_name(var)} is missing, and the model treats missingness "
            f"as part of the borrower risk profile."
        )

    value = float(raw_value)

    if var == "Credit_Score":
        if value < 580:
            base = (
                f"The borrower's credit score is {value:.0f}, which falls in a very weak "
                f"credit range. This level is generally associated with weaker historical "
                f"repayment performance."
            )
        elif value < 670:
            base = (
                f"The borrower's credit score is {value:.0f}, which is below prime quality. "
                f"This typically indicates a somewhat riskier borrower profile."
            )
        elif value < 740:
            base = (
                f"The credit score is {value:.0f}, which is in a near-prime to solid range. "
                f"This usually reflects acceptable repayment history."
            )
        else:
            base = (
                f"The credit score is {value:.0f}, which is strong. Borrowers in this range "
                f"generally demonstrate stronger credit quality."
            )
    elif var == "LTV":
        if value < 60:
            base = (
                f"The loan-to-value ratio is {value:.1f}%, indicating substantial borrower equity. "
                f"Lower leverage generally reduces default incentives."
            )
        elif value < 80:
            base = (
                f"The loan-to-value ratio is {value:.1f}%, which is moderate. This suggests "
                f"the borrower retains a meaningful equity cushion."
            )
        elif value < 90:
            base = (
                f"The loan-to-value ratio is {value:.1f}%, which indicates relatively high leverage. "
                f"At this level, the borrower has less room to absorb financial stress."
            )
        else:
            base = (
                f"The loan-to-value ratio is {value:.1f}%, which is very high. This implies "
                f"limited equity and greater vulnerability if repayment stress occurs."
            )
    elif var == "dtir1":
        if value < 20:
            base = (
                f"The debt-to-income ratio is {value:.1f}, which is very low. This implies "
                f"a relatively light debt burden compared with income."
            )
        elif value < 36:
            base = (
                f"The debt-to-income ratio is {value:.1f}, which is generally manageable. "
                f"This suggests repayment burden is likely sustainable."
            )
        elif value < 43:
            base = (
                f"The debt-to-income ratio is {value:.1f}, which is somewhat elevated. "
                f"At this level, debt obligations place more pressure on cash flow."
            )
        else:
            base = (
                f"The debt-to-income ratio is {value:.1f}, which is high. This indicates "
                f"significant repayment burden relative to income."
            )
    elif var == "rate_of_interest":
        if value < 3.5:
            base = (
                f"The interest rate is {value:.2f}%, which is relatively low. Lower rates reduce "
                f"monthly payment burden and often correspond to stronger borrower quality."
            )
        elif value < 5.0:
            base = (
                f"The interest rate is {value:.2f}%, which is moderate. This still contributes "
                f"to repayment cost but remains within a typical range."
            )
        else:
            base = (
                f"The interest rate is {value:.2f}%, which is high relative to lower-rate loans. "
                f"Higher rates raise monthly payment burden and may reflect risk-based pricing."
            )
    elif var == "income":
        if value < 4000:
            base = (
                f"Reported income is {format_value(var, value)}, which is relatively low. Lower "
                f"income can limit financial flexibility and reduce shock absorption capacity."
            )
        elif value < 8000:
            base = (
                f"Reported income is {format_value(var, value)}, which is moderate. This may "
                f"support repayment, but resilience still depends on leverage and debt burden."
            )
        elif value < 15000:
            base = (
                f"Reported income is {format_value(var, value)}, which is solid. Higher income "
                f"generally improves repayment capacity."
            )
        else:
            base = (
                f"Reported income is {format_value(var, value)}, which is high. Borrowers at this "
                f"level typically have more capacity to sustain payments."
            )
    elif var == "loan_amount":
        if value < 150000:
            base = (
                f"The loan amount is {format_value(var, value)}, which is relatively small. "
                f"Smaller balances may reduce absolute payment burden."
            )
        elif value < 400000:
            base = (
                f"The loan amount is {format_value(var, value)}, which is moderate for this "
                f"portfolio. At this level, risk depends more on affordability and leverage than "
                f"size alone."
            )
        else:
            base = (
                f"The loan amount is {format_value(var, value)}, which is large. Larger loans "
                f"increase exposure size, although they may also reflect stronger underwriting."
            )
    elif var == "property_value":
        base = (
            f"The property value is {format_value(var, value)}. Property value affects leverage "
            f"and borrower equity when considered together with loan size."
        )
    elif var == "term":
        base = (
            f"The loan term is {format_value(var, value)}. Loan maturity affects payment "
            f"structure and repayment horizon."
        )
    else:
        base = f"{pretty_var_name(var)} has value {format_value(var, value)}."

    tail = (
        " The model treats this as increasing predicted risk."
        if direction_up
        else " The model treats this as reducing predicted risk."
    )
    return base + tail


def explain_categorical_variable(var: str, raw_value: object, contribution: float) -> str:
    direction_up = contribution > 0
    value = "missing" if pd.isna(raw_value) else str(raw_value)

    if var == "Gender":
        base = (
            f"The borrower is categorized as {value}. This variable reflects historical "
            f"patterns in the training data rather than a direct causal financial mechanism."
        )
    elif var == "loan_type":
        base = (
            f"The loan is categorized as {value}. Different loan structures may show different "
            f"repayment behavior and underwriting characteristics."
        )
    elif var == "occupancy_type":
        base = (
            f"The occupancy type is {value}. Owner-occupied, investor, and secondary-property "
            f"loans can perform differently because borrower incentives vary."
        )
    elif var == "credit_type":
        base = (
            f"The credit reporting category is {value}. This may proxy differences in bureau "
            f"reporting patterns or borrower segments."
        )
    elif var == "loan_purpose":
        base = (
            f"The loan purpose is {value}. Different borrowing purposes may correspond to "
            f"different underlying risk behavior."
        )
    elif var == "Region":
        base = (
            f"The borrower is associated with the {value} region. Regional effects may reflect "
            f"differences in local economic or housing conditions."
        )
    else:
        base = (
            f"{pretty_var_name(var)} is {value}. This category contributes based on "
            f"relationships learned from the training data."
        )

    tail = (
        " For this borrower, the category pushes predicted risk upward."
        if direction_up
        else " For this borrower, the category helps lower predicted risk."
    )
    return base + tail


@st.cache_resource
def load_bundle() -> dict:
    for candidate in BUNDLE_CANDIDATES:
        if candidate.exists():
            with candidate.open("rb") as handle:
                return pickle.load(handle)
    raise FileNotFoundError("Could not find logit_app_bundle.pkl.")


def build_input_frame(raw_values: dict, all_columns: list[str]) -> pd.DataFrame:
    row = {column: raw_values.get(column, pd.NA) for column in all_columns}
    return pd.DataFrame([row], columns=all_columns)


def compute_variable_contributions(
    x_row: np.ndarray,
    beta_mean: np.ndarray,
    feature_names: np.ndarray,
    raw_vars: list[str],
) -> pd.DataFrame:
    contributions = np.asarray(x_row).reshape(-1) * np.asarray(beta_mean).reshape(-1)
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": contributions,
        }
    )
    frame["variable"] = frame["feature"].map(lambda name: feature_to_raw_var(name, raw_vars))
    grouped = frame.groupby("variable", as_index=False)["contribution"].sum()
    grouped["abs_contribution"] = grouped["contribution"].abs()
    return grouped.sort_values("abs_contribution", ascending=False)


def build_detailed_explanations(
    raw_row: pd.Series,
    contributions: pd.DataFrame,
    numeric_features: list[str],
    top_n: int = 3,
) -> tuple[list[str], list[str]]:
    up = contributions[contributions["contribution"] > 0].head(top_n)
    down = contributions[contributions["contribution"] < 0].head(top_n)

    up_text: list[str] = []
    down_text: list[str] = []

    for _, row in up.iterrows():
        variable = row["variable"]
        raw_value = raw_row.get(variable, pd.NA)
        if variable in numeric_features:
            up_text.append(explain_numeric_variable(variable, raw_value, float(row["contribution"])))
        else:
            up_text.append(explain_categorical_variable(variable, raw_value, float(row["contribution"])))

    for _, row in down.iterrows():
        variable = row["variable"]
        raw_value = raw_row.get(variable, pd.NA)
        if variable in numeric_features:
            down_text.append(explain_numeric_variable(variable, raw_value, float(row["contribution"])))
        else:
            down_text.append(explain_categorical_variable(variable, raw_value, float(row["contribution"])))

    return up_text, down_text


def generate_statement(
    probability: float,
    contributions: pd.DataFrame,
    raw_row: pd.Series,
    numeric_features: list[str],
    top_n: int = 3,
) -> str:
    statement = (
        f"The model predicts an estimated default probability of {probability * 100:.1f}% "
        f"for this borrower, which corresponds to a {risk_label(probability).lower()} risk "
        f"classification. "
    )

    up = contributions[contributions["contribution"] > 0].head(top_n)
    down = contributions[contributions["contribution"] < 0].head(top_n)

    if not up.empty:
        statement += (
            "The strongest factors increasing risk are "
            + ", ".join(pretty_var_name(value).lower() for value in up["variable"].tolist())
            + ". "
        )

    if not down.empty:
        statement += (
            "The strongest offsetting factors are "
            + ", ".join(pretty_var_name(value).lower() for value in down["variable"].tolist())
            + ". "
        )

    up_text, down_text = build_detailed_explanations(raw_row, contributions, numeric_features, top_n=top_n)
    if up_text:
        statement += "Risk-increasing interpretation: " + " ".join(up_text) + " "
    if down_text:
        statement += "Risk-reducing interpretation: " + " ".join(down_text)

    return statement.strip()


def score_borrower(raw_values: dict, bundle: dict) -> dict:
    preprocess = bundle["preprocess_pipeline"]
    numeric_features = list(bundle["numeric_features"])
    categorical_features = list(bundle["categorical_features"])
    raw_vars = numeric_features + categorical_features

    input_frame = build_input_frame(raw_values, raw_vars)
    x_row = preprocess.transform(input_frame)

    beta_mean = np.asarray(bundle["beta_mean"])
    intercept_mean = float(np.asarray(bundle["intercept_mean"]).reshape(-1)[0])
    eta = intercept_mean + np.asarray(x_row @ beta_mean).reshape(-1)[0]
    probability = float(sigmoid(eta))

    contributions = compute_variable_contributions(
        x_row=x_row,
        beta_mean=beta_mean,
        feature_names=np.asarray(bundle["feature_names"]),
        raw_vars=raw_vars,
    )
    raw_row = pd.Series(raw_values)

    return {
        "probability": probability,
        "risk_band": risk_label(probability),
        "statement": generate_statement(probability, contributions, raw_row, numeric_features),
        "contributions": contributions,
    }


def parse_numeric_input(raw_text: str, field_name: str) -> float | pd._libs.missing.NAType:
    value = raw_text.strip()
    if not value:
        return pd.NA

    cleaned = value.replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"{pretty_var_name(field_name)} must be a number.") from exc


def category_options(bundle: dict) -> dict[str, list[str]]:
    preprocess = bundle["preprocess_pipeline"]
    categorical_features = list(bundle["categorical_features"])
    encoder = preprocess.named_transformers_["cat"].named_steps["onehot"]

    options: dict[str, list[str]] = {}
    for feature, values in zip(categorical_features, encoder.categories_):
        cleaned = [str(value) for value in values if str(value) != "__MISSING__"]
        options[feature] = [""] + cleaned
    return options


def main() -> None:
    st.set_page_config(page_title="Credit Risk Analyst App", page_icon=":bar_chart:", layout="wide")

    st.title("Credit Risk Analyst App")
    st.write("Enter borrower and loan values to get the model score and analyst-style response.")

    try:
        bundle = load_bundle()
    except Exception as exc:
        st.error(f"Bundle load failed: {exc}")
        st.stop()

    numeric_features = list(bundle["numeric_features"])
    categorical_features = list(bundle["categorical_features"])
    category_map = category_options(bundle)

    with st.form("credit_form"):
        left, right = st.columns(2)
        numeric_values: dict[str, object] = {}
        categorical_values: dict[str, object] = {}

        for index, feature in enumerate(numeric_features):
            container = left if index % 2 == 0 else right
            with container:
                numeric_values[feature] = st.text_input(
                    pretty_var_name(feature),
                    value=DEFAULT_NUMERIC_VALUES.get(feature, ""),
                    help=NUMERIC_HELP.get(feature, ""),
                )

        for index, feature in enumerate(categorical_features):
            container = left if index % 2 == 0 else right
            with container:
                opts = category_map[feature]
                default_index = 0
                if len(opts) > 1:
                    default_index = 1
                categorical_values[feature] = st.selectbox(
                    pretty_var_name(feature),
                    options=opts,
                    index=default_index,
                )

        submitted = st.form_submit_button("Generate analyst response", use_container_width=True)

    if not submitted:
        st.info("Submit the form to score a borrower.")
        return

    try:
        parsed_numeric = {
            feature: parse_numeric_input(str(value), feature)
            for feature, value in numeric_values.items()
        }
    except ValueError as exc:
        st.error(str(exc))
        return

    parsed_categorical = {
        feature: (value if str(value).strip() else pd.NA)
        for feature, value in categorical_values.items()
    }

    raw_values = {**parsed_numeric, **parsed_categorical}
    results = score_borrower(raw_values, bundle)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Default Probability", f"{results['probability'] * 100:.1f}%")
    metric_cols[1].metric("Risk Band", results["risk_band"])

    st.subheader("Analyst Response")
    st.write(results["statement"])

    contributions = results["contributions"].copy()
    contributions["direction"] = np.where(
        contributions["contribution"] >= 0,
        "Increases risk",
        "Reduces risk",
    )
    contributions["contribution"] = contributions["contribution"].round(4)
    contributions["abs_contribution"] = contributions["abs_contribution"].round(4)
    contributions["variable"] = contributions["variable"].map(pretty_var_name)

    st.subheader("Top Drivers")
    st.dataframe(
        contributions[["variable", "direction", "contribution", "abs_contribution"]].head(10),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
