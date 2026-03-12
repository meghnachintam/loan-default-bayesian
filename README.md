# Bayesian Loan Default Risk Modeling

A practical Bayesian credit-risk project that estimates **loan default probability** from borrower and loan attributes, then exposes predictions through both a **CLI** and a **Streamlit app**.

This repository combines:

- a full notebook workflow for data analysis and model development,
- an exported Bayesian-logistic inference bundle (`logit_app_bundle.pkl`), and
- lightweight interfaces for scoring single or batch records.

---

## Table of Contents

- [What this project does](#what-this-project-does)
- [Repository contents](#repository-contents)
- [Quickstart](#quickstart)
- [Running predictions (CLI)](#running-predictions-cli)
- [Running the Streamlit app](#running-the-streamlit-app)
- [Input schema](#input-schema)
- [Modeling approach](#modeling-approach)
- [Evaluation summary](#evaluation-summary)
- [Interpretability output](#interpretability-output)
- [Deployment notes](#deployment-notes)
- [Troubleshooting](#troubleshooting)

---

## What this project does

This project predicts a borrower’s probability of default (`Status = 1`) using Bayesian methods designed for:

- **probabilistic estimates** (not just hard class labels),
- **calibrated risk scoring**, and
- **analyst-friendly interpretation**.

The currently selected production model is a **Bayesian Logistic Regression** model bundle used for inference.

---

## Repository contents

| File | Purpose |
| --- | --- |
| `BML_Credit_Project.ipynb` | End-to-end analysis, Bayesian model training, and evaluation workflow |
| `logit_app_bundle.pkl` | Serialized inference bundle (preprocess pipeline + coefficients) |
| `predict.py` | Command-line inference utility for JSON records |
| `streamlit_app.py` | Main Streamlit interface for single/batch scoring |
| `app.py` | Compatibility Streamlit entrypoint forwarding to `streamlit_app.py` |
| `loan_desktop_app.py` | Additional local app entrypoint included in the project |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python runtime declaration for hosted Streamlit environments |

---

## Quickstart

### 1) Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Verify required artifact

Make sure this file exists at repo root:

- `logit_app_bundle.pkl`

Without it, both the app and CLI inference will fail.

---

## Running predictions (CLI)

The CLI accepts one JSON object or a list of objects and returns `default_probability` values in `[0, 1]`.

### Example command

```bash
python predict.py --bundle logit_app_bundle.pkl --input-json sample_input.json
```

### Expected output shape

```json
{
  "default_probability": [0.137, 0.482, 0.911]
}
```

### Notes

- If required fields are missing, the script auto-adds known training columns and fills missing values with `NaN` before preprocessing.
- The bundle includes preprocessing and model coefficients, so retraining is not required for inference.

---

## Running the Streamlit app

### Local

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`).

### App behavior

- Paste JSON payload in the text area (`{...}` or `[{...}, {...}]`)
- Click **Predict default probability**
- Review predictions in a table
- Download results as CSV

`app.py` is preserved as a compatibility wrapper for environments still pointing to that filename.

---

## Input schema

### Target variable

| Variable | Meaning |
| --- | --- |
| `Status` | `0 = performing`, `1 = default` |

### Numeric predictors

- `loan_amount`
- `rate_of_interest`
- `Credit_Score`
- `LTV`
- `income`
- `dtir1`
- `Interest_rate_spread`
- `Upfront_charges`
- `property_value`
- `term`

### Categorical predictors

- `loan_type`
- `occupancy_type`
- `credit_type`
- `Gender`
- `loan_purpose`
- `Region`
- `Security_Type`
- `approv_in_adv`
- `loan_limit`
- `business_or_commercial`

---

## Modeling approach

Three Bayesian models were developed and compared:

1. **Bayesian Logistic Regression**
2. **Hierarchical Bayesian Logistic Regression** (region-level partial pooling)
3. **Bayesian GAM** (spline-based nonlinearity)

### Data preparation summary

- Stratified split: **70% train / 15% validation / 15% test**
- Numeric imputation: median
- Categorical missingness: explicit placeholder category
- Standardization + one-hot encoding via preprocessing pipeline

---

## Evaluation summary

### Validation

| Model | AUROC | PR-AUC | LogLoss | Brier | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Bayesian Logistic** | **0.7927** | **0.7115** | **0.3815** | **0.1164** | **0.8559** |
| Hierarchical Logistic | 0.7764 | 0.6936 | 0.5170 | 0.1695 | 0.7573 |
| Bayesian GAM | 0.5355 | 0.2459 | 1.1010 | 0.3438 | 0.4978 |

### Test

| Model | AUROC | Brier | Accuracy |
| --- | ---: | ---: | ---: |
| **Bayesian Logistic** | **0.7951** | **0.1148** | **0.8584** |
| Hierarchical Logistic | 0.7816 | 0.1690 | 0.7610 |
| Bayesian GAM | 0.5290 | 0.3471 | 0.4914 |

**Selected model:** Bayesian Logistic Regression (best combined discrimination + calibration + stability).

---

## Interpretability output

The workflow supports analyst-style explanation of risk by combining:

- predicted probability,
- risk banding (e.g., low/moderate/elevated/high), and
- feature contribution summaries.

Example narrative pattern:

> “Predicted default probability is 33.1% (elevated risk). The strongest risk-increasing drivers are A/B/C, partially offset by X/Y/Z.”

---

## Deployment notes

For Streamlit Community Cloud:

- **Main file path:** `streamlit_app.py`
- **Runtime:** Python `3.11` (from `runtime.txt`)
- Ensure `logit_app_bundle.pkl` is committed at repository root

---

## Troubleshooting

### `Could not find logit_app_bundle.pkl`

Place the `.pkl` bundle in the project root (same folder as `streamlit_app.py` and `predict.py`).

### `Invalid JSON` in Streamlit

Ensure payload is valid JSON (double quotes, no trailing commas).

### Bundle/pickle compatibility issues

`predict.py` includes a small scikit-learn compatibility patch for older serialized column-transformer symbols.

---

## Tech stack

- Python
- PyMC
- ArviZ
- scikit-learn
- NumPy
- pandas
- matplotlib
- Streamlit

---

## Conclusion

This repository demonstrates a complete Bayesian credit-risk workflow from exploratory analysis through deployment-oriented inference. The selected Bayesian logistic model provides a strong practical balance of **performance**, **calibration**, and **interpretability** for loan default risk scoring.
