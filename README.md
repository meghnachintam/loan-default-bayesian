# Bayesian Loan Default Risk Modeling

## Overview

This project studies mortgage-style loan default using a Bayesian modeling framework designed to produce both accurate probabilities and analyst-friendly explanations. The goal is not just binary classification. The goal is to estimate calibrated default risk, compare alternative Bayesian specifications, and translate the final model output into a narrative that explains why a borrower is being flagged as lower or higher risk.

The modeling work compares three approaches:

1. Bayesian logistic regression
2. Hierarchical Bayesian logistic regression
3. Bayesian generalized additive modeling (GAM)

The final deliverable is a scoring workflow that combines preprocessing, posterior mean inference, risk banding, and a natural-language analyst response.

## Business Framing

Loan default modeling is useful when credit teams need more than a yes or no answer. In practice, underwriting and risk management teams care about:

- the estimated probability of default
- whether that probability is well calibrated
- which borrower and loan attributes are driving the result
- whether the explanation is understandable enough for human review

This project is structured around those requirements. The chosen model therefore emphasizes probabilistic reliability and interpretability, not only raw classification performance.

## Dataset

The dataset contains borrower- and loan-level variables typically used in mortgage underwriting and portfolio risk analysis. The target variable is `Status`, where:

| Value | Meaning |
| --- | --- |
| `0` | Performing loan |
| `1` | Defaulted loan |

### Core numeric predictors

| Variable | Role in credit analysis |
| --- | --- |
| `loan_amount` | Exposure size and payment burden |
| `rate_of_interest` | Loan pricing and monthly payment pressure |
| `Credit_Score` | Borrower credit quality |
| `LTV` | Leverage and borrower equity cushion |
| `income` | Repayment capacity |
| `dtir1` | Debt burden relative to income |
| `Interest_rate_spread` | Relative pricing signal |
| `Upfront_charges` | Additional financing cost |
| `property_value` | Collateral scale and equity context |
| `term` | Repayment horizon |

### Core categorical predictors

| Variable | Role in credit analysis |
| --- | --- |
| `loan_purpose` | Borrowing context |
| `Region` | Geographic segmentation |
| `Gender` | Historical segmentation in the source data |
| `loan_type` | Product structure |
| `occupancy_type` | Property usage / borrower incentive structure |
| `credit_type` | Credit bureau / reporting segment |
| `Security_Type` | Loan security classification |
| `approv_in_adv` | Pre-approval signal |
| `loan_limit` | Loan limit category |
| `business_or_commercial` | Commercial vs non-commercial classification |

## Exploratory Analysis

The notebook begins with a basic audit of the source data: schema inspection, target distribution, missingness, univariate numeric distributions, categorical default-rate comparisons, and simple correlation review.

Several variables show moderate missingness, concentrated mostly in financial fields rather than the target:

| Variable | Approx. missing rate |
| --- | --- |
| `Upfront_charges` | 27% |
| `Interest_rate_spread` | 25% |
| `rate_of_interest` | 24% |
| `dtir1` | 16% |
| `property_value` | 10% |
| `LTV` | 10% |

That pattern matters because it argues for preprocessing instead of row deletion. Removing all incomplete records would unnecessarily shrink the dataset and potentially bias the sample toward cleaner applications.

<img width="789" height="590" alt="image" src="https://github.com/user-attachments/assets/63347f47-abc0-48d8-b332-da62ec9cc945" />


The EDA also showed the strongest practical relationships in the expected places:

- higher interest rates aligned with higher default risk
- very high LTV values showed a sharp increase in risk
- higher debt-to-income ratios were associated with repayment stress
- higher income generally reduced predicted risk

These relationships make logistic-style models a strong baseline because the dominant signals are mostly monotonic and financially intuitive.

## Data Preparation

The pipeline standardizes raw records before model estimation:

1. String columns are trimmed and blank-like values are normalized to missing.
2. Numeric columns are coerced to floats.
3. The target is converted to a clean binary indicator.
4. Rows with missing target values are dropped.
5. The data is split into training, validation, and test sets.

The preprocessing layer used for the logistic model is:

- numeric imputation with median values
- numeric standardization with `StandardScaler`
- categorical imputation with a fallback category
- one-hot encoding with `handle_unknown="ignore"`

This design is important for deployment because the exact same transformation logic can be reused for new borrower records without rebuilding the training notebook.

## Modeling Strategy

### 1. Bayesian Logistic Regression

This is the core baseline:

`P(default) = sigmoid(alpha + X beta)`

Each coefficient is assigned a prior and estimated through posterior sampling. This model is attractive because it is:

- straightforward to interpret
- naturally probabilistic
- well aligned with common credit-risk assumptions
- stable when the underlying effects are mostly linear on the log-odds scale

### 2. Hierarchical Bayesian Logistic Regression

The hierarchical version extends the baseline by allowing regional intercept variation:

`P(default) = sigmoid(alpha_region + X beta)`

with region-level effects partially pooled around a global intercept. The intent is to capture baseline geographic variation without fully fitting separate models by region.

This is useful when regional market conditions may matter but the analyst still wants a shared model structure.

### 3. Bayesian GAM

The GAM replaces the simple linear numeric terms with spline expansions. This allows the model to capture nonlinear relationships more flexibly:

`P(default) = sigmoid(alpha + f1(x1) + f2(x2) + ... )`

In principle this can model curved effects in income, credit score, leverage, and debt burden. In practice it also expands the feature space substantially, which raises the complexity of inference.

## Evaluation Framework

The models are compared on multiple dimensions because credit scoring is not just a discrimination problem.

| Metric | Why it matters |
| --- | --- |
| AUROC | Ranking ability across default / non-default cases |
| PR-AUC | Performance on the default class when classes are imbalanced |
| LogLoss | Quality of probabilistic predictions |
| Brier score | Calibration of predicted probabilities |
| Accuracy | Thresholded classification summary |

This combination gives a fuller picture. A model can rank borrowers reasonably well and still be poorly calibrated. In lending contexts, calibration matters because downstream decisions often depend on the actual probability estimate rather than only the rank order.

## Results

### Validation performance

| Model | AUROC | PR-AUC | LogLoss | Brier | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian Logistic | 0.7927 | 0.7115 | 0.3815 | 0.1164 | 0.8559 |
| Hierarchical Logistic | 0.7764 | 0.6936 | 0.5170 | 0.1695 | 0.7573 |
| Bayesian GAM | 0.5355 | 0.2459 | 1.1010 | 0.3438 | 0.4978 |

### Test performance

| Model | AUROC | Brier | Accuracy |
| --- | ---: | ---: | ---: |
| Bayesian Logistic | 0.7951 | 0.1148 | 0.8584 |
| Hierarchical Logistic | 0.7816 | 0.1690 | 0.7610 |
| Bayesian GAM | 0.5290 | 0.3471 | 0.4914 |

## Why the Bayesian Logistic Model Won

The Bayesian logistic regression model was selected as the final model because it delivered the best overall balance of:

- discrimination
- calibration
- stability
- interpretability

The hierarchical model was reasonable but did not add enough predictive value to justify the extra complexity. That suggests borrower-level financial variables already explain most of the useful variation, and region-specific baseline adjustments only add limited signal.

The Bayesian GAM underperformed badly. The most likely explanation is that spline expansion introduced too much flexibility relative to the signal available in the data. Instead of improving fit, that complexity weakened out-of-sample performance and calibration.

## Analyst Explanation Engine

The final logistic model is paired with an interpretation layer that converts a score into an analyst-facing summary.

For a single borrower record, the workflow is:

1. apply the saved preprocessing pipeline
2. compute the posterior-mean linear predictor
3. convert the log-odds to default probability
4. assign a risk band
5. decompose feature-level contributions
6. aggregate encoded features back to raw business variables
7. generate a natural-language explanation

The risk bands are:

| Probability | Risk band |
| --- | --- |
| `< 10%` | low |
| `10% to <25%` | moderate |
| `25% to <40%` | elevated |
| `>= 40%` | high |

The explanation engine highlights both sides of the decision:

- the strongest factors pushing risk upward
- the strongest offsetting factors reducing risk

That makes the final output easier to use in analyst review than a raw probability alone.

## Project Components

| File | Purpose |
| --- | --- |
| `BML_Credit_Project.ipynb` | End-to-end notebook for EDA, preprocessing, model estimation, and interpretation logic |
| `logit_app_bundle.pkl` | Saved preprocessing + coefficient bundle for inference |
| `app.py` | Lightweight scoring interface for entering borrower values and receiving an analyst response |
| `loan_desktop_app.py` | Desktop application variant kept in the repository |
| `requirements.txt` | Runtime dependencies for the app |

## Main Takeaways

- Default risk in this dataset is driven primarily by standard credit fundamentals: leverage, debt burden, pricing, income, and credit quality.
- The simplest Bayesian model performed best because the main predictive relationships were already captured well by a linear log-odds structure.
- More complex model classes do not automatically improve credit-risk performance.
- Interpretability is strongest when modeling choices and explanation logic are aligned from the start.

## Conclusion

This project demonstrates a full Bayesian credit-risk workflow: exploratory analysis, preprocessing, model comparison, probabilistic evaluation, model selection, and explanation generation. The final Bayesian logistic regression model provides the best combination of calibrated probabilities and practical interpretability, making it the most defensible choice for analyst-facing default risk assessment in this repository.
