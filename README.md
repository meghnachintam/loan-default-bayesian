# Bayesian Loan Default Risk Modeling

## Use the uploaded `.pkl` model bundle for inference

## Run in Streamlit

Yes — if you want to use the uploaded `logit_app_bundle.pkl` through Streamlit, run:

```bash
streamlit run streamlit_app.py
```

Then paste a single JSON record or a JSON array of records in the app and click **Predict default probability**.

If you already uploaded `logit_app_bundle.pkl`, you can score new records directly without re-training.

1. Create a JSON file with one record (object) or many records (array of objects).
2. Run:

```bash
python predict.py --bundle logit_app_bundle.pkl --input-json sample_input.json
```

The script prints `default_probability` values (between 0 and 1) in JSON format.

## Project Overview

This project builds a **Bayesian machine learning pipeline to estimate loan default probability** using borrower financial characteristics and loan attributes. The objective is to generate **well-calibrated probabilistic risk estimates** and provide **interpretable explanations** that could support underwriting, portfolio monitoring, and credit risk management.

Rather than using traditional deterministic classification models, this project adopts a **Bayesian modeling framework**. Bayesian methods allow us to estimate uncertainty in model parameters and generate probability distributions over predicted risk, which is particularly valuable in financial decision-making.

Three Bayesian models are developed and compared:

1. **Bayesian Logistic Regression**
2. **Hierarchical Bayesian Logistic Regression**
3. **Bayesian Generalized Additive Model (GAM)**

Model performance is evaluated across multiple metrics, and the best-performing model is selected for deployment in an **analyst interpretation engine** that produces natural-language explanations for predicted default risk.

---

# Dataset

The dataset contains borrower and loan-level attributes commonly used in mortgage underwriting and credit risk modeling.

### Target Variable

**Status**

| Value | Meaning         |
| ----- | --------------- |
| 0     | Loan performing |
| 1     | Loan default    |

---

## Numeric Features

These variables represent borrower financial strength and loan structure.

| Variable             | Description                                     |
| -------------------- | ----------------------------------------------- |
| loan_amount          | Total loan size                                 |
| rate_of_interest     | Interest rate applied to the loan               |
| Credit_Score         | Borrower credit score                           |
| LTV                  | Loan-to-value ratio                             |
| income               | Borrower income                                 |
| dtir1                | Debt-to-income ratio                            |
| Interest_rate_spread | Difference between loan rate and reference rate |
| Upfront_charges      | Upfront fees charged                            |
| property_value       | Estimated property value                        |
| term                 | Loan term length                                |

These variables capture **borrower creditworthiness, repayment capacity, leverage, and loan structure**, which are primary drivers of default risk.

---

## Categorical Features

Categorical variables capture loan structure, borrower attributes, and institutional features.

| Variable               | Description                            |
| ---------------------- | -------------------------------------- |
| loan_type              | Type of loan product                   |
| occupancy_type         | Property occupancy classification      |
| credit_type            | Credit reporting source                |
| Gender                 | Borrower gender category               |
| loan_purpose           | Purpose of the loan                    |
| Region                 | Geographic region                      |
| Security_Type          | Security classification                |
| approv_in_adv          | Whether loan was pre-approved          |
| loan_limit             | Loan limit classification              |
| business_or_commercial | Business vs residential classification |

These variables may capture **structural differences in underwriting, regional markets, and borrower behavior**.

---

# Exploratory Data Analysis

Initial exploratory analysis examined the structure of the dataset, missingness patterns, and relationships between borrower attributes and default probability.

### Missing Data

Several financial variables exhibit moderate missingness.

| Variable             | Missing Rate |
| -------------------- | ------------ |
| Upfront_charges      | ~27%         |
| Interest_rate_spread | ~25%         |
| rate_of_interest     | ~24%         |
| dtir1                | ~16%         |
| property_value       | ~10%         |
| LTV                  | ~10%         |

Handling strategy:

* **Numeric variables:** median imputation
* **Categorical variables:** missing category placeholder

This approach preserves the full dataset while allowing the models to learn patterns associated with missing information.

---

### Feature Distributions

Key numeric variables display several notable characteristics:

* **Loan amounts** are right-skewed with most loans between $150k–$400k.
* **Borrower income** is strongly right-skewed with high-income outliers.
* **Credit scores** span a wide range (~520–900).
* **Loan-to-value ratios** cluster around typical mortgage leverage levels (60–90%).

These distributions indicate a **heterogeneous borrower population**, motivating the use of probabilistic models.

---

### Relationships with Default Risk

Binned default rate analysis revealed several important patterns:

* **Higher interest rates → higher default risk**
* **Higher loan-to-value ratios → sharply increasing risk at extreme leverage**
* **Higher debt-to-income ratios → significantly higher default probability**
* **Higher income → lower default risk**

These patterns confirm that borrower financial variables carry meaningful predictive signal for default outcomes.

---

# Data Preparation

The dataset is partitioned into three subsets using stratified sampling to preserve the default rate.

| Dataset    | Percentage |
| ---------- | ---------- |
| Training   | 70%        |
| Validation | 15%        |
| Test       | 15%        |

Preprocessing includes:

* Numeric feature standardization
* Median imputation
* One-hot encoding for categorical variables
* Missing value sentinel categories

Separate **design matrices** are constructed for each model type.

---

# Modeling Approaches

## Bayesian Logistic Regression

The Bayesian logistic regression model estimates default probability using a standard logistic regression structure with Bayesian priors.

Model form:

P(default) = sigmoid(α + Xβ)

Where:

* **X** = borrower feature matrix
* **β** = regression coefficients
* **α** = intercept

This model is widely used in credit risk modeling because borrower financial variables often affect default risk in **predictable, monotonic ways**.

---

## Hierarchical Bayesian Logistic Regression

The hierarchical model extends the logistic regression framework by allowing **regional variation in baseline default risk**.

P(default) = sigmoid(α_region + Xβ)

Where:

α_region = α + σ_region * offset_region

This allows the model to partially pool information across geographic regions and capture differences in economic conditions or housing markets.

---

## Bayesian Generalized Additive Model (GAM)

The GAM introduces **spline-based nonlinear transformations** for numeric variables.

P(default) = sigmoid(α + f₁(x₁) + f₂(x₂) + ...)

Where each function f(x) is modeled using spline basis expansions.

This approach is designed to capture nonlinear relationships between predictors and default probability.

---

# Model Evaluation

Model performance was evaluated using multiple metrics that measure both discrimination and calibration.

| Metric      | Description                                |
| ----------- | ------------------------------------------ |
| AUROC       | Ability to separate default vs non-default |
| PR-AUC      | Precision-recall performance               |
| LogLoss     | Probabilistic accuracy                     |
| Brier Score | Calibration of predicted probabilities     |
| Accuracy    | Classification performance                 |

---

# Model Performance

### Validation Performance

| Model                 | AUROC      | PR-AUC     | LogLoss    | Brier      | Accuracy   |
| --------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Bayesian Logistic** | **0.7927** | **0.7115** | **0.3815** | **0.1164** | **0.8559** |
| Hierarchical Logistic | 0.7764     | 0.6936     | 0.5170     | 0.1695     | 0.7573     |
| Bayesian GAM          | 0.5355     | 0.2459     | 1.1010     | 0.3438     | 0.4978     |

### Test Performance

| Model                 | AUROC      | Brier      | Accuracy   |
| --------------------- | ---------- | ---------- | ---------- |
| **Bayesian Logistic** | **0.7951** | **0.1148** | **0.8584** |
| Hierarchical Logistic | 0.7816     | 0.1690     | 0.7610     |
| Bayesian GAM          | 0.5290     | 0.3471     | 0.4914     |

---

# Model Selection

The **Bayesian Logistic Regression model performed best across all evaluation metrics**.

Reasons:

1. Highest AUROC and PR-AUC
2. Lowest Brier score (best calibrated probabilities)
3. Highest classification accuracy
4. Stable performance across training, validation, and test datasets

The hierarchical model performed slightly worse, suggesting that **regional variation did not significantly improve predictive performance** beyond borrower-level financial variables.

The GAM model performed poorly due to the **high dimensionality introduced by spline expansions**, which likely led to unstable parameter estimation relative to the available signal in the dataset.

---

# Final Model

The final selected model is **Bayesian Logistic Regression**.

This model offers the best balance between:

* predictive performance
* probabilistic calibration
* interpretability

These characteristics are essential for financial risk modeling where **transparent and reliable probability estimates are required**.

---

# Analyst Explanation Engine

To enhance interpretability, the project includes an **analyst explanation engine** that converts model predictions into natural-language explanations.

For any borrower record, the system:

1. Predicts default probability
2. Assigns a risk classification
3. Identifies the strongest drivers of risk
4. Generates a human-readable explanation

Example output:

> The model predicts a default probability of **33.1%**, corresponding to an **elevated risk classification**.
> The strongest factors increasing risk are **gender classification, regional location, and approval status**, while the strongest offsetting factors are **credit reporting type, security type, and occupancy status**.

This functionality allows analysts to **understand why a borrower is considered higher or lower risk**, improving transparency and decision support.

---

# Key Insights

Several borrower characteristics strongly influence default probability:

* **Debt-to-income ratio**
* **Loan-to-value ratio**
* **Interest rate**
* **Borrower income**
* **Credit score**

These financial variables capture the borrower’s **repayment capacity and leverage**, which are fundamental determinants of credit risk.

The results demonstrate that **relatively simple models with strong financial predictors can outperform more complex models** when the underlying relationships are mostly monotonic and well-behaved.

---

# Technologies Used

* Python
* PyMC
* ArviZ
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

# Repository Structure

```
BML_Credit_Project.ipynb
README.md
Loan_Default.csv
```

---

# Conclusion

This project demonstrates how Bayesian modeling can be applied to credit risk prediction while maintaining **probabilistic rigor and interpretability**. The final Bayesian logistic regression model provides strong predictive performance and produces calibrated risk estimates that can support financial decision-making and risk management workflows.
