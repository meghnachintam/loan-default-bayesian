import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

DEFAULT_CSV_PATH = r"C:\Users\13107\Dropbox\Uchicago\Bayesian Machine Learning with Generative AI Applications\Final\Loan_Default.csv"

DEFAULT_NUMERIC_COLUMNS = [
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

DEFAULT_CATEGORICAL_COLUMNS = [
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

COLORS = {
    "bg": "#f4efe7",
    "panel": "#fbf8f2",
    "card": "#efe6d8",
    "ink": "#1f2933",
    "muted": "#6b7280",
    "accent": "#9a3412",
    "accent_soft": "#f59e0b",
    "good": "#166534",
    "border": "#d6c7b2",
}


def risk_label(prob):
    if prob < 0.10:
        return "low"
    if prob < 0.25:
        return "moderate"
    if prob < 0.40:
        return "elevated"
    return "high"


def pretty_var_name(v):
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
        "Interest_rate_spread": "interest rate spread",
        "Upfront_charges": "upfront charges",
        "property_value": "property value",
        "term": "term",
        "Region": "region",
        "Security_Type": "security type",
        "approv_in_adv": "approval timing",
        "loan_limit": "loan limit category",
        "business_or_commercial": "business/commercial flag",
    }
    return mapping.get(v, v.replace("_", " "))


def format_value(var, value):
    if pd.isna(value):
        return "missing"
    if var == "rate_of_interest":
        return f"{value:.2f}%"
    if var == "Credit_Score":
        return f"{value:.0f}"
    if var == "LTV":
        return f"{value:.1f}%"
    if var == "dtir1":
        return f"{value:.1f}"
    if var in {"income", "loan_amount", "property_value", "Upfront_charges"}:
        return f"${value:,.0f}"
    if var == "term":
        return f"{value:.0f} months"
    return str(value)


def infer_base_variable(feature_name, candidate_variables):
    for var in sorted(candidate_variables, key=len, reverse=True):
        if feature_name == var or feature_name.startswith(var + "_"):
            return var
    return feature_name.split("_sp")[0]


def explain_numeric_variable(var, raw_value, contribution):
    direction_up = contribution > 0

    if pd.isna(raw_value):
        return f"{pretty_var_name(var).capitalize()} is missing, and the model treats missingness as part of the borrower risk profile."

    if var == "Credit_Score":
        if raw_value < 580:
            base = f"The borrower's credit score is {raw_value:.0f}, which falls in a very weak credit range. This level is typically associated with prior repayment issues or limited credit strength."
        elif raw_value < 670:
            base = f"The borrower's credit score is {raw_value:.0f}, which is below prime quality. This range generally signals weaker historical credit performance and more limited borrowing quality."
        elif raw_value < 740:
            base = f"The credit score is {raw_value:.0f}, which is in a near-prime to solid range. This usually reflects acceptable repayment history but not the strongest borrower profile in the portfolio."
        else:
            base = f"The credit score is {raw_value:.0f}, which is strong. Borrowers in this range generally show better historical repayment behavior and stronger credit quality."
        tail = " In this case, the model interprets credit score as increasing risk." if direction_up else " In this case, the model interprets credit score as reducing risk."
        return base + tail

    if var == "LTV":
        if raw_value < 60:
            base = f"The loan-to-value ratio is {raw_value:.1f}%, indicating substantial borrower equity in the property. Low leverage generally provides a stronger financial cushion and reduces default incentives."
        elif raw_value < 80:
            base = f"The loan-to-value ratio is {raw_value:.1f}%, which is moderate. This suggests the borrower still retains meaningful equity, limiting credit risk under normal conditions."
        elif raw_value < 90:
            base = f"The loan-to-value ratio is {raw_value:.1f}%, which indicates relatively high leverage. At this level, the borrower has a smaller equity buffer, making repayment stress more concerning."
        else:
            base = f"The loan-to-value ratio is {raw_value:.1f}%, which is very high. This suggests limited borrower equity, and loans at this leverage level are generally more vulnerable to delinquency if financial stress occurs."
        tail = " The model treats this as a risk-increasing feature." if direction_up else " The model treats this as a risk-reducing feature for this borrower profile."
        return base + tail

    if var == "dtir1":
        if raw_value < 20:
            base = f"The debt-to-income ratio is {raw_value:.1f}, which is very low. This implies debt obligations consume a relatively small share of income, supporting repayment capacity."
        elif raw_value < 36:
            base = f"The debt-to-income ratio is {raw_value:.1f}, which is generally manageable. This indicates the borrower's payment burden is likely sustainable relative to income."
        elif raw_value < 43:
            base = f"The debt-to-income ratio is {raw_value:.1f}, which is somewhat elevated. At this level, debt service begins to place more pressure on household cash flow."
        else:
            base = f"The debt-to-income ratio is {raw_value:.1f}, which is high. This suggests a large share of income is already committed to debt payments, increasing affordability stress."
        tail = " The model associates this with higher delinquency risk." if direction_up else " The model associates this with lower delinquency risk."
        return base + tail

    if var == "rate_of_interest":
        if raw_value < 3.5:
            base = f"The interest rate is {raw_value:.2f}%, which is very low. Lower rates reduce monthly payment burden and are often observed among stronger-quality borrowers."
        elif raw_value < 5.0:
            base = f"The interest rate is {raw_value:.2f}%, which is moderate. This creates a manageable payment burden for most borrowers, although it still contributes to repayment cost."
        else:
            base = f"The interest rate is {raw_value:.2f}%, which is high relative to lower-rate loans. Higher rates increase monthly payments and may also reflect somewhat riskier loan pricing."
        tail = " In this prediction, the rate contributes upward pressure on delinquency risk." if direction_up else " In this prediction, the rate contributes downward pressure on delinquency risk."
        return base + tail

    if var == "income":
        if raw_value < 40000:
            base = f"Reported income is {format_value(var, raw_value)}, which is relatively low. Lower income can leave borrowers with less flexibility to absorb payment shocks or unexpected expenses."
        elif raw_value < 80000:
            base = f"Reported income is {format_value(var, raw_value)}, which is moderate. This can support repayment, although financial resilience may still depend on leverage and debt burden."
        elif raw_value < 150000:
            base = f"Reported income is {format_value(var, raw_value)}, which is solid. Higher income generally improves repayment capacity and reduces sensitivity to moderate financial stress."
        else:
            base = f"Reported income is {format_value(var, raw_value)}, which is high. Borrowers at this income level typically have greater capacity to sustain payments and absorb shocks."
        tail = " In this case, the model still views income as contributing to higher risk." if direction_up else " In this case, the model views income as an offsetting strength that lowers predicted risk."
        return base + tail

    if var == "loan_amount":
        if raw_value < 150000:
            base = f"The loan amount is {format_value(var, raw_value)}, which is relatively small. Smaller balances may reduce absolute payment burden, although risk still depends on borrower income and leverage."
        elif raw_value < 400000:
            base = f"The loan amount is {format_value(var, raw_value)}, which is moderate for the portfolio. At this level, risk depends more on repayment capacity, leverage, and pricing than on size alone."
        else:
            base = f"The loan amount is {format_value(var, raw_value)}, which is large. Larger loans increase exposure size, but they can also be associated with stronger underwriting or higher-income borrowers."
        tail = " Here, the model interprets loan size as increasing risk." if direction_up else " Here, the model interprets loan size as reducing risk relative to the broader borrower profile."
        return base + tail

    return f"{pretty_var_name(var).capitalize()} has value {format_value(var, raw_value)}. " + ("The model treats this as increasing predicted risk." if direction_up else "The model treats this as reducing predicted risk.")


def explain_categorical_variable(var, raw_value, contribution):
    direction_up = contribution > 0

    if var == "Gender":
        base = f"The borrower is categorized as {raw_value}. This variable captures patterns present in the training data rather than a direct causal financial mechanism."
    elif var == "loan_type":
        base = f"The loan is categorized as {raw_value}. Different loan structures may have different repayment behavior, pricing, or underwriting characteristics in the historical data."
    elif var == "occupancy_type":
        base = f"The occupancy type is {raw_value}. Owner-occupied, investor, and secondary-property loans can perform differently because borrower incentives and financial priorities vary."
    elif var == "credit_type":
        base = f"The credit reporting category is {raw_value}. This may proxy differences in credit bureau reporting patterns or borrower segments observed in the training data."
    elif var == "loan_purpose":
        base = f"The loan purpose is {raw_value}. Borrowers taking loans for different purposes may exhibit different delinquency behavior depending on refinancing motives, purchase context, or financial stress."
    else:
        base = f"{pretty_var_name(var).capitalize()} is {raw_value}. This category contributes based on relationships learned from the training data."

    tail = " For this borrower, the category pushes predicted risk upward." if direction_up else " For this borrower, the category helps lower predicted risk."
    return base + tail


class LoanAnalystApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Default Analyst App")
        self.root.geometry("1320x860")
        self.root.minsize(1160, 760)
        self.root.configure(bg=COLORS["bg"])

        self.df = None
        self.numeric_features = []
        self.categorical_features = []
        self.preprocess = None
        self.model = None
        self.feature_names = None

        self._configure_style()
        self._build_ui()
        self.path_var.set(DEFAULT_CSV_PATH)

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["ink"], font=("Georgia", 20, "bold"))
        style.configure("Sub.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Segoe UI", 10))
        style.configure("Field.TLabel", background=COLORS["panel"], foreground=COLORS["ink"], font=("Segoe UI", 10, "bold"))
        style.configure("Section.TLabel", background=COLORS["panel"], foreground=COLORS["ink"], font=("Georgia", 12, "bold"))
        style.configure("MetricLabel.TLabel", background=COLORS["card"], foreground=COLORS["muted"], font=("Segoe UI", 9, "bold"))
        style.configure("MetricValue.TLabel", background=COLORS["card"], foreground=COLORS["ink"], font=("Georgia", 15, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Accent.TButton", background=[("active", COLORS["accent_soft"]), ("!disabled", COLORS["accent"])], foreground=[("!disabled", "white")])
        style.configure("Plain.TButton", font=("Segoe UI", 10), padding=7)
        style.configure("App.TCombobox", padding=4)

    def _build_metric_card(self, parent, label_text, value_var):
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12))
        ttk.Label(card, text=label_text, style="MetricLabel.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=value_var, style="MetricValue.TLabel").pack(anchor="w", pady=(6, 0))
        return card

    def _make_text_panel(self, parent, title, height, wrap="word", font=("Segoe UI", 10)):
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=(14, 14))
        ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor="w")

        text_wrap = tk.WORD if wrap == "word" else tk.NONE
        text_frame = tk.Frame(frame, bg=COLORS["panel"])
        text_frame.pack(fill="both", expand=True, pady=(8, 0))

        text = tk.Text(
            text_frame,
            wrap=text_wrap,
            height=height,
            bg=COLORS["panel"],
            fg=COLORS["ink"],
            insertbackground=COLORS["ink"],
            relief="flat",
            padx=8,
            pady=8,
            font=font,
        )
        y_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=y_scroll.set)
        y_scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)

        return frame, text

    def _build_ui(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Loan Analyst Dashboard", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Fast borrower-level scoring with richer analyst explanations and a lighter local model.", style="Sub.TLabel").pack(anchor="w", pady=(2, 14))

        controls = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        controls.pack(fill="x")

        self.path_var = tk.StringVar()
        ttk.Label(controls, text="CSV path", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        path_entry = tk.Entry(controls, textvariable=self.path_var, width=92, bg="white", fg=COLORS["ink"], relief="flat", font=("Segoe UI", 10))
        path_entry.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(0, 12), pady=(6, 0), ipady=7)
        ttk.Button(controls, text="Browse", command=self.browse_file, style="Plain.TButton").grid(row=1, column=3, padx=(0, 8), pady=(6, 0))
        ttk.Button(controls, text="Train Model", command=self.train_model, style="Accent.TButton").grid(row=1, column=4, pady=(6, 0))

        ttk.Label(controls, text="Top factors", style="Field.TLabel").grid(row=2, column=0, sticky="w", pady=(16, 0))
        self.topn_var = tk.IntVar(value=5)
        top_spin = tk.Spinbox(controls, from_=1, to=10, textvariable=self.topn_var, width=6, relief="flat", font=("Segoe UI", 10))
        top_spin.grid(row=3, column=0, sticky="w", pady=(6, 0), ipady=4)

        ttk.Label(controls, text="Borrower row", style="Field.TLabel").grid(row=2, column=1, sticky="w", pady=(16, 0))
        self.row_var = tk.StringVar()
        self.row_combo = ttk.Combobox(controls, textvariable=self.row_var, state="disabled", width=38, style="App.TCombobox")
        self.row_combo.grid(row=3, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(controls, text="Analyze Borrower", command=self.analyze_selected, style="Accent.TButton").grid(row=3, column=4, sticky="e", pady=(6, 0))

        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.status_var = tk.StringVar(value="Load a CSV and train the model.")
        tk.Label(outer, textvariable=self.status_var, bg=COLORS["bg"], fg=COLORS["accent"], anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x", pady=(12, 8))

        metrics = ttk.Frame(outer, style="App.TFrame")
        metrics.pack(fill="x", pady=(0, 12))
        self.prob_var = tk.StringVar(value="-")
        self.risk_var = tk.StringVar(value="-")
        self.actual_var = tk.StringVar(value="-")
        self._build_metric_card(metrics, "Predicted delinquency probability", self.prob_var).pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._build_metric_card(metrics, "Risk band", self.risk_var).pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._build_metric_card(metrics, "Observed status", self.actual_var).pack(side="left", fill="x", expand=True)

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=3)
        body.rowconfigure(1, weight=2)

        left_top = ttk.Frame(body, style="App.TFrame")
        left_top.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        left_top.rowconfigure(0, weight=3)
        left_top.rowconfigure(1, weight=2)
        left_top.columnconfigure(0, weight=1)

        statement_frame, self.statement_text = self._make_text_panel(left_top, "Analyst Statement", 14, wrap="word", font=("Segoe UI", 10))
        statement_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        contrib_frame, self.contrib_text = self._make_text_panel(left_top, "Top Contributions", 10, wrap="none", font=("Consolas", 10))
        contrib_frame.grid(row=1, column=0, sticky="nsew")

        record_frame, self.record_text = self._make_text_panel(body, "Borrower Record", 30, wrap="none", font=("Consolas", 10))
        record_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def prepare_dataset(self, csv_path):
        df = pd.read_csv(csv_path).copy()
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA
        if "Status" not in df.columns:
            raise ValueError("CSV must include a 'Status' column.")
        df["Status"] = pd.to_numeric(df["Status"], errors="coerce").astype("Int64")
        numeric_features = [c for c in DEFAULT_NUMERIC_COLUMNS if c in df.columns]
        categorical_features = [c for c in DEFAULT_CATEGORICAL_COLUMNS if c in df.columns]
        if not numeric_features and not categorical_features:
            raise ValueError("No expected model features were found in this CSV.")
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Status"]).copy()
        return df, numeric_features, categorical_features

    def train_model(self):
        try:
            csv_path = self.path_var.get().strip()
            self.status_var.set("Loading data and training model...")
            self.root.update_idletasks()
            df, numeric_features, categorical_features = self.prepare_dataset(csv_path)
            X = df[numeric_features + categorical_features].copy()
            y = df["Status"].astype(int).values
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train_num = X_train[numeric_features].apply(pd.to_numeric, errors="coerce").astype(float)
            X_train_cat = X_train[categorical_features].astype("object").where(lambda x: x.notna(), "__MISSING__")
            X_train_skl = pd.concat([X_train_num, X_train_cat], axis=1)
            num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=8, degree=3, include_bias=False)), ("scale", StandardScaler())])
            cat_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))])
            preprocess = ColumnTransformer(transformers=[("num", num_pipe, numeric_features), ("cat", cat_pipe, categorical_features)], remainder="drop", verbose_feature_names_out=False)
            X_train_proc = preprocess.fit_transform(X_train_skl)
            model = LogisticRegression(max_iter=2000)
            model.fit(X_train_proc, y_train)
            self.df = df
            self.numeric_features = numeric_features
            self.categorical_features = categorical_features
            self.preprocess = preprocess
            self.model = model
            self.feature_names = preprocess.get_feature_names_out()
            preview_rows = list(df.index[:1000])
            self.row_combo["values"] = [f"Row {idx}" for idx in preview_rows]
            if preview_rows:
                self.row_combo.current(0)
            self.row_combo.configure(state="readonly")
            self.status_var.set(f"Model ready. Loaded {len(df):,} rows with {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")
            messagebox.showinfo("Success", "Model trained successfully.")
        except Exception as exc:
            messagebox.showerror("Training error", str(exc))
            self.status_var.set("Training failed.")

    def transform_one_row(self, raw_row):
        row_df = pd.DataFrame([raw_row])
        row_num = row_df[self.numeric_features].apply(pd.to_numeric, errors="coerce").astype(float)
        row_cat = row_df[self.categorical_features].astype("object").where(lambda x: x.notna(), "__MISSING__")
        row_skl = pd.concat([row_num, row_cat], axis=1)
        return self.preprocess.transform(row_skl)

    def compute_contributions(self, x_row):
        beta = self.model.coef_[0]
        contrib = x_row.flatten() * beta
        contrib_df = pd.DataFrame({"feature": self.feature_names, "contribution": contrib})
        candidates = self.numeric_features + self.categorical_features
        contrib_df["variable"] = contrib_df["feature"].apply(lambda name: infer_base_variable(str(name), candidates))
        grouped = contrib_df.groupby("variable", as_index=False)["contribution"].sum()
        grouped["abs_contribution"] = grouped["contribution"].abs()
        return grouped.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    def build_detailed_explanations(self, raw_row, grouped, top_n):
        up = grouped[grouped["contribution"] > 0].head(top_n)
        down = grouped[grouped["contribution"] < 0].head(top_n)
        up_text = []
        down_text = []
        for _, row in up.iterrows():
            var = row["variable"]
            raw_val = raw_row[var] if var in raw_row.index else pd.NA
            up_text.append(explain_numeric_variable(var, raw_val, row["contribution"]) if var in self.numeric_features else explain_categorical_variable(var, raw_val, row["contribution"]))
        for _, row in down.iterrows():
            var = row["variable"]
            raw_val = raw_row[var] if var in raw_row.index else pd.NA
            down_text.append(explain_numeric_variable(var, raw_val, row["contribution"]) if var in self.numeric_features else explain_categorical_variable(var, raw_val, row["contribution"]))
        return up_text, down_text

    def build_statement(self, raw_row, prob, grouped, top_n):
        up = grouped[grouped["contribution"] > 0].head(top_n)
        down = grouped[grouped["contribution"] < 0].head(top_n)
        statement = f"The model predicts an estimated delinquency probability of {prob * 100:.1f}% for this borrower, which corresponds to a {risk_label(prob)} risk classification. "
        if not up.empty:
            statement += "The strongest factors increasing risk are " + ", ".join(pretty_var_name(v) for v in up["variable"].tolist()) + ". "
        if not down.empty:
            statement += "The strongest offsetting factors are " + ", ".join(pretty_var_name(v) for v in down["variable"].tolist()) + ". "
        up_text, down_text = self.build_detailed_explanations(raw_row, grouped, top_n)
        if up_text:
            statement += "Risk-increasing interpretation: " + " ".join(up_text) + " "
        if down_text:
            statement += "Risk-reducing interpretation: " + " ".join(down_text)
        return statement

    def analyze_selected(self):
        if self.df is None or self.model is None:
            messagebox.showwarning("Not ready", "Train the model first.")
            return
        selected = self.row_var.get().strip()
        if not selected:
            messagebox.showwarning("No borrower", "Choose a borrower row first.")
            return
        row_id = int(selected.replace("Row", "").strip())
        raw_row = self.df.loc[row_id, self.numeric_features + self.categorical_features]
        x_row = self.transform_one_row(raw_row)
        prob = self.model.predict_proba(x_row)[0, 1]
        grouped = self.compute_contributions(x_row)
        statement = self.build_statement(raw_row, prob, grouped, self.topn_var.get())
        self.prob_var.set(f"{prob * 100:.2f}%")
        self.risk_var.set(risk_label(prob).title())
        self.actual_var.set(str(self.df.loc[row_id, 'Status']))
        self.statement_text.delete("1.0", tk.END)
        self.statement_text.insert(tk.END, statement)
        self.contrib_text.delete("1.0", tk.END)
        self.contrib_text.insert(tk.END, grouped.head(10).to_string(index=False))
        self.record_text.delete("1.0", tk.END)
        self.record_text.insert(tk.END, raw_row.to_frame(name="value").to_string())


def main():
    root = tk.Tk()
    LoanAnalystApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
