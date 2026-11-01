# -*- coding: utf-8 -*-
"""
Student Service Request Monitoring - Streamlit version of Untitled2.ipynb

Excel/CSV upload -> data quality checks -> feature engineering ->
SLA breach analysis -> Decision Tree model -> UML use case diagram.

Run with:  streamlit run app.py
"""
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Student Service Request Monitoring",
                   page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
UML_PATH = BASE_DIR / "assets" / "uml_use_case.png"
DEFAULT_SHEET = "Service Requests"

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score",
]
NUMERIC_COLUMNS = ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]
CATEGORY_CHECK_COLUMNS = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Required_Documents_Complete", "Status", "Reopened",
]
SUBMISSION_PREDICTORS = ["Service_Type", "Requester_Type", "Requested_Channel",
                         "Priority", "Queue_Length"]
FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday", "Queue_Category",
]
TARGET = "SLA_Breached"


def is_text(series: pd.Series) -> bool:
    """True for object or string columns (works on pandas 2.x and 3.x)."""
    return (pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series))


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_sheets(file_bytes: bytes):
    try:
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names, None
    except Exception as exc:
        return None, f"The Excel file could not be opened: {exc}"


@st.cache_data(show_spinner=False)
def load_table(file_bytes: bytes, filename: str, sheet: str):
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    except Exception as exc:
        return None, f"The file could not be read: {exc}"
    if df.empty:
        return None, "The file has no data rows."
    df.columns = df.columns.astype(str).str.strip()
    return df, None


# ---------------------------------------------------------------------------
# Data quality, cleaning and feature engineering (same steps as the notebook)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def process_data(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # --- Initial checks (notebook cell 4) ---
    df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    profile = {
        "missing": df.isnull().sum().to_frame("Missing_Count"),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["Request_ID"].duplicated().sum()),
        "invalid_dates": int(df["Submission_Date"].isna().sum()),
    }
    negatives = {}
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negatives[col] = int((df[col] < 0).sum())
    profile["negatives"] = (pd.Series(negatives, name="Negative_Values")
                            .rename_axis("Column").reset_index())
    profile["categories"] = {
        col: df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="Count")
        for col in CATEGORY_CHECK_COLUMNS
    }

    # --- Cleaning with a quality log (notebook cell 5) ---
    quality_log = []

    def log_issue(issue, before, after, treatment):
        quality_log.append({"Issue": issue, "Count_Before": before,
                            "Count_After": after, "Treatment": treatment})

    for col in df.columns:
        if is_text(df[col]):
            df[col] = df[col].str.strip()

    before = len(df)
    df = df.drop_duplicates().copy()
    log_issue("Exact duplicate rows", before, len(df), "Removed exact duplicate rows")

    before = len(df)
    df = df.dropna(subset=["Request_ID"]).copy()
    df = df.drop_duplicates(subset=["Request_ID"], keep="first")
    log_issue("Missing or duplicate Request_ID", before, len(df),
              "Missing IDs removed; first occurrence retained")

    for col in NUMERIC_COLUMNS:
        negative = int((df[col] < 0).sum())
        log_issue(f"Negative values in {col}", negative, negative,
                  "Flag for investigation; replace invalid negatives with missing")
        df.loc[df[col] < 0, col] = np.nan

    invalid_dates = int(df["Submission_Date"].isna().sum())
    log_issue("Invalid or missing Submission_Date", invalid_dates, invalid_dates,
              "Retained as missing and excluded from date-dependent analysis")
    quality_df = pd.DataFrame(quality_log)

    # --- Feature engineering (notebook cell 6) ---
    df["Submission_Weekday"] = df["Submission_Date"].dt.day_name()
    df["Queue_Category"] = pd.cut(
        df["Queue_Length"], bins=[-np.inf, 5, 15, np.inf],
        labels=["Low", "Medium", "High"])
    df["Status"] = df["Status"].str.strip().str.title()

    eligible = (
        df["Status"].eq("Completed")
        & df["Processing_Days"].notna()
        & df["SLA_Days"].notna()
        & (df["Processing_Days"] >= 0)
        & (df["SLA_Days"] >= 0)
    )
    df[TARGET] = np.nan
    df.loc[eligible, TARGET] = (
        df.loc[eligible, "Processing_Days"] > df.loc[eligible, "SLA_Days"]
    ).astype(int)

    return df, quality_df, profile, int(eligible.sum())


# ---------------------------------------------------------------------------
# Model (notebook cells 11-13)
# ---------------------------------------------------------------------------
def calculate_metrics(name, actual, predicted):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1_Score": f1_score(actual, predicted, zero_division=0),
    }


@st.cache_data(show_spinner="Training Decision Tree...")
def train_model(model_data: pd.DataFrame, max_depth: int, min_leaf: int,
                test_size: float, seed: int):
    X = model_data[FEATURES]
    y = model_data[TARGET].astype(int)

    counts = y.value_counts()
    if y.nunique() < 2 or counts.min() < 2:
        return {"error": "Insufficient examples in one target class for a stratified "
                         "split. Inspect the dataset and document this limitation.",
                "counts": counts}

    categorical = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in FEATURES if c not in categorical]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError as exc:
        return {"error": f"Could not split the data: {exc}", "counts": counts}

    preprocessor = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=seed, max_depth=max_depth, min_samples_leaf=min_leaf)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    evaluation = pd.DataFrame([
        calculate_metrics("Decision Tree", y_test, predictions),
        calculate_metrics("Majority-Class Baseline", y_test, baseline.predict(X_test)),
    ])

    report = pd.DataFrame(classification_report(
        y_test, predictions, zero_division=0, output_dict=True)).T

    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({
        "Feature": names,
        "Importance": model.named_steps["classifier"].feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "error": None,
        "counts": counts,
        "n_records": len(model_data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "evaluation": evaluation,
        "cm": confusion_matrix(y_test, predictions),
        "labels": sorted(y.unique().tolist()),
        "report": report,
        "importance": importance,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_service_volume(service_counts: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))
    service_counts.plot(kind="bar", ax=ax, title="Student Service Requests by Service Type")
    ax.set_xlabel("Service Type")
    ax.set_ylabel("Number of Requests")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def chart_queue_breach(queue_rates: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 5))
    queue_rates.plot(kind="bar", ax=ax, title="SLA Breach Rate by Queue Category")
    ax.set_xlabel("Queue Category")
    ax.set_ylabel("SLA Breach Rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def chart_importance(importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    importance.head(10).sort_values("Importance").plot(
        x="Feature", y="Importance", kind="barh", legend=False, ax=ax,
        title="Decision Tree Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("📊 Student Service Request Monitoring")
st.caption("Data quality, SLA breach analysis and Decision Tree prediction.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Excel or CSV file", type=["xlsx", "xls", "csv"])

    sheet = DEFAULT_SHEET
    file_bytes = None
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        if not uploaded.name.lower().endswith(".csv"):
            sheets, sheet_error = list_sheets(file_bytes)
            if sheet_error:
                st.error(sheet_error)
                st.stop()
            default_index = sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in sheets else 0
            sheet = st.selectbox("Sheet", sheets, index=default_index)

    st.header("2. Model settings")
    max_depth = st.slider("Max tree depth", 1, 15, 5)
    min_leaf = st.slider("Min samples per leaf", 1, 50, 5)
    test_size = st.slider("Test set size", 0.10, 0.50, 0.20, step=0.05)
    seed = int(st.number_input("Random seed", value=42, step=1))

if uploaded is None:
    st.info("👈 Upload `student_service_requests.xlsx` (sheet **Service Requests**) "
            "in the sidebar to begin.")
    tab_uml_only = st.expander("UML use case diagram", expanded=True)
    with tab_uml_only:
        if UML_PATH.exists():
            st.image(str(UML_PATH))
    st.stop()

raw_df, load_error = load_table(file_bytes, uploaded.name, sheet)
if load_error:
    st.error(load_error)
    st.stop()

missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
if missing_cols:
    st.error("The dataset is missing required columns: " + ", ".join(missing_cols))
    st.write("Columns found:", ", ".join(raw_df.columns))
    st.stop()

df, quality_df, profile, eligible_count = process_data(raw_df)
labeled = df[df[TARGET].notna()].copy()

tabs = st.tabs([
    "1. Dataset", "2. Data Quality", "3. Features",
    "4. Analysis", "5. Model", "6. Use Case Diagram",
])

# ------------------------------- 1. Dataset --------------------------------
with tabs[0]:
    st.success("Dataset loaded successfully!")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", raw_df.shape[1])

    st.subheader("First 10 records")
    st.dataframe(raw_df.head(10))

    left, right = st.columns(2)
    with left:
        st.subheader("Column names and data types")
        st.dataframe(raw_df.dtypes.astype(str).to_frame("dtype"))
    with right:
        st.subheader("Variable roles")
        st.write("**Five submission-time predictors:** " + ", ".join(SUBMISSION_PREDICTORS))
        st.write("**Target variable:** SLA_Breached")
        st.write("**Identifier:** Request_ID")
        st.write("**Categorical variables:** Service_Type, Assigned_Office")
        st.write("**Numerical variables:** Queue_Length, SLA_Days")

    st.subheader("Category frequencies")
    text_cols = [c for c in raw_df.columns if is_text(raw_df[c])]
    if text_cols:
        chosen = st.selectbox("Choose a column", text_cols)
        st.dataframe(raw_df[chosen].value_counts(dropna=False).to_frame("Count"))

# ----------------------------- 2. Data Quality -----------------------------
with tabs[1]:
    st.subheader("Initial checks")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate rows", profile["duplicate_rows"])
    c2.metric("Duplicate Request_ID values", profile["duplicate_ids"])
    c3.metric("Invalid or missing dates", profile["invalid_dates"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values**")
        st.dataframe(profile["missing"])
    with right:
        st.markdown("**Negative numeric values**")
        st.dataframe(profile["negatives"], hide_index=True)

    with st.expander("Category values (before cleaning)"):
        for col, table in profile["categories"].items():
            st.markdown(f"**{col}**")
            st.dataframe(table, hide_index=True)

    st.subheader("Data quality log")
    st.dataframe(quality_df, hide_index=True)
    st.download_button("⬇ Download data_quality_log.csv",
                       quality_df.to_csv(index=False).encode("utf-8"),
                       "data_quality_log.csv", "text/csv")

# ------------------------------ 3. Features --------------------------------
with tabs[2]:
    left, right = st.columns(2)
    with left:
        st.subheader("Submission weekday")
        st.dataframe(df["Submission_Weekday"].value_counts(dropna=False).to_frame("Count"))
        st.subheader("Queue category")
        st.dataframe(df["Queue_Category"].value_counts(dropna=False).to_frame("Count"))
    with right:
        st.subheader("SLA breach labels")
        st.dataframe(df[TARGET].value_counts(dropna=False).to_frame("Count"))
        st.metric("Eligible completed records", eligible_count)

    st.subheader("Cleaned data with engineered features")
    st.dataframe(df.head(10))
    st.download_button("⬇ Download student_service_requests_cleaned.csv",
                       df.to_csv(index=False).encode("utf-8"),
                       "student_service_requests_cleaned.csv", "text/csv")

# ------------------------------- 4. Analysis -------------------------------
with tabs[3]:
    st.subheader("Requests per service type")
    service_counts = df["Service_Type"].value_counts()
    left, right = st.columns([1, 2])
    left.dataframe(service_counts.rename("Request_Count").to_frame())
    fig1 = chart_service_volume(service_counts)
    right.pyplot(fig1)
    st.info(f"**Most requested service:** {service_counts.idxmax()}")
    st.download_button("⬇ Download chart 1", fig_to_png(fig1),
                       "chart1_service_volume.png", "image/png")
    plt.close(fig1)

    if labeled.empty:
        st.warning("No eligible completed requests, so breach analysis is unavailable.")
    else:
        st.subheader("SLA breach rate by assigned office")
        office_breach = (
            labeled.groupby("Assigned_Office", observed=True)[TARGET]
            .agg(["count", "mean"])
            .rename(columns={"count": "Completed_Requests", "mean": "Breach_Rate"})
        )
        office_breach["Breach_Rate_Percent"] = office_breach["Breach_Rate"] * 100
        st.dataframe(office_breach.sort_values("Breach_Rate_Percent", ascending=False))
        st.info("**Office with highest observed breach rate:** "
                f"{office_breach['Breach_Rate_Percent'].idxmax()}")

        st.subheader("SLA breach rate by queue category")
        queue_analysis = (
            labeled.groupby("Queue_Category", observed=True)[TARGET].agg(["count", "mean"])
        )
        queue_analysis["Breach_Rate_Percent"] = queue_analysis["mean"] * 100
        left, right = st.columns([1, 2])
        left.dataframe(queue_analysis)
        queue_rates = (labeled.groupby("Queue_Category", observed=True)[TARGET].mean() * 100)
        fig2 = chart_queue_breach(queue_rates)
        right.pyplot(fig2)
        st.download_button("⬇ Download chart 2", fig_to_png(fig2),
                           "chart2_queue_breach.png", "image/png")
        plt.close(fig2)

# -------------------------------- 5. Model ---------------------------------
with tabs[4]:
    model_data = df.loc[df[TARGET].notna()].dropna(subset=FEATURES + [TARGET]).copy()
    st.write(f"**Model records:** {len(model_data)}")

    if model_data.empty:
        st.warning("There are no complete eligible records to train a model.")
    else:
        result = train_model(model_data, max_depth, min_leaf, test_size, seed)
        st.markdown("**Target class counts**")
        st.dataframe(result["counts"].rename("Count").to_frame())

        if result["error"]:
            st.error(result["error"])
        else:
            st.success("Decision Tree training completed!")
            st.caption(f"Training records: {result['n_train']} | "
                       f"Testing records: {result['n_test']}")

            st.subheader("Evaluation")
            st.dataframe(result["evaluation"].round(3), hide_index=True)

            left, right = st.columns(2)
            with left:
                st.markdown("**Decision Tree confusion matrix**")
                fig_cm, ax = plt.subplots(figsize=(4.5, 4))
                ConfusionMatrixDisplay(result["cm"], display_labels=result["labels"]).plot(
                    cmap="Blues", values_format="d", ax=ax)
                fig_cm.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            with right:
                st.markdown("**Classification report**")
                st.dataframe(result["report"].round(3))

            st.subheader("Top 10 influential features")
            left, right = st.columns([1, 2])
            left.dataframe(result["importance"].head(10), hide_index=True)
            fig3 = chart_importance(result["importance"])
            right.pyplot(fig3)
            st.download_button("⬇ Download feature_importance.png", fig_to_png(fig3),
                               "feature_importance.png", "image/png")
            plt.close(fig3)

# ------------------------------ 6. Use Case --------------------------------
with tabs[5]:
    st.subheader("UML Use Case Diagram")
    if UML_PATH.exists():
        st.image(str(UML_PATH), caption="Student Service Request Monitoring System")
    else:
        st.error("assets/uml_use_case.png was not found.")
    st.markdown(
        "- **Administrator:** View Service Volume, View SLA Performance, Review Delay Risk\n"
        "- **Service Office Staff:** View SLA Performance, Review Delay Risk, "
        "Manage Service Requests\n"
        "- **Student:** Submit Service Request, View Request Status"
    )
