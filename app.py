import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from imblearn.over_sampling import SMOTE


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


#  ============================================================
# CUSTOM CSS - FIXED VISIBILITY
# ============================================================

st.markdown("""
<style>

/* ============================================================
   MAIN APP BACKGROUND
   ============================================================ */

.stApp {
    background-color: #f5f7fb !important;
}

/* Main content area */
[data-testid="stAppViewContainer"] .main {
    background-color: #f5f7fb !important;
}

/* Main block container */
[data-testid="stAppViewContainer"] .main .block-container {
    background-color: #f5f7fb !important;
}


/* ============================================================
   MAIN CONTENT TEXT
   IMPORTANT: Do NOT apply this to sidebar
   ============================================================ */

/* Normal markdown text */
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] p {
    color: #111827 !important;
}

/* Markdown headings */
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] h1,
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] h2,
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] h3,
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] h4,
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] h5,
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] h6 {
    color: #111827 !important;
}

/* General main content text */
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main li,
[data-testid="stAppViewContainer"] .main label {
    color: #111827 !important;
}

/* Main headings */
[data-testid="stAppViewContainer"] .main h1,
[data-testid="stAppViewContainer"] .main h2,
[data-testid="stAppViewContainer"] .main h3,
[data-testid="stAppViewContainer"] .main h4 {
    color: #111827 !important;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08) !important;
}

/* Metric label */
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
[data-testid="stMetric"] label,
[data-testid="stMetric"] label * {
    color: #4b5563 !important;
    font-weight: 600 !important;
}

/* Metric value */
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] *,
[data-testid="stMetric"] [data-testid="stMetricValue"] div,
[data-testid="stMetric"] [data-testid="stMetricValue"] span {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* Metric delta */
[data-testid="stMetric"] [data-testid="stMetricDelta"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
    color: #374151 !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

/* Sidebar title */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #ffffff !important;
}


/* ============================================================
   SIDEBAR FILE UPLOADER
   ============================================================ */

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: #0b1220 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #0b1220 !important;
    border-color: #374151 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #374151 !important;
    color: #ffffff !important;
    border: 1px solid #4b5563 !important;
}


/* ============================================================
   SIDEBAR SELECTBOX / RADIO
   ============================================================ */

[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #ffffff !important;
}


/* ============================================================
   MAIN SELECTBOX
   ============================================================ */

[data-testid="stAppViewContainer"] .main
[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-testid="stAppViewContainer"] .main
[data-baseweb="select"] * {
    color: #111827 !important;
}


/* ============================================================
   TEXT INPUT / NUMBER INPUT
   ============================================================ */

[data-testid="stAppViewContainer"] .main input,
[data-testid="stAppViewContainer"] .main textarea {
    color: #111827 !important;
    background-color: #ffffff !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

[data-testid="stAppViewContainer"] .main button {
    color: #ffffff !important;
    background-color: #2563eb !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}

[data-testid="stAppViewContainer"] .main button:hover {
    background-color: #1d4ed8 !important;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
}


/* ============================================================
   ALERT / INFO / WARNING / ERROR BOXES
   ============================================================ */

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #111827 !important;
}


/* ============================================================
   EXPANDERS
   ============================================================ */

[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] label {
    color: #111827 !important;
}


/* ============================================================
   CAPTION
   ============================================================ */

[data-testid="stAppViewContainer"] .main
[data-testid="stCaptionContainer"] {
    color: #4b5563 !important;
}

[data-testid="stAppViewContainer"] .main
[data-testid="stCaptionContainer"] * {
    color: #4b5563 !important;
}


/* ============================================================
   DIVIDER
   ============================================================ */

[data-testid="stAppViewContainer"] .main hr {
    border-color: #d1d5db !important;
}


/* ============================================================
   LINKS
   ============================================================ */

[data-testid="stAppViewContainer"] .main a {
    color: #2563eb !important;
}


/* ============================================================
   SLIDER
   ============================================================ */

[data-testid="stAppViewContainer"] .main
[data-testid="stSlider"] label {
    color: #111827 !important;
}


/* ============================================================
   CHECKBOX
   ============================================================ */

[data-testid="stAppViewContainer"] .main
[data-testid="stCheckbox"] label {
    color: #111827 !important;
}


/* ============================================================
   PLOTLY CONTAINER
   ============================================================ */

[data-testid="stPlotlyChart"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px !important;
}


/* ============================================================
   DOWNLOAD BUTTON
   ============================================================ */

[data-testid="stDownloadButton"] button {
    background-color: #2563eb !important;
    color: #ffffff !important;
}


/* ============================================================
   REMOVE ANY WHITE TEXT FROM MAIN AREA
   ============================================================ */

/* Markdown bold text */
[data-testid="stAppViewContainer"] .main strong {
    color: #111827 !important;
}

/* Markdown italic text */
[data-testid="stAppViewContainer"] .main em {
    color: #374151 !important;
}

/* Code / inline code */
[data-testid="stAppViewContainer"] .main code {
    color: #111827 !important;
    background-color: #e5e7eb !important;
}


/* ============================================================
   SCROLLBAR
   ============================================================ */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
    background: #94a3b8;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748b;
}

</style>
""", unsafe_allow_html=True)
முக்கியமானது
உன் code-ல இப்போ இருக்கும்:

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb !important;
}

/* METRIC CARD */

[data-testid="stMetric"] {
    ...
}

/* ALL METRIC TEXT */

[data-testid="stMetric"] * {
    color: #111827 !important;
}

...
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="Financial Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
...
""", unsafe_allow_html=True)
# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "df": None,
    "target": None,
    "model": None,
    "preprocessor": None,
    "anomaly_model": None,
    "metrics": None,
    "feature_importance": None,
    "feature_names": None,
    "X_reference": None,
    "numeric_columns": None,
    "categorical_columns": None,
    "training_columns": None,
    "training_means": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# FUNCTIONS
# ============================================================

def load_file(uploaded_file):

    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if uploaded_file.name.lower().endswith(
        (".xlsx", ".xls")
    ):
        return pd.read_excel(uploaded_file)

    return None


# ============================================================
# UPDATED CLEAN DATAFRAME FUNCTION
# ============================================================

def clean_dataframe(df):

    df = df.copy()

    new_columns = []

    for column in df.columns:
        new_column = str(column).strip()
        new_column = new_column.replace(" ", "_")
        new_column = new_column.replace("-", "_")
        new_column = new_column.replace("/", "_")
        new_columns.append(new_column)

    df.columns = new_columns

    df = df.drop_duplicates()

    return df


def detect_target(columns):

    candidates = [
        "class",
        "fraud",
        "is_fraud",
        "fraud_flag",
        "fraudulent",
        "isfraud",
        "fraud_status",
        "target",
        "label",
        "risk"
    ]

    normalized = {
        str(c).lower().replace(" ", "_"): c
        for c in columns
    }

    for candidate in candidates:

        if candidate in normalized:
            return normalized[candidate]

    for column in columns:

        name = str(column).lower()

        if "fraud" in name:
            return column

    return None


def encode_target(series):

    if pd.api.types.is_numeric_dtype(series):

        values = sorted(
            series.dropna().unique().tolist()
        )

        if len(values) == 2:

            mapping = {
                values[0]: 0,
                values[1]: 1
            }

            return series.map(mapping), mapping

    text = (
        series
        .astype(str)
        .str.lower()
        .str.strip()
    )

    unique_values = text.unique()

    if len(unique_values) == 2:

        fraud_words = [
            "fraud",
            "fraudulent",
            "yes",
            "true",
            "positive",
            "suspicious",
            "1"
        ]

        mapping = {}

        for value in unique_values:

            if value in fraud_words:
                mapping[value] = 1
            else:
                mapping[value] = 0

        return text.map(mapping), mapping

    return None, None


def create_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers = []

    if numeric_columns:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return preprocessor


def get_feature_names(preprocessor):

    try:
        return preprocessor.get_feature_names_out()

    except Exception:

        return []


def calculate_risk(probability, anomaly_score):

    ml_score = probability * 100
    anomaly_part = anomaly_score * 100

    risk = (
        ml_score * 0.70
        +
        anomaly_part * 0.30
    )

    return min(
        100,
        max(0, risk)
    )


def risk_category(score):

    if score >= 85:
        return "CRITICAL"

    if score >= 70:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


def risk_symbol(category):

    symbols = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }

    return symbols.get(
        category,
        "⚪"
    )


def find_amount_column(columns):

    keywords = [
        "amount",
        "transaction_amount",
        "amt",
        "value",
        "price"
    ]

    for column in columns:

        name = str(column).lower()

        for keyword in keywords:

            if keyword in name:
                return column

    return None


def find_time_column(columns):

    keywords = [
        "time",
        "timestamp",
        "date",
        "datetime"
    ]

    for column in columns:

        name = str(column).lower()

        for keyword in keywords:

            if keyword in name:
                return column

    return None


def generate_reasons(
    row,
    probability,
    anomaly_score,
    amount_column=None,
    reference_amount=None
):

    reasons = []

    if probability >= 0.80:
        reasons.append(
            "Very high ML fraud probability"
        )

    elif probability >= 0.60:
        reasons.append(
            "High ML fraud probability"
        )

    if anomaly_score >= 0.75:
        reasons.append(
            "Transaction behavior is highly anomalous"
        )

    elif anomaly_score >= 0.50:
        reasons.append(
            "Transaction behavior differs from normal patterns"
        )

    if (
        amount_column is not None
        and reference_amount is not None
    ):

        try:

            amount = float(
                row[amount_column]
            )

            if amount > reference_amount * 3:

                reasons.append(
                    "Transaction amount is significantly higher than normal"
                )

        except Exception:
            pass

    if not reasons:

        reasons.append(
            "Transaction shows relatively normal behavior"
        )

    return reasons


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ FRAUD INTELLIGENCE")

st.sidebar.caption(
    "AI-Powered Financial Risk Analytics"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Financial Fraud Dataset",
    type=[
        "csv",
        "xlsx",
        "xls"
    ]
)


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None:

    try:

        df = load_file(
            uploaded_file
        )

        df = clean_dataframe(
            df
        )

        st.session_state.df = df

        detected_target = detect_target(
            df.columns
        )

        if detected_target is not None:

            st.session_state.target = (
                detected_target
            )

    except Exception as e:

        st.error(
            f"Dataset loading error: {e}"
        )


if st.session_state.df is None:

    st.title(
        "🛡️ Financial Fraud Intelligence Platform"
    )

    st.markdown(
        """
        ### AI-Powered Fraud Detection & Risk Analytics

        This platform combines:

        **Machine Learning + Anomaly Detection + Risk Scoring
        + Explainable Analytics + Investigation Intelligence**
        """
    )

    st.info(
        "Upload your Zidio Financial Fraud dataset from the sidebar."
    )

    st.markdown(
        """
        ### Advanced capabilities

        🔹 Fraud probability prediction

        🔹 Unsupervised anomaly detection

        🔹 Combined AI risk score

        🔹 Fraud pattern analytics

        🔹 Investigation priority queue

        🔹 Transaction explanation

        🔹 Model performance

        🔹 What-if risk analysis

        🔹 Batch fraud screening

        🔹 Data drift monitoring
        """
    )

    st.stop()


df = st.session_state.df


# ============================================================
# TARGET SELECTION
# ============================================================

target_options = df.columns.tolist()

default_index = 0

if st.session_state.target in target_options:

    default_index = target_options.index(
        st.session_state.target
    )


target = st.sidebar.selectbox(
    "Fraud / Target Column",
    target_options,
    index=default_index
)

st.session_state.target = target


# ============================================================
# TARGET ENCODING
# ============================================================

target_encoded, target_mapping = encode_target(
    df[target]
)

if target_encoded is None:

    st.error(
        "Selected target column must contain exactly two classes."
    )

    st.stop()


analysis_df = df.copy()

analysis_df["Fraud_Label"] = (
    target_encoded
)

analysis_df = analysis_df.dropna(
    subset=["Fraud_Label"]
)

analysis_df["Fraud_Label"] = (
    analysis_df["Fraud_Label"]
    .astype(int)
)


# ============================================================
# GLOBAL KPIs
# ============================================================

total_transactions = len(
    analysis_df
)

fraud_count = int(
    (
        analysis_df["Fraud_Label"] == 1
    ).sum()
)

normal_count = int(
    (
        analysis_df["Fraud_Label"] == 0
    ).sum()
)

fraud_rate = (
    fraud_count
    /
    total_transactions
    *
    100
)


amount_column = find_amount_column(
    df.columns
)

time_column = find_time_column(
    df.columns
)


# ============================================================
# NAVIGATION
# ============================================================

pages = [
    "🏠 Executive Intelligence",
    "📊 Transaction Analytics",
    "🚨 Fraud Pattern Intelligence",
    "🤖 AI Risk Engine",
    "🔍 Investigation Center",
    "🧪 What-If Analysis",
    "📈 Model Performance",
    "📡 Model Monitoring",
    "📋 Dataset Explorer"
]


page = st.sidebar.radio(
    "Navigation",
    pages
)


st.sidebar.divider()

st.sidebar.metric(
    "Transactions",
    f"{total_transactions:,}"
)

st.sidebar.metric(
    "Fraud Cases",
    f"{fraud_count:,}"
)

st.sidebar.metric(
    "Fraud Rate",
    f"{fraud_rate:.3f}%"
)


# ============================================================
# EXECUTIVE INTELLIGENCE
# ============================================================

if page == "🏠 Executive Intelligence":

    st.title(
        "🛡️ Financial Fraud Intelligence"
    )

    st.caption(
        "AI-powered transaction risk analytics platform"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

    c3.metric(
        "Normal Transactions",
        f"{normal_count:,}"
    )

    c4.metric(
        "Fraud Rate",
        f"{fraud_rate:.3f}%"
    )

    st.divider()

    if amount_column is not None:

        total_value = pd.to_numeric(
            analysis_df[amount_column],
            errors="coerce"
        ).sum()

        fraud_value = pd.to_numeric(
            analysis_df.loc[
                analysis_df["Fraud_Label"] == 1,
                amount_column
            ],
            errors="coerce"
        ).sum()

        c1, c2 = st.columns(2)

        c1.metric(
            "Total Transaction Value",
            f"{total_value:,.2f}"
        )

        c2.metric(
            "Fraud Transaction Value",
            f"{fraud_value:,.2f}"
        )

    st.subheader(
        "Transaction Distribution"
    )

    distribution = pd.DataFrame(
        {
            "Type": [
                "Normal",
                "Fraud"
            ],
            "Count": [
                normal_count,
                fraud_count
            ]
        }
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            distribution,
            x="Type",
            y="Count",
            text="Count",
            title="Normal vs Fraud"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            distribution,
            names="Type",
            values="Count",
            hole=0.5,
            title="Transaction Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Business Intelligence"
    )

    if fraud_rate < 1:

        st.warning(
            f"Fraud represents only {fraud_rate:.3f}% "
            "of transactions. This indicates a highly "
            "imbalanced fraud detection problem."
        )

    else:

        st.info(
            f"Fraud represents {fraud_rate:.2f}% "
            "of transactions."
        )


# ============================================================
# TRANSACTION ANALYTICS
# ============================================================

elif page == "📊 Transaction Analytics":

    st.title(
        "📊 Transaction Intelligence"
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns = [
        c for c in numeric_columns
        if c != target
    ]

    if numeric_columns:

        selected_column = st.selectbox(
            "Select transaction feature",
            numeric_columns
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                analysis_df,
                x=selected_column,
                color="Fraud_Label",
                nbins=50,
                title=f"{selected_column} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.box(
                analysis_df,
                x="Fraud_Label",
                y=selected_column,
                color="Fraud_Label",
                title=f"{selected_column}: Normal vs Fraud"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader(
        "Correlation Intelligence"
    )

    numeric_df = analysis_df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            aspect="auto",
            title="Feature Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if time_column is not None:

        st.subheader(
            "Transaction Time Intelligence"
        )

        temp = analysis_df.copy()

        try:

            if pd.api.types.is_numeric_dtype(
                temp[time_column]
            ):

                temp["Hour"] = (
                    temp[time_column] / 3600
                ) % 24

            else:

                temp["Parsed_Time"] = pd.to_datetime(
                    temp[time_column],
                    errors="coerce"
                )

                temp["Hour"] = (
                    temp["Parsed_Time"].dt.hour
                )

            hourly = temp.groupby(
                "Hour"
            )["Fraud_Label"].sum().reset_index()

            hourly.columns = [
                "Hour",
                "Fraud_Count"
            ]

            fig = px.line(
                hourly,
                x="Hour",
                y="Fraud_Count",
                markers=True,
                title="Fraud Activity by Hour"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception:
            pass


# ============================================================
# FRAUD PATTERN INTELLIGENCE
# ============================================================

elif page == "🚨 Fraud Pattern Intelligence":

    st.title(
        "🚨 Fraud Pattern Intelligence"
    )

    st.subheader(
        "Fraud Concentration"
    )

    fraud_features = []

    for column in df.columns:

        if column == target:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            normal_mean = analysis_df.loc[
                analysis_df["Fraud_Label"] == 0,
                column
            ].mean()

            fraud_mean = analysis_df.loc[
                analysis_df["Fraud_Label"] == 1,
                column
            ].mean()

            if pd.notna(normal_mean) and pd.notna(
                fraud_mean
            ):

                difference = abs(
                    fraud_mean - normal_mean
                )

                fraud_features.append(
                    {
                        "Feature": column,
                        "Normal Mean": normal_mean,
                        "Fraud Mean": fraud_mean,
                        "Difference": difference
                    }
                )

    if fraud_features:

        feature_df = pd.DataFrame(
            fraud_features
        ).sort_values(
            "Difference",
            ascending=False
        )

        st.dataframe(
            feature_df.head(20),
            use_container_width=True
        )

        fig = px.bar(
            feature_df.head(15).sort_values(
                "Difference"
            ),
            x="Difference",
            y="Feature",
            orientation="h",
            title="Features Showing Strongest Fraud/Normal Difference"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if amount_column is not None:

        st.subheader(
            "Fraud Amount Intelligence"
        )

        amount_series = pd.to_numeric(
            analysis_df[amount_column],
            errors="coerce"
        )

        normal_amount = amount_series[
            analysis_df["Fraud_Label"] == 0
        ]

        fraud_amount = amount_series[
            analysis_df["Fraud_Label"] == 1
        ]

        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Average",
                    "Median",
                    "Maximum"
                ],
                "Normal": [
                    normal_amount.mean(),
                    normal_amount.median(),
                    normal_amount.max()
                ],
                "Fraud": [
                    fraud_amount.mean(),
                    fraud_amount.median(),
                    fraud_amount.max()
                ]
            }
        )

        st.dataframe(
            metrics,
            use_container_width=True
        )


# ============================================================
# AI RISK ENGINE
# ============================================================

elif page == "🤖 AI Risk Engine":

    st.title(
        "🤖 AI Risk Engine"
    )

    st.write(
        """
        This engine combines supervised machine learning
        with unsupervised anomaly detection.
        """
    )

    st.markdown(
        """
        **Random Forest**
        → learns known fraud patterns

        **Isolation Forest**
        → detects unusual transaction behavior

        **Risk Engine**
        → combines both signals into a 0–100 risk score
        """
    )

    if st.button(
        "🚀 Train Advanced Fraud Engine",
        use_container_width=True
    ):

        with st.spinner(
            "Training AI fraud and anomaly models..."
        ):

            X = analysis_df.drop(
                columns=[
                    target,
                    "Fraud_Label"
                ]
            )

            y = analysis_df[
                "Fraud_Label"
            ]

            if y.nunique() != 2:

                st.error(
                    "Target must contain exactly two classes."
                )

                st.stop()

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )
            )

            preprocessor = create_preprocessor(
                X_train
            )

            X_train_processed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_test_processed = (
                preprocessor.transform(
                    X_test
                )
            )

            try:

                smote = SMOTE(
                    random_state=42
                )

                X_train_balanced, y_train_balanced = (
                    smote.fit_resample(
                        X_train_processed,
                        y_train
                    )
                )

            except Exception:

                X_train_balanced = (
                    X_train_processed
                )

                y_train_balanced = y_train

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )

            model.fit(
                X_train_balanced,
                y_train_balanced
            )

            predictions = model.predict(
                X_test_processed
            )

            probabilities = model.predict_proba(
                X_test_processed
            )[:, 1]

            anomaly_model = IsolationForest(
                n_estimators=150,
                contamination="auto",
                random_state=42,
                n_jobs=-1
            )

            anomaly_model.fit(
                X_train_processed
            )

            anomaly_raw = (
                -anomaly_model.decision_function(
                    X_test_processed
                )
            )

            anomaly_min = anomaly_raw.min()
            anomaly_max = anomaly_raw.max()

            if anomaly_max != anomaly_min:

                anomaly_scores = (
                    anomaly_raw - anomaly_min
                ) / (
                    anomaly_max - anomaly_min
                )

            else:

                anomaly_scores = np.zeros(
                    len(anomaly_raw)
                )

            risk_scores = [
                calculate_risk(
                    probability,
                    anomaly
                )
                for probability, anomaly
                in zip(
                    probabilities,
                    anomaly_scores
                )
            ]

            categories = [
                risk_category(score)
                for score in risk_scores
            ]

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            auc = roc_auc_score(
                y_test,
                probabilities
            )

            metrics = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "ROC-AUC"
                    ],
                    "Score": [
                        accuracy,
                        precision,
                        recall,
                        f1,
                        auc
                    ]
                }
            )

            st.session_state.model = {
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "predictions": predictions,
                "probabilities": probabilities,
                "anomaly_scores": anomaly_scores,
                "risk_scores": risk_scores,
                "categories": categories
            }

            st.session_state.preprocessor = (
                preprocessor
            )

            st.session_state.anomaly_model = (
                anomaly_model
            )

            st.session_state.metrics = (
                metrics
            )

            st.session_state.feature_names = (
                get_feature_names(
                    preprocessor
                )
            )

            st.session_state.numeric_columns = (
                X_train.select_dtypes(
                    include=np.number
                ).columns.tolist()
            )

            st.session_state.categorical_columns = (
                X_train.select_dtypes(
                    exclude=np.number
                ).columns.tolist()
            )

            st.session_state.training_columns = (
                X.columns.tolist()
            )

            st.session_state.training_means = (
                X_train.select_dtypes(
                    include=np.number
                ).mean()
            )

            feature_names = (
                st.session_state.feature_names
            )

            if hasattr(
                model,
                "feature_importances_"
            ):

                importance_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance":
                            model.feature_importances_
                    }
                ).sort_values(
                    "Importance",
                    ascending=False
                )

                st.session_state.feature_importance = (
                    importance_df
                )

            joblib.dump(
                model,
                "advanced_fraud_model.pkl"
            )

            joblib.dump(
                preprocessor,
                "advanced_preprocessor.pkl"
            )

            joblib.dump(
                anomaly_model,
                "anomaly_model.pkl"
            )

            st.success(
                "Advanced AI Risk Engine trained successfully!"
            )

    if st.session_state.model is not None:

        model_data = st.session_state.model

        risk_array = np.array(
            model_data["risk_scores"]
        )

        critical = int(
            (risk_array >= 85).sum()
        )

        high = int(
            (
                (risk_array >= 70)
                &
                (risk_array < 85)
            ).sum()
        )

        medium = int(
            (
                (risk_array >= 40)
                &
                (risk_array < 70)
            ).sum()
        )

        low = int(
            (risk_array < 40).sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "🔴 Critical",
            critical
        )

        c2.metric(
            "🟠 High",
            high
        )

        c3.metric(
            "🟡 Medium",
            medium
        )

        c4.metric(
            "🟢 Low",
            low
        )

        st.subheader(
            "Risk Distribution"
        )

        risk_df = pd.DataFrame(
            {
                "Risk": [
                    "Critical",
                    "High",
                    "Medium",
                    "Low"
                ],
                "Count": [
                    critical,
                    high,
                    medium,
                    low
                ]
            }
        )

        fig = px.bar(
            risk_df,
            x="Risk",
            y="Count",
            text="Count",
            title="AI Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# INVESTIGATION CENTER
# ============================================================

elif page == "🔍 Investigation Center":

    st.title(
        "🔍 Fraud Investigation Center"
    )

    if st.session_state.model is None:

        st.warning(
            "Train the AI Risk Engine first."
        )

        st.stop()

    model_data = st.session_state.model

    test_data = (
        model_data["X_test"]
        .copy()
        .reset_index(drop=True)
    )

    result = test_data.copy()

    result["Fraud Probability"] = (
        model_data["probabilities"] * 100
    ).round(2)

    result["Anomaly Score"] = (
        model_data["anomaly_scores"] * 100
    ).round(2)

    result["Risk Score"] = (
        np.array(
            model_data["risk_scores"]
        ).round(2)
    )

    result["Risk Level"] = (
        model_data["categories"]
    )

    result["Priority"] = (
        result["Risk Score"]
        .rank(
            ascending=False,
            method="first"
        )
        .astype(int)
    )

    result = result.sort_values(
        "Risk Score",
        ascending=False
    )

    st.subheader(
        "🚨 Investigation Priority Queue"
    )

    priority_filter = st.multiselect(
        "Risk Levels",
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"
        ],
        default=[
            "CRITICAL",
            "HIGH"
        ]
    )

    filtered = result[
        result["Risk Level"].isin(
            priority_filter
        )
    ]

    st.dataframe(
        filtered.head(100),
        use_container_width=True
    )

    st.download_button(
        "📥 Download Investigation Queue",
        filtered.to_csv(index=False),
        "fraud_investigation_queue.csv",
        "text/csv"
    )

    st.divider()

    if len(result) > 0:

        selected_index = st.selectbox(
            "Select transaction for investigation",
            result.index.tolist()
        )

        selected = result.loc[
            selected_index
        ]

        probability = (
            selected["Fraud Probability"]
            /
            100
        )

        anomaly = (
            selected["Anomaly Score"]
            /
            100
        )

        risk = selected[
            "Risk Score"
        ]

        category = selected[
            "Risk Level"
        ]

        st.subheader(
            "Transaction Investigation"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        c2.metric(
            "Anomaly Score",
            f"{anomaly * 100:.2f}%"
        )

        c3.metric(
            "Final Risk Score",
            f"{risk:.1f}/100"
        )

        c4.metric(
            "Risk Level",
            f"{risk_symbol(category)} {category}"
        )

        reference_amount = None

        if amount_column is not None:

            try:

                reference_amount = pd.to_numeric(
                    analysis_df.loc[
                        analysis_df["Fraud_Label"] == 0,
                        amount_column
                    ],
                    errors="coerce"
                ).median()

            except Exception:
                reference_amount = None

        reasons = generate_reasons(
            selected,
            probability,
            anomaly,
            amount_column,
            reference_amount
        )

        st.subheader(
            "🧠 Why Was This Transaction Flagged?"
        )

        for reason in reasons:

            st.warning(
                f"• {reason}"
            )

        if category in [
            "CRITICAL",
            "HIGH"
        ]:

            st.error(
                "Recommended Action: Send transaction "
                "for manual fraud investigation."
            )

        elif category == "MEDIUM":

            st.warning(
                "Recommended Action: Perform additional verification."
            )

        else:

            st.success(
                "Recommended Action: Low immediate risk."
            )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

elif page == "🧪 What-If Analysis":

    st.title(
        "🧪 What-If Risk Analysis"
    )

    if st.session_state.model is None:

        st.warning(
            "Train the AI Risk Engine first."
        )

        st.stop()

    model = st.session_state.model[
        "model"
    ]

    preprocessor = (
        st.session_state.preprocessor
    )

    training_columns = (
        st.session_state.training_columns
    )

    base = {}

    for column in training_columns:

        if column in df.columns:

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                base[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).median()

            else:

                mode = df[column].mode()

                if len(mode) > 0:
                    base[column] = mode.iloc[0]
                else:
                    base[column] = ""

    st.write(
        "Change available numeric transaction features "
        "and observe how the predicted fraud probability changes."
    )

    editable = {}

    numeric_training = [
        column
        for column in training_columns
        if column in df.columns
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    important_numeric = numeric_training[:12]

    for column in important_numeric:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        min_value = float(
            values.quantile(0.01)
        )

        max_value = float(
            values.quantile(0.99)
        )

        if min_value == max_value:
            continue

        editable[column] = st.slider(
            column,
            min_value=min_value,
            max_value=max_value,
            value=float(
                np.clip(
                    values.median(),
                    min_value,
                    max_value
                )
            )
        )

    if st.button(
        "🔮 Calculate New Risk",
        use_container_width=True
    ):

        input_row = base.copy()

        for column, value in editable.items():
            input_row[column] = value

        input_df = pd.DataFrame(
            [input_row]
        )

        input_df = input_df[
            training_columns
        ]

        processed = preprocessor.transform(
            input_df
        )

        probability = model.predict_proba(
            processed
        )[0, 1]

        anomaly_model = (
            st.session_state.anomaly_model
        )

        anomaly_raw = -anomaly_model.decision_function(
            processed
        )[0]

        anomaly_score = float(
            np.clip(
                (
                    anomaly_raw + 0.5
                ),
                0,
                1
            )
        )

        risk = calculate_risk(
            probability,
            anomaly_score
        )

        category = risk_category(
            risk
        )

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        c2.metric(
            "Anomaly Score",
            f"{anomaly_score * 100:.2f}%"
        )

        c3.metric(
            "AI Risk Score",
            f"{risk:.1f}/100"
        )

        st.subheader(
            f"{risk_symbol(category)} {category} RISK"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk,
                title={
                    "text": "AI Risk Score"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Model Performance & Explainability"
    )

    if st.session_state.model is None:

        st.warning(
            "Train the AI Risk Engine first."
        )

        st.stop()

    model_data = st.session_state.model

    y_test = model_data["y_test"]

    predictions = model_data[
        "predictions"
    ]

    probabilities = model_data[
        "probabilities"
    ]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.3f}"
    )

    c3.metric(
        "Recall",
        f"{recall:.3f}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    c5.metric(
        "ROC-AUC",
        f"{auc:.3f}"
    )

    st.divider()

    cm = confusion_matrix(
        y_test,
        predictions
    )

    st.subheader(
        "Confusion Matrix"
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Predicted Normal",
            "Predicted Fraud"
        ],
        y=[
            "Actual Normal",
            "Actual Fraud"
        ],
        title="Fraud Detection Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "ROC Curve"
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Random Forest AUC = {auc:.3f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline"
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Feature Importance"
    )

    importance_df = (
        st.session_state.feature_importance
    )

    if importance_df is not None:

        top_features = (
            importance_df
            .head(20)
            .sort_values(
                "Importance"
            )
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Fraud Detection Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )


# ============================================================
# MODEL MONITORING
# ============================================================

elif page == "📡 Model Monitoring":

    st.title(
        "📡 Model & Data Monitoring"
    )

    if st.session_state.training_means is None:

        st.warning(
            "Train the AI Risk Engine first."
        )

        st.stop()

    st.subheader(
        "Training Data Reference"
    )

    training_means = (
        st.session_state.training_means
    )

    current_means = df[
        training_means.index
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).mean()

    drift_rows = []

    for column in training_means.index:

        train_value = training_means[
            column
        ]

        current_value = current_means[
            column
        ]

        if pd.isna(
            train_value
        ) or pd.isna(
            current_value
        ):

            continue

        if train_value == 0:

            drift_percentage = 0

        else:

            drift_percentage = abs(
                current_value - train_value
            ) / abs(
                train_value
            ) * 100

        if drift_percentage >= 50:

            status = "🔴 High Drift"

        elif drift_percentage >= 20:

            status = "🟠 Moderate Drift"

        else:

            status = "🟢 Low Drift"

        drift_rows.append(
            {
                "Feature": column,
                "Training Mean": train_value,
                "Current Mean": current_value,
                "Drift %": drift_percentage,
                "Status": status
            }
        )

    if drift_rows:

        drift_df = pd.DataFrame(
            drift_rows
        ).sort_values(
            "Drift %",
            ascending=False
        )

        st.dataframe(
            drift_df,
            use_container_width=True
        )

        fig = px.bar(
            drift_df.head(20),
            x="Drift %",
            y="Feature",
            orientation="h",
            title="Top Feature Distribution Changes"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "This monitoring page provides a simple analytical drift indicator "
        "by comparing feature means. It is intended for project monitoring, "
        "not production regulatory monitoring."
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📋 Dataset Explorer":

    st.title(
        "📋 Dataset Explorer"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Duplicates",
        f"{df.duplicated().sum():,}"
    )

    c4.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader(
        "Column Information"
    )

    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing": [
                df[c].isnull().sum()
                for c in df.columns
            ],
            "Unique": [
                df[c].nunique()
                for c in df.columns
            ]
        }
    )

    st.dataframe(
        info,
        use_container_width=True
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        df.describe(
            include="all"
        ).transpose(),
        use_container_width=True
    )

    st.download_button(
        "📥 Download Dataset",
        df.to_csv(index=False),
        "financial_fraud_dataset.csv",
        "text/csv"
    )


