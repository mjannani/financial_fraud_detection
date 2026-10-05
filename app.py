
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE


st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)


def load_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def clean_data(df):
    df = df.copy()
    df.columns = [
        str(c).strip().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df.drop_duplicates()


def detect_target(columns):
    candidates = [
        "class", "fraud", "is_fraud", "fraud_flag",
        "fraudulent", "isfraud", "fraud_status",
        "label", "target"
    ]
    normalized = {
        str(c).lower().replace(" ", "_").replace("-", "_"): c
        for c in columns
    }

    for c in candidates:
        if c in normalized:
            return normalized[c]

    for c in columns:
        if "fraud" in str(c).lower():
            return c

    return None


def encode_target(series):
    if pd.api.types.is_numeric_dtype(series):
        values = sorted(series.dropna().unique())
        if len(values) == 2:
            return series.map({values[0]: 0, values[1]: 1})

    text = series.astype(str).str.lower().str.strip()
    fraud_words = {
        "fraud", "yes", "true", "1",
        "fraudulent", "positive", "suspicious"
    }
    values = list(text.dropna().unique())

    if len(values) == 2:
        mapping = {
            value: 1 if value in fraud_words else 0
            for value in values
        }
        return text.map(mapping)

    return None


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []

    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, numeric))

    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])
        transformers.append(("cat", cat_pipe, categorical))

    return ColumnTransformer(transformers=transformers)


def feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array([])


if "df" not in st.session_state:
    st.session_state.df = None
if "target" not in st.session_state:
    st.session_state.target = None
if "model" not in st.session_state:
    st.session_state.model = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "results" not in st.session_state:
    st.session_state.results = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None


st.sidebar.title("💳 Fraud Intelligence")
uploaded = st.sidebar.file_uploader(
    "Upload Financial Fraud Dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded is not None:
    try:
        st.session_state.df = clean_data(load_file(uploaded))
        detected = detect_target(st.session_state.df.columns)
        if detected:
            st.session_state.target = detected
    except Exception as e:
        st.error(f"Could not load file: {e}")
        st.stop()

if st.session_state.df is None:
    st.title("💳 Financial Fraud Detection & Risk Analytics")
    st.markdown("""
    ### Streamlit Data Analytics Application

    Upload one of your Zidio fraud datasets from the sidebar.

    **Included modules**
    - Executive KPI dashboard
    - Exploratory Data Analysis
    - Fraud pattern analysis
    - Transaction amount analysis
    - SMOTE class balancing
    - Logistic Regression
    - Random Forest
    - Model comparison
    - Confusion matrix
    - ROC-AUC
    - Feature importance
    - Batch fraud-risk prediction
    """)
    st.info("Upload your CSV/XLSX dataset to begin.")
    st.stop()


df = st.session_state.df

target = st.sidebar.selectbox(
    "Fraud / Target Column",
    df.columns.tolist(),
    index=(
        df.columns.tolist().index(st.session_state.target)
        if st.session_state.target in df.columns
        else 0
    )
)
st.session_state.target = target

pages = [
    "🏠 Executive Overview",
    "📊 Data Analysis",
    "🚨 Fraud Analysis",
    "🤖 Model Training",
    "📈 Model Performance",
    "🔍 Risk Prediction",
    "📋 Dataset"
]
page = st.sidebar.radio("Navigation", pages)

target_encoded = encode_target(df[target])

if target_encoded is None:
    st.error(
        "The selected target column must contain exactly two classes, "
        "such as 0/1, Normal/Fraud, or Yes/No."
    )
    st.stop()

work_df = df.copy()
work_df["_Fraud_Label"] = target_encoded

total = len(work_df)
fraud = int((work_df["_Fraud_Label"] == 1).sum())
normal = int((work_df["_Fraud_Label"] == 0).sum())
fraud_rate = fraud / total * 100 if total else 0


if page == "🏠 Executive Overview":
    st.title("💳 Financial Fraud Detection & Risk Analytics")
    st.caption("Data Analytics + Machine Learning + Streamlit")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Transactions", f"{fraud:,}")
    c3.metric("Normal Transactions", f"{normal:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    distribution = pd.DataFrame({
        "Status": ["Normal", "Fraud"],
        "Count": [normal, fraud]
    })

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            distribution, x="Status", y="Count",
            text="Count", title="Normal vs Fraud Transactions"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.pie(
            distribution, names="Status", values="Count",
            hole=.45, title="Transaction Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    amount_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if "amount" in str(c).lower()
    ]

    if amount_cols:
        amount_col = amount_cols[0]
        c1, c2 = st.columns(2)
        c1.metric(
            "Total Transaction Value",
            f"{df[amount_col].sum():,.2f}"
        )
        c2.metric(
            "Fraud Transaction Value",
            f"{work_df.loc[work_df['_Fraud_Label']==1, amount_col].sum():,.2f}"
        )

    st.subheader("Business Summary")
    st.success(
        f"The dataset contains {total:,} transactions, including "
        f"{fraud:,} fraudulent transactions. The observed fraud rate "
        f"is {fraud_rate:.3f}%."
    )


elif page == "📊 Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicates", f"{df.duplicated().sum():,}")
    c4.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.subheader("Missing Values")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values found.")
    else:
        m = missing.reset_index()
        m.columns = ["Column", "Missing Values"]
        fig = px.bar(m, x="Column", y="Missing Values")
        st.plotly_chart(fig, use_container_width=True)

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    if numeric:
        col = st.selectbox("Select numeric feature", numeric)
        fig = px.histogram(
            df, x=col, nbins=50,
            title=f"Distribution of {col}"
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(numeric) >= 2:
            corr = df[numeric].corr()
            fig = px.imshow(
                corr, aspect="auto",
                title="Numeric Feature Correlation"
            )
            st.plotly_chart(fig, use_container_width=True)


elif page == "🚨 Fraud Analysis":
    st.title("🚨 Fraud Pattern Analysis")

    amount_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if "amount" in str(c).lower()
    ]

    if amount_cols:
        amount_col = amount_cols[0]
        fig = px.box(
            work_df,
            x="_Fraud_Label",
            y=amount_col,
            title="Transaction Amount: Normal vs Fraud"
        )
        st.plotly_chart(fig, use_container_width=True)

        summary = work_df.groupby("_Fraud_Label")[amount_col].agg(
            ["count", "mean", "median", "max"]
        ).reset_index()
        summary["_Fraud_Label"] = summary["_Fraud_Label"].map(
            {0: "Normal", 1: "Fraud"}
        )
        st.dataframe(summary, use_container_width=True)

    time_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if str(c).lower() == "time"
    ]

    if time_cols:
        time_col = time_cols[0]
        temp = work_df.copy()
        temp["Hour"] = (temp[time_col] / 3600) % 24
        hourly = temp.groupby("Hour")["_Fraud_Label"].sum().reset_index()
        hourly.columns = ["Hour", "Fraud Transactions"]

        fig = px.line(
            hourly, x="Hour", y="Fraud Transactions",
            markers=True, title="Fraud Transactions by Hour"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Rate")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_rate,
        title={"text": "Fraud Rate (%)"},
        gauge={"axis": {"range": [0, max(1, fraud_rate * 2)]}}
    ))
    st.plotly_chart(fig, use_container_width=True)


elif page == "🤖 Model Training":
    st.title("🤖 Fraud Detection Model Training")

    X = work_df.drop(columns=[target, "_Fraud_Label"])
    y = work_df["_Fraud_Label"].astype(int)

    if y.nunique() != 2:
        st.error("Target must contain exactly two classes.")
        st.stop()

    if y.value_counts().min() < 2:
        st.error("The minority class has fewer than two records.")
        st.stop()

    test_size = st.slider("Test Size", .10, .40, .20, .05)
    random_state = st.number_input(
        "Random State", min_value=1, max_value=999, value=42
    )

    if st.button("🚀 Train Models", use_container_width=True):
        with st.spinner("Training models..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=int(random_state),
                stratify=y
            )

            preprocessor = make_preprocessor(X_train)

            X_train_p = preprocessor.fit_transform(X_train)
            X_test_p = preprocessor.transform(X_test)

            try:
                smote = SMOTE(random_state=int(random_state))
                X_train_b, y_train_b = smote.fit_resample(
                    X_train_p, y_train
                )
            except Exception:
                X_train_b, y_train_b = X_train_p, y_train

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=150,
                    random_state=int(random_state),
                    n_jobs=-1,
                    class_weight="balanced"
                )
            }

            results = []
            trained = {}

            for name, model in models.items():
                model.fit(X_train_b, y_train_b)
                pred = model.predict(X_test_p)
                prob = model.predict_proba(X_test_p)[:, 1]

                result = {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(
                        y_test, pred, zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test, pred, zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_test, pred, zero_division=0
                    ),
                    "ROC-AUC": roc_auc_score(y_test, prob)
                }
                results.append(result)

                trained[name] = {
                    "model": model,
                    "predictions": pred,
                    "probabilities": prob,
                    "y_test": y_test
                }

            results_df = pd.DataFrame(results)
            st.session_state.results = results_df
            st.session_state.model = trained["Random Forest"]
            st.session_state.preprocessor = preprocessor

            names = feature_names(preprocessor)
            rf = trained["Random Forest"]["model"]

            if hasattr(rf, "feature_importances_"):
                importance = pd.DataFrame({
                    "Feature": names,
                    "Importance": rf.feature_importances_
                }).sort_values("Importance", ascending=False)
                st.session_state.feature_importance = importance

            joblib.dump(rf, "fraud_model.pkl")
            joblib.dump(preprocessor, "fraud_preprocessor.pkl")

            st.success("Models trained successfully!")
            st.dataframe(
                results_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}"
                }),
                use_container_width=True
            )

            st.download_button(
                "Download Model Results",
                results_df.to_csv(index=False),
                "model_results.csv",
                "text/csv"
            )


elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    if st.session_state.results is None:
        st.warning("Train the models first.")
        st.stop()

    results = st.session_state.results

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True
    )

    metric = st.selectbox(
        "Performance Metric",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )

    fig = px.bar(
        results, x="Model", y=metric,
        text=metric, title=f"{metric} Comparison"
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.feature_importance is not None:
        st.subheader("Top Fraud Detection Features")
        top = st.session_state.feature_importance.head(20)
        fig = px.bar(
            top.sort_values("Importance"),
            x="Importance", y="Feature",
            orientation="h",
            title="Top 20 Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

    data = st.session_state.model
    cm = confusion_matrix(data["y_test"], data["predictions"])

    st.subheader("Random Forest Confusion Matrix")
    fig = px.imshow(
        cm, text_auto=True,
        x=["Predicted Normal", "Predicted Fraud"],
        y=["Actual Normal", "Actual Fraud"]
    )
    st.plotly_chart(fig, use_container_width=True)

    report = classification_report(
        data["y_test"],
        data["predictions"],
        output_dict=True,
        zero_division=0
    )
    st.subheader("Classification Report")
    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )


elif page == "🔍 Risk Prediction":
    st.title("🔍 Transaction Risk Prediction")

    if st.session_state.model is None:
        st.warning("Train the model first.")
        st.stop()

    prediction_file = st.file_uploader(
        "Upload transactions for prediction",
        type=["csv", "xlsx", "xls"],
        key="prediction_upload"
    )

    if prediction_file is not None:
        try:
            pred_df = load_file(prediction_file)
            original = pred_df.copy()

            if target in pred_df.columns:
                pred_df = pred_df.drop(columns=[target])

            model = st.session_state.model["model"]
            prep = st.session_state.preprocessor

            X_pred = prep.transform(pred_df)
            probabilities = model.predict_proba(X_pred)[:, 1]
            predictions = model.predict(X_pred)

            result = original.copy()
            result["Fraud Probability (%)"] = np.round(
                probabilities * 100, 2
            )
            result["Prediction"] = np.where(
                predictions == 1,
                "Potential Fraud",
                "Normal"
            )
            result["Risk Level"] = pd.cut(
                probabilities,
                bins=[-0.01, .30, .70, 1.01],
                labels=["Low", "Medium", "High"]
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions Analyzed", len(result))
            c2.metric(
                "Potential Fraud",
                int((predictions == 1).sum())
            )
            c3.metric(
                "High Risk",
                int((probabilities >= .70).sum())
            )

            st.dataframe(result, use_container_width=True)

            st.download_button(
                "Download Prediction Results",
                result.to_csv(index=False),
                "fraud_prediction_results.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.info(
        "Risk thresholds used for dashboard presentation: "
        "Low <30%, Medium 30–70%, High >70%."
    )


elif page == "📋 Dataset":
    st.title("📋 Dataset Explorer")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

    st.subheader("Dataset Preview")
    n = st.slider(
        "Rows to display",
        5, min(100, len(df)), 10
    )
    st.dataframe(
        df.head(n),
        use_container_width=True
    )

    st.subheader("Column Information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(x) for x in df.dtypes],
        "Missing Values": [df[c].isnull().sum() for c in df.columns],
        "Unique Values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

st.sidebar.divider()
st.sidebar.caption("Financial Fraud Detection & Risk Analytics")
