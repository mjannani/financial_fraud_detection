import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
    roc_curve,
)

from imblearn.over_sampling import SMOTE


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates.default = "plotly_white"


# ============================================================
# FIXED LIGHT THEME CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL APP
   ========================================================= */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
.stApp {
    background-color: #f5f7fb !important;
    color: #111827 !important;
}

/* Main content area */
[data-testid="stAppViewContainer"] .main {
    background-color: #f5f7fb !important;
    color: #111827 !important;
}

/* Main block */
[data-testid="stAppViewContainer"] .main .block-container {
    background-color: #f5f7fb !important;
    color: #111827 !important;
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
}


/* =========================================================
   FORCE MAIN TEXT TO DARK
   ========================================================= */

[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main span,
[data-testid="stAppViewContainer"] .main label,
[data-testid="stAppViewContainer"] .main li,
[data-testid="stAppViewContainer"] .main small,
[data-testid="stAppViewContainer"] .main div {
    color: #111827;
}

/* Markdown text */
[data-testid="stMarkdownContainer"] {
    color: #111827 !important;
}
