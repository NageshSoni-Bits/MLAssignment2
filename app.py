import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Bank Marketing ML App", layout="wide")
st.title("Bank Marketing Classification App")
st.write("Predict whether a customer will subscribe to a term deposit.")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return models, scaler


models, scaler = load_models()

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
st.sidebar.header("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())

    if "deposit" not in data.columns:
        st.error("Uploaded file must contain 'deposit' column as target.")
    else:

        # Encode target
        data["deposit"] = data["deposit"].map({"yes": 1, "no": 0})

        # One-hot encoding
        data = pd.get_dummies(data, drop_first=True)

        X = data.drop("deposit", axis=1)
        y = data["deposit"]
        X = X.reindex(columns=feature_columns, fill_value=0)
        # Scale features
        X = scaler.transform(X)

        # ---------------------------------------------------
        # Model Selection
        # ---------------------------------------------------
        model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
        model = models[model_name]

        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # ---------------------------------------------------
        # Evaluation Metrics
        # ---------------------------------------------------
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # ---------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

else:
    st.info("Please upload a CSV test dataset to begin.")
