"""
app.py
------
Streamlit web app to demonstrate 5 pre-trained classification models on the
Iris dataset. Supports multi-class (3 species) classification.

Features:
    a. CSV upload (test data only)
    b. Model selection dropdown
    c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    d. Confusion matrix + classification report
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

MODEL_DIR = "model"
TARGET_COL = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

# Iris species names for better readability
SPECIES_NAMES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.set_page_config(page_title="Iris ML Classification Demo", layout="wide")


@st.cache_resource
def load_artifacts(model_filename: str):
    """Load a trained model plus the shared scaler and feature list."""
    model = joblib.load(os.path.join(MODEL_DIR, model_filename))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, scaler, feature_names


def compute_metrics(y_true, y_pred, y_score, n_classes=3):
    """Compute 6 evaluation metrics for multi-class classification."""
    
    # Binarize for AUC calculation
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    try:
        auc = roc_auc_score(y_bin, y_score, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("🌸 Iris Species Classification Demo")
    st.write(
        "Upload the test dataset (CSV), pick a trained model, and view "
        "its evaluation metrics and confusion matrix."
    )

    # --- a. Dataset upload -------------------------------------------------
    st.sidebar.header("1. Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv", type=["csv"]
    )

    # --- b. Model selection ------------------------------------------------
    st.sidebar.header("2. Select Model")
    model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

    if uploaded_file is None:
        st.info("👈 Upload `test_data.csv` from the sidebar to get started.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(10))

    if TARGET_COL not in df.columns:
        st.error(
            f"Uploaded CSV must contain a '{TARGET_COL}' column with the true "
            "labels (as produced by train_models.py)."
        )
        st.stop()

    try:
        model, scaler, feature_names = load_artifacts(MODEL_FILES[model_name])
    except FileNotFoundError:
        st.error(
            "Trained model files not found. Run `train_models.py` first "
            "so the `model/` folder is populated."
        )
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df[TARGET_COL]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Get prediction probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_scaled)
    else:
        y_score = model.decision_function(X_scaled)
        if len(y_score.shape) == 1:
            y_score = np.column_stack([y_score, -y_score])

    # --- c. Evaluation metrics ---------------------------------------------
    st.subheader(f"📊 Evaluation Metrics — {model_name}")
    metrics = compute_metrics(y_true, y_pred, y_score, n_classes=3)

    cols = st.columns(len(metrics))
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    # --- d. Confusion matrix + classification report -----------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[SPECIES_NAMES[i] for i in range(3)],
        yticklabels=[SPECIES_NAMES[i] for i in range(3)],
        ax=ax,
    )
    ax.set_ylabel("Actual Species")
    ax.set_xlabel("Predicted Species")
    st.pyplot(fig)

    st.subheader("Classification Report")
    target_names = [SPECIES_NAMES[i] for i in range(3)]
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

    st.subheader("Prediction Details")
    results_df = pd.DataFrame({
        "True Species": [SPECIES_NAMES[i] for i in y_true],
        "Predicted Species": [SPECIES_NAMES[i] for i in y_pred],
        "Correct": y_true.values == y_pred,
    })
    st.dataframe(results_df)

    st.success("Prediction and evaluation complete ✅")


if __name__ == "__main__":
    main()
