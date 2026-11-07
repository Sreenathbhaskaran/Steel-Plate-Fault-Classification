
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Steel Plate Fault Classification",
    page_icon="🏭",
    layout="wide"
)

# Application heading
st.title("🏭 Steel Plate Fault Classification System")

st.write(
    "Machine Learning application for predicting "
    "seven different types of steel plate faults."
)

st.divider()

# Model paths
model_paths = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# Load models
@st.cache_resource
def load_models():

    loaded_models = {}

    for name, path in model_paths.items():

        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)

    return loaded_models


models = load_models()

# Sidebar
st.sidebar.header("Model Selection")

if len(models) == 0:
    st.error("No trained models found. Check the model folder.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select a classification model",
    list(models.keys())
)

# Upload CSV
st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if "Fault_Type" not in data.columns:

        st.error("Target column Fault_Type is missing.")

    else:

        # Separate features and target
        X_test = data.drop(columns=["Fault_Type"])
        y_test = data["Fault_Type"]

        # Verify features
        selected_pipeline = models[selected_model]

        expected_features = (
            selected_pipeline
            .named_steps["preprocessor"]
            .feature_names_in_
        )

        if set(X_test.columns) != set(expected_features):

            st.error(
                "Uploaded CSV does not contain the expected features."
            )

        else:

            # Ensure expected column order
            X_test = X_test[list(expected_features)]

            # Predictions
            y_pred = selected_pipeline.predict(X_test)
            y_proba = selected_pipeline.predict_proba(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)

            auc = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=selected_pipeline.classes_
            )

            precision = precision_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            )

            mcc = matthews_corrcoef(y_test, y_pred)

            # Display metrics
            st.header("2. Model Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            # Confusion matrix
            st.header("3. Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=selected_pipeline.classes_
            )

            fig, ax = plt.subplots(figsize=(10, 7))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=selected_pipeline.classes_,
                yticklabels=selected_pipeline.classes_,
                ax=ax
            )

            ax.set_xlabel("Predicted Fault")
            ax.set_ylabel("Actual Fault")
            ax.set_title(selected_model + " Confusion Matrix")

            st.pyplot(fig)
            plt.close(fig)

            # Classification report
            st.header("4. Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
                output_dict=True
            )

            st.dataframe(
                pd.DataFrame(report).transpose()
            )

            # Predictions preview
            st.header("5. Prediction Results")

            prediction_results = pd.DataFrame({
                "Actual Fault": y_test.values,
                "Predicted Fault": y_pred
            })

            st.dataframe(prediction_results)

else:

    st.info("Please upload test_data.csv to view model results.")

# Footer
st.divider()

st.caption(
    "Steel Plate Fault Classification | "
    "Machine Learning Academic Project"
)
