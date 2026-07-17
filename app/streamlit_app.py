
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="HELOC Credit Risk",
    layout="wide"
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_model.pkl"

model = joblib.load(MODEL_PATH)

st.title("🏦 HELOC Credit Risk Decision Support")

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "🏠 Home",
        "📊 Prediction",
        "📈 Model Insights"
    ]
)

# -------------------
# HOME
# -------------------

if page == "🏠 Home":

    st.markdown("""
    ### Project Overview

    This application demonstrates an end-to-end machine learning decision support system for Home Equity Line of Credit (HELOC) risk assessment.

    **The project includes**

    - Data preprocessing and feature engineering
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost
    - Model evaluation and comparison
    - Interactive credit risk prediction
    - Model interpretability and visualization

    The application supports credit lending decisions by estimating applicant default risk based on historical credit behavior.
    """)

# -------------------
# Prediction
# -------------------

if page == "📊 Prediction":

    st.title("📊 Credit Risk Prediction")

    st.write(
        "Estimate applicant default risk using the trained logistic regression model."
    )

    st.info("""
    For demonstration purposes, this application exposes 8 key applicant features selected based on business relevance and model importance.
    The remaining model features are assigned representative baseline values to simulate backend-generated credit bureau data.
    """)

    ExternalRiskEstimate = st.number_input(
        "External Risk Estimate", value=70
    )

    MSinceOldestTradeOpen = st.number_input(
        "Months Since Oldest Trade Open", value=200
    )

    NumTotalTrades = st.number_input(
        "Total Number of Trades", value=20
    )

    NumInqLast6M = st.number_input(
        "Number of Inquiries (Last 6 Months)", value=1
    )

    PercentTradesNeverDelq = st.number_input(
        "Percent of Trades Never Delinquent", value=95
    )

    MaxDelqEver = st.number_input(
        "Maximum Delinquency Ever", value=0
    )

    NetFractionRevolvingBurden = st.number_input(
        "Net Revolving Burden", value=30
    )

    NumRevolvingTradesWBalance = st.number_input(
        "Number of Revolving Trades with Balance", value=5
    )

    predict = st.button("Predict")

    if predict:

        input_data = pd.DataFrame([{

            "ExternalRiskEstimate": ExternalRiskEstimate,
            "MSinceOldestTradeOpen": MSinceOldestTradeOpen,
            "MSinceMostRecentTradeOpen": 10,
            "AverageMInFile": 100,
            "NumSatisfactoryTrades": 20,
            "NumTrades60Ever2DerogPubRec": 0,
            "NumTrades90Ever2DerogPubRec": 0,
            "PercentTradesNeverDelq": PercentTradesNeverDelq,
            "MSinceMostRecentDelq": 12,
            "MaxDelq2PublicRecLast12M": 0,
            "MaxDelqEver": MaxDelqEver,
            "NumTotalTrades": NumTotalTrades,
            "NumTradesOpeninLast12M": 2,
            "PercentInstallTrades": 40,
            "MSinceMostRecentInqexcl7days": 6,
            "NumInqLast6M": NumInqLast6M,
            "NumInqLast6Mexcl7days": 1,
            "NetFractionRevolvingBurden": NetFractionRevolvingBurden,
            "NetFractionInstallBurden": 20,
            "NumRevolvingTradesWBalance": NumRevolvingTradesWBalance,
            "NumInstallTradesWBalance": 3,
            "NumBank2NatlTradesWHighUtilization": 1,
            "PercentTradesWBalance": 40

        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Probability of Default",
            f"{probability:.1%}"
        )

        if probability < 0.30:
            st.success("Risk Level: Low")
        elif probability < 0.50:
            st.warning("Risk Level: Moderate")
        else:
            st.error("Risk Level: High")

        st.subheader("Business Recommendation")

        if probability < 0.30:
            st.success("✅ Recommendation: Approve")
            st.write("The applicant appears to have relatively low credit risk.")

        elif probability < 0.50:
            st.warning("🟡 Recommendation: Approve with Manual Review")
            st.write("The applicant may be acceptable, but additional review is recommended.")

        elif probability < 0.70:
            st.warning("🟠 Recommendation: Further Investigation Required")
            st.write("The applicant shows moderate credit risk and should be reviewed carefully.")

        else:
            st.error("🔴 Recommendation: High Risk / Consider Decline")
            st.write("The applicant shows elevated risk and may not be suitable for approval.")
        
        st.subheader("Key Risk Drivers")

        risk_drivers = []

        if ExternalRiskEstimate < 65:
            risk_drivers.append("Low external risk estimate")
        if NumInqLast6M >= 3:
            risk_drivers.append("High number of recent credit inquiries")
        if PercentTradesNeverDelq < 80:
            risk_drivers.append("Lower percentage of trades never delinquent")
        if MaxDelqEver > 0:
            risk_drivers.append("Historical delinquency record")
        if NetFractionRevolvingBurden > 50:
            risk_drivers.append("High revolving credit burden")

        if risk_drivers:
            for driver in risk_drivers:
                st.write(f"- {driver}")
        else:
            st.write("No major risk drivers detected based on the selected applicant inputs.")


# -------------------
# Model Insights
# -------------------

if page == "📈 Model Insights":

    st.title("📈 Model Insights")

    st.subheader("Top Important Features")
    st.write(
        "The chart below shows the variables with the largest influence on the logistic regression model."
    )
    st.image(
        str(BASE_DIR / "images" / "logistic_features.png"),
        use_column_width=True
    )

    st.subheader("ROC Curve")
    st.write(
        "The ROC curve evaluates the model's ability to distinguish between good and bad credit risk applicants."
    )
    st.image(
        str(BASE_DIR / "images" / "logistic_roc.png"),
        use_column_width=True
    )

    st.subheader("Confusion Matrix")
    st.write(
        "The confusion matrix summarizes correct and incorrect predictions on the hold-out test set."
    )
    st.image(
        str(BASE_DIR / "images" / "logistic_confusion.png"),
        use_column_width=True
    )