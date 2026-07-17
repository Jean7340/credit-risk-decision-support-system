import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Paths & model loading
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Make src/ importable: required so joblib can unpickle the pipeline,
# whose first step is src.preprocessing.SpecialValueTransformer
sys.path.insert(0, str(BASE_DIR))
from src.preprocessing import SpecialValueTransformer  # noqa: E402,F401

st.set_page_config(page_title="HELOC Credit Risk", layout="wide")


@st.cache_resource
def load_artifact():
    return joblib.load(BASE_DIR / "models" / "heloc_pipeline.pkl")


artifact = load_artifact()
pipe = artifact["pipeline"]                 # calibrated end-to-end pipeline
FEATURES = artifact["feature_columns"]      # exact training column order
T_LOW = artifact["thresholds"]["approve_below"]
T_HIGH = artifact["thresholds"]["decline_from"]

st.title("🏦 HELOC Credit Risk Decision Support")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["🏠 Home", "📊 Prediction", "📈 Model Insights"],
)

# ------------------------------------------------------------------
# HOME
# ------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(f"""
    ### Project Overview

    This application demonstrates an end-to-end machine learning **decision-support
    system** for Home Equity Line of Credit (HELOC) risk assessment. It is designed
    to assist human underwriters — not to replace them: the model provides a
    calibrated probability of default and a recommendation tier, and ambiguous
    cases are routed to manual review.

    **The project includes**

    - Business-aware preprocessing of FICO credit-bureau special values
    - Four model candidates (Logistic Regression, Decision Tree, Random Forest, XGBoost)
      compared with hold-out and 5-fold cross-validated ROC-AUC
    - A calibrated Logistic Regression pipeline selected for its intrinsic,
      regulator-friendly interpretability
    - Cost-based decision thresholds:
      **Approve** below {T_LOW:.0%} · **Review** {T_LOW:.0%}–{T_HIGH:.0%} · **Decline** above {T_HIGH:.0%}

    The entire preprocessing-to-prediction chain is a single saved pipeline shared
    between training and this application, so predictions here exactly match the
    training-time methodology.
    """)

# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
if page == "📊 Prediction":
    st.title("📊 Credit Risk Prediction")
    st.write("Estimate applicant default risk using the calibrated logistic regression pipeline.")

    st.info("""
    For demonstration purposes, this application exposes 8 key applicant features selected
    based on business relevance and model importance. The remaining model features are
    assigned representative baseline values to simulate backend-generated credit bureau data.
    """)

    ExternalRiskEstimate = st.number_input("External Risk Estimate", value=70)
    MSinceOldestTradeOpen = st.number_input("Months Since Oldest Trade Open", value=200)
    NumTotalTrades = st.number_input("Total Number of Trades", value=20)
    NumInqLast6M = st.number_input("Number of Inquiries (Last 6 Months)", value=1)
    PercentTradesNeverDelq = st.number_input("Percent of Trades Never Delinquent", value=95)
    MaxDelqEver = st.number_input("Maximum Delinquency Ever", value=0)
    NetFractionRevolvingBurden = st.number_input("Net Revolving Burden", value=30)
    NumRevolvingTradesWBalance = st.number_input("Number of Revolving Trades with Balance", value=5)

    if st.button("Predict"):
        # Baselines for non-exposed features. Raw FICO-style values are fine here:
        # the pipeline's SpecialValueTransformer handles special codes such as -7.
        input_data = pd.DataFrame([{
            "ExternalRiskEstimate": ExternalRiskEstimate,
            "MSinceOldestTradeOpen": MSinceOldestTradeOpen,
            "MSinceMostRecentTradeOpen": 10,
            "AverageMInFile": 100,
            "NumSatisfactoryTrades": 20,
            "NumTrades60Ever2DerogPubRec": 0,
            "NumTrades90Ever2DerogPubRec": 0,
            "PercentTradesNeverDelq": PercentTradesNeverDelq,
            "MSinceMostRecentDelq": -7,          # -7 = never delinquent (clean baseline)
            "MaxDelq2PublicRecLast12M": 0,
            "MaxDelqEver": MaxDelqEver,
            "NumTotalTrades": NumTotalTrades,
            "NumTradesOpeninLast12M": 2,
            "PercentInstallTrades": 40,
            "MSinceMostRecentInqexcl7days": 6,
            "NumInqLast6M": NumInqLast6M,
            "NetFractionRevolvingBurden": NetFractionRevolvingBurden,
            "NetFractionInstallBurden": 20,
            "NumRevolvingTradesWBalance": NumRevolvingTradesWBalance,
            "NumInstallTradesWBalance": 3,
            "NumBank2NatlTradesWHighUtilization": 1,
            "PercentTradesWBalance": 40,
        }])

        # Align to the exact training column order (fails loudly if a column is missing)
        input_data = input_data[FEATURES]

        # Positive class = default (Bad = 1), so this IS the probability of default
        probability = pipe.predict_proba(input_data)[0, 1]

        st.subheader("Prediction Result")
        st.metric("Probability of Default (calibrated)", f"{probability:.1%}")

        st.subheader("Business Recommendation")
        if probability < T_LOW:
            st.success("✅ Recommendation: Approve")
            st.write("The applicant's estimated default risk is below the approval threshold.")
        elif probability < T_HIGH:
            st.warning("🟡 Recommendation: Manual Review")
            st.write(
                "The applicant falls in the ambiguous band where model errors are most "
                "likely — exactly where a human underwriter adds the most value."
            )
        else:
            st.error("🔴 Recommendation: Decline")
            st.write("The applicant's estimated default risk exceeds the decline threshold.")

        st.caption(
            f"Tiers from cost-based threshold analysis (missed defaulter ≈ 5× the cost of a "
            f"wrongly declined good applicant): Approve < {T_LOW:.0%} ≤ Review < {T_HIGH:.0%} ≤ Decline."
        )

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

# ------------------------------------------------------------------
# MODEL INSIGHTS
# ------------------------------------------------------------------
if page == "📈 Model Insights":
    st.title("📈 Model Insights")

    st.subheader("Top Important Features (Standardized Coefficients)")
    st.write(
        "Because features are standardized inside the pipeline, coefficient magnitudes are "
        "directly comparable. Red bars increase default risk; green bars decrease it."
    )
    st.image(str(BASE_DIR / "images" / "logistic_features.png"), use_column_width=True)

    st.subheader("Model Comparison")
    st.write("Hold-out ROC-AUC across the four candidate models.")
    st.image(str(BASE_DIR / "images" / "model_comparison.png"), use_column_width=True)

    st.subheader("Probability Calibration")
    st.write(
        "Isotonic calibration aligns predicted probabilities with observed default rates, "
        "so the percentage shown on the Prediction page is meaningful."
    )
    st.image(str(BASE_DIR / "images" / "calibration_curve.png"), use_column_width=True)

    st.subheader("Cost-Based Threshold Selection")
    st.write(
        "Expected cost across decision thresholds under an illustrative 5:1 cost ratio; "
        "the recommendation tiers are derived from the cost-optimal region."
    )
    st.image(str(BASE_DIR / "images" / "threshold_cost.png"), use_column_width=True)

    st.subheader("ROC Curve")
    st.image(str(BASE_DIR / "images" / "logistic_roc.png"), use_column_width=True)

    st.subheader("Confusion Matrix")
    st.image(str(BASE_DIR / "images" / "logistic_confusion.png"), use_column_width=True)
