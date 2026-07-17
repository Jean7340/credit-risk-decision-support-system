# 🏦 HELOC Credit Risk Decision Support System

An end-to-end machine learning decision support application for Home Equity Line of Credit (HELOC) risk assessment.

The project demonstrates a complete analytics workflow, including:

- Data preprocessing and feature engineering
- Predictive modeling using Logistic Regression, Decision Tree, Random Forest and XGBoost
- Model evaluation and comparison
- Interactive credit risk prediction with Streamlit
- Business-oriented lending recommendations
- Model interpretability and visualization

---

## Project Overview

Financial institutions need to evaluate whether an applicant is likely to default before approving a Home Equity Line of Credit (HELOC).

This project develops a machine learning decision support system that predicts applicant default risk based on historical credit behavior and translates model outputs into business recommendations suitable for lending decisions.

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Joblib

## Results

🔗 Live Demo: https://xxxxx.streamlit.app

- Built an end-to-end machine learning application for credit risk assessment and lending decision support
- Achieved 72.2% prediction accuracy on the hold-out test set using Logistic Regression
- Generated applicant risk probabilities and lending recommendations
- Visualized feature importance, ROC curve, and confusion matrix

## Run Locally

```bash
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

## Project Structure

```text
HELOC-Credit-Risk-Decision-Support/
├── app/
├── data/
├── images/
├── models/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```