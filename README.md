Fraud Detection System with Explainable AI
📌 Project Overview

This project builds an end-to-end financial fraud detection system using machine learning and explainable AI (XAI). The system is designed to identify fraudulent transactions with high accuracy while also providing transparent, auditable explanations for every prediction using SHAP (SHapley Additive Explanations).
It is built to meet real-world financial risk management and regulatory (Basel II/III) requirements.

🎯 Objectives

Detect fraudulent financial transactions

Minimize false positives

Provide human-interpretable explanations

Enable auditability for compliance and risk teams

Support deployment and CI/CD readiness

Task-1: Exploratory Data Analysis (EDA)

The EDA phase investigates:

Fraud vs non-fraud distribution

Class imbalance

Correlations between transaction features

Outliers and suspicious patterns

Key insights:

Fraud cases are extremely rare but have distinct value and behavior patterns

Certain transaction features strongly influence fraud likelihood

Heavy skewness requires careful preprocessing

⚙️ Task-2: Model Training

A supervised classification model is trained using:

Cleaned and scaled transaction features

Stratified train/test split

Optimized hyperparameters

Models evaluated:

Logistic Regression

Random Forest

Gradient Boosting

Final model selected based on:

ROC-AUC

Recall (Fraud Detection Rate)

Precision

F1-Score

The trained model is stored using:

joblib.dump(model, "models/fraud_model.joblib")


This format is production-safe, fast, and compatible with SHAP.

🔍 Task-3: Explainable AI (SHAP)

This task makes the model interpretable.

What SHAP Does

SHAP calculates how much each feature contributed to a prediction by comparing it against all possible feature combinations.

It provides:

Global explanations (What drives fraud in general?)

Local explanations (Why was this transaction flagged?)

Outputs

SHAP Summary Plot (global importance)

Feature ranking

Transaction-level explanations:

True Positive (correct fraud)

False Positive (mistaken fraud)

False Negative (missed fraud)

This ensures:

Model trust

Regulatory compliance (Basel II/III)

Analyst-friendly validation
