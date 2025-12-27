import pandas as pd
import numpy as np

# -------------------------------
# Function to summarize model performance
# -------------------------------
def summarize_models(results, dataset_name):
    summary = []
    for model_name, metrics in results.items():
        test = metrics["test_metrics"]
        cv = metrics["cv_metrics"]
        summary.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "AUC_PR_test": test["AUC_PR"],
            "F1_test": test["F1"],
            "AUC_PR_CV_mean": cv["AUC_PR_mean"],
            "AUC_PR_CV_std": cv["AUC_PR_std"],
            "F1_CV_mean": cv["F1_mean"],
            "F1_CV_std": cv["F1_std"]
        })
    return pd.DataFrame(summary)

def select_best_model(results):
    # Priority: High AUC-PR, High F1, Interpretability (Logistic > RF if comparable)
    best_model = None
    best_auc = -np.inf
    for model_name, metrics in results.items():
        if metrics["test_metrics"]["AUC_PR"] > best_auc:
            best_auc = metrics["test_metrics"]["AUC_PR"]
            best_model = model_name
    return best_model