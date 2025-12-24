import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)

class FraudModelPipeline:
    def __init__(self, csv_path, target):
        self.df = pd.read_csv(csv_path)
        self.target = target

    def split_data(self, test_size=0.2):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=42
        )

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return {
            "AUC_PR": average_precision_score(y_test, y_prob),
            "F1": f1_score(y_test, y_pred),
            "ConfusionMatrix": confusion_matrix(y_test, y_pred)
        }

    # -------------------------------
    # Logistic Regression Baseline
    # -------------------------------
    def train_logistic(self, X_train, y_train):
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    # -------------------------------
    # Random Forest Ensemble
    # -------------------------------
    def train_random_forest(self, X_train, y_train):
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None]
        }

        rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        grid = GridSearchCV(
            rf,
            param_grid,
            cv=3,
            scoring="average_precision",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        return grid.best_estimator_

    # -------------------------------
    # Stratified K-Fold CV
    # -------------------------------
    def cross_validate(self, model):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        aucs, f1s = [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            aucs.append(average_precision_score(y_test, y_prob))
            f1s.append(f1_score(y_test, y_pred))

        return {
            "AUC_PR_mean": np.mean(aucs),
            "AUC_PR_std": np.std(aucs),
            "F1_mean": np.mean(f1s),
            "F1_std": np.std(f1s)
        }

    # -------------------------------
    # Full Pipeline
    # -------------------------------
    def run(self):
        X_train, X_test, y_train, y_test = self.split_data()

        # Logistic Regression
        log_model = self.train_logistic(X_train, y_train)
        log_metrics = self.evaluate(log_model, X_test, y_test)
        log_cv = self.cross_validate(log_model)

        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate(rf_model, X_test, y_test)
        rf_cv = self.cross_validate(rf_model)

        return {
            "LogisticRegression": {
                "test_metrics": log_metrics,
                "cv_metrics": log_cv
            },
            "RandomForest": {
                "test_metrics": rf_metrics,
                "cv_metrics": rf_cv
            }
        }
