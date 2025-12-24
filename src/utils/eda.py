import pandas as pd

class EDA:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    # Class imbalance
    def class_distribution(self):
        return self.df[self.target].value_counts(), \
               self.df[self.target].value_counts(normalize=True)

    # Global stats
    def univariate(self):
        return self.df.describe()

    # Numeric features vs fraud
    def bivariate(self):
        return self.df.groupby(self.target).mean(numeric_only=True)

    # For Fraud_Data.csv
    def categorical_vs_target(self, col):
        return pd.crosstab(self.df[col], self.df[self.target], normalize="index")

    # 🔥 Credit-card specific
    def amount_by_class(self):
        return self.df.groupby(self.target)["Amount"].describe()

    def fraud_over_time(self, time_col="Time", bins=10):
        self.df["time_bin"] = pd.qcut(self.df[time_col], bins)
        return self.df.groupby("time_bin")[self.target].mean()
