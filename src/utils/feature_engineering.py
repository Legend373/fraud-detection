import pandas as pd
import numpy as np

class FraudFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def time_features(self):
        # Ensure datetime types
        self.df["purchase_time"] = pd.to_datetime(self.df["purchase_time"], errors="coerce")
        self.df["signup_time"] = pd.to_datetime(self.df["signup_time"], errors="coerce")

        # Fill missing purchase_time or signup_time with a placeholder (or drop)
        self.df["purchase_time"] = self.df["purchase_time"].fillna(method="ffill")
        self.df["signup_time"] = self.df["signup_time"].fillna(self.df["purchase_time"])

        # Extract time features
        self.df["hour_of_day"] = self.df["purchase_time"].dt.hour
        self.df["day_of_week"] = self.df["purchase_time"].dt.dayofweek
        self.df["time_since_signup"] = (
            self.df["purchase_time"] - self.df["signup_time"]
        ).dt.total_seconds()

        return self

    def transaction_velocity(self, window="1h"):
     """
     Count number of transactions per user in a rolling window.
     Handles duplicate timestamps per user.
     """
     self.df = self.df.sort_values("purchase_time").copy()

     # Function to apply rolling per user
     def rolling_count(group):
        group = group.copy()

        # Make timestamps unique by adding tiny delta to duplicates
        group["purchase_time"] += pd.to_timedelta(np.arange(len(group)) * 1e-6, unit="s")

        # Set index for rolling
        group.set_index("purchase_time", inplace=True)

        # Rolling count
        group["txn_count_1h"] = group["user_id"].rolling(window=window).count()
        return group.reset_index()

    # Apply rolling per user
     self.df = pd.concat(
        [rolling_count(g) for _, g in self.df.groupby("user_id")],
        ignore_index=True
     )

     return self


    def get_data(self):
        return self.df
