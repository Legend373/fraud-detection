import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def encode(self, df):
        df = df.copy()

        # ❌ Drop columns that must never go into ML
        drop_cols = [
            "user_id",
            "device_id",
            "ip_address",
            "ip_int",
            "purchase_time",
            "signup_time"
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Encode only safe low-cardinality categoricals
        cat_cols = [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() < 50
        ]

        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        return df

    def scale(self, df):
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        return df
