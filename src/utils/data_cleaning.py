import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def handle_missing(self):
        # Numerical → median, Categorical → mode
        for col in self.df.columns:
            if self.df[col].dtype in ["int64", "float64"]:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self

    def correct_dtypes(self):
        time_cols = ["signup_time", "purchase_time"]
        for col in time_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
        return self

    def get_data(self):
        return self.df
