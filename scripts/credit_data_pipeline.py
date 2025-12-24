import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
from src.utils.data_cleaning import DataCleaner

class CreditDataPipeline:
    def __init__(self, data_path):
        self.data_path = data_path

    def run(self):
        df = pd.read_csv(self.data_path)

        df = (
            DataCleaner(df)
            .handle_missing()
            .remove_duplicates()
            .get_data()
        )

        df.to_csv("../data/processed/credit_clean.csv", index=False)
        return df
