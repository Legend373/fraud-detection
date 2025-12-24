import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
from src.utils.data_cleaning import DataCleaner
from src.utils.geoip import GeoIPMapper
from src.utils.feature_engineering import FraudFeatureEngineer

class FraudDataPipeline:
    def __init__(self, fraud_path, ip_path):
        self.fraud_path = fraud_path
        self.ip_path = ip_path

    def run(self):
        df = pd.read_csv(self.fraud_path)
        ip = pd.read_csv(self.ip_path)

        # Data cleaning
        df = (
            DataCleaner(df)
            .handle_missing()
            .remove_duplicates()
            .correct_dtypes()
            .get_data()
        )

        # GeoIP merge
        df = GeoIPMapper(df, ip).convert_ips().merge_country()

        # Feature engineering
        df = (
            FraudFeatureEngineer(df)
            .time_features()
            .transaction_velocity(window="1h")
            .get_data()
        )

        df.to_csv("../data/processed/fraud_features.csv", index=False)
        return df
