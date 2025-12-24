import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.eda import EDA

class CreditCardEDAPipeline:
    def __init__(self, df):
        self.df = df

    def run(self):
        eda = EDA(self.df, target="Class")

        return {
            "class_counts": eda.class_distribution()[0],
            "class_ratio": eda.class_distribution()[1],
            "univariate": eda.univariate(),
            "bivariate": eda.bivariate(),
            "amount_analysis": eda.amount_by_class(),
            "fraud_over_time": eda.fraud_over_time()
        }
