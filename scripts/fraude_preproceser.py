
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
from src.utils.preprocessing import Preprocessor
from src.utils.imbalance import ImbalanceHandler

class FraudPreprocessPipeline:
    def __init__(self, df, target, output_path):
        """
        df: pandas DataFrame
        target: name of the target column (e.g., "class" or "Class")
        output_path: path to save the processed, balanced dataset
        """
        self.df = df
        self.target = target
        self.output_path = output_path

    def run(self):
        # Split features and target
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

        # Encode categorical + scale numerical
        prep = Preprocessor()
        X = prep.encode(X)
        X = prep.scale(X)
        X = pd.DataFrame(X)

        # Handle class imbalance
        imb = ImbalanceHandler()
        before = imb.report(y)
        X_res, y_res = imb.resample(X, y)
        after = imb.report(y_res)

        # Combine features and target
        df_balanced = pd.concat([X_res, y_res], axis=1)

        # Save to configurable path
        df_balanced.to_csv(self.output_path, index=False)

        return {
            "before": before,
            "after": after,
            "data": df_balanced
        }
