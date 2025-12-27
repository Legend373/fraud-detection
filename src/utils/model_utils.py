import pandas as pd

def get_feature_importance(model, feature_names, top_n=10):
    importance = model.feature_importances_

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
