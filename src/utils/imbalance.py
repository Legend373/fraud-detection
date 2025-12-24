from imblearn.over_sampling import SMOTE

class ImbalanceHandler:
    def __init__(self):
        self.smote = SMOTE(random_state=42)

    def report(self, y):
        return y.value_counts(normalize=True)

    def resample(self, X, y):
        return self.smote.fit_resample(X, y)
