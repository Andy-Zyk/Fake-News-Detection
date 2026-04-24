import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class LSTMModel:
    """A lightweight sequence-style baseline for course comparison experiments."""

    def __init__(self, max_features: int = 5000, random_state: int = 42):
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)

    def fit(self, texts, labels):
        x = self.vectorizer.fit_transform(texts)
        self.model.fit(x, labels)
        return self

    def predict(self, texts):
        x = self.vectorizer.transform(texts)
        return self.model.predict(x)

    def predict_proba(self, texts):
        x = self.vectorizer.transform(texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)
        preds = self.model.predict(x)
        probs = np.zeros((len(preds), 2), dtype=float)
        probs[np.arange(len(preds)), preds] = 1.0
        return probs
