from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


def train_tfidf_model(x_train, y_train) -> LogisticRegression:
    """Train a logistic regression classifier on TF-IDF features."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    return model


class TfidfBaselineModel:
    """A simple logistic regression baseline on TF-IDF features."""

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)

    def fit(self, x_train, y_train) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x_test, y_test) -> Dict[str, float]:
        preds = self.predict(x_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "report": classification_report(y_test, preds),
        }
