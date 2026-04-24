from typing import Dict

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    """Compute common binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred),
    }


def evaluate(y_true, y_pred):
    """Course-friendly API: return acc, precision, recall, f1."""
    metrics = evaluate_predictions(y_true, y_pred)
    return metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]
