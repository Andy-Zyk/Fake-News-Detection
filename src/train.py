from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import load_raw_data
from src.evaluate import evaluate_predictions
from src.features import build_tfidf_features
from src.models.bert_model import BertModel
from src.models.lstm_model import LSTMModel
from src.models.tfidf_model import train_tfidf_model
from src.preprocess import preprocess_dataframe
from src.utils import ensure_dir, set_seed


def split_data(texts, labels, test_size: float = 0.2):
    """Split data for training and evaluation."""
    label_counts = labels.value_counts()
    if label_counts.shape[0] < 2:
        raise ValueError("Training data must contain at least two classes.")
    if int(label_counts.min()) < 2 or len(labels) < 10:
        # Tiny bootstrap data fallback: train and evaluate on full data.
        return texts, texts, labels, labels

    stratify_labels = labels
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels,
    )


def _prepare_data(data_source: str, data_dir: str, kaggle_dataset: str) -> pd.DataFrame:
    df = load_raw_data(
        data_source=data_source,
        data_dir=data_dir,
        kaggle_dataset=kaggle_dataset,
    )
    df = preprocess_dataframe(df, text_col="text")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a label column.")
    return df


def _train_tfidf(x_train_text, x_test_text, y_train, y_test, output_dir: str):
    x_train, x_test, vectorizer = build_tfidf_features(x_train_text, x_test_text)
    model = train_tfidf_model(x_train, y_train)
    preds = model.predict(x_test)
    metrics = evaluate_predictions(y_test, preds)

    model_path = Path(output_dir) / "tfidf_model.joblib"
    vec_path = Path(output_dir) / "tfidf_vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    return metrics


def _train_lstm(x_train_text, x_test_text, y_train, y_test, output_dir: str):
    model = LSTMModel(max_features=5000, random_state=42)
    model.fit(x_train_text, y_train)
    preds = model.predict(x_test_text)
    metrics = evaluate_predictions(y_test, preds)

    model_path = Path(output_dir) / "lstm_model.joblib"
    joblib.dump(model, model_path)
    return metrics


def _train_bert(x_train_text, x_test_text, y_train, y_test, output_dir: str):
    model = BertModel(random_state=42)
    model.fit(x_train_text, y_train)
    preds = model.predict(x_test_text)
    metrics = evaluate_predictions(y_test, preds)

    model_path = Path(output_dir) / "bert_model.joblib"
    joblib.dump(model, model_path)
    return metrics


def train_and_compare_models(
    data_source: str = "local",
    data_dir: str = "data/raw",
    kaggle_dataset: str = "clmentbisaillon/fake-and-real-news-dataset",
    output_dir: str = "results",
):
    """Train TF-IDF, LSTM-style baseline, and BERT-style baseline and compare metrics."""
    set_seed(42)
    ensure_dir(output_dir)

    df = _prepare_data(data_source=data_source, data_dir=data_dir, kaggle_dataset=kaggle_dataset)
    x_train_text, x_test_text, y_train, y_test = split_data(df["clean_text"], df["label"])

    results = {
        "tfidf": _train_tfidf(x_train_text, x_test_text, y_train, y_test, output_dir),
        "lstm": _train_lstm(x_train_text, x_test_text, y_train, y_test, output_dir),
        "bert": _train_bert(x_train_text, x_test_text, y_train, y_test, output_dir),
    }

    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(by="f1", ascending=False)
    comparison_path = Path(output_dir) / "metrics_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    return {
        "metrics": results,
        "comparison_path": str(comparison_path),
        "data_source": data_source,
    }
