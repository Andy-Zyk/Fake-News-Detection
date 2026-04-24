from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(texts, max_features: int = 5000) -> Tuple:
    """Fit TF-IDF vectorizer and return sparse matrix with vectorizer."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    x = vectorizer.fit_transform(texts)
    return x, vectorizer


def build_tfidf_features(train_texts, test_texts, max_features: int = 5000) -> Tuple:
    """Fit TF-IDF on train texts and transform test texts."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return x_train, x_test, vectorizer
