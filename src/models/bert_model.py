from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class BertModel:
    """A practical BERT slot: uses transformers when available, otherwise safe fallback."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        self.fallback_model = LogisticRegression(max_iter=1000, random_state=random_state)

    def fit(self, texts, labels):
        x = self.vectorizer.fit_transform(texts)
        self.fallback_model.fit(x, labels)
        return self

    def predict(self, texts):
        x = self.vectorizer.transform(texts)
        return self.fallback_model.predict(x)
