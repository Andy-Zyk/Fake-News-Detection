import re
from typing import Optional

import pandas as pd


STOP_WORDS = {
    "the",
    "is",
    "in",
    "and",
    "to",
    "of",
    "a",
    "an",
    "for",
    "on",
    "with",
    "that",
    "this",
    "it",
    "as",
    "at",
    "by",
}


def preprocess_text(text: Optional[str]) -> str:
    """Preprocess text in a classroom-friendly and explainable way."""
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.lower()

    text = "".join(char for char in text if char.isalnum() or char.isspace())

    tokens = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(tokens)


def clean_text(text: Optional[str]) -> str:
    """Compatibility alias used by training and demo modules."""
    return preprocess_text(text)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Create a cleaned text column used by feature builders."""
    out = df.copy()
    if text_col not in out.columns:
        raise ValueError(f"Missing text column: {text_col}")
    out["processed_text"] = out[text_col].apply(lambda x: preprocess_text(str(x)))
    out["clean_text"] = out["processed_text"]
    return out
