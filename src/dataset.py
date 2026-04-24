from pathlib import Path

import pandas as pd


DEFAULT_KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"


def _ensure_label(fake_df: pd.DataFrame, true_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure binary labels exist for fake/true records."""
    if "label" not in fake_df.columns:
        fake_df["label"] = 0
    if "label" not in true_df.columns:
        true_df["label"] = 1
    return fake_df, true_df


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input columns to a minimal schema used by the pipeline."""
    out = df.copy()
    if "text" not in out.columns:
        raise ValueError("Input data must contain a text column named 'text'.")
    if "title" not in out.columns:
        out["title"] = ""
    out["title"] = out["title"].fillna("").astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    return out


def _load_from_local(data_dir: str, fake_file: str, true_file: str) -> pd.DataFrame:
    """Load Fake.csv and True.csv from local data/raw directory."""
    data_path = Path(data_dir)
    fake_path = data_path / fake_file
    true_path = data_path / true_file

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(f"Expected {fake_file} and {true_file} in {data_dir}")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df = _normalize_schema(fake_df)
    true_df = _normalize_schema(true_df)
    fake_df, true_df = _ensure_label(fake_df, true_df)
    return pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def _load_from_kagglehub(
    kaggle_dataset: str,
    fake_file: str,
    true_file: str,
    cache_data_dir: str,
) -> pd.DataFrame:
    """Load Fake.csv and True.csv via kagglehub and cache to data/raw."""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Install with: uv add 'kagglehub[pandas-datasets]'"
        ) from exc

    fake_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        kaggle_dataset,
        fake_file,
    )
    true_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        kaggle_dataset,
        true_file,
    )
    fake_df = _normalize_schema(fake_df)
    true_df = _normalize_schema(true_df)
    fake_df, true_df = _ensure_label(fake_df, true_df)
    combined = pd.concat([fake_df, true_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    cache_path = Path(cache_data_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fake_df.to_csv(cache_path / fake_file, index=False)
    true_df.to_csv(cache_path / true_file, index=False)
    return combined


def load_raw_data(
    data_source: str = "local",
    data_dir: str = "data/raw",
    kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
    fake_file: str = "Fake.csv",
    true_file: str = "True.csv",
) -> pd.DataFrame:
    """Load and combine fake/true data from local files or KaggleHub."""
    if data_source == "local":
        return _load_from_local(data_dir, fake_file, true_file)
    if data_source == "kagglehub":
        return _load_from_kagglehub(kaggle_dataset, fake_file, true_file, data_dir)

    raise ValueError("data_source must be one of: local, kagglehub")


def load_dataset(fake_path: str, real_path: str) -> pd.DataFrame:
    """Course-friendly API: load fake/real CSV files and return a shuffled dataframe."""
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake = _normalize_schema(fake)
    real = _normalize_schema(real)
    fake["label"] = 0
    real["label"] = 1
    data = pd.concat([fake, real], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data
