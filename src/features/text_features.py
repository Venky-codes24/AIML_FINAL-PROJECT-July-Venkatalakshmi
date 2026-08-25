"""TF-IDF vectorizer helpers for resume and job text representation."""

from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from src import config


def build_tfidf_vectorizer(
    max_features: int = config.MAX_FEATURES,
    ngram_range: Tuple[int, int] = config.NGRAM_RANGE,
    stop_words: str = "english",
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> TfidfVectorizer:
    """Create a configured TfidfVectorizer instance.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: Lower and upper boundary of range of n-values.
        stop_words: Stop words removal list/language.
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        min_df: Minimum document frequency threshold.

    Returns:
        Unfitted TfidfVectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )


def fit_transform_text(
    texts,
    vectorizer: Optional[TfidfVectorizer] = None,
    **kwargs
) -> Tuple[sp.csr_matrix, TfidfVectorizer]:
    """Fit a vectorizer on texts and return the transformed feature matrix and fitted vectorizer."""
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer(**kwargs)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def transform_text(
    texts,
    vectorizer: TfidfVectorizer,
) -> sp.csr_matrix:
    """Transform given texts using an already fitted TfidfVectorizer."""
    if isinstance(texts, str):
        texts = [texts]
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: Union[str, Path]) -> None:
    """Save a fitted TfidfVectorizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Union[str, Path]) -> TfidfVectorizer:
    """Load a saved TfidfVectorizer from disk.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vectorizer artifact not found at {path}. Please run 'python -m src.models.train' to train and save artifacts."
        )
    return joblib.load(path)
