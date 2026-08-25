"""Supervised resume category classifier (train / predict / save / load).

Uses TF-IDF feature extraction and Logistic Regression to classify resume texts
into target job roles/categories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src import config
from src.data.preprocess import clean_text
from src.evaluate import evaluate_classifier
from src.features.text_features import (
    build_tfidf_vectorizer,
    load_vectorizer,
    save_vectorizer,
    transform_text,
)


def train_classifier(
    resumes_df: Optional[pd.DataFrame] = None,
    save: bool = True,
    test_size: float = 0.2,
    random_state: int = config.RANDOM_STATE,
    max_iter: int = 1000,
    C: float = 1.0,
) -> Dict[str, Any]:
    """Train a TF-IDF + Logistic Regression classifier on resume dataset.

    Args:
        resumes_df: Optional DataFrame with 'Category' and 'text' columns.
        save: If True, saves model and vectorizer artifacts to models/.
        test_size: Ratio of data reserved for test evaluation.
        random_state: Random state seed.
        max_iter: Maximum iterations for solver convergence.
        C: Inverse regularization strength.

    Returns:
        Dict with model, vectorizer, categories, and test evaluation metrics.
    """
    if resumes_df is None:
        if not config.RESUMES_CLEAN_CSV.exists():
            raise FileNotFoundError(
                f"Cleaned resumes dataset not found at {config.RESUMES_CLEAN_CSV}. "
                "Please run 'python -m src.data.preprocess' first."
            )
        resumes_df = pd.read_csv(config.RESUMES_CLEAN_CSV)

    resumes_df = resumes_df.dropna(subset=["Category", "text"])
    resumes_df = resumes_df[resumes_df["text"].str.strip() != ""]

    X = np.array(resumes_df["text"].tolist(), dtype=object)
    y = np.array(resumes_df["Category"].tolist(), dtype=object)

    # Stratified train-test split for balanced category representation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Feature extraction via TF-IDF
    vectorizer = build_tfidf_vectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(X_train_vec, y_train)

    # Evaluate on held-out test split
    y_pred = clf.predict(X_test_vec)
    metrics = evaluate_classifier(y_test, y_pred)

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, config.RESUME_CLASSIFIER_PATH)
        save_vectorizer(vectorizer, config.RESUME_VECTORIZER_PATH)
        print(f"Classifier saved -> {config.RESUME_CLASSIFIER_PATH}")
        print(f"Vectorizer saved -> {config.RESUME_VECTORIZER_PATH}")

    return {
        "model": clf,
        "vectorizer": vectorizer,
        "classes": sorted(list(clf.classes_)),
        "metrics": metrics,
    }


def load_classifier() -> Tuple[LogisticRegression, Any]:
    """Load the saved resume classifier and TF-IDF vectorizer.

    Returns:
        Tuple of (model, vectorizer).

    Raises:
        FileNotFoundError: If model artifacts are missing.
    """
    if not config.RESUME_CLASSIFIER_PATH.exists() or not config.RESUME_VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {config.MODELS_DIR}. "
            "Please run 'python -m src.models.train' to build and save them."
        )

    model = joblib.load(config.RESUME_CLASSIFIER_PATH)
    vectorizer = load_vectorizer(config.RESUME_VECTORIZER_PATH)
    return model, vectorizer


def predict_category(
    resume_text: str,
    model: Optional[LogisticRegression] = None,
    vectorizer: Optional[Any] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Predict the target role category and confidence for a resume text.

    Args:
        resume_text: Raw or cleaned resume text string.
        model: Preloaded model instance (or None to load automatically).
        vectorizer: Preloaded vectorizer instance (or None to load automatically).
        top_k: Number of top alternative categories to return.

    Returns:
        Dict containing:
            - 'predicted_category': str
            - 'confidence': float (0.0 to 1.0)
            - 'top_categories': List[Tuple[str, float]]
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_classifier()

    cleaned = clean_text(resume_text)
    if not cleaned:
        return {
            "predicted_category": "Unknown",
            "confidence": 0.0,
            "top_categories": [],
        }

    vec = transform_text(cleaned, vectorizer)
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    # Sort classes by descending probability
    sorted_indices = np.argsort(probs)[::-1]
    top_class = classes[sorted_indices[0]]
    top_prob = float(probs[sorted_indices[0]])

    top_categories = [
        (classes[idx], round(float(probs[idx]), 4))
        for idx in sorted_indices[:top_k]
    ]

    return {
        "predicted_category": top_class,
        "confidence": round(top_prob, 4),
        "top_categories": top_categories,
    }
