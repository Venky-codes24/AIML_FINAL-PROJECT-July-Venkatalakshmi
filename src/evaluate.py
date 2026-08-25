"""Shared evaluation metrics and reporting utilities for all models.

Supported Evaluation Metrics:
1. Classification: Accuracy, Precision, Recall, F1-Score (macro/weighted), ROC-AUC, Confusion Matrix.
2. Clustering: Silhouette Score, Inertia (Elbow Method).
3. Recommender: Precision@K, Catalog Coverage, Qualitative alignment.
4. Regression: MAE, RMSE, R².
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from src import config


# ----------------------------------------------------------------------------
# 1. Classification Evaluation (Resume Classifier & Fit Predictor)
# ----------------------------------------------------------------------------
def evaluate_classifier(
    y_true: List[str],
    y_pred: List[str],
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    save_fig: bool = True,
    fig_name: str = "confusion_matrix.png",
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics and optionally save a confusion matrix plot."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    roc_auc = None
    if y_prob is not None and len(labels) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if save_fig:
        fig_dir = config.PROJECT_ROOT / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / fig_name

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            cm,
            annot=len(labels) <= 15,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Resume Category Classifier — Confusion Matrix", fontsize=14, pad=15)
        plt.xlabel("Predicted Category", fontsize=12)
        plt.ylabel("True Category", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "classification_report": report,
    }
    if roc_auc is not None:
        metrics["roc_auc_ovr"] = round(roc_auc, 4)

    return metrics


# ----------------------------------------------------------------------------
# 2. Clustering Evaluation (KMeans & Topics)
# ----------------------------------------------------------------------------
def evaluate_clustering(
    matrix,
    labels: np.ndarray,
    kmeans_model: Optional[Any] = None,
    sample_size: int = 1500,
) -> Dict[str, float]:
    """Calculate Silhouette Score and Inertia (for Elbow Method)."""
    results: Dict[str, float] = {}

    if kmeans_model is not None and hasattr(kmeans_model, "inertia_"):
        results["inertia"] = round(float(kmeans_model.inertia_), 2)

    try:
        sub_size = min(sample_size, len(labels))
        sub_matrix = matrix[:sub_size] if matrix.shape[0] > sub_size else matrix
        sub_labels = labels[:sub_size]
        sil = silhouette_score(sub_matrix, sub_labels)
        results["silhouette_score"] = round(float(sil), 4)
    except Exception:
        results["silhouette_score"] = 0.0

    return results


# ----------------------------------------------------------------------------
# 3. Recommender Evaluation (Precision@K & Coverage)
# ----------------------------------------------------------------------------
def evaluate_precision_at_k(
    recommendations: List[Dict[str, Any]],
    target_category_or_keywords: Union[str, List[str]],
    k: int = 5,
) -> float:
    """Calculate Precision@K: fraction of top-K recommendations matching target role/keywords."""
    if not recommendations or k <= 0:
        return 0.0

    if isinstance(target_category_or_keywords, str):
        target_keywords = [target_category_or_keywords.lower()]
    else:
        target_keywords = [kw.lower() for kw in target_category_or_keywords]

    top_k_recs = recommendations[:k]
    relevant_count = 0

    for rec in top_k_recs:
        text = f"{rec.get('title', '')} {rec.get('description', '')}".lower()
        if any(kw in text for kw in target_keywords):
            relevant_count += 1

    return round(float(relevant_count / len(top_k_recs)), 4)


def evaluate_recommender_coverage(
    recommendations_list: List[List[Dict[str, Any]]],
    total_catalog_size: int,
) -> Dict[str, float]:
    """Compute catalog coverage and average score distribution for recommendations."""
    recommended_job_ids = set()
    scores = []

    for recs in recommendations_list:
        for r in recs:
            if "job_id" in r:
                recommended_job_ids.add(r["job_id"])
            elif "title" in r and "company" in r:
                recommended_job_ids.add((r["title"], r["company"]))
            if "match_score" in r:
                scores.append(r["match_score"])

    coverage = len(recommended_job_ids) / max(1, total_catalog_size)
    avg_score = float(np.mean(scores)) if scores else 0.0

    return {
        "catalog_coverage": round(coverage, 4),
        "unique_recommended_items": len(recommended_job_ids),
        "avg_match_score": round(avg_score, 2),
    }


# ----------------------------------------------------------------------------
# 4. Regression Evaluation (Salary Band Predictor)
# ----------------------------------------------------------------------------
def evaluate_regression(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R² for regression models."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
    }
