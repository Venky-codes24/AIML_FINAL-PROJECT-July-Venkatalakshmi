"""Unsupervised job and resume clustering via KMeans and NMF/LDA topic extraction."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score

from src import config


def fit_job_clusters(
    job_matrix,
    vectorizer,
    n_clusters: int = 10,
    random_state: int = config.RANDOM_STATE,
) -> Dict[str, Any]:
    """Cluster job postings using KMeans and extract representative terms per cluster."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        n_init=5,
        random_state=random_state,
    )
    cluster_labels = kmeans.fit_predict(job_matrix)

    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    cluster_topics: Dict[int, List[str]] = {}
    for cluster_id in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[cluster_id, :10]]
        cluster_topics[cluster_id] = top_terms

    return {
        "model": kmeans,
        "labels": cluster_labels,
        "cluster_topics": cluster_topics,
        "n_clusters": n_clusters,
    }


def evaluate_elbow_silhouette(
    job_matrix,
    k_range: List[int] = [3, 5, 8, 10],
    sample_size: int = 2000,
    random_state: int = config.RANDOM_STATE,
) -> Dict[str, Any]:
    """Calculate inertia (Elbow method) and Silhouette scores across multiple k values on a matrix sample."""
    sub_matrix = job_matrix[:sample_size] if job_matrix.shape[0] > sample_size else job_matrix
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=3, random_state=random_state)
        labels = km.fit_predict(sub_matrix)
        inertias.append(float(km.inertia_))
        try:
            sil = float(silhouette_score(sub_matrix, labels, sample_size=min(1000, len(labels))))
            silhouette_scores.append(round(sil, 4))
        except Exception:
            silhouette_scores.append(0.0)

    return {
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }


def fit_nmf_topics(
    job_matrix,
    vectorizer,
    n_topics: int = 8,
    top_words: int = 8,
    random_state: int = config.RANDOM_STATE,
) -> Dict[int, List[str]]:
    """Extract in-demand domain and skill themes using Non-Negative Matrix Factorization (NMF)."""
    nmf = NMF(n_components=n_topics, random_state=random_state, max_iter=200)
    nmf.fit(job_matrix)

    terms = vectorizer.get_feature_names_out()
    topics: Dict[int, List[str]] = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[:-top_words - 1:-1]
        topics[topic_idx] = [terms[i] for i in top_indices]

    return topics
