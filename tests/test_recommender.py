"""Unit tests for unsupervised job recommender and explainable fit predictor."""

import pandas as pd
import pytest

from src.models.fit_predictor import compute_candidate_fit
from src.models.recommender import load_recommender, recommend_jobs


def test_recommend_jobs_inference():
    resume_text = "Senior Python Developer with 5 years experience in Django, FastAPI, SQL, Docker, AWS"
    recs = recommend_jobs(resume_text, top_n=3)

    assert len(recs) == 3
    for r in recs:
        assert "title" in r
        assert "company" in r
        assert "match_score" in r
        assert 0 <= r["match_score"] <= 100
        assert "matched_skills" in r
        assert "missing_skills" in r


def test_recommend_jobs_category_filter():
    resume_text = "Full Stack Engineer with React, Node.js, and Java"
    recs = recommend_jobs(resume_text, top_n=3, category_filter="Java")
    assert len(recs) > 0


def test_compute_candidate_fit():
    resume = "Data Scientist with Python, SQL, Machine Learning, and Scikit-learn."
    job = {
        "title": "Machine Learning Engineer",
        "skills": "Python, SQL, Machine Learning, TensorFlow, AWS",
        "description": "Seeking ML engineer with strong Python and SQL background.",
    }
    fit = compute_candidate_fit(
        resume_text=resume,
        job_info=job,
        predicted_category="Data Science",
        semantic_similarity=0.55,
    )

    assert "fit_score" in fit
    assert "fit_tier" in fit
    assert 0 <= fit["fit_score"] <= 100
    assert "Python" in fit["matched_skills"]
    assert "TensorFlow" in fit["missing_skills"]
    assert "breakdown" in fit


def test_clustering_and_topics():
    from src.models.clustering import fit_job_clusters, evaluate_elbow_silhouette, fit_nmf_topics
    from src.features.text_features import build_tfidf_vectorizer, fit_transform_text

    docs = [
        "python machine learning deep learning",
        "data science python pandas statistics",
        "java spring boot microservices backend",
        "react javascript fullstack frontend web",
        "aws devops docker kubernetes ci cd",
    ]
    vec = build_tfidf_vectorizer(max_features=50, min_df=1)
    matrix, fitted_vec = fit_transform_text(docs, vectorizer=vec)

    cluster_res = fit_job_clusters(matrix, fitted_vec, n_clusters=2)
    assert "model" in cluster_res
    assert len(cluster_res["cluster_topics"]) == 2

    eval_res = evaluate_elbow_silhouette(matrix, k_range=[2, 3])
    assert len(eval_res["inertias"]) == 2

    nmf_topics = fit_nmf_topics(matrix, fitted_vec, n_topics=2)
    assert len(nmf_topics) == 2
