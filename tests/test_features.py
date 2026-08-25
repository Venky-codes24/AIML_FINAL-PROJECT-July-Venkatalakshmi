"""Unit tests for text preprocessing, TF-IDF feature extraction, and skill-gap logic."""

import pytest
from src.data.preprocess import clean_text
from src.features.match_features import (
    compute_skill_gap,
    extract_skills,
    recommend_skills_to_learn,
)
from src.features.text_features import (
    build_tfidf_vectorizer,
    fit_transform_text,
    transform_text,
)


def test_clean_text():
    raw = "Senior Software Engineer @ Google! URL: https://example.com with C++ & Python 3.9."
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "google" in cleaned
    assert "c++" in cleaned
    assert "python" in cleaned
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_skill_extraction_canonical_and_aliases():
    text = "Proficient in Python, Scikit-learn, sklearn, C++, React.js, Node.js, AWS, and Docker."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "Scikit-learn" in skills
    assert "C++" in skills
    assert "React" in skills
    assert "Node.js" in skills
    assert "AWS" in skills
    assert "Docker" in skills


def test_skill_gap_computation():
    resume_skills = ["Python", "SQL", "Machine Learning"]
    job_skills = ["Python", "SQL", "Docker", "AWS", "Machine Learning"]
    gap = compute_skill_gap(resume_skills, job_skills)

    assert set(gap["matched_skills"]) == {"Python", "SQL", "Machine Learning"}
    assert set(gap["missing_skills"]) == {"Docker", "AWS"}
    assert gap["overlap_ratio"] == 0.6
    assert gap["match_percentage"] == 60


def test_recommend_skills_to_learn():
    resume_skills = ["Python", "SQL"]
    mock_jobs = [
        {"title": "Job 1", "skills_list": ["Python", "Docker", "AWS"]},
        {"title": "Job 2", "skills_list": ["Python", "Docker", "Kubernetes"]},
        {"title": "Job 3", "skills_list": ["Python", "AWS", "Terraform"]},
    ]
    recommendations = recommend_skills_to_learn(resume_skills, mock_jobs, top_k=3)
    skills = [rec[0] for rec in recommendations]
    assert "Docker" in skills
    assert "AWS" in skills
    assert "Python" not in skills  # Candidate already knows Python


def test_tfidf_vectorization():
    corpus = [
        "python machine learning data science",
        "java spring boot microservices",
        "react node javascript fullstack",
    ]
    vec = build_tfidf_vectorizer(max_features=50, ngram_range=(1, 1), min_df=1)
    matrix, fitted_vec = fit_transform_text(corpus, vectorizer=vec)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 5

    test_transform = transform_text(["python data science"], fitted_vec)
    assert test_transform.shape[0] == 1
    assert test_transform.shape[1] == matrix.shape[1]


def test_evaluation_metrics():
    from src.evaluate import (
        evaluate_classifier,
        evaluate_clustering,
        evaluate_precision_at_k,
        evaluate_regression,
    )

    # 1. Classification metrics
    clf_res = evaluate_classifier(["Data Science", "Java"], ["Data Science", "Java"], save_fig=False)
    assert clf_res["accuracy"] == 1.0
    assert clf_res["f1_macro"] == 1.0

    # 2. Precision@K
    mock_recs = [
        {"title": "Senior Data Scientist", "description": "Python ML"},
        {"title": "Machine Learning Engineer", "description": "PyTorch"},
        {"title": "Frontend React Developer", "description": "JavaScript"},
    ]
    p_at_2 = evaluate_precision_at_k(mock_recs, "Data Scientist", k=2)
    assert p_at_2 == 0.5

    # 3. Regression metrics (MAE, RMSE, R2)
    reg_res = evaluate_regression([10.0, 20.0, 30.0], [10.0, 22.0, 28.0])
    assert reg_res["mae"] == 1.3333
    assert reg_res["rmse"] > 0
    assert reg_res["r2_score"] > 0.9
