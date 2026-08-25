"""Unit tests for supervised resume category classifier."""

import numpy as np
import pandas as pd
import pytest

from src.models.classifier import load_classifier, predict_category, train_classifier


@pytest.fixture
def mock_resumes_df():
    data = {
        "Category": [
            "Data Science", "Data Science", "Data Science", "Data Science",
            "Java Developer", "Java Developer", "Java Developer", "Java Developer",
            "DevOps", "DevOps", "DevOps", "DevOps",
        ],
        "text": [
            "python machine learning deep learning tensorflow data science",
            "pandas numpy statistics regression scikit learn python",
            "natural language processing transformers ml models python sql",
            "computer vision opencv deep neural networks python",
            "java spring boot microservices hibernate rest api maven",
            "core java multi-threading spring framework postgresql backend",
            "java j2ee jpa rest web services junit spring boot",
            "backend java developer spring cloud distributed systems",
            "aws terraform docker kubernetes ci cd linux jenkins",
            "cloud infrastructure kubernetes helm docker ansible devops",
            "ci cd automation pipeline monitoring prometheus grafana linux",
            "devops engineer aws cloudformation bash python gitlab",
        ],
    }
    return pd.DataFrame(data)


def test_train_classifier_on_mock(mock_resumes_df):
    results = train_classifier(resumes_df=mock_resumes_df, save=False, test_size=0.25)
    assert "model" in results
    assert "metrics" in results
    assert len(results["classes"]) == 3
    assert results["metrics"]["accuracy"] >= 0.5


def test_predict_category_inference():
    # Test using saved model
    resume = "Experienced Data Scientist skilled in Python, Scikit-learn, TensorFlow, and statistical analysis."
    pred = predict_category(resume)

    assert "predicted_category" in pred
    assert "confidence" in pred
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["top_categories"]) > 0


def test_predict_category_empty_string():
    pred = predict_category("")
    assert pred["predicted_category"] == "Unknown"
    assert pred["confidence"] == 0.0


def test_salary_band_predictor():
    from src.models.salary_predictor import parse_salary_inr, parse_experience_years, predict_salary_band

    assert parse_salary_inr("3,00,000 - 6,50,000 P.A") == 4.75
    assert parse_salary_inr("Not Disclosed by Recruiter") is None
    assert parse_experience_years("3 - 5 yrs") == 4.0

    salary_est = predict_salary_band("Python Developer", "Django FastAPI SQL", experience_years=3.0)
    assert "estimated_salary_lakhs" in salary_est
    assert "salary_range_lakhs" in salary_est
    assert salary_est["estimated_salary_lakhs"] > 0
