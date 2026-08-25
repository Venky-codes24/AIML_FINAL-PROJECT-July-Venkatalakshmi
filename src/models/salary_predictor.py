"""Optional Supervised Learning: Salary Band Regression on Job Features.

Extracts numeric salary targets (e.g. INR per annum) from job listings and trains a
regression model (Ridge / Random Forest) on job features and experience to predict salary range.
Evaluated with MAE and RMSE.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src import config
from src.data.preprocess import clean_text


def parse_salary_inr(payrate_str: str) -> Optional[float]:
    """Parse salary range string (e.g. '3,00,000 - 6,50,000 P.A') into average annual salary in INR (Lakhs)."""
    if not isinstance(payrate_str, str) or "not disclosed" in payrate_str.lower():
        return None
    # Find all numeric patterns with commas
    nums = re.findall(r"[\d,]+", payrate_str)
    values = []
    for n in nums:
        cleaned_n = n.replace(",", "").strip()
        if cleaned_n.isdigit() and len(cleaned_n) >= 4:
            val = float(cleaned_n)
            # Filter out reasonable annual salary bounds in INR (50k to 5Cr)
            if 50_000 <= val <= 50_000_000:
                values.append(val / 100_000.0)  # Convert to Lakhs INR
    if len(values) >= 2:
        return float(np.mean(values[:2]))
    elif len(values) == 1:
        return float(values[0])
    return None


def parse_experience_years(exp_str: str) -> float:
    """Parse experience string (e.g. '4 - 8 yrs') into average years of experience."""
    if not isinstance(exp_str, str):
        return 0.0
    nums = re.findall(r"\d+", exp_str)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    return 0.0


def train_salary_predictor(
    naukri_csv_path: Optional[Path] = None,
    save: bool = False,
) -> Dict[str, Any]:
    """Train a Ridge regression model to predict salary band from job title, skills, and experience."""
    csv_path = naukri_csv_path or config.NAUKRI_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Naukri dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, usecols=["jobtitle", "skills", "experience", "payrate"]).dropna()
    df["salary_lakhs"] = df["payrate"].apply(parse_salary_inr)
    df["exp_years"] = df["experience"].apply(parse_experience_years)
    df["text"] = (df["jobtitle"].fillna("") + " " + df["skills"].fillna("")).apply(clean_text)

    # Filter out rows with valid salary labels
    valid_df = df.dropna(subset=["salary_lakhs"]).copy()
    valid_df = valid_df[(valid_df["salary_lakhs"] >= 1.0) & (valid_df["salary_lakhs"] <= 80.0)]

    if len(valid_df) < 50:
        return {"error": "Insufficient valid salary records."}

    # Vectorize text features
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 1), min_df=2)
    X_text = vectorizer.fit_transform(valid_df["text"])
    X_exp = valid_df[["exp_years"]].values

    from scipy.sparse import hstack
    X = hstack([X_text, X_exp])
    y = valid_df["salary_lakhs"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "num_samples": len(valid_df),
        "mae_lakhs": round(mae, 2),
        "rmse_lakhs": round(rmse, 2),
        "r2_score": round(r2, 4),
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
    }


def predict_salary_band(
    job_title: str,
    skills: str,
    experience_years: float = 3.0,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Estimate expected annual salary range in Lakhs INR."""
    if model is None or vectorizer is None:
        train_res = train_salary_predictor()
        model = train_res["model"]
        vectorizer = train_res["vectorizer"]

    from scipy.sparse import hstack
    text = clean_text(f"{job_title} {skills}")
    x_text = vectorizer.transform([text])
    x_exp = np.array([[float(experience_years)]])
    x_input = hstack([x_text, x_exp])

    pred = float(model.predict(x_input)[0])
    pred = max(1.5, round(pred, 1))

    low = max(1.0, round(pred * 0.8, 1))
    high = round(pred * 1.25, 1)

    return {
        "estimated_salary_lakhs": pred,
        "salary_range_lakhs": f"₹{low}L - ₹{high}L P.A.",
        "currency": "INR",
    }
