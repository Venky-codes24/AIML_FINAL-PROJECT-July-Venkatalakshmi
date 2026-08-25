"""Unsupervised content-based recommender: TF-IDF + cosine-similarity job ranking."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.data.preprocess import clean_text
from src.features.match_features import compute_skill_gap, extract_skills
from src.features.text_features import (
    build_tfidf_vectorizer,
    load_vectorizer,
    save_vectorizer,
    transform_text,
)


def build_and_save_recommender(
    jobs_df: Optional[pd.DataFrame] = None,
    save: bool = True,
    max_features: int = 10000,
) -> Dict[str, Any]:
    """Fit TF-IDF on job corpus, precalculate job feature matrix, and save artifacts.

    Args:
        jobs_df: Optional DataFrame with job listings and cleaned 'text' column.
        save: Whether to save artifacts to models/.
        max_features: Maximum vocabulary size for job vectorizer.

    Returns:
        Dict with vectorizer, job_matrix, and jobs DataFrame.
    """
    if jobs_df is None:
        jobs_df = pd.read_csv(config.JOBS_CLEAN_CSV, nrows=50000, engine="c", dtype=str, keep_default_na=False)

    jobs_df = jobs_df.fillna("").reset_index(drop=True)
    job_texts = jobs_df["text"].tolist()

    # Keep essential metadata columns (exclude redundant heavy 'text' column)
    metadata_cols = [col for col in config.COMMON_JOB_COLUMNS + ["source"] if col in jobs_df.columns]
    job_metadata = jobs_df[metadata_cols].copy()

    # Fit TF-IDF on job corpus text
    print("Fitting job TF-IDF vectorizer...")
    vectorizer = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
        min_df=5,
        sublinear_tf=True,
    )
    job_matrix = vectorizer.fit_transform(job_texts).astype(np.float32)

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_vectorizer(vectorizer, config.JOB_VECTORIZER_PATH)
        joblib.dump(job_matrix, config.JOB_MATRIX_PATH, compress=3)
        joblib.dump(job_metadata, config.JOB_METADATA_PATH, compress=3)
        print(f"Job recommender artifacts saved to {config.MODELS_DIR}")

    return {
        "vectorizer": vectorizer,
        "job_matrix": job_matrix,
        "job_metadata": job_metadata,
    }


def load_recommender() -> Tuple[Any, Any, pd.DataFrame]:
    """Load precalculated job vectorizer, job matrix, and job metadata table.

    Returns:
        Tuple of (job_vectorizer, job_matrix, job_metadata_df).

    Raises:
        FileNotFoundError: If artifacts are missing.
    """
    if (
        not config.JOB_VECTORIZER_PATH.exists()
        or not config.JOB_MATRIX_PATH.exists()
        or not config.JOB_METADATA_PATH.exists()
    ):
        raise FileNotFoundError(
            f"Job recommender artifacts not found in {config.MODELS_DIR}. "
            "Please run 'python -m src.models.train' to build them."
        )

    vectorizer = load_vectorizer(config.JOB_VECTORIZER_PATH)
    job_matrix = joblib.load(config.JOB_MATRIX_PATH)
    job_metadata = joblib.load(config.JOB_METADATA_PATH)
    return vectorizer, job_matrix, job_metadata


def recommend_jobs(
    resume_text: str,
    top_n: int = config.TOP_N,
    category_filter: Optional[str] = None,
    job_vectorizer: Optional[Any] = None,
    job_matrix: Optional[Any] = None,
    job_metadata: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Recommend top-N matching jobs for a candidate resume using cosine similarity.

    Args:
        resume_text: Extracted raw or cleaned resume text.
        top_n: Number of recommendations to return.
        category_filter: Optional string keyword to filter job titles/descriptions.
        job_vectorizer: Preloaded vectorizer (or None to load automatically).
        job_matrix: Preloaded sparse job matrix (or None to load automatically).
        job_metadata: Preloaded job metadata DataFrame (or None to load automatically).

    Returns:
        List of recommendation dictionaries sorted by Match Score descending.
    """
    if job_vectorizer is None or job_matrix is None or job_metadata is None:
        job_vectorizer, job_matrix, job_metadata = load_recommender()

    cleaned_resume = clean_text(resume_text)
    if not cleaned_resume:
        return []

    # Transform resume text into TF-IDF vector
    resume_vec = transform_text(cleaned_resume, job_vectorizer)

    # Compute cosine similarity between resume and all job listings in catalog
    similarities = cosine_similarity(resume_vec, job_matrix)[0]

    # Candidate skills extracted from resume
    resume_skills = extract_skills(resume_text)

    # Fast Top Candidate Ranking: Sort top matching candidates first
    # This avoids expensive full-table regex/string scans across 136k items
    candidate_pool_size = min(len(similarities), max(500, top_n * 50))
    top_candidate_indices = np.argpartition(similarities, -candidate_pool_size)[-candidate_pool_size:]
    top_candidate_indices = top_candidate_indices[np.argsort(similarities[top_candidate_indices])[::-1]]

    # Apply category filter and deduplication to ensure diverse, high-value recommendations
    selected_indices: List[int] = []
    seen_signatures = set()

    for idx in top_candidate_indices:
        row = job_metadata.iloc[idx]
        title_str = str(row.get("title", "")).strip()
        comp_str = str(row.get("company", "")).strip()
        desc_str = str(row.get("description", "")).strip()

        # Category filter check
        if category_filter and category_filter.strip().lower() != "all":
            cat_lower = category_filter.strip().lower()
            if cat_lower not in title_str.lower() and cat_lower not in desc_str.lower():
                continue

        # Deduplication signature: normalized title + company or first 100 chars of description
        norm_title = "".join(c for c in title_str.lower() if c.isalnum())
        norm_comp = "".join(c for c in comp_str.lower() if c.isalnum())
        desc_snippet_sig = desc_str[:100].lower().strip()
        sig = (norm_title, norm_comp, desc_snippet_sig)

        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        selected_indices.append(idx)
        if len(selected_indices) >= top_n:
            break

    recommendations: List[Dict[str, Any]] = []
    for rank, idx in enumerate(selected_indices, start=1):
        row = job_metadata.iloc[idx]
        sim = float(similarities[idx])

        # Normalize cosine similarity to an intuitive 0-100 Match Score
        # TF-IDF cosine similarity in text retrieval typically peaks between 0.15 - 0.70;
        # Scaled smoothly using min-max linear scaling for user clarity
        raw_pct = min(100.0, max(0.0, sim * 100))
        match_score = int(round(min(98.0, max(15.0, raw_pct * 1.6))))

        # Skill gap analysis for this job
        job_skills_list = row.get("skills_list") or extract_skills(
            f"{row.get('title', '')} {row.get('skills', '')} {row.get('description', '')}"
        )
        gap_info = compute_skill_gap(resume_skills, job_skills_list)

        title = str(row.get("title", "Untitled Job")).strip() or "Untitled Job"
        company = str(row.get("company", "Company Confidential")).strip() or "Company Confidential"
        location = str(row.get("location", "Not Specified")).strip() or "Not Specified"
        description = str(row.get("description", "")).strip()
        experience = str(row.get("experience", "Not Specified")).strip() or "Not Specified"
        source = str(row.get("source", "job_board")).strip()

        # Clean description snippet for UI display
        snippet = description[:350] + "..." if len(description) > 350 else description

        recommendations.append({
            "rank": rank,
            "job_id": int(idx),
            "title": title,
            "company": company,
            "location": location,
            "description": description,
            "description_snippet": snippet,
            "experience": experience,
            "source": source,
            "raw_similarity": round(sim, 4),
            "match_score": match_score,
            "match_score_display": f"Match Score: {match_score}/100",
            "skills_list": job_skills_list,
            "matched_skills": gap_info["matched_skills"],
            "missing_skills": gap_info["missing_skills"],
            "skill_overlap_ratio": gap_info["overlap_ratio"],
        })

    return recommendations
