"""Explainable candidate-job fit score engine.

Note on Methodology:
Public job post and resume datasets do not contain ground-truth outcome labels
(e.g., 'hired' vs 'rejected'). In accordance with ethical ML standards, this module
does NOT train a fabricated supervised classifier or claim to predict an ungrounded
'probability of hiring'. Instead, it computes an explainable multi-factor fit index
based on semantic alignment, skill match percentage, and role relevance.
"""

from typing import Any, Dict, List, Optional

from src.data.preprocess import clean_text
from src.features.match_features import compute_skill_gap, extract_skills


def compute_candidate_fit(
    resume_text: str,
    job_info: Dict[str, Any],
    predicted_category: Optional[str] = None,
    semantic_similarity: float = 0.0,
) -> Dict[str, Any]:
    """Compute an explainable, multi-factor fit score between a candidate resume and a specific job.

    Components:
        1. Semantic Text Alignment (40% weight): Overall contextual similarity.
        2. Skill Gap Overlap (40% weight): Ratio of job skills possessed by candidate.
        3. Role / Keyword Relevance (20% weight): Alignment with target role category.

    Args:
        resume_text: Candidate resume raw or cleaned text.
        job_info: Dictionary containing job title, description, skills, and experience.
        predicted_category: Optional predicted category from classifier.
        semantic_similarity: Optional precomputed cosine similarity (0.0 to 1.0).

    Returns:
        Dict containing composite score (0-100), breakdown scores, fit tier, and explanation.
    """
    resume_skills = extract_skills(resume_text)
    job_skills = job_info.get("skills_list") or extract_skills(
        f"{job_info.get('title', '')} {job_info.get('skills', '')} {job_info.get('description', '')}"
    )

    # 1. Skill overlap score (0 to 100)
    gap = compute_skill_gap(resume_skills, job_skills)
    skill_score = float(gap["match_percentage"])

    # 2. Semantic alignment score (0 to 100)
    semantic_score = min(100.0, max(0.0, semantic_similarity * 150)) if semantic_similarity > 0 else 50.0

    # 3. Role relevance score (0 to 100)
    role_score = 50.0
    job_title = str(job_info.get("title", "")).lower()
    if predicted_category:
        cat_tokens = clean_text(predicted_category).split()
        if any(token in job_title for token in cat_tokens if len(token) > 2):
            role_score = 95.0
        elif any(token in clean_text(job_info.get("description", "")) for token in cat_tokens if len(token) > 2):
            role_score = 75.0
        else:
            role_score = 40.0

    # Composite weighted score
    composite = (0.40 * skill_score) + (0.40 * semantic_score) + (0.20 * role_score)
    final_score = int(round(min(98.0, max(15.0, composite))))

    if final_score >= 80:
        fit_tier = "Strong Match"
        summary = "Candidate demonstrates high skill coverage and strong contextual alignment with the role."
    elif final_score >= 65:
        fit_tier = "Good Match"
        summary = "Candidate meets core competencies with minor skill gaps that can be addressed on the job."
    elif final_score >= 50:
        fit_tier = "Moderate Match"
        summary = "Candidate has transferable foundational skills but would require upskilling in several key areas."
    else:
        fit_tier = "Developmental Match"
        summary = "Significant divergence between candidate's current background and target job requirements."

    return {
        "fit_score": final_score,
        "fit_tier": fit_tier,
        "summary": summary,
        "breakdown": {
            "skill_match_score": round(skill_score, 1),
            "semantic_similarity_score": round(semantic_score, 1),
            "role_relevance_score": round(role_score, 1),
        },
        "matched_skills": gap["matched_skills"],
        "missing_skills": gap["missing_skills"],
    }
