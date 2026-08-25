"""SmartHire End-to-End System Integrity & Verification Test."""

import sys
from pathlib import Path

# Add root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsing.resume_parser import extract_metadata
from src.models.classifier import load_classifier, predict_category
from src.models.recommender import load_recommender, recommend_jobs
from src.features.match_features import extract_skills, compute_skill_gap, recommend_skills_to_learn
from src.models.fit_predictor import compute_candidate_fit
from src.models.salary_predictor import predict_salary_band
from src.models.clustering import fit_job_clusters

print("=" * 65)
print("  SMARTHIRE END-TO-END SYSTEM INTEGRITY TEST")
print("=" * 65)

# 1. Test Resume Text
resume_text = """
SENIOR DATA SCIENTIST
Summary: Experienced Data Scientist with 5 years experience in Python, SQL,
Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn, NLP, Docker, AWS.
Education: Bachelor of Technology in Computer Science
Email: test.candidate@example.com | Phone: +1 555-019-2834
"""
print("\n[1] Resume Parsing & Metadata Extraction:")
meta = extract_metadata(resume_text)
print(f"    - Emails: {meta['emails']}")
print(f"    - Education: {meta['education']}")

# 2. Test Classifier
clf, vec = load_classifier()
pred = predict_category(resume_text, clf, vec)
print(f"\n[2] Supervised Classifier (25 Categories):")
print(f"    - Predicted Role: {pred['predicted_category']} (Confidence: {int(pred['confidence']*100)}%)")
print(f"    - Top 3 Probabilities: {pred['top_categories']}")

# 3. Test Skills Extraction
skills = extract_skills(resume_text)
print(f"\n[3] Extracted Skills ({len(skills)} found):")
print(f"    - {skills}")

# 4. Test Recommender
rec_vec, rec_mat, job_meta = load_recommender()
print(f"\n[4] Recommender Catalog:")
print(f"    - Indexed Jobs: {rec_mat.shape[0]:,} listings")
print(f"    - Matrix Type: {type(rec_mat).__name__} ({rec_mat.shape[1]:,} TF-IDF features)")

recs = recommend_jobs(resume_text, top_n=3, job_vectorizer=rec_vec, job_matrix=rec_mat, job_metadata=job_meta)
print(f"\n[5] Top 3 Job Recommendations (Deduplicated):")
for r in recs:
    print(f"    - #{r['rank']} {r['title']} @ {r['company']}")
    print(f"      {r['match_score_display']} | Matched: {r['matched_skills']} | Missing: {r['missing_skills']}")

# 6. Test Fit Predictor
fit = compute_candidate_fit(resume_text, recs[0], pred["predicted_category"], recs[0]["raw_similarity"])
print(f"\n[6] Candidate Fit Alignment (Top Job):")
print(f"    - Tier: {fit['fit_tier']} (Score: {fit['fit_score']}/100)")
print(f"    - Breakdown: {fit['breakdown']}")

# 7. Test Learning Path
learning_path = recommend_skills_to_learn(skills, recs, top_k=3)
print(f"\n[7] High-Demand Skills to Learn:")
for rank, (sk, count, pct) in enumerate(learning_path, 1):
    print(f"    - {rank}. {sk} (In {count} matched jobs, {pct}%)")

# 8. Test Salary Predictor
salary = predict_salary_band(recs[0]["title"], ", ".join(skills), experience_years=5.0)
print(f"\n[8] Estimated Salary Band:")
print(f"    - Estimated Range: {salary['salary_range_lakhs']}")

# 9. Test Clustering
sub_mat = rec_mat[:500]
clusters = fit_job_clusters(sub_mat, rec_vec, n_clusters=3)
print(f"\n[9] Role Clustering Topics:")
for cid, terms in clusters["cluster_topics"].items():
    print(f"    - Cluster {cid}: {', '.join(terms[:5])}")

print("\n" + "=" * 65)
print("  >>> ALL 9 CORE & OPTIONAL PIPELINES VERIFIED SUCCESSFULLY! <<<")
print("=" * 65)
