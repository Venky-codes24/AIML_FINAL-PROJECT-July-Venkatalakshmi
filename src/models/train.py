"""Unified model training and artifact generation script.

Run:
    python -m src.models.train
"""

import sys
import time

from src import config
from src.features.match_features import save_skills_vocab
from src.models.classifier import train_classifier
from src.models.recommender import build_and_save_recommender


def main():
    print("=" * 60)
    print(" SmartHire — Model Training & Artifact Generation Pipeline ")
    print("=" * 60)

    start_time = time.time()

    # 1. Verify datasets exist
    if not config.RESUMES_CLEAN_CSV.exists() or not config.JOBS_CLEAN_CSV.exists():
        print("Preprocessed datasets not found in data/processed/.")
        print("Running data preprocessing first (python -m src.data.preprocess)...")
        from src.data.preprocess import main as preprocess_main
        preprocess_main()

    # 2. Train Resume Classifier
    print("\n[1/3] Training Resume Category Classifier (TF-IDF + Logistic Regression)...")
    clf_results = train_classifier(save=True)
    metrics = clf_results["metrics"]
    print(f"  [OK] Classifier trained on {len(clf_results['classes'])} categories.")
    print(f"  [OK] Test Accuracy:     {metrics['accuracy'] * 100:.2f}%")
    print(f"  [OK] Macro F1-Score:    {metrics['f1_macro']:.4f}")
    print(f"  [OK] Weighted F1-Score: {metrics['f1_weighted']:.4f}")

    # 3. Build & Index Job Recommender
    print("\n[2/3] Building Job Recommender Catalog & Feature Matrix...")
    rec_results = build_and_save_recommender(save=True)
    num_jobs = len(rec_results["job_metadata"])
    matrix_shape = rec_results["job_matrix"].shape
    print(f"  [OK] Indexed {num_jobs:,} job postings.")
    print(f"  [OK] TF-IDF Matrix Shape: {matrix_shape[0]:,} documents x {matrix_shape[1]:,} features.")

    # 4. Save Skills Vocabulary
    print("\n[3/3] Exporting Canonical Skills Vocabulary...")
    save_skills_vocab()
    print(f"  [OK] Skills vocabulary saved -> {config.SKILLS_VOCAB_PATH}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Pipeline completed successfully in {elapsed:.2f}s! ")
    print(f" All model artifacts are saved in '{config.MODELS_DIR}'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
