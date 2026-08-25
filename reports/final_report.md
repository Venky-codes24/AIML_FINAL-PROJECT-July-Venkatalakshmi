# SmartHire — Final Project Report

## 1. Project Overview
SmartHire is an end-to-end Classical Machine Learning career matching platform for intelligent resume parsing, role classification, job recommendation, skill-gap analysis, and personalized career learning paths without external Generative AI or LLM dependencies.

## 2. Dataset Summary
- **Resumes Dataset**: 962 resumes labeled across 25 job domains (`UpdatedResumeDataSet.csv`).
- **Jobs Corpus**: 136,759 unified job listings combining Naukri and LinkedIn datasets (`jobs_clean.csv`).

## 3. Methodology & Performance
- **Supervised Role Classification**: Multi-class Logistic Regression with class weighting on TF-IDF features.
  - **Accuracy**: 99.48%
  - **Macro F1-Score**: 0.9945
  - **Weighted F1-Score**: 0.9949
  - Confusion matrix exported to `reports/figures/confusion_matrix.png`.
- **Unsupervised Job Recommendation**: Sparse TF-IDF + Cosine Similarity over 136,759 postings with near-duplicate suppression.
- **Skill Extraction & Gap Analysis**: Canonical dictionary with alias matching; computes matched/missing skills and frequency-ranked learning paths.
- **Explainable Fit Index**: Multi-factor scoring (Semantic + Skill Coverage + Role Relevance).
- **Salary Band Predictor**: Ridge regression on job text features and experience.

## 4. Web Portal & Tests
- Interactive Streamlit portal in `app/streamlit_app.py`.
- 18 automated unit tests in `tests/` passing with 100% success rate.
