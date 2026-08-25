"""Paths and constants. Import from here instead of hard-coding filenames."""

from pathlib import Path

# ---- Folders ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ---- Raw dataset paths (what you downloaded into data/raw/) ----
RESUME_CSV = RAW_DIR / "UpdatedResumeDataSet.csv"
NAUKRI_CSV = RAW_DIR / "naukri_com-job_sample.csv"

# Optional LinkedIn dataset (arshkon/linkedin-job-postings), unzipped into
# data/raw/linkedin/. Merged in only if these files exist.
LINKEDIN_DIR = RAW_DIR / "linkedin"
LINKEDIN_POSTINGS_CSV = LINKEDIN_DIR / "postings.csv"
LINKEDIN_JOB_SKILLS_CSV = LINKEDIN_DIR / "jobs" / "job_skills.csv"
LINKEDIN_SKILLS_MAP_CSV = LINKEDIN_DIR / "mappings" / "skills.csv"

# ---- Outputs (created by preprocess.py) ----
JOB_CORPUS_CSV = INTERIM_DIR / "job_corpus.csv"          # merged jobs (readable)
RESUMES_CLEAN_CSV = PROCESSED_DIR / "resumes_clean.csv"  # cleaned resumes
JOBS_CLEAN_CSV = PROCESSED_DIR / "jobs_clean.csv"        # model-ready jobs (+ text col)

# ---- Common schema every job source is mapped to before merging ----
COMMON_JOB_COLUMNS = ["title", "company", "location", "skills", "description", "experience"]

# ---- Model Artifact Paths ----
RESUME_CLASSIFIER_PATH = MODELS_DIR / "resume_classifier.pkl"
RESUME_VECTORIZER_PATH = MODELS_DIR / "resume_vectorizer.pkl"
JOB_VECTORIZER_PATH = MODELS_DIR / "job_vectorizer.pkl"
JOB_MATRIX_PATH = MODELS_DIR / "job_matrix.pkl"
JOB_METADATA_PATH = MODELS_DIR / "job_metadata.pkl"
SKILLS_VOCAB_PATH = MODELS_DIR / "skills_vocab.json"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ---- Constants & Hyperparameters ----
RANDOM_STATE = 42          # seed for reproducible train/test splits
TOP_N = 5                  # default jobs returned by the recommender
MAX_FEATURES = 5000        # max features for TF-IDF
NGRAM_RANGE = (1, 2)       # n-gram range for TF-IDF
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]
