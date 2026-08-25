# SmartHire — AI-Powered Resume & Career Matching Engine

SmartHire is a classical machine learning application and AI/ML prototype (no external LLMs or black-box APIs) for resume parsing, supervised role categorization, cosine-similarity job matching, skill-gap analysis, and career learning path recommendations.

---

## 🌟 Key Features

1. **Multi-Format Resume Parser**: Supports text extraction from **PDF**, **DOCX**, and **TXT** files with automatic normalization and verifiable metadata extraction (education, experience, contact details).
2. **Supervised Resume Category Classifier**: Predicts candidate role category across 25 target specializations using **TF-IDF + Logistic Regression** with confidence scoring and probability distributions.
3. **Unsupervised Job Recommender**: Ranks and matches the candidate's resume against **136,000+ job listings** using memory-efficient **Sparse TF-IDF + Cosine Similarity** with active duplicate suppression.
4. **Transparent Match Scoring**: Evaluates candidate-to-job similarity scored as `Match Score: X/100` (clearly distinguished from hiring probabilities).
5. **Automated Skill Extraction**: Identifies technical and professional skills using canonical normalization and alias matching (e.g. `scikit-learn`, `sklearn`, `React.js`, `C++`, `Docker`, `AWS`).
6. **Skill-Gap Analysis**: Performs exact matching against required job skills, categorizing them into **Matched Skills** and **Missing Skills**.
7. **Prioritized Career Learning Path**: Identifies and ranks missing skills by demand frequency across recommended job matches.
8. **Explainable Multi-Factor Alignment**: Combines semantic text similarity, skill overlap, and role relevance into a transparent breakdown without synthetic labels.
9. **Interactive Streamlit Web Portal**: Clean, responsive UI with distinct tabs for uploading resumes or trying demo sample documents.

---

## ⚠️ Match Score Disclaimer

> **Important:** The **Match Score** displayed by the recommender represents text and skill similarity/relevance between the uploaded resume and job postings. It is **not** a prediction of hiring probability, selection odds, or an assessment of candidate suitability beyond textual alignment.

---

## 🏗️ Architecture

```
User Resume (PDF / DOCX / TXT)
             │
             ▼
[ Resume Parser & Cleaner ] ──────────────► [ Extract Metadata & Skills ]
             │
             ├───► [ TF-IDF Vectorizer ] ───► [ Logistic Regression Classifier ]
             │                                          │
             │                                          ▼
             │                                   Predicted Role & Confidence
             │
             └───► [ Sparse Job TF-IDF Matrix ] ───► [ Cosine Similarity Engine ]
                                                           │
                                                           ▼
                                                Deduplicated Top-N Matches
                                                           │
                                                           ├───► [ Skill Gap Analysis ]
                                                           │      (Matched vs Missing)
                                                           │
                                                           └───► [ Recommended Skills ]
                                                                  (High-Demand Learning Path)
```

---

## 📂 Project Structure

```
Smart_hire_AI_ML_June/
├── app/
│   └── streamlit_app.py              # Streamlit web portal
├── data/
│   ├── raw/                          # Original raw downloads (git-ignored)
│   ├── interim/                      # Merged job corpus (job_corpus.csv)
│   └── processed/                    # Final model-ready datasets
│       ├── jobs_clean.csv            # 136,759 cleaned job postings
│       └── resumes_clean.csv         # 962 cleaned resumes (25 categories)
├── models/                           # Saved model artifacts (git-ignored)
│   ├── resume_classifier.pkl         # Trained Logistic Regression classifier
│   ├── resume_vectorizer.pkl         # Fitted TF-IDF resume vectorizer
│   ├── job_vectorizer.pkl            # Fitted TF-IDF job vectorizer
│   ├── job_matrix.pkl                # Precalculated sparse CSR job matrix
│   ├── job_metadata.pkl              # Fast-lookup job metadata table
│   └── skills_vocab.json             # Canonical skills mapping dictionary
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_resume_classifier.ipynb    # Supervised classification experiments
│   ├── 03_recommender.ipynb          # Recommender experiments
│   ├── 04_clustering_topics.ipynb    # KMeans clustering and topic discovery
│   └── 05_fit_predictor.ipynb        # Explainable candidate fit index
├── reports/
│   └── figures/                      # Evaluation figures & confusion matrices
├── src/
│   ├── config.py                     # Centralized paths and settings
│   ├── evaluate.py                   # Classification & recommender metrics
│   ├── data/
│   │   ├── load_data.py              # Safe CSV loaders
│   │   └── preprocess.py             # Preprocessing & deduplication
│   ├── features/
│   │   ├── text_features.py          # TF-IDF feature helpers
│   │   └── match_features.py         # Skill extraction, gaps & recommendations
│   ├── models/
│   │   ├── classifier.py             # Classifier training & prediction
│   │   ├── recommender.py            # Cosine similarity matching & deduplication
│   │   ├── clustering.py             # KMeans topic clustering
│   │   ├── fit_predictor.py          # Explainable multi-factor fit scorer
│   │   └── train.py                  # Unified CLI training script
│   └── parsing/
│       └── resume_parser.py          # PDF, DOCX, TXT parser & metadata extractor
├── tests/
│   ├── test_features.py              # Tests for preprocessing & TF-IDF
│   ├── test_parser.py                # Tests for PDF, DOCX, TXT parsing
│   ├── test_classifier.py            # Tests for category classification
│   └── test_recommender.py           # Tests for job matching & fit index
├── download_data.py                  # Automated Kaggle dataset downloader
├── requirements.txt                  # Pinned Python dependencies
├── pytest.ini                        # Pytest configuration
└── README.md                         # Documentation
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.11 and 3.12)
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Anjali05R/Smart_hire_AI_ML_June.git
cd Smart_hire_AI_ML_June

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Datasets
```bash
python download_data.py
```

### 4. Preprocess Data
```bash
python -m src.data.preprocess
```

### 5. Train Models & Build Sparse Indices
Train the resume classifier and build the sparse recommender index (run once):
```bash
python -m src.models.train
```

### 6. Launch the Web Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧪 Running Unit Tests

Run the complete test suite with pytest:
```bash
python -m pytest -v
```

All 16 test cases validate:
- Text normalization and token sanitization
- PDF, DOCX, and TXT parsing with corrupted/empty file handling
- Skill extraction and alias canonicalization
- Skill-gap computation and frequency ranking
- TF-IDF vectorization and sparse matrix transformation
- Resume classifier training and probability prediction
- Job recommendation ranking and category filtering
- Explainable multi-factor fit scoring calculation

---

## 📊 Model Evaluation Results

- **Resume Classifier Architecture**: TF-IDF (5,000 features, unigrams + bigrams) + Logistic Regression (L2 regularization, balanced weights)
- **Classifier Performance**:
  - Test Accuracy: **99.48%**
  - Macro F1-Score: **0.9945**
  - Weighted F1-Score: **0.9949**
- **Recommender Architecture**: Memory-efficient Sparse TF-IDF (10,000 unigrams) + Cosine Similarity over 136,759 postings with near-duplicate suppression.

---

## 🔒 Methodology & Project Scope

- **Scope Definition**: SmartHire is a **production-quality local application / AI/ML prototype** designed for transparent, explainable career matching and skill discovery.
- **Ethical Standards**: Public resume and job post datasets do not contain ground-truth hiring outcomes. The system avoids training ungrounded binary hire/reject models and instead provides transparent textual relevance metrics and skill coverage indicators.
- **Privacy**: File parsing and scoring execute strictly in local memory without remote telemetry or cloud transmission.

---

## 📄 License
Open source project for educational and career-matching purposes.
