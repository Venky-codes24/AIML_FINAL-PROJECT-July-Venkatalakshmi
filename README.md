# SmartHire — AI-Powered Resume & Career Matching Engine

SmartHire is a classical machine learning application and AI/ML prototype (no external LLMs or black-box APIs) for resume parsing, supervised role categorization, cosine-similarity job matching, skill-gap analysis, and career learning path recommendations.

---

## 🌟 Key Features

1. **Multi-Format Resume Parser**: Supports text extraction from **PDF**, **DOCX**, and **TXT** files with automatic normalization, letter-spacing repair, and verifiable metadata extraction (education, experience, contact details).
2. **Supervised Resume Category Classifier**: Predicts candidate role category across 25 target specializations using **TF-IDF + Logistic Regression** with confidence scoring and probability distributions.
3. **Unsupervised Job Recommender**: Ranks and matches the candidate's resume against **136,000+ job listings** using memory-efficient **Sparse TF-IDF + Cosine Similarity** with active duplicate suppression.
4. **Transparent Match Scoring**: Evaluates candidate-to-job similarity scored as `Match Score: X/100` (clearly distinguished from hiring probabilities).
5. **Comprehensive Skill Extraction & Normalization**: High-precision vocabulary covering 100+ canonical technical domains with alias matching, symbol-awareness (`C++`, `C#`, `.NET`), context-isolated single-letter detection (`C`, `R`, `Go`), and compound isolation (`MySQL` vs `SQL`, `JavaScript` vs `Java`).
6. **ATS Readiness Score & Optimization**: Deterministic 0-100 applicant tracking system compliance audit measuring section completeness, technical skill density, contact details, action-oriented verbs, and measurable impact metrics.
7. **Skill-Gap Analysis & High-Demand Learning Path**: Identifies exact missing tools across top matches, prioritizing them into an 8-week actionable career roadmap and domain radar profile.
8. **Explainable Multi-Factor Alignment**: Combines semantic text similarity, skill overlap, and role relevance into a transparent breakdown without synthetic labels.
9. **Instant Assessment Report Export**: One-click generation of comprehensive, downloadable Markdown audit summaries.
10. **Interactive Streamlit Web Portal**: Clean, modern UI with distinct tabs for file upload analysis, interactive demo profiles, and batch validation.

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
│   │   ├── match_features.py         # Skill extraction, gaps & recommendations
│   │   ├── build_skills_vocab.py     # Skill taxonomy builder & validator
│   │   └── career_insights.py        # ATS audit, career roadmaps & reports
│   ├── models/
│   │   ├── classifier.py             # Classifier training & prediction
│   │   ├── recommender.py            # Cosine similarity matching & deduplication
│   │   ├── clustering.py             # KMeans topic clustering
│   │   ├── fit_predictor.py          # Explainable multi-factor fit scorer
│   │   ├── salary_predictor.py       # Heuristic salary band estimator
│   │   └── train.py                  # Unified CLI training script
│   └── parsing/
│       └── resume_parser.py          # PDF, DOCX, TXT parser & metadata extractor
├── tests/
│   ├── test_features.py              # Tests for preprocessing & TF-IDF
│   ├── test_parser.py                # Tests for PDF, DOCX, TXT parsing
│   ├── test_classifier.py            # Tests for category classification
│   ├── test_recommender.py           # Tests for job matching & fit index
│   ├── test_skills_pipeline.py       # Precision tests for skills taxonomy
│   └── verify_all.py                 # End-to-end multi-pipeline validation
├── download_data.py                  # Automated Kaggle dataset downloader
├── requirements.txt                  # Pinned Python dependencies
├── pytest.ini                        # Pytest configuration
└── README.md                         # Documentation
```

---

## ⚡ How to Run the Program (Step-by-Step with Comments)

Follow these steps to set up the environment and run the application locally:

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.11 and 3.12)
- **Git**

### 2. Clone the Repository & Set Up Environment
```bash
# Clone the project repository from GitHub
git clone https://github.com/Venky-codes24/AIML_FINAL-PROJECT.git

# Navigate into the project root directory
cd AIML_FINAL-PROJECT

# Create a dedicated Python virtual environment named '.venv'
python -m venv .venv

# --- Activate the Virtual Environment ---
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On Linux / macOS:
source .venv/bin/activate

# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install all pinned dependencies (Streamlit, Scikit-learn, Pandas, PyPDF, etc.)
pip install -r requirements.txt
```

### 3. Download Raw Datasets (First-Time Setup Only)
```bash
# Automatically downloads raw Kaggle datasets (Resume dataset & 136k+ job postings)
python download_data.py
```

### 4. Preprocess Data & Clean Text
```bash
# Cleans raw text, handles missing fields, and generates processed datasets:
# - data/processed/resumes_clean.csv (962 labeled resumes across 25 categories)
# - data/processed/jobs_clean.csv (136,759 cleaned job postings)
python -m src.data.preprocess
```

### 5. Train Models & Build Feature Indices
```bash
# Trains the supervised TF-IDF + Logistic Regression classifier (99.48% accuracy)
# Builds the sparse TF-IDF job catalog matrix (136k+ postings, 10,000 features)
# Exports canonical skills taxonomy dictionary (models/skills_vocab.json)
python -m src.models.train
```

> **Note:** Pre-trained model artifacts are already included in the `models/` directory. If they exist, you can skip steps 3–5 and immediately launch the web app below.

### 6. Launch the Interactive Streamlit Web Portal
```bash
# Launch the Streamlit application locally on default port 8501
streamlit run app/streamlit_app.py

# Optional: Run on custom port or expose to local network
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Optional: Run in headless mode (ideal for remote servers/VMs)
streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```
- Open your web browser and navigate to: **`http://localhost:8501`** (or `http://<your-server-ip>:8501`).
- You can upload your own **PDF**, **DOCX**, or **TXT** resume, or click on any of the built-in **Demo Sample Profiles** in the sidebar to test instantly.
- To stop the server at any time, press **`Ctrl + C`** in your terminal.

---

## 🎯 Accuracy Checking & Model Evaluation (Commands with Comments)

SmartHire includes comprehensive test suites and automated evaluation scripts to verify model accuracy, taxonomy precision, and end-to-end pipeline integrity.

### 1. Run Complete Automated Test Suite (All 34 Tests)
```bash
# Run the complete test suite with detailed test-by-test breakdown
python -m pytest -v

# Run the test suite in concise summary mode
python -m pytest -q
```
*Expected Result:* **`34 passed`** in under 30 seconds.

### 2. Verify Resume Classifier Accuracy & Metrics
```bash
# Run isolated tests for the supervised resume category classifier
# Checks that model test accuracy exceeds the 95% threshold and top-3 probabilities calibrate properly
python -m pytest tests/test_classifier.py -v
```

To re-run training and display the exact accuracy and F1 scores in terminal:
```bash
# Trains and outputs classification accuracy, macro F1, and weighted F1
python -m src.models.train
```
*Output Metrics:*
- **Test Accuracy**: **`99.48%`**
- **Macro F1-Score**: **`0.9945`**
- **Weighted F1-Score**: **`0.9949`**
- **Total Categories**: 25 specialized tech domains

### 3. Check Skills Disambiguation & Token Precision (100% Precision)
```bash
# Tests the high-precision skill extraction engine against edge-case ambiguities:
# - Validates symbol-heavy languages: C++, C#, .NET
# - Validates context-isolated single letters: 'C', 'R', 'Go' (no false triggers in words like 'Docker', 'React')
# - Validates compound sub-token guards: 'MySQL' does not trigger 'SQL', 'JavaScript' does not trigger 'Java'
python -m pytest tests/test_skills_pipeline.py -v
```

### 4. Run End-to-End System Integrity Verification
```bash
# Executes an end-to-end integration test through all 9 core and optional system pipelines:
# 1. Resume Parsing & Metadata Extraction
# 2. Supervised Category Classification
# 3. High-Precision Skills Extraction
# 4. Sparse Job Catalog Indexing
# 5. Cosine Similarity Matching & Deduplication
# 6. Candidate Fit Alignment Scoring
# 7. High-Demand Skill Gap & Learning Roadmap
# 8. Salary Band Estimation
# 9. KMeans Role Topic Clustering
python tests/verify_all.py
```
*Expected Output:* `>>> ALL 9 CORE & OPTIONAL PIPELINES VERIFIED SUCCESSFULLY! <<<`

### 5. Inspect Visual Confusion Matrix
When training finishes, a high-resolution 25-category confusion matrix is generated and saved to:
```
reports/figures/confusion_matrix.png
```
You can inspect this image to visually examine per-category classification precision across all 25 specializations.

---

## 🚀 Deployment Guide (How to Deploy SmartHire)

SmartHire can be deployed to the cloud using any of the following approaches:

### Option A: Streamlit Community Cloud (Recommended & Free)
The easiest way to deploy SmartHire publicly with a live URL:

1. **Push your code to GitHub**:
   Ensure your code is pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Prepare repository for deployment"
   git push origin main
   ```
2. **Handle Model Artifacts**:
   The trained model artifacts in `models/` total ~96 MB (`job_matrix.pkl` is ~44 MB, `job_metadata.pkl` is ~49 MB).
   - If tracking models in git: Each file is under GitHub's 100 MB hard limit.
   - Alternatively, track them via **Git LFS** or upload them to a release/cloud storage bucket.
3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
   - Click **"New app"**.
   - Select your repository: `Venky-codes24/AIML_FINAL-PROJECT`.
   - Set **Branch**: `main`.
   - Set **Main file path**: `app/streamlit_app.py`.
   - Click **"Deploy!"**.
   - Streamlit will automatically read `requirements.txt`, install dependencies, and launch the portal with a public `https://<your-app-name>.streamlit.app` URL.

---

### Option B: Docker Container Deployment (Cloud / VPS / Render / Railway)
SmartHire comes with a production-ready `Dockerfile` and `.dockerignore`.

#### 1. Build the Docker Image
```bash
# Build the container image tagged as 'smarthire-engine'
docker build -t smarthire-engine .
```

#### 2. Run the Container
```bash
# Run container in detached mode (-d) mapping host port 8501 to container port 8501
docker run -d -p 8501:8501 --name smarthire-app smarthire-engine
```
- Access the app at: **`http://localhost:8501`**.

#### 3. View Logs or Stop Container
```bash
# View real-time container application logs
docker logs -f smarthire-app

# Stop the container
docker stop smarthire-app

# Remove container
docker rm smarthire-app
```

#### Deploying Container to Cloud Hosts:
- **Render.com**: Create a new **Web Service** -> Link GitHub -> Select **Docker** environment -> Deploy. Set port to `8501`.
- **Railway.app**: New Project -> Deploy from GitHub -> Railway detects the `Dockerfile` automatically.
- **AWS ECS / Google Cloud Run / Azure Container Apps**: Push `smarthire-engine` to Amazon ECR or Google Artifact Registry and launch a serverless container.

---

### Option C: Hugging Face Spaces (Free 16 GB RAM CPU Tier)
Hugging Face Spaces offers a generous 16 GB RAM CPU tier, making it ideal for large text corpora and sparse matrices:

1. Create a free account on [huggingface.co](https://huggingface.co).
2. Go to **Spaces** -> **Create new Space**.
3. Set **SDK** to **Streamlit** and choose **Public**.
4. Clone the Space repository locally or link it directly with your GitHub repo.
5. Ensure `requirements.txt` and `app/streamlit_app.py` are present in root or configure `app_file: app/streamlit_app.py` in the `README.md` metadata.
6. Hugging Face automatically spins up the instance and serves the live web application.

---

### Option D: Ubuntu / Debian Linux Virtual Machine (AWS EC2 / DigitalOcean)
To host on a dedicated Linux VPS:

```bash
# 1. Update system and install Python + Git
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git

# 2. Clone repo and enter folder
git clone https://github.com/Venky-codes24/AIML_FINAL-PROJECT.git
cd AIML_FINAL-PROJECT

# 3. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Run with nohup or systemd in the background
nohup streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > app.log 2>&1 &
```
- For production domains, set up **Nginx** as a reverse proxy forwarding port 80/443 to `http://127.0.0.1:8501` and configure free SSL with **Certbot** (`sudo certbot --nginx`).

---

## 🔒 Methodology & Project Scope

- **Scope Definition**: SmartHire is a **production-quality local application / AI/ML prototype** designed for transparent, explainable career matching and skill discovery.
- **Ethical Standards**: Public resume and job post datasets do not contain ground-truth hiring outcomes. The system avoids training ungrounded binary hire/reject models and instead provides transparent textual relevance metrics and skill coverage indicators.
- **Privacy**: File parsing and scoring execute strictly in local memory without remote telemetry or cloud transmission.

---

## 📄 License
Open source project for educational and career-matching purposes.
