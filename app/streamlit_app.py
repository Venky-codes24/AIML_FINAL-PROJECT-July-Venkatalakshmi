"""SmartHire — AI-Powered Resume & Career Matching Web Application.

Production-Quality Local Application & AI/ML Prototype.
Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path for direct module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.features.career_insights import (
    compute_ats_score,
    compute_domain_skill_distribution,
    generate_assessment_report_markdown,
    generate_career_roadmap,
)
from src.features.match_features import (
    compute_skill_gap,
    extract_sections,
    extract_skills,
    recommend_skills_to_learn,
)
from src.models.classifier import load_classifier, predict_category
from src.models.fit_predictor import compute_candidate_fit
from src.models.recommender import load_recommender, recommend_jobs
from src.parsing.resume_parser import ResumeParsingError, extract_metadata, parse_resume

# Page configuration
st.set_page_config(
    page_title="SmartHire — Career Matching & ATS Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich styling, card layout, and orange-accented modern UI matching mockup
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1E293B;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    /* Titles */
    .title-dark {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
    }
    .title-orange {
        font-size: 2.3rem;
        font-weight: 800;
        color: #FF5722;
        letter-spacing: -0.5px;
    }
    .title-underline-accent {
        width: 48px;
        height: 4px;
        background: #FF5722;
        border-radius: 2px;
        margin-top: -2px;
        margin-bottom: 0.8rem;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #FFF7ED;
        border-left: 4px solid #FF5722;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #9A3412;
        margin-bottom: 1.2rem;
    }

    /* Top Navigation Bar: Brand + Profile & Theme */
    .top-nav-bar {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        align-items: center !important;
        width: 100% !important;
        padding: 0.4rem 0 0.8rem 0;
        margin-bottom: 0.4rem;
    }
    .brand-title-group {
        display: flex;
        flex-direction: column;
    }
    .profile-header-group {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.75rem !important;
        white-space: nowrap !important;
    }
    .theme-icon-static {
        width: 38px;
        height: 38px;
        min-width: 38px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        cursor: pointer;
        line-height: 1;
        user-select: none;
    }
    .user-avatar-circle {
        width: 38px;
        height: 38px;
        min-width: 38px;
        border-radius: 50%;
        background: #FF5722;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(255, 87, 34, 0.35);
        flex-shrink: 0;
        line-height: 1;
    }
    .user-info-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: left;
    }
    .user-welcome-sub {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 600;
        line-height: 1.15;
    }
    .user-welcome-name {
        font-size: 0.92rem;
        color: #0F172A;
        font-weight: 800;
        line-height: 1.2;
    }

    /* Feature Pill Badges */
    .feature-pills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1.4rem;
        margin-top: 0.4rem;
    }
    .feature-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.95rem;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .pill-purple {
        background-color: #F3E8FF;
        color: #7E22CE;
        border: 1px solid #E9D5FF;
    }
    .pill-green {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .pill-pink {
        background-color: #FCE7F3;
        color: #BE185D;
        border: 1px solid #FBCFE8;
    }
    .pill-orange {
        background-color: #FFEDD5;
        color: #C2410C;
        border: 1px solid #FED7AA;
    }

    /* "What You'll Get" 4-Card Grid */
    .section-headline {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin: 1.6rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .what-you-get-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .benefit-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.15rem 1rem;
        display: flex;
        gap: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .benefit-icon-wrapper {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        flex-shrink: 0;
    }
    .benefit-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .benefit-desc {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Bottom Action Banner */
    .cta-banner-card {
        background-color: #FFF7ED;
        border: 1px solid #FFEDD5;
        border-radius: 14px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.6rem;
    }
    .cta-left-box {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .cta-lightbulb {
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    .cta-text-content {
        font-size: 0.88rem;
        color: #9A3412;
        font-weight: 500;
    }

    /* Cards & Badges */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease-in-out;
    }
    .card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    }
    .match-badge {
        display: inline-block;
        background: #FF5722;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        float: right;
        box-shadow: 0 2px 6px rgba(255, 87, 34, 0.3);
    }

    .skill-pill {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.22rem 0.6rem;
        border-radius: 8px;
        margin: 0.2rem 0.25rem 0.2rem 0;
        border: 1px solid #E2E8F0;
    }
    .skill-pill-matched {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .skill-pill-missing {
        background-color: #FFF7ED;
        color: #C2410C;
        border: 1px solid #FFD8A8;
    }
    .meta-tag {
        font-size: 0.85rem;
        color: #475569;
        margin-right: 1.2rem;
        display: inline-block;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.4rem;
    }
    .stat-box {
        text-align: center;
        padding: 1.1rem 0.9rem;
        background-color: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .stat-number {
        font-size: 1.55rem;
        font-weight: 800;
        color: #FF5722;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .roadmap-step {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #FF5722;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .ats-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Buttons: Modern Orange */
    div.stButton > button {
        background-color: #FF5722 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(255, 87, 34, 0.25) !important;
    }
    div.stButton > button:hover {
        background-color: #F4511E !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(255, 87, 34, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stDownloadButton"] > button {
        background-color: #FF5722 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 2px 8px rgba(255, 87, 34, 0.25) !important;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-bottom-card {
        background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
        border: 1px solid #FED7AA;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Demo sample resumes strictly for demonstration/testing
DEMO_SAMPLE_RESUMES = {
    "— Select a Demo Sample Resume —": "",
    "Demo Sample 1: Full-Stack & Python / C Candidate": """
JOHN DOE — SOFTWARE ENGINEER
Email: john.doe@example.com | Phone: +1 555-0199
GitHub: github.com/johndoe | Portfolio: johndoe.dev

PROFESSIONAL SUMMARY:
Proactive Software Engineer with 3+ years of experience designing and implementing scalable web applications and distributed backend systems. Engineered robust REST APIs and reduced API response latency by 35%.

TECHNICAL SKILLS:
• Programming Languages: Java, Python, C, JavaScript
• Web Technologies: HTML, CSS, Bootstrap, React.js, Node.js
• Database Management: SQL, MySQL
• Tools & Platforms: GitHub, VS Code, Eclipse, Jupyter Notebook

WORK EXPERIENCE:
Software Engineer at Tech Innovations (2022 - Present)
- Developed and deployed full-stack client portals using React.js and Node.js backend services.
- Architected RESTful microservices with Python and FastAPI, handling over 25k daily requests.
- Optimized relational SQL queries and database schemas in MySQL, improving throughput by 20%.

EDUCATION:
Bachelor of Technology in Computer Science and Engineering
    """.strip(),
    "Demo Sample 2: Senior Data Scientist / ML Engineer": """
SENIOR DATA SCIENTIST & MACHINE LEARNING ENGINEER (DEMO SAMPLE)
Summary: Experienced Data Scientist with 4+ years of expertise in Statistical Modeling, Machine Learning, Deep Learning, and NLP.
Skills: Python, SQL, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, NLP, BERT, AWS, Docker, Git, Tableau, Statistics, Jupyter Notebook.
Experience:
- Developed predictive classification and regression models in Python and Scikit-learn, achieving 94% accuracy.
- Built end-to-end NLP pipelines using Hugging Face Transformers for text analytics and document classification.
- Containerized and deployed ML microservices on AWS EC2 with FastAPI and Docker, handling 50k+ inferences daily.
Education: Bachelor of Technology in Computer Science and Engineering.
Contact: demo.candidate.ml@example.com | Phone: +1 555-0144
    """.strip(),
    "Demo Sample 3: QA Automation Engineer (SDET)": """
JOHN DOE — SOFTWARE DEVELOPMENT ENGINEER IN TEST (SDET)
Email: john.sdet@example.com | Phone: +1 555-0188
Summary: Test automation specialist with 4+ years designing robust hybrid test frameworks.
Skills: Java, Python, HTML, CSS, JavaScript, Spring Boot, MySQL, Selenium, Cucumber, TestNG, Git, Postman, JMeter.
Experience:
- Developed hybrid test automation frameworks using Selenium WebDriver, TestNG, and Cucumber BDD, reducing regression test cycles by 40%.
- Integrated Spring Boot microservices with MySQL backend databases.
- Built automated API regression suites with Postman and RestAssured.
Education: B.Tech in Computer Science and Engineering.
    """.strip(),
    "Demo Sample 4: DevOps & Cloud Infrastructure Engineer": """
DEVOPS & CLOUD INFRASTRUCTURE ENGINEER (DEMO SAMPLE)
Summary: Cloud engineer with hands-on experience in AWS Cloud, Kubernetes, Docker, Terraform, and CI/CD pipelines.
Skills: AWS, Linux, Docker, Kubernetes, Terraform, Ansible, CI/CD, Jenkins, Git, Python, Bash/Shell, Nginx.
Experience:
- Automated multi-region AWS cloud infrastructure using Terraform and CloudFormation, achieving 99.9% uptime.
- Managed Kubernetes (EKS) clusters orchestrating microservices in staging and production.
- Built Jenkins and GitHub Actions CI/CD automation pipelines, reducing release deployment times by 50%.
Education: Bachelor of Engineering in Information Technology.
Contact: demo.candidate.devops@example.com | Phone: +1 555-0177
    """.strip(),
}


@st.cache_resource(show_spinner=False)
def get_classifier():
    """Load classifier model and vectorizer with resource caching."""
    return load_classifier()


@st.cache_resource(show_spinner=False)
def get_recommender():
    """Load recommender vectorizer, sparse matrix, and metadata table with caching."""
    return load_recommender()


def check_model_status() -> Dict[str, bool]:
    """Check availability of trained model artifacts on disk."""
    return {
        "classifier": config.RESUME_CLASSIFIER_PATH.exists() and config.RESUME_VECTORIZER_PATH.exists(),
        "recommender": config.JOB_VECTORIZER_PATH.exists() and config.JOB_MATRIX_PATH.exists() and config.JOB_METADATA_PATH.exists(),
        "processed_data": config.RESUMES_CLEAN_CSV.exists() and config.JOBS_CLEAN_CSV.exists(),
    }


def render_radar_chart(domain_dist: Dict[str, int]) -> go.Figure:
    """Generate interactive Plotly Radar / Spider Chart for skill domain distribution."""
    categories = list(domain_dist.keys())
    values = list(domain_dist.values())
    
    # Close polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(255, 87, 34, 0.22)",
            line=dict(color="#FF5722", width=2.5),
            marker=dict(size=7, color="#E64A19"),
            name="Your Skills",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(4, max(values) + 1)],
                tickfont=dict(size=10, color="#64748B"),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#1E293B", family="Plus Jakarta Sans"),
                gridcolor="#E2E8F0",
            ),
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_market_demand_chart(recommendations: List[Dict[str, Any]]) -> go.Figure:
    """Generate interactive Plotly horizontal bar chart for top in-demand skills in matched jobs."""
    skill_counts = {}
    for job in recommendations:
        for skill in job.get("skills_list", []):
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    top_demanded = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    if not top_demanded:
        top_demanded = [("Core Technical Skills", 1)]

    skills_names = [item[0] for item in top_demanded][::-1]
    counts = [item[1] for item in top_demanded][::-1]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=skills_names,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale=[[0, "#FED7AA"], [1, "#FF5722"]],
                line=dict(color="#EA580C", width=1),
            ),
            text=[f"{c} jobs" for c in counts],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text="Top Skills Demanded Across Recommended Jobs", font=dict(size=13, color="#1E293B", family="Plus Jakarta Sans")),
        xaxis=dict(title="Occurrence Count in Matches", tickfont=dict(size=10), gridcolor="#F1F5F9"),
        yaxis=dict(tickfont=dict(size=11, color="#1E293B")),
        margin=dict(l=20, r=20, t=40, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/briefcase.png", width=56)
    st.sidebar.title("SmartHire")
    st.sidebar.caption("AI/ML Career Matching & ATS Screening Hub")

    # Primary Mode Switcher
    app_mode = st.sidebar.radio(
        "🔀 Select Platform View:",
        ["👤 Candidate Career Hub", "👔 Recruiter & ATS Screening Hub"],
        help="Switch between Candidate Career Guidance Mode and Recruiter Bulk Screening Mode.",
    )

    status = check_model_status()

    # Recommender Settings
    st.sidebar.markdown("### ⚙️ Matching Configuration")
    top_n = st.sidebar.slider("Number of Top Matches", min_value=3, max_value=15, value=config.TOP_N)
    
    cat_filter = st.sidebar.text_input(
        "Filter by Job Title / Keyword",
        placeholder="e.g. Data Scientist, Java, Engineer",
        help="Filters the candidate job pool to titles/descriptions containing this keyword.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Engine Status")
    if status["classifier"] and status["recommender"]:
        st.sidebar.success("✅ Models Loaded & Ready")
        if st.sidebar.button("🔄 Reload Engine & Clear Cache"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.sidebar.warning("⚠️ Model Artifacts Missing")
        if st.sidebar.button("Train Models"):
            with st.spinner("Training models and indexing job corpus..."):
                from src.models.train import main as train_main
                train_main()
                st.cache_resource.clear()
                st.success("Training completed! Please proceed.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Methodology & Standards:**
        - **Supervised Classifier:** TF-IDF + Logistic Regression
        - **Recommender:** Sparse TF-IDF + Cosine Similarity
        - **Deduplication:** Active
        - **Match Score:** Resume/Job text & skill similarity (not hiring probability)
        """
    )

    # Sidebar Bottom Brand Card (from Mockup)
    st.sidebar.markdown(
        """
        <div class="sidebar-bottom-card">
            <div style="font-size: 2.2rem; margin-bottom: 0.35rem;">🚀</div>
            <div style="font-weight: 800; font-size: 0.95rem; color: #9A3412; margin-bottom: 0.2rem;">AI Career Copilot</div>
            <div style="font-size: 0.78rem; color: #C2410C; line-height: 1.35;">Instant ATS feedback, tailored job matches, & skill gap analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if models exist
    if not status["classifier"] or not status["recommender"]:
        st.error(
            "⚠️ Model artifacts were not found in `models/`. "
            "Please train them using the sidebar **'Train Models'** button or by running `python -m src.models.train` in your terminal."
        )
        return

    # Load cached models
    try:
        clf_model, clf_vec = get_classifier()
        rec_vec, rec_matrix, job_meta = get_recommender()
    except Exception as err:
        st.error(f"Error loading model artifacts: {err}")
        return

    # =========================================================================
    # VIEW 1: CANDIDATE CAREER HUB
    # =========================================================================
    if app_mode == "👤 Candidate Career Hub":
        # Top Header Bar: SmartHire Career Hub (Left) + Profile & Theme (Right)
        st.markdown(
            """
            <div class="top-nav-bar">
                <div class="brand-title-group">
                    <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                        <span class="title-dark">SmartHire</span>
                        <span class="title-orange">Career Hub</span>
                    </div>
                    <div class="title-underline-accent"></div>
                </div>
                <div class="profile-header-group">
                    <div class="theme-icon-static" title="Toggle Theme">🌙</div>
                    <div class="user-avatar-circle">V</div>
                    <div class="user-info-text">
                        <span class="user-welcome-sub">Welcome</span>
                        <span class="user-welcome-name">Venky <span style="font-size:0.7rem; color:#94A3B8;">▾</span></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feature Pill Badges Row (Matching Mockup)
        st.markdown(
            """
            <div class="feature-pills-container">
                <div class="feature-pill-badge pill-purple">
                    <span>📄</span> <span>Resume Analysis</span>
                </div>
                <div class="feature-pill-badge pill-green">
                    <span>🛡️</span> <span>ATS Optimization</span>
                </div>
                <div class="feature-pill-badge pill-pink">
                    <span>🎯</span> <span>Job Matching</span>
                </div>
                <div class="feature-pill-badge pill-orange">
                    <span>📊</span> <span>Career Insights</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sub-header">AI-powered resume role categorization, ATS optimization, interactive job matching, and career roadmaps.</div>',
            unsafe_allow_html=True,
        )

        # Step 1: Input Section
        st.markdown('<div class="section-title">1. Provide Resume</div>', unsafe_allow_html=True)
        
        input_tab1, input_tab2, input_tab3 = st.tabs(["📁 Upload Your Resume", "✍️ Paste Resume Text", "🧪 Try a Demo Sample Resume"])

        resume_raw_text = ""
        source_label = ""
        uploaded_name = "resume_document.pdf"

        with input_tab1:
            uploaded_file = st.file_uploader(
                "Upload Resume File (PDF, DOCX, or TXT)",
                type=["pdf", "docx", "txt"],
                help="Your resume is parsed locally in-memory and is not stored remotely.",
                key="file_uploader",
            )
            if uploaded_file is not None:
                try:
                    raw_bytes = uploaded_file.getvalue()
                    uploaded_name = uploaded_file.name
                    parsed_dict = parse_resume(raw_bytes, filename=uploaded_file.name)
                    resume_raw_text = parsed_dict["raw_text"]
                    source_label = f"Uploaded File: `{uploaded_file.name}`"
                    st.success(f"✓ Parsed `{uploaded_file.name}` ({len(resume_raw_text):,} characters extracted).")
                except ResumeParsingError as parse_err:
                    st.error(f"Parsing error: {parse_err}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error reading file: {e}")
                    return

        with input_tab2:
            pasted_text = st.text_area(
                "Paste your resume text directly:",
                placeholder="Paste your skills, experience, and summary here...",
                height=150,
                key="pasted_text_input",
            )
            if pasted_text.strip() and not resume_raw_text:
                resume_raw_text = pasted_text.strip()
                source_label = "Direct Text Input"
                uploaded_name = "pasted_resume.txt"

        with input_tab3:
            st.caption("Select a pre-built sample resume to test the ML pipeline without uploading your own document:")
            selected_sample = st.selectbox(
                "Demo Samples:",
                options=list(DEMO_SAMPLE_RESUMES.keys()),
                key="sample_selector",
            )
            if selected_sample and DEMO_SAMPLE_RESUMES[selected_sample] and not resume_raw_text:
                resume_raw_text = DEMO_SAMPLE_RESUMES[selected_sample]
                source_label = f"Demo Sample: `{selected_sample}`"
                uploaded_name = f"{selected_sample.split(':')[0]}.txt"
                st.info(f"ℹ️ Loaded {source_label} for demonstration.")

        if not resume_raw_text.strip():
            # What You'll Get Cards (from Mockup)
            st.markdown('<div class="section-headline">✨ What You\'ll Get with SmartHire</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="what-you-get-grid">
                    <div class="benefit-card">
                        <div class="benefit-icon-wrapper" style="background:#F3E8FF; color:#7E22CE;">🎯</div>
                        <div>
                            <div class="benefit-title">Role Prediction</div>
                            <div class="benefit-desc">Supervised ML classifier trained on thousands of tech resumes with confidence scoring.</div>
                        </div>
                    </div>
                    <div class="benefit-card">
                        <div class="benefit-icon-wrapper" style="background:#DCFCE7; color:#15803D;">🛡️</div>
                        <div>
                            <div class="benefit-title">ATS Optimizer</div>
                            <div class="benefit-desc">Comprehensive 100-point audit across contact info, section structure, and action verbs.</div>
                        </div>
                    </div>
                    <div class="benefit-card">
                        <div class="benefit-icon-wrapper" style="background:#FCE7F3; color:#BE185D;">💼</div>
                        <div>
                            <div class="benefit-title">Job Matching</div>
                            <div class="benefit-desc">Cosine similarity ranking across live job corpus with skill overlap breakdown.</div>
                        </div>
                    </div>
                    <div class="benefit-card">
                        <div class="benefit-icon-wrapper" style="background:#FFEDD5; color:#C2410C;">📈</div>
                        <div>
                            <div class="benefit-title">Career Roadmap</div>
                            <div class="benefit-desc">Personalized phased learning steps to close critical skill gaps and boost interview calls.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # CTA Banner (from Mockup)
            st.markdown(
                """
                <div class="cta-banner-card">
                    <div class="cta-left-box">
                        <span class="cta-lightbulb">💡</span>
                        <div class="cta-text-content">
                            <strong>Ready to accelerate your career?</strong> Upload your resume above or choose a demo profile to experience our end-to-end ML career engine.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info("👆 Please upload a resume file in the 'Upload Your Resume' tab, paste text, or select a demo sample to begin analysis.")
            return

        # Step 2: Resume Analysis & Classification
        with st.spinner("Analyzing resume content, identifying skills, and computing matches..."):
            # Prediction
            pred_results = predict_category(resume_raw_text, clf_model, clf_vec, top_k=3)
            predicted_cat = pred_results["predicted_category"]
            confidence_pct = int(round(pred_results["confidence"] * 100))

            # Skill extraction & Metadata
            skills_found = extract_skills(resume_raw_text)
            metadata = extract_metadata(resume_raw_text)
            sections = extract_sections(resume_raw_text)

            # ATS Readiness Evaluation
            ats_data = compute_ats_score(resume_raw_text, skills_found, metadata, sections)

            # Job Recommendations with deduplication and diversity
            recommendations = recommend_jobs(
                resume_text=resume_raw_text,
                top_n=top_n,
                category_filter=cat_filter if cat_filter else None,
                job_vectorizer=rec_vec,
                job_matrix=rec_matrix,
                job_metadata=job_meta,
            )

            # Recommended Skills to Learn
            skills_to_learn = recommend_skills_to_learn(skills_found, recommendations, top_k=6)

            # Domain Skill Distribution for Radar
            domain_dist = compute_domain_skill_distribution(skills_found)

            # Career Learning Roadmap
            roadmap = generate_career_roadmap(predicted_cat, skills_to_learn, skills_found)

        # Step 3: Display Resume Overview & Role Prediction
        st.markdown('<div class="section-title">2. Resume Overview & Role Prediction</div>', unsafe_allow_html=True)
        st.caption(f"Active Document: {source_label}")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{predicted_cat}</div>
                    <div class="stat-label">Predicted Role Specialization</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_col2:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{confidence_pct}%</div>
                    <div class="stat-label">Classification Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_col3:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{len(skills_found)}</div>
                    <div class="stat-label">Skills Identified in Resume</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_col4:
            st.markdown(
                f"""
                <div class="stat-box" style="border-top: 3px solid {ats_data['color']};">
                    <div class="stat-number" style="color: {ats_data['color']};">{ats_data['ats_score']}/100</div>
                    <div class="stat-label">ATS Readiness Score ({ats_data['tier']})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Skills Found & Alternative Roles
        st.markdown("<br>", unsafe_allow_html=True)
        detail_col1, detail_col2 = st.columns([3, 2])

        with detail_col1:
            st.markdown("**Skills Found in Resume:**")
            if skills_found:
                pill_html = "".join([f'<span class="skill-pill skill-pill-matched">✓ {s}</span>' for s in skills_found])
                st.markdown(f"<div>{pill_html}</div>", unsafe_allow_html=True)
            else:
                st.warning("No standard canonical skills detected in the uploaded text.")

        with detail_col2:
            st.markdown("**Top Predicted Role Distribution:**")
            top_cats = pred_results.get("top_categories", [])
            for cat_name, prob in top_cats:
                st.progress(float(prob), text=f"{cat_name} ({int(prob * 100)}%)")

        # Feature 3: ATS Resume Optimizer & Actionable Audit
        with st.expander("🎯 ATS Resume Optimizer & Optimization Audit", expanded=False):
            ats_left, ats_right = st.columns([1, 1])
            with ats_left:
                st.markdown(f"### Overall ATS Score: **{ats_data['ats_score']}/100**")
                st.markdown(f"**Readiness Level:** `{ats_data['tier']}`")
                st.markdown("**Evaluation Checklist:**")
                for title, desc, passed, score_pts, max_pts in ats_data["checks"]:
                    icon = "✅" if passed else "⚠️"
                    st.markdown(f"- {icon} **{title}** ({score_pts}/{max_pts} pts) — {desc}")

            with ats_right:
                st.markdown("### 💡 Actionable Improvement Tips")
                for tip in ats_data["tips"]:
                    st.markdown(f"- 📌 {tip}")

        # Metadata Preview
        with st.expander("📄 View Parsed Resume Text & Extracted Details"):
            meta_sub1, meta_sub2 = st.columns(2)
            with meta_sub1:
                if metadata.get("education"):
                    st.markdown("**Education References:**")
                    for edu in metadata["education"]:
                        st.markdown(f"- {edu}")
                if metadata.get("experience_mentions"):
                    st.markdown("**Experience References:**")
                    for exp in metadata["experience_mentions"]:
                        st.markdown(f"- {exp}")
            with meta_sub2:
                if metadata.get("emails") or metadata.get("phones"):
                    st.markdown("**Contact Information Detected:**")
                    for em in metadata.get("emails", []):
                        st.markdown(f"- ✉️ `{em}`")
                    for ph in metadata.get("phones", []):
                        st.markdown(f"- 📞 `{ph}`")
            st.markdown("**Parsed Resume Text:**")
            st.text_area("Extracted Text Content", resume_raw_text, height=180, disabled=True)

        # Feature 1: Interactive Visualizations & Analytics (Plotly Radar + Market Demand)
        st.markdown('<div class="section-title">3. Interactive Skill Profile & Market Analytics</div>', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**📊 Technical Domain Distribution (Radar Chart)**")
            st.caption("Visual breakdown of your verified skills across 6 technical competency areas:")
            radar_fig = render_radar_chart(domain_dist)
            st.plotly_chart(radar_fig, use_container_width=True)

        with chart_col2:
            st.markdown("**📈 Employer Skill Demand In Your Matches**")
            st.caption("Most frequent skill requirements across recommended jobs:")
            demand_fig = render_market_demand_chart(recommendations)
            st.plotly_chart(demand_fig, use_container_width=True)

        # Step 4: Top Job Recommendations
        st.markdown('<div class="section-title">4. Job Recommendations</div>', unsafe_allow_html=True)
        
        # Required Disclaimer Banner
        st.markdown(
            """
            <div class="disclaimer-banner">
                ℹ️ <strong>Match Score Disclaimer:</strong> Match Score represents resume-to-job text and skill similarity/relevance. It is not a prediction of hiring probability or selection.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not recommendations:
            st.warning("No job recommendations found matching the current keyword filters. Try adjusting your search.")
        else:
            for job in recommendations:
                rank = job["rank"]
                score = job["match_score"]
                title = job["title"]
                company = job["company"]
                location = job["location"]
                matched_skills = job["matched_skills"]
                missing_skills = job["missing_skills"]
                exp = job["experience"]
                snippet = job["description_snippet"]
                source = job["source"]

                matched_html = "".join([f'<span class="skill-pill skill-pill-matched">✓ {s}</span>' for s in matched_skills]) if matched_skills else '<span style="color:#94A3B8; font-size:0.85rem;">None</span>'
                missing_html = "".join([f'<span class="skill-pill skill-pill-missing">+ {s}</span>' for s in missing_skills]) if missing_skills else '<span style="color:#10B981; font-size:0.85rem;">All matched!</span>'

                card_html = textwrap.dedent(f"""
                    <div class="card">
                        <span class="match-badge">Match: {score}/100</span>
                        <h4 style="margin: 0 0 0.35rem 0; color: #0F172A;">#{rank}. {title}</h4>
                        <div style="margin-bottom: 0.6rem;">
                            <span class="meta-tag">🏢 <strong>{company}</strong></span>
                            <span class="meta-tag">📍 {location}</span>
                            <span class="meta-tag">⏳ Experience: {exp}</span>
                            <span class="meta-tag" style="text-transform: capitalize;">🌐 Source: {source}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #334155; margin-bottom: 0.8rem; line-height: 1.45;">
                            {snippet}
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <div style="font-size: 0.82rem; font-weight: 700; color: #065F46; margin-bottom: 0.2rem;">Matched Skills:</div>
                            <div style="margin-bottom: 0.5rem;">{matched_html}</div>
                            <div style="font-size: 0.82rem; font-weight: 700; color: #92400E; margin-bottom: 0.2rem;">Missing Job Skills:</div>
                            <div>{missing_html}</div>
                        </div>
                    </div>
                """).strip()
                st.markdown(card_html, unsafe_allow_html=True)

        # Step 5: Skill Gap & Learning Path
        st.markdown('<div class="section-title">5. Skill-Gap & Career Learning Roadmap</div>', unsafe_allow_html=True)
        gap_col1, gap_col2 = st.columns([1, 1])

        with gap_col1:
            st.markdown("### 🎯 Recommended Skills to Learn")
            st.markdown(
                "These high-demand skills appear most frequently across your matched jobs, but are currently missing from your resume:"
            )
            if skills_to_learn:
                for rank_idx, (skill_name, count, pct) in enumerate(skills_to_learn, start=1):
                    st.markdown(
                        f"**{rank_idx}. {skill_name}** — In **{count}** of your top matches ({pct}% occurrence)"
                    )
            else:
                st.success("🎉 You possess all key skills identified across the recommended jobs!")

        with gap_col2:
            st.markdown("### 📈 Alignment Analysis (Top Match)")
            if recommendations:
                top_job = recommendations[0]
                fit_report = compute_candidate_fit(
                    resume_text=resume_raw_text,
                    job_info=top_job,
                    predicted_category=predicted_cat,
                    semantic_similarity=top_job["raw_similarity"],
                )
                st.markdown(f"**Relevance Tier:** `{fit_report['fit_tier']}`")
                st.info(fit_report["summary"])

                breakdown = fit_report["breakdown"]
                st.markdown(
                    f"""
                    - **Skill Overlap Component:** {breakdown['skill_match_score']}/100
                    - **Semantic Context Similarity:** {breakdown['semantic_similarity_score']}/100
                    - **Role Alignment Component:** {breakdown['role_relevance_score']}/100
                    """
                )

        # Feature 5: Interactive Step-by-Step 3-Phase Career Roadmap
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🗺️ Tailored 8-Week Career Learning Roadmap")
        st.caption("A structured action plan to master your missing skills and prepare a capstone portfolio project:")

        for stage in roadmap:
            st.markdown(
                textwrap.dedent(f"""
                    <div class="roadmap-step">
                        <div style="font-size: 1.05rem; font-weight: 700; color: #1E293B; margin-bottom: 0.3rem;">
                            {stage['phase']} <span style="font-size: 0.85rem; color: #2563EB; font-weight: 600;">({stage['timeline']})</span>
                        </div>
                        <div style="font-size: 0.92rem; color: #475569; margin-bottom: 0.5rem;">
                            <strong>Target Focus:</strong> {stage['goal']}
                        </div>
                        <ul style="font-size: 0.88rem; color: #334155; margin-bottom: 0; padding-left: 1.2rem;">
                            {''.join([f'<li>{item}</li>' for item in stage['action_items']])}
                        </ul>
                    </div>
                """).strip(),
                unsafe_allow_html=True,
            )

        # Feature 2: Downloadable Assessment Report
        st.markdown('<div class="section-title">6. Download Career Assessment & Resume Audit Report</div>', unsafe_allow_html=True)
        st.markdown("Download a complete, structured summary report of your career assessment, skills inventory, and learning roadmap:")
        
        report_content = generate_assessment_report_markdown(
            filename=uploaded_name,
            predicted_role=predicted_cat,
            confidence=confidence_pct,
            skills_found=skills_found,
            ats_data=ats_data,
            recommendations=recommendations,
            skills_to_learn=skills_to_learn,
            roadmap=roadmap,
        )

        st.download_button(
            label="📥 Download Full Assessment Report (.md)",
            data=report_content,
            file_name=f"SmartHire_Career_Report_{predicted_cat.replace(' ', '_')}.md",
            mime="text/markdown",
            help="Download the complete assessment report as a Markdown document.",
        )

        # Step 7: Interactive Rule-Based Career Assistant / Mentor
        st.markdown('<div class="section-title">7. Rule-Based Career Assistant (ML-Driven Mentor)</div>', unsafe_allow_html=True)
        st.caption("Ask common career questions answered deterministically using the ML model outputs (no generative AI/LLMs).")

        mentor_expander = st.expander("💬 Open Career Assistant Q&A", expanded=False)
        with mentor_expander:
            question_option = st.selectbox(
                "Select a question:",
                options=[
                    f"What skills am I missing for my predicted role ({predicted_cat})?",
                    "How can I improve my Match Score on top recommended jobs?",
                    f"Why did the classifier predict '{predicted_cat}' for my resume?",
                    "What are the top skills demanded by employers in my match results?",
                ],
            )

            if "missing" in question_option.lower():
                if skills_to_learn:
                    missing_str = ", ".join([f"**{s[0]}**" for s in skills_to_learn[:5]])
                    st.info(
                        f"Based on our analysis of jobs matching your profile, the most critical missing skills you should learn are: {missing_str}."
                    )
                else:
                    st.success("Your resume already includes all major skills commonly required in your matched roles!")
            elif "improve" in question_option.lower():
                st.info(
                    f"To improve your match score:\n"
                    f"1. **Acquire missing target skills**: Focus on high-demand skills like {', '.join([s[0] for s in skills_to_learn[:3]]) if skills_to_learn else 'advanced domain tools'}.\n"
                    f"2. **Add detailed project descriptions**: Explicitly describe tools, libraries, and frameworks used in your past projects.\n"
                    f"3. **Ensure clear role keywords**: Use standard industry terminology in your summary and experience bullet points."
                )
            elif "why did the classifier" in question_option.lower():
                top_3 = pred_results.get("top_categories", [])
                summary_cats = ", ".join([f"{c} ({int(p*100)}%)" for c, p in top_3])
                st.info(
                    f"The Logistic Regression model evaluated your resume's TF-IDF term frequencies. "
                    f"The highest scoring category is **{predicted_cat}** ({confidence_pct}% confidence).\n\n"
                    f"Top probability distribution: {summary_cats}."
                )
            elif "demanded by employers" in question_option.lower():
                if recommendations:
                    all_demanded = {}
                    for job in recommendations:
                        for skill in job.get("skills_list", []):
                            all_demanded[skill] = all_demanded.get(skill, 0) + 1
                    top_demanded = sorted(all_demanded.items(), key=lambda x: x[1], reverse=True)[:6]
                    dem_str = ", ".join([f"**{k}** ({v} jobs)" for k, v in top_demanded])
                    st.info(f"Top skills demanded across your recommended jobs: {dem_str}.")

    # =========================================================================
    # VIEW 2: RECRUITER & ATS SCREENING HUB (FEATURE 4)
    # =========================================================================
    else:
        st.markdown('<div class="main-header">👔 Recruiter & ATS Candidate Screening Hub</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Screen and rank candidate resumes against target Job Descriptions with multi-factor fit scoring, skill-gap analysis, and company-wide role matching.</div>',
            unsafe_allow_html=True,
        )

        RECRUITER_JOB_PRESETS = {
            "Custom / Paste Your Own Job Description": "",
            "Preset 1: Software Engineer & Full-Stack (Java, Python, C, React, Node.js, SQL)": (
                "We are hiring a Software Engineer / Full-Stack Developer with hands-on experience in "
                "Java, Python, C, JavaScript, HTML, CSS, React.js, Node.js, and SQL databases. "
                "Proficiency with Git, VS Code, and building scalable RESTful web applications is required."
            ),
            "Preset 2: Java Microservices & Cloud Backend (Java, Spring Boot, MySQL, Docker, REST APIs)": (
                "We are seeking a Full Stack Java Developer with expertise in Core Java, Spring Boot, "
                "Microservices, React.js, MySQL, Docker, and REST APIs. Candidate should have hands-on experience "
                "building scalable web architectures and automated testing."
            ),
            "Preset 3: Data Scientist & AI/ML Specialist (Python, SQL, Machine Learning, Deep Learning)": (
                "Looking for a Senior Data Scientist with expertise in Python, SQL, Machine Learning, "
                "Deep Learning (TensorFlow/PyTorch), NLP, Scikit-learn, Pandas, Statistics, and Docker containerization."
            ),
            "Preset 4: SDET / QA Automation Engineer (Selenium, Cucumber, TestNG, Java, Python, SQL)": (
                "Hiring a Software Development Engineer in Test (SDET) with strong automation skills in "
                "Selenium WebDriver, TestNG, Cucumber BDD, Java or Python, SQL, Postman API testing, and Git CI/CD."
            ),
            "Preset 5: DevOps & Cloud Infrastructure Engineer (AWS, Kubernetes, Docker, Terraform, CI/CD)": (
                "Seeking a DevOps Engineer proficient in AWS, Linux, Docker, Kubernetes, Terraform, "
                "Ansible, CI/CD automation with Jenkins or GitHub Actions, and Python/Bash scripting."
            ),
        }

        rec_col1, rec_col2 = st.columns([1, 1])

        with rec_col1:
            st.markdown("### 1. Target Job Description & Requirements")
            selected_jd_preset = st.selectbox(
                "Select a Role Template or Custom Description:",
                options=list(RECRUITER_JOB_PRESETS.keys()),
                key="recruiter_jd_preset_select",
            )

            default_jd_text = RECRUITER_JOB_PRESETS[selected_jd_preset] if RECRUITER_JOB_PRESETS[selected_jd_preset] else (
                "We are hiring a Software Engineer / Full-Stack Developer with hands-on experience in "
                "Java, Python, C, JavaScript, HTML, CSS, React.js, Node.js, and SQL databases. "
                "Proficiency with Git, VS Code, and building scalable RESTful web applications is required."
            )

            jd_input = st.text_area(
                "Job Description / Hiring Requirements Text:",
                value=default_jd_text,
                height=160,
                key="recruiter_jd_input",
            )
            jd_skills = extract_skills(jd_input)

        with rec_col2:
            st.markdown("### 2. Candidate Resumes Pool")
            screening_source = st.radio(
                "Candidate Resumes Source:",
                ["📁 Upload Your Candidate Resumes (PDF, DOCX, TXT)", "🧪 Use Benchmark Demo Candidate Pool (4 Profiles)"],
                key="screening_source_choice",
            )

            candidate_pool: List[Tuple[str, str]] = []

            if "Upload" in screening_source:
                batch_files = st.file_uploader(
                    "Upload One or Multiple Resumes:",
                    type=["pdf", "docx", "txt"],
                    accept_multiple_files=True,
                    key="batch_uploader",
                )
                if batch_files:
                    for f in batch_files:
                        try:
                            parsed = parse_resume(f.getvalue(), filename=f.name)
                            candidate_pool.append((f.name, parsed["raw_text"]))
                        except Exception as parse_err:
                            st.warning(f"Skipped '{f.name}': {parse_err}")
                    if candidate_pool:
                        st.success(f"✓ Successfully loaded {len(candidate_pool)} uploaded candidate resume(s).")
            else:
                for label, r_text in DEMO_SAMPLE_RESUMES.items():
                    if r_text.strip():
                        candidate_pool.append((label.split(":")[1].strip() if ":" in label else label, r_text))
                st.info(f"Loaded {len(candidate_pool)} benchmark candidate profiles.")

        # Real-Time Dynamic Target Requirements Match Display based on candidate resume
        if candidate_pool and jd_skills:
            cand_names = [c[0] for c in candidate_pool]
            selected_cand_name = cand_names[0]
            if len(cand_names) > 1:
                selected_cand_name = st.selectbox(
                    "🎯 Preview Live Skill Match for Candidate:",
                    options=cand_names,
                    key="selected_preview_cand",
                )

            selected_cand_text = next(c[1] for c in candidate_pool if c[0] == selected_cand_name)
            selected_cand_skills = extract_skills(selected_cand_text)
            matched_with_resume = [s for s in jd_skills if s in selected_cand_skills]
            missing_from_resume = [s for s in jd_skills if s not in selected_cand_skills]

            m_pill_html = " ".join([f'<span class="skill-pill skill-pill-matched">✓ {s}</span>' for s in matched_with_resume]) if matched_with_resume else '<span style="color:#94A3B8; font-size:0.85rem;">None</span>'
            mis_pill_html = " ".join([f'<span class="skill-pill skill-pill-missing">+ {s}</span>' for s in missing_from_resume]) if missing_from_resume else '<span style="color:#10B981; font-size:0.85rem;">All Job Requirements Matched!</span>'

            st.markdown(
                textwrap.dedent(f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:1rem 1.25rem; margin-top:0.75rem; margin-bottom:1rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                            <span style="font-weight:700; color:#0F172A; font-size:0.95rem;">
                                🎯 Target Job Skills Matched with Resume ({selected_cand_name}):
                            </span>
                            <span style="font-size:0.85rem; font-weight:700; color:#065F46; background:#ECFDF5; padding:0.2rem 0.6rem; border-radius:12px; border:1px solid #A7F3D0;">
                                {len(matched_with_resume)} of {len(jd_skills)} Matched ({round((len(matched_with_resume)/len(jd_skills))*100, 1) if jd_skills else 0}%)
                            </span>
                        </div>
                        <div style="margin-bottom: 0.5rem;">
                            <div style="font-size:0.82rem; font-weight:700; color:#065F46; margin-bottom:0.2rem;">Matched Requirements ({len(matched_with_resume)}):</div>
                            <div>{m_pill_html}</div>
                        </div>
                        <div>
                            <div style="font-size:0.82rem; font-weight:700; color:#92400E; margin-bottom:0.2rem;">Missing Target Requirements ({len(missing_from_resume)}):</div>
                            <div>{mis_pill_html}</div>
                        </div>
                    </div>
                """).strip(),
                unsafe_allow_html=True,
            )
        elif jd_skills:
            all_jd_pill_html = " ".join([f'<span class="skill-pill" style="background:#F1F5F9; color:#334155;">{s}</span>' for s in jd_skills])
            st.markdown(
                f"**Identified Target Job Skills ({len(jd_skills)}):** *(Upload a candidate resume to calculate live matches)*"
            )
            st.markdown(f"<div>{all_jd_pill_html}</div>", unsafe_allow_html=True)
        else:
            st.info("Paste requirements containing technical skills above to enable skill matching.")

        if not jd_input.strip() or not candidate_pool:
            st.info("👆 Please ensure a Job Description and at least one candidate resume are provided above to view the screening results.")
            return

        st.markdown('<div class="section-title">Candidate Screening Leaderboard & Multi-Factor Fit</div>', unsafe_allow_html=True)

        from sklearn.metrics.pairwise import cosine_similarity
        from src.data.preprocess import clean_text
        from src.features.text_features import transform_text

        # Vectorize JD
        jd_clean = clean_text(jd_input)
        jd_vec = transform_text(jd_clean, rec_vec)

        leaderboard_data = []
        for cand_name, cand_text in candidate_pool:
            c_clean = clean_text(cand_text)
            c_vec = transform_text(c_clean, rec_vec)
            sim = float(cosine_similarity(c_vec, jd_vec)[0][0])
            
            # Predict category
            cand_pred = predict_category(cand_text, clf_model, clf_vec, top_k=1)
            cand_cat = cand_pred["predicted_category"]

            # Skills
            cand_skills = extract_skills(cand_text)
            gap = compute_skill_gap(cand_skills, jd_skills)

            # Additional candidate strengths (skills candidate has beyond the target JD requirements)
            cand_set = set(cand_skills)
            jd_set = set(jd_skills)
            matched_reqs = sorted(cand_set.intersection(jd_set))
            missing_reqs = sorted(jd_set.difference(cand_set))
            extra_strengths = sorted(cand_set.difference(jd_set))

            # Explainable multi-factor fit calculation
            fit_report = compute_candidate_fit(
                resume_text=cand_text,
                job_info={"title": jd_input[:100], "skills_list": jd_skills, "description": jd_input},
                predicted_category=cand_cat,
                semantic_similarity=sim,
            )

            # ATS Score
            meta = extract_metadata(cand_text)
            secs = extract_sections(cand_text)
            ats = compute_ats_score(cand_text, cand_skills, meta, secs)

            # Top matching roles from catalog for this candidate
            cand_recommendations = recommend_jobs(
                resume_text=cand_text,
                top_n=3,
                job_vectorizer=rec_vec,
                job_matrix=rec_matrix,
                job_metadata=job_meta,
            )

            leaderboard_data.append({
                "Candidate": cand_name,
                "Match Score": fit_report["fit_score"],
                "Fit Tier": fit_report["fit_tier"],
                "Summary": fit_report["summary"],
                "Breakdown": fit_report["breakdown"],
                "Predicted Role": cand_cat,
                "All Skills": cand_skills,
                "Matched Skills": matched_reqs,
                "Missing Skills": missing_reqs,
                "Extra Strengths": extra_strengths,
                "ATS Score": ats["ats_score"],
                "ATS Tier": ats["tier"],
                "Top Catalog Matches": cand_recommendations,
            })

        leaderboard_data.sort(key=lambda x: x["Match Score"], reverse=True)

        # Leaderboard Cards
        for rank_pos, cand in enumerate(leaderboard_data, 1):
            m_skills = cand["Matched Skills"]
            mis_skills = cand["Missing Skills"]
            extra_skills = cand["Extra Strengths"]
            all_skills = cand["All Skills"]
            bk = cand["Breakdown"]

            m_html = "".join([f'<span class="skill-pill skill-pill-matched">✓ {s}</span>' for s in m_skills]) if m_skills else '<span style="color:#94A3B8; font-size:0.85rem;">None</span>'
            mis_html = "".join([f'<span class="skill-pill skill-pill-missing">+ {s}</span>' for s in mis_skills]) if mis_skills else '<span style="color:#10B981; font-size:0.85rem;">All Job Requirements Matched!</span>'
            extra_html = "".join([f'<span class="skill-pill" style="background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE;">★ {s}</span>' for s in extra_skills]) if extra_skills else '<span style="color:#94A3B8; font-size:0.85rem;">None</span>'
            all_html = "".join([f'<span class="skill-pill">{s}</span>' for s in all_skills]) if all_skills else '<span style="color:#94A3B8; font-size:0.85rem;">None</span>'

            card_html = textwrap.dedent(f"""
                <div class="card">
                    <span class="match-badge">Fit Score: {cand['Match Score']}/100 ({cand['Fit Tier']})</span>
                    <h4 style="margin: 0 0 0.35rem 0; color: #0F172A;">#{rank_pos}. {cand['Candidate']}</h4>
                    <div style="margin-bottom: 0.6rem;">
                        <span class="meta-tag">🎯 Predicted Role: <strong>{cand['Predicted Role']}</strong></span>
                        <span class="meta-tag">📄 ATS Readiness: <strong>{cand['ATS Score']}/100</strong> ({cand['ATS Tier']})</span>
                        <span class="meta-tag">🧠 Skills Coverage: <strong>{bk['skill_match_score']}%</strong></span>
                        <span class="meta-tag">📊 Semantic Match: <strong>{bk['semantic_similarity_score']}%</strong></span>
                    </div>
                    <div style="font-size: 0.88rem; color: #475569; margin-bottom: 0.8rem; background: #F8FAFC; padding: 0.5rem 0.8rem; border-radius: 6px;">
                        ℹ️ <strong>Recruiter Summary:</strong> {cand['Summary']}
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <div style="font-size: 0.82rem; font-weight: 700; color: #065F46; margin-bottom: 0.2rem;">Matched Target Requirements ({len(m_skills)} of {len(jd_skills)}):</div>
                        <div style="margin-bottom: 0.5rem;">{m_html}</div>
                        <div style="font-size: 0.82rem; font-weight: 700; color: #1D4ED8; margin-bottom: 0.2rem;">Additional Candidate Strengths ({len(extra_skills)} skills):</div>
                        <div style="margin-bottom: 0.5rem;">{extra_html}</div>
                        <div style="font-size: 0.82rem; font-weight: 700; color: #92400E; margin-bottom: 0.2rem;">Missing Target Requirements ({len(mis_skills)}):</div>
                        <div style="margin-bottom: 0.5rem;">{mis_html}</div>
                    </div>
                </div>
            """).strip()
            st.markdown(card_html, unsafe_allow_html=True)

            # Expandable preview of all skills and alternative company openings
            with st.expander(f"🔍 Detailed Profile & Top Company Openings for {cand['Candidate']}"):
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.markdown(f"**Complete Verified Skills Inventory ({len(all_skills)}):**")
                    st.markdown(f"<div>{all_html}</div>", unsafe_allow_html=True)
                with exp_col2:
                    st.markdown("**Top Alternative Matching Openings in Catalog:**")
                    for top_m in cand["Top Catalog Matches"]:
                        st.markdown(
                            f"- **{top_m['title']}** ({top_m['company']}) — Match Score: `{top_m['match_score']}/100`"
                        )

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption("SmartHire AI/ML — Classical Machine Learning & ATS Prototype • Built with Scikit-learn, Plotly & Streamlit")


if __name__ == "__main__":
    main()
