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


def render_custom_styles(is_dark: bool = False):
    """Render dynamic CSS tokens supporting smooth Light & Dark Mode transitions and polished buttons."""
    bg_app = "#0B0F19" if is_dark else "#F8FAFC"
    bg_sidebar = "#111827" if is_dark else "#FFFFFF"
    bg_card = "#1E293B" if is_dark else "#FFFFFF"
    border_card = "#334155" if is_dark else "#E2E8F0"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#64748B"
    text_body = "#CBD5E1" if is_dark else "#334155"
    upload_bg = "#151F32" if is_dark else "#F8FAFC"
    upload_border = "#475569" if is_dark else "#CBD5E1"
    banner_bg = "#23150D" if is_dark else "#FFF7ED"
    banner_border = "#7C2D12" if is_dark else "#FFEDD5"
    banner_text = "#FDBA74" if is_dark else "#9A3412"
    theme_btn_bg = "#334155" if is_dark else "#FFFFFF"
    theme_btn_border = "#475569" if is_dark else "#E2E8F0"
    theme_btn_color = "#F8FAFC" if is_dark else "#0F172A"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg-app: {bg_app};
            --bg-card: {bg_card};
            --border-card: {border_card};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --text-body: {text_body};
            --upload-bg: {upload_bg};
            --upload-border: {upload_border};
        }}

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: {text_primary};
        }}

        .stApp {{
            background-color: {bg_app};
            color: {text_primary};
        }}

        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {border_card};
        }}

        /* Titles */
        .title-dark {{
            font-size: 2.3rem;
            font-weight: 800;
            color: {text_primary};
            letter-spacing: -0.5px;
        }}

        .title-orange {{
            font-size: 2.3rem;
            font-weight: 800;
            color: #FF5722;
            letter-spacing: -0.5px;
        }}

        .title-underline-accent {{
            width: 48px;
            height: 4px;
            background: #FF5722;
            border-radius: 2px;
            margin-top: -2px;
        }}

        /* Header Right Cluster (Theme Button + Profile Avatar Compact Spacing) */
        .user-profile-header-compact {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-top: 3px;
        }}

        .user-avatar-circle {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #FF5722;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 1.15rem;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(255, 87, 34, 0.35);
            flex-shrink: 0;
        }}

        .user-welcome-sub {{
            font-size: 0.75rem;
            color: {text_secondary};
            font-weight: 500;
            line-height: 1.1;
        }}

        .user-welcome-name {{
            font-size: 0.95rem;
            color: {text_primary};
            font-weight: 700;
            line-height: 1.2;
        }}

        /* Theme Toggle Round Button */
        div.st-key-theme_toggle_btn > button,
        div[data-testid="stButton"].st-key-theme_toggle_btn button {{
            width: 40px !important;
            height: 40px !important;
            min-width: 40px !important;
            max-width: 40px !important;
            border-radius: 50% !important;
            background-color: {theme_btn_bg} !important;
            border: 1px solid {theme_btn_border} !important;
            color: {theme_btn_color} !important;
            font-size: 1.2rem !important;
            padding: 0 !important;
            margin: 3px 0 0 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.08) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }}
        div.st-key-theme_toggle_btn > button:hover {{
            border-color: #FF5722 !important;
            transform: scale(1.08) !important;
            box-shadow: 0 4px 10px rgba(255, 87, 34, 0.25) !important;
        }}

        /* Feature Pill Badges */
        .feature-pills-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            margin-top: 0.8rem;
        }}

        .feature-pill-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.95rem;
            border-radius: 10px;
            font-size: 0.88rem;
            font-weight: 600;
            transition: transform 0.15s ease;
        }}
        .feature-pill-badge:hover {{
            transform: translateY(-1px);
        }}

        .pill-purple {{
            background-color: {'#2E1065' if is_dark else '#F3E8FF'};
            color: {'#DDD6FE' if is_dark else '#7E22CE'};
            border: 1px solid {'#581C87' if is_dark else '#E9D5FF'};
        }}
        .pill-green {{
            background-color: {'#064E3B' if is_dark else '#DCFCE7'};
            color: {'#A7F3D0' if is_dark else '#15803D'};
            border: 1px solid {'#065F46' if is_dark else '#BBF7D0'};
        }}
        .pill-pink {{
            background-color: {'#831843' if is_dark else '#FCE7F3'};
            color: {'#FBCFE8' if is_dark else '#BE185D'};
            border: 1px solid {'#9D174D' if is_dark else '#FBCFE8'};
        }}
        .pill-orange {{
            background-color: {'#7C2D12' if is_dark else '#FFEDD5'};
            color: {'#FED7AA' if is_dark else '#C2410C'};
            border: 1px solid {'#9A3412' if is_dark else '#FED7AA'};
        }}

        /* Cards */
        .custom-card {{
            background-color: {bg_card};
            border: 1px solid {border_card};
            border-radius: 16px;
            padding: 1.35rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, {'0.25' if is_dark else '0.03'});
            transition: all 0.2s ease-in-out;
        }}
        .custom-card:hover {{
            border-color: {'#64748B' if is_dark else '#CBD5E1'};
            box-shadow: 0 6px 24px -2px rgba(0, 0, 0, {'0.35' if is_dark else '0.06'});
        }}

        .card-header-icon {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }}

        .icon-box-orange {{
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background-color: {'#3F1A0A' if is_dark else '#FFF3E0'};
            color: #FF5722;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            flex-shrink: 0;
        }}
        .icon-box-blue {{
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background-color: {'#0C2B47' if is_dark else '#E0F2FE'};
            color: #0284C7;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            flex-shrink: 0;
        }}
        .icon-box-green {{
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background-color: {'#0A341E' if is_dark else '#DCFCE7'};
            color: #16A34A;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            flex-shrink: 0;
        }}

        .card-title-text {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {text_primary};
            margin: 0;
        }}
        .card-subtitle-text {{
            font-size: 0.84rem;
            color: {text_secondary};
            margin: 0;
        }}

        /* Dashed Upload Visual Box */
        .dashed-upload-visual {{
            border: 2px dashed {upload_border};
            border-radius: 12px;
            background: {upload_bg};
            padding: 1.5rem 1rem;
            text-align: center;
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s;
        }}
        .dashed-upload-visual:hover {{
            border-color: #FF5722;
        }}

        .upload-cloud-icon {{
            font-size: 2rem;
            color: #FF5722;
            margin-bottom: 0.35rem;
        }}
        .upload-main-text {{
            font-size: 0.95rem;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 0.15rem;
        }}
        .upload-sub-text {{
            font-size: 0.82rem;
            color: {text_secondary};
        }}

        /* "What You'll Get" Grid */
        .section-headline {{
            font-size: 1.25rem;
            font-weight: 800;
            color: {text_primary};
            margin: 1.6rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }}

        .what-you-get-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.4rem;
        }}

        .benefit-card {{
            background: {bg_card};
            border: 1px solid {border_card};
            border-radius: 14px;
            padding: 1.15rem 1rem;
            display: flex;
            gap: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,{'0.2' if is_dark else '0.02'});
        }}

        .benefit-icon-wrapper {{
            width: 38px;
            height: 38px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.15rem;
            flex-shrink: 0;
        }}

        .benefit-title {{
            font-size: 0.92rem;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 0.25rem;
        }}

        .benefit-desc {{
            font-size: 0.78rem;
            color: {text_secondary};
            line-height: 1.4;
        }}

        /* Bottom Action Banner */
        .cta-banner-card {{
            background-color: {banner_bg};
            border: 1px solid {banner_border};
            border-radius: 14px;
            padding: 1rem 1.4rem;
            margin-bottom: 2rem;
        }}

        .cta-left-box {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }}

        .cta-lightbulb {{
            font-size: 1.6rem;
            flex-shrink: 0;
        }}

        .cta-text-content {{
            font-size: 0.88rem;
            color: {banner_text};
            font-weight: 500;
        }}

        /* All Standard Buttons: Modern Orange Styling */
        div.stButton > button:not([key="theme_toggle_btn"]) {{
            background-color: #FF5722 !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 0.55rem 1.25rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 8px rgba(255, 87, 34, 0.25) !important;
        }}
        div.stButton > button:not([key="theme_toggle_btn"]):hover {{
            background-color: #F4511E !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(255, 87, 34, 0.4) !important;
            transform: translateY(-1px) !important;
        }}

        /* File Uploader browse button */
        [data-testid="stFileUploader"] section button {{
            background-color: #FF5722 !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 0.45rem 1.1rem !important;
            box-shadow: 0 2px 8px rgba(255, 87, 34, 0.25) !important;
        }}
        [data-testid="stFileUploader"] section button:hover {{
            background-color: #F4511E !important;
        }}

        /* Download Button */
        div[data-testid="stDownloadButton"] > button {{
            background-color: #FF5722 !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 2px 8px rgba(255, 87, 34, 0.25) !important;
        }}
        div[data-testid="stDownloadButton"] > button:hover {{
            background-color: #F4511E !important;
            transform: translateY(-1px) !important;
        }}

        /* Skill Pill */
        .skill-pill {{
            display: inline-block;
            background-color: {'#1E293B' if is_dark else '#F1F5F9'};
            color: {text_primary};
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.22rem 0.6rem;
            border-radius: 8px;
            margin: 0.2rem 0.25rem 0.2rem 0;
            border: 1px solid {border_card};
        }}
        .skill-pill-matched {{
            background-color: {'#064E3B' if is_dark else '#ECFDF5'};
            color: {'#A7F3D0' if is_dark else '#065F46'};
            border: 1px solid {'#059669' if is_dark else '#A7F3D0'};
        }}
        .skill-pill-missing {{
            background-color: {'#451A03' if is_dark else '#FFF7ED'};
            color: {'#FDBA74' if is_dark else '#C2410C'};
            border: 1px solid {'#7C2D12' if is_dark else '#FFD8A8'};
        }}

        .match-badge {{
            display: inline-block;
            background: #FF5722;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            float: right;
            box-shadow: 0 2px 6px rgba(255, 87, 34, 0.3);
        }}

        .stat-box {{
            text-align: center;
            padding: 1.1rem 0.9rem;
            background-color: {bg_card};
            border-radius: 14px;
            border: 1px solid {border_card};
            box-shadow: 0 2px 8px rgba(0,0,0,{'0.2' if is_dark else '0.02'});
        }}
        .stat-number {{
            font-size: 1.55rem;
            font-weight: 800;
            color: #FF5722;
        }}
        .stat-label {{
            font-size: 0.82rem;
            color: {text_secondary};
            font-weight: 600;
            margin-top: 0.2rem;
        }}

        .roadmap-step {{
            background: {bg_card};
            border: 1px solid {border_card};
            border-left: 5px solid #FF5722;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 2px 6px rgba(0,0,0,{'0.2' if is_dark else '0.02'});
        }}

        .sidebar-brand-box {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 0 1rem 0;
            margin-bottom: 0.5rem;
        }}
        .sidebar-brand-icon {{
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: {'#3F1A0A' if is_dark else '#FFF3E0'};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
        }}
        .sidebar-brand-title {{
            font-size: 1.45rem;
            font-weight: 800;
            color: {text_primary};
        }}

        .sidebar-bottom-card {{
            background: {'linear-gradient(135deg, #1E293B 0%, #0F172A 100%)' if is_dark else 'linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%)'};
            border: 1px solid {'#334155' if is_dark else '#FED7AA'};
            border-radius: 16px;
            padding: 1.2rem 1rem;
            text-align: center;
            margin-top: 2rem;
        }}
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


def render_radar_chart(domain_dist: Dict[str, int], is_dark: bool = False) -> go.Figure:
    """Generate interactive Plotly Radar / Spider Chart for skill domain distribution."""
    categories = list(domain_dist.keys())
    values = list(domain_dist.values())

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(255, 87, 34, 0.25)",
            line=dict(color="#FF5722", width=2.5),
            marker=dict(size=7, color="#E64A19"),
            name="Your Skills",
        )
    )

    tick_col = "#94A3B8" if is_dark else "#64748B"
    grid_col = "#334155" if is_dark else "#E2E8F0"
    label_col = "#F8FAFC" if is_dark else "#1E293B"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(4, max(values) + 1)],
                tickfont=dict(size=10, color=tick_col),
                gridcolor=grid_col,
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=label_col, family="Plus Jakarta Sans"),
                gridcolor=grid_col,
            ),
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_market_demand_chart(recommendations: List[Dict[str, Any]], is_dark: bool = False) -> go.Figure:
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

    title_col = "#F8FAFC" if is_dark else "#1E293B"
    grid_col = "#1E293B" if is_dark else "#F1F5F9"
    axis_col = "#94A3B8" if is_dark else "#64748B"

    fig.update_layout(
        title=dict(text="Top Skills Demanded Across Recommended Jobs", font=dict(size=13, color=title_col, family="Plus Jakarta Sans")),
        xaxis=dict(title="Occurrence Count in Matches", tickfont=dict(size=10, color=axis_col), gridcolor=grid_col),
        yaxis=dict(tickfont=dict(size=11, color=title_col)),
        margin=dict(l=20, r=20, t=40, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def main():
    # Initialize session state for persistent theme and resume text
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    if "active_resume_text" not in st.session_state:
        st.session_state["active_resume_text"] = ""
    if "active_source_label" not in st.session_state:
        st.session_state["active_source_label"] = ""
    if "saved_jobs" not in st.session_state:
        st.session_state["saved_jobs"] = []

    is_dark = st.session_state["dark_mode"]
    render_custom_styles(is_dark)

    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-brand-icon">💼</div>
                <span class="sidebar-brand-title">SmartHire</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_selection = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🔍 Career Matching",
                "📄 ATS Analysis",
                "📊 Career Insights",
                "🗺️ Roadmap",
                "🔖 Saved Jobs",
                "👔 Recruiter Screening",
                "⚙️ Settings",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        status = check_model_status()
        if status["classifier"] and status["recommender"]:
            st.success("✅ Models Loaded & Ready")
        else:
            st.warning("⚠️ Model Artifacts Missing")
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    from src.models.train import main as train_main
                    train_main()
                    st.cache_resource.clear()
                    st.rerun()

        # Sidebar illustration matching mockup
        st.markdown(
            f"""
            <div class="sidebar-bottom-card">
                <svg width="130" height="95" viewBox="0 0 130 95" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="65" cy="48" r="40" fill="#FED7AA" fill-opacity="0.35"/>
                    <rect x="25" y="68" width="80" height="5" rx="2.5" fill="#C2410C"/>
                    <rect x="50" y="54" width="30" height="14" rx="2" fill="#1E293B"/>
                    <rect x="53" y="56" width="24" height="10" rx="1" fill="#FF5722"/>
                    <path d="M46 68H84L80 71H50L46 68Z" fill="#64748B"/>
                    <circle cx="65" cy="34" r="10" fill="#EA580C"/>
                    <path d="M53 64C53 50 77 50 77 64H53Z" fill="#F97316"/>
                    <rect x="33" y="61" width="7" height="7" rx="1" fill="#EA580C"/>
                    <path d="M36.5 54C34 56 35 61 36.5 61C38 61 39 56 36.5 54Z" fill="#10B981"/>
                </svg>
                <div style="font-size:0.82rem; font-weight:700; color:{'#FED7AA' if is_dark else '#9A3412'}; margin-top:0.4rem;">AI Career Assistant</div>
                <div style="font-size:0.72rem; color:{'#94A3B8' if is_dark else '#C2410C'};">Empowering your career journey</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Check model files
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
    # VIEW: RECRUITER SCREENING
    # =========================================================================
    if nav_selection == "👔 Recruiter Screening":
        st.markdown(
            """
            <div class="top-header-row">
                <div class="main-title-container">
                    <span class="title-dark">SmartHire</span>
                    <span class="title-orange">Recruiter Hub</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Screen and rank candidate resumes against target Job Descriptions with multi-factor fit scoring.")

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
                "Java, Python, C, JavaScript, HTML, CSS, React.js, Node.js, and SQL databases."
            )
            jd_input = st.text_area("Job Description / Hiring Requirements Text:", value=default_jd_text, height=160, key="recruiter_jd_input")
            jd_skills = extract_skills(jd_input)

        with rec_col2:
            st.markdown("### 2. Candidate Resumes Pool")
            screening_source = st.radio(
                "Candidate Resumes Source:",
                ["📁 Upload Candidate Resumes (PDF, DOCX, TXT)", "🧪 Use Benchmark Demo Candidate Pool (4 Profiles)"],
                key="screening_source_choice",
            )
            candidate_pool: List[Tuple[str, str]] = []
            if "Upload" in screening_source:
                batch_files = st.file_uploader("Upload One or Multiple Resumes:", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="batch_uploader")
                if batch_files:
                    for f in batch_files:
                        try:
                            parsed = parse_resume(f.getvalue(), filename=f.name)
                            candidate_pool.append((f.name, parsed["raw_text"]))
                        except Exception as parse_err:
                            st.warning(f"Skipped '{f.name}': {parse_err}")
            else:
                for label, r_text in DEMO_SAMPLE_RESUMES.items():
                    if r_text.strip():
                        candidate_pool.append((label.split(":")[1].strip() if ":" in label else label, r_text))

        if candidate_pool and jd_input.strip():
            st.markdown("---")
            st.markdown("### 🏆 Ranked Candidate Fit Table")
            from sklearn.metrics.pairwise import cosine_similarity
            from src.data.preprocess import clean_text
            from src.features.text_features import transform_text

            jd_cleaned = clean_text(jd_input)
            jd_vec = transform_text(jd_cleaned, rec_vec)
            scored_candidates = []

            for cand_name, cand_text in candidate_pool:
                cand_cleaned = clean_text(cand_text)
                cand_vec = transform_text(cand_cleaned, rec_vec)
                sim = float(cosine_similarity(cand_vec, jd_vec)[0][0])
                c_skills = extract_skills(cand_text)
                matched_s = [s for s in jd_skills if s in c_skills]
                skill_pct = round((len(matched_s) / len(jd_skills)) * 100, 1) if jd_skills else 0
                composite = round((sim * 50) + (min(100, skill_pct) * 0.5), 1)

                scored_candidates.append({
                    "Candidate": cand_name,
                    "Match Score": composite,
                    "Skill Match (%)": f"{skill_pct}% ({len(matched_s)}/{len(jd_skills)})",
                    "Matched Skills": ", ".join(matched_s) if matched_s else "None",
                    "Extracted Skills Count": len(c_skills),
                })

            scored_candidates = sorted(scored_candidates, key=lambda x: x["Match Score"], reverse=True)
            for idx, c in enumerate(scored_candidates, start=1):
                c["Rank"] = f"#{idx}"

            df_display = pd.DataFrame(scored_candidates)[["Rank", "Candidate", "Match Score", "Skill Match (%)", "Matched Skills", "Extracted Skills Count"]]
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        return

    # =========================================================================
    # VIEW: SETTINGS
    # =========================================================================
    if nav_selection == "⚙️ Settings":
        st.markdown(
            """
            <div class="top-header-row">
                <div class="main-title-container">
                    <span class="title-dark">SmartHire</span>
                    <span class="title-orange">Settings</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### 🎨 Display Preferences")
        new_theme = st.toggle("🌙 Dark Mode Theme", value=is_dark, key="dark_toggle_setting")
        if new_theme != is_dark:
            st.session_state["dark_mode"] = new_theme
            st.rerun()

        st.markdown("### ⚙️ Matching Configuration")
        top_n = st.slider("Number of Top Matches", min_value=3, max_value=15, value=config.TOP_N)
        cat_filter = st.text_input("Filter by Job Title / Keyword", placeholder="e.g. Data Scientist, Java, Engineer")

        st.markdown("---")
        st.markdown("### 🔄 Cache & Engine Management")
        if st.button("Reload Engine & Clear Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading...")
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            **Methodology & Standards:**
            - **Supervised Classifier:** TF-IDF + Logistic Regression
            - **Recommender:** Sparse TF-IDF + Cosine Similarity
            - **Deduplication:** Active
            - **Match Score:** Resume/Job text & skill similarity (not hiring probability)
            """
        )
        return

    # =========================================================================
    # VIEW: SAVED JOBS
    # =========================================================================
    if nav_selection == "🔖 Saved Jobs":
        st.markdown(
            """
            <div class="top-header-row">
                <div class="main-title-container">
                    <span class="title-dark">SmartHire</span>
                    <span class="title-orange">Saved Jobs</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not st.session_state["saved_jobs"]:
            st.info("No saved jobs yet! Recommended jobs will appear here when saved.")
        else:
            for job in st.session_state["saved_jobs"]:
                st.markdown(
                    f"""
                    <div class="custom-card">
                        <span class="match-badge">Match: {job.get('match_score', 'N/A')}/100</span>
                        <h4 style="margin:0 0 0.3rem 0; color:{'#F8FAFC' if is_dark else '#0F172A'};">{job.get('title')}</h4>
                        <div style="font-size:0.85rem; color:{'#94A3B8' if is_dark else '#475569'}; margin-bottom:0.5rem;">
                            🏢 <strong>{job.get('company')}</strong> | 📍 {job.get('location')} | ⏳ Experience: {job.get('experience')}
                        </div>
                        <div style="font-size:0.88rem; color:{'#CBD5E1' if is_dark else '#334155'};">{job.get('description_snippet')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    # =========================================================================
    # MAIN CANDIDATE CAREER HUB HEADER (Tightly Grouped Theme & Profile Buttons)
    # =========================================================================
    head_col_left, head_col_right = st.columns([68, 32])

    with head_col_left:
        st.markdown(
            """
            <div>
                <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                    <span class="title-dark">SmartHire</span>
                    <span class="title-orange">Career Hub</span>
                </div>
                <div class="title-underline-accent"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with head_col_right:
        # Sub-columns to place theme button immediately beside the profile avatar
        r_theme_col, r_user_col = st.columns([1.15, 3.2])
        with r_theme_col:
            theme_icon = "☀️" if is_dark else "🌙"
            theme_tooltip = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
            if st.button(theme_icon, key="theme_toggle_btn", help=theme_tooltip):
                st.session_state["dark_mode"] = not is_dark
                st.rerun()

        with r_user_col:
            st.markdown(
                f"""
                <div class="user-profile-header-compact">
                    <div class="user-avatar-circle">V</div>
                    <div style="display: flex; flex-direction: column;">
                        <span class="user-welcome-sub">Welcome</span>
                        <span class="user-welcome-name">Venky <span style="font-size:0.75rem; color:#64748B;">▾</span></span>
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

    # =========================================================================
    # TWO-COLUMN INPUT SECTION (Matching Mockup)
    # =========================================================================
    col_left, col_right = st.columns([58, 42], gap="large")

    with col_left:
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-header-icon">
                    <div class="icon-box-orange">📤</div>
                    <div>
                        <div class="card-title-text">Upload Your Resume</div>
                        <div class="card-subtitle-text">Supports PDF, DOCX, or TXT files (Max 5MB)</div>
                    </div>
                </div>
                <div class="dashed-upload-visual">
                    <div class="upload-cloud-icon">☁️ ⬆️</div>
                    <div class="upload-main-text">Drag & drop your resume here</div>
                    <div class="upload-sub-text">or click to browse files below</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt"],
            help="Your resume is parsed locally in-memory and is not stored remotely.",
            key="file_uploader_widget",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                parsed_dict = parse_resume(raw_bytes, filename=uploaded_file.name)
                st.session_state["active_resume_text"] = parsed_dict["raw_text"]
                st.session_state["active_source_label"] = f"Uploaded File: {uploaded_file.name}"
                st.success(f"✓ Parsed `{uploaded_file.name}` ({len(parsed_dict['raw_text']):,} characters extracted).")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    with col_right:
        # Card 1: Try a Sample Resume
        st.markdown(
            """
            <div class="custom-card" style="margin-bottom: 1rem; padding: 1.15rem 1.25rem;">
                <div class="card-header-icon" style="margin-bottom: 0.5rem;">
                    <div class="icon-box-blue">📄</div>
                    <div>
                        <div class="card-title-text">Try a Sample Resume</div>
                        <div class="card-subtitle-text">Explore with our demo resume to see how it works.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sample_choice = st.selectbox(
            "Select Demo Sample:",
            options=list(DEMO_SAMPLE_RESUMES.keys()),
            key="sample_select_box",
            label_visibility="collapsed",
        )
        if st.button("🔄 Use Sample Resume", use_container_width=True):
            if sample_choice and DEMO_SAMPLE_RESUMES[sample_choice]:
                st.session_state["active_resume_text"] = DEMO_SAMPLE_RESUMES[sample_choice]
                st.session_state["active_source_label"] = f"Demo Sample: {sample_choice.split(':')[1] if ':' in sample_choice else sample_choice}"
                st.success(f"✓ Loaded {st.session_state['active_source_label']}")
                st.rerun()

        # Card 2: Paste Resume Text
        st.markdown(
            """
            <div class="custom-card" style="margin-top: 1rem; margin-bottom: 0.6rem; padding: 1.15rem 1.25rem;">
                <div class="card-header-icon" style="margin-bottom: 0.4rem;">
                    <div class="icon-box-green">✏️</div>
                    <div>
                        <div class="card-title-text">Paste Resume Text</div>
                        <div class="card-subtitle-text">Or paste your resume content directly.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pasted_content = st.text_area(
            "Paste Text:",
            placeholder="Paste your resume here...",
            height=110,
            key="pasted_text_box",
            label_visibility="collapsed",
        )
        if st.button("✨ Analyze Text", use_container_width=True):
            if pasted_content.strip():
                st.session_state["active_resume_text"] = pasted_content.strip()
                st.session_state["active_source_label"] = "Direct Text Input"
                st.success("✓ Text loaded for analysis!")
                st.rerun()

    # =========================================================================
    # "WHAT YOU'LL GET" 4-CARD SHOWCASE SECTION (Matching Mockup)
    # =========================================================================
    st.markdown(
        """
        <div class="section-headline">
            <span style="color:#FF5722;">✨</span> What You'll Get
        </div>
        <div class="what-you-get-grid">
            <div class="benefit-card">
                <div class="benefit-icon-wrapper" style="background:#F3E8FF; color:#7E22CE;">📄</div>
                <div>
                    <div class="benefit-title">Role Prediction</div>
                    <div class="benefit-desc">AI predicts the best-suited job roles based on your skills and experience.</div>
                </div>
            </div>
            <div class="benefit-card">
                <div class="benefit-icon-wrapper" style="background:#DCFCE7; color:#15803D;">🛡️</div>
                <div>
                    <div class="benefit-title">ATS Score & Optimization</div>
                    <div class="benefit-desc">Get your ATS score and actionable suggestions to improve it.</div>
                </div>
            </div>
            <div class="benefit-card">
                <div class="benefit-icon-wrapper" style="background:#FCE7F3; color:#BE185D;">🎯</div>
                <div>
                    <div class="benefit-title">Job Matching</div>
                    <div class="benefit-desc">Find relevant job opportunities matched to your profile.</div>
                </div>
            </div>
            <div class="benefit-card">
                <div class="benefit-icon-wrapper" style="background:#FEF3C7; color:#B45309;">🗺️</div>
                <div>
                    <div class="benefit-title">Career Roadmap</div>
                    <div class="benefit-desc">Get a personalized learning path and career growth roadmap.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bottom Action / Tip Banner Card with Interactive Get Started Button
    st.markdown('<div class="cta-banner-card">', unsafe_allow_html=True)
    cta_left, cta_right = st.columns([78, 22])
    with cta_left:
        st.markdown(
            """
            <div class="cta-left-box">
                <span class="cta-lightbulb">💡</span>
                <span class="cta-text-content">
                    Ready to accelerate your career? Upload your resume or try a sample above to get AI-powered role matches, ATS scoring, and learning roadmaps.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cta_right:
        if st.button("Get Started →", key="cta_get_started_btn", use_container_width=True):
            if not st.session_state.get("active_resume_text"):
                sample_k = "Demo Sample 1: Full-Stack & Python / C Candidate"
                st.session_state["active_resume_text"] = DEMO_SAMPLE_RESUMES[sample_k]
                st.session_state["active_source_label"] = "Demo Sample: Full-Stack & Python / C Candidate"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # RESUME PROCESSING & ANALYSIS
    # =========================================================================
    resume_raw_text = st.session_state.get("active_resume_text", "").strip()
    source_label = st.session_state.get("active_source_label", "No document active")

    if not resume_raw_text:
        st.info("👆 Upload your resume file or click **'Use Sample Resume'** or **'Get Started →'** above to view full AI career analysis.")
        return

    # Execute ML Pipelines
    with st.spinner("Analyzing resume content, predicting role, and computing job matches..."):
        pred_results = predict_category(resume_raw_text, clf_model, clf_vec, top_k=3)
        predicted_cat = pred_results["predicted_category"]
        confidence_pct = int(round(pred_results["confidence"] * 100))

        skills_found = extract_skills(resume_raw_text)
        metadata = extract_metadata(resume_raw_text)
        sections = extract_sections(resume_raw_text)

        ats_data = compute_ats_score(resume_raw_text, skills_found, metadata, sections)

        recommendations = recommend_jobs(
            resume_text=resume_raw_text,
            top_n=config.TOP_N,
            category_filter=None,
            job_vectorizer=rec_vec,
            job_matrix=rec_matrix,
            job_metadata=job_meta,
        )

        skills_to_learn = recommend_skills_to_learn(skills_found, recommendations, top_k=6)
        domain_dist = compute_domain_skill_distribution(skills_found)
        roadmap = generate_career_roadmap(predicted_cat, skills_to_learn, skills_found)

    # =========================================================================
    # TABS / SPECIFIC NAVIGATION VIEW ROUTING
    # =========================================================================
    st.markdown("---")
    st.caption(f"📌 Active Analysis: **{source_label}**")

    # Overview Metrics Row
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
                <div class="stat-label">ATS Readiness ({ats_data['tier']})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Skills Pill Display
    st.markdown("**Verified Skills Detected in Resume:**")
    if skills_found:
        pill_html = "".join([f'<span class="skill-pill skill-pill-matched">✓ {s}</span>' for s in skills_found])
        st.markdown(f"<div>{pill_html}</div>", unsafe_allow_html=True)
    else:
        st.warning("No standard canonical skills detected in the uploaded text.")

    # Specific view routing based on sidebar selection
    show_all = nav_selection == "🏠 Home"

    # 1. CAREER MATCHING SECTION
    if show_all or nav_selection == "🔍 Career Matching":
        st.markdown('<div class="section-headline"><span style="color:#FF5722;">🎯</span> Top Job Matches</div>', unsafe_allow_html=True)
        st.caption("ℹ️ **Match Score Disclaimer:** Match Score represents resume-to-job text and skill similarity/relevance. It is not a prediction of hiring probability.")

        if not recommendations:
            st.warning("No job recommendations found matching current filters.")
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

                st.markdown(
                    f"""
                    <div class="custom-card">
                        <span class="match-badge">Match: {score}/100</span>
                        <h4 style="margin: 0 0 0.35rem 0; color: {'#F8FAFC' if is_dark else '#0F172A'};">#{rank}. {title}</h4>
                        <div style="margin-bottom: 0.6rem;">
                            <span class="meta-tag">🏢 <strong>{company}</strong></span>
                            <span class="meta-tag">📍 {location}</span>
                            <span class="meta-tag">⏳ Experience: {exp}</span>
                            <span class="meta-tag" style="text-transform: capitalize;">🌐 Source: {source}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: {'#CBD5E1' if is_dark else '#334155'}; margin-bottom: 0.8rem; line-height: 1.45;">
                            {snippet}
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <div style="font-size: 0.82rem; font-weight: 700; color: #065F46; margin-bottom: 0.2rem;">Matched Skills:</div>
                            <div style="margin-bottom: 0.5rem;">{matched_html}</div>
                            <div style="font-size: 0.82rem; font-weight: 700; color: #C2410C; margin-bottom: 0.2rem;">Missing Job Skills:</div>
                            <div>{missing_html}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 2. ATS ANALYSIS SECTION
    if show_all or nav_selection == "📄 ATS Analysis":
        st.markdown('<div class="section-headline"><span style="color:#FF5722;">🛡️</span> ATS Readiness Audit</div>', unsafe_allow_html=True)
        ats_left, ats_right = st.columns([1, 1])
        with ats_left:
            st.markdown(f"### Overall ATS Score: **{ats_data['ats_score']}/100** ({ats_data['tier']})")
            st.markdown("**Evaluation Checklist:**")
            for title, desc, passed, score_pts, max_pts in ats_data["checks"]:
                icon = "✅" if passed else "⚠️"
                st.markdown(f"- {icon} **{title}** ({score_pts}/{max_pts} pts) — {desc}")

        with ats_right:
            st.markdown("### 💡 Actionable Improvement Tips")
            for tip in ats_data["tips"]:
                st.markdown(f"- 📌 {tip}")

        report_content = generate_assessment_report_markdown(
            filename=source_label,
            predicted_role=predicted_cat,
            confidence=confidence_pct,
            skills_found=skills_found,
            ats_data=ats_data,
            recommendations=recommendations,
            skills_to_learn=skills_to_learn,
            roadmap=roadmap,
        )
        st.download_button(
            label="📥 Download Full ATS Audit Report (.md)",
            data=report_content,
            file_name="SmartHire_Career_Report.md",
            mime="text/markdown",
        )

    # 3. CAREER INSIGHTS SECTION (Plotly Charts)
    if show_all or nav_selection == "📊 Career Insights":
        st.markdown('<div class="section-headline"><span style="color:#FF5722;">📊</span> Career Analytics & Skill Profile</div>', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Technical Domain Distribution (Radar Chart)**")
            radar_fig = render_radar_chart(domain_dist, is_dark=is_dark)
            st.plotly_chart(radar_fig, use_container_width=True)
        with chart_col2:
            st.markdown("**Employer Skill Demand Across Matches**")
            demand_fig = render_market_demand_chart(recommendations, is_dark=is_dark)
            st.plotly_chart(demand_fig, use_container_width=True)

    # 4. ROADMAP SECTION
    if show_all or nav_selection == "🗺️ Roadmap":
        st.markdown('<div class="section-headline"><span style="color:#FF5722;">🗺️</span> Tailored 8-Week Career Learning Roadmap</div>', unsafe_allow_html=True)
        st.caption("A structured step-by-step action plan to master your missing skills:")

        for stage in roadmap:
            st.markdown(
                f"""
                <div class="roadmap-step">
                    <div style="font-size: 1.05rem; font-weight: 700; color: {'#F8FAFC' if is_dark else '#0F172A'}; margin-bottom: 0.3rem;">
                        {stage['phase']} <span style="font-size: 0.85rem; color: #FF5722; font-weight: 600;">({stage['timeline']})</span>
                    </div>
                    <div style="font-size: 0.92rem; color: {'#94A3B8' if is_dark else '#475569'}; margin-bottom: 0.5rem;">
                        <strong>Target Focus:</strong> {stage['goal']}
                    </div>
                    <ul style="font-size: 0.88rem; color: {'#CBD5E1' if is_dark else '#334155'}; margin-bottom: 0; padding-left: 1.2rem;">
                        {''.join([f'<li>{item}</li>' for item in stage['action_items']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
