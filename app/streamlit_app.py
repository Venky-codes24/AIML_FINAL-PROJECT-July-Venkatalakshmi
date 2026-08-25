"""SmartHire — AI-Powered Resume & Career Matching Web Application.

Production-Quality Local Application & AI/ML Prototype.
Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure project root is in sys.path for direct module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.features.match_features import (
    compute_skill_gap,
    extract_skills,
    recommend_skills_to_learn,
)
from src.models.classifier import load_classifier, predict_category
from src.models.fit_predictor import compute_candidate_fit
from src.models.recommender import load_recommender, recommend_jobs
from src.parsing.resume_parser import ResumeParsingError, extract_metadata, parse_resume

# Page configuration
st.set_page_config(
    page_title="SmartHire — Career Matching Prototype",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich styling and responsive UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 1.2rem;
    }
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease-in-out;
    }
    .card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
    }
    .match-badge {
        display: inline-block;
        background: #2563EB;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.35rem 0.8rem;
        border-radius: 16px;
        float: right;
    }
    .skill-pill {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        margin: 0.2rem 0.25rem 0.2rem 0;
        border: 1px solid #E2E8F0;
    }
    .skill-pill-matched {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .skill-pill-missing {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .meta-tag {
        font-size: 0.85rem;
        color: #475569;
        margin-right: 1.2rem;
        display: inline-block;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.4rem;
    }
    .stat-box {
        text-align: center;
        padding: 0.9rem;
        background-color: #F8FAFC;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2563EB;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Demo sample resumes strictly for demonstration/testing
DEMO_SAMPLE_RESUMES = {
    "— Select a Demo Sample Resume —": "",
    "Demo Sample 1: Data Scientist / ML Engineer": """
SENIOR DATA SCIENTIST & MACHINE LEARNING ENGINEER (DEMO SAMPLE)
Summary: Experienced Data Scientist with 4+ years of expertise in Statistical Modeling, Machine Learning, Deep Learning, and NLP.
Skills: Python, SQL, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, NLP, BERT, AWS, Docker, Git, Tableau, Statistics.
Experience:
- Developed predictive classification and regression models in Python and Scikit-learn, achieving high accuracy.
- Built end-to-end NLP pipelines using Hugging Face Transformers for text analytics and document classification.
- Containerized and deployed ML microservices on AWS EC2 with FastAPI and Docker.
Education: Bachelor of Technology in Computer Science and Engineering.
Contact: demo.candidate.ml@example.com
    """.strip(),
    "Demo Sample 2: Java Full Stack Developer": """
JAVA FULL STACK SOFTWARE ENGINEER (DEMO SAMPLE)
Summary: Software engineering specialist with expertise in Core Java, Spring Boot, Microservices, and React.
Skills: Java, Spring Boot, Spring MVC, REST APIs, Microservices, React, JavaScript, HTML, CSS, MySQL, PostgreSQL, Docker, Kubernetes, Git, CI/CD, Jenkins.
Experience:
- Designed and maintained scalable RESTful web APIs and microservices using Java 11 and Spring Boot.
- Built responsive client-facing web applications using React.js and JavaScript.
- Integrated PostgreSQL and MySQL databases with Hibernate ORM.
Education: Master of Computer Applications (MCA).
Contact: demo.candidate.java@example.com
    """.strip(),
    "Demo Sample 3: DevOps & Cloud Infrastructure Engineer": """
DEVOPS & CLOUD INFRASTRUCTURE ENGINEER (DEMO SAMPLE)
Summary: Cloud engineer with hands-on experience in AWS Cloud, Kubernetes, Docker, Terraform, and CI/CD pipelines.
Skills: AWS, Linux, Docker, Kubernetes, Terraform, Ansible, CI/CD, Jenkins, GitHub Actions, Python, Bash/Shell, Nginx, Prometheus, Grafana, Git.
Experience:
- Automated multi-region AWS cloud infrastructure using Terraform and CloudFormation.
- Managed Kubernetes (EKS) clusters orchestrating microservices in staging and production.
- Built Jenkins and GitHub Actions CI/CD automation pipelines.
Education: Bachelor of Engineering in Information Technology.
Contact: demo.candidate.devops@example.com
    """.strip(),
    "Demo Sample 4: HR & Talent Acquisition Specialist": """
HR SPECIALIST & TALENT ACQUISITION (DEMO SAMPLE)
Summary: HR professional with experience managing full-cycle recruitment, talent acquisition, onboarding, and employee relations.
Skills: Recruitment & HR, Talent Acquisition, Sourcing, Screening, Onboarding, Payroll, HR Policies, Employee Relations, HRMS, Agile/Scrum.
Experience:
- Led end-to-end talent sourcing and hiring processes for technical and operational roles.
- Managed HR policies, payroll processing, performance management reviews, and compliance.
Education: Master of Business Administration (MBA) in Human Resources.
Contact: demo.candidate.hr@example.com
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


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/briefcase.png", width=56)
    st.sidebar.title("SmartHire")
    st.sidebar.caption("Career Guidance & Job Matching Engine\n*(Local Prototype)*")

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

    # Main Header
    st.markdown('<div class="main-header">SmartHire</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered resume role categorization, job relevance matching, and skill-gap career pathing.</div>',
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

    # Step 1: Input Section
    st.markdown('<div class="section-title">1. Provide Resume</div>', unsafe_allow_html=True)
    
    input_tab1, input_tab2, input_tab3 = st.tabs(["📁 Upload Your Resume", "✍️ Paste Resume Text", "🧪 Try a Demo Sample Resume"])

    resume_raw_text = ""
    source_label = ""

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
            st.info(f"ℹ️ Loaded {source_label} for demonstration.")

    if not resume_raw_text.strip():
        st.info("👆 Please upload a resume file in the 'Upload Your Resume' tab or select a demo sample to begin analysis.")
        return

    # Step 2: Resume Analysis & Classification
    with st.spinner("Analyzing resume content, identifying skills, and computing matches..."):
        # Prediction
        pred_results = predict_category(resume_raw_text, clf_model, clf_vec, top_k=3)
        predicted_cat = pred_results["predicted_category"]
        confidence_pct = int(round(pred_results["confidence"] * 100))

        # Skill extraction
        skills_found = extract_skills(resume_raw_text)
        metadata = extract_metadata(resume_raw_text)

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

    # Step 3: Display Resume Overview
    st.markdown('<div class="section-title">2. Resume Overview & Role Prediction</div>', unsafe_allow_html=True)
    st.caption(f"Active Document: {source_label}")

    stat_col1, stat_col2, stat_col3 = st.columns(3)
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

    # Step 4: Top Job Matches
    st.markdown('<div class="section-title">3. Job Recommendations</div>', unsafe_allow_html=True)
    
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

            card_html = f"""
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
            """
            st.markdown(card_html, unsafe_allow_html=True)

    # Step 5: Skill Gap & Learning Path
    st.markdown('<div class="section-title">4. Skill-Gap & Career Learning Path</div>', unsafe_allow_html=True)
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

    # Step 6: Interactive Rule-Based Career Assistant / Mentor
    st.markdown('<div class="section-title">5. Rule-Based Career Assistant (ML-Driven Mentor)</div>', unsafe_allow_html=True)
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

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption("SmartHire AI/ML — Classical Machine Learning Prototype • Built with Scikit-learn & Streamlit")


if __name__ == "__main__":
    main()
