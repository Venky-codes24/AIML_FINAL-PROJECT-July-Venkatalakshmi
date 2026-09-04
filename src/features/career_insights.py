"""Career insights, ATS scoring, career roadmaps, and report generation for SmartHire."""

import re
from typing import Any, Dict, List, Tuple


# Standard action verbs commonly favored in ATS resume parsing
ATS_ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "engineered", "created",
    "optimized", "deployed", "automated", "maintained", "integrated", "managed",
    "led", "orchestrated", "architected", "delivered", "configured", "reduced",
    "increased", "scaled", "improved", "refactored", "monitored", "tested",
]

# Tech domains taxonomy for radar chart profiling
TECH_DOMAINS = {
    "Frontend & Web": ["HTML", "CSS", "JavaScript", "TypeScript", "Angular", "React", "Vue.js", "Next.js", "Tailwind CSS", "Bootstrap", "jQuery", "UI/UX Design"],
    "Backend & APIs": ["Java", "Python", "C++", "C#", "C", "Go", "Rust", "Ruby", "PHP", "Spring Boot", ".NET", "Node.js", "Express.js", "Django", "Flask", "FastAPI", "REST APIs", "GraphQL", "Microservices", "Hibernate"],
    "Databases & Storage": ["MySQL", "PostgreSQL", "SQL", "MongoDB", "Redis", "Elasticsearch", "Snowflake", "DynamoDB", "Oracle DB"],
    "QA & Automation": ["Selenium", "Cucumber", "TestNG", "JUnit", "PyTest", "Cypress", "Playwright", "Appium", "Postman", "JMeter", "Software Testing", "Automation Testing"],
    "Cloud & DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins", "Git", "Terraform", "Ansible", "Linux", "Nginx"],
    "Data Science & AI": ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy", "SciPy", "XGBoost", "Statistics", "Data Analysis", "Data Visualization", "Tableau", "Power BI", "Apache Spark", "Hadoop", "Databricks", "MLOps", "Jupyter Notebook"],
}

# Role-specific capstone project recommendations
ROLE_CAPSTONE_PROJECTS = {
    "Java Developer": {
        "title": "Cloud-Native Enterprise Microservices Platform",
        "description": "Design and build a scalable e-commerce or banking backend using Spring Boot, Spring Security JWT, Docker, MySQL/PostgreSQL, and Kafka for async event messaging.",
    },
    "Python Developer": {
        "title": "Async High-Throughput REST & GraphQL API Engine",
        "description": "Develop a high-performance backend using FastAPI or Django REST Framework with Celery task queues, Redis caching, and automated PyTest CI/CD pipelines.",
    },
    "Data Science": {
        "title": "End-to-End MLOps Predictive Intelligence Pipeline",
        "description": "Train and deploy a production Machine Learning model (Scikit-Learn/PyTorch) with automated data preprocessing, FastAPI inference endpoints, Docker containerization, and Streamlit dashboard.",
    },
    "DevOps Engineer": {
        "title": "Multi-Region Kubernetes GitOps & Infrastructure-as-Code",
        "description": "Automate cloud deployment with Terraform on AWS/GCP, containerize microservices in Docker, and set up automated CI/CD deployment with GitHub Actions and Kubernetes (EKS/GKE).",
    },
    "Testing": {
        "title": "Enterprise Hybrid Test Automation Framework",
        "description": "Construct an end-to-end test framework using Selenium WebDriver, TestNG/PyTest, Cucumber BDD, and integrate it with Jenkins CI/CD for automated regression testing and HTML report generation.",
    },
    "Web Designing": {
        "title": "Modern Interactive SaaS Dashboard Web App",
        "description": "Build a responsive, accessible single-page application using React/Next.js or Angular, Tailwind CSS, TypeScript, and state management with RESTful API integration.",
    },
    "Database": {
        "title": "High-Availability Distributed Database & ETL Pipeline",
        "description": "Design an optimized relational and NoSQL database architecture with indexed queries, automated backup procedures, and Apache Spark ETL pipeline for data warehousing.",
    },
    "HR": {
        "title": "Digital HRMS & Talent Analytics Platform",
        "description": "Develop an automated candidate screening and HR metrics analytics dashboard tracking time-to-hire, employee retention, and recruitment pipeline efficiency.",
    },
}


def compute_ats_score(
    raw_text: str,
    skills_found: List[str],
    metadata: Dict[str, Any],
    sections: Dict[str, str],
) -> Dict[str, Any]:
    """Compute a deterministic ATS (Applicant Tracking System) Readiness Score (0-100).
    
    Evaluates:
    - Section Architecture (25 pts): Skills, Experience, Education, Summary
    - Technical Skill Density (30 pts): Number & variety of recognized canonical skills
    - Contact Information (15 pts): Verifiable email & phone
    - Action-Oriented Verbs (15 pts): Strong active engineering verbs
    - Measurable Metrics (15 pts): Numbers, percentages, quantifiable achievements
    """
    score = 0
    checks = []
    tips = []

    # 1. Section Completeness (Max 25 pts)
    sec_count = len(sections)
    if sec_count >= 4:
        score += 25
        checks.append(("Section Structure", "Complete (Skills, Experience, Education, Summary)", True, 25, 25))
    elif sec_count >= 2:
        score += 15
        checks.append(("Section Structure", f"Partial ({sec_count} standard sections detected)", True, 15, 25))
        tips.append("Add explicit standard section headers like 'EXPERIENCE', 'EDUCATION', and 'TECHNICAL SKILLS'.")
    else:
        score += 5
        checks.append(("Section Structure", "Unstructured text layout", False, 5, 25))
        tips.append("Format resume with clear capitalized section titles for ATS parsing compatibility.")

    # 2. Technical Skill Density (Max 30 pts)
    skill_count = len(skills_found)
    if skill_count >= 10:
        score += 30
        checks.append(("Technical Skills", f"Excellent density ({skill_count} canonical skills recognized)", True, 30, 30))
    elif skill_count >= 6:
        score += 22
        checks.append(("Technical Skills", f"Good coverage ({skill_count} skills recognized)", True, 22, 30))
        tips.append("Incorporate additional complementary tools (e.g. Git, Docker, CI/CD) into your skills list.")
    elif skill_count >= 3:
        score += 14
        checks.append(("Technical Skills", f"Moderate coverage ({skill_count} skills recognized)", True, 14, 30))
        tips.append("Expand your skills list with specific frameworks, databases, and testing tools.")
    else:
        score += 5
        checks.append(("Technical Skills", f"Low skill density ({skill_count} skills detected)", False, 5, 30))
        tips.append("Explicitly list technical skills, programming languages, and frameworks used in your work.")

    # 3. Contact Information (Max 15 pts)
    has_email = bool(metadata.get("emails"))
    has_phone = bool(metadata.get("phones"))
    if has_email and has_phone:
        score += 15
        checks.append(("Contact Details", "Email & Phone number detected", True, 15, 15))
    elif has_email or has_phone:
        score += 8
        checks.append(("Contact Details", "Partial contact details detected", True, 8, 15))
        tips.append("Ensure both your professional email address and phone number are clearly visible at the top.")
    else:
        score += 0
        checks.append(("Contact Details", "No email/phone identified", False, 0, 15))
        tips.append("Include valid professional contact details (Email, Phone number, LinkedIn/GitHub URL).")

    # 4. Action Verbs (Max 15 pts)
    text_lower = raw_text.lower()
    verbs_found = [v for v in ATS_ACTION_VERBS if re.search(r"\b" + v + r"\b", text_lower)]
    if len(verbs_found) >= 5:
        score += 15
        checks.append(("Action Verbs", f"Strong verb usage ({len(verbs_found)} active verbs)", True, 15, 15))
    elif len(verbs_found) >= 2:
        score += 10
        checks.append(("Action Verbs", f"Moderate verb usage ({len(verbs_found)} active verbs)", True, 10, 15))
        tips.append("Begin project bullet points with strong action verbs like 'Architected', 'Optimized', 'Automated'.")
    else:
        score += 4
        checks.append(("Action Verbs", "Passive phrasing detected", False, 4, 15))
        tips.append("Use strong action verbs (e.g., Developed, Engineered, Deployed, Reduced, Built) in work descriptions.")

    # 5. Quantifiable Impact & Metrics (Max 15 pts)
    metric_matches = re.findall(r"\b(?:\d+%\s*|\d+\+\s*(?:years?|users?|million|billion|k|gb|tb|req)|reduced\s+by\s+\d+|increased\s+by\s+\d+)\b", text_lower)
    if len(metric_matches) >= 3:
        score += 15
        checks.append(("Measurable Impact", f"Quantifiable metrics found ({len(metric_matches)} instances)", True, 15, 15))
    elif len(metric_matches) >= 1:
        score += 9
        checks.append(("Measurable Impact", "Limited quantifiable metrics", True, 9, 15))
        tips.append("Include numerical metrics in bullet points (e.g. 'boosted performance by 25%', 'handled 10k+ requests').")
    else:
        score += 3
        checks.append(("Measurable Impact", "No metrics found", False, 3, 15))
        tips.append("Quantify your achievements with numbers, percentages, user counts, or latency reductions.")

    score = min(100, max(0, score))

    if score >= 85:
        tier = "Excellent (ATS Ready)"
        color = "#10B981"
    elif score >= 70:
        tier = "Strong Candidate"
        color = "#3B82F6"
    elif score >= 50:
        tier = "Fair / Needs Polish"
        color = "#F59E0B"
    else:
        tier = "Needs Optimization"
        color = "#EF4444"

    return {
        "ats_score": score,
        "tier": tier,
        "color": color,
        "checks": checks,
        "tips": tips[:4] if tips else ["Your resume demonstrates strong ATS compliance and formatting standards."],
    }


def compute_domain_skill_distribution(skills_found: List[str]) -> Dict[str, int]:
    """Calculate candidate skill coverage across 6 core technical domains for radar chart visualization."""
    skills_set = set(skills_found)
    distribution = {}

    for domain_name, domain_skills in TECH_DOMAINS.items():
        matched_count = sum(1 for s in domain_skills if s in skills_set)
        distribution[domain_name] = matched_count

    return distribution


def generate_career_roadmap(
    predicted_role: str,
    missing_skills: List[Tuple[str, int, float]],
    resume_skills: List[str],
) -> List[Dict[str, Any]]:
    """Construct an actionable 3-phase career & skill learning roadmap."""
    # Top missing skills
    missing_names = [s[0] for s in missing_skills] if missing_skills else ["Advanced Architecture", "System Design", "Cloud Native Tools"]
    
    phase1_skills = missing_names[:2] if len(missing_names) >= 2 else (missing_names[:1] or ["Core Fundamentals"])
    phase2_skills = missing_names[2:4] if len(missing_names) >= 4 else (missing_names[1:2] or ["CI/CD & Cloud Deployment"])
    
    capstone = ROLE_CAPSTONE_PROJECTS.get(
        predicted_role,
        {
            "title": f"Production-Grade {predicted_role} Full-Stack Application",
            "description": f"Build a comprehensive portfolio project incorporating your existing skills ({', '.join(resume_skills[:3]) if resume_skills else 'core tools'}) and new target tools with automated testing and deployment.",
        },
    )

    return [
        {
            "phase": "Phase 1: High-Priority Missing Fundamentals",
            "timeline": "Weeks 1 - 3",
            "goal": f"Master immediate high-demand job prerequisites: {', '.join(phase1_skills)}",
            "action_items": [
                f"Complete hands-on tutorials and build working sample apps using {phase1_skills[0] if phase1_skills else 'target libraries'}.",
                "Write clean, unit-tested code implementing standard design patterns.",
                "Publish small focused demonstration repositories on your GitHub profile.",
            ],
        },
        {
            "phase": "Phase 2: Scalability, Cloud & Testing Practices",
            "timeline": "Weeks 4 - 6",
            "goal": f"Integrate production-grade tooling: {', '.join(phase2_skills) if phase2_skills else 'Docker & Automated Testing'}",
            "action_items": [
                f"Implement automated workflows, containerization, or APIs with {phase2_skills[0] if phase2_skills else 'Docker'}.",
                "Write integration and automated unit tests to ensure high test coverage.",
                "Integrate GitHub Actions / CI/CD pipeline for automated builds.",
            ],
        },
        {
            "phase": "Phase 3: Portfolio Capstone Project & Job Applications",
            "timeline": "Weeks 7 - 8",
            "goal": capstone["title"],
            "action_items": [
                capstone["description"],
                "Document architecture in a comprehensive GitHub README with system diagrams and live demo link.",
                "Update your resume bullet points with measurable impact metrics and apply to top matched job listings.",
            ],
        },
    ]


def generate_assessment_report_markdown(
    filename: str,
    predicted_role: str,
    confidence: int,
    skills_found: List[str],
    ats_data: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    skills_to_learn: List[Tuple[str, int, float]],
    roadmap: List[Dict[str, Any]],
) -> str:
    """Generate a clean, structured Markdown Career & Resume Assessment Report for download."""
    skills_str = ", ".join(skills_found) if skills_found else "None detected"
    
    top_jobs_md = ""
    for job in recommendations[:5]:
        top_jobs_md += f"""
### #{job['rank']}. {job['title']} — {job['company']}
- **Match Score:** {job['match_score']}/100
- **Location:** {job['location']} | **Experience:** {job['experience']}
- **Matched Skills:** {', '.join(job['matched_skills']) if job['matched_skills'] else 'None'}
- **Missing Skills:** {', '.join(job['missing_skills']) if job['missing_skills'] else 'All key skills matched!'}
"""

    missing_md = ""
    for rank_idx, (skill, count, pct) in enumerate(skills_to_learn[:6], 1):
        missing_md += f"- **{rank_idx}. {skill}** (In {count} top matched roles, {pct}% occurrence)\n"

    roadmap_md = ""
    for stage in roadmap:
        roadmap_md += f"""
#### {stage['phase']} ({stage['timeline']})
- **Primary Goal:** {stage['goal']}
- **Key Milestones:**
"""
        for item in stage["action_items"]:
            roadmap_md += f"  - {item}\n"

    report = f"""# 📄 SmartHire AI Career & Resume Audit Report
**Source Document:** `{filename}`  
**Report Generated By:** SmartHire ML Engine  
**Predicted Specialization:** **{predicted_role}** ({confidence}% Confidence)  
**ATS Readiness Score:** **{ats_data['ats_score']}/100** ({ats_data['tier']})

---

## 1. Resume Skills Inventory
**Total Skills Identified:** {len(skills_found)}  
**Identified Canonical Skills:**  
`{skills_str}`

---

## 2. ATS Readiness & Optimization Audit
- **Overall ATS Score:** {ats_data['ats_score']}/100
- **Assessment Tier:** {ats_data['tier']}

### Key Actionable Improvement Tips:
"""
    for tip in ats_data["tips"]:
        report += f"- {tip}\n"

    report += f"""
---

## 3. Top Recommended Job Matches
{top_jobs_md}

---

## 4. Priority Skill Gap Analysis
High-demand skills currently missing from your resume:
{missing_md if missing_md else "- You possess all key skills found across top matched jobs!"}

---

## 5. Tailored 8-Week Career Learning Roadmap
{roadmap_md}

---
*Disclaimer: Match scores represent resume-to-job text and skill relevance calculated via TF-IDF cosine similarity. They do not constitute a hiring guarantee.*
"""
    return report.strip()
