"""Build and synchronize the canonical skill taxonomy and alias mapping.

Combines:
1. Curated industry-standard technical skills taxonomy with comprehensive aliases/abbreviations.
2. Domain skills discovered from dataset job postings and labeled resumes.

Run:
    python -m src.features.build_skills_vocab
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from src import config

# Curated high-precision canonical skill vocabulary with comprehensive aliases
BASE_SKILLS_CANONICAL: Dict[str, List[str]] = {
    # Web & Frontend Development
    "HTML": ["html", "html5", "dhtml", "xhtml", "html 5"],
    "CSS": ["css", "css3", "sass", "scss", "less", "css 3"],
    "JavaScript": ["javascript", "js", "ecmascript", "java script", "es6", "es7", "es8", "es2015", "es2020", "vanilla js"],
    "TypeScript": ["typescript", "ts"],
    "Angular": ["angular", "angularjs", "angular.js", "angular 2+", "angular 8", "angular 10", "angular 14", "angular 15", "angular 16", "angular 17", "angular 2", "angular 4"],
    "React": ["react", "react.js", "reactjs", "react native", "react-native"],
    "Vue.js": ["vue", "vue.js", "vuejs", "vue 3", "vuex", "nuxt", "nuxt.js"],
    "Next.js": ["next.js", "nextjs"],
    "Tailwind CSS": ["tailwind", "tailwind css", "tailwindcss"],
    "Bootstrap": ["bootstrap", "bootstrap 4", "bootstrap 5"],
    "jQuery": ["jquery"],

    # Programming Languages
    "Python": ["python", "python3", "python2", "cpython", "py3"],
    "Java": ["java", "core java", "java 8", "java 11", "java 17", "java 21", "j2ee", "j2se", "advanced java", "java se", "java ee"],
    "C++": ["c++", "cpp", "c plus plus"],
    "C#": ["c#", "csharp", "c sharp", "c#.net", "c-sharp"],
    "C": ["c language", "c programming", "c/c++", "embedded c", "ansi c"],
    "R": ["r language", "r programming", "rstudio", "r studio", "r shiny", "r package", "r-project", "cran"],
    "Go": ["golang", "go language", "go programming", "go developer"],
    "Rust": ["rust", "rustlang"],
    "Ruby": ["ruby", "ruby on rails", "rails"],
    "PHP": ["php", "laravel", "codeigniter", "symfony"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift", "swiftui"],
    "Scala": ["scala"],
    "Bash/Shell": ["bash", "shell scripting", "shell script", "powershell", "zsh"],

    # Backend Frameworks & Architecture
    "Spring Boot": ["spring boot", "springboot", "spring-boot", "spring framework", "spring mvc", "spring security", "spring cloud", "spring data"],
    ".NET": [".net", "dotnet", "asp.net", "asp.net core", ".net core", "vb.net", ".net framework"],
    "Node.js": ["node.js", "nodejs", "node js", "node"],
    "Express.js": ["express", "express.js", "expressjs"],
    "Django": ["django", "django rest framework", "drf"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "REST APIs": ["rest api", "rest apis", "restful api", "restful apis", "rest", "web apis", "api design", "rest services"],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservice architecture", "distributed systems", "service oriented architecture"],
    "Hibernate": ["hibernate", "hibernate orm", "jpa"],

    # Quality Assurance, Automation & Testing Tools
    "Selenium": ["selenium", "selenium webdriver", "selenium grid", "webdriver", "selenium ide"],
    "Cucumber": ["cucumber", "cucumber bdd", "cucumber testing", "bdd", "cucumber-jvm", "gherkin"],
    "TestNG": ["testng", "test ng", "test-ng", "testng framework"],
    "JUnit": ["junit", "junit 5", "junit 4"],
    "PyTest": ["pytest", "py.test", "unittest"],
    "Cypress": ["cypress", "cypress.io"],
    "Playwright": ["playwright"],
    "Appium": ["appium", "appium mobile testing"],
    "Postman": ["postman", "postman api", "newman"],
    "JMeter": ["jmeter", "apache jmeter", "load testing"],
    "Software Testing": ["qa", "quality assurance", "software testing", "manual testing", "test cases", "test planning", "bug tracking", "system testing", "regression testing", "sanity testing", "functional testing", "uat", "stlc", "sdlc", "black box testing", "white box testing", "test execution", "defect tracking", "test strategy"],
    "Automation Testing": ["automation testing", "test automation", "automated testing", "automation framework", "hybrid framework", "page object model", "pom", "data driven framework", "keyword driven framework", "sdet"],

    # Databases & Storage
    "MySQL": ["mysql", "my sql", "mysql database"],
    "PostgreSQL": ["postgresql", "postgres", "psql", "postgre sql"],
    "SQL": ["sql", "t-sql", "pl/sql", "plsql", "sqlite", "oracle sql", "transact-sql"],
    "MongoDB": ["mongodb", "mongo", "nosql", "documentdb"],
    "Redis": ["redis", "redis cache"],
    "Elasticsearch": ["elasticsearch", "elastic search", "opensearch"],
    "Snowflake": ["snowflake", "snowflake data warehouse"],
    "DynamoDB": ["dynamodb", "aws dynamodb"],
    "Oracle DB": ["oracle database", "oracle db", "oracle 11g", "oracle 12c", "oracle 19c", "oracle rdbms"],

    # Data Science, Machine Learning & AI
    "Machine Learning": ["machine learning", "ml", "statistical learning", "supervised learning", "unsupervised learning"],
    "Deep Learning": ["deep learning", "neural networks", "cnn", "rnn", "lstm", "artificial neural networks", "ann"],
    "Natural Language Processing": ["natural language processing", "nlp", "text mining", "ner", "bert", "transformers", "llm", "large language models", "spacy", "nltk", "huggingface", "hugging face"],
    "Computer Vision": ["computer vision", "opencv", "image processing", "object detection", "yolo"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "Keras": ["keras"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "SciPy": ["scipy"],
    "XGBoost": ["xgboost", "lightgbm", "catboost"],
    "Statistics": ["statistics", "statistical modeling", "hypothesis testing", "regression analysis", "time series"],
    "Data Analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "matplotlib", "seaborn", "plotly", "d3.js"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Apache Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop", "mapreduce", "hive", "hdfs"],
    "Databricks": ["databricks"],
    "MLOps": ["mlops", "model deployment", "mlflow", "kubeflow"],

    # Cloud & DevOps
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "cloudformation", "iam", "cloudwatch", "sqs", "sns"],
    "Azure": ["azure", "microsoft azure", "azure devops", "blob storage", "azure functions"],
    "GCP": ["gcp", "google cloud", "google cloud platform", "bigquery"],
    "Docker": ["docker", "containerization", "containers", "docker-compose"],
    "Kubernetes": ["kubernetes", "k8s", "helm"],
    "CI/CD": ["ci/cd", "ci cd", "cicd", "continuous integration", "continuous deployment"],
    "Jenkins": ["jenkins"],
    "Git": ["git", "github", "gitlab", "bitbucket", "version control"],
    "Terraform": ["terraform", "infrastructure as code", "iac"],
    "Ansible": ["ansible"],
    "Linux": ["linux", "ubuntu", "centos", "redhat", "debian", "unix"],
    "Nginx": ["nginx", "apache web server"],

    # Project Management & Business
    "Agile/Scrum": ["agile", "scrum", "kanban", "sprint planning", "scrum master"],
    "Jira": ["jira", "confluence", "trello"],
    "Project Management": ["project management", "pmp", "stakeholder management", "risk management"],
    "UI/UX Design": ["ui/ux", "ui design", "ux design", "user experience", "user interface", "wireframing", "prototyping", "figma", "adobe xd"],
    "Recruitment & HR": ["recruitment", "talent acquisition", "sourcing", "screening", "onboarding", "payroll", "hr policies", "employee relations", "hr operations"],
    "Digital Marketing": ["digital marketing", "seo", "sem", "google ads", "social media marketing", "content marketing", "email marketing"],
    "Financial Analysis": ["financial analysis", "accounting", "auditing", "taxation", "tally", "financial modeling", "budgeting"],
}


def mine_dataset_skills() -> Dict[str, List[str]]:
    """Mine potential domain skills from dataset job descriptions and resume texts."""
    mined_skills: Dict[str, List[str]] = {}

    files_to_check = [
        config.JOBS_CLEAN_CSV,
        config.RESUMES_CLEAN_CSV,
        config.NAUKRI_CSV,
        config.RESUME_CSV,
    ]

    for file_path in files_to_check:
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path, nrows=500)
            text_cols = [c for c in ["skills", "Resume", "text", "description"] if c in df.columns]
            for col in text_cols:
                series = df[col].dropna().astype(str)
                for val in series:
                    tokens = re.split(r"[,|\n/]+", val)
                    for tok in tokens:
                        tok_clean = tok.strip()
                        if 3 <= len(tok_clean) <= 30 and re.match(r"^[A-Za-z0-9+#. -]+$", tok_clean):
                            lower = tok_clean.lower()
                            if lower in ["experience", "years", "management", "skills", "team", "work", "responsibilities"]:
                                continue
                            canonical_candidate = tok_clean.title()
                            if canonical_candidate not in BASE_SKILLS_CANONICAL and canonical_candidate not in mined_skills:
                                if any(kw in lower for kw in ["developer", "engineer", "framework", "database", "analytics", "design", "security"]):
                                    mined_skills[canonical_candidate] = [lower]
        except Exception:
            continue

    return mined_skills


def build_and_save_vocab(
    output_path: Path = config.SKILLS_VOCAB_PATH,
    min_alias_len: int = 2,
) -> Dict[str, List[str]]:
    """Generate, validate, and save the reproducible canonical skill vocabulary.

    Combines curated taxonomy with discovered dataset skills, deduplicates, and saves JSON.
    """
    final_vocab: Dict[str, List[str]] = {}

    # 1. Load curated base skills
    for canonical, aliases in BASE_SKILLS_CANONICAL.items():
        clean_aliases = set()
        for a in aliases:
            a_clean = a.strip().lower()
            if len(a_clean) >= min_alias_len or a_clean in ["c", "r"]:
                clean_aliases.add(a_clean)
        # Ensure canonical lower is in aliases
        clean_aliases.add(canonical.strip().lower())
        final_vocab[canonical] = sorted(clean_aliases, key=lambda x: -len(x))

    # 2. Integrate mined dataset domain skills
    try:
        mined = mine_dataset_skills()
        for canonical, aliases in mined.items():
            if canonical not in final_vocab:
                clean_aliases = set()
                for a in aliases:
                    a_clean = a.strip().lower()
                    if len(a_clean) >= min_alias_len:
                        clean_aliases.add(a_clean)
                clean_aliases.add(canonical.strip().lower())
                final_vocab[canonical] = sorted(clean_aliases, key=lambda x: -len(x))
    except Exception:
        pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_vocab, f, indent=2)

    print(f"Generated vocabulary with {len(final_vocab)} canonical skills and saved to {output_path}")
    return final_vocab


if __name__ == "__main__":
    build_and_save_vocab()
