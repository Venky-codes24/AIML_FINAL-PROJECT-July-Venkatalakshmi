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
    "CSS": ["css", "css3", "sass", "scss", "css 3", "less"],
    "JavaScript": ["javascript", "js", "ecmascript", "java script", "es6", "es7", "es8", "es2015", "es2020", "vanilla js"],
    "TypeScript": ["typescript", "ts", "type script"],
    "Angular": ["angular", "angularjs", "angular.js", "angular 2+", "angular 8", "angular 10", "angular 14", "angular 15", "angular 16", "angular 17", "angular 2", "angular 4"],
    "React": ["react", "react.js", "reactjs", "react native", "react-native"],
    "Vue.js": ["vue", "vue.js", "vuejs", "vue 3", "vuex", "nuxt", "nuxt.js"],
    "Next.js": ["next.js", "nextjs"],
    "Svelte": ["svelte", "sveltekit", "svelte.js"],
    "Tailwind CSS": ["tailwind", "tailwind css", "tailwindcss"],
    "Bootstrap": ["bootstrap", "bootstrap 4", "bootstrap 5"],
    "Redux": ["redux", "redux toolkit", "rtk", "mobx", "zustand"],
    "Webpack": ["webpack", "vite", "rollup", "babel", "parcel"],
    "GraphQL": ["graphql", "graph ql", "apollo client", "relay"],
    "WebSockets": ["websockets", "websocket", "socket.io"],
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
    "Kotlin": ["kotlin", "android kotlin"],
    "Swift": ["swift", "swiftui", "ios swift"],
    "Scala": ["scala"],
    "Dart": ["dart", "flutter dart"],
    "MATLAB": ["matlab", "simulink"],
    "Bash/Shell": ["bash", "shell scripting", "shell script", "powershell", "zsh", "unix shell"],

    # Backend Frameworks & Architecture
    "Spring Boot": ["spring boot", "springboot", "spring-boot", "spring framework", "spring mvc", "spring security", "spring cloud", "spring data"],
    ".NET": [".net", "dotnet", "asp.net", "asp.net core", ".net core", "vb.net", ".net framework"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express", "express.js", "expressjs"],
    "NestJS": ["nestjs", "nest.js", "nest js"],
    "Django": ["django", "django rest framework", "drf"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "REST APIs": ["rest api", "rest apis", "restful api", "restful apis", "restful", "rest web services", "restful web services", "api design", "rest"],
    "Microservices": ["microservices", "microservice", "microservice architecture", "distributed systems"],
    "Hibernate": ["hibernate", "hibernate orm", "jpa", "spring data jpa"],
    "Kafka": ["kafka", "apache kafka"],
    "RabbitMQ": ["rabbitmq", "rabbit mq", "activemq"],
    "gRPC": ["grpc", "protocol buffers", "protobuf"],
    "Celery": ["celery", "celery task queue"],
    "Maven": ["maven", "apache maven"],
    "Gradle": ["gradle"],

    # Quality Assurance, Automation & Testing Tools
    "Selenium": ["selenium", "selenium webdriver", "selenium grid", "webdriver", "selenium ide"],
    "Cucumber": ["cucumber", "cucumber bdd", "cucumber testing", "cucumber-jvm", "gherkin"],
    "TestNG": ["testng", "test ng", "test-ng", "testng framework"],
    "JUnit": ["junit", "junit 5", "junit 4"],
    "PyTest": ["pytest", "py.test", "unittest"],
    "Cypress": ["cypress", "cypress.io"],
    "Playwright": ["playwright"],
    "Appium": ["appium", "appium mobile testing"],
    "Postman": ["postman", "postman api", "newman"],
    "RestAssured": ["restassured", "rest-assured", "rest assured"],
    "JMeter": ["jmeter", "apache jmeter", "load testing", "locust"],
    "Software Testing": ["software testing", "manual testing", "functional testing", "regression testing", "sanity testing", "system testing", "performance testing", "test planning", "test cases", "black box testing", "white box testing", "quality assurance", "qa engineer"],
    "Automation Testing": ["automation testing", "test automation", "automated testing", "automation framework", "hybrid framework", "data driven framework", "keyword driven framework"],

    # Databases & Storage
    "MySQL": ["mysql", "my sql", "mysql database"],
    "PostgreSQL": ["postgresql", "postgres", "psql", "postgre sql", "pgsql"],
    "SQL": ["sql", "t-sql", "pl/sql", "plsql", "sqlite", "oracle sql", "transact-sql"],
    "MongoDB": ["mongodb", "mongo db", "mongo", "nosql", "documentdb"],
    "Redis": ["redis", "redis cache"],
    "Elasticsearch": ["elasticsearch", "elastic search", "opensearch"],
    "Snowflake": ["snowflake", "snowflake data warehouse"],
    "DynamoDB": ["dynamodb", "dynamo db", "aws dynamodb"],
    "Oracle DB": ["oracle database", "oracle db", "oracle 11g", "oracle 12c", "oracle 19c", "oracle rdbms"],
    "Cassandra": ["cassandra", "apache cassandra", "scylladb"],
    "Neo4j": ["neo4j", "graph database", "cypher query"],
    "Firebase": ["firebase", "firestore", "cloud firestore"],
    "Supabase": ["supabase"],

    # Generative AI, LLMs & Modern AI
    "Generative AI": ["generative ai", "genai", "gen ai", "synthetic data", "diffusion models"],
    "Large Language Models (LLMs)": ["large language models", "llm", "llms", "gpt-4", "chatgpt", "llama", "claude", "gemini", "mistral", "prompt engineering"],
    "LangChain": ["langchain", "llamaindex", "llama-index", "crewai", "autogen"],
    "RAG (Retrieval-Augmented Generation)": ["rag", "retrieval augmented generation", "retrieval-augmented generation", "vector search", "hybrid search"],
    "Vector Databases": ["vector database", "vector db", "pinecone", "chromadb", "chroma", "weaviate", "qdrant", "milvus", "faiss"],
    "Hugging Face": ["huggingface", "hugging face", "transformers library", "diffusers"],

    # Data Science, Machine Learning & AI
    "Machine Learning": ["machine learning", "ml", "statistical learning", "supervised learning", "unsupervised learning", "reinforcement learning"],
    "Deep Learning": ["deep learning", "neural networks", "cnn", "rnn", "lstm", "artificial neural networks", "transformers"],
    "Natural Language Processing": ["natural language processing", "nlp", "text mining", "bert", "spacy", "nltk", "named entity recognition", "sentiment analysis"],
    "Computer Vision": ["computer vision", "opencv", "open cv", "image processing", "object detection", "yolo", "segmentation"],
    "TensorFlow": ["tensorflow", "tensor flow"],
    "PyTorch": ["pytorch", "py torch"],
    "Keras": ["keras"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "SciPy": ["scipy"],
    "XGBoost": ["xgboost", "lightgbm", "catboost"],
    "Statistics": ["statistics", "statistical modeling", "hypothesis testing", "regression analysis", "time series", "a/b testing"],
    "Data Analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "matplotlib", "seaborn", "plotly", "d3.js"],
    "Tableau": ["tableau", "tableau desktop"],
    "Power BI": ["power bi", "powerbi", "dax"],
    "Apache Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop", "mapreduce", "hive", "hdfs"],
    "Databricks": ["databricks", "delta lake"],
    "Apache Airflow": ["airflow", "apache airflow", "data pipelines", "etl pipelines", "etl", "elt"],
    "dbt": ["dbt", "data build tool"],
    "MLOps": ["mlops", "model deployment", "mlflow", "kubeflow", "wandb", "dvc"],

    # Cloud & DevOps
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "cloudformation", "cloudwatch", "aws iam", "aws sqs", "aws sns", "ecs", "eks"],
    "Azure": ["azure", "microsoft azure", "azure devops", "blob storage", "azure functions", "aks"],
    "GCP": ["gcp", "google cloud", "google cloud platform", "bigquery", "gke", "cloud run"],
    "Docker": ["docker", "containerization", "containers", "docker-compose"],
    "Kubernetes": ["kubernetes", "k8s", "helm"],
    "CI/CD": ["ci/cd", "ci cd", "cicd", "continuous integration", "continuous deployment"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions", "gitlab ci", "gitlab ci/cd", "circleci", "travis ci"],
    "Git": ["git", "github", "gitlab", "bitbucket", "version control"],
    "Terraform": ["terraform", "infrastructure as code", "iac"],
    "Ansible": ["ansible"],
    "Linux": ["linux", "ubuntu", "centos", "redhat", "debian", "unix"],
    "Nginx": ["nginx", "apache web server"],
    "Monitoring & Observability": ["prometheus", "grafana", "datadog", "new relic", "splunk", "elk stack", "opentelemetry"],

    # Mobile Development
    "Mobile App Development": ["mobile app development", "mobile application", "android app", "ios app"],
    "Flutter": ["flutter", "dart framework"],
    "React Native": ["react native", "react-native", "expo"],

    # Software Engineering & Computer Science Fundamentals
    "Data Structures & Algorithms": ["data structures", "algorithms", "dsa", "leetcode", "problem solving"],
    "Object-Oriented Programming (OOP)": ["oop", "oops", "object oriented programming", "solid principles", "design patterns"],
    "System Design": ["system design", "distributed systems design", "high level design", "low level design", "hld", "lld"],

    # Cybersecurity
    "Cybersecurity": ["cybersecurity", "cyber security", "network security", "penetration testing", "ethical hacking", "vulnerability assessment", "owasp", "siem", "soc", "cryptography", "zero trust", "firewalls"],

    # Developer Tools & IDEs
    "VS Code": ["vs code", "vscode", "visual studio code"],
    "IntelliJ IDEA": ["intellij", "intellij idea", "pycharm", "webstorm"],
    "Eclipse": ["eclipse", "eclipse ide"],
    "Jupyter Notebook": ["jupyter notebook", "jupyter", "jupyterlab", "jupyter lab"],

    # Project Management & Business
    "Agile/Scrum": ["agile", "scrum", "kanban", "sprint planning", "scrum master"],
    "Jira": ["jira", "confluence", "trello"],
    "Project Management": ["project management", "pmp certified", "pmp certification", "stakeholder management", "risk management"],
    "UI/UX Design": ["ui/ux", "ui design", "ux design", "user experience", "user interface", "wireframing", "prototyping", "figma", "adobe xd"],
    "Recruitment & HR": ["recruitment", "talent acquisition", "sourcing", "screening", "onboarding", "payroll", "hr policies", "employee relations", "hr operations"],
    "Digital Marketing": ["digital marketing", "seo", "search engine marketing", "google ads", "social media marketing", "content marketing", "email marketing"],
    "Financial Analysis": ["financial analysis", "accounting", "auditing", "taxation", "tally", "financial modeling", "budgeting"],
}


def build_and_save_vocab(
    output_path: Path = config.SKILLS_VOCAB_PATH,
    min_alias_len: int = 2,
) -> Dict[str, List[str]]:
    """Generate, validate, and save the reproducible canonical skill vocabulary.

    Uses curated high-precision taxonomy, deduplicates aliases, and saves JSON.
    """
    final_vocab: Dict[str, List[str]] = {}

    # Load curated base skills
    for canonical, aliases in BASE_SKILLS_CANONICAL.items():
        clean_aliases = set()
        for a in aliases:
            a_clean = a.strip().lower()
            if len(a_clean) >= min_alias_len or a_clean in ["c", "r"]:
                clean_aliases.add(a_clean)
        # Ensure canonical lower is in aliases
        clean_aliases.add(canonical.strip().lower())
        final_vocab[canonical] = sorted(clean_aliases, key=lambda x: -len(x))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_vocab, f, indent=2)

    print(f"Generated vocabulary with {len(final_vocab)} canonical skills and saved to {output_path}")
    return final_vocab


if __name__ == "__main__":
    build_and_save_vocab()

