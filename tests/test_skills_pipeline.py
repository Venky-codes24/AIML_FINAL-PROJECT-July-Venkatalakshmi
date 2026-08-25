"""Comprehensive regression & unit tests for SmartHire skill extraction and normalization pipeline."""

import io
import pytest
from src.features.match_features import (
    extract_sections,
    extract_skills,
    normalize_text_for_skills,
    compute_skill_gap,
    recommend_skills_to_learn,
    load_skills_vocab,
)
from src.features.build_skills_vocab import build_and_save_vocab
from src.parsing.resume_parser import parse_resume


def test_user_resume_exact_11_skills_regression():
    """Test against the exact resume skills reported by user:
    HTML, CSS, JavaScript, Java, Python, Angular, SpringBoot, MySQL, Selenium, Cucumber, TestNG.
    """
    resume_text = """
    JOHN DOE — SOFTWARE DEVELOPMENT ENGINEER IN TEST (SDET)
    Email: john.doe@example.com | Phone: +1 555-0199
    
    PROFESSIONAL SUMMARY:
    Experienced test automation engineer with 4+ years of experience designing robust test frameworks.
    
    TECHNICAL SKILLS:
    • Web Technologies: HTML, CSS, JavaScript, Angular
    • Programming Languages: Java, Python
    • Backend Frameworks: SpringBoot
    • Databases: MySQL
    • Automation & Testing: Selenium, Cucumber, TestNG
    
    WORK EXPERIENCE:
    SDET at Tech Corp (2021 - Present)
    - Developed hybrid test automation frameworks using Selenium WebDriver, TestNG, and Cucumber BDD.
    - Integrated SpringBoot microservices with MySQL backend databases.
    - Built frontend dashboards using Angular and JavaScript.
    
    EDUCATION:
    B.Tech in Computer Science and Engineering
    """
    
    detected = extract_skills(resume_text)
    
    # Must detect all 11 required skills
    expected_skills = [
        "HTML",
        "CSS",
        "JavaScript",
        "Java",
        "Python",
        "Angular",
        "Spring Boot",
        "MySQL",
        "Selenium",
        "Cucumber",
        "TestNG",
    ]
    
    for skill in expected_skills:
        assert skill in detected, f"Expected '{skill}' to be detected in resume, but got: {detected}"


def test_pdf_font_bullet_artifacts_and_slashes():
    """Test that PDF extraction artifacts (PUA bullets like \uf0b7, slashes, attached tokens) are cleaned and matched."""
    pdf_extracted_text = (
        "\uf0b7HTML/CSS/JavaScript\n"
        "\uf0b7Java/Python\n"
        "\uf0b7SpringBoot microservices\n"
        "\uf0b7MySQL database\n"
        "\uf0b7Selenium, Cucumber, TestNG automation frameworks\n"
        "\uf0b7AngularJS frontend development"
    )
    skills = extract_skills(pdf_extracted_text)
    
    expected = ["HTML", "CSS", "JavaScript", "Java", "Python", "Spring Boot", "MySQL", "Selenium", "Cucumber", "TestNG", "Angular"]
    for s in expected:
        assert s in skills, f"Failed to extract '{s}' from PDF artifact text. Extracted: {skills}"


def test_text_normalization_unicode_and_artifacts():
    """Test ligature normalization, bullet replacement, line-break dehyphenation, and Unicode cleanup."""
    raw = "•\tPython\u00a0programming\n▪\tﬁle\u00a0handling\n⁃\tSpringBoot\n–\tCI/CD\nJava-\nScript\nSpring-\nBoot"
    normalized = normalize_text_for_skills(raw)
    
    assert "Python" in normalized
    assert "Spring Boot" in normalized or "SpringBoot" in normalized
    assert "CI/CD" in normalized
    assert "JavaScript" in normalized


def test_case_insensitive_and_alias_matching():
    """Test case insensitivity and common alias variations."""
    sample_text = "Experienced in SPRING BOOT, react.js, angularjs, postgres, and nodejs."
    skills = extract_skills(sample_text)
    
    assert "Spring Boot" in skills
    assert "React" in skills
    assert "Angular" in skills
    assert "PostgreSQL" in skills
    assert "Node.js" in skills


def test_symbol_based_skills():
    """Test symbols like C++, C#, .NET, CI/CD, UI/UX."""
    text = "Core skills include C++, C#, .NET Core, CI/CD automation, and UI/UX design."
    skills = extract_skills(text)
    
    assert "C++" in skills
    assert "C#" in skills
    assert ".NET" in skills
    assert "CI/CD" in skills
    assert "UI/UX Design" in skills


def test_false_positive_prevention():
    """Ensure single letters like 'r' and 'c' or English words like 'go' are not falsely extracted."""
    text = "John C. Smith evaluated section r. We go to market with high performance."
    skills = extract_skills(text)
    
    assert "C" not in skills
    assert "R" not in skills
    assert "Go" not in skills


def test_true_positive_for_single_letter_languages():
    """Ensure C, R, and Go are detected when mentioned in programming context."""
    text = "Experienced in C programming, R language statistical modeling, and Golang backend development."
    skills = extract_skills(text)
    
    assert "C" in skills
    assert "R" in skills
    assert "Go" in skills


def test_multi_word_skills_matching():
    """Test extraction of multi-word technical skills."""
    text = "Specialist in Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, Software Testing, and Test Automation."
    skills = extract_skills(text)
    
    assert "Machine Learning" in skills
    assert "Deep Learning" in skills
    assert "Natural Language Processing" in skills
    assert "Computer Vision" in skills
    assert "Software Testing" in skills
    assert "Automation Testing" in skills


def test_section_aware_extraction():
    """Test section detection and extraction from explicit Skills sections."""
    text = """
    EXPERIENCE
    Worked at ABC Corp.
    
    TECHNICAL SKILLS:
    Python, Docker, Kubernetes, AWS
    
    EDUCATION
    B.S. in Computer Science
    """
    sections = extract_sections(text)
    assert "skills" in sections
    assert "experience" in sections
    assert "education" in sections
    
    skills = extract_skills(text)
    assert "Python" in skills
    assert "Docker" in skills
    assert "Kubernetes" in skills
    assert "AWS" in skills


def test_end_to_end_parsing_and_skill_extraction():
    """Test resume parsing from in-memory text buffer into skill extraction."""
    raw_content = b"Skills: Java, Selenium, Cucumber, TestNG, HTML, CSS, JavaScript, MySQL, SpringBoot, Python, Angular"
    parsed = parse_resume(raw_content, filename="resume.txt")
    
    assert len(parsed["raw_text"]) > 0
    skills = extract_skills(parsed["raw_text"])
    
    assert len(skills) >= 11
    assert "Selenium" in skills
    assert "Spring Boot" in skills
    assert "Angular" in skills


def test_skill_gap_with_extracted_skills():
    """Test that extracted skills integrate cleanly with skill-gap computation."""
    resume_skills = ["Java", "Selenium", "TestNG", "Cucumber", "MySQL"]
    job_reqs = "Requirements: Java, Spring Boot, MySQL, Docker, Kubernetes, Selenium"
    
    gap = compute_skill_gap(resume_skills, job_reqs)
    
    assert "Java" in gap["matched_skills"]
    assert "Selenium" in gap["matched_skills"]
    assert "MySQL" in gap["matched_skills"]
    assert "Spring Boot" in gap["missing_skills"]
    assert "Docker" in gap["missing_skills"]
    assert gap["overlap_ratio"] > 0


def test_reproducible_vocab_generation(tmp_path):
    """Test that vocabulary generation is reproducible and outputs valid canonical structure."""
    vocab_file = tmp_path / "test_skills_vocab.json"
    vocab = build_and_save_vocab(vocab_file)
    
    assert isinstance(vocab, dict)
    assert "HTML" in vocab
    assert "Java" in vocab
    assert "Spring Boot" in vocab
    assert "Selenium" in vocab
    assert "Cucumber" in vocab
    assert "TestNG" in vocab
    assert len(vocab) >= 90
    assert vocab_file.exists()


def test_letter_spaced_pdf_text_extraction_and_skills():
    """Test regression on real-world letter-spaced PDF text (e.g., 'H T M L', 'S p r i n g B o o t', 'T e s t N G')."""
    letter_spaced_raw = """
    A B O U T  M E
    I  a m  a  B . T e c h  g r a d u a t e  i n  C o m p u t e r  S c i e n c e  a n d  E n g i n e e r i n g  w i t h  a  s t r o n g  i n t e r e s t  i n  s o f t w a r e  d e v e l o p m e n t  a n d  t e s t i n g .
    
    S K I L L S
    H T M L
    M y S Q L
    J a v a
    C S S
    J a v a S c r i p t
    S p r i n g B o o t
    P y t h o n
    S e l e n i u m
    T e s t N G
    C u c u m b e r
    A n g u l a r
    """
    
    skills = extract_skills(letter_spaced_raw)
    
    expected_skills = [
        "HTML",
        "CSS",
        "JavaScript",
        "Java",
        "Python",
        "Angular",
        "Spring Boot",
        "MySQL",
        "Selenium",
        "Cucumber",
        "TestNG",
    ]
    for s in expected_skills:
        assert s in skills, f"Failed to extract '{s}' from letter-spaced text. Got: {skills}"

