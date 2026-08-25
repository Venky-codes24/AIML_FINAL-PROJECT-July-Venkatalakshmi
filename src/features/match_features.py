"""Skill extraction, text normalization, skill-gap analysis, and learning path recommendation."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src import config
from src.features.build_skills_vocab import BASE_SKILLS_CANONICAL, build_and_save_vocab


# Protected compound tokens with slashes/dots that must not be split
PROTECTED_TOKENS = {
    "c/c++": "___TOKEN_C_CPP___",
    "ci/cd": "___TOKEN_CICD___",
    "ui/ux": "___TOKEN_UIUX___",
    "pl/sql": "___TOKEN_PLSQL___",
    "tcp/ip": "___TOKEN_TCPIP___",
    ".net": "___TOKEN_DOTNET___",
    "asp.net": "___TOKEN_ASPDOTNET___",
    "node.js": "___TOKEN_NODEJS___",
    "react.js": "___TOKEN_REACTJS___",
    "vue.js": "___TOKEN_VUEJS___",
    "next.js": "___TOKEN_NEXTJS___",
    "express.js": "___TOKEN_EXPRESSJS___",
    "angular.js": "___TOKEN_ANGULARJS___",
    "d3.js": "___TOKEN_D3JS___",
    "three.js": "___TOKEN_THREEJS___",
    "chart.js": "___TOKEN_CHARTJS___",
    "nuxt.js": "___TOKEN_NUXTJS___",
    "nest.js": "___TOKEN_NESTJS___",
    "py.test": "___TOKEN_PYTEST___",
    "vb.net": "___TOKEN_VBDOTNET___",
}


def unkern_spaced_text(text: str) -> str:
    """De-space text with artificial letter spacing or kerning artifacts."""
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        tokens = stripped.split()
        if len(tokens) >= 2:
            single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
            if single_char_ratio > 0.4:
                words = re.split(r"\s{2,}", stripped)
                fixed_words = []
                for w in words:
                    collapsed = re.sub(r"(?<=\S)\s+(?=\S)", "", w)
                    fixed_words.append(collapsed)
                lines.append(" ".join(fixed_words))
                continue
        lines.append(line)
    return "\n".join(lines)


def normalize_text_for_skills(text: str) -> str:
    """Normalize raw text for high-recall, high-precision skill extraction.

    Handles:
    - Letter spacing / kerning artifacts ('H T M L', 'S p r i n g B o o t')
    - Unicode normalization & ligatures (e.g., 'ﬁ' -> 'fi', 'ﬂ' -> 'fl', 'ﬀ' -> 'ff')
    - Private Use Area Unicode and PDF bullet points (•, ·, ⁃, ◦, ▪, ▫, ★, etc.)
    - Non-breaking spaces, zero-width spaces, and soft hyphens
    - Line-break dehyphenation ('Java-\nScript' -> 'JavaScript')
    - Slash separation for skills lists ('HTML/CSS/JS' -> 'HTML / CSS / JS')
    - Protected token preservation ('C/C++', 'CI/CD', 'Node.js', '.NET', 'C++')
    - Punctuation spacing (colons, commas, parentheses, brackets, quotes)
    - CamelCase / PascalCase token expansion ('SpringBoot' -> 'SpringBoot Spring Boot')
    - Whitespace and newline standardization
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 0. Unkern spaced characters
    text = unkern_spaced_text(text)

    # 1. Unicode decomposition & ligature normalization
    text = unicodedata.normalize("NFKD", text)

    # Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Dehyphenate words split across line breaks in PDFs (e.g. 'Java-\nScript' -> 'JavaScript')
    text = re.sub(r"([A-Za-z0-9]+)-\s*\n\s*([A-Za-z0-9]+)", r"\1\2", text)

    # 3. Strip Private Use Area unicode (common PDF font bullet artifacts like \uf0b7, \uf0a7)
    text = re.sub(r"[\ue000-\uf8ff]", " \n ", text)

    # 4. Replace common PDF visual bullets and artifact characters with spaces/newlines
    artifact_map = {
        "\u00a0": " ",      # non-breaking space
        "\u00ad": "",       # soft hyphen
        "\u200b": " ",      # zero-width space
        "\u200c": " ",      # zero-width non-joiner
        "\u200d": " ",      # zero-width joiner
        "\ufeff": " ",      # zero-width no-break space
        "\u200e": " ",      # left-to-right mark
        "\u200f": " ",      # right-to-left mark
        "\u2022": " \n ",   # bullet •
        "\u2023": " \n ",   # triangular bullet ‣
        "\u2043": " \n ",   # hyphen bullet ⁃
        "\u25cb": " \n ",   # white circle bullet ○
        "\u25cf": " \n ",   # black circle bullet ●
        "\u25aa": " \n ",   # small square bullet ▪
        "\u25ab": " \n ",   # small white square ▫
        "\u25e6": " \n ",   # white bullet ◦
        "\u00b7": " \n ",   # middle dot ·
        "\u2219": " \n ",   # bullet operator ∙
        "\u25b8": " \n ",   # black right-pointing small triangle ▸
        "\u25be": " \n ",   # black down-pointing small triangle ▾
        "\u2713": " \n ",   # check mark ✓
        "\u2714": " \n ",   # heavy check mark ✔
        "\u27a4": " \n ",   # arrowhead ➤
        "\u2605": " \n ",   # black star ★
        "\u2606": " \n ",   # white star ☆
        "\u25c6": " \n ",   # black diamond ◆
        "\u25c7": " \n ",   # white diamond ◇
        "\u25a0": " \n ",   # black square ■
        "\u25a1": " \n ",   # white square □
        "\u2013": "-",      # en dash –
        "\u2014": " - ",    # em dash —
        "\u2015": " - ",    # horizontal bar ―
        "\u2018": "'",      # left single quote ‘
        "\u2019": "'",      # right single quote ’
        "\u201c": '"',      # left double quote “
        "\u201d": '"',      # right double quote ”
        "\t": " ",
        "|": " \n ",
        "~": " ",
        "^": " ",
    }
    for orig, repl in artifact_map.items():
        text = text.replace(orig, repl)

    # 5. Protect compound tokens before general punctuation splitting
    placeholder_reverse = {}
    for token, placeholder in PROTECTED_TOKENS.items():
        # Case insensitive replacement of protected tokens
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        matches = pattern.findall(text)
        for m in matches:
            text = text.replace(m, placeholder)
            placeholder_reverse[placeholder] = m

    # 6. Separate slashes in lists (e.g. 'HTML/CSS/JS' -> 'HTML / CSS / JS')
    text = re.sub(r"(?<=[A-Za-z0-9+#])/(?=[A-Za-z0-9+#])", " / ", text)

    # 7. Space out punctuation delimiters (colons, commas, semicolons, brackets, parens, quotes)
    text = re.sub(r"([:,;()\[\]{}<>\"'?*!])", r" \1 ", text)

    # 8. Restore protected compound tokens
    for placeholder, orig in placeholder_reverse.items():
        text = text.replace(placeholder, f" {orig} ")

    # 9. CamelCase expansion (e.g. 'SpringBoot' -> 'SpringBoot Spring Boot', 'TestNG' -> 'TestNG Test NG')
    def camel_expand(match):
        w = match.group(0)
        # Skip acronyms like AWS, HTML, CSS, SQL, GCP
        if w.isupper() or len(w) <= 2:
            return w
        # Split on lower-to-upper transition
        split_w = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w)
        if split_w != w:
            return f"{w} {split_w}"
        return w

    text = re.sub(r"\b[A-Za-z0-9_+#.-]+\b", camel_expand, text)

    # 10. Collapse consecutive whitespace and normalize lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def extract_sections(text: str) -> Dict[str, str]:
    """Extract standard sections from resume text (Skills, Experience, Projects, Education, Summary)."""
    normalized = normalize_text_for_skills(text)
    section_patterns = {
        "skills": r"(?:^|\n)\s*(?:technical\s+skills|core\s+skills|skills\s*(?:&|and|/)?\s*(?:competencies|expertise|proficiencies|summary|abilities)?|key\s+skills|technologies|tech\s+stack|tools\s*(?:&|and|/)?\s*technologies|programming\s+languages|technical\s+expertise|development\s+skills|areas\s+of\s+expertise|technical\s+proficiencies|skills)\s*[:\-\n]",
        "experience": r"(?:^|\n)\s*(?:work\s+experience|professional\s+experience|employment\s+history|experience|internships?|work\s+history)\s*[:\-\n]",
        "projects": r"(?:^|\n)\s*(?:academic\s+projects|personal\s+projects|key\s+projects|projects|project\s+experience)\s*[:\-\n]",
        "education": r"(?:^|\n)\s*(?:education|academic\s+background|qualifications|academic\s+credentials|educational\s+details)\s*[:\-\n]",
        "summary": r"(?:^|\n)\s*(?:professional\s+summary|career\s+summary|executive\s+summary|summary|profile|about\s+me|career\s+objective)\s*[:\-\n]",
    }

    # Find section header matches and their positions
    matches: List[Tuple[int, str]] = []
    for sec_name, pattern in section_patterns.items():
        for m in re.finditer(pattern, normalized, re.IGNORECASE):
            matches.append((m.start(), sec_name))

    matches.sort(key=lambda x: x[0])

    sections: Dict[str, str] = {}
    for i, (pos, name) in enumerate(matches):
        start_idx = pos
        end_idx = matches[i + 1][0] if i + 1 < len(matches) else len(normalized)
        content = normalized[start_idx:end_idx].strip()
        if name in sections:
            sections[name] += "\n" + content
        else:
            sections[name] = content

    return sections


def load_skills_vocab(path: Union[str, Path] = config.SKILLS_VOCAB_PATH) -> Dict[str, List[str]]:
    """Load canonical skills vocabulary, building it if missing or invalid."""
    global _VOCAB_CACHE, _MATCHERS_CACHE
    path = Path(path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
                if isinstance(vocab, dict) and len(vocab) > 20:
                    _VOCAB_CACHE = vocab
                    _MATCHERS_CACHE = _compile_skill_matcher(vocab)
                    return vocab
        except Exception:
            pass
    vocab = build_and_save_vocab(path)
    _VOCAB_CACHE = vocab
    _MATCHERS_CACHE = _compile_skill_matcher(vocab)
    return vocab


def _compile_skill_matcher(vocab: Dict[str, List[str]]) -> List[Tuple[str, str, re.Pattern]]:
    """Compile optimized regex matchers sorted by alias length (longest match first)."""
    alias_entries: List[Tuple[str, str]] = []
    for canonical, aliases in vocab.items():
        for alias in aliases:
            a_clean = alias.strip().lower()
            if a_clean:
                alias_entries.append((canonical, a_clean))

    # Sort aliases by length descending (longest phrase first)
    alias_entries.sort(key=lambda x: -len(x[1]))

    compiled_matchers: List[Tuple[str, str, re.Pattern]] = []
    for canonical, alias in alias_entries:
        escaped = re.escape(alias)
        # Symbols requiring lookarounds for word boundaries
        if alias in ["c++", "c#", ".net", "ci/cd", "ui/ux", "node.js", "react.js", "vue.js", "next.js", "d3.js", "express.js", "angular.js", "nuxt.js", "nest.js", "py.test", "asp.net"]:
            pattern = re.compile(rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])", re.IGNORECASE)
        elif alias in ["c", "r", "go"]:
            # Strict contextual pattern requirement for single-letter / common word language names
            continue
        elif len(alias) <= 3:
            pattern = re.compile(rf"(?<![a-zA-Z0-9+#.]){escaped}(?![a-zA-Z0-9+#.])", re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        compiled_matchers.append((canonical, alias, pattern))

    return compiled_matchers


# Global precompiled matcher instance
_VOCAB_CACHE: Dict[str, List[str]] = {}
_MATCHERS_CACHE: List[Tuple[str, str, re.Pattern]] = []
_VOCAB_CACHE = load_skills_vocab()


def extract_skills(
    text: str,
    vocab: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Extract canonical skills from free-form text using robust normalization and section weighting.

    Args:
        text: Free-form text (resume or job description).
        vocab: Optional custom skill vocabulary dict mapping canonical name to alias list.

    Returns:
        Sorted list of distinct matched canonical skill names.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    global _VOCAB_CACHE, _MATCHERS_CACHE
    if vocab is not None:
        matchers = _compile_skill_matcher(vocab)
    else:
        if not _MATCHERS_CACHE:
            load_skills_vocab()
        matchers = _MATCHERS_CACHE

    normalized = normalize_text_for_skills(text)
    padded_text = f" {normalized} "
    found_skills: Set[str] = set()

    # 1. First pass: Match across the entire normalized document
    for canonical, alias, pattern in matchers:
        if canonical not in found_skills and pattern.search(padded_text):
            found_skills.add(canonical)

    # 2. Second pass: Check explicitly recognized Skills section (if present) for matches
    sections = extract_sections(text)
    skills_sec = sections.get("skills", "")
    if skills_sec:
        padded_sec = f" {normalize_text_for_skills(skills_sec)} "
        for canonical, alias, pattern in matchers:
            if canonical not in found_skills and pattern.search(padded_sec):
                found_skills.add(canonical)

    # 3. Explicit check for C, R, and Go programming when contextually mentioned
    if "C" not in found_skills:
        c_pattern = re.compile(r"\b(?:c\s*language|c\s*programming|c/c\+\+|embedded\s*c|ansi\s*c)\b", re.IGNORECASE)
        if c_pattern.search(padded_text):
            found_skills.add("C")

    if "R" not in found_skills:
        r_pattern = re.compile(r"\b(?:r\s*language|r\s*programming|rstudio|r\s*shiny|r\s*package|cran|r-project)\b", re.IGNORECASE)
        if r_pattern.search(padded_text):
            found_skills.add("R")

    if "Go" not in found_skills:
        go_pattern = re.compile(r"\b(?:golang|go\s*language|go\s*programming|go\s*developer)\b", re.IGNORECASE)
        if go_pattern.search(padded_text):
            found_skills.add("Go")

    return sorted(found_skills)


def compute_skill_gap(
    resume_skills: Union[List[str], Set[str]],
    job_skills_source: Union[str, List[str], Set[str]],
) -> Dict[str, Any]:
    """Compute matched and missing skills between candidate resume skills and job requirements.

    Args:
        resume_skills: List/set of skills candidate possesses.
        job_skills_source: Either raw job skills string/text or list of job skills.

    Returns:
        Dict containing:
            - 'matched_skills': List[str]
            - 'missing_skills': List[str]
            - 'job_skills': List[str]
            - 'overlap_ratio': float (0.0 to 1.0)
            - 'match_percentage': int (0 to 100)
    """
    resume_set = set(resume_skills)

    if isinstance(job_skills_source, str):
        # Extract canonical skills from the job string/description
        job_set = set(extract_skills(job_skills_source))
        if not job_set:
            # Fallback split if comma-separated
            parts = [s.strip() for s in job_skills_source.split(",") if s.strip()]
            for p in parts:
                extracted = extract_skills(p)
                if extracted:
                    job_set.update(extracted)
                elif len(p) > 1:
                    job_set.add(p.title())
    elif isinstance(job_skills_source, (list, set, tuple)):
        job_set = set()
        for item in job_skills_source:
            ext = extract_skills(str(item))
            if ext:
                job_set.update(ext)
            elif str(item).strip():
                job_set.add(str(item).strip().title())
    else:
        job_set = set()

    matched = sorted(resume_set.intersection(job_set))
    missing = sorted(job_set.difference(resume_set))

    total_required = len(job_set)
    if total_required > 0:
        overlap_ratio = len(matched) / total_required
    else:
        overlap_ratio = 1.0 if len(matched) > 0 else 0.5

    match_percentage = int(round(overlap_ratio * 100))

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "job_skills": sorted(job_set),
        "overlap_ratio": round(overlap_ratio, 3),
        "match_percentage": min(100, max(0, match_percentage)),
    }


def recommend_skills_to_learn(
    resume_skills: List[str],
    recommended_jobs: List[Dict[str, Any]],
    top_k: int = 6,
) -> List[Tuple[str, int, float]]:
    """Prioritize and recommend missing skills based on frequency in recommended job matches.

    Args:
        resume_skills: List of skills the candidate already has.
        recommended_jobs: Top recommended job items returned by recommender.
        top_k: Maximum number of skills to recommend.

    Returns:
        List of tuples: (Skill Name, Frequency Count, Frequency Percentage).
    """
    resume_set = set(resume_skills)
    missing_counter: Counter = Counter()
    total_jobs = max(1, len(recommended_jobs))

    for job in recommended_jobs:
        job_skills = job.get("skills_list") or extract_skills(
            f"{job.get('skills', '')} {job.get('description', '')} {job.get('title', '')}"
        )
        for skill in set(job_skills):
            if skill not in resume_set:
                missing_counter[skill] += 1

    ranked = []
    for skill, count in missing_counter.most_common(top_k):
        pct = round((count / total_jobs) * 100, 1)
        ranked.append((skill, count, pct))

    return ranked


def save_skills_vocab(path: Union[str, Path] = config.SKILLS_VOCAB_PATH) -> None:
    """Save the skill vocabulary mapping as JSON for transparency."""
    build_and_save_vocab(path)
    load_skills_vocab(path)
