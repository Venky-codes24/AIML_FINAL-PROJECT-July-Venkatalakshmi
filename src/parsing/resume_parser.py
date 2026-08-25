"""Extract and clean raw text from uploaded resumes (PDF / DOCX / TXT).

Provides a unified `parse_resume` function with error handling for corrupted,
empty, or unsupported file formats.
"""

import io
import re
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

import docx
import pypdf

from src.data.preprocess import clean_text


class ResumeParsingError(Exception):
    """Custom exception raised when resume parsing fails with a user-friendly message."""
    pass


def unkern_spaced_text(text: str) -> str:
    """De-space text with artificial letter spacing or kerning artifacts.

    Examples:
        'S p r i n g B o o t' -> 'SpringBoot'
        'H T M L  C S S' -> 'HTML CSS'
        'I  a m  a  B . T e c h  g r a d u a t e' -> 'I am a B.Tech graduate'
    """
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


def extract_text_from_pdf(file_source: Union[str, Path, BinaryIO, bytes]) -> str:
    """Extract text from a PDF file using pdfplumber (with layout awareness), falling back to pypdf.

    Args:
        file_source: File path, file-like object, or bytes.

    Returns:
        Extracted text string.

    Raises:
        ResumeParsingError: If file is empty, corrupted, or cannot be parsed.
    """
    if isinstance(file_source, (str, Path)):
        file_path = Path(file_source)
        if not file_path.exists():
            raise ResumeParsingError(f"PDF file not found: {file_path}")
        if file_path.stat().st_size == 0:
            raise ResumeParsingError("The uploaded PDF file is empty.")
        stream = open(file_path, "rb")
        should_close = True
    elif isinstance(file_source, bytes):
        if len(file_source) == 0:
            raise ResumeParsingError("The uploaded PDF file is empty.")
        stream = io.BytesIO(file_source)
        should_close = False
    else:
        stream = file_source
        should_close = False

    if hasattr(stream, "seek"):
        stream.seek(0)

    text_pages: List[str] = []
    
    # 1. Preferred primary extraction: pdfplumber (superior layout, column & font metrics)
    try:
        import pdfplumber
        with pdfplumber.open(stream) as pdf:
            if len(pdf.pages) == 0:
                raise ResumeParsingError("The PDF contains no pages.")
            for page in pdf.pages:
                plumber_text = page.extract_text()
                if plumber_text:
                    text_pages.append(plumber_text)
    except Exception:
        text_pages = []

    # 2. Fallback extraction: pypdf if pdfplumber extracted nothing or failed
    if not text_pages:
        try:
            if hasattr(stream, "seek"):
                stream.seek(0)
            reader = pypdf.PdfReader(stream)
            if len(reader.pages) == 0:
                raise ResumeParsingError("The PDF contains no pages.")
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        except Exception as pdf_err:
            raise ResumeParsingError(
                f"Could not read PDF file. The file may be corrupted, password-protected, or invalid: {pdf_err}"
            )
        finally:
            if should_close:
                stream.close()
    else:
        if should_close:
            stream.close()

    full_text = "\n".join(text_pages).replace("\x00", " ").strip()
    full_text = unkern_spaced_text(full_text)

    if not full_text:
        raise ResumeParsingError(
            "No readable text found in PDF. If this is a scanned/image PDF, please upload a text-based PDF, DOCX, or TXT file."
        )
    return full_text


def extract_text_from_docx(file_source: Union[str, Path, BinaryIO, bytes]) -> str:
    """Extract text from a DOCX file using python-docx.

    Args:
        file_source: File path, file-like object, or bytes.

    Returns:
        Extracted text string.

    Raises:
        ResumeParsingError: If file is empty, corrupted, or cannot be parsed.
    """
    try:
        if isinstance(file_source, (str, Path)):
            file_path = Path(file_source)
            if not file_path.exists():
                raise ResumeParsingError(f"DOCX file not found: {file_path}")
            if file_path.stat().st_size == 0:
                raise ResumeParsingError("The uploaded DOCX file is empty.")
            doc = docx.Document(str(file_path))
        elif isinstance(file_source, bytes):
            if len(file_source) == 0:
                raise ResumeParsingError("The uploaded DOCX file is empty.")
            doc = docx.Document(io.BytesIO(file_source))
        else:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            doc = docx.Document(file_source)

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n".join(paragraphs).strip()
        if not full_text:
            raise ResumeParsingError("The DOCX file contains no readable text content.")
        return full_text
    except ResumeParsingError:
        raise
    except Exception as e:
        raise ResumeParsingError(f"Could not parse DOCX file. It might be corrupted or in an unsupported format: {e}")


def extract_text_from_txt(file_source: Union[str, Path, BinaryIO, bytes]) -> str:
    """Extract text from a TXT file with charset fallbacks.

    Args:
        file_source: File path, file-like object, or bytes.

    Returns:
        Extracted text string.
    """
    raw_bytes: bytes
    if isinstance(file_source, (str, Path)):
        file_path = Path(file_source)
        if not file_path.exists():
            raise ResumeParsingError(f"TXT file not found: {file_path}")
        raw_bytes = file_path.read_bytes()
    elif isinstance(file_source, bytes):
        raw_bytes = file_source
    else:
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        raw_bytes = file_source.read()

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ResumeParsingError("The uploaded TXT file is empty.")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue

    raise ResumeParsingError("Failed to decode text file. Please ensure it is saved as UTF-8 plain text.")


def parse_resume(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: Optional[str] = None,
) -> Dict[str, str]:
    """Unified interface to parse any supported resume format (PDF, DOCX, TXT).

    Args:
        file_source: File path, file-like object (e.g. Streamlit UploadedFile), or raw bytes.
        filename: Optional filename hint (useful when passing BinaryIO or bytes).

    Returns:
        Dict with keys:
            - 'raw_text': Original extracted text with line structure preserved.
            - 'cleaned_text': Normalized text cleaned for ML pipelines.

    Raises:
        ResumeParsingError: If file cannot be read, is empty, or format is unsupported.
    """
    # Determine extension
    ext = ""
    if filename:
        ext = Path(filename).suffix.lower()
    elif isinstance(file_source, (str, Path)):
        ext = Path(file_source).suffix.lower()
    elif hasattr(file_source, "name") and file_source.name:
        ext = Path(file_source.name).suffix.lower()

    if not ext:
        # Attempt detection or default to txt if text readable
        if isinstance(file_source, bytes) and file_source.startswith(b"%PDF"):
            ext = ".pdf"
        elif isinstance(file_source, bytes) and file_source.startswith(b"PK\x03\x04"):
            ext = ".docx"
        else:
            ext = ".txt"

    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_source)
    elif ext in [".docx", ".doc"]:
        if ext == ".doc":
            raise ResumeParsingError("Legacy .doc format is not supported. Please save/convert as .docx or .pdf.")
        raw_text = extract_text_from_docx(file_source)
    elif ext == ".txt":
        raw_text = extract_text_from_txt(file_source)
    else:
        raise ResumeParsingError(
            f"Unsupported file format '{ext}'. Please upload a PDF (.pdf), Word Document (.docx), or Text file (.txt)."
        )

    raw_text = raw_text.strip()
    if not raw_text or len(raw_text) < 10:
        raise ResumeParsingError("Resume content is too short or empty to analyze.")

    cleaned = clean_text(raw_text)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
    }


def extract_metadata(raw_text: str) -> Dict[str, Union[str, List[str]]]:
    """Extract transparent metadata from resume text (emails, phones, education keywords, experience mentions).

    Does NOT fabricate information; returns only verifiable regex/keyword patterns found in the text.
    """
    # Email extraction
    email_match = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", raw_text)
    emails = list(dict.fromkeys(email_match))

    # Phone extraction (common global/Indian formats)
    phone_match = re.findall(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text)
    phones = [p.strip() for p in phone_match if len(re.sub(r"\D", "", p)) >= 10]
    phones = list(dict.fromkeys(phones))

    # Education mentions
    edu_keywords = [
        "bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e", "bca", "mca",
        "b.sc", "m.sc", "b.com", "m.com", "bba", "mba", "diploma", "high school",
        "computer science", "information technology", "engineering",
    ]
    edu_found: List[str] = []
    lines = raw_text.split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        for kw in edu_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", line_lower):
                cleaned_line = line.strip()
                if 5 < len(cleaned_line) < 100 and cleaned_line not in edu_found:
                    edu_found.append(cleaned_line)
                break

    # Experience mentions (e.g., "X years of experience", "2019 - 2023", etc.)
    exp_matches = re.findall(r"\b(\d{1,2}\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience)\b", raw_text, re.IGNORECASE)
    exp_summary = list(dict.fromkeys(exp_matches))

    return {
        "emails": emails,
        "phones": phones,
        "education": edu_found[:5],
        "experience_mentions": exp_summary[:3],
    }
