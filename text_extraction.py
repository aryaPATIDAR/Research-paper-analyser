import io
import os
import re
import shutil
import logging
import tempfile
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Extraction backends are imported defensively: a missing optional package
# (or a missing Tesseract/Poppler install for OCR) must not stop the API
# from starting - the remaining backends are simply used instead.
try:
    import pymupdf as fitz  # PyMuPDF (current import name)
except Exception:
    try:
        import fitz  # PyMuPDF (legacy import name)
    except Exception:  # pragma: no cover
        fitz = None

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
except Exception:  # pragma: no cover
    pdfminer_extract = None

try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:  # pragma: no cover
    convert_from_path = None
    pytesseract = None

MIN_TEXT_LENGTH = 500  # heuristic


def _normalize_extracted_text(text: str) -> str:
    """
    Normalize extracted text to fix common PDF extraction issues.
    """
    if not text:
        return ""
    
    # 1. Replace various unicode spaces with regular space
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
    
    # 2. Fix ligatures and special characters
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        '—': '-', '–': '-', ''': "'", ''': "'", '"': '"', '"': '"',
    }
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)
    
    # 3. Add spaces around punctuation that might be missing spaces
    text = re.sub(r'\.([A-Z])', r'. \1', text)  # Period followed by capital
    text = re.sub(r',([A-Za-z])', r', \1', text)  # Comma followed by letter
    
    # 4. Fix hyphenated line breaks (word- continuation on next line)
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # 5. Normalize multiple spaces and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 6. Add space between lowercase and uppercase (camelCase from PDF issues)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    return text.strip()


def _count_valid_words(text: str) -> int:
    """Count words that look like valid English words (not truncated)."""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # Common word patterns - valid words usually start with common letter combinations
    valid_starts = {'th', 'an', 'in', 're', 'on', 'at', 'en', 'er', 'ou', 'it', 'es', 'st', 
                   'or', 'te', 'of', 'ed', 'is', 'ar', 'al', 'to', 'co', 'de', 'ra', 'ri',
                   'ro', 'be', 'wa', 'ha', 'wi', 'as', 'fo', 'pr', 'no', 'se', 'so', 'un',
                   'us', 'ab', 'ac', 'ad', 'af', 'ag', 'ai', 'am', 'ap', 'au', 'av', 'ba',
                   'bi', 'bl', 'bo', 'br', 'bu', 'ca', 'ce', 'ch', 'ci', 'cl', 'cr', 'cu',
                   'da', 'di', 'do', 'dr', 'du', 'ea', 'ec', 'ef', 'el', 'em', 'ev', 'ex',
                   'fa', 'fe', 'fi', 'fl', 'fr', 'fu', 'ga', 'ge', 'gi', 'gl', 'go', 'gr',
                   'gu', 'he', 'hi', 'ho', 'hu', 'id', 'ig', 'im', 'io', 'ir', 'ja', 'jo',
                   'ju', 'ke', 'ki', 'kn', 'la', 'le', 'li', 'lo', 'lu', 'ma', 'me', 'mi',
                   'mo', 'mu', 'na', 'ne', 'ni', 'nu', 'ob', 'oc', 'of', 'op', 'ot', 'ov',
                   'pa', 'pe', 'ph', 'pi', 'pl', 'po', 'qu', 'ra', 'sc', 'sh', 'si', 'sk',
                   'sl', 'sm', 'sn', 'sp', 'sq', 'su', 'sw', 'sy', 'ta', 'tr', 'tw', 'ty',
                   'va', 've', 'vi', 'vo', 'we', 'wh', 'wo', 'wr', 'ye', 'yo'}
    
    valid_count = 0
    for word in words:
        if len(word) >= 2 and word[:2] in valid_starts:
            valid_count += 1
    
    return valid_count


def extract_with_pymupdf(pdf_path: str) -> str:
    """Extract using PyMuPDF with multiple strategies."""
    best_text = ""
    best_valid_words = 0
    
    try:
        doc = fitz.open(pdf_path)
        
        # Strategy 1: Default text extraction
        text1 = []
        for page in doc:
            text1.append(page.get_text("text"))
        text1 = "\n".join(text1)
        valid1 = _count_valid_words(text1)
        if valid1 > best_valid_words:
            best_text = text1
            best_valid_words = valid1
        
        # Strategy 2: With whitespace preservation
        text2 = []
        for page in doc:
            text2.append(page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE))
        text2 = "\n".join(text2)
        valid2 = _count_valid_words(text2)
        if valid2 > best_valid_words:
            best_text = text2
            best_valid_words = valid2
        
        # Strategy 3: Using blocks for better structure
        text3 = []
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                if block[6] == 0:  # Text block (not image)
                    text3.append(block[4])
        text3 = "\n".join(text3)
        valid3 = _count_valid_words(text3)
        if valid3 > best_valid_words:
            best_text = text3
            best_valid_words = valid3
        
        # Strategy 4: Raw text with sorting
        text4 = []
        for page in doc:
            text4.append(page.get_text("text", sort=True))
        text4 = "\n".join(text4)
        valid4 = _count_valid_words(text4)
        if valid4 > best_valid_words:
            best_text = text4
            best_valid_words = valid4
        
        doc.close()
        logger.debug(f"PyMuPDF best strategy had {best_valid_words} valid words")
        
    except Exception as e:
        logger.warning(f"PyMuPDF extraction error: {e}")
    
    return best_text


def extract_with_pdfminer(pdf_path: str) -> str:
    if pdfminer_extract is None:
        return ""
    try:
        return pdfminer_extract(pdf_path) or ""
    except Exception:
        return ""


def ocr_available() -> bool:
    """OCR needs the pytesseract + pdf2image packages AND the Tesseract and Poppler programs."""
    if convert_from_path is None or pytesseract is None:
        return False
    return bool(shutil.which("tesseract")) and bool(shutil.which("pdftoppm"))


def extract_with_ocr(pdf_path: str, max_pages: int = 15) -> str:
    if not ocr_available():
        return ""
    text = []
    try:
        images = convert_from_path(pdf_path, dpi=200, last_page=max_pages)
        for img in images:
            text.append(pytesseract.image_to_string(img))
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
    return "\n".join(text)


def extract_text_robust(pdf_path: str) -> str:
    """
    Extract text from PDF using multiple methods with fallback.
    Returns normalized text with proper spacing.
    """
    # 1. PyMuPDF (fast, good for most PDFs)
    best = extract_with_pymupdf(pdf_path)
    if len(best.strip()) >= MIN_TEXT_LENGTH:
        return _normalize_extracted_text(best)

    # 2. pdfminer (better for complex layouts)
    text = extract_with_pdfminer(pdf_path)
    if len(text.strip()) >= MIN_TEXT_LENGTH:
        return _normalize_extracted_text(text)
    if len(text.strip()) > len(best.strip()):
        best = text

    # 3. OCR (scanned PDFs) - only replaces the text layer if it finds more.
    # Previously a short-but-valid text layer was thrown away here and the
    # (usually empty) OCR output returned instead.
    text = extract_with_ocr(pdf_path)
    if len(text.strip()) > len(best.strip()):
        best = text
    return _normalize_extracted_text(best)


# -------------------------------------------------------------------------
# Bytes-level entry point used by the API
# -------------------------------------------------------------------------

# Below this much usable text a similarity / bias analysis is meaningless.
MIN_USABLE_CHARS = 400
MIN_USABLE_WORDS = 60


class PDFExtractionError(Exception):
    """The upload cannot be analysed. `code` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text or ""))


def _inspect_pdf(data: bytes) -> Tuple[bool, Optional[int], bool]:
    """Returns (opened, page_count, encrypted) using PyMuPDF."""
    if fitz is None:
        return False, None, False
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return False, None, False
    try:
        encrypted = bool(doc.needs_pass)
        return True, (None if encrypted else doc.page_count), encrypted
    except Exception:
        return False, None, False
    finally:
        doc.close()


def _extract_pdfplumber(data: bytes, x_tolerance: Optional[float] = None) -> str:
    """Original extraction path (kept first so section detection behaves as before)."""
    if pdfplumber is None:
        return ""
    kwargs = {} if x_tolerance is None else {"x_tolerance": x_tolerance}
    text = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(**kwargs) or ""
            if page_text:
                text += page_text + "\n"
    return text


# English prose averages 5-6 characters per whitespace-separated token.
# Well above that means the extractor glued words together
# ("AttentionIsAllYouNeed", "headedself"), which happens with PDFs that
# position words individually with narrow gaps (common for LaTeX papers).
RUN_TOGETHER_AVG_TOKEN = 8.5


def _avg_token_length(text: str) -> float:
    tokens = (text or "").split()
    return sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0


def _extract_pymupdf_bytes(data: bytes) -> str:
    if fitz is None:
        return ""
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _fix_run_together(data: bytes, text: str) -> Tuple[str, str]:
    """If `text` has missing spaces, re-extract with stricter word-gap settings.

    Returns (method, text) of the candidate with the most natural word lengths.
    """
    best_method, best_text, best_avg = "pdfplumber", text, _avg_token_length(text)
    for method, fn in (
        ("pdfplumber (tight word spacing)", lambda: _extract_pdfplumber(data, x_tolerance=1.5)),
        ("pymupdf", lambda: _extract_pymupdf_bytes(data)),
    ):
        try:
            candidate = fn()
        except Exception as e:
            logger.warning(f"Re-extraction with {method} failed: {type(e).__name__}")
            continue
        # must be a comparable amount of text, with clearly better spacing
        if len(candidate.strip()) < 0.8 * len(text.strip()):
            continue
        avg = _avg_token_length(candidate)
        if 0 < avg < best_avg - 0.5:
            best_method, best_text, best_avg = method, candidate, avg
    return best_method, best_text


def extract_text_from_bytes(data: bytes) -> Tuple[str, Dict[str, object]]:
    """
    Extract text from an uploaded PDF.

    Returns (text, info) or raises PDFExtractionError with a message that can
    be shown to the user (empty file, not a PDF, corrupted, password
    protected, scanned without a text layer, too little text).
    """
    if not data:
        raise PDFExtractionError("empty_file", "The uploaded file is empty.", 400)
    if b"%PDF-" not in data[:1024]:
        raise PDFExtractionError(
            "not_a_pdf", "The uploaded file is not a PDF. Please upload a research paper in PDF format.", 415
        )

    opened, pages, encrypted = _inspect_pdf(data)
    if encrypted:
        raise PDFExtractionError(
            "encrypted_pdf", "This PDF is password-protected. Remove the password and upload it again."
        )

    candidates: List[Tuple[str, str]] = []
    parser_errors: List[str] = []

    # 1. pdfplumber
    try:
        text = _extract_pdfplumber(data)
        if len(text.strip()) >= MIN_USABLE_CHARS and _avg_token_length(text) > RUN_TOGETHER_AVG_TOKEN:
            logger.info(
                f"pdfplumber text has run-together words (avg token {_avg_token_length(text):.1f} chars) - re-extracting"
            )
            method, text = _fix_run_together(data, text)
            candidates.append((method, text))
        else:
            candidates.append(("pdfplumber", text))
    except Exception as e:
        parser_errors.append(f"pdfplumber: {type(e).__name__}")
        text = ""

    # 2. Fallbacks when the text layer looks weak or pdfplumber could not read the file
    if len(text.strip()) < MIN_USABLE_CHARS:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            for name, fn in (("pymupdf", extract_with_pymupdf), ("pdfminer", extract_with_pdfminer)):
                try:
                    candidates.append((name, fn(tmp_path) or ""))
                except Exception as e:
                    parser_errors.append(f"{name}: {type(e).__name__}")
                if len(candidates[-1][1].strip()) >= MIN_USABLE_CHARS:
                    break
            if max((len(t.strip()) for _, t in candidates), default=0) < MIN_USABLE_CHARS and ocr_available():
                logger.info("No usable text layer - trying OCR")
                candidates.append(("ocr", extract_with_ocr(tmp_path)))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    method, best = max(candidates, key=lambda c: len(c[1].strip()), default=("none", ""))
    if not method.startswith("pdfplumber"):
        best = _normalize_extracted_text(best)
    chars, words = len(best.strip()), _word_count(best)
    info = {"method": method, "characters": chars, "words": words, "pages": pages}
    logger.info(f"PDF extraction: {info}")

    if chars == 0:
        if not opened:
            raise PDFExtractionError(
                "corrupted_pdf",
                "This PDF could not be read - the file appears to be corrupted or incomplete. "
                "Try re-downloading or re-exporting it.",
            )
        hint = "" if ocr_available() else " (OCR is not installed on the server)"
        raise PDFExtractionError(
            "no_text_layer",
            "No text could be extracted from this PDF. It looks like a scanned or image-only document"
            + hint + ". Please upload a text-based PDF.",
        )
    if chars < MIN_USABLE_CHARS or words < MIN_USABLE_WORDS:
        raise PDFExtractionError(
            "insufficient_text",
            f"Only {words} words could be extracted from this PDF, which is not enough for a meaningful "
            "analysis. Upload the full paper as a text-based PDF (not a scan or a single cover page).",
        )
    return best, info
