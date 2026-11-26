import re
import time
import os
import logging
import threading
import requests
from requests.adapters import HTTPAdapter
from typing import Callable, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.integrations.ieee_explore import ieee_search

logger = logging.getLogger(__name__)

# -------------------- CONFIG -------------------- #

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_SEARCH = "https://api.openalex.org/works"
CROSSREF_SEARCH = "https://api.crossref.org/works"
ARXIV_SEARCH = "https://export.arxiv.org/api/query"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


# Network limits. Every outbound call is bounded, and the whole search phase
# for one paper has a wall-clock budget, so a slow provider can never hang
# or crash an analysis.
CONNECT_TIMEOUT = 4.0   # seconds to establish a connection
READ_TIMEOUT = 10.0     # seconds to wait for a provider response
USER_AGENT = "ResearchPaperAnalyzer/1.0"


def _search_budget() -> float:
    """Wall-clock budget (seconds) for all literature searches of one paper."""
    return _env_float("PLAGIARISM_SEARCH_BUDGET_SECONDS", 40.0)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds (exponential backoff: 1s, 2s, then fail)
SEMANTIC_SCHOLAR_WAIT = 4  # seconds between retries (rate limit: 1 req per 3.5s without API key)

# Similarity thresholds
IEEE_THRESHOLD = 12.0
SS_THRESHOLD = 10.0
OPENALEX_THRESHOLD = 6.0


# -------------------- UTILS -------------------- #

def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u00A0\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fix_concatenated_text(text: str) -> str:
    """
    Fix text that has concatenated words without spaces.
    Handles camelCase, run-together words, and PDF extraction issues.
    """
    if not text:
        return ""
    
    # 1. Add spaces before capital letters in camelCase (e.g., "deepFake" -> "deep Fake")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 2. Add spaces between lowercase and numbers (e.g., "model2" -> "model 2")
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # 3. Fix common concatenation patterns from PDF extraction
    # Pattern: lowercaseUPPERCASE or multiple capitals followed by lowercase
    text = re.sub(r'([a-z]{2,})([A-Z][a-z])', r'\1 \2', text)
    
    # 4. Split on common word boundaries that might be missing spaces
    # ("thepaperpresentsanewmodel" -> "the paper presents ...").
    # Only applied to abnormally long tokens: run on every word it broke normal
    # English ("Attention" -> "A ttenti on", "Ashish" -> "A shish",
    # "Recognition" -> "Recogniti on"), which corrupted the search queries.
    common_articles = ['the', 'a', 'an', 'in', 'on', 'of', 'to', 'for', 'and', 'or', 'by', 'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'we', 'our', 'their', 'they']

    def _split_long_token(match):
        token = match.group(0)
        for article in common_articles:
            token = re.sub(rf'\b({article})([a-z]{{3,}})', r'\1 \2', token, flags=re.IGNORECASE)
            token = re.sub(rf'([a-z]{{3,}})({article})\b', r'\1 \2', token, flags=re.IGNORECASE)
        return token

    text = re.sub(rf'[A-Za-z]{{{MAX_KEYWORD_LENGTH + 1},}}', _split_long_token, text)

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# Longer "words" are almost always several words run together by PDF extraction
MAX_KEYWORD_LENGTH = 15


def _retry_request(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Retry a request with exponential backoff."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.info(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
    raise last_exception if last_exception else Exception("Max retries exceeded")


def similarity_percent(a: str, b: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        X = vectorizer.fit_transform([a or "", b or ""])
        sim = cosine_similarity(X[0:1], X[1:2]).ravel()[0]
    except ValueError:
        sim = 0.0
    return max(0.0, min(1.0, float(sim))) * 100.0


def categorize_similarity(pct: float) -> str:
    """
    Categorize similarity percentage:
    - 0-10%: Low similarity (mostly original)
    - 10-25%: Moderate similarity (some overlap)
    - >25%: High similarity (review recommended)
    """
    if pct <= 10:
        return "Low similarity"
    if pct <= 25:
        return "Moderate similarity"
    return "High similarity (review recommended)"


# -------------------- METADATA -------------------- #

def extract_metadata(full_text: str) -> Dict[str, str]:
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    title = lines[0][:200] if lines else ""

    year_match = re.search(r"(19|20)\d{2}", full_text)
    year = year_match.group(0) if year_match else ""

    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", full_text, re.I)
    doi = doi_match.group(0) if doi_match else ""

    return {"title": title, "year": year, "doi": doi}


# -------------------- SECTIONS -------------------- #

import re

# Try to load spaCy - graceful fallback if not installed
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        # Model not downloaded yet
        SPACY_AVAILABLE = False
        _nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    _nlp = None


def _extract_sections_regex(text: str) -> dict:
    """
    Enhanced regex-based section extraction.
    Handles section numbers (1., I., A., 1.1) and many naming variations.
    """
    text = text.replace("\r", "\n")
    lower = text.lower()
    
    # Enhanced patterns with section numbers and more variations
    # Format: (category, pattern) - patterns include optional section numbers
    section_patterns = [
        # Abstract patterns
        ("Abstract", r"(?:^|\n)\s*(?:\d+\.?\s*|[IVX]+\.?\s*|[A-Z]\.?\s*)?(?:abstract)\s*[:\.\n]", re.IGNORECASE | re.MULTILINE),
        
        # Methodology/Methods patterns - many variations
        ("Methodology", r"(?:^|\n)\s*(?:\d+\.?\s*|[IVX]+\.?\s*|[A-Z]\.?\s*)?(?:methodology|methods?|materials?\s+and\s+methods?|approach|proposed\s+(?:system|method|approach|framework|architecture)|system\s+(?:design|architecture|overview)|implementation|experimental\s+(?:setup|design)|technical\s+approach|our\s+approach|framework|design)\s*[:\.\n]", re.IGNORECASE | re.MULTILINE),
        
        # Conclusions patterns
        ("Conclusions", r"(?:^|\n)\s*(?:\d+\.?\s*|[IVX]+\.?\s*|[A-Z]\.?\s*)?(?:conclusions?|discussion|summary|concluding\s+remarks?|final\s+remarks?)\s*[:\.\n]", re.IGNORECASE | re.MULTILINE),
        
        # Introduction (useful for finding paper start)
        ("Introduction", r"(?:^|\n)\s*(?:\d+\.?\s*|[IVX]+\.?\s*|[A-Z]\.?\s*)?(?:introduction)\s*[:\.\n]", re.IGNORECASE | re.MULTILINE),
    ]
    
    # Find all section matches with their positions
    found_sections = []
    for category, pattern, flags in section_patterns:
        for match in re.finditer(pattern, text, flags):
            found_sections.append({
                "category": category,
                "start": match.start(),
                "end": match.end(),
                "match_text": match.group().strip()
            })
    
    # Sort by position in document
    found_sections.sort(key=lambda x: x["start"])
    
    # Extract content between sections
    sections = {}
    for i, section in enumerate(found_sections):
        category = section["category"]
        start = section["end"]  # Start after the header
        
        # End at next section or document end
        if i + 1 < len(found_sections):
            end = found_sections[i + 1]["start"]
        else:
            end = len(text)
        
        content = text[start:end].strip()
        
        # Keep the longest content for each category
        if category not in sections or len(content) > len(sections[category]):
            sections[category] = content
    
    # Extract title from first non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title = lines[0][:200] if lines else ""
    
    return {
        "Title": title,
        "Abstract": sections.get("Abstract", ""),
        "Methodology": sections.get("Methodology", ""),
        "Conclusions": sections.get("Conclusions", ""),
        "confidence": "regex"
    }


def _extract_sections_spacy(text: str) -> dict:
    """
    spaCy-based section extraction using NLP.
    Detects section headers based on linguistic patterns.
    """
    if not SPACY_AVAILABLE or _nlp is None:
        return {"Title": "", "Abstract": "", "Methodology": "", "Conclusions": "", "confidence": "spacy_unavailable"}
    
    # Process with spaCy (limit text length for speed)
    max_chars = 50000  # ~12-15 pages
    doc = _nlp(text[:max_chars])
    
    # Section header keywords for each category
    section_keywords = {
        "Abstract": {"abstract", "summary"},
        "Methodology": {
            "methodology", "methods", "method", "approach", "implementation",
            "proposed", "system", "framework", "architecture", "design",
            "experimental", "setup", "technique", "algorithm", "model"
        },
        "Conclusions": {
            "conclusion", "conclusions", "discussion", "summary",
            "concluding", "remarks", "future"
        },
        "Introduction": {"introduction", "background", "overview"}
    }
    
    # Find potential section headers (short sentences at start of paragraphs)
    sections = {}
    current_section = None
    current_content = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_lower = sent_text.lower()
        
        # Check if this looks like a section header
        is_header = False
        header_category = None
        
        # Header heuristics:
        # 1. Short (< 10 words)
        # 2. Contains section keyword
        # 3. Often starts with number or is ALL CAPS
        words = sent_text.split()
        if len(words) <= 10:
            # Check for section keywords
            for category, keywords in section_keywords.items():
                if any(kw in sent_lower for kw in keywords):
                    # Additional checks: starts with number or has section-like pattern
                    if (re.match(r'^[\d\.]+\s', sent_text) or  # Starts with "1.", "1.1", etc.
                        re.match(r'^[IVX]+\.?\s', sent_text) or  # Roman numerals
                        re.match(r'^[A-Z]\.?\s', sent_text) or  # Letter sections
                        sent_text.isupper() or  # ALL CAPS
                        len(words) <= 5):  # Very short = likely header
                        is_header = True
                        header_category = category
                        break
        
        if is_header and header_category:
            # Save previous section content
            if current_section and current_content:
                content = " ".join(current_content)
                if current_section not in sections or len(content) > len(sections[current_section]):
                    sections[current_section] = content
            
            # Start new section
            current_section = header_category
            current_content = []
        elif current_section:
            current_content.append(sent_text)
    
    # Save last section
    if current_section and current_content:
        content = " ".join(current_content)
        if current_section not in sections or len(content) > len(sections[current_section]):
            sections[current_section] = content
    
    # Extract title
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title = lines[0][:200] if lines else ""
    
    return {
        "Title": title,
        "Abstract": sections.get("Abstract", ""),
        "Methodology": sections.get("Methodology", ""),
        "Conclusions": sections.get("Conclusions", ""),
        "confidence": "spacy"
    }


def extract_sections(text: str) -> dict:
    """
    Hybrid section extraction using both enhanced regex and spaCy IN PARALLEL.
    Combines results for best accuracy.
    """
    if not text:
        return {
            "Title": "",
            "Abstract": "",
            "Methodology": "",
            "Conclusions": ""
        }
    
    # Regex extraction is instant. spaCy (optional) can take several seconds on
    # a long paper, so it runs inline with no artificial timeout: the old
    # as_completed(timeout=5) raised TimeoutError and failed the whole request.
    empty = {"Title": "", "Abstract": "", "Methodology": "", "Conclusions": ""}
    try:
        regex_result = _extract_sections_regex(text)
    except Exception as e:
        logger.warning(f"Section extraction (regex) failed: {e}")
        regex_result = dict(empty)

    spacy_result = dict(empty)
    if SPACY_AVAILABLE:
        try:
            spacy_result = _extract_sections_spacy(text)
        except Exception as e:
            logger.warning(f"Section extraction (spacy) failed: {e}")

    # Merge results: prefer longer, non-empty content
    def best_content(key):
        regex_val = regex_result.get(key, "")
        spacy_val = spacy_result.get(key, "")
        
        # If one is empty, use the other
        if not regex_val:
            return spacy_val
        if not spacy_val:
            return regex_val
        
        # Both have content - prefer longer one (more complete extraction)
        # But penalize if it looks like garbage (too many truncated words)
        def quality_score(content):
            words = content.split()
            if len(words) < 10:
                return 0
            # Check for truncated words (rough heuristic)
            truncated = sum(1 for w in words[:50] if len(w) > 2 and not w[0].isupper() and w[0] not in 'aeiou')
            return len(content) - (truncated * 10)
        
        return regex_val if quality_score(regex_val) >= quality_score(spacy_val) else spacy_val
    
    final_sections = {
        "Title": regex_result.get("Title") or spacy_result.get("Title", ""),
        "Abstract": best_content("Abstract"),
        "Methodology": best_content("Methodology"),
        "Conclusions": best_content("Conclusions")
    }
    
    # Fallbacks if sections are still empty or too short
    min_length = 100
    
    if len(final_sections["Abstract"]) < min_length:
        # Try to find abstract in first 2000 chars
        abstract_match = re.search(r'abstract[:\.\s]*(.{100,2000}?)(?=\n\s*\n|\n\s*\d+\.|\n\s*[IVX]+\.|\n\s*introduction)', 
                                    text[:3000], re.IGNORECASE | re.DOTALL)
        if abstract_match:
            final_sections["Abstract"] = abstract_match.group(1).strip()
        else:
            final_sections["Abstract"] = text[:1500]
    
    if len(final_sections["Methodology"]) < min_length:
        final_sections["Methodology"] = text[1500:6000]
    
    if len(final_sections["Conclusions"]) < min_length:
        final_sections["Conclusions"] = text[-2500:]
    
    logger.debug(f"Section extraction - Abstract: {len(final_sections['Abstract'])} chars, "
                 f"Methodology: {len(final_sections['Methodology'])} chars, "
                 f"Conclusions: {len(final_sections['Conclusions'])} chars")
    
    return final_sections

# -------------------- SEARCH PROVIDERS -------------------- #
#
# Each provider has a raw "_fetch_*" function that raises on any failure, and
# is called through a _Provider gate that adds:
#   - a result cache (same query is never sent twice within the TTL)
#   - politeness spacing between requests to the same API
#   - a cool-down after HTTP 429 / repeated failures (circuit breaker), so a
#     rate-limited API is skipped instead of being hit again for every section
#   - a status for every call, so the report can say what really happened

STATUS_OK = "ok"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
NO_QUERY = "_no_query"  # marker: the text had no searchable words

CACHE_TTL_SECONDS = 3600

# HTTP 429 handling: retry with exponential backoff, then cool the provider down
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_BACKOFF_SECONDS = 2.0      # 2s, then 4s (or the server's Retry-After)
RATE_LIMIT_MAX_WAIT_SECONDS = 10.0    # a longer Retry-After means: skip it for now

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})
_adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


class _RateLimited(Exception):
    def __init__(self, retry_after: Optional[float] = None):
        super().__init__("rate limited")
        self.retry_after = retry_after


def _contact_email() -> str:
    """Optional contact address for the Crossref / OpenAlex 'polite pool'."""
    email = (os.getenv("CONTACT_EMAIL") or "").strip()
    return email if "@" in email and "example." not in email else ""


def _http_get(url: str, params: Dict, timeout: Tuple[float, float], headers: Optional[Dict] = None) -> requests.Response:
    resp = _session.get(url, params=params, timeout=timeout, headers=headers)
    if resp.status_code == 429:
        retry_after = None
        try:
            retry_after = float(resp.headers.get("Retry-After", ""))
        except ValueError:
            pass
        raise _RateLimited(retry_after)
    resp.raise_for_status()
    return resp


def _fetch_semantic_scholar(query: str, max_results: int, timeout: Tuple[float, float]) -> List[Dict]:
    headers = {}
    api_key = (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    resp = _http_get(
        SEMANTIC_SCHOLAR_SEARCH,
        {"query": query, "limit": max_results, "fields": "title,abstract,url,paperId"},
        timeout,
        headers,
    )
    data = resp.json() or {}
    return [{
        "title": p.get("title", "") or "",
        "abstract": p.get("abstract", "") or "",
        "url": p.get("url", "") or f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
        "source": "SemanticScholar"
    } for p in (data.get("data") or [])]


def _fetch_openalex(query: str, max_results: int, timeout: Tuple[float, float]) -> List[Dict]:
    params = {"search": query, "per_page": max_results}
    if _contact_email():
        params["mailto"] = _contact_email()
    api_key = (os.getenv("OPENALEX_API_KEY") or "").strip()
    if api_key:
        params["api_key"] = api_key
    resp = _http_get(OPENALEX_SEARCH, params, timeout)
    data = resp.json() or {}
    results = []
    for r in (data.get("results") or []):
        # OpenAlex stores the abstract as an inverted index: {"word": [positions]}
        abstract = ""
        inverted_index = r.get("abstract_inverted_index") or {}
        if inverted_index:
            words_with_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    words_with_positions.append((pos, word))
            words_with_positions.sort()
            abstract = " ".join(w for _, w in words_with_positions)

        results.append({
            "title": r.get("title", "") or r.get("display_name", "") or "",
            "abstract": abstract,
            "url": r.get("doi") or r.get("id", "") or "",
            "source": "OpenAlex"
        })
    return results


def _fetch_crossref(query: str, max_results: int, timeout: Tuple[float, float]) -> List[Dict]:
    params = {
        "query": query,
        "rows": max_results,
        "select": "title,abstract,URL,DOI",
    }
    headers = None
    if _contact_email():
        params["mailto"] = _contact_email()
        headers = {"User-Agent": f"{USER_AGENT} (mailto:{_contact_email()})"}
    resp = _http_get(CROSSREF_SEARCH, params, timeout, headers)
    data = resp.json() or {}

    results = []
    for item in ((data.get("message") or {}).get("items") or []):
        title_list = item.get("title") or []
        title = title_list[0] if title_list else ""
        abstract = re.sub(r'<[^>]+>', '', item.get("abstract", "") or "")
        url = item.get("URL", "") or f"https://doi.org/{item.get('DOI', '')}"
        if title:
            results.append({
                "title": title,
                "abstract": abstract,
                "url": url,
                "source": "CrossRef"
            })
    return results


def _fetch_arxiv(query: str, max_results: int, timeout: Tuple[float, float]) -> List[Dict]:
    import xml.etree.ElementTree as ET

    # arXiv uses its own query syntax; requests encodes the spaces for us.
    words = [w for w in query.split() if len(w) >= 4 and w not in ACADEMIC_STOP_WORDS] or query.split()
    search_query = " AND ".join(f"all:{word}" for word in words[:5])
    resp = _http_get(
        ARXIV_SEARCH,
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        },
        timeout,
    )
    root = ET.fromstring(resp.content)
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    results = []
    for entry in root.findall('atom:entry', ns):
        title_elem = entry.find('atom:title', ns)
        title = (title_elem.text or "").strip().replace('\n', ' ') if title_elem is not None else ""

        summary_elem = entry.find('atom:summary', ns)
        abstract = (summary_elem.text or "").strip().replace('\n', ' ') if summary_elem is not None else ""

        url = ""
        for link in entry.findall('atom:link', ns):
            if link.get('type') == 'text/html':
                url = link.get('href', '')
                break
        if not url:
            id_elem = entry.find('atom:id', ns)
            url = id_elem.text if id_elem is not None else ""

        if title:
            results.append({
                "title": title,
                "abstract": abstract,
                "url": url,
                "source": "arXiv"
            })
    return results


class _Provider:
    """Serialises, spaces out, caches and circuit-breaks calls to one API."""

    def __init__(self, name: str, fetch: Callable, min_interval: float, cooldown: float = 60.0):
        self.name = name
        self._fetch = fetch
        self._min_interval = min_interval
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._blocked_until = 0.0
        self._blocked_reason = STATUS_ERROR
        self._failures = 0
        self._cache: Dict[Tuple[str, int], Tuple[float, List[Dict]]] = {}

    def _cached(self, key) -> Optional[List[Dict]]:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]
        return None

    def reset(self) -> None:
        """Forget cool-downs and cached results (used by tests)."""
        self._blocked_until = 0.0
        self._failures = 0
        self._cache.clear()

    def search(self, query: str, max_results: int, deadline: float) -> Tuple[List[Dict], str]:
        """Returns (results, status). Never raises."""
        key = (query, max_results)
        cached = self._cached(key)
        if cached is not None:
            return list(cached), STATUS_OK

        if time.time() < self._blocked_until:
            return [], self._blocked_reason

        # One request at a time per provider; give up if the budget runs out.
        if not self._lock.acquire(timeout=max(0.0, deadline - time.time())):
            return [], STATUS_TIMEOUT
        try:
            cached = self._cached(key)
            if cached is not None:
                return list(cached), STATUS_OK
            if time.time() < self._blocked_until:
                return [], self._blocked_reason

            pause = self._last_call + self._min_interval - time.time()
            if pause > 0:
                if time.time() + pause >= deadline:
                    return [], STATUS_TIMEOUT
                time.sleep(pause)

            remaining = deadline - time.time()
            if remaining < 1.5:
                return [], STATUS_TIMEOUT
            timeout = (min(CONNECT_TIMEOUT, remaining), min(READ_TIMEOUT, remaining))

            try:
                results = self._fetch_with_backoff(query, max_results, timeout, deadline)
            except _RateLimited as e:
                wait_for = e.retry_after if e.retry_after and e.retry_after > 0 else self._cooldown
                self._block(min(wait_for, 300.0), STATUS_RATE_LIMITED)
                logger.warning(
                    f"{self.name}: still rate limited (429) after retries - skipping it for {int(min(wait_for, 300))}s"
                )
                return [], STATUS_RATE_LIMITED
            except requests.exceptions.Timeout:
                self._note_failure(STATUS_TIMEOUT)
                logger.warning(f"{self.name}: request timed out")
                return [], STATUS_TIMEOUT
            except Exception as e:
                self._note_failure(STATUS_ERROR)
                logger.warning(f"{self.name}: search failed ({type(e).__name__}: {str(e)[:120]})")
                return [], STATUS_ERROR
            finally:
                self._last_call = time.time()

            self._failures = 0
            self._cache[key] = (time.time(), results)
            logger.info(f"{self.name} returned {len(results)} results")
            return list(results), STATUS_OK
        finally:
            self._lock.release()

    def _fetch_with_backoff(self, query: str, max_results: int, timeout, deadline: float) -> List[Dict]:
        """Fetch, retrying HTTP 429 with exponential backoff.

        Waits for the server's Retry-After when it sends one, otherwise
        2s, then 4s. Gives up (re-raising _RateLimited) when the retries are
        used, the requested wait is long, or the search budget would be
        exceeded - the caller then cools the provider down and the other
        databases carry the search.
        """
        delay = RATE_LIMIT_BACKOFF_SECONDS
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return self._fetch(query, max_results, timeout)
            except _RateLimited as e:
                wait_for = e.retry_after if e.retry_after and e.retry_after > 0 else delay
                out_of_time = time.time() + wait_for + 2.0 >= deadline
                if attempt == RATE_LIMIT_RETRIES or wait_for > RATE_LIMIT_MAX_WAIT_SECONDS or out_of_time:
                    raise
                logger.info(f"{self.name}: rate limited (429) - retry {attempt + 1}/{RATE_LIMIT_RETRIES} in {wait_for:.0f}s")
                time.sleep(wait_for)
                delay *= 2
                remaining = deadline - time.time()
                timeout = (min(CONNECT_TIMEOUT, remaining), min(READ_TIMEOUT, remaining))
        raise _RateLimited()  # not reached

    def _block(self, seconds: float, reason: str) -> None:
        self._blocked_until = time.time() + seconds
        self._blocked_reason = reason

    def _note_failure(self, reason: str) -> None:
        self._failures += 1
        if self._failures >= 2:  # two failures in a row: stop hammering it for a while
            self._block(self._cooldown, reason)
            self._failures = 0


# min_interval follows each API's published etiquette (arXiv asks for 3s).
PROVIDERS: List[_Provider] = [
    _Provider("CrossRef", _fetch_crossref, min_interval=0.4),
    _Provider("arXiv", _fetch_arxiv, min_interval=3.0),
    _Provider("OpenAlex", _fetch_openalex, min_interval=0.3),
    _Provider("SemanticScholar", _fetch_semantic_scholar, min_interval=1.1),
]
_PROVIDER_BY_NAME = {p.name: p for p in PROVIDERS}

# Shared worker pool for provider calls. Not used as a context manager on
# purpose: leaving a `with ThreadPoolExecutor()` block waits for every worker,
# which is how one slow provider used to stall the whole request.
_search_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="paper-search")


def reset_search_state() -> None:
    """Clear provider caches and cool-downs (used by tests)."""
    for provider in PROVIDERS:
        provider.reset()
    _gemini_paper_cache.clear()


def _safe_single_search(provider_name: str, query: str, max_results: int) -> List[Dict]:
    results, _ = _PROVIDER_BY_NAME[provider_name].search(query, max_results, time.time() + CONNECT_TIMEOUT + READ_TIMEOUT)
    return results


# Backwards-compatible helpers: return [] on any failure.
def _semantic_scholar_search(query: str, max_results: int) -> List[Dict]:
    return _safe_single_search("SemanticScholar", query, max_results)


def _openalex_search(query: str, max_results: int) -> List[Dict]:
    return _safe_single_search("OpenAlex", query, max_results)


def _crossref_search(query: str, max_results: int = 10) -> List[Dict]:
    return _safe_single_search("CrossRef", query, max_results)


def _arxiv_search(query: str, max_results: int = 10) -> List[Dict]:
    return _safe_single_search("arXiv", query, max_results)


# -------------------- CORE ANALYSIS -------------------- #

# Common academic stop words to filter out
ACADEMIC_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'their',
    'they', 'them', 'which', 'who', 'whom', 'what', 'where', 'when', 'how',
    'paper', 'study', 'research', 'proposed', 'method', 'approach', 'using',
    'based', 'results', 'show', 'shows', 'shown', 'also', 'however', 'thus',
    'figure', 'table', 'section', 'following', 'provide', 'present', 'work',
    'used', 'use', 'case', 'given', 'describe', 'described', 'introduction',
    'abstract', 'conclusion', 'conclusions', 'related', 'previous', 'other'
}

def is_likely_truncated(word: str) -> bool:
    """
    Detect if a word is likely truncated from PDF extraction.
    Returns True if the word appears to be malformed/truncated.
    """
    if len(word) < 4:
        return True
    
    word = word.lower()
    
    # Known truncated words from PDF extraction (missing first letter(s))
    truncated_patterns = {
        # Missing 'a' prefix
        'bstract', 'nalysis', 'pproach', 'lgorithm', 'udio', 'reas', 'uthenticity',
        'uthor', 'rticle', 'cademic', 'ccuracy', 'chieve', 'dvanced',
        # Missing 'in' prefix  
        'ternational', 'terdisciplinary', 'troduction', 'vestigation', 'telligent',
        'formation', 'cluding', 'crease', 'creasing', 'ternet', 'tegrates', 'tegrate',
        'tegrated', 'tegration', 'terface', 'terfaces', 'teractive', 'teresting',
        # Missing 'con' prefix
        'clusion', 'conclusi', 'cluded', 'ference', 'tribution', 'sidering',
        # Missing 'de' prefix
        'tection', 'detecti', 'velop', 'velopment', 'tector', 'tected', 'epfake',
        # Missing 're' prefix
        'sults', 'search', 'searc', 'cognition', 'liable', 'levant',
        # Missing 'ex' prefix
        'periment', 'perimental', 'traction', 'plore', 'isting',
        # Missing 'me' prefix
        'thods', 'thodology', 'thod',
        # Missing 'or' prefix
        'iginality', 'iginal',
        # Missing other prefixes
        'udio', 'ignal', 'ignals', 'ystem', 'ystems', 'ection', 'ections',
        'pplication', 'fficient', 'fficiency', 'valuate', 'valuation',
        'roposed', 'resent', 'resented', 'erformance', 'earning',
    }
    
    # Check exact matches
    if word in truncated_patterns:
        return True
    
    # Words ending in incomplete suffixes (likely cut off)
    incomplete_suffixes = ('ti', 'si', 'ri', 'di', 'ni', 'gi', 'ci', 'cti', 'ndi')
    if word.endswith(incomplete_suffixes) and len(word) > 4:
        return True
    
    # Invalid word starts - consonant clusters that don't occur at word beginnings
    invalid_starts = {
        'bstr', 'bsc', 'ppr', 'ntr', 'ncl', 'nst', 'nth', 'xpl', 'xtr', 'xpr',
        'bs', 'bt', 'ck', 'ct', 'dl', 'dm', 'dn', 'ds', 'dt', 'dv',
        'fs', 'ft', 'gm', 'gn', 'gt', 'hm', 'hn', 'hr', 'hs', 'ht',
        'lb', 'lc', 'ld', 'lf', 'lg', 'lk', 'll', 'lm', 'ln', 'lp', 'lr', 'ls', 'lt',
        'mb', 'mc', 'md', 'mf', 'mg', 'mk', 'ml', 'mm', 'mn', 'mp', 'mr', 'ms', 'mt',
        'nb', 'nc', 'nd', 'nf', 'ng', 'nk', 'nl', 'nm', 'nn', 'np', 'nr', 'ns', 'nt', 'nv',
        'pb', 'pc', 'pd', 'pf', 'pg', 'pk', 'pm', 'pn', 'pp', 'pt', 'pv',
        'rb', 'rc', 'rd', 'rf', 'rg', 'rk', 'rl', 'rm', 'rn', 'rp', 'rr', 'rs', 'rt', 'rv',
        'sb', 'sd', 'sf', 'sg', 'sr', 'ss', 'sv',
        'tb', 'tc', 'td', 'tf', 'tg', 'tk', 'tl', 'tm', 'tn', 'tp', 'ts', 'tt', 'tv',
        'vb', 'vc', 'vd', 'vf', 'vg', 'vk', 'vl', 'vm', 'vn', 'vp', 'vr', 'vs', 'vt',
        'wb', 'wc', 'wd', 'wf', 'wg', 'wk', 'wl', 'wm', 'wp', 'ws', 'wt',
        'xb', 'xc', 'xd', 'xf', 'xg', 'xk', 'xl', 'xm', 'xn', 'xp', 'xr', 'xs', 'xt',
        'thm', 'thn', 'ths', 'rch', 'rth', 'lst', 'rst', 'ngl', 'ngs', 'ncr',
    }
    
    # Check first 2-4 characters against invalid starts
    for length in [4, 3, 2]:
        if len(word) >= length and word[:length] in invalid_starts:
            return True
    
    return False


SHORT_TEXT_WORDS = 20


def build_search_query(text: str) -> str:
    """
    Build a short, effective academic search query
    from section text. Extracts meaningful technical keywords.
    """
    # First, fix any concatenated text from PDF extraction
    text = _fix_concatenated_text(text)
    
    # A title or other short passage is the best possible query as it stands.
    # Keyword filtering would strip it to nothing: every word of
    # "Attention Is All You Need" is either short or a stop word.
    plain_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text)
    if 0 < len(plain_words) <= SHORT_TEXT_WORDS:
        query = " ".join(plain_words).lower()
        logger.debug(f"Built search query (short text used as is): {query}")
        return query

    # Extract words (4+ characters, alphabetic)
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    
    # Filter out stop words and deduplicate
    keywords = []
    seen = set()
    for word in words:
        # Skip if it's a stop word or already seen
        if word in ACADEMIC_STOP_WORDS or word in seen:
            continue
        # Skip very long words (likely concatenated garbage from PDF extraction)
        if len(word) > MAX_KEYWORD_LENGTH:
            logger.debug(f"Skipping long word: '{word[:20]}...'")
            continue
        # Skip truncated words
        if is_likely_truncated(word):
            logger.debug(f"Skipping likely truncated word: '{word}'")
            continue
        keywords.append(word)
        seen.add(word)
        if len(keywords) >= 8:  # Keep query focused
            break
    
    # Never return an empty query for text that has words: fall back to its
    # first words, unfiltered.
    if len(keywords) < 2 and plain_words:
        keywords = [w.lower() for w in plain_words[:8]]

    query = " ".join(keywords)
    logger.debug(f"Built search query: {query}")
    return query


_gemini_paper_cache: Dict[str, Tuple[float, List[Dict]]] = {}


def _gemini_find_papers(text: str, max_results: int = 5, budget_seconds: float = 20.0) -> List[Dict]:
    """
    Use Gemini AI to suggest related academic papers.
    This is a fallback when the academic APIs fail or return few results.
    Suggestions come from a language model and are NOT verified against a
    database, so they are labelled with source "Gemini" in the report.
    """
    from src import gemini_client

    if os.getenv("GEMINI_PAPER_SEARCH", "true").lower() != "true":
        return []
    if not gemini_client.is_configured():
        return []

    excerpt = text[:2000]  # Limit text size
    cached = _gemini_paper_cache.get(excerpt)
    if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
        return list(cached[1])

    prompt = f"""Based on this academic paper excerpt, identify 5 real, published academic papers that discuss similar topics or methodologies.

Paper excerpt:
{excerpt}

Return ONLY a JSON array of papers with this exact format (no markdown, no explanation):
[
  {{"title": "Exact Paper Title", "authors": "Author Names", "year": "2023", "url": "https://doi.org/..."}}
]

Requirements:
- Only include real, verifiable papers that actually exist
- Include DOI links when possible
- Focus on papers from major publishers (IEEE, ACM, Springer, Nature, arXiv)
- Return exactly 5 papers"""

    try:
        papers, _model = gemini_client.generate_json(
            prompt,
            max_output_tokens=4096,
            temperature=0.1,  # Low temperature for factual responses
            budget_seconds=budget_seconds,
            attempt_timeout=min(15.0, budget_seconds),
        )
    except gemini_client.GeminiError as e:
        logger.warning(f"Gemini paper search failed: {e.message}")
        return []
    except Exception as e:
        logger.warning(f"Gemini paper search failed: {e}")
        return []

    if isinstance(papers, dict):  # tolerate {"papers": [...]}
        papers = next((v for v in papers.values() if isinstance(v, list)), [])
    if not isinstance(papers, list):
        return []

    results = []
    for paper in papers[:max_results]:
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title") or "")
        if title:
            results.append({
                "title": title,
                "abstract": f"By {paper.get('authors', 'Unknown')} ({paper.get('year', '')})",
                "url": str(paper.get("url") or ""),
                "source": "Gemini"
            })

    _gemini_paper_cache[excerpt] = (time.time(), results)
    return results


def search_related_papers_detailed(
    section_text: str,
    max_results: int = 10,
    deadline: Optional[float] = None,
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Search CrossRef, arXiv, OpenAlex and Semantic Scholar IN PARALLEL.

    Returns (unique_results, provider_status) where provider_status maps each
    provider name to "ok" | "rate_limited" | "timeout" | "error".
    A provider that is slow, rate limited or down only loses its own results;
    it can never raise out of this function or delay it past `deadline`.
    """
    if deadline is None:
        deadline = time.time() + _search_budget()

    # Fix concatenated text first
    fixed_text = _fix_concatenated_text(section_text)
    query = build_search_query(fixed_text)

    if not query:
        logger.warning("Could not build search query from text")
        return [], {NO_QUERY: NO_QUERY}

    logger.info(f"Search query: '{query}'")

    futures = {
        _search_pool.submit(provider.search, query, max_results, deadline): provider.name
        for provider in PROVIDERS
    }
    # +1s grace: each call already limits itself to the deadline.
    done, not_done = wait(futures, timeout=max(0.0, deadline - time.time()) + 1.0)

    all_results: List[Dict] = []
    status: Dict[str, str] = {}
    for future in done:
        name = futures[future]
        try:
            results, state = future.result()
        except Exception as e:  # defensive: provider.search should not raise
            logger.warning(f"{name} search failed: {e}")
            results, state = [], STATUS_ERROR
        status[name] = state
        all_results.extend(results)
    for future in not_done:
        # Still running: ignore it. Its own request timeout ends it shortly and
        # a late answer is cached for next time.
        status[futures[future]] = STATUS_TIMEOUT
        logger.warning(f"{futures[future]}: no answer within the search budget, continuing without it")

    # Deduplicate results by title
    seen_titles = set()
    unique_results = []
    for item in all_results:
        title = (item.get("title") or "").lower().strip()
        if title and len(title) > 10 and title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(item)

    # Gemini AI fallback (if we have few results and still have time)
    remaining = deadline - time.time()
    if len(unique_results) < 5 and remaining > 6:
        logger.info("Few database results - asking Gemini for additional papers...")
        gemini_results = _gemini_find_papers(fixed_text, max_results=5, budget_seconds=min(20.0, remaining))
        if gemini_results:
            status["Gemini"] = STATUS_OK
        for item in gemini_results:
            title = (item.get("title") or "").lower().strip()
            if title and len(title) > 10 and title not in seen_titles:
                seen_titles.add(title)
                unique_results.append(item)

    logger.info(f"Total: Found {len(unique_results)} unique papers ({status})")
    return unique_results, status


def search_related_papers(section_text: str, max_results: int = 10) -> List[Dict]:
    """Backwards-compatible wrapper: just the list of related papers."""
    results, _ = search_related_papers_detailed(section_text, max_results)
    return results


def analyze_section(section_text: str, top_k: int = 5, deadline: Optional[float] = None):
    """
    Similarity analysis for one section.

    "status" tells the caller whether the number can be trusted:
      ok          - at least one academic database answered
      degraded    - no database answered; only unverified AI suggestions were compared
      unavailable - nothing could be searched, so no similarity was measured
      skipped     - the section has too little text to analyse
    """
    # ---- EARLY EXIT ----
    # (A short title such as "Attention Is All You Need" must still be checked;
    # the old 30-character minimum silently skipped it.)
    if not section_text or len(section_text.strip()) < 3:
        return {
            "best_similarity_percent": 0.0,
            "category": "Low similarity",
            "matches": [],
            "status": "skipped",
            "providers": {}
        }

    # ---- NORMAL LOGIC ----
    candidates, providers = search_related_papers_detailed(section_text, deadline=deadline)

    if NO_QUERY in providers:
        # Nothing searchable in this text (e.g. only numbers or symbols).
        # No service failed, so this must not be reported as an outage.
        return {
            "best_similarity_percent": 0.0,
            "category": "Not checked",
            "matches": [],
            "status": "skipped",
            "providers": {}
        }

    database_ok = any(
        state == STATUS_OK for name, state in providers.items() if name != "Gemini"
    )
    if database_ok:
        status = "ok"
    elif candidates:
        status = "degraded"
    else:
        status = "unavailable"

    scored = []
    for paper in candidates:
        content = f"{paper.get('title','')} {paper.get('abstract','')}"
        pct = similarity_percent(section_text, content)
        scored.append((pct, paper))

    scored.sort(key=lambda x: x[0], reverse=True)

    best_pct = scored[0][0] if scored else 0.0

    return {
        "best_similarity_percent": round(best_pct, 2),
        "category": categorize_similarity(best_pct) if status != "unavailable" else "Not checked",
        "matches": [
            {
                "percent": round(p, 2),
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "source": d.get("source", "")
            }
            for p, d in scored[:top_k]
        ],
        "status": status,
        "providers": providers
    }


SECTION_NAMES = ["Title", "Abstract", "Methodology", "Conclusions"]
# Weights: Title, Abstract, Methodology, Conclusions
SECTION_WEIGHTS = {"Title": 0.05, "Abstract": 0.45, "Methodology": 0.45, "Conclusions": 0.05}

_PROVIDER_PROBLEM_TEXT = {
    STATUS_RATE_LIMITED: "rate limited (HTTP 429)",
    STATUS_TIMEOUT: "timed out",
    STATUS_ERROR: "unavailable",
}


def analyze_plagiarism(full_text: str, sections: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """
    Similarity report for the four key sections.

    report["status"]:
      ok      - every section with text was checked against at least one database
      partial - some sections could not be checked (see report["warnings"])
      failed  - no section could be checked; overall_percent is NOT a result
    """
    if sections is None:
        sections = extract_sections(full_text)

    deadline = time.time() + _search_budget()

    def run(name: str):
        try:
            return analyze_section(sections.get(name, ""), deadline=deadline)
        except Exception as e:  # one section must never take the report down
            logger.exception(f"Section '{name}' analysis failed")
            return {
                "best_similarity_percent": 0.0,
                "category": "Not checked",
                "matches": [],
                "status": "unavailable",
                "providers": {},
                "error": str(e)[:200]
            }

    # Sections are searched concurrently; per-provider gates keep the
    # requests to each API polite and sequential.
    section_pool = ThreadPoolExecutor(max_workers=len(SECTION_NAMES), thread_name_prefix="section")
    try:
        results = dict(zip(SECTION_NAMES, section_pool.map(run, SECTION_NAMES)))
    finally:
        section_pool.shutdown(wait=False)

    report: Dict[str, object] = {"sections": {}, "overall_percent": 0.0}
    provider_states: Dict[str, List[str]] = {}

    for name in SECTION_NAMES:
        result = results[name]
        # INCLUDE SECTION TEXT (the frontend / bias stage rely on it)
        report["sections"][name] = {
            "text": sections.get(name, ""),
            "best_similarity_percent": result.get("best_similarity_percent", 0.0),
            "category": result.get("category", ""),
            "matches": result.get("matches", []),
            "status": result.get("status", "ok"),
        }
        for provider, state in (result.get("providers") or {}).items():
            provider_states.setdefault(provider, []).append(state)

    # Overall weighted similarity over the sections that were really checked
    checked = [n for n in SECTION_NAMES if results[n]["status"] in ("ok", "degraded")]
    unchecked = [n for n in SECTION_NAMES if results[n]["status"] == "unavailable"]
    degraded = [n for n in SECTION_NAMES if results[n]["status"] == "degraded"]

    weight_sum = sum(SECTION_WEIGHTS[n] for n in checked)
    if weight_sum > 0:
        overall = sum(SECTION_WEIGHTS[n] * results[n]["best_similarity_percent"] for n in checked) / weight_sum
        report["overall_percent"] = float(np.round(overall, 2))

    # Per-provider summary: "ok" if it answered at least once
    providers_summary = {}
    for provider, states in provider_states.items():
        providers_summary[provider] = STATUS_OK if STATUS_OK in states else states[0]
    report["providers"] = providers_summary

    warnings: List[str] = []
    problems = [
        f"{provider} {_PROVIDER_PROBLEM_TEXT.get(state, state)}"
        for provider, state in providers_summary.items() if state != STATUS_OK
    ]
    # A database that was skipped while the others covered every section does
    # not change the result, so it is not shown as a notice. It is still
    # listed in report["providers"] and in the server log.
    if problems and (degraded or unchecked):
        warnings.append("Some literature databases did not respond: " + ", ".join(problems) + ".")
    if degraded:
        warnings.append(
            "No literature database answered for: " + ", ".join(degraded)
            + ". Those sections were compared only against AI-suggested references, which are unverified."
        )

    if not checked:
        report["status"] = "failed"
        report["overall_category"] = "Not checked"
        report["error"] = (
            "The similarity check could not be performed: none of the literature databases "
            "(CrossRef, arXiv, OpenAlex, Semantic Scholar) could be reached. "
            + ("; ".join(problems) + ". " if problems else "")
            + "Check the internet connection and try again in a minute."
        )
    else:
        if unchecked:
            report["status"] = "partial"
            warnings.append(
                "Could not check: " + ", ".join(unchecked)
                + ". The overall similarity is based on the remaining sections only."
            )
        else:
            report["status"] = "partial" if degraded else "ok"
        report["overall_category"] = categorize_similarity(report["overall_percent"])

    report["warnings"] = warnings
    return report
