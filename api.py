import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

# Load backend/.env before any module reads its settings, whatever the
# current working directory is.
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import gemini_client
from src.bias_analyzer import BiasAnalyzer, result_to_dict
from src.plagiarism_checker import analyze_plagiarism, extract_sections
from src.text_extraction import PDFExtractionError, extract_text_from_bytes

# -------------------------------------------------
# Setup
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bias_analyzer = BiasAnalyzer()
ai_provider = os.getenv("AI_PROVIDER", "gemini")
logger.info(f"Bias analysis enabled: {bias_analyzer.enabled}, Primary provider: {ai_provider}")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
RESULT_CACHE_TTL = 3600  # seconds

app = FastAPI(title="Paper Similarity & Bias Detection API")

# Comma-separated list of allowed origins, or "*" (default) for any origin.
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Two long-running stages (literature search + AI bias review) run side by side.
_stage_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="stage")

# Complete, fully successful reports keyed by the PDF's SHA-256, so analysing
# the same file again does not repeat every external request.
_result_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_result_cache_lock = threading.Lock()


class AnalysisError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    # "error" is what the frontend displays; "detail" mirrors FastAPI's convention.
    return JSONResponse(status_code=status_code, content={"error": message, "detail": message, "code": code})


@app.exception_handler(AnalysisError)
async def _analysis_error_handler(request: Request, exc: AnalysisError):
    return _error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    return _error_response(
        400, "invalid_request",
        "No PDF was received. Send the paper as multipart/form-data in a field named 'file'.",
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")
    return _error_response(
        500, "internal_error",
        f"The server hit an unexpected error while analysing the paper ({type(exc).__name__}). "
        "Check the backend console for details.",
    )


# -------------------------------------------------
# Pipeline
# -------------------------------------------------

def _run_plagiarism(full_text: str, sections: Dict[str, str]) -> Dict[str, Any]:
    try:
        return analyze_plagiarism(full_text, sections=sections)
    except Exception as e:
        logger.exception("Plagiarism stage crashed")
        return {
            "status": "failed",
            "error": f"The similarity check failed unexpectedly ({type(e).__name__}: {str(e)[:150]}).",
            "overall_percent": 0.0,
            "overall_category": "Not checked",
            "sections": {},
            "providers": {},
            "warnings": [],
        }


def _run_bias(sections: Dict[str, str]) -> Dict[str, Any]:
    try:
        return result_to_dict(bias_analyzer.analyze_paper_sections(sections))
    except Exception as e:
        logger.exception("Bias stage crashed")
        message = f"Bias analysis failed unexpectedly ({type(e).__name__}: {str(e)[:150]})."
        return {
            "status": "failed",
            "error": message,
            "overall_score": 0,
            "severity": "low",
            "summary": message,
            "biases": [],
            "strengths": [],
            "provider": None,
            "model": None,
        }


def _analyze_pdf(data: bytes, filename: str) -> Dict[str, Any]:
    started = time.time()

    digest = hashlib.sha256(data).hexdigest()
    with _result_cache_lock:
        hit = _result_cache.get(digest)
        if hit and time.time() - hit[0] < RESULT_CACHE_TTL:
            logger.info(f"Returning cached report for {filename!r}")
            return {**hit[1], "cached": True}

    # 1. PDF text extraction (raises PDFExtractionError with a user-facing message)
    try:
        full_text, extraction = extract_text_from_bytes(data)
    except PDFExtractionError as e:
        raise AnalysisError(e.status_code, e.code, e.message)
    t_extract = time.time()

    # 2. Section detection (shared by both stages)
    sections = extract_sections(full_text)

    # 3 + 4. Plagiarism search and bias analysis, in parallel
    plagiarism_future = _stage_pool.submit(_run_plagiarism, full_text, sections)
    bias_future = _stage_pool.submit(_run_bias, sections)
    plagiarism_report = plagiarism_future.result()
    bias_dict = bias_future.result()

    plagiarism_ok = plagiarism_report.get("status") != "failed"
    bias_status = bias_dict.get("status", "ok")
    bias_ok = bias_status == "ok"

    # A stage that failed is never presented as a result.
    if not plagiarism_ok and not bias_ok:
        raise AnalysisError(
            503, "analysis_unavailable",
            "The paper was read successfully, but neither analysis could be completed. "
            f"Similarity check: {plagiarism_report.get('error')} "
            f"Bias analysis: {bias_dict.get('error')}",
        )

    if plagiarism_ok:
        warnings = list(plagiarism_report.get("warnings") or [])
    else:
        # Full detail is in plagiarism.error; keep the notice short.
        warnings = ["The similarity check could not be completed - see the plagiarism section for details."]
    if bias_status == "disabled":
        warnings.append("Bias analysis is not configured on the server (set GEMINI_API_KEY in backend/.env).")
    elif not bias_ok:
        warnings.append(f"Bias analysis failed: {bias_dict.get('error')}")

    status = "ok" if (plagiarism_report.get("status") == "ok" and bias_ok) else "partial"

    response = {
        "status": status,
        "warnings": warnings,
        "filename": filename,
        "extraction": extraction,
        "plagiarism": {
            "status": plagiarism_report.get("status", "ok"),
            "error": plagiarism_report.get("error"),
            "overall_percent": plagiarism_report.get("overall_percent", 0),
            "overall_category": plagiarism_report.get("overall_category", ""),
            "sections": plagiarism_report.get("sections", {}),
            "providers": plagiarism_report.get("providers", {}),
        },
        "bias_analysis": bias_dict,
        "timings": {
            "extraction_seconds": round(t_extract - started, 2),
            "total_seconds": round(time.time() - started, 2),
        },
        "cached": False,
    }

    if status == "ok":
        with _result_cache_lock:
            _result_cache[digest] = (time.time(), response)
            if len(_result_cache) > 50:  # keep the cache small
                oldest = min(_result_cache, key=lambda k: _result_cache[k][0])
                _result_cache.pop(oldest, None)

    logger.info(f"Analysis of {filename!r} finished: status={status} in {response['timings']['total_seconds']}s")
    return response


# -------------------------------------------------
# API
# -------------------------------------------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise AnalysisError(
            413, "file_too_large",
            f"The PDF is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller file.",
        )
    # The pipeline is blocking (PDF parsing, HTTP calls), so it runs in a worker
    # thread. The event loop stays free and /health keeps answering meanwhile.
    return await run_in_threadpool(_analyze_pdf, data, file.filename or "upload.pdf")


@app.get("/")
async def root():
    return {
        "message": "Paper Bias Detection API",
        "status": "running",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "analyze": "/analyze (POST with PDF file)"
        },
        "bias_analysis_enabled": bias_analyzer.enabled
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "bias_analysis_enabled": bias_analyzer.enabled,
        "gemini_configured": gemini_client.is_configured(),
        "gemini_models": gemini_client.model_chain(),
    }
