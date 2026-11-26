# Repair summary

What was wrong, what was changed, and where. Test evidence is in `TESTING_REPORT.md`.

## Modified files

| File | Change |
|------|--------|
| `backend/api.py` | Rewritten request pipeline (see below) |
| `backend/src/plagiarism_checker.py` | Search providers, concurrency, timeouts, status reporting |
| `backend/src/bias_analyzer.py` | Uses shared Gemini client, validates output, reports failure honestly |
| `backend/src/gemini_client.py` | **New.** Timeout, model fallback, retry, JSON parsing for Gemini |
| `backend/src/text_extraction.py` | Bytes-level extraction with specific errors; optional imports |
| `backend/requirements.txt` | Removed upper pins that break on current Python |
| `backend/.env.example` | New settings documented, placeholders only |
| `web/src/ui/App.tsx` | API base handling, timeout, error and loading messages, partial results |
| `web/src/ui/BiasAnalysisSection.tsx` | Failure states, unknown-severity guard |
| `web/.env.example` | Local backend as the default |
| `test_gemini_api.py` | No longer prints part of the API key; uses `GEMINI_MODEL` |
| `README.md` | Windows PowerShell instructions, API contract, troubleshooting |
| `tests/test_resilience.py`, `tests/smoke_test.py`, `tests/sample_paper.pdf` | **New.** Tests and demo paper |

Not changed: page layout and styling, `GridDistortion`, `package.json`, the similarity algorithm (TF-IDF cosine), the bias prompt, and the unused legacy modules (`gemini_bias_analyzer.py`, `integrations/ieee_explore.py`, etc.).

## Root causes and fixes

### 1. `timeout of 120000ms exceeded` in the frontend
Causes, all confirmed in the code or by measurement:
- The Gemini SDK was called with no timeout. An overloaded model can hold a request for 30 seconds or more before answering 504, and the code then waited on a second provider for up to 120 seconds.
- `OPENROUTER_API_KEY` still held the `.env.example` placeholder, which the code treated as a real key, so every Gemini failure triggered a pointless OpenRouter call (HTTP 401).
- The four sections were searched one after another, and the Gemini "find papers" fallback added up to four more unbounded Gemini calls.
- With no `VITE_API_BASE`, the frontend silently used the remote Render deployment, which cold-starts slowly.

Fixes: every outbound call has a hard timeout; the literature search has a 40s budget and the bias analysis a 100s budget; the two stages run in parallel; placeholder keys are ignored; the frontend timeout is 180s (above the backend's worst case) and, in dev, the default API is the local backend.

### 2. `500 Internal Server Error`
- Any unreadable PDF made pdfplumber raise, and the catch-all returned 500 with a raw parser message (`No /Root object! - Is this really a PDF?`).
- `/analyze` was `async` but did blocking work, so the whole server (including `/health`) froze during an analysis. Measured: `/health` timed out during an analysis before the fix, 0.03s after.

Fixes: uploads are validated and rejected with a specific 4xx status and a readable message; the pipeline runs in a worker thread; each stage is isolated so one failing does not take down the request.

### 3. `TimeoutError: 1 (of 4) futures unfinished`
`as_completed(..., timeout=N)` raises when a future is not finished in time, and nothing caught it. The same pattern was in `extract_sections` (`timeout=5`, hit whenever spaCy is slow). In addition, `with ThreadPoolExecutor()` waits for all workers on exit, so the timeout did not even shorten the wait.

Fix: provider calls go through `concurrent.futures.wait()` on a shared pool. Unfinished providers are recorded as `timeout` and ignored; nothing is raised. Section extraction no longer uses threads or a timeout.

### 4. HTTP 429 / timeouts from academic APIs
- Every section hit every provider again immediately after a 429.
- Requests carried a fake `mailto=research@example.com`.
- arXiv was called over `http://` (redirect) and faster than its 3-second etiquette.

Fixes, per provider: one request at a time with polite spacing, a cool-down after 429 (honouring `Retry-After`) or two consecutive failures, a one-hour result cache, connect/read timeouts of 4s/10s. `CONTACT_EMAIL`, `SEMANTIC_SCHOLAR_API_KEY` and `OPENALEX_API_KEY` are optional settings. Each provider's outcome is reported in `plagiarism.providers`.

### 5. Gemini 503 and malformed JSON
- A 503 was returned to the user as HTTP 200 with a made-up score of 50 ("moderate").
- JSON was parsed by stripping code fences only; anything else failed, and a parse failure also became score 50.
- The paper-search call allowed only 1000 output tokens, which thinking models can use up before writing any JSON.

Fixes (`gemini_client.py`): JSON response mode, tolerant parsing, strict validation of the bias result, a model chain (`GEMINI_MODEL` then `GEMINI_FALLBACK_MODELS`), one retry on a fast 5xx, cool-downs for failing models, and hedging (if a model has not answered in 12s the next one is started and the first valid answer wins). A failed bias analysis is reported as `status: "failed"` with the reason.

### 6. PDF extraction returning 400
The fallback extractor discarded any text shorter than 500 characters and returned the OCR result instead, which is empty when Tesseract is not installed. Top-level imports of the OCR packages also meant a missing optional package stopped the server from starting.

Fixes: the best result across pdfplumber, PyMuPDF and pdfminer is kept; OCR is used only when installed; damaged files are recovered where PyMuPDF can repair them; distinct errors for empty, non-PDF, corrupted, password-protected, scanned and too-short documents.

### 7. Results that looked successful but were not
- All databases down produced "0.0% - Low similarity".
- Gemini down produced a bias score of 50.

Fixes: the response has `status` (`ok` / `partial`) and `warnings`; each stage and section has its own `status`. If both stages fail the API returns 503. The frontend shows a notice for partial results, a "not checked" state for sections, and never a number for a stage that did not run. Sections that could only be compared with Gemini-suggested references are marked `degraded` and the references are labelled as unverified.

### 8. Installation
`numpy<2.0.0` has no wheels for Python 3.13+, so `pip install` had to compile numpy from source (or failed). `fastapi<0.115` and `uvicorn<0.25` were similarly dated. The upper pins are removed; the project was reinstalled from scratch and retested on Python 3.14.

### 9. Frontend
- API base: `VITE_API_BASE` if set; otherwise the Vite proxy in dev and the deployed backend in production builds.
- The manual `Content-Type: multipart/form-data` header (no boundary) is removed.
- Errors: server message when there is one; otherwise specific text for timeout, unreachable backend, oversized file and 5xx.
- Loading text follows the real stages of the request.
- Rendering no longer assumes fields exist (`percent.toFixed` on a missing value, unknown severity).

## Things to know

- `backend/.env` is not included in the ZIP because it contains your real key. Your existing `backend/.env` in the project folder is untouched.
- On the day of testing, `gemini-3.8-flash` and `gemini-3.6-flash` were overloaded (503/504) for this key and every analysis was answered by `gemini-3.5-flash-lite`. That is handled automatically, but for a demo you can save about 25 seconds per first analysis by setting `GEMINI_MODEL=gemini-3.5-flash-lite` in `backend/.env`.
- Semantic Scholar answered 429 in every run (shared anonymous pool). A free `SEMANTIC_SCHOLAR_API_KEY` fixes this; the other three databases are enough for a result.
- The similarity score compares each section with titles and abstracts of related papers. It is a signal for manual review, not a full-text plagiarism verdict.

---

# Follow-up: "Could not check: Title" (2026-11-20)

Reported: title shown as "Not checked - search services unavailable" together with a Semantic Scholar 429 notice.

## Root cause

The Semantic Scholar 429 was **not** what stopped the title check. The backend log of the failing run shows `Could not build search query from text` for the title, while CrossRef, arXiv and OpenAlex answered normally for the other sections. No request was ever sent for the title. Four defects combined:

1. **Words glued together during PDF extraction.** For PDFs that position each word separately with narrow gaps (typical for LaTeX papers), pdfplumber's default settings drop the spaces: `AttentionIsAllYouNeed`, `headedself`, `nikidesigned`. The failing run extracted 35,525 characters but only 2,237 words.
2. **The "fix concatenated text" step damaged normal words.** It split every word that starts or ends with a short function word: `Attention` -> `A ttenti on`, `Ashish` -> `A shish`, `Recognition` -> `Recogniti on`. The fragments were then discarded as "truncated".
3. **Keyword filtering emptied short titles.** In "Attention Is All You Need" every word is under 4 letters, a stop word (`need`), or was destroyed by step 2, so the query was empty.
4. **Titles under 30 characters were never checked**, and an empty query was reported as "search services unavailable", which pointed at the wrong component.

Separately, Semantic Scholar's anonymous access really is rate limited: a direct probe returned 429, 429, then 200 a few seconds apart, with no `Retry-After` header. The existing code made one request and then skipped the provider for 60 seconds.

Requests were not duplicated: one request per provider per section, results cached.

## Changes

| File | Change |
|------|--------|
| `backend/src/text_extraction.py` | Detects run-together text (average token length over 8.5 characters) and re-extracts with tighter word-gap settings or PyMuPDF, keeping whichever has natural word lengths |
| `backend/src/plagiarism_checker.py` | Function-word splitting only applies to abnormally long tokens; short texts (20 words or fewer, i.e. titles) are searched as they are; a query is never empty if the text has words; the 30-character minimum is removed; text with nothing searchable is `skipped`, not `unavailable`; arXiv searches the significant words of the query |
| `backend/src/plagiarism_checker.py` | HTTP 429: up to 2 retries with exponential backoff (2s, 4s), using `Retry-After` when the server sends it; gives up if the wait is over 10s or would exceed the search budget, then cools the provider down |
| `web/src/ui/App.tsx` | Wording of the "skipped" state only |
| `tests/test_resilience.py`, `tests/sample_paper_tight_spacing.pdf` | 8 new regression tests and a PDF that reproduces the glued-words extraction |

The title is searched in CrossRef, arXiv, OpenAlex and Semantic Scholar in parallel; any one of them answering is enough.

## Not changed

Semantic Scholar can still answer 429 after the retries; that is its anonymous shared quota and only an API key (`SEMANTIC_SCHOLAR_API_KEY` in `backend/.env`) removes it. The other databases are used. No notice is shown for this as long as every section was checked by at least one database; the skipped database is still listed in `plagiarism.providers` and the backend log. A notice appears only if a section could not be checked at all.
