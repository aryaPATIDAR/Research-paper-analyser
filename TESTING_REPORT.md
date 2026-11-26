# Testing report

Date: 2026-11-20. Environment: Windows 11, Python 3.14.6 (fresh `backend/.venv` installed from `requirements.txt`), Node 24.11.0, real Gemini key, real internet.

Test paper: `tests/sample_paper.pdf`, a 2-page text PDF written for this test (deepfake detection) with deliberately biased wording. No third-party PDF was used.

## Result

- Passed: all steps of the requested workflow, 22 offline failure-mode tests, 15 live checks, 4 browser scenarios.
- Not passing / not tested: see "Open items" at the end.

## Before the fixes (original code, same machine)

| Check | Result |
|-------|--------|
| `/health` while an analysis runs | No answer within 10s (server blocked) |
| Sample paper | HTTP 200, but bias analysis was a Gemini 503 shown as score 50 "moderate" |
| Random bytes named `.pdf`, truncated PDF, empty file, `.txt` | HTTP 500 with raw parser text |
| PDF with a short text layer | HTTP 400 "No text extracted" (text was discarded) |
| Literature APIs | CrossRef 429 on 2 of 4 sections, Semantic Scholar 429 on all |

## Requested workflow (after the fixes)

| # | Step | Result | Evidence |
|---|------|--------|----------|
| 1 | Start the backend | Pass | `uvicorn api:app` starts, no errors |
| 2 | `/health` | Pass | 200, `status: ok`, `gemini_configured: true`; 0.03s while an analysis was running |
| 3 | `/docs` | Pass | 200, Swagger UI lists `/analyze`, `/`, `/health` (opened in a browser) |
| 4 | Upload a text-based PDF | Pass | HTTP 200 |
| 5 | PDF extraction | Pass | pdfplumber, 5183 characters, 696 words, 2 pages |
| 6 | Plagiarism analysis | Pass | `status: ok`, overall 16.2%, 5 matches in each of 4 sections; CrossRef, arXiv, OpenAlex ok; Semantic Scholar 429 (reported as a warning) |
| 7 | Gemini bias analysis | Pass | `status: ok`, score 65-75 across runs, 4 biases (confirmation, selection, methodology, publication) matching the wording planted in the paper; answered by fallback model `gemini-3.5-flash-lite` |
| 8 | Complete `/analyze` response | Pass | All fields present; 23-31s total; repeat upload served from cache in 0.01s |
| 9 | Same workflow in the React frontend | Pass | See browser scenarios |
| 10 | Unavailable external APIs and malformed PDFs | Pass | See the two tables below |

## Malformed uploads (live server)

| Upload | HTTP | Message |
|--------|------|---------|
| Empty file | 400 | The uploaded file is empty. |
| Random bytes named `.pdf` | 415 | The uploaded file is not a PDF... |
| Text file | 415 | The uploaded file is not a PDF... |
| `%PDF` header followed by garbage | 422 | This PDF could not be read - the file appears to be corrupted... |
| Image-only (scanned) PDF | 422 | No text could be extracted... scanned or image-only document (OCR is not installed on the server)... |
| PDF with 2 words | 422 | Only 2 words could be extracted... |
| Request without a file | 400 | No PDF was received... |
| Real PDF cut to one third of its size | 200 | Recovered by PyMuPDF (218 words) and analysed |

The server stayed healthy after every case; no tracebacks in the log.

## External services unavailable

Live server, forced by routing outbound traffic to a dead proxy:

| Scenario | Result |
|----------|--------|
| Every external host refuses connections | HTTP 503 in 24s, message names both failed stages |
| Every external host hangs (black hole) | HTTP 503 in 85s (worst case; under the 180s frontend timeout) |
| Literature databases down, Gemini up | HTTP 200 `partial`: similarity "unavailable", bias analysis shown |
| Gemini key invalid, databases up | HTTP 200 `partial`: similarity shown, bias "API configuration issue" |
| Gemini primary and second model overloaded (real, on the test day) | Answered by the third model; 69s before hedging was added, 31s after |

Offline tests, `python -m unittest tests.test_resilience` - 22 tests, all pass:

- A provider that hangs for 30s: report returns in under 8s, provider marked `timeout`, others used (regression test for `TimeoutError: 1 (of 4) futures unfinished`).
- One provider down: others' results kept.
- All providers down: `status: failed`, not "0%".
- Provider answers 429: called once, then skipped.
- Same query twice: no second request.
- Gemini: 503 falls back to the next model with one retry; 429 and malformed JSON fall through; a slow model is hedged; all models down gives a clean error; an invalid key stops immediately.
- Bias analyzer: 503 and three kinds of malformed JSON give `status: failed`; valid output is normalised; placeholder keys disable the provider.
- Pipeline: ok, bias down (partial), search down (partial), both down (503).

## Browser scenarios (Vite dev server at `localhost:5173`, backend at `127.0.0.1:8000`)

| Scenario | Result |
|----------|--------|
| Upload sample paper | Staged loading text, then results: 16.2% overall, four section cards, expanded match table with links, bias score, 4 biases, strengths. `POST /analyze` 200, no console errors |
| Upload scanned PDF | "Analysis Failed" with the server's explanation; page still usable |
| Backend stopped | "Cannot reach the analysis server at http://127.0.0.1:8000. Make sure the backend is running and try again." |
| Partial results (both directions) | Yellow notice plus an "unavailable" panel for the failed stage; the other stage's results shown |

`npm run build` completes without errors.

The file was attached to the page's file input by script, since an automated browser cannot use the native file dialog. Checks were made on the rendered page text; no screenshots were taken, so visual appearance (colours, spacing) of the new notice was not inspected by eye.

## Open items

- **Semantic Scholar** returned 429 in every run. It is skipped and reported, not fixed. A free API key (`SEMANTIC_SCHOLAR_API_KEY`) should resolve it.
- **`gemini-3.8-flash` and `gemini-3.6-flash`** never answered successfully during testing (503/504). The fallback covers it, so results in this report come from `gemini-3.5-flash-lite`.
- **OCR of scanned PDFs** not tested: Tesseract and Poppler are not installed on this machine. Such PDFs get a clear error.
- **Password-protected PDFs**: the error path is implemented but was not exercised with a real encrypted file.
- **OpenRouter fallback** not tested: no real OpenRouter key is configured.
- **spaCy section extraction** not tested: spaCy is not installed, so the regex extractor was used throughout.
- **Real published papers** (multi-column layouts, 20+ pages) were not tested; only the generated sample was. Try your demo PDF with `tests\smoke_test.py path\to\paper.pdf` beforehand.
- **Drag-and-drop** upload was not exercised in the browser; the click-to-select path was.
- **Production deployment** (Render/Vercel) was not tested; only local runs.

---

# Follow-up tests: title check and Semantic Scholar 429 (2026-11-20)

**Reproduced before the fix**
- `build_search_query("Attention Is All You Need")` returned an empty string (`_fix_concatenated_text` produced `A ttenti on Is All You Need`).
- A PDF with narrow word gaps (`tests/sample_paper_tight_spacing.pdf`) was extracted by pdfplumber as 29 tokens averaging 71.5 characters (`AttentionIsAllYouNeed...`).
- Semantic Scholar, anonymous, three requests 3s apart: 429, 429, 200. No `Retry-After` header.

**After the fix**

Offline, `python -m unittest tests.test_resilience`: 30 tests, all pass. New ones cover: short titles give a real query; normal words are not split; the title is checked while Semantic Scholar is rate limited; text with no words is `skipped` and not reported as an outage; 429 is retried with 2s then 4s and then succeeds; `Retry-After` is used; a persistent 429 stops after 3 requests and cools down; backoff never sleeps past the search budget; the tight-spacing PDF is extracted with spaces and the title "Attention Is All You Need".

Live, `tests/sample_paper_tight_spacing.pdf` (first page of text modelled on "Attention Is All You Need") through `POST /analyze`:

| Check | Result |
|-------|--------|
| HTTP / status | 200, `status: ok`, no warnings, 26.7s |
| Extraction | `pdfplumber (tight word spacing)`, 345 words, spaces restored |
| Title query sent | `attention is all you need` |
| Title | `ok`, best match 100% ("Is Attention All You Need?", CrossRef), 5 matches |
| Abstract | `ok`, 57.5%, top match the real "Attention Is All You Need" (arXiv) |
| Methodology / Conclusions | `ok`, 6.2% / 31.4% |
| 429 handling in the log | CrossRef: 429, retried after 2s, succeeded. Semantic Scholar: 429, retried after 2s and 4s, then skipped for 60s |
| Bias analysis | `ok` (gemini-3.5-flash-lite) |

The original smoke test (`tests/smoke_test.py`) and `npm run build` still pass.

**Not verified**
- Your own `demo_paper_test.pdf` was not available on this machine. The fix was verified on a PDF built to reproduce the same symptoms (glued words, same title). Please re-run your file.
- The browser UI was not re-tested for this change; only one text label changed in the frontend.
- Section detection still sometimes starts the "Methodology" text mid-sentence (seen in the live run: query began with `enerated symbols`). That predates these changes and was left alone.
