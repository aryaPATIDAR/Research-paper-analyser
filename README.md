# Research Paper Analyzer

A comprehensive, open-source web tool for analyzing academic papers for plagiarism and bias detection.

![Landing Page](assets/screenshot-landing.png)

## Why This Tool?

### The Problem with Existing Solutions

Academic plagiarism detection is dominated by expensive, proprietary tools that lack critical features researchers actually need:

| Tool | Cost | Bias Detection | Self-Hosted | Academic DBs | Open Source |
|------|------|----------------|-------------|--------------|-------------|
| **Turnitin** | $3+/student/year (institutional only) | No | No | Limited | No |
| **iThenticate** | $125/document | No | No | Yes | No |
| **Grammarly** | $12-30/month | No | No | Limited | No |
| **Copyscape** | $0.03/search | No | No | No (web only) | No |
| **This Tool** | **Free** | **Yes (AI-powered)** | **Yes** | **Yes (6 sources)** | **Yes** |

### What Existing Tools Are Missing

**1. No Bias Detection**
Turnitin, iThenticate, and Grammarly only detect text similarity. They completely ignore academic bias—a critical factor in research integrity. Our tool uses Google's Gemini AI to detect:
- Confirmation bias, Selection bias, Publication bias
- Funding bias, Citation bias, Methodology bias

**2. Prohibitive Pricing**
- Turnitin requires institutional licensing ($10,000+ for universities)
- iThenticate charges **$125 per document**—impractical for students
- Students cannot purchase Turnitin individually

**3. No Data Privacy**
Commercial tools upload your research to their servers. Some index submissions permanently. Our tool is **self-hosted**—your research never leaves your machine.

**4. Limited Transparency**
Proprietary tools are black boxes. You can't verify their algorithms or customize them. Our tool is fully open-source.

**5. No Actionable Feedback**
Traditional tools show similarity percentages without context. We provide:
- Section-by-section analysis (Title, Abstract, Methodology, Conclusions)
- Direct links to matching sources
- AI-generated suggestions for improvement

---

## Project Overview

Research Paper Analyzer is an AI-powered tool that reads academic papers (PDF), extracts main sections, and provides:

1. **Plagiarism Detection**: Evaluates similarity against published academic work from **6 different sources** including CrossRef (130M+ works), arXiv (2M+ papers), OpenAlex (250M+ works), Semantic Scholar (200M+ papers), IEEE Xplore, and Gemini AI-powered search
2. **AI Bias Analysis**: Detects 6 types of academic bias using Google's Gemini AI with actionable improvement suggestions

## Screenshots

### After Uploading pdf
![Landing Page](assets/Screenshot%20uploaded%20pdf.png)

### Plagiarism Analysis Results
![Plagiarism Results](assets/screenshot-plagiarism.png)

### AI Bias Detection
![Bias Analysis](assets/screenshot-bias.png)

---

## Features

### Plagiarism Detection
- Extracts text from uploaded PDFs (with robust fallbacks using PyMuPDF, pdfminer, and OCR)
- **Smart text normalization** to handle PDF extraction issues (concatenated words, missing spaces)
- **Hybrid section extraction** using Enhanced Regex + spaCy NLP running in parallel
- Splits content into four sections: Title, Abstract, Methodology, Conclusions
- **Parallel API searches** - All 4 academic databases queried simultaneously for 3x faster results
- Searches **6 academic databases** (all FREE, no API keys required for most):

| Source | Database Size | API Key | Best For |
|--------|---------------|---------|----------|
| **CrossRef** | 130M+ works | Not required | All academic papers |
| **arXiv** | 2M+ papers | Not required | CS, Physics, Math |
| **OpenAlex** | 250M+ works | Not required | All disciplines |
| **Semantic Scholar** | 200M+ papers | Not required | AI/ML papers |
| **IEEE Xplore** | 6M+ documents | Optional | Engineering papers |
| **Gemini AI** | Web search | Required | Fallback for rare topics |

- Computes TF-IDF cosine similarity to estimate overlap
- **Similarity Classification:**

| Similarity | Category | Color | Meaning |
|------------|----------|-------|---------|
| **0–10%** | Low | Green | Mostly original content |
| **10–25%** | Moderate | Yellow | Some overlap detected |
| **>25%** | High | Red | Review recommended |

- Displays top matching sources with direct links
- Automatic retry with exponential backoff for rate-limited APIs

### AI Bias Analysis
Powered by Google's Gemini AI, detects:
- **Confirmation Bias**: Language assuming conclusions before evidence
- **Selection Bias**: Non-representative sampling indicators
- **Publication Bias**: Overly positive framing, suppressed negative results
- **Funding Bias**: Undisclosed conflicts of interest
- **Citation Bias**: Selective citing supporting predetermined views
- **Methodology Bias**: Flawed experimental design

**Bias Score Classification:**
| Score | Severity | Meaning |
|-------|----------|---------|
| 0-25 | Low | Excellent objectivity |
| 26-50 | Moderate | Some concerns present |
| 51-75 | High | Significant issues detected |
| 76-100 | Severe | Major credibility concerns |

Provides:
- Overall bias score (0-100)
- Detailed explanations for each detected bias
- Actionable suggestions for improvement
- Identified strengths in the paper

---

## Tech Stack

### Backend
- Python 3.11+
- FastAPI
- Google GenAI SDK (Gemini)
- PDF Processing: PyMuPDF, pdfplumber, pdfminer.six, pytesseract
- NLP: spaCy (en_core_web_sm model)
- ML: scikit-learn, numpy, scipy

### Frontend
- React 18 (Vite)
- TypeScript
- Axios
- Modern dark theme UI

---

## How Section Extraction Works

The tool uses a **hybrid approach** combining Enhanced Regex and spaCy NLP, running **in parallel** for maximum accuracy and speed.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF Text Input                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              ThreadPoolExecutor (Parallel)                  │
│  ┌─────────────────────┐     ┌─────────────────────┐       │
│  │   Enhanced Regex    │     │     spaCy NLP       │       │
│  │     (~0.01s)        │     │     (~0.3s)         │       │
│  │                     │     │                     │       │
│  │ • Section numbers   │     │ • Sentence bounds   │       │
│  │   (1., IV., A.)     │     │ • Header detection  │       │
│  │ • 20+ name variants │     │ • Linguistic        │       │
│  │ • ALL CAPS headers  │     │   patterns          │       │
│  └──────────┬──────────┘     └──────────┬──────────┘       │
│             └────────────┬───────────────┘                  │
│                          ▼                                  │
│              Smart Merge (Best Content)                     │
│         • Prefer longer extractions                         │
│         • Quality scoring                                   │
│         • Fallback to text slices                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│     4 Sections: Title, Abstract, Methodology, Conclusions   │
└─────────────────────────────────────────────────────────────┘
```

### Section Name Detection

The Enhanced Regex now recognizes many naming variations:

| Section | Detected Names |
|---------|----------------|
| **Abstract** | "Abstract", "Summary" |
| **Methodology** | "Methodology", "Methods", "Proposed System", "Framework", "Architecture", "Implementation", "Experimental Setup", "Our Approach", "System Design", "Technical Approach" |
| **Conclusions** | "Conclusion", "Conclusions", "Discussion", "Summary", "Concluding Remarks", "Final Remarks" |

Plus section numbers: `1.`, `2.`, `I.`, `II.`, `IV.`, `A.`, `1.1`, etc.

### PDF Text Cleaning

The tool includes intelligent filtering for common PDF extraction issues:

**Truncated Word Detection:**
Words like `"bstract"` (abstract), `"tegrates"` (integrates), `"ternational"` (international) are automatically filtered from search queries.

**Concatenated Word Detection:**
Long garbage strings like `"minantsequencetransductionmodels"` (>15 characters) are filtered out.

---

## Performance Optimizations

### Parallel API Execution

All academic database searches run **simultaneously** instead of sequentially:

```
BEFORE (Sequential):
[CrossRef 3s] → [arXiv 2s] → [OpenAlex 2s] → [Semantic Scholar 4s]
Total: ~11 seconds per section × 4 sections = ~44+ seconds

AFTER (Parallel):
┌─────────────┐
│ CrossRef    │ ─┐
│ arXiv       │  │ All run simultaneously
│ OpenAlex    │  ├─→ ~3-4 seconds per section
│ Sem.Scholar │ ─┘
└─────────────┘
Total: ~4 seconds × 4 sections = ~16 seconds
```

### Performance Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total analysis time | ~100s | ~35s | **3x faster** |
| API calls per section | Sequential | Parallel | **4x faster** |
| Section extraction | Basic regex | Hybrid (Regex + spaCy) | More accurate |
| Timeout per API | 15-20s | 8s | Faster failure recovery |

### Reliability

Every external call is bounded, so one slow service cannot hang or crash an analysis:

| Limit | Default | Setting |
|-------|---------|---------|
| Literature API connect / read timeout | 4s / 10s | `plagiarism_checker.py` |
| All literature searches for one paper | 40s | `PLAGIARISM_SEARCH_BUDGET_SECONDS` |
| One Gemini model call | 35s | `GEMINI_TIMEOUT_SECONDS` |
| Start the next Gemini model if no answer | 12s | `GEMINI_HEDGE_SECONDS` |
| Whole bias analysis | 100s | `GEMINI_BUDGET_SECONDS` |
| Frontend request timeout | 180s | `web/src/ui/App.tsx` |

- A database that answers HTTP 429 is skipped for a cool-down period instead of being retried for every section.
- Search results and complete reports are cached in memory for one hour.
- If a stage fails, the response says so (`status`, `warnings`); a failed stage is never shown as a result.

---

## Requirements

- Python 3.11 - 3.14 (tested on 3.14)
- Node.js 18+ (tested on 24)
- A Gemini API key (free tier available) - required for bias analysis
- Internet access (literature databases and Gemini)

---

## Setup & Run (Windows PowerShell)

Run these from the project folder (the one containing `backend` and `web`). Use two PowerShell windows: one for the backend, one for the frontend.

### 1. Backend - first-time setup

```powershell
cd backend
```

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
Copy-Item .env.example .env
```

Open `backend\.env` in a text editor and set `GEMINI_API_KEY` (get one at https://aistudio.google.com/app/apikey). Skip the `Copy-Item` step if you already have a `.env`.

### 2. Backend - start

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Leave this window open. Check it in a browser:

- http://127.0.0.1:8000/health - should show `"status":"ok"` and `"gemini_configured":true`
- http://127.0.0.1:8000/docs - interactive API page

### 3. Frontend - first-time setup (second PowerShell window)

```powershell
cd web
```

```powershell
npm install
```

```powershell
Copy-Item .env.example .env
```

`web\.env` should contain `VITE_API_BASE=http://127.0.0.1:8000`.

### 4. Frontend - start

```powershell
npm run dev
```

Open http://localhost:5173, choose a PDF and click **Analyze Paper**. An analysis normally takes 20-60 seconds. `tests\sample_paper.pdf` is a small demo paper.

### 5. Check everything before a demo

With the backend running, from the project folder:

```powershell
.\backend\.venv\Scripts\python.exe tests\smoke_test.py
```

It checks `/health`, `/docs`, a full analysis of the sample paper, and the handling of bad uploads, and prints PASS/FAIL per step.

Offline failure-mode tests (no internet or key needed):

```powershell
.\backend\.venv\Scripts\python.exe -m unittest tests.test_resilience
```

### Optional

- **spaCy section detection:** `.\.venv\Scripts\python.exe -m pip install -r requirements-spacy.txt` then `.\.venv\Scripts\python.exe -m spacy download en_core_web_sm`. Without it the regex extractor is used.
- **OCR for scanned PDFs:** install Tesseract and Poppler and add both to `PATH`. Without them, scanned PDFs are rejected with a clear message.

### Troubleshooting

| Message | Meaning / fix |
|---------|---------------|
| "Cannot reach the analysis server" | Backend is not running, or `VITE_API_BASE` in `web\.env` is wrong. Restart `npm run dev` after editing `.env`. |
| "Bias Analysis Unavailable" | Gemini key missing/invalid, quota used up, or all models overloaded. Check `/health`, then retry in a minute. |
| "Similarity check unavailable" | No literature database could be reached. Check the internet connection. |
| Yellow notice naming a database | Shown only when a section could not be checked against any database. A database that is merely rate limited while the others answer is not shown; see `plagiarism.providers` in the API response or the backend log. |
| "No text could be extracted" | Scanned / image-only PDF. Use a text-based PDF or install OCR. |
| `uvicorn` / `python` not recognized | Use the full `.\.venv\Scripts\python.exe -m ...` form shown above. |

---

## Deployment

### Quick Deploy to Production

**Ready to deploy?** We provide comprehensive guides for deploying your application:

📚 **[Complete Deployment Guide](DEPLOYMENT.md)** - Detailed step-by-step instructions

⚡ **[Quick Start Guide](DEPLOYMENT_QUICKSTART.md)** - Deploy in 5 minutes

### Recommended Stack
- **Backend**: Render Web Service ($7/month for Starter plan)
- **Frontend**: Vercel (Free tier)
- **Total Cost**: ~$7/month

### One-Click Deploy Options

**Deploy Backend to Render:**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

Just connect your GitHub repo and Render will use the `render.yaml` configuration.

**Deploy Frontend to Vercel:**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/rk0802p/paper_bias_detection&root-directory=web&env=VITE_API_BASE&project-name=paper-bias-frontend)

### Deployment Checklist
- [ ] Backend deployed on Render
- [ ] Frontend deployed on Vercel
- [ ] Environment variables configured (`GEMINI_API_KEY`, `VITE_API_BASE`)
- [ ] CORS settings updated
- [ ] Test with real PDF upload
- [ ] Monitor logs for errors

For detailed instructions, see the [deployment guides](DEPLOYMENT.md).

---

## API Endpoints

### POST /analyze
Upload a PDF file for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (PDF file, 25 MB max)

**Response (200):**
```json
{
  "status": "ok",
  "warnings": [],
  "extraction": { "method": "pdfplumber", "characters": 5183, "words": 696, "pages": 2 },
  "plagiarism": {
    "status": "ok",
    "overall_percent": 16.2,
    "overall_category": "Moderate similarity",
    "providers": { "CrossRef": "ok", "arXiv": "ok", "OpenAlex": "ok", "SemanticScholar": "rate_limited" },
    "sections": {
      "Abstract": {
        "status": "ok",
        "best_similarity_percent": 10.1,
        "category": "Low similarity",
        "matches": [{ "percent": 10.1, "title": "...", "url": "...", "source": "arXiv" }]
      }
    }
  },
  "bias_analysis": {
    "status": "ok",
    "provider": "gemini",
    "model": "gemini-3.5-flash-lite",
    "overall_score": 65,
    "severity": "high",
    "summary": "...",
    "biases": [],
    "strengths": [],
    "error": null
  },
  "timings": { "total_seconds": 31.3 },
  "cached": false
}
```

- `status` is `"ok"` or `"partial"`. With `"partial"`, `warnings` explains what could not be done, and the affected stage has `status: "failed"` (or a section has `status: "unavailable"`).
- A section `status` is `ok`, `degraded` (only unverified AI-suggested references were available), `unavailable` or `skipped`.

**Errors** always have the form `{"error": "message", "detail": "message", "code": "..."}`:

| HTTP | code | When |
|------|------|------|
| 400 | `empty_file`, `invalid_request` | Empty upload / no `file` field |
| 413 | `file_too_large` | PDF over the size limit |
| 415 | `not_a_pdf` | The file is not a PDF |
| 422 | `corrupted_pdf`, `encrypted_pdf`, `no_text_layer`, `insufficient_text` | The PDF cannot be analysed |
| 503 | `analysis_unavailable` | Neither the similarity check nor the bias analysis could be completed |
| 500 | `internal_error` | Unexpected server error |

### GET /health
Returns `status`, whether bias analysis is enabled, whether a Gemini key is configured, and the Gemini model chain.

### GET /docs
Interactive Swagger UI.

---

---

## Comparison with Industry Leaders

### vs Turnitin
| Feature | Turnitin | This Tool |
|---------|----------|-----------|
| Price | $10,000+/institution | Free |
| Individual Access | No | Yes |
| Bias Detection | No | Yes |
| Self-Hosted | No | Yes |
| Open Source | No | Yes |
| AI Suggestions | No | Yes |

### vs iThenticate
| Feature | iThenticate | This Tool |
|---------|-------------|-----------|
| Price | $125/document | Free |
| Bias Detection | No | Yes |
| Real-time Analysis | No | Yes |
| Section Analysis | No | Yes |
| Improvement Tips | No | Yes |

### vs Grammarly
| Feature | Grammarly | This Tool |
|---------|-----------|-----------|
| Academic Focus | Limited | Full |
| Scholarly Databases | ProQuest only | 6 sources (CrossRef, arXiv, OpenAlex, Semantic Scholar, IEEE, Gemini) |
| Bias Detection | No | Yes |
| Self-Hosted | No | Yes |
| Research Integrity | Grammar focus | Plagiarism + Bias |

---

## Use Cases

- **Students**: Check papers before submission without institutional access
- **Researchers**: Verify originality and identify potential bias before peer review
- **Institutions**: Self-hosted solution with complete data privacy
- **Journals**: Pre-screen submissions for integrity issues

---

## Notes & Limitations

- Similarity scores are approximate signals for manual review, not legal plagiarism determinations
- Bias analysis quality depends on Gemini AI model and API availability
- For scanned/image PDFs, OCR support requires Tesseract and Poppler installed on the system
- When the literature databases return few results, Gemini suggests related papers; these are marked "AI-suggested" and are not verified against a database
- Most plagiarism detection APIs (CrossRef, arXiv, OpenAlex, Semantic Scholar) work without API keys
- IEEE Xplore requires a valid API key for enhanced results
- Gemini API key is required for bias analysis and fallback paper search

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
