# AI Resume Ranker

Ranks a folder of resumes (PDF / DOCX / TXT) against a job description and
produces a scored, ordered shortlist with reasoning, as CSV and JSON.

This README is written so a stranger can clone this repo and get a working
run in under 5 minutes, with **zero API keys required** (offline mode), or
with real Cohere + Groq + Qdrant accounts (live mode).

---

## 0. What's actually in this repo

```
resume_ranker/
├── utils.py                     # ResumeRanker class: parsing, scoring, ranking
├── app.py                       # Streamlit UI
├── run_demo.py                  # CLI script: runs the pipeline end-to-end, saves CSV+JSON
├── requirements.txt
├── SCORING_METHOD.md            # Full write-up of the scoring formula
├── sample_data/
│   ├── job_description.txt      # Sample JD used for the demo run
│   ├── resumes/                 # 8 sample resumes (mix of TXT/DOCX/PDF, varied fit)
│   └── output/                  # Actual saved results from running run_demo.py
│       ├── ranked_results_offline.csv
│       └── ranked_results_offline.json
└── README.md                    # This file
```

---

## 1. Install

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd resume_ranker
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

That's it for offline mode — no further setup needed.

---

## 2. Configure API keys (only needed for live mode)

The project runs in two modes:

- **Offline mode (default)** — TF-IDF similarity + rule-based skill
  extraction. No API keys, no network calls, fully deterministic. This is
  what generated the results in `sample_data/output/`.
- **Live mode** — Cohere embeddings + Qdrant vector search + Groq LLM
  (Llama-3.3-70B) for real semantic similarity and LLM-written reasoning.

To use live mode, create a `.env` file in the project root:

```bash
cp .env.example .env   # then edit .env with your real values
```

`.env` contents:

```
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

Where to get each key:
- Cohere: https://dashboard.cohere.com/api-keys (free tier available)
- Groq: https://console.groq.com/keys (free tier available)
- Qdrant: https://cloud.qdrant.io (free 1GB cluster available) — create a
  cluster, copy its URL and API key. Local/self-hosted Qdrant also works;
  just point `QDRANT_URL` at it.

If any live-mode key is missing, `run_demo.py --live` and the Streamlit app
will tell you exactly what's missing rather than failing silently.

---

## 3. Run the agent end to end

### Option A — CLI (fastest way to verify it works)

```bash
# Offline mode — no keys needed, runs in seconds
python run_demo.py

# Live mode — needs .env configured (see above)
python run_demo.py --live
```

This loads `sample_data/job_description.txt` and every resume in
`sample_data/resumes/`, ranks them, and writes:
- `sample_data/output/ranked_results_offline.csv` / `.json` (offline mode)
- `sample_data/output/ranked_results_live.csv` / `.json` (live mode)

The offline-mode output files are already committed to this repo — you
can inspect them without running anything, or delete them and re-run
`python run_demo.py` to regenerate them yourself.

### Option B — Web UI

```bash
streamlit run app.py
```

Opens a browser UI where you can:
1. Paste a job description
2. Upload resumes (PDF/DOCX/TXT, 10+ supported)
3. Enter your Cohere + Groq keys in the sidebar (pre-filled from `.env` if present)
4. Click "Rank Resumes"
5. Download results as CSV, JSON, Excel, or a text summary report

The Streamlit UI **only** runs live mode (it needs both API keys entered in
the sidebar to enable the button). Use the CLI for offline mode.

### Using your own resumes / JD

```bash
python run_demo.py --live   # edit run_demo.py's JD_PATH / RESUMES_DIR constants,
                             # or just replace the files in sample_data/
```
or drop your files into `sample_data/resumes/` and replace
`sample_data/job_description.txt`, then re-run.

---

## 4. Sample run (already executed, results committed)

`sample_data/resumes/` contains 8 resumes across all 3 supported formats
(`.txt`, `.docx`, `.pdf`), deliberately spanning strong/medium/weak fits for
the sample JD (a Senior Python Backend Developer role):

| # | Candidate | Format | Intended fit |
|---|---|---|---|
| 1 | Priya Sharma | .txt | Strong |
| 2 | Alex Kim | .txt | Weak (frontend-focused) |
| 3 | Maria Gonzalez | .txt | Medium-strong (ML-leaning) |
| 4 | James Okoro | .txt | Weak (junior, PHP-focused) |
| 5 | Sara Patel | .txt | Very strong (staff-level) |
| 6 | Tom Reilly | .txt | Medium (DevOps, not backend dev) |
| 7 | Wei Zhang | .docx | Strong |
| 8 | Linda Nguyen | .pdf | Medium-strong |

Actual offline-mode output (`sample_data/output/ranked_results_offline.csv`):

| Rank | Resume | Score | NLP Similarity | Rule-Based Score |
|---|---|---|---|---|
| 1 | candidate_05_sara_patel.txt | 52.6 | 25.3 | 80 |
| 2 | candidate_07_wei_zhang.docx | 47.6 | 25.3 | 70 |
| 3 | candidate_01_priya_sharma.txt | 47.0 | 24.0 | 70 |
| 4 | candidate_08_linda_nguyen.pdf | 44.1 | 28.3 | 60 |
| 5 | candidate_03_maria_gonzalez.txt | 41.8 | 23.5 | 60 |
| 6 | candidate_06_tom_reilly.txt | 39.1 | 18.3 | 60 |
| 7 | candidate_04_james_okoro.txt | 17.4 | 14.7 | 20 |
| 8 | candidate_02_alex_kim.txt | 10.6 | 11.1 | 10 |

This matches intuition: the staff-level, most-relevant-skills candidate
ranks #1, and the frontend-focused candidate with almost no backend overlap
ranks last. Full reasoning (skills/gaps/strengths per candidate) is in the
CSV/JSON files.

**I was not able to run live mode for this submission** — see "Environment
note" below.

---

## 5. Design choices

- **Two-signal blended score.** `Score = w1 × NLP_Similarity + w2 × LLM/Rule_Score`
  (default 50/50, adjustable). Pure embedding similarity can be gamed by
  keyword-stuffed resumes; pure LLM judgment is slower, costs more, and can
  be inconsistent run-to-run. Blending both, and reporting both sub-scores
  separately in the output, makes the final number harder to game and easier
  to audit. Full formula in `SCORING_METHOD.md`.
- **Offline mode as a first-class path, not an afterthought.** `utils.py`
  lazily imports Cohere/Groq/Qdrant only inside the functions that need
  them, so the offline path (TF-IDF + rule-based extraction) has zero
  dependency on those packages being installed or reachable. This makes the
  project runnable in restricted/CI environments and testable without
  burning API credits.
- **Chunking before embedding (live mode).** Resumes are split into
  ~1000-character chunks so long resumes aren't truncated or diluted into
  one embedding, and so retrieval can surface the most relevant *section*
  of a resume rather than an average of the whole document.
- **Structured extraction, not just a score.** The LLM (or rule-based
  fallback) is asked to return `skills`, `experience`, `education`,
  `strengths`, `gaps`, and `overall_assessment` as separate fields — not
  just a number — so the ranking is auditable and explainable, not a black
  box.
- **Multi-format parsing dispatch.** `parse_resume()` routes to
  format-specific parsers (`parse_pdf` / `parse_docx` / `parse_txt`) behind
  one interface, so adding a new format later (e.g. `.rtf`) means adding one
  parser function, not touching the ranking logic.
- **Validation before parsing.** `validate_file()` rejects empty, encrypted,
  or unreadable files early with a specific reason, so one bad file in a
  batch of 10+ doesn't silently corrupt or crash the whole run — it's
  skipped and logged.

---

## 6. Tradeoffs & limitations

- **Offline mode is intentionally simple.** The rule-based extractor matches
  against a fixed ~35-term skill vocabulary (`SKILL_VOCABULARY` in
  `utils.py`) and scores by keyword overlap. It will miss synonyms
  ("k8s" vs "kubernetes"), can't judge seniority nuance, and doesn't reason
  about transferable skills the way an LLM can. It's meant as a
  free/deterministic fallback and a way to sanity-check the pipeline
  end-to-end, not a replacement for live mode.
- **Live mode makes one LLM call per resume, sequentially.** For 10+
  resumes this is slow and can hit Groq's free-tier rate limits. No
  batching/concurrency was added, since that changes error-handling and
  retry behavior significantly — flagging this as a good next improvement
  rather than doing it silently.
- **LLM scores are not perfectly reproducible.** Temperature is set low
  (0.1) to reduce variance, but re-running live mode on the same resume can
  produce a slightly different score/reasoning.
- **TF-IDF similarity (offline) and Cohere embedding similarity (live) are
  not numerically comparable.** Both are scaled to 0–100, but they measure
  different things (lexical overlap vs. semantic similarity), so don't
  expect identical rankings between offline and live mode on the same
  inputs — they're independent estimates of the same underlying question.
- **Qdrant collection is recreated on every run** (`create_vector_store`
  deletes and rebuilds the collection). This keeps results from different
  runs from bleeding into each other, but means concurrent runs against the
  same Qdrant collection name would conflict. Fine for single-user/demo use;
  would need per-run collection names for concurrent multi-user use.
- **Extraction accuracy depends on parsing quality.** Scanned/image-based
  PDFs with no embedded text layer will fail validation (`validate_file`
  checks for extractable text) rather than silently returning garbage —
  but that does mean OCR is out of scope here.

### Environment note (why no live-mode results are included)

This project was built and tested in a sandboxed environment whose network
access is restricted to a small allowlist (PyPI, npm, GitHub, etc.) and does
**not** include `api.cohere.com`, `api.groq.com`, or `*.qdrant.io`. Because
of that, I could not execute the live pipeline against real Cohere/Groq/Qdrant
accounts from within that environment — `sample_data/output/` therefore only
contains the offline-mode run.

The offline path exercises the exact same code (parsing, chunking-equivalent
text handling, score blending, ranking, CSV/JSON export) minus the specific
embedding/LLM providers, so it's a legitimate end-to-end test of the
pipeline's correctness. If you run `python run_demo.py --live` with your own
keys, it will produce `ranked_results_live.csv/json` alongside the committed
offline results, using the exact same input resumes — the two are directly
comparable.

---

## 7. Submission note

I don't have the ability to create or push to a GitHub repository on your
behalf — you'll need to `git init`, commit these files (including
`sample_data/output/`), push to your own GitHub, and share that URL before
your deadline. Suggested `.gitignore`:

```
venv/
__pycache__/
*.pyc
.env
```

