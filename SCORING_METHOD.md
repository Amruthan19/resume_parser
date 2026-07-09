# Scoring Methodology — AI Resume Ranker

This document explains how each resume gets its final `Score`, as required by the
agent deliverables.

## 1. Overview

Every resume is scored on a **0–100** scale, produced by blending two independent
signals:

| Component | What it measures | Source |
|---|---|---|
| **NLP Similarity Score** | How semantically close the resume text is to the job description | Cohere `embed-english-v3.0` embeddings + cosine similarity, retrieved via Qdrant |
| **LLM Assessment Score** | A qualitative judgement of skills/experience/education fit | Llama-3.3-70B (via Groq), prompted to reason about the match |

```
Final Score = (similarity_weight × NLP_Similarity_Score) + (llm_weight × LLM_Assessment_Score)
```

By default both weights are `0.5`, so the final score is an even split between
"does the resume look similar to the JD in vector space" and "does an LLM,
reading the resume, believe this candidate fits." The weights are adjustable
in the sidebar of the app.

## 2. Step-by-step pipeline

1. **Parsing** — Each uploaded file (PDF, DOCX, or TXT) is converted into plain
   text. PDFs use `PyPDFLoader`, DOCX files use `python-docx` (including table
   content), and TXT files are read directly.

2. **Chunking** — Resume text is split into ~1000-character chunks
   (100-character overlap) using `RecursiveCharacterTextSplitter`, so long
   resumes don't get truncated or diluted in a single embedding.

3. **Embedding + indexing** — Every chunk is embedded with Cohere
   `embed-english-v3.0` (1024-dim) and stored in a Qdrant collection using
   cosine distance.

4. **Retrieval (NLP similarity)** — The job description is embedded once and
   used to query Qdrant for the most similar chunks across all resumes. For
   each resume, the top 5 matching chunks are kept, and their **average
   cosine similarity** (0–1) is scaled to 0–100 to produce the
   `NLP_Similarity_Score`. This is a pure vector-similarity signal — no LLM
   is involved in this number.

5. **Extraction + qualitative scoring (LLM)** — The top-matching chunks for
   each resume are passed to Llama-3.3-70B with a structured prompt that
   asks it to return:
   - `skills` — list of skills found in the resume text
   - `experience` — a short description of relevant experience
   - `education` — degrees/institutions found in the resume text
   - `llm_score` (0–100) — qualitative match score
   - `strengths`, `gaps`, `overall_assessment` — written reasoning

   The model is explicitly instructed to base these fields only on what's
   actually present in the resume text, not to infer or invent information.

6. **Blending** — The final `Score` is the weighted average of
   `NLP_Similarity_Score` and `LLM_Assessment_Score` described above. Both
   sub-scores are also kept in the output so the two signals can be audited
   separately (e.g., a resume could rank high on similarity but the LLM
   flags it as lacking seniority).

7. **Ranking** — All candidates are sorted by final `Score` descending, and
   the top `k` are returned.

## 3. Why blend two signals instead of one?

- **Pure embedding similarity** is fast and language-agnostic but can be
  fooled by keyword-stuffed resumes, or miss a strong candidate who phrases
  things differently than the JD.
- **Pure LLM judgement** is more nuanced (it can recognize equivalent skills,
  seniority signals, etc.) but is slower, costs more per resume, and can be
  inconsistent between runs.

Combining both gives a score that's harder to game with keyword stuffing
alone, while still being explainable through the LLM's written reasoning.

## 4. Score interpretation

| Score | Match Level |
|---|---|
| 90–100 | 🎯 Excellent Match |
| 80–89 | ⭐ Very Good Match |
| 70–79 | ✓ Good Match |
| 60–69 | ↻ Partial Match |
| < 60 | ⚠️ Needs Review |

## 5. Known limitations

- Similarity retrieval quality depends on chunk boundaries; a resume's most
  relevant section could theoretically be split across two chunks.
- The LLM's `llm_score` reflects the model's judgement and, like any LLM
  output, is not deterministic between runs (temperature is set low — `0.1`
  — to reduce this).
- Extracted `skills`/`experience`/`education` reflect only what is stated in
  the resume text; OCR/parsing errors in the source PDF/DOCX will propagate.
