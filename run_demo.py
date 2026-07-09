"""
run_demo.py
-----------
Runs the ResumeRanker pipeline end-to-end against the sample job description
and sample resumes in sample_data/, and saves the ranked output as CSV + JSON
into sample_data/output/.

By default this uses OFFLINE mode (TF-IDF + rule-based scoring), which needs
no API keys and no network access — this is what was used to generate the
sample_data/output/ files checked into this repo.

To run the SAME resumes through the live Cohere + Groq + Qdrant pipeline
instead, set COHERE_API_KEY / GROQ_API_KEY / QDRANT_URL / QDRANT_API_KEY in
your environment (or .env file) and run:

    python run_demo.py --live

See README.md for full setup instructions.
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from utils import ResumeRanker

load_dotenv()

BASE_DIR = Path(__file__).parent
JD_PATH = BASE_DIR / "sample_data" / "job_description.txt"
RESUMES_DIR = BASE_DIR / "sample_data" / "resumes"
OUTPUT_DIR = BASE_DIR / "sample_data" / "output"


def load_resume_files():
    files_data = []
    for path in sorted(RESUMES_DIR.iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            files_data.append({
                "name": path.name,
                "content": path.read_bytes()
            })
    return files_data


def main():
    parser = argparse.ArgumentParser(description="Run the Resume Ranker demo")
    parser.add_argument("--live", action="store_true",
                         help="Use live Cohere + Groq + Qdrant APIs instead of offline mode")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    job_desc = JD_PATH.read_text()
    resume_files = load_resume_files()
    print(f"Loaded job description ({len(job_desc)} chars) and {len(resume_files)} resumes:")
    for f in resume_files:
        print(f"  - {f['name']}")

    ranker = ResumeRanker(similarity_weight=0.5, llm_weight=0.5)

    if args.live:
        cohere_key = os.getenv("COHERE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if not cohere_key or not groq_key:
            raise SystemExit(
                "ERROR: --live requires COHERE_API_KEY and GROQ_API_KEY to be set "
                "(in your environment or a .env file). See README.md."
            )
        print("\nRunning LIVE pipeline (Cohere embeddings + Qdrant + Groq LLM)...\n")
        df = ranker.rank_resumes(
            job_desc=job_desc,
            resume_files_data=resume_files,
            cohere_key=cohere_key,
            groq_key=groq_key,
            k=args.top_k
        )
        mode = "live"
    else:
        print("\nRunning OFFLINE pipeline (TF-IDF similarity + rule-based extraction)...\n")
        df = ranker.rank_resumes_offline(
            job_desc=job_desc,
            resume_files_data=resume_files,
            k=args.top_k
        )
        mode = "offline"

    csv_path = OUTPUT_DIR / f"ranked_results_{mode}.csv"
    json_path = OUTPUT_DIR / f"ranked_results_{mode}.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        f.write(ranker.to_json(df, job_desc=job_desc))

    print(f"\nSaved {len(df)} ranked results to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print("\nTop result:")
    print(df[["Rank", "Resume", "Score", "NLP_Similarity_Score", "LLM_Assessment_Score"]].to_string(index=False))


if __name__ == "__main__":
    main()
