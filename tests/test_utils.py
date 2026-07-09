# tests/test_utils.py
"""
Test suite for the ResumeRanker offline pipeline (parsing, extraction,
scoring, ranking, and export). Runs with zero API keys and zero network
access — pure pytest.

Run with:
    pytest -v
"""
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import ResumeRanker, SmoothException  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
JD_PATH = BASE_DIR / "sample_data" / "job_description.txt"
RESUMES_DIR = BASE_DIR / "sample_data" / "resumes"


@pytest.fixture
def ranker():
    return ResumeRanker(similarity_weight=0.5, llm_weight=0.5)


@pytest.fixture
def sample_job_desc():
    return (
        "Senior Python Developer with Django, FastAPI, PostgreSQL, and AWS "
        "experience. Must know Docker and REST APIs."
    )


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------

def test_weights_normalize_to_one():
    r = ResumeRanker(similarity_weight=1, llm_weight=1)  # 2:1 ratio -> 0.5/0.5
    assert r.similarity_weight == pytest.approx(0.5)
    assert r.llm_weight == pytest.approx(0.5)

    r2 = ResumeRanker(similarity_weight=3, llm_weight=1)  # 3:1 ratio
    assert r2.similarity_weight == pytest.approx(0.75)
    assert r2.llm_weight == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def test_validate_file_rejects_empty(ranker):
    is_valid, msg = ranker.validate_file(b"", "empty.txt")
    assert is_valid is False
    assert "empty" in msg.lower()


def test_validate_file_rejects_unsupported_extension(ranker):
    is_valid, msg = ranker.validate_file(b"some content", "resume.rtf")
    assert is_valid is False
    assert "unsupported" in msg.lower()


def test_validate_file_accepts_valid_txt(ranker):
    content = b"John Doe\nSenior Python Developer with 5 years experience."
    is_valid, msg = ranker.validate_file(content, "resume.txt")
    assert is_valid is True


def test_validate_file_rejects_txt_too_short(ranker):
    is_valid, msg = ranker.validate_file(b"hi", "resume.txt")
    assert is_valid is False


# ---------------------------------------------------------------------------
# Parsing: TXT
# ---------------------------------------------------------------------------

def test_parse_txt_returns_document_with_metadata(ranker):
    content = b"Jane Smith\nPython, Django, AWS. 5 years experience."
    docs = ranker.parse_txt(content, "jane.txt")
    assert len(docs) == 1
    assert "Jane Smith" in docs[0].page_content
    assert docs[0].metadata["source"] == "jane.txt"


def test_parse_resume_dispatches_txt(ranker):
    content = b"Some resume text here that is long enough to pass validation."
    docs = ranker.parse_resume(content, "candidate.txt")
    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "candidate.txt"


def test_parse_resume_raises_on_unsupported_extension(ranker):
    with pytest.raises(Exception):
        ranker.parse_resume(b"content", "resume.rtf")


# ---------------------------------------------------------------------------
# Parsing: DOCX (generated on the fly, no fixture file needed)
# ---------------------------------------------------------------------------

def test_parse_docx_extracts_paragraphs_and_tables(ranker):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Alex Doe - Backend Engineer")
    d.add_paragraph("5 years of Python and Django experience.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skill"
    table.rows[0].cells[1].text = "AWS"
    buf = io.BytesIO()
    d.save(buf)
    content = buf.getvalue()

    docs = ranker.parse_docx(content, "alex.docx")
    assert len(docs) == 1
    assert "Alex Doe" in docs[0].page_content
    assert "AWS" in docs[0].page_content  # table content was pulled in
    assert docs[0].metadata["source"] == "alex.docx"


# ---------------------------------------------------------------------------
# Offline skill extraction (regression test for the word-boundary bug:
# "go" must NOT match inside "django", "java" must NOT match inside
# "javascript")
# ---------------------------------------------------------------------------

def test_offline_extract_no_substring_false_positives(ranker, sample_job_desc):
    resume_text = "5 years experience with Django and JavaScript development."
    analysis = ranker._offline_extract(resume_text, sample_job_desc)
    assert "go" not in analysis["skills"], "false positive: 'go' matched inside 'django'"
    assert "java" not in analysis["skills"], "false positive: 'java' matched inside 'javascript'"
    assert "django" in analysis["skills"]


def test_offline_extract_finds_real_skills(ranker, sample_job_desc):
    resume_text = "Expert in Python, Django, FastAPI, PostgreSQL, AWS, and Docker."
    analysis = ranker._offline_extract(resume_text, sample_job_desc)
    for skill in ["python", "django", "fastapi", "postgresql", "aws", "docker"]:
        assert skill in analysis["skills"]


def test_offline_extract_score_reflects_overlap(ranker, sample_job_desc):
    strong_resume = "Python, Django, FastAPI, PostgreSQL, AWS, Docker, REST API expert."
    weak_resume = "JavaScript, React, and CSS developer with no backend experience."

    strong_analysis = ranker._offline_extract(strong_resume, sample_job_desc)
    weak_analysis = ranker._offline_extract(weak_resume, sample_job_desc)

    assert strong_analysis["llm_score"] > weak_analysis["llm_score"]


def test_offline_extract_years_of_experience(ranker, sample_job_desc):
    resume_text = "I have 7 years of experience building Python services."
    analysis = ranker._offline_extract(resume_text, sample_job_desc)
    assert "7" in analysis["experience"]


def test_offline_extract_education_detection(ranker, sample_job_desc):
    resume_text = "B.Sc. Computer Science, MIT. 5 years Python experience."
    analysis = ranker._offline_extract(resume_text, sample_job_desc)
    assert analysis["education"] != "No degree information detected"


# ---------------------------------------------------------------------------
# End-to-end offline ranking
# ---------------------------------------------------------------------------

def test_rank_resumes_offline_orders_by_relevance(ranker, sample_job_desc):
    strong = {
        "name": "strong_fit.txt",
        "content": b"Senior Python Developer. 8 years experience with Django, "
                    b"FastAPI, PostgreSQL, AWS, Docker, and REST API design."
    }
    weak = {
        "name": "weak_fit.txt",
        "content": b"Graphic designer with 3 years experience in Photoshop, "
                    b"Illustrator, and print layout design."
    }

    df = ranker.rank_resumes_offline(
        job_desc=sample_job_desc,
        resume_files_data=[weak, strong],  # weak listed first on purpose
        k=10
    )

    assert len(df) == 2
    assert df.iloc[0]["Resume"] == "strong_fit.txt"  # ranked first despite input order
    assert df.iloc[0]["Score"] > df.iloc[1]["Score"]
    assert df.iloc[0]["Rank"] == 1
    assert df.iloc[1]["Rank"] == 2


def test_rank_resumes_offline_raises_with_no_files(ranker, sample_job_desc):
    with pytest.raises(Exception):
        ranker.rank_resumes_offline(job_desc=sample_job_desc, resume_files_data=[], k=10)


def test_rank_resumes_offline_skips_invalid_files_gracefully(ranker, sample_job_desc):
    valid = {
        "name": "valid.txt",
        "content": b"Python Django FastAPI PostgreSQL AWS Docker developer with experience."
    }
    empty = {"name": "empty.txt", "content": b""}

    df = ranker.rank_resumes_offline(
        job_desc=sample_job_desc,
        resume_files_data=[valid, empty],
        k=10
    )
    assert len(df) == 1
    assert df.iloc[0]["Resume"] == "valid.txt"


def test_rank_resumes_offline_score_matches_weight_formula(sample_job_desc):
    r = ResumeRanker(similarity_weight=0.7, llm_weight=0.3)
    resume = {
        "name": "candidate.txt",
        "content": b"Python Django FastAPI PostgreSQL AWS Docker developer, 5 years."
    }
    df = r.rank_resumes_offline(job_desc=sample_job_desc, resume_files_data=[resume], k=10)
    row = df.iloc[0]

    expected = round(0.7 * row["NLP_Similarity_Score"] + 0.3 * row["LLM_Assessment_Score"], 1)
    assert row["Score"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def test_to_json_produces_valid_structured_output(ranker, sample_job_desc):
    resume = {
        "name": "candidate.txt",
        "content": b"Python Django FastAPI PostgreSQL AWS Docker developer, 5 years."
    }
    df = ranker.rank_resumes_offline(job_desc=sample_job_desc, resume_files_data=[resume], k=10)
    json_str = ranker.to_json(df, job_desc=sample_job_desc)
    payload = json.loads(json_str)

    assert payload["job_description"] == sample_job_desc
    assert payload["total_candidates"] == 1
    assert "scoring_method" in payload
    assert payload["scoring_method"]["nlp_similarity_weight"] == pytest.approx(0.5)
    assert len(payload["results"]) == 1
    assert payload["results"][0]["Resume"] == "candidate.txt"


# ---------------------------------------------------------------------------
# Integration test against the real committed sample data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not JD_PATH.exists(), reason="sample_data/job_description.txt not found")
def test_integration_sample_data_end_to_end():
    """
    Runs the exact same pipeline as `python run_demo.py` against the
    committed sample_data/ resumes, and checks the known-good ranking
    (Sara Patel, the staff-level strongest-fit candidate, should rank #1;
    Alex Kim, the frontend-only weakest-fit candidate, should rank last).
    """
    job_desc = JD_PATH.read_text()
    resume_files = []
    for path in sorted(RESUMES_DIR.iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            resume_files.append({"name": path.name, "content": path.read_bytes()})

    assert len(resume_files) >= 5, "expected at least 5 sample resumes"

    ranker = ResumeRanker(similarity_weight=0.5, llm_weight=0.5)
    df = ranker.rank_resumes_offline(job_desc=job_desc, resume_files_data=resume_files, k=10)

    assert len(df) == len(resume_files)
    # scores should be sorted descending
    scores = df["Score"].tolist()
    assert scores == sorted(scores, reverse=True)

    top_resume = df.iloc[0]["Resume"]
    bottom_resume = df.iloc[-1]["Resume"]
    assert "sara_patel" in top_resume, f"expected sara_patel to rank #1, got {top_resume}"
    assert "alex_kim" in bottom_resume, f"expected alex_kim to rank last, got {bottom_resume}"
