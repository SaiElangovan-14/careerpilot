import pytest

from app.services.resume_parser import parse_resume_text
from app.services.retrieval import rrf_fuse
from evals.metrics import recall_at_k, reciprocal_rank

def test_recall_at_k():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=3) == pytest.approx(1 / 3)

def test_reciprocal_rank():
    assert reciprocal_rank([9, 8, 7], {7}) == pytest.approx(1 / 3)
    assert reciprocal_rank([9, 8, 7], {1}) == 0.0

def test_rrf_prefers_jobs_in_both_lists():
    fused = rrf_fuse([[1, 2, 3], [3, 4, 1]])
    top_two = {job_id for job_id,_ in fused[:2]}
    assert top_two =={1, 3}

def test_parser_ignores_skills_inside_other_words():
    parsed = parse_resume_text("I know javascript and digital marketing.")
    assert "javascript" in parsed["skills"]
    assert "java" not in parsed["skills"]
    assert "git" not in parsed ["skills"]
