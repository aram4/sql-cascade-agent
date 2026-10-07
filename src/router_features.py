"""Feature extraction for the router classifier.

Predicts, from a question and its schema, whether the cheap `generate` model is
likely sufficient or whether the strong model is needed — before paying for
either call. Trained in train_router.py on logged results from running both
models over the same question set (see that file for the labeling logic).
"""

from __future__ import annotations

import re

AGG_RE = re.compile(r"\b(average|avg|sum|count|total|how many|number of)\b", re.IGNORECASE)
RANK_RE = re.compile(r"\b(rank|top \d+|highest|lowest|maximum|minimum|most|least)\b", re.IGNORECASE)
GROUP_RE = re.compile(r"\b(each|every|per|group)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(date|year|month|day|between|before|after)\b", re.IGNORECASE)
COMPARISON_RE = re.compile(r"\b(more than|less than|greater|fewer|at least|at most)\b", re.IGNORECASE)


def extract_features(question: dict, schema_text: str) -> dict:
    """question: a BIRD question dict (question, difficulty, evidence, ...).
    schema_text: this db_id's full schema description (same text the generate
    role is prompted with), used only to measure its size."""
    q = question["question"]
    difficulty = question.get("difficulty", "")
    evidence = question.get("evidence") or ""

    return {
        "question_len": len(q),
        "question_word_count": len(q.split()),
        "evidence_len": len(evidence),
        "has_evidence": bool(evidence.strip()),
        "difficulty_simple": int(difficulty == "simple"),
        "difficulty_moderate": int(difficulty == "moderate"),
        "difficulty_challenging": int(difficulty == "challenging"),
        "schema_table_count": schema_text.count("CREATE TABLE"),
        "schema_len": len(schema_text),
        "has_agg_keyword": int(bool(AGG_RE.search(q))),
        "has_rank_keyword": int(bool(RANK_RE.search(q))),
        "has_group_keyword": int(bool(GROUP_RE.search(q))),
        "has_date_keyword": int(bool(DATE_RE.search(q))),
        "has_comparison_keyword": int(bool(COMPARISON_RE.search(q))),
        "and_or_count": q.lower().count(" and ") + q.lower().count(" or "),
    }


FEATURE_NAMES = list(extract_features(
    {"question": "", "difficulty": "", "evidence": ""}, ""
).keys())
