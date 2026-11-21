"""Annotation workflow tests use synthetic declarations, not real human labels."""
from __future__ import annotations

import copy
import json

import pytest

from evaluation.annotations import (
    export_tasks, main, review_tasks, validate_tasks, weighted_kappa,
)


@pytest.fixture
def dataset(tmp_path):
    rows = [
        {"query_id": "Q001", "query": "राम अयोध्या", "query_language": "hi",
         "passage_id": "P1", "text": "Rama is associated with Ayodhya.",
         "query_relevance": 3, "indian_context_relevance": 3, "rank": 1,
         "final_score": 0.99,
         "source_metadata": {"url": "https://example.org/book", "source_type": "book"}},
        {"query_id": "Q001", "query": "राम अयोध्या", "passage_id": "P2",
         "text": "An unrelated weather forecast.",
         "query_relevance": 0, "indian_context_relevance": 0,
         "extraction_metadata": {"page_number": 2, "method": "ocr", "ocr_confidence": 0.7}},
    ]
    path = tmp_path / "sample.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def completed(tasks, reviewer="SyntheticUnitTestReviewerA", labels=None):
    rows = copy.deepcopy(tasks)
    for i, row in enumerate(rows):
        row.update(review_status="complete", annotator_id=reviewer,
                   review_method="human_declared", human_review_attested=True,
                   reviewed_at="2026-11-12", review_note="Synthetic test fixture.")
        row["query_relevance"], row["indian_context_relevance"] = labels[i] if labels else (2, 3)
        if row.get("translation_source_query_id"):
            row["translation_status"] = "approved"
    return rows


def test_export_is_blinded_pending_and_does_not_modify_source(dataset):
    original = dataset.read_bytes()
    tasks = export_tasks(dataset)
    assert dataset.read_bytes() == original
    assert len(tasks) == 2
    for task in tasks:
        assert task["origin"] == "existing_sample_unverified"
        assert task["query_language"] == "hi"
        assert task["query_family"] == "Q001"
        assert task["review_status"] == "pending"
        assert task["query_relevance"] is None
        assert task["indian_context_relevance"] is None
        assert task["annotator_id"] is None
        assert task["human_review_attested"] is False
        assert "rank" not in task and "final_score" not in task
    assert tasks == export_tasks(dataset, seed=42)
    assert len(validate_tasks(tasks)) == 2


def test_roman_templates_keep_family_and_never_inherit_labels(dataset):
    tasks = export_tasks(dataset, include_roman=True)
    roman = [task for task in tasks if task["query_language"] == "hi-Latn"]
    assert len(tasks) == 4 and len(roman) == 2
    assert all(task["query_id"] == "Q001-ROMAN" for task in roman)
    assert all(task["query_family"] == task["translation_source_query_id"] == "Q001" for task in roman)
    assert all(task["translation_status"] == "draft_pending_review" for task in roman)
    assert all(task["query_relevance"] is None for task in roman)
    reviews = completed(tasks)
    for row in reviews:
        if row["query_language"] == "hi-Latn":
            row["translation_status"] = "draft_pending_review"
    with pytest.raises(ValueError, match="translation"):
        review_tasks(tasks, [reviews])


def test_pending_and_partial_reviews_cannot_produce_benchmark(dataset):
    tasks = export_tasks(dataset)
    pending = review_tasks(tasks, [copy.deepcopy(tasks)])
    assert not pending["report"]["reviewed_benchmark_ready"]
    assert pending["report"]["pending_task_count"] == 2
    assert pending["benchmark"] == []
    partial = review_tasks(tasks, [completed(tasks[:1])])
    assert partial["report"]["pending_task_count"] == 1
    assert partial["benchmark"] == []


def test_one_declared_reviewer_is_explicitly_single_review(dataset):
    tasks = export_tasks(dataset)
    result = review_tasks(tasks, [completed(tasks)])
    assert result["report"]["reviewed_benchmark_ready"]
    assert result["report"]["benchmark_review_level"] == "single_declared_reviewer"
    assert result["report"]["agreement"] is None
    assert result["report"]["human_identity_verified_by_tool"] is False
    assert result["report"]["source_authenticity_verified_by_tool"] is False
    assert len(result["benchmark"]) == len(tasks)
    for row in result["benchmark"]:
        assert row["annotation_metadata"]["review_method"] == "human_declared"
        assert row["annotation_metadata"]["human_review_attested"] is True
        assert row["annotation_metadata"]["origin"] == "existing_sample_unverified"
        assert row["query_family"] == "Q001"


@pytest.mark.parametrize("field,value", [
    ("query_relevance", True), ("query_relevance", -1),
    ("indian_context_relevance", 4), ("query_relevance", 2.0),
    ("query_relevance", None), ("review_method", "automatic"),
    ("human_review_attested", False), ("reviewed_at", None),
    ("annotator_id", "TODO"),
    ("annotator_id", "<a consistent reviewer ID or pseudonym>"),
])
def test_invalid_labels_or_automatic_provenance_are_rejected(dataset, field, value):
    tasks = export_tasks(dataset)
    rows = completed(tasks)
    rows[0][field] = value
    with pytest.raises(ValueError):
        review_tasks(tasks, [rows])


def test_pending_rows_cannot_carry_labels_or_completed_claims(dataset):
    tasks = export_tasks(dataset)
    rows = copy.deepcopy(tasks)
    rows[0]["query_relevance"] = 2
    with pytest.raises(ValueError, match="pending"):
        review_tasks(tasks, [rows])


def test_reviewer_ids_must_be_distinct_and_consistent(dataset):
    tasks = export_tasks(dataset)
    first = completed(tasks)
    with pytest.raises(ValueError, match="distinct"):
        review_tasks(tasks, [first, copy.deepcopy(first)])
    mixed = completed(tasks)
    mixed[1]["annotator_id"] = "SyntheticUnitTestReviewerB"
    with pytest.raises(ValueError, match="one reviewer file"):
        review_tasks(tasks, [mixed])


def test_changed_task_identity_and_repeated_reviews_are_rejected(dataset):
    tasks = export_tasks(dataset)
    changed = completed(tasks)
    changed[0]["text"] = "A different passage."
    with pytest.raises(ValueError, match="immutable"):
        review_tasks(tasks, [changed])
    rows = completed(tasks)
    with pytest.raises(ValueError, match="repeated"):
        review_tasks(tasks, [[rows[0], rows[0]]])
    changed_tasks = copy.deepcopy(tasks)
    changed_tasks[0]["query"] = "A changed query."
    with pytest.raises(ValueError, match="identity changed"):
        validate_tasks(changed_tasks)


def test_two_reviewers_need_explicit_adjudication_of_disagreement(dataset):
    tasks = export_tasks(dataset)
    first = completed(tasks, "SyntheticUnitTestReviewerA")
    second = completed(tasks, "SyntheticUnitTestReviewerB")
    second[0]["query_relevance"] = 1
    result = review_tasks(tasks, [first, second])
    assert result["report"]["disagreement_count"] == 1
    assert result["report"]["unresolved_disagreement_count"] == 1
    assert not result["report"]["reviewed_benchmark_ready"]
    assert result["benchmark"] == []
    adjudication = {
        "task_id": tasks[0]["task_id"], "query_relevance": 2,
        "indian_context_relevance": 3, "adjudicator_id": "SyntheticUnitTestAdjudicator",
        "review_status": "complete", "review_method": "human_declared",
        "human_review_attested": True, "reviewed_at": "2026-11-12",
        "reason": "Synthetic adjudication used only for a software test.",
    }
    resolved = review_tasks(tasks, [first, second], [adjudication])
    assert resolved["report"]["reviewed_benchmark_ready"]
    assert resolved["report"]["resolved_disagreement_count"] == 1
    assert resolved["disagreements"] == []
    assert any(row["annotation_metadata"]["adjudicator_id"] == "SyntheticUnitTestAdjudicator"
               for row in resolved["benchmark"])
    adjudication["adjudicator_id"] = "SyntheticUnitTestReviewerA"
    with pytest.raises(ValueError, match="must differ"):
        review_tasks(tasks, [first, second], [adjudication])


def test_weighted_kappa_has_fixed_four_grade_scale():
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 3, 3]) == pytest.approx(11 / 12)
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1
    assert weighted_kappa([], []) is None
    assert weighted_kappa([2, 2], [2, 2]) is None
    with pytest.raises(ValueError):
        weighted_kappa([0], [])
    with pytest.raises(ValueError):
        weighted_kappa([True], [1])


def test_cli_pending_review_writes_report_without_reviewed_dataset(dataset, tmp_path):
    output = tmp_path / "tasks.jsonl"
    assert main(["export", str(dataset), "--output", str(output)]) == 0
    report = tmp_path / "review-report"
    assert main(["review", "--tasks", str(output), "--reviewer", str(output),
                 "--output", str(report)]) == 0
    assert (report / "review_report.json").is_file()
    assert (report / "reviews.jsonl").read_text(encoding="utf-8") == ""
    assert not (report / "reviewed_dataset.jsonl").exists()
    metadata = json.loads((report / "review_report.json").read_text(encoding="utf-8"))
    assert metadata["pending_task_count"] == 2
    assert "sha256" in metadata["input_files"]["tasks"]
    with pytest.raises(SystemExit):
        main(["export", str(dataset), "--output", str(output)])
