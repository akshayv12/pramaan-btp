"""Task 3 output -> Task 4 handoff. Inputs follow the real task3_output.json shape."""
from __future__ import annotations
import copy
import json
from pathlib import Path

import pytest

import process_task3_output as handoff
from contracts import validate_task5_package

ROOT = Path(__file__).resolve().parents[1]
SAVED_TASK3 = ROOT / "task2_search_collect" / "task3_output.json"
LEXICAL = {"relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0},
           "evidence": {"reference_date": "2026-11-19"},
           "output": {"include_source_support": True}}
Q1 = "भारतीय कृषि पर हरित क्रांति का क्या प्रभाव पड़ा?"
Q2 = "What is the Green Revolution?"


def passage(doc, n, text, **changes):
    return {"document_id": doc, "passage_id": f"{doc}_p1_chunk{n}", "source_file": f"{doc}.html",
            "source_path": f"downloads\\html_pages\\{doc}.html", "page_number": None,
            "extraction_method": "DIRECT_EXTRACTION", "ocr_engine": None, "detected_script": "Latin",
            "detected_language": "English", "ocr_confidence": 100.0, "reliability": "HIGH",
            "text": text, "word_count": len(text.split()), **changes}


def document(doc, query, passages, status="processed", **source):
    record = {"doc_id": doc, "url": f"https://example.org/{doc}", "canonical_url": f"https://example.org/{doc}",
              "domain": "example.org", "title": f"Title {doc}", "author": None, "pub_date": None,
              "source_queries": [query], "original_query": query, "query_slug": "slug-" + doc[:1],
              "content_type": "html", "needs_ocr": False, "extraction_status": "success",
              "licence": "unknown", "licence_category": "unknown", "source_type": "commercial",
              "parent_doc_id": None, "source_metadata": {"language": "en"}, **source}
    item = {"document_id": doc, "source": record, "passages": passages,
            "total_passages": len(passages), "status": status}
    if status == "skipped":
        item["reason"] = "fetch_failed"
    return item


def task3_output(documents, original_query=Q2):
    counts = {s: sum(d["status"] == s for d in documents) for s in handoff.DOCUMENT_STATUSES}
    return {"task3_output": {
        "schema_version": "1.0", "metadata": {"original_query": original_query},
        "documents": documents,
        "summary": {**counts, "total_documents": len(documents),
                    "total_passages": sum(d["total_passages"] for d in documents)}}}


def sample():
    return task3_output([
        document("a1", Q2, [passage("a1", 1, "The Green Revolution raised wheat yields in Punjab."),
                            passage("a1", 2, "High-yielding seed varieties and irrigation spread in India.")],
                 source_type="government", author="Ministry", pub_date="2024-10-18"),
        document("b2", Q2, [passage("b2", 1, "हरित क्रांति: भारत की कृषि", page_number=1,
                                    extraction_method="OCR", ocr_engine="TESSERACT_BASELINE",
                                    ocr_confidence=64.89, detected_script="Devanagari", reliability="MEDIUM")],
                 content_type="image", needs_ocr=True, source_type="academic",
                 pub_date="2025-01-03T10:00:00Z", source_metadata={"language": "hi"}),
        document("c3", Q2, [], status="skipped"),
        document("d4", Q2, [], status="empty"),
    ])


def test_mapping_preserves_provenance_and_converts_task3_fields():
    payload = sample()
    before = copy.deepcopy(payload)
    [package] = handoff.task3_output_to_packages(payload)
    assert payload == before
    assert package["query"] == Q2 and package["provenance"]["kind"] == "task3_output"
    assert package["provenance"]["document_status_counts"] == {
        "processed": 2, "duplicate": 0, "empty": 1, "skipped": 1, "failed": 0}
    assert [p["passage_id"] for p in package["passages"]] == ["a1_p1_chunk1", "a1_p1_chunk2", "b2_p1_chunk1"]
    direct, _, scan = package["passages"]
    # Task 3's conventional 100 for direct text is not a measurement and is omitted.
    assert "ocr_confidence" not in direct["extraction_metadata"]
    assert direct["extraction_metadata"]["method"] == "DIRECT_EXTRACTION"
    assert "page_number" not in direct["extraction_metadata"]
    expected = {
        "document_id": "a1", "url": "https://example.org/a1", "title": "Title a1", "author": "Ministry",
        "publication_date": "2024-10-18", "source_type": "institutional", "task2_source_type": "government",
        "language": "en", "content_type": "html"}
    assert {key: direct["source_metadata"].get(key) for key in expected} == expected
    assert scan["extraction_metadata"]["ocr_confidence"] == pytest.approx(0.6489)
    assert scan["extraction_metadata"]["page_number"] == 1
    assert scan["extraction_metadata"]["task3_reliability"] == "MEDIUM"
    assert scan["source_metadata"]["source_type"] == "academic"
    assert scan["source_metadata"]["publication_date"] == "2025-01-03"
    assert scan["source_metadata"]["language"] == "hi"


def test_unmapped_source_types_stay_unknown():
    [package] = handoff.task3_output_to_packages(task3_output([
        document("a1", Q2, [passage("a1", 1, "Text")], source_type="commercial")]))
    meta = package["passages"][0]["source_metadata"]
    assert "source_type" not in meta and meta["task2_source_type"] == "commercial"


def test_multiple_questions_are_ranked_separately(fake_semantic):
    payload = task3_output([
        document("a1", Q1, [passage("a1", 1, "हरित क्रांति ने भारतीय कृषि में गेहूं उत्पादन बढ़ाया।")]),
        document("b2", Q2, [passage("b2", 1, "The Green Revolution in India increased wheat output.")]),
        document("c3", Q1, [], status="failed"),
    ], original_query=[Q1, Q2])
    result = handoff.rank_task3_output(payload, config=LEXICAL, top_k=5)["task4_output"]
    assert [q["query"] for q in result["queries"]] == [Q1, Q2]
    assert [[r["passage_id"] for r in q["ranked_evidence"]] for q in result["queries"]] == [
        ["a1_p1_chunk1"], ["b2_p1_chunk1"]]
    assert result["queries"][0]["provenance"]["document_status_counts"]["failed"] == 1
    assert result["summary"] == {"questions": 2, "questions_with_evidence": 2,
                                 "candidate_passages": 2, "ranked_passages": 2}
    for envelope in result["queries"]:
        assert validate_task5_package(envelope)
        assert envelope["is_simulated"] is False
        assert envelope["integration"]["task3"] == "supplied_package"


def test_ranked_output_keeps_scores_and_traceability(fake_semantic):
    result = handoff.rank_task3_output(sample(), config=LEXICAL, top_k=2)["task4_output"]
    [envelope] = result["queries"]
    assert envelope["status"] == "ok" and envelope["candidate_count"] == 3
    assert [r["rank"] for r in envelope["ranked_evidence"]] == [1, 2]
    for row in envelope["ranked_evidence"]:
        for key in ("query_relevance", "indian_context_relevance", "source_signal", "final_score"):
            assert 0 <= row[key] <= 1
        assert row["provenance"]["source_metadata"]["document_id"] in {"a1", "b2"}
        assert "source_support" in row
    assert envelope["traceability"]["references_available"] is True
    assert result["task3_summary"]["total_documents"] == 4
    assert result["ranking"]["top_k"] == 2


def test_ocr_passage_uses_converted_confidence(fake_semantic):
    payload = task3_output([document("b2", Q2, [passage(
        "b2", 1, "Green Revolution", extraction_method="OCR", ocr_confidence=40.0, page_number=2)])])
    [row] = handoff.rank_task3_output(payload, config=LEXICAL)["task4_output"]["queries"][0]["ranked_evidence"]
    assert row["feature_breakdown"]["evidence"]["ocr_score"] == 0.4


def test_question_without_passages_reports_no_evidence():
    result = handoff.rank_task3_output(task3_output([document("c3", Q2, [], status="skipped")]),
                                       config=LEXICAL)["task4_output"]
    [envelope] = result["queries"]
    assert envelope["status"] == "no_relevant_evidence" and envelope["ranked_evidence"] == []
    assert validate_task5_package(envelope)


def test_empty_collection_keeps_declared_question():
    result = handoff.rank_task3_output(task3_output([]), config=LEXICAL)["task4_output"]
    assert [q["query"] for q in result["queries"]] == [Q2]
    assert result["summary"]["questions_with_evidence"] == 0


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["task3_output"].update(schema_version="9"), "schema_version"),
    (lambda p: p.update(task3_output=[]), "schema_version"),
    (lambda p: p["task3_output"]["documents"][0].update(status="odd"), "unknown status"),
    (lambda p: p["task3_output"]["documents"][0]["passages"][0].update(
        extraction_method="OCR", ocr_confidence=140), "0-100"),
    (lambda p: p["task3_output"]["documents"][0]["passages"][0].update(page_number=0), "page_number"),
    (lambda p: p["task3_output"]["documents"][1]["passages"][0].update(passage_id="a1_p1_chunk1"),
     "duplicate passage_id"),
])
def test_invalid_task3_output_is_rejected(mutate, message):
    payload = sample()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        handoff.rank_task3_output(payload, config=LEXICAL)


def test_document_without_query_in_multi_question_batch_is_rejected():
    payload = task3_output([document("a1", Q2, [passage("a1", 1, "Text")])], original_query=[Q1, Q2])
    del payload["task3_output"]["documents"][0]["source"]["original_query"]
    with pytest.raises(ValueError, match="original_query"):
        handoff.task3_output_to_packages(payload)


def test_relative_model_paths_resolve_against_project_root():
    config = {"semantic_model": {"name": "configs", "cache_folder": "missing-folder"}}
    resolved = handoff._resolve_model_paths(config)
    assert resolved["semantic_model"]["name"] == str((ROOT / "configs").resolve())
    assert resolved["semantic_model"]["cache_folder"] == "missing-folder"
    assert config["semantic_model"]["name"] == "configs"


def test_cli_writes_output_and_protects_previous_result_on_error(tmp_path):
    source = tmp_path / "task3_output.json"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(LEXICAL), encoding="utf-8")
    source.write_text(json.dumps(sample(), ensure_ascii=False), encoding="utf-8")
    assert handoff.main([str(source), "--config", str(config), "--lexical-only", "--top-k", "1"]) == 0
    output = tmp_path / "task4_output.json"
    first = json.loads(output.read_text(encoding="utf-8"))["task4_output"]
    assert len(first["queries"][0]["ranked_evidence"]) == 1
    source.write_text(json.dumps({"task3_output": {"schema_version": "9"}}), encoding="utf-8")
    previous = output.read_bytes()
    assert handoff.main([str(source), "--config", str(config), "--lexical-only"]) == 2
    assert output.read_bytes() == previous
    with pytest.raises(SystemExit):
        handoff.main([str(source), "--top-k", "-1"])
    with pytest.raises(ValueError, match="differ"):
        handoff.process_task3_output(source, source, config=LEXICAL)


@pytest.mark.skipif(not SAVED_TASK3.is_file(), reason="saved Task 3 output is not present")
def test_saved_task3_output_flows_through_task4(fake_semantic):
    payload = json.loads(SAVED_TASK3.read_text(encoding="utf-8"))
    passages = {p["passage_id"]: p for d in payload["task3_output"]["documents"] for p in d["passages"]}
    result = handoff.rank_task3_output(payload, config=LEXICAL, top_k=5)["task4_output"]
    assert result["summary"]["candidate_passages"] == sum(
        d["total_passages"] for d in payload["task3_output"]["documents"] if d["status"] == "processed")
    for envelope in result["queries"]:
        assert validate_task5_package(envelope)
        for row in envelope["ranked_evidence"]:
            original = passages[row["passage_id"]]
            assert row["text"] == original["text"].strip()
            assert row["provenance"]["source_metadata"]["document_id"] == original["document_id"]


@pytest.mark.model
@pytest.mark.skipif(not SAVED_TASK3.is_file(), reason="saved Task 3 output is not present")
def test_saved_task3_output_with_real_model_and_deployment_config():
    payload = json.loads(SAVED_TASK3.read_text(encoding="utf-8"))
    # One question keeps the real-model check short on CPU.
    documents = [d for d in payload["task3_output"]["documents"]
                 if d["source"].get("original_query") == payload["task3_output"]["documents"][0]["source"]["original_query"]]
    payload["task3_output"]["documents"] = documents
    payload["task3_output"]["metadata"]["original_query"] = documents[0]["source"]["original_query"]
    result = handoff.rank_task3_output(payload, config=handoff.load_config(), top_k=5)["task4_output"]
    [envelope] = result["queries"]
    assert envelope["status"] == "ok" and validate_task5_package(envelope)
