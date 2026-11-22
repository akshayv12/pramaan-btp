"""Task 4 proposed-interface tests. Inputs and consumer are simulated."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from contracts import (
    SCHEMA_VERSION, normalize_task3_package, validate_task3_package,
    build_task5_package, validate_task5_package, task5_consumer_stub,
)
from task4_rank_evidence import rank_evidence

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/fixtures/task3_example.json"


def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_constructed_and_represents_multiple_input_types():
    package = fixture()
    assert package["provenance"]["kind"] == "constructed_example"
    assert len(package["passages"]) > 10
    assert {p["source_metadata"]["document_type"] for p in package["passages"]} == {
        "html", "text_pdf", "scanned_pdf", "image",
    }
    assert all(".invalid/" in p["url"] for p in package["passages"])
    assert any(p["passage_id"] == "FOREIGN-REL" for p in package["passages"])
    assert any(p["passage_id"] == "INDIA-UNREL" for p in package["passages"])


def test_normalization_preserves_traceability_and_maps_extraction_method():
    package = fixture()
    before = copy.deepcopy(package)
    normalized = normalize_task3_package(package)
    assert package == before
    scan = next(p for p in normalized["passages"] if p["passage_id"] == "SCAN-HI")
    assert scan["document_id"] == "DOC-SCAN"
    assert scan["source_id"] == "SOURCE-DOC-SCAN"
    assert scan["language"] == "hi"
    assert scan["source_metadata"]["title"]
    assert scan["source_metadata"]["url"].startswith("https://fixtures.invalid/")
    assert scan["extraction_metadata"]["method"] == "ocr"
    assert scan["extraction_metadata"]["ocr_confidence"] == 0.92
    assert scan["extraction_metadata"]["page_id"] == "DOC-SCAN-PAGE-1"
    image = next(p for p in normalized["passages"] if p["passage_id"] == "IMAGE-HI")
    assert image["extraction_metadata"]["image_id"] == "DOC-IMAGE-IMAGE-1"


def test_existing_nested_passage_format_is_supported():
    package = fixture()
    package["passages"] = [{
        "passage_id": "NESTED", "text": "A prepared passage.",
        "source_metadata": {"document_id": "D", "source_id": "S", "url": "https://fixtures.invalid/n"},
        "extraction_metadata": {"extraction_method": "ocr", "ocr_confidence": 0.6},
    }]
    normalized = normalize_task3_package(package)
    passage = normalized["passages"][0]
    assert passage["extraction_metadata"]["method"] == "ocr"
    assert "extraction_method" not in passage["extraction_metadata"]
    assert passage["source_metadata"]["document_id"] == "D"


@pytest.mark.parametrize("mutation", [
    {"extraction_method": "direct", "extraction_metadata": {"method": "ocr"}},
    {"extraction_method": "ocr", "extraction_metadata": {"extraction_method": "direct"}},
    {"document_id": "D1", "source_metadata": {"document_id": "D2"}},
    {"url": "https://fixtures.invalid/a", "source_metadata": {"url": "https://fixtures.invalid/b"}},
    {"ocr_confidence": 0.5, "extraction_metadata": {"ocr_confidence": 0.9}},
])
def test_conflicting_flat_and_nested_aliases_are_rejected(mutation):
    package = fixture()
    package["passages"] = [dict({"passage_id": "P", "text": "text"}, **mutation)]
    with pytest.raises(ValueError, match="conflicting"):
        normalize_task3_package(package)


@pytest.mark.parametrize("version", [None, "0.1", "2.0.0"])
def test_unknown_input_schema_versions_are_rejected(version):
    package = fixture()
    package["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        validate_task3_package(package)


def test_missing_ocr_confidence_is_not_misclassified_as_direct(fake_semantic):
    package = fixture()
    package["passages"] = [{
        "passage_id": "SCAN", "text": "A scanned manuscript passage.",
        "document_id": "D", "extraction_method": "ocr",
    }]
    normalized = normalize_task3_package(package)
    ranked = rank_evidence(package["query"], normalized["passages"])
    assert ranked[0]["feature_breakdown"]["evidence"]["ocr_score"] == 0.5


def test_local_task4_to_consumer_stub_preserves_scope_and_references(fake_semantic):
    normalized = normalize_task3_package(fixture())
    config = {"output": {"include_source_support": True},
              "evidence": {"reference_date": "2026-11-12"}}
    ranked = rank_evidence(normalized["query"], normalized["passages"], config=config)
    package = build_task5_package(normalized["query"], ranked[:4], provenance=normalized["provenance"])
    assert package["schema_version"] == SCHEMA_VERSION
    assert package["is_simulated"] is True
    assert package["status"] == "ok"
    assert package["traceability"]["selected_evidence_ids"] == [p["passage_id"] for p in ranked[:4]]
    assert len(package["traceability"]["selected_references"]) == 4
    assert package["traceability"]["references_available"] is True
    assert package["traceability"]["source_support_scope"] == "all_input_candidates_before_filtering"
    assert package["traceability"]["source_independence_established"] is False
    assert package["traceability"]["text_similarity_establishes_agreement"] is False
    assert package["ranked_evidence"][0]["source_support"] == ranked[0]["source_support"]
    result = task5_consumer_stub(package)
    assert result["accepted"] is True
    assert result["actual_task5_available"] is False
    assert "summary" not in result and "confidence" not in result


def test_no_evidence_is_explicit_without_fabricated_answer():
    package = fixture()
    result = build_task5_package(package["query"], [], provenance=package["provenance"])
    assert result["status"] == "no_relevant_evidence"
    assert result["ranked_evidence"] == []
    assert result["traceability"]["references_available"] is False
    assert task5_consumer_stub(result)["received_evidence_ids"] == []
    assert "summary" not in result and "confidence" not in result


def test_output_validation_rejects_inconsistent_selected_ids():
    source = fixture()
    result = build_task5_package(source["query"], [], provenance=source["provenance"])
    result["traceability"]["selected_evidence_ids"] = ["invented"]
    with pytest.raises(ValueError, match="selected_evidence_ids"):
        validate_task5_package(result)


def test_lexical_demo_cli_writes_valid_simulated_report(tmp_path):
    output = tmp_path / "integration.json"
    proc = subprocess.run([
        sys.executable, "-B", str(ROOT / "examples/integration_demo.py"),
        "--lexical-only", "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["is_simulated"] is True
    assert report["live_handoff_available"] is False
    assert report["candidate_count"] > 10
    assert report["config_used"]["filter"]["enabled"] is True
    assert report["config_used"]["filter"]["threshold"] > 0
    assert validate_task5_package(report["task5_envelope"])


def test_unrelated_query_demo_reports_actual_empty_selection(tmp_path):
    output = tmp_path / "negative.json"
    proc = subprocess.run([
        sys.executable, "-B", str(ROOT / "examples/integration_demo.py"),
        "--lexical-only", "--query", "sourdough sourdough fermentation",
        "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ranked_count"] == 0
    assert report["status"] == "no_relevant_evidence"


def one_ranked_package():
    source = fixture()
    row = {"rank": 1, "passage_id": "P", "text": "A returned passage.",
           "query_relevance": 0.8, "indian_context_relevance": 0.5,
           "source_signal": 0.7, "final_score": 0.6,
           "provenance": {"source_metadata": {"document_id": "D"},
                          "extraction_metadata": {}}}
    return build_task5_package(source["query"], [row], provenance=source["provenance"])


def test_output_validation_rejects_boolean_rank():
    package = one_ranked_package()
    package["ranked_evidence"][0]["rank"] = True
    with pytest.raises(ValueError, match="consecutive ranks"):
        validate_task5_package(package)


def test_output_validation_rejects_mismatched_reference_passage_ids():
    package = one_ranked_package()
    package["traceability"]["selected_references"][0]["passage_id"] = "OTHER"
    with pytest.raises(ValueError, match="selected_references passage IDs"):
        validate_task5_package(package)


def test_wrapper_reports_invalid_rows_as_value_error():
    source = fixture()
    with pytest.raises(ValueError, match="invalid ranked_evidence"):
        build_task5_package(source["query"], [{}], provenance=source["provenance"])
