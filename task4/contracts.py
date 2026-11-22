"""Versioned adapters for Tasks 3 -> 4 -> 5; no retrieval or summarization."""
from __future__ import annotations
import copy
import math

SCHEMA_VERSION = "1.0.0"


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _alias(destination, key, value, label):
    if value is None:
        return
    if key in destination and destination[key] is not None and destination[key] != value:
        raise ValueError(f"conflicting aliases for {label}")
    destination[key] = copy.deepcopy(value)


def normalize_task3_package(package):
    """Accept flat Task 3 fields or the existing nested passage format.

    Flat fields are moved into source/extraction metadata. Conflicting aliases
    are rejected, and the original package is not mutated.
    """
    _object(package, "Task 3 package")
    if package.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version; expected {SCHEMA_VERSION}")
    query = _string(package.get("query"), "query")
    provenance = copy.deepcopy(_object(package.get("provenance"), "provenance"))
    if provenance.get("kind") not in ("constructed_example", "task3_output"):
        raise ValueError("provenance.kind must be constructed_example or task3_output")
    _string(provenance.get("description"), "provenance.description")
    if not isinstance(package.get("passages"), list):
        raise ValueError("passages must be a list")
    normalized = []
    for index, original in enumerate(package["passages"]):
        row = _object(original, f"passages[{index}]")
        source = copy.deepcopy(row.get("source_metadata"))
        extraction = copy.deepcopy(row.get("extraction_metadata"))
        source = {} if source is None else _object(source, "source_metadata")
        extraction = {} if extraction is None else _object(extraction, "extraction_metadata")
        for key in ("document_id", "source_id", "url", "title", "language",
                    "source_group_id", "origin_document_id"):
            _alias(source, key, row.get(key), key)
        # Task 4 reads extraction_metadata.method, never extraction_method.
        for value in (row.get("extraction_method"), row.get("method"),
                      extraction.pop("extraction_method", None)):
            _alias(extraction, "method", value, "extraction_method/method")
        for key in ("page_number", "page_id", "image_id", "ocr_confidence"):
            _alias(extraction, key, row.get(key), key)
        item = {
            "passage_id": row.get("passage_id"),
            "text": row.get("text"),
            "source_metadata": source,
            "extraction_metadata": extraction,
        }
        for key in ("document_id", "source_id", "language"):
            if source.get(key) is not None:
                item[key] = source[key]
        for key in ("document_id", "source_id", "title", "language",
                    "source_group_id", "origin_document_id"):
            if source.get(key) is not None:
                _string(source[key], f"source_metadata.{key}")
        for key in ("page_id", "image_id"):
            if extraction.get(key) is not None:
                _string(extraction[key], f"extraction_metadata.{key}")
        normalized.append(item)
    from task4_rank_evidence import validate_inputs
    validate_inputs(query, normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "passages": normalized,
        "provenance": provenance,
    }


def validate_task3_package(package):
    normalize_task3_package(package)
    return True


def _reference(row):
    provenance = row.get("provenance") or {}
    source = provenance.get("source_metadata") or {}
    extraction = provenance.get("extraction_metadata") or {}
    reference = {"passage_id": row["passage_id"]}
    for key in ("document_id", "source_id", "url", "title", "language",
                "source_group_id", "origin_document_id"):
        if source.get(key) is not None:
            reference[key] = copy.deepcopy(source[key])
    for key in ("method", "page_number", "page_id", "image_id", "ocr_confidence"):
        if extraction.get(key) is not None:
            reference[key] = copy.deepcopy(extraction[key])
    reference["reference_available"] = bool(
        source.get("url") or source.get("document_id") or source.get("source_id")
    )
    return reference


def build_task5_package(query, ranked_evidence, *, provenance):
    """Wrap the public list API; this does not call a Task 5 summarizer."""
    _string(query, "query")
    if not isinstance(ranked_evidence, list):
        raise ValueError("ranked_evidence must be a list")
    _object(provenance, "provenance")
    try:
        references = [_reference(row) for row in ranked_evidence]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid ranked_evidence row or provenance") from exc
    has_support = any("source_support" in row for row in ranked_evidence)
    result = {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "status": "ok" if ranked_evidence else "no_relevant_evidence",
        "ranked_evidence": copy.deepcopy(ranked_evidence),
        "is_simulated": provenance.get("kind") == "constructed_example",
        "provenance": copy.deepcopy(provenance),
        "traceability": {
            "selected_evidence_ids": [row["passage_id"] for row in ranked_evidence],
            "selected_references": references,
            "references_available": bool(references) and all(
                reference["reference_available"] for reference in references
            ),
            "source_support_scope": (
                "all_input_candidates_before_filtering" if has_support else "not_included"
            ),
            "source_independence_established": False,
            "text_similarity_establishes_agreement": False,
        },
        "integration": {
            "task3": "constructed_fixture" if provenance.get("kind") == "constructed_example"
                     else "supplied_package",
            "task5": "consumer_stub_only",
            "actual_task5_available": False,
        },
    }
    validate_task5_package(result)
    return result


def validate_task5_package(package):
    _object(package, "Task 5 package")
    if package.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Task 5 schema_version")
    _string(package.get("query"), "query")
    rows = package.get("ranked_evidence")
    if not isinstance(rows, list):
        raise ValueError("ranked_evidence must be a list")
    expected_status = "ok" if rows else "no_relevant_evidence"
    if package.get("status") != expected_status:
        raise ValueError("status must agree with ranked_evidence")
    seen = set()
    for rank, row in enumerate(rows, 1):
        _object(row, "ranked passage")
        pid = _string(row.get("passage_id"), "passage_id")
        if pid in seen or isinstance(row.get("rank"), bool) or not isinstance(row.get("rank"), int) or row["rank"] != rank:
            raise ValueError("ranked evidence needs unique IDs and consecutive ranks")
        seen.add(pid)
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("ranked evidence text must be nonempty")
        for key in ("query_relevance", "indian_context_relevance", "source_signal", "final_score"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and within [0,1]")
        if "source_support" in row and not isinstance(row["source_support"], dict):
            raise ValueError("source_support must be an object")
    trace = _object(package.get("traceability"), "traceability")
    if trace.get("selected_evidence_ids") != [row["passage_id"] for row in rows]:
        raise ValueError("selected_evidence_ids must match returned evidence")
    if not isinstance(trace.get("selected_references"), list) or len(trace["selected_references"]) != len(rows):
        raise ValueError("selected_references must cover returned evidence")
    for reference, row in zip(trace["selected_references"], rows):
        _object(reference, "selected reference")
        if reference.get("passage_id") != row["passage_id"]:
            raise ValueError("selected_references passage IDs must match returned evidence")
        if not isinstance(reference.get("reference_available"), bool):
            raise ValueError("reference_available must be a boolean")
    expected_available = bool(rows) and all(
        reference["reference_available"] for reference in trace["selected_references"]
    )
    if trace.get("references_available") is not expected_available:
        raise ValueError("references_available must agree with selected references")
    expected_scope = "all_input_candidates_before_filtering" if any(
        "source_support" in row for row in rows
    ) else "not_included"
    if trace.get("source_support_scope") != expected_scope:
        raise ValueError("source_support_scope must describe the preserved counters")
    if trace.get("source_independence_established") is not False:
        raise ValueError("this contract does not establish source independence")
    if trace.get("text_similarity_establishes_agreement") is not False:
        raise ValueError("text similarity does not establish agreement")
    integration = _object(package.get("integration"), "integration")
    if integration.get("actual_task5_available") is not False or integration.get("task5") != "consumer_stub_only":
        raise ValueError("the Task 5 consumer is a stub")
    provenance = _object(package.get("provenance"), "provenance")
    if provenance.get("kind") not in ("constructed_example", "task3_output"):
        raise ValueError("invalid provenance.kind")
    _string(provenance.get("description"), "provenance.description")
    if package.get("is_simulated") is not (provenance["kind"] == "constructed_example"):
        raise ValueError("is_simulated must agree with provenance.kind")
    return True


def task5_consumer_stub(package):
    """Contract check only: no generated answer, reliability score or real Task 5."""
    validate_task5_package(package)
    return {
        "accepted": True,
        "consumer": "task5_contract_stub",
        "actual_task5_available": False,
        "received_evidence_ids": package["traceability"]["selected_evidence_ids"][:],
        "status": package["status"],
    }
