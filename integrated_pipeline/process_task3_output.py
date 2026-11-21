"""Rank the passages of a Task 3 ``task3_output.json`` with Task 4.

Task 3 writes ``{"task3_output": {"schema_version": "1.0", "metadata", "documents",
"summary"}}``. One file can hold several user questions; each document records its
own ``source.original_query``. This adapter groups passages per question, maps the
Task 2/Task 3 field names onto the existing Task 4 input contract (contracts.py),
ranks them with ``rank_evidence`` and wraps each result with ``build_task5_package``.

Field mapping decisions (documented in docs/INTEGRATION.md):
  * OCR confidence: Task 3 reports a 0-100 percentage. OCR passages are converted
    to 0-1. Direct text carries a conventional 100 that is not a measurement, so it
    is omitted and Task 4 applies its existing direct-text treatment.
  * Task 2 source_type: "academic" -> "academic", "government" -> "institutional".
    Other categories (commercial, non_profit, other) are not reliability claims and
    map to unknown; the raw value is kept as task2_source_type.
  * Only documents with status "processed" contribute passages. Other statuses are
    counted in the provenance so nothing disappears silently.

Usage (root virtual environment, which holds the BGE-M3 model):
    .venv\\Scripts\\python.exe -X utf8 process_task3_output.py task2_search_collect\\task3_output.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts import SCHEMA_VERSION, build_task5_package, normalize_task3_package  # noqa: E402

TASK3_SCHEMA_VERSION = "1.0"
OUTPUT_SCHEMA_VERSION = "1.0"
DEFAULT_CONFIG = ROOT / "configs" / "deployment.json"
DEFAULT_TOP_K = 10
SOURCE_TYPE_MAP = {"academic": "academic", "government": "institutional"}
DOCUMENT_STATUSES = ("processed", "duplicate", "empty", "skipped", "failed")


def _text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _publication_date(value):
    """Keep ISO dates; reduce ISO datetimes to their date; leave others for Task 4's neutral score."""
    value = _text(value)
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value


def _ocr_confidence(passage, label):
    """Return a 0-1 confidence for OCR passages, None for direct text or unknown values."""
    method = (passage.get("extraction_method") or "").upper()
    if "OCR" not in method:
        return None
    value = passage.get("ocr_confidence")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: ocr_confidence must be a finite number")
    if not 0 <= value <= 100:
        raise ValueError(f"{label}: Task 3 ocr_confidence must be a 0-100 percentage")
    return round(value / 100.0, 6)


def _page_number(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}: page_number must be a positive integer or null")
    return value


def _map_passage(document, passage, index):
    label = f"document {document.get('document_id')!r} passage {index}"
    if not isinstance(passage, dict):
        raise ValueError(f"{label} must be an object")
    source = document.get("source") or {}
    if not isinstance(source, dict):
        raise ValueError(f"document {document.get('document_id')!r}: source must be an object")
    declared = source.get("source_metadata") if isinstance(source.get("source_metadata"), dict) else {}
    raw_type = _text(source.get("source_type"))
    source_metadata = {
        "document_id": _text(passage.get("document_id")) or _text(document.get("document_id")),
        "url": _text(source.get("url")),
        "canonical_url": _text(source.get("canonical_url")),
        "domain": _text(source.get("domain")),
        "title": _text(source.get("title")),
        "author": _text(source.get("author")),
        "publication_date": _publication_date(source.get("pub_date")),
        "language": _text(declared.get("language")),
        "source_type": SOURCE_TYPE_MAP.get((raw_type or "").casefold()),
        "task2_source_type": raw_type,
        "content_type": _text(source.get("content_type")),
        "licence": _text(source.get("licence")),
        "licence_category": _text(source.get("licence_category")),
        "parent_document_id": _text(source.get("parent_doc_id")),
        "query_slug": _text(source.get("query_slug")),
        "source_queries": copy.deepcopy(source.get("source_queries")),
    }
    extraction_metadata = {
        "method": _text(passage.get("extraction_method")),
        "ocr_engine": _text(passage.get("ocr_engine")),
        "ocr_confidence": _ocr_confidence(passage, label),
        "page_number": _page_number(passage.get("page_number"), label),
        "detected_script": _text(passage.get("detected_script")),
        "detected_language": _text(passage.get("detected_language")),
        "task3_reliability": _text(passage.get("reliability")),
        "source_file": _text(passage.get("source_file")),
        "source_path": _text(passage.get("source_path")),
        "word_count": passage.get("word_count"),
    }
    return {
        "passage_id": passage.get("passage_id"),
        "text": passage.get("text"),
        "source_metadata": {k: v for k, v in source_metadata.items() if v is not None},
        "extraction_metadata": {k: v for k, v in extraction_metadata.items() if v is not None},
    }


def _unwrap(payload):
    package = payload.get("task3_output") if isinstance(payload, dict) else None
    if not isinstance(package, dict) or package.get("schema_version") != TASK3_SCHEMA_VERSION:
        raise ValueError(f"expected task3_output with Task 3 schema_version '{TASK3_SCHEMA_VERSION}'")
    if not isinstance(package.get("documents"), list) or not isinstance(package.get("metadata"), dict):
        raise ValueError("expected documents and metadata in task3_output")
    return package


def task3_output_to_packages(payload):
    """Convert a Task 3 output into one contracts.py Task 4 input package per question.

    Questions are kept in first-appearance order: those named in the metadata first,
    then any additional question recorded on a document.
    """
    package = _unwrap(payload)
    metadata = package["metadata"]
    declared = metadata.get("original_query")
    declared = declared if isinstance(declared, list) else [declared]
    fallback = _text(declared[0]) if len(declared) == 1 else None
    groups = {}

    def group(query):
        return groups.setdefault(query, {"passages": [], "statuses": Counter(), "slugs": [], "documents": 0})

    for query in declared:
        if _text(query):
            group(_text(query))
    for index, document in enumerate(package["documents"]):
        if not isinstance(document, dict):
            raise ValueError(f"documents[{index}] must be an object")
        source = document.get("source") if isinstance(document.get("source"), dict) else {}
        query = _text(source.get("original_query")) or fallback
        if query is None:
            raise ValueError(f"documents[{index}] has no original_query and the metadata names several questions")
        status = document.get("status")
        if status not in DOCUMENT_STATUSES:
            raise ValueError(f"documents[{index}] has unknown status {status!r}")
        entry = group(query)
        entry["documents"] += 1
        entry["statuses"][status] += 1
        slug = _text(source.get("query_slug"))
        if slug and slug not in entry["slugs"]:
            entry["slugs"].append(slug)
        passages = document.get("passages") or []
        if not isinstance(passages, list):
            raise ValueError(f"documents[{index}].passages must be a list")
        if status == "processed":
            entry["passages"].extend(_map_passage(document, p, i) for i, p in enumerate(passages))

    packages = []
    for query, entry in groups.items():
        packages.append({
            "schema_version": SCHEMA_VERSION,
            "query": query,
            "provenance": {
                "kind": "task3_output",
                "description": "Passages extracted by Task 3 from documents collected by Task 2.",
                "task3_schema_version": package["schema_version"],
                "query_slugs": entry["slugs"],
                "documents_considered": entry["documents"],
                "document_status_counts": {s: entry["statuses"][s] for s in DOCUMENT_STATUSES},
                "passages_supplied": len(entry["passages"]),
                "field_mapping": {
                    "ocr_confidence": "Task 3 0-100 OCR percentage divided by 100; omitted for direct text",
                    "source_type": "Task 2 academic->academic, government->institutional, others unknown",
                },
            },
            "passages": entry["passages"],
        })
    return packages


def _resolve_model_paths(config):
    """Resolve project-relative model folders so the ranker works from any directory."""
    config = copy.deepcopy(config or {})
    model = config.get("semantic_model")
    if isinstance(model, dict):
        for key in ("name", "cache_folder"):
            value = model.get(key)
            if isinstance(value, str) and value and not Path(value).is_absolute() and (ROOT / value).exists():
                model[key] = str((ROOT / value).resolve())
    return config


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    return _resolve_model_paths(json.loads(path.read_text(encoding="utf-8-sig")))


def rank_task3_output(payload, *, config=None, top_k=DEFAULT_TOP_K):
    """Return the Task 4 output object for a parsed Task 3 output."""
    from task4_rank_evidence import rank_evidence, validated_config

    config = _resolve_model_paths(config)
    effective = validated_config(config)
    package = _unwrap(payload)
    results = []
    for raw in task3_output_to_packages(payload):
        normalized = normalize_task3_package(raw)
        ranked = rank_evidence(normalized["query"], normalized["passages"], top_k=top_k, config=config)
        envelope = build_task5_package(normalized["query"], ranked, provenance=normalized["provenance"])
        envelope["candidate_count"] = len(normalized["passages"])
        results.append(envelope)
    summary = {
        "questions": len(results),
        "questions_with_evidence": sum(r["status"] == "ok" for r in results),
        "candidate_passages": sum(r["candidate_count"] for r in results),
        "ranked_passages": sum(len(r["ranked_evidence"]) for r in results),
    }
    return {"task4_output": {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metadata": copy.deepcopy(package["metadata"]),
        "task3_summary": copy.deepcopy(package.get("summary")),
        "ranking": {"top_k": top_k, "config": effective},
        "queries": results,
        "summary": summary,
    }}


def _write_output(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_task3_output(input_path, output_path=None, *, config=None, top_k=DEFAULT_TOP_K):
    """Read Task 3 output, rank it and write task4_output.json. Errors leave existing output intact."""
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve() if output_path else source.with_name("task4_output.json")
    if destination == source:
        raise ValueError("output must differ from the Task 3 input")
    with source.open(encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    result = rank_task3_output(payload, config=config, top_k=top_k)
    _write_output(destination, result)
    return result["task4_output"]["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank Task 3 passages with Task 4.")
    parser.add_argument("task3_output", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to task4_output.json beside the input")
    parser.add_argument("--config", type=Path, help=f"Task 4 JSON config (default: {DEFAULT_CONFIG.relative_to(ROOT)})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Ranked passages kept per question")
    parser.add_argument("--lexical-only", action="store_true", help="TF-IDF relevance only; no encoder is loaded")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True,
                        help="Forbid model downloads (default: on)")
    args = parser.parse_args(argv)
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    output = args.output or args.task3_output.with_name("task4_output.json")
    try:
        config = load_config(args.config)
        if args.lexical_only:
            config["relevance_weights"] = {"lexical_weight": 1.0, "semantic_weight": 0.0}
        summary = process_task3_output(args.task3_output, output, config=config, top_k=args.top_k)
    except Exception as exc:  # Invalid input, missing model, or I/O errors are all fatal here.
        print(f"[TASK 4] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"[TASK 4] {summary} -> {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
