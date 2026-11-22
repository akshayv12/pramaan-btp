"""Inspect an annotated JSONL candidate dataset without loading a model.

The default invocation reads and prints only. Use --output explicitly to save
the same JSON audit. Source URL checks establish syntax, not source truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse


def _http_url(value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlparse(value.strip())
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not any(char.isspace() for char in parsed.hostname)
            and (parsed.port is None or 1 <= parsed.port <= 65535)
        )
    except ValueError:
        return False


def audit_dataset(path: str | Path) -> dict:
    """Report observable dataset properties without inferring annotation truth."""
    path = Path(path)
    raw = path.read_bytes()
    errors, rows = [], []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_number}: record must be a JSON object")
            continue
        for key in ("query_id", "query", "passage_id", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f"line {line_number}: {key} must be a nonempty string")
        for key in ("query_relevance", "indian_context_relevance"):
            label = row.get(key)
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label <= 3:
                errors.append(f"line {line_number}: {key} must be an integer from 0 to 3")
        for key in ("source_metadata", "extraction_metadata"):
            if row.get(key) is not None and not isinstance(row[key], dict):
                errors.append(f"line {line_number}: {key} must be an object or null")
        extraction = row.get("extraction_metadata")
        if isinstance(extraction, dict) and extraction.get("ocr_confidence") is not None:
            confidence = extraction["ocr_confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                errors.append(f"line {line_number}: ocr_confidence must be a finite number in [0,1]")
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        if isinstance(row.get("query_id"), str) and row["query_id"].strip():
            groups[row["query_id"]].append(row)

    seen, duplicate_pairs = set(), []
    for row in rows:
        qid, pid = row.get("query_id"), row.get("passage_id")
        if not isinstance(qid, str) or not isinstance(pid, str):
            continue
        pair = (qid, pid)
        if pair in seen:
            duplicate_pairs.append({"query_id": qid, "passage_id": pid})
        seen.add(pair)

    languages = Counter()
    per_query = []
    for qid, group in sorted(groups.items()):
        queries = {row["query"] for row in group if isinstance(row.get("query"), str)}
        if len(queries) != 1:
            errors.append(f"{qid}: inconsistent or missing query text")
        declared = {
            row["query_language"] for row in group
            if isinstance(row.get("query_language"), str) and row["query_language"].strip()
        }
        if len(declared) > 1:
            errors.append(f"{qid}: inconsistent query_language values")
        language = next(iter(declared)) if len(declared) == 1 else "unspecified"
        languages[language] += 1
        relevant = sum(
            isinstance(row.get("query_relevance"), int)
            and not isinstance(row["query_relevance"], bool)
            and row["query_relevance"] >= 2
            for row in group
        )
        per_query.append({
            "query_id": qid,
            "query": next(iter(queries)) if len(queries) == 1 else None,
            "query_language": language,
            "candidate_count": len(group),
            "relevant_candidates_label_ge_2": relevant,
        })

    texts = Counter(row["text"] for row in rows if isinstance(row.get("text"), str))
    sources = [
        row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
        for row in rows
    ]
    extractions = [
        row.get("extraction_metadata") if isinstance(row.get("extraction_metadata"), dict) else {}
        for row in rows
    ]
    exhaustive_at_ten = bool(per_query) and all(row["candidate_count"] <= 10 for row in per_query)
    warnings = []
    if exhaustive_at_ten:
        warnings.append(
            "Every query has at most 10 candidates. With filtering disabled, "
            "Recall@10 equals 1 for every query containing a relevant candidate, "
            "regardless of ranking quality. Use smaller K values and nDCG to assess ordering."
        )
    if languages and not any(value in {"hi-Latn", "roman_hi", "roman-hindi", "hinglish"} for value in languages):
        warnings.append(
            "No query group is explicitly labeled as Roman Hindi. Latin-script "
            "query language cannot be established from character script alone."
        )
    warnings.append(
        "This audit verifies record structure and counts. It does not establish "
        "human annotation provenance, factual correctness, or reliability of sources."
    )
    return {
        "audit_schema_version": 1,
        "dataset": str(path.resolve()),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(rows),
        "query_count": len(groups),
        "query_languages": dict(sorted(languages.items())),
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(Counter(len(group) for group in groups.values()).items())
        },
        "query_relevance_label_distribution": {
            str(label): sum(row.get("query_relevance") == label and not isinstance(row.get("query_relevance"), bool) for row in rows)
            for label in range(4)
        },
        "context_label_distribution": {
            str(label): sum(row.get("indian_context_relevance") == label and not isinstance(row.get("indian_context_relevance"), bool) for row in rows)
            for label in range(4)
        },
        "source_types": dict(sorted(Counter(
            meta.get("source_type") if isinstance(meta.get("source_type"), str) else "missing"
            for meta in sources
        ).items())),
        "source_metadata_present": sum(bool(meta) for meta in sources),
        "source_urls_provided": sum(bool(meta.get("url")) for meta in sources),
        "structurally_valid_http_source_urls": sum(_http_url(meta.get("url")) for meta in sources),
        "extraction_metadata_present": sum(bool(meta) for meta in extractions),
        "ocr_confidence_provided": sum(meta.get("ocr_confidence") is not None for meta in extractions),
        "page_reference_provided": sum(meta.get("page_number") is not None for meta in extractions),
        "duplicate_query_passage_pairs": duplicate_pairs,
        "exact_repeated_passage_text_count": sum(count - 1 for count in texts.values() if count > 1),
        "recall_at_10_covers_entire_candidate_pool_without_filtering": exhaustive_at_ten,
        "per_query": per_query,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", type=Path,
        default=Path(__file__).with_name("sample_dataset.jsonl"),
    )
    parser.add_argument("--output", type=Path, help="Explicitly save the JSON audit.")
    args = parser.parse_args()
    try:
        result = audit_dataset(args.dataset)
    except (OSError, UnicodeDecodeError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        if args.output.resolve() == args.dataset.resolve():
            parser.error("--output must not overwrite the input dataset")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
