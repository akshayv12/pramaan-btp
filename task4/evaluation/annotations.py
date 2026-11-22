"""Blinded annotation exports and explicitly declared human-review imports.

No labels or reviewer declarations are invented by export. Review files record
self-declared reviewers; this tool cannot verify a person's identity or source truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

LABELS = ("query_relevance", "indian_context_relevance")
IDENTITY = ("annotation_schema_version", "task_id", "query_id", "query", "query_language",
            "query_family", "passage_id", "text", "source_metadata", "extraction_metadata",
            "origin", "source_dataset_sha256", "translation_source_query_id")
ROMAN_TEMPLATES = {
    "Q001": "Ram aur Ayodhya se judi pracheen jaankari kya hai?",
    "Q002": "ISRO ke Chandrayaan mission ki kya uplabdhiyan hain?",
    "Q013": "Bharatiya Samvidhan mein maulik adhikar kya hain?",
    "Q028": "Bharat mein jal sankat aur jal sanrakshan ke kya upay hain?",
}
PLACEHOLDERS = {"todo", "tbd", "pending", "unknown", "annotator_id", "<annotator>", "<id>"}


def read_jsonl(path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        rows.append(row)
    return rows


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _label(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
        raise ValueError(f"{field} must be an integer in [0,3]")
    return value


def _review_declaration(row: dict, identifier: str) -> str:
    reviewer = _text(row.get(identifier), identifier)
    if reviewer.casefold() in PLACEHOLDERS or (reviewer.startswith("<") and reviewer.endswith(">")):
        raise ValueError(f"{identifier} is a placeholder")
    if row.get("review_method") != "human_declared" or row.get("human_review_attested") is not True:
        raise ValueError("completed review requires review_method='human_declared' and explicit human_review_attested=true")
    stamp = _text(row.get("reviewed_at"), "reviewed_at")
    try:
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an explicitly supplied ISO date/time") from exc
    for field in LABELS:
        _label(row.get(field), field)
    return reviewer


def _task_identity(task: dict) -> dict:
    return {key: task.get(key) for key in IDENTITY if key != "task_id"}


def _task_id(task: dict) -> str:
    payload = json.dumps(_task_identity(task), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def export_tasks(dataset, include_roman=False, seed=42, origin="existing_sample_unverified") -> list[dict]:
    """Remove supplied labels/scores and optionally add pending translated tasks."""
    if origin not in {"existing_sample_unverified", "retrieved_unverified"}:
        raise ValueError("origin must describe an unverified source collection")
    dataset = Path(dataset)
    source_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    rows = read_jsonl(dataset)
    if not rows:
        raise ValueError("dataset is empty")
    groups = {}
    for row in rows:
        qid = _text(row.get("query_id"), "query_id")
        query = _text(row.get("query"), "query")
        family = _text(row.get("query_family", qid), "query_family")
        language = row.get("query_language")
        if language is not None:
            language = _text(language, "query_language")
        group = groups.setdefault(qid, {"query": query, "family": family, "language": None, "rows": []})
        if group["query"] != query or group["family"] != family:
            raise ValueError(f"inconsistent query/family for {qid}")
        if language:
            if group["language"] and group["language"] != language:
                raise ValueError(f"inconsistent query language for {qid}")
            group["language"] = language
        group["rows"].append(row)
    tasks, seen = [], set()
    for qid, group in sorted(groups.items()):
        variants = [(qid, group["query"], group["language"] or "unspecified", None)]
        if include_roman and qid in ROMAN_TEMPLATES:
            variants.append((qid + "-ROMAN", ROMAN_TEMPLATES[qid], "hi-Latn", qid))
        for variant_id, query, language, source_query in variants:
            for row in group["rows"]:
                pid = _text(row.get("passage_id"), "passage_id")
                text = _text(row.get("text"), "text")
                pair = (variant_id, pid)
                if pair in seen:
                    raise ValueError(f"duplicate query/passage pair: {pair}")
                seen.add(pair)
                for field in ("source_metadata", "extraction_metadata"):
                    if row.get(field) is not None and not isinstance(row[field], dict):
                        raise ValueError(f"{field} must be an object or null")
                task = {
                    "annotation_schema_version": 1,
                    "query_id": variant_id, "query": query, "query_language": language,
                    "query_family": group["family"], "passage_id": pid, "text": text,
                    "source_metadata": copy.deepcopy(row.get("source_metadata") or {}),
                    "extraction_metadata": copy.deepcopy(row.get("extraction_metadata") or {}),
                    "origin": origin, "source_dataset_sha256": source_hash,
                    "translation_source_query_id": source_query,
                    "translation_status": "draft_pending_review" if source_query else "not_applicable",
                    "query_relevance": None, "indian_context_relevance": None,
                    "annotator_id": None, "review_status": "pending",
                    "review_method": None, "human_review_attested": False,
                    "reviewed_at": None, "review_note": "",
                }
                task["task_id"] = _task_id(task)
                tasks.append(task)
    random.Random(seed).shuffle(tasks)
    return tasks


def validate_tasks(tasks: list[dict]) -> dict[str, dict]:
    if not tasks:
        raise ValueError("task set is empty")
    indexed = {}
    for task in tasks:
        if task.get("annotation_schema_version") != 1:
            raise ValueError("unsupported annotation task schema")
        for field in ("query_id", "query", "query_language", "query_family", "passage_id", "text", "origin"):
            _text(task.get(field), field)
        if task.get("origin") not in {"existing_sample_unverified", "retrieved_unverified"}:
            raise ValueError("task source origin must remain explicitly unverified")
        if task.get("review_status") != "pending" or any(task.get(field) is not None for field in LABELS):
            raise ValueError("the original task file must remain blinded and pending")
        if task.get("human_review_attested") is True or task.get("review_method") is not None:
            raise ValueError("unreviewed task files must not claim human review")
        task_id = _text(task.get("task_id"), "task_id")
        if task_id != _task_id(task):
            raise ValueError("task identity changed; regenerate tasks before annotation")
        if task_id in indexed:
            raise ValueError("duplicate task_id")
        indexed[task_id] = task
    return indexed


def import_reviewer(rows: list[dict], tasks: dict[str, dict]) -> tuple[str | None, dict[str, dict]]:
    """Validate one reviewer file; incomplete rows remain pending."""
    complete, seen, reviewers = {}, set(), set()
    for row in rows:
        task_id = _text(row.get("task_id"), "task_id")
        if task_id not in tasks or task_id in seen:
            raise ValueError("unknown or repeated task_id in reviewer file")
        seen.add(task_id)
        expected = tasks[task_id]
        for field in IDENTITY:
            if row.get(field) != expected.get(field):
                raise ValueError(f"review changed immutable task field: {field}")
        status = row.get("review_status")
        if status == "pending":
            if any(row.get(field) is not None for field in LABELS):
                raise ValueError("pending reviews must leave both labels null")
            if row.get("human_review_attested") is True or row.get("review_method") is not None:
                raise ValueError("pending rows must not claim completed human review")
            continue
        if status != "complete":
            raise ValueError("review_status must be pending or complete")
        reviewer = _review_declaration(row, "annotator_id")
        reviewers.add(reviewer)
        if expected.get("translation_source_query_id") and row.get("translation_status") != "approved":
            raise ValueError("Roman-Hindi translation must be explicitly approved before labeling")
        complete[task_id] = copy.deepcopy(row)
    if len(reviewers) > 1:
        raise ValueError("one reviewer file must identify exactly one declared reviewer")
    return next(iter(reviewers)) if reviewers else None, complete


def weighted_kappa(first: list[int], second: list[int]) -> float | None:
    """Quadratic-weighted Cohen kappa on the fixed four-label scale."""
    if len(first) != len(second):
        raise ValueError("agreement lists must have equal length")
    if not first:
        return None
    for value in first + second:
        _label(value, "agreement label")
    count = len(first)
    left, right = Counter(first), Counter(second)
    observed = sum((a - b) ** 2 for a, b in zip(first, second)) / count
    expected = sum(left[a] * right[b] * (a - b) ** 2 for a in range(4) for b in range(4)) / count ** 2
    return 1 - observed / expected if expected else None


def review_tasks(tasks: list[dict], reviewer_files: list[list[dict]], adjudications=None) -> dict:
    """Return reports and, only when fully reviewed, a benchmark."""
    indexed = validate_tasks(tasks)
    if len(reviewer_files) not in (1, 2):
        raise ValueError("supply one or two reviewer files")
    imports = [import_reviewer(rows, indexed) for rows in reviewer_files]
    declared = [reviewer for reviewer, _ in imports if reviewer is not None]
    if len(declared) != len(set(declared)):
        raise ValueError("two reviewer files must declare distinct reviewer IDs")
    completed = [records for _, records in imports]
    overlap = sorted(set(completed[0]) & set(completed[1])) if len(completed) == 2 else []
    agreement = None
    conflicts = {}
    if len(completed) == 2:
        agreement = {}
        for field in LABELS:
            left = [completed[0][task][field] for task in overlap]
            right = [completed[1][task][field] for task in overlap]
            agreement[field] = {
                "paired_tasks": len(overlap),
                "exact_agreement": sum(a == b for a, b in zip(left, right)) / len(overlap) if overlap else None,
                "quadratic_weighted_kappa": weighted_kappa(left, right),
            }
        for task in overlap:
            if any(completed[0][task][field] != completed[1][task][field] for field in LABELS):
                conflicts[task] = {
                    "task_id": task,
                    "query_id": indexed[task]["query_id"], "passage_id": indexed[task]["passage_id"],
                    "reviews": [{key: record[key] for key in (*LABELS, "annotator_id", "reviewed_at", "review_note") if key in record}
                                for record in (completed[0][task], completed[1][task])],
                }
    resolutions = {}
    for row in adjudications or []:
        task_id = _text(row.get("task_id"), "task_id")
        if task_id not in conflicts or task_id in resolutions:
            raise ValueError("adjudication must reference one unresolved disagreement exactly once")
        if row.get("review_status") != "complete":
            raise ValueError("adjudication review_status must be complete")
        person = _review_declaration(row, "adjudicator_id")
        if person in declared:
            raise ValueError("adjudicator_id must differ from the two original reviewers")
        _text(row.get("reason"), "adjudication reason")
        resolutions[task_id] = copy.deepcopy(row)
    fully_reviewed = set.intersection(*(set(records) for records in completed))
    unresolved = sorted(set(conflicts) - set(resolutions))
    ready = len(fully_reviewed) == len(indexed) and not unresolved and len(declared) == len(imports)
    benchmark = []
    if ready:
        for task_id, task in sorted(indexed.items()):
            selected = resolutions.get(task_id) or completed[0][task_id]
            row = {key: copy.deepcopy(task[key]) for key in (
                "query_id", "query", "query_language", "query_family", "passage_id", "text",
                "source_metadata", "extraction_metadata",
            )}
            row.update({field: selected[field] for field in LABELS})
            row["annotation_metadata"] = {
                "task_id": task_id, "origin": task["origin"],
                "source_dataset_sha256": task["source_dataset_sha256"],
                "review_status": "complete", "review_method": "human_declared",
                "human_review_attested": True,
                "declared_reviewer_ids": declared,
                "reviewed_at": [records[task_id]["reviewed_at"] for records in completed],
                "adjudicator_id": resolutions.get(task_id, {}).get("adjudicator_id"),
                "adjudication_reason": resolutions.get(task_id, {}).get("reason"),
                "translation_source_query_id": task.get("translation_source_query_id"),
                "translation_status": "approved" if task.get("translation_source_query_id") else "not_applicable",
                "human_identity_verified_by_tool": False,
                "source_authenticity_verified_by_tool": False,
            }
            benchmark.append(row)
    report = {
        "annotation_schema_version": 1, "task_count": len(indexed),
        "reviewer_file_count": len(imports), "declared_reviewer_ids": declared,
        "complete_reviews_by_reviewer": {reviewer or f"file_{i + 1}_not_declared": len(records)
                                         for i, (reviewer, records) in enumerate(imports)},
        "tasks_with_all_requested_reviews": len(fully_reviewed),
        "pending_task_count": len(indexed) - len(fully_reviewed),
        "disagreement_count": len(conflicts), "resolved_disagreement_count": len(resolutions),
        "unresolved_disagreement_count": len(unresolved),
        "agreement": agreement, "reviewed_benchmark_ready": ready,
        "benchmark_review_level": ("two_declared_reviewers" if len(imports) == 2 else "single_declared_reviewer") if ready else "pending",
        "human_identity_verified_by_tool": False, "source_authenticity_verified_by_tool": False,
        "warning": "Declarations document claimed human review; this software cannot establish person identity, truth, or source authenticity.",
    }
    return {"report": report, "disagreements": [conflicts[task] for task in unresolved],
            "resolutions": list(resolutions.values()),
            "reviews": [record for _, records in imports for record in records.values()],
            "benchmark": benchmark}


def write_jsonl(path, rows, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Create pending tasks without supplied labels")
    export.add_argument("dataset", type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--include-roman", action="store_true", help="Add proposed, unapproved Roman-Hindi translations")
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--origin", choices=("existing_sample_unverified", "retrieved_unverified"), default="existing_sample_unverified")
    review = sub.add_parser("review", help="Validate declared reviews and queue disagreements")
    review.add_argument("--tasks", required=True, type=Path)
    review.add_argument("--reviewer", required=True, type=Path, action="append")
    review.add_argument("--adjudications", type=Path)
    review.add_argument("--output", required=True, type=Path, help="A new report directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            tasks = export_tasks(args.dataset, args.include_roman, args.seed, args.origin)
            if args.output.resolve() == args.dataset.resolve():
                raise ValueError("output must not overwrite the source dataset")
            write_jsonl(args.output, tasks)
            result = {"task_count": len(tasks), "pending_task_count": len(tasks),
                      "query_count": len({task["query_id"] for task in tasks}),
                      "reviewed_benchmark_ready": False, "output": str(args.output)}
        else:
            result = review_tasks(read_jsonl(args.tasks), [read_jsonl(path) for path in args.reviewer],
                                  read_jsonl(args.adjudications) if args.adjudications else None)
            inputs = {"tasks": args.tasks, **{f"reviewer_{i + 1}": path for i, path in enumerate(args.reviewer)}}
            if args.adjudications:
                inputs["adjudications"] = args.adjudications
            result["report"]["input_files"] = {
                name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in inputs.items()
            }
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "review_report.json").write_text(
                json.dumps(result["report"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            write_jsonl(args.output / "reviews.jsonl", result["reviews"])
            write_jsonl(args.output / "disagreements.jsonl", result["disagreements"])
            write_jsonl(args.output / "adjudications.jsonl", result["resolutions"])
            if result["report"]["reviewed_benchmark_ready"]:
                write_jsonl(args.output / "reviewed_dataset.jsonl", result["benchmark"])
            result = result["report"]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
