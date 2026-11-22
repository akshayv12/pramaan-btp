"""Evaluate passage ranking on labeled query groups without rounding objectives.

The supplied sample annotations are a development benchmark; their authorship
and factual/source reliability have not been independently validated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scipy.stats import spearmanr
from task4_rank_evidence import (
    CONTEXT_WEIGHT_KEYS, prepare_features, rank_prepared, validated_config,
    validate_inputs,
)
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

DEFAULT_KS = (1, 3, 5, 10)


def load_dataset(path: str | Path) -> dict[str, dict]:
    """Validate JSONL rows, retaining labels separately from ranker inputs."""
    grouped = {}
    with open(path, encoding="utf-8") as dataset:
        for number, line in enumerate(dataset, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {number}: expected a JSON object")
            for key in ("query_id", "query", "passage_id", "text",
                        "query_relevance", "indian_context_relevance"):
                if key not in row:
                    raise ValueError(f"line {number}: missing {key}")
            if not isinstance(row["query_id"], str) or not row["query_id"].strip():
                raise ValueError(f"line {number}: query_id must be a non-empty string")
            for label in ("query_relevance", "indian_context_relevance"):
                value = row[label]
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
                    raise ValueError(f"line {number}: {label} must be an integer in [0, 3]")
            qid = row["query_id"]
            family = row.get("query_family", qid)
            if not isinstance(family, str) or not family.strip():
                raise ValueError(f"line {number}: query_family must be a non-empty string")
            group = grouped.setdefault(qid, {
                "query": row["query"], "passages": [],
                "query_relevance_labels": {}, "context_labels": {},
                "query_family": family, "query_language": row.get("query_language"),
            })
            if group["query"] != row["query"] or group["query_family"] != family:
                raise ValueError(f"line {number}: inconsistent query or family for {qid}")
            pid = row["passage_id"]
            if pid in group["query_relevance_labels"]:
                raise ValueError(f"line {number}: duplicate passage_id {pid!r} within {qid}")
            passage = {
                "passage_id": pid, "text": row["text"],
                "source_metadata": row.get("source_metadata", {}),
                "extraction_metadata": row.get("extraction_metadata", {}),
            }
            group["passages"].append(passage)
            group["query_relevance_labels"][pid] = row["query_relevance"]
            group["context_labels"][pid] = row["indian_context_relevance"]
    if not grouped:
        raise ValueError("dataset must contain at least one labeled query")
    for group in grouped.values():
        validate_inputs(group["query"], group["passages"])
    return grouped



def feature_cache_sources():
    """All project modules that determine cache interpretation and reporting."""
    names = ("task4_rank_evidence.py", "context_features.py", "evidence_groups.py", "contracts.py")
    return [path for path in [*(_ROOT / name for name in names),
                              *sorted((_ROOT / "evaluation").glob("*.py"))] if path.exists()]


def load_feature_cache(path, dataset_path) -> dict:
    """Require a completed experiment manifest binding features to this dataset."""
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("feature cache requires its experiment manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("feature cache experiment is not complete")
    digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    if manifest.get("dataset_sha256") != digest:
        raise ValueError("feature cache dataset hash differs")
    if manifest.get("feature_cache_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("feature cache content hash differs")
    for source in feature_cache_sources():
        name = str(source.relative_to(_ROOT))
        if source.exists() and manifest.get("source_sha256", {}).get(name) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"feature cache extraction source changed: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_dataset_features(grouped_dataset: dict, config=None) -> dict:
    """Encode each query once; no labels are supplied to feature extraction."""
    cfg = validated_config(config)
    return {
        qid: prepare_features(data["query"], data["passages"], config=cfg)
        for qid, data in sorted(grouped_dataset.items())
    }


def evaluate_ranker(grouped_dataset: dict[str, dict], config: dict | None = None,
                    ks: tuple[int, ...] = DEFAULT_KS, rel_threshold: int = 2,
                    prepared_features: dict | None = None) -> dict:
    """Macro-average full-precision metrics across queries.

    Context agreement uses the full candidate pool, independently of ranking
    filters. Precision divides by the returned count up to K, as metrics.py
    specifies. All objective values remain unrounded; formatting is display-only.
    """
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    cfg = validated_config(config)
    cache = (prepare_dataset_features(grouped_dataset, cfg)
             if prepared_features is None else prepared_features)
    rows = []
    correlations = []
    for qid, data in sorted(grouped_dataset.items()):
        if qid not in cache:
            raise ValueError(f"missing prepared features for {qid}")
        features = cache[qid]
        expected = [passage["passage_id"] for passage in data["passages"]]
        if [feature["passage_id"] for feature in features] != expected:
            raise ValueError(f"prepared passage ids/order differ for {qid}")
        if [feature["text"] for feature in features] != [
                (passage.get("text") or "").strip() for passage in data["passages"]]:
            raise ValueError(f"prepared passage text differs for {qid}")
        ranked = rank_prepared(features, config=cfg)
        ids = [item["passage_id"] for item in ranked]
        labels = data["query_relevance_labels"]
        row = {
            "query_id": qid, "query_language": data.get("query_language") or "unknown",
            "candidate_count": len(features),
            "returned_count": len(ranked),
            "relevant_count": sum(label >= rel_threshold for label in labels.values()),
            "ranked_ids": ids,
        }
        for k in ks:
            row[f"P@{k}"] = precision_at_k(ids, labels, k, rel_threshold)
            row[f"R@{k}"] = recall_at_k(ids, labels, k, rel_threshold)
            row[f"nDCG@{k}"] = ndcg_at_k(ids, labels, k)
        system = [
            sum(cfg["context_weights"][weight] * feature["context_features"][key]
                for key, weight in CONTEXT_WEIGHT_KEYS.items())
            for feature in features
        ]
        human = [data["context_labels"][feature["passage_id"]] for feature in features]
        correlation = None
        if len(set(system)) > 1 and len(set(human)) > 1:
            correlation = float(spearmanr(system, human).statistic)
            if not math.isfinite(correlation):
                correlation = None
        if correlation is not None:
            correlations.append(correlation)
        row["context_agreement_spearman"] = correlation
        rows.append(row)
    summary = {
        "per_query": rows, "config_used": copy.deepcopy(cfg),
        "query_count": len(rows), "context_agreement_query_count": len(correlations),
        "precision_definition": "relevant_returned / min(k, returned_count); zero if none returned",
        "context_agreement_population": "all prepared candidates, before filtering",
    }
    for k in ks:
        for metric in ("P", "R", "nDCG"):
            summary[f"mean_{metric}@{k}"] = (
                mean(row[f"{metric}@{k}"] for row in rows) if rows else 0.0)
    raw_correlation = mean(correlations) if correlations else None
    context_used = cfg["ranking_weights"]["indian_context_weight"] > 0
    summary["context_used_in_ranking"] = context_used
    summary["raw_context_agreement_spearman"] = raw_correlation
    summary["mean_context_agreement_spearman"] = raw_correlation if context_used else None
    summary["context_agreement_interpretation"] = (
        "raw context-feature agreement over all candidates; a missing ranking value means context has zero final weight")
    summary["stratified"] = _stratify(grouped_dataset, rows, ks, rel_threshold, context_used)
    from evaluation.calibrate_filter import evaluate_filter
    summary["filter_metrics"] = evaluate_filter(grouped_dataset, cache, cfg, rel_threshold)
    return summary



def _stratify(grouped, rows, ks, rel_threshold, context_used):
    languages = {}
    for row in rows:
        languages.setdefault(row["query_language"], []).append(row)
    by_language = {}
    for language, members in sorted(languages.items()):
        report = {"query_count": len(members),
                  "candidate_count": sum(row["candidate_count"] for row in members)}
        for k in ks:
            for metric in ("P", "R", "nDCG"):
                report[f"mean_{metric}@{k}"] = mean(row[f"{metric}@{k}"] for row in members)
        correlations = [row["context_agreement_spearman"] for row in members
                        if row["context_agreement_spearman"] is not None]
        raw = mean(correlations) if correlations else None
        report["context_used_in_ranking"] = context_used
        report["raw_context_agreement_spearman"] = raw
        report["mean_context_agreement_spearman"] = raw if context_used else None
        by_language[language] = report
    methods = {}
    for row in rows:
        data = grouped[row["query_id"]]
        method_by_id = {}
        for passage in data["passages"]:
            method = ((passage.get("extraction_metadata") or {}).get("method") or "unknown").casefold()
            method_by_id[passage["passage_id"]] = method
        for method in sorted(set(method_by_id.values())):
            ids = {pid for pid, candidate_method in method_by_id.items() if candidate_method == method}
            relevant = {pid for pid in ids if data["query_relevance_labels"][pid] >= rel_threshold}
            report = methods.setdefault(method, {
                "query_count": 0, "candidate_count": 0, "relevant_candidate_count": 0,
                "returned_count": 0, **{f"top{k}_count": 0 for k in ks},
                **{f"relevant_top{k}_count": 0 for k in ks},
            })
            report["query_count"] += 1
            report["candidate_count"] += len(ids)
            report["relevant_candidate_count"] += len(relevant)
            report["returned_count"] += sum(pid in ids for pid in row["ranked_ids"])
            for k in ks:
                top = row["ranked_ids"][:k]
                report[f"top{k}_count"] += sum(pid in ids for pid in top)
                report[f"relevant_top{k}_count"] += sum(pid in relevant for pid in top)
    for report in methods.values():
        for k in ks:
            count, hit = report[f"top{k}_count"], report[f"relevant_top{k}_count"]
            report[f"P@{k}"] = hit / count if count else 0.0
            total = report["relevant_candidate_count"]
            report[f"R@{k}"] = hit / total if total else None
    return {
        "query_language": by_language, "extraction_method": methods,
        "language_definition": "macro metrics by supplied query_language; missing metadata is unknown",
        "extraction_definition": "micro precision/recall of each method among original full-ranking top-K positions; ranks are never recomputed within method",
    }


def print_report(summary: dict) -> None:
    for key, value in summary.items():
        if key.startswith("mean_"):
            print(f"{key}: {value:.6f}" if isinstance(value, (int, float)) else f"{key}: {value}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(_ROOT / "evaluation/sample_dataset.jsonl"))
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--features", help="Feature cache exported by an experiment")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    cache = load_feature_cache(args.features, args.dataset) if args.features else None
    report = evaluate_ranker(load_dataset(args.dataset), config=config, prepared_features=cache)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    print_report(report)


if __name__ == "__main__":
    main()
