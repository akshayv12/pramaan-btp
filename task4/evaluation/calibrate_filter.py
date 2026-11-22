"""Select a relevance gate using training labels only.

Predeclared policy: maximize micro passage F1, subject to retaining at least
95% of label>=2 passages. Break ties by greater retention, then lower threshold.
The previously inspected six-query split remains a development holdout.
"""
from __future__ import annotations
import copy
import math

from task4_rank_evidence import validated_config


def _examples(grouped, prepared, config, rel_threshold):
    weights = config["relevance_weights"]
    examples = []
    for qid, data in sorted(grouped.items()):
        if qid not in prepared:
            raise ValueError(f"missing prepared features for {qid}")
        expected = [p["passage_id"] for p in data["passages"]]
        if [f["passage_id"] for f in prepared[qid]] != expected:
            raise ValueError(f"prepared IDs differ for {qid}")
        for feature in prepared[qid]:
            if weights["semantic_weight"] > 0 and feature.get("semantic_available") is False:
                raise ValueError("calibration requires available semantic features")
            score = weights["lexical_weight"] * feature["lexical_score"] + weights["semantic_weight"] * feature["semantic_score"]
            if not math.isfinite(score):
                raise ValueError("calibration relevance score must be finite")
            examples.append({
                "query_id": qid, "passage_id": feature["passage_id"],
                "score": min(1.0, max(0.0, float(score))),
                "positive": data["query_relevance_labels"][feature["passage_id"]] >= rel_threshold,
                "has_text": bool(feature["text"].strip()),
            })
    return examples


def _metrics(examples, threshold):
    tp = fp = fn = tn = 0
    per_query = {}
    for item in examples:
        selected = item["has_text"] and item["score"] >= threshold
        positive = item["positive"]
        if selected and positive:
            tp += 1
        elif selected:
            fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1
        row = per_query.setdefault(item["query_id"], {"positive_count": 0, "returned_count": 0})
        row["positive_count"] += int(positive)
        row["returned_count"] += int(selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    retention = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negatives = [row for row in per_query.values() if not row["positive_count"]]
    rejected = sum(not row["returned_count"] for row in negatives)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "positive_retention": retention, "f1": f1,
        "negative_rejection_rate": tn / (tn + fp) if tn + fp else None,
        "all_negative_query_count": len(negatives),
        "all_negative_queries_rejected": rejected,
        "all_negative_query_rejection_rate": rejected / len(negatives) if negatives else None,
        "query_count": len(per_query),
    }


def evaluate_filter(grouped_dataset, prepared_features, config=None, rel_threshold=2):
    """Describe gate decisions without selecting or changing the threshold."""
    cfg = validated_config(config)
    threshold = cfg["filter"]["threshold"] if cfg["filter"]["enabled"] else 0.0
    return _metrics(_examples(grouped_dataset, prepared_features, cfg, rel_threshold), threshold)


def calibrate_filter(train_dataset, prepared_features, config=None, min_retention=0.95, rel_threshold=2):
    """Only train_dataset labels are read, even when cache includes other IDs."""
    if isinstance(min_retention, bool) or not math.isfinite(min_retention) or not 0 < min_retention <= 1:
        raise ValueError("min_retention must be in (0, 1]")
    cfg = validated_config(config)
    examples = _examples(train_dataset, prepared_features, cfg, rel_threshold)
    if not any(item["positive"] for item in examples):
        raise ValueError("calibration requires at least one positively labeled training passage")
    candidates = {0.0, 1.0}
    scores = sorted({item["score"] for item in examples})
    # Midpoints avoid fragile one-ULP boundaries after normalization/replay.
    # When adjacent floats have no representable midpoint, the upper endpoint
    # gives the equivalent attainable partition.
    for low, high in zip(scores, scores[1:]):
        midpoint = low + (high - low) / 2
        candidates.add(midpoint if midpoint > low else high)
    trace = []
    for threshold in sorted(candidates):
        metrics = _metrics(examples, threshold)
        feasible = metrics["positive_retention"] >= min_retention
        trace.append({"threshold": threshold, "feasible": feasible, **metrics})
    feasible = [row for row in trace if row["feasible"]]
    if not feasible:
        raise ValueError("retention constraint cannot be met, including at threshold zero")
    best = max(feasible, key=lambda row: (row["f1"], row["positive_retention"], -row["threshold"]))
    deployment = copy.deepcopy(cfg)
    deployment["filter"] = {"enabled": True, "threshold": best["threshold"]}
    deployment.setdefault("output", {})["include_source_support"] = True
    return {
        "schema_version": 1, "selection_population": "training queries only",
        "training_query_ids": sorted(train_dataset),
        "policy": {
            "objective": "maximize micro passage F1", "positive_label_threshold": rel_threshold,
            "minimum_positive_retention": min_retention,
            "tie_break": ["higher positive retention", "lower threshold"],
            "candidate_rule": "zero, one, and midpoints between consecutive distinct observed relevance scores",
        },
        "selected_threshold": best["threshold"], "selected_metrics": best,
        "threshold_trace": trace, "input_config": cfg,
        "deployment_config": validated_config(deployment),
    }
