"""Training-only constrained calibration, including all-negative queries."""
import copy
import json
import math
import pytest
from evaluation.calibrate_filter import calibrate_filter, evaluate_filter
from evaluation.optimize_weights import run_experiment
from task4_rank_evidence import CONTEXT_WEIGHT_KEYS


def inputs():
    data, cache = {}, {}
    for qid, scores, labels in (
        ("train", (0.9, 0.6, 0.2, 0.1), (3, 0, 2, 0)),
        ("negative", (0.05, 0.03), (0, 0)),
    ):
        data[qid] = {
            "query": "query " + qid, "query_language": "en", "query_family": qid,
            "passages": [{"passage_id": f"{qid}-{i}", "text": f"text {i}"} for i in range(len(scores))],
            "query_relevance_labels": {f"{qid}-{i}": label for i, label in enumerate(labels)},
            "context_labels": {f"{qid}-{i}": 0 for i in range(len(scores))},
        }
        cache[qid] = [{
            "passage_id": f"{qid}-{i}", "text": f"text {i}", "lexical_score": score,
            "semantic_score": score, "semantic_available": True,
            "context_features": {key: 0 for key in CONTEXT_WEIGHT_KEYS},
            "source_signal": 0.5, "evidence_features": {}, "source_support": {},
            "provenance": {"source_metadata": {}, "extraction_metadata": {}},
        } for i, score in enumerate(scores)]
    return data, cache


def test_calibration_enforces_retention_and_excludes_cache_only_holdout():
    data, cache = inputs()
    cache["heldout"] = object()  # Any access would fail.
    before = copy.deepcopy(data)
    result = calibrate_filter(data, cache, min_retention=0.95)
    assert data == before
    assert result["training_query_ids"] == ["negative", "train"]
    assert result["selected_metrics"]["positive_retention"] == 1
    assert result["selected_metrics"]["true_positive"] == 2
    assert result["selected_metrics"]["false_negative"] == 0
    assert result["selected_metrics"]["all_negative_query_count"] == 1
    assert result["selected_metrics"]["all_negative_queries_rejected"] == 1
    assert result["deployment_config"]["filter"]["enabled"] is True
    assert all(row["positive_retention"] >= 0.95 for row in result["threshold_trace"] if row["feasible"])


def test_calibration_is_deterministic_and_tie_break_prefers_lower_threshold():
    data, cache = inputs()
    first = calibrate_filter(data, cache)
    second = calibrate_filter(dict(reversed(list(data.items()))), cache)
    assert first == second
    assert first["selected_threshold"] == pytest.approx(0.15)
    assert first["selected_metrics"]["f1"] == pytest.approx(0.8)


def test_all_negative_training_data_cannot_select_a_retention_threshold():
    data, cache = inputs()
    with pytest.raises(ValueError, match="positively"):
        calibrate_filter({"negative": data["negative"]}, cache)


def test_all_negative_evaluation_has_no_fabricated_positive_retention():
    data, cache = inputs()
    report = evaluate_filter({"negative": data["negative"]}, cache,
                             {"filter": {"enabled": True, "threshold": 0.5}})
    assert report["positive_retention"] is None
    assert report["true_negative"] == 2
    assert report["all_negative_query_rejection_rate"] == 1


@pytest.mark.parametrize("retention", [0, -1, 1.01, float("nan"), True])
def test_invalid_retention_constraint_is_rejected(retention):
    data, cache = inputs()
    with pytest.raises(ValueError):
        calibrate_filter(data, cache, min_retention=retention)


def test_calibrated_experiment_exports_both_profiles_and_training_trace(tmp_path, monkeypatch):
    data, cache = inputs()
    rows = []
    for qid, group in data.items():
        for passage in group["passages"]:
            pid = passage["passage_id"]
            rows.append({
                "query_id": qid, "query": group["query"], "query_language": group["query_language"],
                **passage, "query_relevance": group["query_relevance_labels"][pid],
                "indian_context_relevance": group["context_labels"][pid],
            })
    # Seed42 puts one query in train. Use a frozen split to require positive training data.
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    from evaluation.optimize_weights import sha256_file
    split = {
        "train_query_ids": ["train"], "test_query_ids": ["negative"],
        "query_families": {"train": "train", "negative": "negative"},
        "dataset_sha256": sha256_file(path),
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr("evaluation.optimize_weights.prepare_dataset_features", lambda *args: cache)
    output = tmp_path / "experiment"
    result = run_experiment(path, output, trials=2, seed=42, reference_date="2026-11-12",
                            split_path=split_path, calibrate_filter=True)
    saved = json.loads((output / "filter_calibration.json").read_text(encoding="utf-8"))
    assert saved["optimized"]["training_query_ids"] == ["train"]
    assert saved["test_query_ids_excluded_from_selection"] == ["negative"]
    assert result["deployment_config"]["filter"]["enabled"] is True
    assert set(result["reports"]["deployment"]) == {"train", "test", "full"}
    assert len(result["reports"]["baselines"]["full"]) == 4
    assert result["reports"]["deployment"]["test"]["filter_metrics"]["all_negative_query_count"] == 1
    for name in ("filter_calibration.json", "deployment_config.json", "baseline_deployment_config.json", "baselines.json"):
        assert name in result["manifest"]["artifact_sha256"]
