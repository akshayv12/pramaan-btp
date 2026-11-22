"""Offline comparisons must truly isolate query relevance."""
import copy
import pytest
from evaluation.baselines import build_baselines, run_baselines
from evaluation.evaluate import evaluate_ranker
from task4_rank_evidence import CONTEXT_WEIGHT_KEYS


def fixture_data():
    data = {"Q1": {
        "query": "query", "query_language": "hi",
        "passages": [
            {"passage_id": "A", "text": "A", "extraction_metadata": {"method": "ocr"}},
            {"passage_id": "B", "text": "B", "extraction_metadata": {"method": "direct"}},
        ],
        "query_relevance_labels": {"A": 3, "B": 0},
        "context_labels": {"A": 0, "B": 3},
    }}
    cache = {"Q1": [{
        "passage_id": pid, "text": pid,
        "lexical_score": lexical, "semantic_score": semantic,
        "semantic_available": True,
        "context_features": {key: context for key in CONTEXT_WEIGHT_KEYS},
        "source_signal": context, "evidence_features": {}, "source_support": {},
        "provenance": {"source_metadata": {}, "extraction_metadata": {}},
    } for pid, lexical, semantic, context in (("A", 0.9, 0.1, 0), ("B", 0.1, 0.9, 1))]}
    return data, cache


def test_pure_baselines_have_zero_context_and_evidence_weights():
    variants = build_baselines()
    assert len(variants) == 4
    for name in ("B0_tfidf_query_only", "B1_semantic_query_only", "B2_hybrid_query_only"):
        assert variants[name]["ranking_weights"] == {
            "query_relevance_weight": 1, "indian_context_weight": 0, "evidence_weight": 0}
        assert variants[name]["filter"]["enabled"] is False
    assert variants["B3_full_model"]["ranking_weights"]["indian_context_weight"] > 0


def test_baselines_use_cached_scores_and_report_unused_context_honestly(monkeypatch):
    data, cache = fixture_data()
    monkeypatch.setattr("evaluation.baselines.prepare_dataset_features",
                        lambda *args, **kwargs: pytest.fail("unexpected feature extraction"))
    reports = run_baselines(data, prepared_features=cache)
    lexical = reports["B0_tfidf_query_only"]
    assert lexical["per_query"][0]["ranked_ids"] == ["A", "B"]
    assert reports["B1_semantic_query_only"]["per_query"][0]["ranked_ids"] == ["B", "A"]
    assert reports["B3_full_model"]["per_query"][0]["ranked_ids"] == ["B", "A"]
    assert lexical["context_used_in_ranking"] is False
    assert lexical["mean_context_agreement_spearman"] is None
    assert lexical["raw_context_agreement_spearman"] == pytest.approx(1)


def test_language_metrics_keep_groups_and_extraction_counts_keep_global_ranks():
    data, cache = fixture_data()
    report = evaluate_ranker(data, build_baselines()["B0_tfidf_query_only"], prepared_features=cache)
    assert report["stratified"]["query_language"]["hi"]["query_count"] == 1
    assert report["stratified"]["query_language"]["hi"]["mean_nDCG@1"] == 1
    methods = report["stratified"]["extraction_method"]
    assert methods["ocr"]["top1_count"] == 1
    assert methods["ocr"]["R@1"] == 1
    assert methods["direct"]["top1_count"] == 0
    assert methods["direct"]["R@1"] is None


def test_baseline_construction_does_not_mutate_supplied_profile():
    config = {"filter": {"enabled": True, "threshold": 0.6}}
    before = copy.deepcopy(config)
    build_baselines(config)
    assert config == before

def test_zero_score_ties_are_input_order_invariant_and_label_independent():
    data, cache = fixture_data()
    for feature in cache["Q1"]:
        feature["lexical_score"] = feature["semantic_score"] = 0.0
        feature["context_features"] = {key: 0.0 for key in CONTEXT_WEIGHT_KEYS}
        feature["source_signal"] = 0.0
    original_order = [passage["passage_id"] for passage in data["Q1"]["passages"]]
    first = run_baselines(data, prepared_features=cache)
    reverse_data, reverse_cache = copy.deepcopy(data), copy.deepcopy(cache)
    reverse_data["Q1"]["passages"].reverse()
    reverse_cache["Q1"].reverse()
    second = run_baselines(reverse_data, prepared_features=reverse_cache)
    changed_labels = copy.deepcopy(data)
    changed_labels["Q1"]["query_relevance_labels"] = {"A": 0, "B": 3}
    third = run_baselines(changed_labels, prepared_features=cache)
    for name in first:
        expected = first[name]["per_query"][0]["ranked_ids"]
        assert second[name]["per_query"][0]["ranked_ids"] == expected
        assert third[name]["per_query"][0]["ranked_ids"] == expected
        assert "label-independent" in first[name]["tie_policy"]
    assert [passage["passage_id"] for passage in data["Q1"]["passages"]] == original_order
    assert [feature["passage_id"] for feature in cache["Q1"]] == original_order
