"""Offline regression tests for split isolation, caching and experiment artifacts."""
import copy
import json
from pathlib import Path

import pytest

from evaluation import evaluate
from evaluation.optimize_weights import (
    create_split, validate_split, make_objective, run_experiment,
)
from task4_rank_evidence import CONTEXT_WEIGHT_KEYS


def config():
    return {
        "relevance_weights": {"lexical_weight": 0.3, "semantic_weight": 0.7},
        "semantic_model": {"name": "fake-offline-model"},
        "evidence": {"reference_date": "2026-11-09"},
    }


def group(qid="Q1"):
    return {
        "query": "query " + qid, "query_family": qid,
        "passages": [{"passage_id": f"{qid}-{i}", "text": f"passage {i}"}
                     for i in range(3)],
        "query_relevance_labels": {f"{qid}-{i}": label for i, label in enumerate((0, 2, 3))},
        "context_labels": {f"{qid}-{i}": i for i in range(3)},
    }


def prepared(data):
    return [{
        "passage_id": passage["passage_id"], "text": passage["text"],
        "lexical_score": score, "semantic_score": score,
        "semantic_available": True,
        "context_features": {key: score for key in CONTEXT_WEIGHT_KEYS},
        "source_signal": score, "evidence_features": {}, "source_support": {},
        "provenance": {"source_metadata": {}, "extraction_metadata": {}},
    } for passage, score in zip(data["passages"], (0.1, 0.5, 0.9))]


def test_split_is_stable_and_22_6_for_main_dataset():
    data = {f"Q{i:03}": group(f"Q{i:03}") for i in range(28)}
    first = create_split(data)
    second = create_split(dict(reversed(list(data.items()))))
    assert first == second
    assert len(first["train_query_ids"]) == 22
    assert len(first["test_query_ids"]) == 6
    assert not set(first["train_query_ids"]) & set(first["test_query_ids"])


def test_equivalent_query_families_stay_together():
    data = {f"Q{i}": group(f"Q{i}") for i in range(8)}
    data["Q0"]["query_family"] = data["Q1"]["query_family"] = "same-information-need"
    split = create_split(data)
    assert ("Q0" in split["train_query_ids"]) == ("Q1" in split["train_query_ids"])
    validate_split(split, data)


def test_frozen_split_rejects_different_dataset():
    data = {"Q1": group("Q1"), "Q2": group("Q2")}
    split = create_split(data)
    split["dataset_sha256"] = "original"
    with pytest.raises(ValueError, match="hash"):
        validate_split(split, data, "changed")


def test_objective_receives_only_train_labels_and_cache(monkeypatch):
    train = {"Qtrain": group("Qtrain")}
    cache = {qid: prepared(group(qid)) for qid in ("Qtrain", "Qheldout")}
    calls = []

    def fake_evaluate(data, **kwargs):
        calls.append((set(data), set(kwargs["prepared_features"])))
        return {"mean_nDCG@10": 0.123456789}

    monkeypatch.setattr("evaluation.optimize_weights.evaluate_ranker", fake_evaluate)

    class Trial:
        def __init__(self):
            self.names = []

        def suggest_float(self, name, low, high):
            self.names.append(name)
            assert (low, high) == (0.01, 1.0)
            return 0.5

    trial = Trial()
    objective = make_objective(train, cache, config())
    assert objective(trial) == 0.123456789
    assert calls == [({"Qtrain"}, {"Qtrain"})]
    assert len(trial.names) == len(set(trial.names)) == 12


def test_preparation_never_receives_labels(monkeypatch):
    data = {"Q1": group()}
    calls = []

    def fake_prepare(query, passages, config):
        calls.append(passages)
        return []

    monkeypatch.setattr(evaluate, "prepare_features", fake_prepare)
    assert evaluate.prepare_dataset_features(data, config()) == {"Q1": []}
    assert all(set(passage) == {"passage_id", "text"} for passage in calls[0])


def test_cached_evaluation_does_not_reextract(monkeypatch):
    data = {"Q1": group()}
    monkeypatch.setattr(evaluate, "prepare_dataset_features",
                        lambda *args, **kwargs: pytest.fail("features were recomputed"))
    report = evaluate.evaluate_ranker(data, config(), prepared_features={"Q1": prepared(data["Q1"])})
    assert report["mean_nDCG@10"] == 1.0
    assert report["mean_R@1"] == 0.5
    assert report["mean_P@3"] == pytest.approx(2 / 3)
    assert report["mean_P@3"] != 0.667


def test_context_agreement_uses_entire_pool_before_filtering():
    data = {"Q1": group()}
    cache = {"Q1": prepared(data["Q1"])}
    settings = config()
    settings["filter"] = {"enabled": True, "threshold": 0.8}
    report = evaluate.evaluate_ranker(data, settings, prepared_features=cache)
    assert report["per_query"][0]["returned_count"] == 1
    assert report["mean_context_agreement_spearman"] == pytest.approx(1.0)
    assert report["context_agreement_query_count"] == 1


@pytest.mark.parametrize("problem", ["missing_label", "duplicate", "label_range"])
def test_dataset_loader_rejects_bad_annotation_rows(tmp_path, problem):
    row = {"query_id": "Q1", "query": "query", "passage_id": "P1", "text": "text",
           "query_relevance": 3, "indian_context_relevance": 2}
    rows = [row]
    if problem == "missing_label":
        del row["query_relevance"]
    elif problem == "duplicate":
        rows.append(copy.deepcopy(row))
    else:
        row["query_relevance"] = 4
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate.load_dataset(path)


def test_small_experiment_exports_reproducible_artifacts_without_model(tmp_path, monkeypatch):
    path = tmp_path / "dataset.jsonl"
    rows = []
    for number in range(4):
        qid = f"Q{number}"
        for i, relevance in enumerate((0, 2, 3)):
            rows.append({
                "query_id": qid, "query": "query " + qid,
                "passage_id": f"{qid}-{i}", "text": f"passage {i}",
                "query_relevance": relevance, "indian_context_relevance": i,
            })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr("evaluation.optimize_weights.prepare_dataset_features",
                        lambda data, config: {qid: prepared(group) for qid, group in data.items()})
    output = tmp_path / "experiment"
    result = run_experiment(path, output, config(), trials=2, seed=42,
                            reference_date="2026-11-09")
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["completed_trials"] == 2
    assert len(result["manifest"]["search_space"]) == 12
    assert len(result["reports"]["optimized"]["train"]["per_query"]) == 3
    assert len(result["reports"]["optimized"]["test"]["per_query"]) == 1
    assert len(result["reports"]["ablations"]["test"]) == 6
    for section in ("relevance_weights", "context_weights", "ranking_weights"):
        assert sum(result["config"][section].values()) == pytest.approx(1)
    for name in ("manifest.json", "split.json", "features.json", "trials.jsonl",
                 "reports.json", "ablations.json", "base_config.json", "chosen_config.json"):
        assert (output / name).is_file()


def test_feature_cache_requires_manifest_and_matching_dataset(tmp_path):
    import hashlib
    cache = tmp_path / "features.json"
    cache.write_text("{}", encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        evaluate.load_feature_cache(cache, dataset)
    manifest = {
        "status": "complete",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "feature_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(evaluate._ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in evaluate.feature_cache_sources()
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert evaluate.load_feature_cache(cache, dataset) == {}
    dataset.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset hash"):
        evaluate.load_feature_cache(cache, dataset)


def test_cache_passage_text_must_match_dataset():
    data = {"Q1": group()}
    cache = {"Q1": prepared(data["Q1"])}
    cache["Q1"][0]["text"] = "different passage"
    with pytest.raises(ValueError, match="text differs"):
        evaluate.evaluate_ranker(data, config(), prepared_features=cache)
