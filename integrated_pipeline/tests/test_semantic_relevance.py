"""Offline contracts plus explicitly opted-in real BGE-M3 integration tests."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

import task4_rank_evidence as t4

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
requires_model = pytest.mark.model

EN_ON = {"passage_id": "EN-ON", "text": "Rama is a central figure in the Ramayana, and the epic associates him with Ayodhya."}
HI_ON = {"passage_id": "HI-ON", "text": "रामायण में राम का संबंध अयोध्या से बताया गया है। राम भारतीय परंपरा के प्रमुख पात्र हैं।"}
EN_OFF = {"passage_id": "EN-OFF", "text": "Today's weather forecast predicts sunshine followed by rain in the evening."}
HI_OFF = {"passage_id": "HI-OFF", "text": "आज मौसम साफ रहेगा और शाम को हल्की बारिश होने की संभावना है।"}
MOON_ON = {"passage_id": "MOON-ON", "text": "Chandrayaan-3 explored the Moon near its south pole and measured the lunar surface and soil."}
EN_QUERY = "Rama and Ayodhya in the Ramayana"
HI_QUERY = "रामायण में राम और अयोध्या का क्या संबंध है?"
MIXED_QUERY = "Ayodhya में राम और Ramayana का क्या संबंध है?"

DEMO_PASSAGES = [
    dict(HI_ON, source_metadata={"source_type": "academic", "author": "Example researcher",
                               "url": "https://example.org/paper", "has_citations": True,
                               "publication_date": "2023-02-10"}),
    dict(EN_ON, source_metadata={"source_type": "academic", "organization": "Example university",
                               "url": "https://example.org/book", "has_citations": True}),
    {"passage_id": "DUP-A", "text": "Ayodhya is associated with Rama in the Ramayana epic.",
     "source_metadata": {"source_type": "blog"}},
    {"passage_id": "DUP-B", "text": "Ayodhya is associated with Rama in the Ramayana epic.",
     "source_metadata": {"source_type": "blog"}},
    dict(EN_OFF, source_metadata={"source_type": "blog"}),
]
EXPECTED_KEYS = {
    "rank", "passage_id", "text", "query_relevance", "indian_context_relevance",
    "source_signal", "final_score", "feature_breakdown", "provenance",
}


@requires_model
@pytest.mark.parametrize("query,passages,on_topic_count", [
    (EN_QUERY, [EN_ON, EN_OFF], 1),
    (HI_QUERY, [HI_ON, HI_OFF], 1),
    (HI_QUERY, [EN_ON, EN_OFF], 1),
    (MIXED_QUERY, [HI_ON, EN_ON, HI_OFF, EN_OFF], 2),
    ("Ram ka Ayodhya aur Ramayan se kya sambandh hai?", [EN_ON, HI_ON, EN_OFF], 2),
    ("Chandrayaan ne chaand ki satah aur mitti ke baare mein kya pata lagaya?",
     [MOON_ON, EN_OFF, HI_OFF], 1),
], ids=["english", "hindi", "cross-lingual", "mixed", "roman-hindi-rama", "roman-hindi-moon"])
def test_real_multilingual_semantic_quality(query, passages, on_topic_count):
    scores = t4.semantic_relevance(query, passages)
    assert len(scores) == len(passages)
    assert min(scores[:on_topic_count]) > max(scores[on_topic_count:])


@requires_model
def test_real_model_is_reused_and_batch_results_are_repeatable(monkeypatch):
    import sentence_transformers

    model = t4._get_semantic_model()

    def unexpected_constructor(*args, **kwargs):
        raise AssertionError("cached encoder should be reused")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", unexpected_constructor)
    first = t4.semantic_relevance(MIXED_QUERY, [HI_ON, EN_ON, EN_OFF], {"batch_size": 1})
    second = t4.semantic_relevance(MIXED_QUERY, [HI_ON, EN_ON, EN_OFF], {"batch_size": 16})
    assert first == pytest.approx(second, abs=1e-5)
    assert t4._SEMANTIC_MODEL is model


def test_semantic_encoder_receives_only_nonempty_text_and_preserves_positions(monkeypatch):
    calls = []

    class FakeEncoder:
        def encode(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            return np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])

    monkeypatch.setattr(t4, "_get_semantic_model", lambda *args, **kwargs: FakeEncoder())
    passages = [EN_ON, {"passage_id": "EMPTY", "text": " "}, EN_OFF,
                {"passage_id": "NONE", "text": None}, {"passage_id": "MISSING"}]
    scores = t4.semantic_relevance(EN_QUERY, passages, {"batch_size": 2})
    assert scores == pytest.approx([0.8, 0.0, 0.0, 0.0, 0.0])
    assert calls[0][0] == [t4.normalize_text(text) for text in [EN_QUERY, EN_ON["text"], EN_OFF["text"]]]
    assert calls[0][1]["batch_size"] == 2
    assert calls[0][1]["normalize_embeddings"] is True
    assert calls[0][1]["convert_to_numpy"] is True


def test_all_empty_passages_need_no_encoder(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty input must not load a model")
    monkeypatch.setattr(t4, "_get_semantic_model", unexpected)
    assert list(t4.semantic_relevance(HI_QUERY, [{"passage_id": "E", "text": ""}])) == [0.0]


def test_import_does_not_load_model():
    code = (
        "import sys, task4_rank_evidence as t4;"
        "assert t4._SEMANTIC_MODEL is None;"
        "assert 'sentence_transformers' not in sys.modules;"
        "assert 'torch' not in sys.modules"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_test_module_collection_does_not_load_encoder():
    code = """
import importlib.util
import pathlib
import socket
import sys
import types

attempts = []
def forbidden(*args, **kwargs):
    attempts.append((args, kwargs))
    raise AssertionError("test collection attempted to load a model or use network")

fake = types.ModuleType("sentence_transformers")
fake.SentenceTransformer = forbidden
sys.modules["sentence_transformers"] = fake
socket.socket.connect = forbidden
path = pathlib.Path("tests/test_semantic_relevance.py")
spec = importlib.util.spec_from_file_location("collection_probe", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import task4_rank_evidence as t4
assert t4._SEMANTIC_MODEL is None
assert not attempts, "test module attempted model loading or network during collection"
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_unavailable_model_error_is_actionable_and_preserves_cache(monkeypatch):
    fake = types.ModuleType("sentence_transformers")

    def missing_weights(*args, **kwargs):
        raise OSError("test weights unavailable")

    fake.SentenceTransformer = missing_weights
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    cached = object()
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL", cached)
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL_KEY", ("cached-test-model", None))
    with pytest.raises(t4.SemanticModelUnavailableError) as error:
        t4.semantic_relevance(EN_QUERY, [EN_ON], {"name": "missing-test-model"})
    assert "pip install" in str(error.value)
    assert "sentence-transformers" in str(error.value)
    assert "test weights unavailable" in str(error.value)
    assert t4._SEMANTIC_MODEL is cached
    assert t4._SEMANTIC_MODEL_KEY == ("cached-test-model", None)


def test_model_constructor_is_cached_offline(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    constructions = []
    model = object()

    def construct(*args, **kwargs):
        constructions.append((args, kwargs))
        return model

    fake.SentenceTransformer = construct
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL", None)
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL_KEY", None)
    assert t4._get_semantic_model({"name": "offline-test-model"}) is model
    assert t4._get_semantic_model({"name": "offline-test-model", "batch_size": 1}) is model
    assert len(constructions) == 1


def test_defaults_and_deep_merge_do_not_mutate_shared_config():
    before = copy.deepcopy(t4.DEFAULT_CONFIG)
    merged = t4.merge_config(t4.DEFAULT_CONFIG, {"output": {"include_provenance": False}})
    assert merged["output"]["include_provenance"] is False
    assert merged["output"]["include_feature_breakdown"] is True
    for section in ("relevance_weights", "context_weights", "ranking_weights", "semantic_model", "filter"):
        assert merged[section] == before[section]
    merged["context_weights"]["w_script"] = 999
    assert t4.DEFAULT_CONFIG == before
    assert before["relevance_weights"] == {"lexical_weight": 0.30, "semantic_weight": 0.70}
    assert before["semantic_model"]["name"] == "BAAI/bge-m3"


@pytest.mark.parametrize("weights", [
    {"lexical_weight": -0.1, "semantic_weight": 0.9},
    {"lexical_weight": 0.3, "semantic_weight": -0.7},
    {"lexical_weight": 0.0, "semantic_weight": 0.0},
    {"lexical_weight": 0.3},
    {"semantic_weight": 0.7},
    {"lexical_weight": "a lot", "semantic_weight": 0.7},
    {"lexical_weight": None, "semantic_weight": 0.7},
    {"lexical_weight": float("nan"), "semantic_weight": 0.7},
    {"lexical_weight": float("inf"), "semantic_weight": 0.7},
    "not-a-dict",
])
def test_invalid_relevance_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        t4.validate_relevance_weights(weights)


@pytest.mark.parametrize("given,expected", [
    ({"lexical_weight": 3.0, "semantic_weight": 7.0}, (0.3, 0.7)),
    ({"lexical_weight": 30, "semantic_weight": 70}, (0.3, 0.7)),
    ({"lexical_weight": 1.0, "semantic_weight": 0.0}, (1.0, 0.0)),
])
def test_valid_relevance_weights_are_normalized(given, expected):
    weights = t4.validate_relevance_weights(given)
    assert (weights["lexical_weight"], weights["semantic_weight"]) == pytest.approx(expected)


def test_partial_relevance_weights_are_backfilled(fake_semantic):
    merged = t4.merge_config(t4.DEFAULT_CONFIG, {"relevance_weights": {"lexical_weight": 0.5}})
    assert merged["relevance_weights"] == {"lexical_weight": 0.5, "semantic_weight": 0.7}
    assert t4.rank_evidence(EN_QUERY, [EN_ON], config={"relevance_weights": {"lexical_weight": 0.5}})


def test_lexical_only_config_bypasses_model(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("lexical-only ranking must not load the encoder")
    monkeypatch.setattr(t4, "_get_semantic_model", unexpected)
    ranked = t4.rank_evidence(EN_QUERY, [EN_ON, EN_OFF],
        config={"relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0}})
    assert ranked[0]["passage_id"] == "EN-ON"


def test_output_schema_scores_and_provenance(fake_semantic):
    ranked = t4.rank_evidence(HI_QUERY, DEMO_PASSAGES)
    assert len(ranked) == len(DEMO_PASSAGES)
    assert [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1))
    assert [row["final_score"] for row in ranked] == sorted(
        [row["final_score"] for row in ranked], reverse=True)
    for row in ranked:
        assert set(row) == EXPECTED_KEYS
        for field in ("query_relevance", "indian_context_relevance", "source_signal", "final_score"):
            assert 0 <= row[field] <= 1
        assert set(row["feature_breakdown"]["indian_context"]) == {
            "script_feature", "entity_feature", "place_feature", "institution_feature",
            "primary_text_feature", "contextual_vocab_feature", "query_context_match",
        }
        assert set(row["feature_breakdown"]["evidence"]) == {
            "source_type_score", "attribution_score", "date_score", "citation_score",
            "ocr_score", "provenance_score", "duplication_score",
        }
        assert set(row["provenance"]) == {"source_metadata", "extraction_metadata"}


def test_output_overrides_are_partial(fake_semantic):
    ranked = t4.rank_evidence(HI_QUERY, DEMO_PASSAGES,
                             config={"output": {"include_provenance": False}})
    assert "provenance" not in ranked[0]
    assert "feature_breakdown" in ranked[0]


def test_exact_duplicates_have_half_independence_score(fake_semantic):
    ranked = t4.rank_evidence(HI_QUERY, DEMO_PASSAGES)
    duplicates = [row for row in ranked if row["passage_id"].startswith("DUP-")]
    assert len(duplicates) == 2
    assert all(row["feature_breakdown"]["evidence"]["duplication_score"] == 0.5 for row in duplicates)


def test_top_k_and_empty_passage_exclusion(fake_semantic):
    passages = DEMO_PASSAGES + [{"passage_id": "EMPTY", "text": " \n\t"}]
    full = t4.rank_evidence(HI_QUERY, passages)
    assert len(full) == len(DEMO_PASSAGES)
    assert t4.rank_evidence(HI_QUERY, passages, top_k=3) == full[:3]
    assert t4.rank_evidence(HI_QUERY, passages, top_k=0) == []
    assert t4.rank_evidence(HI_QUERY, passages, top_k=99) == full
    assert t4.rank_evidence(HI_QUERY, passages, top_k=None) == full
    assert t4.rank_evidence(HI_QUERY, []) == []


def test_explicit_relevance_gate_rejects_unrelated_trusted_context(monkeypatch):
    monkeypatch.setattr(t4, "lexical_relevance", lambda q, ps: np.array([0.0, 0.8]))
    monkeypatch.setattr(t4, "semantic_relevance", lambda q, ps, model_config=None: np.array([0.0, 0.8]))
    passages = [
        {"passage_id": "UNRELATED", "text": "राम अयोध्या गंगा धर्म वेद",
         "source_metadata": {"source_type": "academic", "author": "Example",
                             "url": "https://example.org/unrelated", "has_citations": True}},
        {"passage_id": "RELEVANT", "text": "The requested technical explanation."},
    ]
    ranked = t4.rank_evidence("a technical question", passages,
                             config={"filter": {"enabled": True, "threshold": 0.1}})
    assert [row["passage_id"] for row in ranked] == ["RELEVANT"]


def test_missing_metadata_is_supported(fake_semantic):
    ranked = t4.rank_evidence(EN_QUERY, [{"passage_id": "BARE", "text": "Rama in Ayodhya."}])
    assert ranked[0]["passage_id"] == "BARE"
    assert 0 <= ranked[0]["source_signal"] <= 1


def test_rank_prepared_matches_direct_without_repeating_feature_extraction(fake_semantic, monkeypatch):
    config = {"evidence": {"reference_date": "2026-11-09"}}
    direct = t4.rank_evidence(HI_QUERY, DEMO_PASSAGES, config=config)
    prepared = t4.prepare_features(HI_QUERY, DEMO_PASSAGES, config=config)

    def unexpected(*args, **kwargs):
        raise AssertionError("cached ranking must not repeat relevance extraction")

    monkeypatch.setattr(t4, "lexical_relevance", unexpected)
    monkeypatch.setattr(t4, "semantic_relevance", unexpected)
    assert t4.rank_prepared(prepared, config=config) == direct


def test_lexical_only_prepared_features_cannot_silently_enable_semantics(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("lexical-only preparation must not load the model")

    monkeypatch.setattr(t4, "_get_semantic_model", unexpected)
    config = {"relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0}}
    prepared = t4.prepare_features(EN_QUERY, [EN_ON], config=config)
    with pytest.raises(ValueError, match="semantic"):
        t4.rank_prepared(prepared)


def test_ranking_uses_full_precision_before_rounding_display_scores(fake_semantic):
    config = {
        "relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0},
        "ranking_weights": {"query_relevance_weight": 1.0,
                            "indian_context_weight": 0.0, "evidence_weight": 0.0},
    }
    prepared = t4.prepare_features("query", [
        {"passage_id": "LOW", "text": "first candidate"},
        {"passage_id": "HIGH", "text": "second candidate"},
    ], config=config)
    prepared[0]["lexical_score"] = 0.50041
    prepared[1]["lexical_score"] = 0.50049
    ranked = t4.rank_prepared(prepared, config=config)
    assert [item["passage_id"] for item in ranked] == ["HIGH", "LOW"]
    assert ranked[0]["final_score"] == ranked[1]["final_score"]
